import argparse
import math
import random
import os
import gc

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from models import Generator, Discriminator
from dataset import CreateDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

def showTensorImage(image):
    fig = plt.figure()
    image = image.cpu().detach().numpy()
    image = np.swapaxes(np.swapaxes(image, 0, 2), 0, 1)
    plt.imshow(image)
    plt.show()
def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()

def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths

def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises

def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]

def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

device = "cuda"

PATH = "C:/Users/quent/Desktop/art generation/all/1/"
ITER = 800000
BATCH = 1
NSAMPLE = 8
SIZEX = 768
SIZEY = 1024
R1 = 10
PATH_REGULARIZE = 2
PATH_BATCH_SHRINK = 2

d_reg_every = 16
g_reg_every = 4

MIXING = 0.9
SAVE_DIR = None # "C:/Users/quent/Desktop/art generation/StyleGan2_pytorch/checkpoint/stylegan2-ffhq-config-f.pt"
LR = 0.002
CHANNEL_MULTIPLIER = 2

LOCAL_RANK = 0

n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
DISTRIBUTED = n_gpu > 1

if DISTRIBUTED:
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    synchronize()

LATENT = 512
N_MLP = 8
START_ITER = 0
    

generator = Generator(SIZEX, SIZEY, LATENT, N_MLP, channel_multiplier=CHANNEL_MULTIPLIER).to(device)
discriminator = Discriminator(SIZEX, SIZEY, channel_multiplier=CHANNEL_MULTIPLIER).to(device)

from torchsummary import summary

summary(generator, (1, LATENT))

torch.cuda.empty_cache()

# average of the weights of generator to visualize each epochs
g_ema = Generator(LATENT, N_MLP, channel_multiplier=CHANNEL_MULTIPLIER).cuda()

# eval mode
g_ema.eval()

# slowly move through each generator steps
accumulate(g_ema, generator, 0)

g_reg_ratio = g_reg_every / (g_reg_every + 1)
d_reg_ratio = d_reg_every / (d_reg_every + 1)

g_optim = optim.Adam(generator.parameters(), lr=LR * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio))
d_optim = optim.Adam(discriminator.parameters(), lr=LR * d_reg_ratio, betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio))

if SAVE_DIR is not None:
    print("load model:", SAVE_DIR)

    saved = torch.load(SAVE_DIR, map_location=lambda storage, loc: storage)

    try:
        saved_name = SAVE_DIR.split("/")[-1]
        START_ITER = int(saved_name.split(".")[0])

    except ValueError:
        pass

    generator.load_state_dict(saved["g"], strict=False)
    discriminator.load_state_dict(saved["d"], strict=False)
    g_ema.load_state_dict(saved["g_ema"], strict=False)

    # g_optim.load_state_dict(saved["g_optim"])
    # d_optim.load_state_dict(saved["d_optim"])

if DISTRIBUTED:
    generator = nn.parallel.DistributedDataParallel(
        generator,
        device_ids=[LOCAL_RANK],
        output_device=LOCAL_RANK,
        broadcast_buffers=False,
    )

    discriminator = nn.parallel.DistributedDataParallel(
        discriminator,
        device_ids=[LOCAL_RANK],
        output_device=LOCAL_RANK,
        broadcast_buffers=False,
    )

transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize(max(SIZEX, SIZEY), Image.BILINEAR),
            transforms.CenterCrop((SIZEX, SIZEY)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

dataset = CreateDataset(PATH, transform)
loader = data.DataLoader(
    dataset,
    batch_size=BATCH,
    sampler=data_sampler(dataset, shuffle=True, distributed=DISTRIBUTED),
    drop_last=True,
)

loader = sample_data(loader)

pbar = range(ITER)

if get_rank() == 0:
    pbar = tqdm(pbar, initial=START_ITER, dynamic_ncols=True, smoothing=0.01)

mean_path_length = 0

d_loss_val = 0
r1_loss = torch.tensor(0.0, device=device)
g_loss_val = 0
path_loss = torch.tensor(0.0, device=device)
path_lengths = torch.tensor(0.0, device=device)
mean_path_length_avg = 0
loss_dict = {}

if DISTRIBUTED:
    g_module = generator.module
    d_module = discriminator.module

else:
    g_module = generator
    d_module = discriminator

accum = 0.5 ** (32 / (10 * 1000))

sample_z = torch.randn(NSAMPLE, LATENT, device=device)

for idx in pbar:
    i = idx + START_ITER

    if i > ITER:
        print("Done!")

        break
        
    real_img = next(loader)
    real_img = real_img.to(device)
    
    # train the discriminator
    requires_grad(generator, False)
    requires_grad(discriminator, True)

    noise = mixing_noise(BATCH, LATENT, MIXING, device)
    fake_img, _ = generator(noise)
    fake_pred = discriminator(fake_img)

    real_pred = discriminator(real_img)
    d_loss = d_logistic_loss(real_pred, fake_pred)

    loss_dict["d"] = d_loss
    loss_dict["real_score"] = real_pred.mean()
    loss_dict["fake_score"] = fake_pred.mean()

    discriminator.zero_grad()
    d_loss.backward()
    d_optim.step()

    d_regularize = i % d_reg_every == 0

    if d_regularize:
        real_img.requires_grad = True
        real_pred = discriminator(real_img)
        r1_loss = d_r1_loss(real_pred, real_img)

        discriminator.zero_grad()
        (R1 / 2 * r1_loss * d_reg_every + 0 * real_pred[0]).backward()

        d_optim.step()

    loss_dict["r1"] = r1_loss      
    
    requires_grad(generator, True)
    requires_grad(discriminator, False)

    noise = mixing_noise(BATCH, LATENT, MIXING, device)
    fake_img, _ = generator(noise)
    fake_pred = discriminator(fake_img)
    g_loss = g_nonsaturating_loss(fake_pred)

    loss_dict["g"] = g_loss

    generator.zero_grad()
    g_loss.backward()
    g_optim.step()

    g_regularize = i % g_reg_every == 0

    if g_regularize:
        path_batch_size = max(1, BATCH // PATH_BATCH_SHRINK)
        noise = mixing_noise(path_batch_size, LATENT, MIXING, device)
        fake_img, latents = generator(noise, return_latents=True)

        path_loss, mean_path_length, path_lengths = g_path_regularize(
            fake_img, latents, mean_path_length
        )

        generator.zero_grad()
        weighted_path_loss = PATH_REGULARIZE * g_reg_every * path_loss

        if PATH_BATCH_SHRINK:
            weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

        weighted_path_loss.backward()

        g_optim.step()

        mean_path_length_avg = (
            reduce_sum(mean_path_length).item() / get_world_size()
        )

    
    loss_dict["path"] = path_loss
    loss_dict["path_length"] = path_lengths.mean()

    accumulate(g_ema, g_module, accum)

    loss_reduced = reduce_loss_dict(loss_dict)

    d_loss_val = loss_reduced["d"].mean().item()
    g_loss_val = loss_reduced["g"].mean().item()
    r1_val = loss_reduced["r1"].mean().item()
    path_loss_val = loss_reduced["path"].mean().item()
    real_score_val = loss_reduced["real_score"].mean().item()
    fake_score_val = loss_reduced["fake_score"].mean().item()
    path_length_val = loss_reduced["path_length"].mean().item()

    
    gc.collect()
    torch.cuda.empty_cache()
    
    if get_rank() == 0:
        pbar.set_description(
                    (
                        f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                        f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}"
                    )
                )
        
        gc.collect()
        torch.cuda.empty_cache()
    
        if i % 100  == 0:
            with torch.no_grad():
                 g_ema.eval()
                 sample, _ = g_ema([sample_z])
                 utils.save_image(
                     sample,
                     f"sample/{str(i).zfill(6)}.png",
                     nrow=int(NSAMPLE ** 0.5),
                     normalize=True,
                     range=(-1, 1),
                 )
                 
        if i % 500 == 0:
            torch.save(
                {
                    "g": g_module.state_dict(),
                    "d": d_module.state_dict(),
                    "g_ema": g_ema.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
                },
                f"checkpoint/{str(i).zfill(6)}.pt",
            )
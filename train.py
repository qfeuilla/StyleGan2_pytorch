from PIL import Image
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

from dataset import CreateDataset
import matplotlib.pyplot as plt

from models import Generator, Discriminator
import time

def showTensorImage(image):
    fig = plt.figure()
    image = image.cpu().detach().numpy()
    image = np.swapaxes(np.swapaxes(image, 0, 2), 0, 1)
    plt.imshow(image)
    plt.show()

def get_world_size():
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()


def reduce_sum(tensor):
    if not dist.is_available():
        return tensor

    if not dist.is_initialized():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return tensor

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch
            
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def make_noise(batch, latent_dim, n_noise):
    if n_noise == 1:
        return torch.randn(batch, latent_dim).cuda()

    noises = torch.randn(n_noise, batch, latent_dim).cuda().unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2)

    else:
        return [make_noise(batch, latent_dim, 1)]

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

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

def reduce_loss_dict(loss_dict):
    world_size = get_world_size()

    if world_size < 2:
        return loss_dict

    with torch.no_grad():
        keys = []
        losses = []

        for k in sorted(loss_dict.keys()):
            keys.append(k)
            losses.append(loss_dict[k])

        losses = torch.stack(losses, 0)
        dist.reduce(losses, dst=0)

        if dist.get_rank() == 0:
            losses /= world_size

        reduced_losses = {k: v for k, v in zip(keys, losses)}

    return reduced_losses

BATCH = 2
SIZE = 256
LR = 0.002
SAVE_DIR = None
START_ITER = 0
EPOCHS = 800000
LATENT = 512
MIXING = 0.9

generator = nn.DataParallel(Generator(SIZE, 512, 8, channel_multiplier=2)).cuda()
discriminator = nn.DataParallel(Discriminator(SIZE, channel_multiplier=2)).cuda()

# average of the weights of generator to visualize each epochs
g_ema = nn.DataParallel(Generator(SIZE, 512, 8, channel_multiplier=2)).cuda()

# eval mode
g_ema.eval()

# slowly move through each generator steps
accumulate(g_ema, generator, 0)

# step for regulatisation
d_reg_every = 16
g_reg_every = 4

g_reg_ratio = g_reg_every / (g_reg_every + 1)
d_reg_ratio = d_reg_every / (d_reg_every + 1)

g_optim = optim.Adam(generator.parameters(), lr=LR * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio))
d_optim = optim.Adam(discriminator.parameters(), lr=LR * d_reg_ratio, betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio))

if SAVE_DIR is not None:
        print("load model:", SAVE_DIR)

        saved = torch.load(SAVE_DIR, map_location=lambda storage, loc: storage)

        try:
            saved_name = os.path.basename(saved)
            START_ITER = int(os.path.splitext(saved_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(saved["g"])
        discriminator.load_state_dict(saved["d"])
        g_ema.load_state_dict(saved["g_ema"])

        g_optim.load_state_dict(saved["g_optim"])
        d_optim.load_state_dict(saved["d_optim"])



transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize(SIZE, Image.LANCZOS),
            transforms.CenterCrop((SIZE, SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

dataset = CreateDataset("C:/Users/quent/Desktop/art generation/all/1/", transform)
loader = data.DataLoader(
    dataset,
    batch_size=BATCH,
    sampler=data_sampler(dataset, shuffle=True, distributed=False),
    drop_last=True,
)

loader = sample_data(loader)

g_module = generator
d_module = discriminator

pbar = tqdm(range(EPOCHS), dynamic_ncols=True, smoothing=0.01)
# pbar = range(EPOCHS)

mean_path_length = 0

device = torch.device("cuda")

d_loss_val = 0
r1_loss = torch.tensor(0.0, device=device)
g_loss_val = 0
path_loss = torch.tensor(0.0, device=device)
path_lengths = torch.tensor(0.0, device=device)
mean_path_length_avg = 0
loss_dict = {}

accum = 0.5 ** (32 / (10 * 1000))

sample_z = torch.randn(8, 512, device=device)

for idx in pbar:
    i = idx
    
    start = time.time()
    
    real_img = next(loader)
    real_img = real_img.cuda()

    # train the discriminator
    requires_grad(generator, False)
    requires_grad(discriminator, True)

    noise = mixing_noise(BATCH, LATENT, 0.9)
    fake_img, _ = generator(noise)

    fake_pred = discriminator(real_img)
    real_pred = discriminator(real_img)
    
    d_loss = d_logistic_loss(real_pred, fake_pred)
    
    loss_dict["d"] = d_loss
    loss_dict["real_score"] = real_pred.mean()
    loss_dict["fake_score"] = fake_pred.mean()
    
    discriminator.zero_grad()
    
    d_loss.backward()
    
    d_optim.step()

    if i % d_reg_every == 0:
        real_img.requires_grad = True
        
        real_pred = discriminator(real_img)
        
        r1_loss = d_r1_loss(real_pred, real_img)
        
        discriminator.zero_grad()
        
        (10 / 2 * r1_loss * d_reg_every + 0 * real_pred[0]).backward()
        
        d_optim.step()
        
    loss_dict["r1"] = r1_loss        
    
    requires_grad(generator, True)
    requires_grad(discriminator, False)
    
    noise = mixing_noise(BATCH, LATENT, MIXING)
    fake_img, _ = generator(noise)
    fake_pred = discriminator(fake_img)
    g_loss = g_nonsaturating_loss(fake_pred)

    loss_dict["g"] = g_loss

    generator.zero_grad()
    g_loss.backward()
    
    g_optim.step()


    if i % g_reg_every == 0:
        path_batch_size = max(1, BATCH // 2)
        noise = mixing_noise(path_batch_size, LATENT, MIXING)
        fake_img, latents = generator(noise, return_latents=True)

        path_loss, mean_path_length, path_lengths = g_path_regularize(fake_img, latents, mean_path_length)

        generator.zero_grad()
        weighted_path_loss = 2 * g_reg_every * path_loss

        weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

        weighted_path_loss.backward()

        g_optim.step()

        mean_path_length_avg = (reduce_sum(mean_path_length).item() / get_world_size())
    
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
    
    pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}"
                )
            )
    
    gc.collect()
    torch.cuda.empty_cache()

    if i % 50 == 0:
        with torch.no_grad():
            g_ema.eval()
            sample, _ = g_ema([sample_z])
            utils.save_image(
                sample,
                f"sample/{str(i).zfill(6)}.png",
                nrow=int(8 ** 0.5),
                normalize=True,
                range=(-1, 1),
            )
    if i % 200 == 0:
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
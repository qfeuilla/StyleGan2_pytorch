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
from torchsummary import summary

from models import Generator, Discriminator

def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to('cpu')
        .numpy()
    )

latent_path = ["C:/Users/quent/Desktop/art generation/test/1_.pt", "C:/Users/quent/Desktop/art generation/test/69005_.pt"]
latentDict1 = torch.load(latent_path[0], map_location=lambda storage, loc: storage)
latentDict2 = torch.load(latent_path[1], map_location=lambda storage, loc: storage)


CHANNEL_MULTIPLIER = 2
Size = 1024
    
generator = Generator(Size, 512, 8, sizex=4, sizey=2).cuda()

summary(generator, (1, 512))

generator_save = "C:/Users/quent/Desktop/art generation/StyleGan2_pytorch/checkpoint/stylegan2-ffhq-config-f.pt"

if generator_save is not None:
    print("load model:", generator_save)

    saved = torch.load(generator_save, map_location=lambda storage, loc: storage)

    generator.load_state_dict(saved['g_ema'], strict=False)

generator.eval()

keys1 = list(latentDict1.keys())
keys2 = list(latentDict2.keys())

def mix(i):
    latent_n1 = latentDict1[keys1[1]]['latent'].reshape(1, -1, 512)
    latent_n2 = latentDict2[keys2[1]]['latent'].reshape(1, -1, 512)
    
    latent = torch.cat((latent_n1[:, :i], latent_n2[:, i:]), 1)
            
    img_gen, _ = generator([latent], inject_index=i, input_is_latent=True)
    
    
    img_ar = make_image(img_gen)
    
    plt.figure()
    plt.imshow(img_ar[0])
    plt.show()
    
    pil_img = Image.fromarray(img_ar[0])
    pil_img.save("C:/Users/quent/Desktop/art generation/test/mix2_" + str(i) + ".jpg")

for i in range(18):
    mix(i)

from torch import nn
import torch
import torchvision
from torchsummary import summary

import math
from torch.functional import F
from upfirdn2d import upfirdn2d
from PIL import Image

from torchvision import transforms
import numpy as np

#Linear Layer
class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=False):
        super().__init__()
        
        # random weight initialisation and lr scaling
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        
        # if bias, init bias with bias_init
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else :
            self.bias = None
        
        # using leaky relu if activation
        self.activation = activation
        
        # used to scale the weight
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
            
    def forward(self, input):
        # if activation bias is used on leaky relu else, directly in the linear function
        if self.activation:
            out = F.linear(input, self.weight * self.scale)

            # fused version of leaky relu
            out = out + (self.bias * self.lr_mul)
            out = F.leaky_relu(out)

            
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
        
        return out
    
    def __repr__(self):
        # return the representation of weight for summary
        return (f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})')

# Normalization for latent vector
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

# Input layer 
class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        # batch for multiple image
        batch = input.shape[0]
        
        out = self.input.repeat(batch, 1, 1, 1)

        return out


def make_kernel(k):
    # Create the tensor kernel
    k = torch.tensor(k, dtype=torch.float32)
    
    # Expand the dimension if 1d
    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    # Normalize the kernel
    k /= k.sum()
    
    return k

# Using blur for more realisitic image (and less harsh color changing)
class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        # create blur kernel (for each pixel, add a small portion of suronding pixel)
        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        # save without training
        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        # Apply blur kernel on input
        out = upfirdn2d(input, self.kernel, pad=self.pad)
        
        return out

# Modulated version of the Conv2d to replace AdaIN for style embedding
class ModulatedConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, demodulate=True, up_sample=False, down_sample=False, blur_kernel=[1, 3, 3, 1]):
        
        super().__init__()
        
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.up_sample = up_sample
        self.down_sample = down_sample
        
        if up_sample:
            # define padding parameters
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            
            # create blur Layer
            self.blur = Blur(blur_kernel, (pad0, pad1))
        
        if down_sample:
            # define padding parameters
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            
            # create blur Layer
            self.blur = Blur(blur_kernel, (pad0, pad1))
        
        # Each pixel multiplied by the kernel
        fan_in = in_channel * kernel_size ** 2
        
        # Scale for weight scaling
        self.scale = 1 / math.sqrt(fan_in)
        # Padding after Conv2d
        self.padding = kernel_size // 2
        
        # random weight initialization
        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
        
        #modulation for style embedding
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        
        self.demodulate = demodulate
    
    def forward(self, input, style):
        batch, in_channel, height, width = input.shape
        
        # Modulate the style
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        
        # Adding style modulation to weight
        weight = self.weight * style * self.scale
        
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
        #A good explenation of modulation/demodulation https://youtu.be/MYCTn80qSk0?t=142
        
        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)
        
        if self.up_sample:
            
            # Reshape input for deconvolution
            input = input.view(1, batch * in_channel, height, width)
            
            # Reshape weights for deconvolution
            weight = weight.view(batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size)
            
            # Upsampling convolution
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            
            # Reshape output to initial shape
            out = out.view(batch, self.out_channel, height, width)
            
            # Blur the output
            out = self.blur(out)
        
        elif self.down_sample:
            # blur image before convolution
            input = self.blur(input)
            
            # Reshape input for convolution
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            
            # Downsampling convolution
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            
            # Reshape output to initial shape
            out = out.view(batch, self.out_channel, height, width)
        
        else:
            # Reshape input for convolution
            input = input.view(1, batch * in_channel, height, width)
            
            # Downsampling convolution
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            
            # Reshape output to initial shape
            out = out.view(batch, self.out_channel, height, width)
        
        return out
    
    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, 'f'upsample={self.upsample}, downsample={self.downsample})')
    

# Inject noise on image (for detail jittering like hair ...)
class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise

# scaled version of Leaky ReLU
class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)

#Modulated conv plus noise
class StyledConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, upsample=False, blur_kernel=[1, 3, 3, 1], demodulate=True):
        super().__init__()
        
        self.conv = ModulatedConv2d(in_channel, out_channel, kernel_size, style_dim, up_sample=upsample, blur_kernel=blur_kernel, demodulate=demodulate)
        self.noise = NoiseInjection()
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        self.activate = ScaledLeakyReLU(0.2)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = out + self.bias
        out = self.activate(out)

        return out

# Augment input size
class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        # Scaling factor
        self.factor = factor
        
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


# Reduce input size
class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out

# Convert from high-dimensional per-pixeldata to RGB
class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out

class Generator(nn.Module):
    def __init__(self, size, style_dim, n_mlp, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], lr_mlp=0.01):
        super().__init__()

        self.size = size

        self.style_dim = style_dim
        
        # First normalization layer
        layers = [PixelNorm()]
        
        # The Mapping network 
        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation=True
                )
            )

        self.style = nn.Sequential(*layers)
        
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])

        self.conv1 = StyledConv(self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel)
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))
            
        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            # Double Style Convolution with one upsample
            self.convs.append(
                StyledConv(in_channel, out_channel, 3, style_dim, upsample=True, blur_kernel=blur_kernel)
            )

            self.convs.append(
                StyledConv(out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel)
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises
    
    def mean_latent(self, n_latent):
        latent_in = torch.randn(n_latent, self.style_dim, device=self.input.input.device)
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(self, styles, return_latents=False, inject_index=None, truncation=1, truncation_latent=None, input_is_latent=False, noise=None, randomize_noise=True):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]
        
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
                ]
        
        # Quality/variety trade-off 
        # More trucation is more quality but less variety
        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(truncation_latent + truncation * (style - truncation_latent))

            styles = style_t
        
        # If only one style
        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]
        
        # Choose where to inject the second style
        else:
            if inject_index is None:
                inject_index = np.random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)
                
        # mapping network
        out = self.input(latent)
        
        # first conv
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        
        # convert to RGB
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        
        # loop through conv and RGB converter
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        if return_latents:
            return image, latent

        else:
            return image, None




import math
import random
import os

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

BATCH = 4

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

SIZE = 256

generator = Generator(SIZE, 512, 8, channel_multiplier=2).cuda()

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

for b in loader:
    a = b
    break;

print(a.shape)

def showTensorImage(image):
    fig = plt.figure()
    image = image.cpu().detach().numpy()
    image = np.swapaxes(np.swapaxes(image, 0, 2), 0, 1)
    plt.imshow(image)
    plt.show()

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

loader = sample_data(loader)

g_module = generator

pbar = range(2)

for idx in pbar:
    i = idx
    print(i)
    
    real_img = next(loader)
    real_img = real_img.cuda()
    
    requires_grad(generator, False)

    noise = mixing_noise(BATCH, 512, 0.9)
    fake_img, _ = generator(noise)
    showTensorImage(fake_img[0])


















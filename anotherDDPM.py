#%%
# Import of libraries
import random
import imageio
import numpy as np
from argparse import ArgumentParser

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import einops
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import MNIST, FashionMNIST


# Import Pytorch
import torch
from torch import nn
# Import plot
import matplotlib.pyplot as plt
# Import computer vision
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
# Batch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import T

# setup train data
train_data = datasets.FashionMNIST(root = "data",
                                   train = True,
                                   download =True,
                                   transform = torchvision.transforms.ToTensor(),
                                   target_transform = None
                                  )


# setup train data
test_data = datasets.FashionMNIST(root = "data",
                                   train = False,
                                   download =True,
                                   transform = torchvision.transforms.ToTensor(),
                                   target_transform = None
                                  )

# visualize the data
image, label = train_data[1]
#plt.imshow(image.squeeze())
plt.imshow(image.permute(1,2,0))
batch_size = 16 
# Dataloader 
loader = DataLoader(train_data, batch_size, shuffle=True)

# %%
def show_images(images):
    # turn images into numpy array
    if type(images) is torch.Tensor: 
        images = images.detach().cpu().numpy()
    # figure size
    fig = plt.figure(figsize=(8,8)) # 8x8 image size
    rows = int(len(images) ** (1/2) ) # root of size
    colms = int(len(images) / rows)
    
    idx = 0
    for r in range(rows):
        for c in range(colms):
            # add image to subplot
            fig.add_subplot(rows, colms, idx + 1)
            if idx < len(images): 
                plt.imshow(images[idx][0], cmap="gray")
                idx += 1
    # Showing the figure
    plt.show()    

# Shows the first batch of images
def show_first_batch(loader):
    for idx, batch in enumerate(loader):
        show_images(batch[0])
        break

show_first_batch(loader)
# %%
# device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

class DDPM(nn.Module): 
    def __init__(self, 
                 network, 
                 n_steps = 200, 
                 start_beta_schedule = 10**(-4), 
                 end_beta_schedule = 0.02,
                 image_dim = (1, 28, 28),
                 device = "cpu"): 
        super(DDPM, self).__init__()
        # Init
        self.n_steps = n_steps
        self.device = device
        self.image_dim = image_dim
        self.network = network.to(device)
        # Pre-defined 
        betas = torch.linspace(start_beta_schedule, end_beta_schedule, n_steps).to(device)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas)

    def forward(self, x0, t, eps): 
        # get components of image
        n, c, h, w = x0.shape
        alphas_cumprod_t = self.alphas_cumprod[t]

        # produce random noise: 
        if eps is None: 
            noise = torch.randn_like(x0)
        # reparam trick 
        # reshape is similar to gather()
        x0_noised = torch.sqrt(alphas_cumprod_t).x0.reshape(n, 1, 1, 1) + torch.sqrt(1 - alphas_cumprod_t).eps.reshape(n, 1, 1, 1)
        x0_noised = x0_noised.to(self.device)
        return x0_noised.to(self.device)
    
    def backward(self, x0, t): 
        # Unet
        return self.network(x0, t)

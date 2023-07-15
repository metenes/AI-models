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



# temporary here
#%%

# Import Pytorch
import torch
from torch import nn
import numpy as np
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
import torch.nn.functional as F
from PIL import Image
# tqdm 
from torch.optim import Adam
from tqdm import tqdm



#%% 
# Datasets 

train_data = torchvision.datasets.CIFAR10(root = ".", 
                                          train= True, 
                                          transform= torchvision.transforms.ToTensor(),
                                          download= True,
                                          target_transform=None)


test_data = torchvision.datasets.CIFAR10(root = ".", 
                                          train= True, 
                                          transform= torchvision.transforms.ToTensor(),
                                          download= True,
                                          target_transform= None)

""" 
for idx, data in enumerate(test_data):
    print(idx)
    (image,label) = data
    plt.imshow(image.permute(1,2,0))
"""

#%% 

from torch.optim import Adam
from tqdm import tqdm

# DDPM class 
class DDPM: 
    def __init__( self, 
                  timestep = 1000,
                  betas_start = 3e-4,
                  betas_end = 0.02,
                  img_size = 64, 
                  device = "cpu" ):
        # selfs
        self.timestep = timestep
        self.betas_start = betas_start
        self.betas_end = betas_end
        self.img_size = img_size
        self.device = device
        # pre_defined varaibles 
        self.betas = linear_beta_schedule(self.betas_start, self.betas_end, timestep).to(self.device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas)

        def linear_beta_schedule(betas_start, betas_end, timestep ):
          return torch.linspace(betas_start, betas_end, timestep )
        
        def noise_image(self, x0, t): 
          # In this option we use cumulative preoduct to get the noised image in one time
          # get random noise 
          noise = torch.rand_like(x0)
          sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
          sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
          # reparm trick 
          return sqrt_alphas_cumprod.to(device)*x0.to(device) + sqrt_one_minus_alphas_cumprod.to(device)*noise.to(device), noise.to(device)
        def sample_rand_timestep(self, n): 
           return torch.randint(low= 1, high = self.timestep, size=(n,))
        
        def sample_img(self, model, n): 
          # produce n new generated images 
          # set model to evualte mode
          model.eval()
          with torch.no_grad(): 
            # n : number of gaussian distr
            # c : number of channels 
            c = 3
            x_T = torch.randn(n, c, self.img_size, self.img_size).to(self.device)
            for i in tgdm(range(1,self.timestep)): 

# %%
          
          
          
          
          


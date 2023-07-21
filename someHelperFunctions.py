#%%
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
import os 
# Image settings
from nilearn import plotting
import pylab as plt

import numpy as np
import nibabel as nb

# Access data test directly from second folder
img = nb.load(filename = "YOUR_FOLDER_PATH_NOT_IMGs_PATH")
print(img.shape)
# shape of one the images is (256, 320, 320)
print(f"shows the type of the data on disk {img.get_data_dtype()}")
# get the numpy array with fdata()
data_nummpy = img.get_fdata()
# display the nii.gz 
plt.imshow(data_nummpy[:, :, data_nummpy.shape[2] // 2].T, cmap='Greys_r')
print(data_nummpy.shape)

# %%

def display_data_from_dataset(file_dir_path, num_samples=20, cols=4):

    """ Plots some samples from the dataset """
    plt.figure(figsize=(15,15)) # figure size for display
    i = 0 
    list_tensor_imgs = []
    for filename in os.listdir(file_dir_path):
        if i == num_samples : 
            break
        if filename.endswith('.nii.gz'): # niffy 
            nb.load( os.path.join(file_dir_path, filename) ) 
            print(os.path.join(file_dir_path, filename)) # file names
            data_nummpy = img.get_fdata()
            list_tensor_imgs.append(torch.from_numpy(data_nummpy)) # save the numpy to torch tensor
            plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
            plt.imshow(data_nummpy[:, :, data_nummpy.shape[2] // 2].T, cmap='Greys_r')
            i += 1 
    return list_tensor_imgs
    
file_dir_path = "YOUR_FOLDER_PATH_NOT_IMGs_PATH"
list_tensor_imgs = display_data_from_dataset(file_dir_path, 30)

print(list_tensor_imgs[0].dtype) # dtype = float64 , type = torch.float64 tensor
print(len(list_tensor_imgs))

# split array 
train_data_initial = list_tensor_imgs[0:int(len(list_tensor_imgs)*(1/3))]
print(len(train_data_initial))
test_data_initial = list_tensor_imgs[int(len(list_tensor_imgs)*(1/3)):]
print(len(test_data_initial))


# %%
# Custom Datasets 
import os
import pathlib
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Custom Dataset 
class CustomDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
    def __getitem__(self, idx):
        image = self.images[idx]      
        return image
    def __len__(self):
        return len(self.images)

# custom dataset for the niffty file MRI 
custom_dataset = CustomDataset(list_tensor_imgs)

# Data visualizer for datasets
def visualize_dataset(dataset, num_images = 20): 
    fig = plt.figure(figsize=(20, 20))
    for idx in range(0,num_images): 
        data = dataset[idx]
        ax = fig.add_subplot(1, num_images, idx+1)
        # print(f"data shape {data.shape}") # Debug 
        ax.imshow(data[:, :, data.shape[2] // 2].T, cmap='Greys_r')
    plt.show()

visualize_dataset(custom_dataset)

# load dataset to the dataloader
BATCH_SIZE = 128
dataloader = DataLoader(custom_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# %%
# Sampling
@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    # print(x.shape)
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

@torch.no_grad()
def sample_plot_image():
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 1, img_size, img_size), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')

    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu())
    plt.show() 


# %%
# Training / Save
from torch.optim import Adam

def saveCheckPoint(state, filename = "my_checkpoint.pth.tar"): 
    print("-- Checkpoint reached --")
    torch.save(state,filename)

def loadCheckPoint(state):
    print(" Checkpoint loading ")
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])


# device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 100

load_model = False
load_model_filename = "my_checkpoint.pth.tar"

if(load_model == True): 
    loadCheckPoint(torch.load(load_model_filename))

for epoch in range(epochs):
    for step, batch in enumerate(tensor_littel_all_images_dataloader):
      optimizer.zero_grad()

      t = torch.randint(0, T, (1,), device = device).long()
      loss = get_loss(model, batch.to(device), t)
      loss.backward()
      optimizer.step()

      if step % 16 == 0 :
        # save the model
        checkpoint = {'state_dict' : model.state_dict(), 
                      'optimizer': optimizer.state_dict() }
        saveCheckPoint(checkpoint)

        print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
        sample_plot_image()
        # andb.log({"epoch": epoch, "loss": loss}, step=step)

# U-net Arthitecture \ Backward

import math
from typing import Optional, Tuple, Union, List
import torch
from torch import nn
from labml_helpers.module import Module

# Swish actiavation function
class Swish(Module):
    def forward(self, x):
        return x * torch.sigmoid(x) 
    
# classic time embedding 
class TimeEmbedding(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels  
        self.linear_1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.linear_2 = nn.Linear(self.n_channels, n_channels)
        self.activation = Swish()

    def forward(self, t: torch.Tensor):

        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.activation(self.linear_1(emb))
        emb = self.linear_2(emb)
        return emb
    
"""
### Residual block
 A residual block has two convolution layers with group normalization.
 Each resolution is processed with two residual blocks.
"""
class ResidualBlock(Module):

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 time_channels: int,
                 n_groups: int = 32, 
                 dropout: float = 0.1):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        * `dropout` is the dropout rate
        """
        super().__init__()
        # for x_0
        self.norm_1 = nn.GroupNorm(n_groups, in_channels)
        self.activation_1 = Swish()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=(1,1))

        self.norm_2 = nn.GroupNorm(n_groups, out_channels)
        self.activation_2 = Swish()
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=(1,1))

        # To adjust the dimension if out != in dimension 
        if in_channels != out_channels: 
            self.adjust = nn.Conv2d(in_channels, out_channels,  kernel_size=(1, 1))
        else : 
            self.adjust = nn.Identity() # Identity block = conv layer with no filters = no kernel

        # For t = time embedding 
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_activation = Swish()

    def forward(self,
                x: torch.Tensor, 
                t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # first convolution layer 
        hidden = self.conv_1(self.activation_1(self.norm_1(x)))
        # time layer
        hidden += self.time_emb(self.time_activation(t))[:, :, None, None]
        # second convolution layer 
        hidden = self.conv_2(self.activation_2(self.norm_2(hidden))) 
        # adjust the dimension 
        return hidden + self.adjust(x) 
    
"""
### Attention block = BLACK BOX 
"""
class AttentionBlock(Module):


    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k ** -0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=2)
        # Multiply by values
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        #
        return res

class DownBlock(Module):
    """
    ### Down block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 time_channels: int, 
                 has_attn: bool):
        super().__init__()
        self.resBlock = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn : 
            self.att = AttentionBlock(out_channels)
        else: 
            self.att = nn.Identity() # Identity block = conv layer with no filters = no kernel
     
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.resBlock(x,t)
        x = self.att(x)
        return x 

    """
    ### Up block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """
class UpBlock(Module):

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution = skip connections 
        # from the first half of the U-Net

        self.resBlock = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn : 
            self.att = AttentionBlock(out_channels)
        else: 
            self.att = nn.Identity() # Identity block = conv layer with no filters = no kernel
     
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.resBlock(x,t)
        x = self.att(x)
        return x 

class MiddleBlock(Module):
    """
    ### Middle block
    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.resBlock_1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.resBlock_2 = ResidualBlock(n_channels, n_channels, time_channels)
        self.att = AttentionBlock(n_channels)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.resBlock_1(x,t)
        x = self.att(x)
        x = self.resBlock_2(x,t)
        return x

class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)
# Training
# Training
from typing import List
import torch
import torch.utils.data
import torchvision
from PIL import Image
from labml import lab, tracker, experiment, monit
from labml.configs import BaseConfigs, option
from labml_helpers.device import DeviceConfigs
from labml_nn.diffusion.ddpm import DenoiseDiffusion
from labml_nn.diffusion.ddpm.unet import UNet

# checkpoint features 
load_model = True
load_model_filename = "my_checkpoint_4.pth.tar"
save_model_filename = "my_checkpoint_5.pth.tar"

def saveCheckPoint(state, filename = "my_checkpoint_5.pth.tar"): 
    print("-- Checkpoint reached --")
    torch.save(state,filename)

def loadCheckPoint(state):
    print(" Checkpoint loading ")
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])

# checkpoint features 
load_model = False
load_model_filename = "my_checkpoint_new.pth.tar"
save_model_filename = "my_checkpoint_new.pth.tar"

def saveCheckPoint(state, filename = "ERROR"): 
    print("-- Checkpoint reached --")
    torch.save(state,filename)

def loadCheckPoint(state, model, optimizer ):
    print(" Checkpoint loading ")
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])

class Configs():
    def init(self):

        # Same as writting 
        # device agnostic code
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # device: torch.device = DeviceConfigs()
        # Number of channels in the image. 3 for RGB.
        self.image_channels: int = 1 # as gray MRI image
        # Image size
        self.image_size: int = 320 

        # Unet parameters
        # Number of channels in the initial feature map
        self.n_channels: int = 64
        # The list of channel numbers at each resolution.
        # The number of channels is `channel_multipliers[i] * n_channels`
        self.channel_multipliers: List[int] = [1, 2, 2, 4]
        # The list of booleans that indicate whether to use attention at each resolution
        self.is_attention: List[int] = [False, False, False, True]
    
        # Number of time steps $T$
        self.n_steps: int = 150
        # Batch size
        self.batch_size: int = 4
        # Number of samples to generate
        self.n_samples: int = 16
        # Learning rate
        self.learning_rate: float = 2e-5
        # Number of training epochs
        self.epochs: int = 10000
        # Dataset
        self.dataset: torch.utils.data.Dataset
        # Dataloader
        self.data_loader: torch.utils.data.DataLoader
        # Adam optimizer
        self.optimizer: torch.optim.Adam

        # Unet init with Unet parameters
        self.eps_model = UNet(
                                image_channels=self.image_channels,
                                n_channels=self.n_channels,
                                ch_mults=self.channel_multipliers,
                                is_attn=self.is_attention, ).to(self.device)

        # Create ddpm class 
        self.ddpm = DDPM(
                        eps_model=self.eps_model,
                        n_steps=self.n_steps,
                        device=self.device, )
        # Create Dataset 
        self.dataset = tensor_littel_all_images_dataset
        # Create dataloader
        self.data_loader = DataLoader(self.dataset, batch_size= self.batch_size, shuffle=True, drop_last=True)
        # Create optimizer
        self.optimizer = torch.optim.Adam( self.eps_model.parameters(), lr = self.learning_rate)

    def sample(self):
        """
        ### Sample images
        """
        with torch.no_grad():
            # $x_T 
            x = torch.randn([self.n_samples, self.image_channels, self.image_size, self.image_size],device=self.device )
            print("currently sample ")
            # Remove noise for current t step 

            plt.figure(figsize=(15,15))
            plt.axis('off')
            for idx in  range(self.n_steps):
                print(f"Sampling step : {idx}")
                plt.subplot(1, self.n_samples + 1, idx + 1)
                # generate images 
                t = self.n_steps - idx - 1
                x = self.ddpm.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))
                show_tensor_image(x.to(device))


    def train(self):
        """
        ### Train
        """
        # Iterate through the dataset
        # five steps of the train process

        if(load_model == True): 
            # loadCheckPoint(torch.load(load_model_filename, self.eps_model, self.optimizer, map_location=torch.device('cpu')))
            loadCheckPoint(torch.load(load_model_filename), self.eps_model, self.optimizer)

        with torch.no_grad():
         for epoch in range(self.epochs):
            for step, batch in enumerate(self.data_loader):
                print(f"currently at step {step}, epoch {epoch}")
                # optimizer zero_grad
                self.optimizer.zero_grad()
                # rand time 
                t = torch.randint(0, self.n_steps, (1,), device = device).long()
                # loss
                loss = self.ddpm.loss(batch.to(device)).float()
                loss.requires_grad = True # need to get the gradients before loss.backward()
                # backward
                loss.backward()
                # step
                optimizer.step()
                if step % 16 == 0 :
                    # print(model.state_dict())
                    # save the model
                    checkpoint = {'state_dict' : self.eps_model.state_dict(), 'optimizer': self.optimizer.state_dict() }
                    saveCheckPoint(checkpoint, save_model_filename)
                    print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                    # self.sample() # TODO fix the bug freeze in worklflow
    
    def generate(self): 
        if torch.cuda.is_available : 
            state = torch.load(load_model_filename, map_location=torch.device('cpu'))
        else : 
            state = torch.load(load_model_filename)

        self.eps_model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        # loadCheckPoint(state,  self.eps_model, self.optimizer)
        for i in range(self.epochs):
            self.sample()

# Create configurations
configs = Configs()



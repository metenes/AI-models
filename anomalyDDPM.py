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
img = nb.load(filename = "FILE PATH")
print(img.shape)
# shape of one the images is (256, 320, 320)
print(f"shows the type of the data on disk {img.get_data_dtype()}")
# get the numpy array with fdata()
data_nummpy = img.get_fdata()
# display the nii.gz 
plt.imshow(data_nummpy[:, :, data_nummpy.shape[2] // 2].T, cmap='Greys_r')
print(data_nummpy.shape)

# %%
# Display all the nii.gz images in dataset
def prep_data_from_dir(file_dir_path, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    i = 0 
    list_tensor_imgs = []
    # list_img_paths = []
    for filename in os.listdir(file_dir_path):
        if i == num_samples : 
            break
        if filename.endswith('.nii.gz'): # niffy 
            nb.load( os.path.join(file_dir_path, filename) ) 
            print(os.path.join(file_dir_path, filename)) # file names
            data_nummpy = img.get_fdata()
            list_tensor_imgs.append(torch.from_numpy(data_nummpy)) # save the numpy to torch tensor
            # list_img_paths.append(os.path.join(file_dir_path, filename)) # save the file_path
            i += 1 
    return list_tensor_imgs
    
# Display all the nii.gz images in dataset
def display_data_from_dir(file_dir_path, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15,15)) # figure size for display
    i = 0 
    for filename in os.listdir(file_dir_path):
        if i == num_samples : 
            break
        if filename.endswith('.nii.gz'): # niffy 
            nb.load( os.path.join(file_dir_path, filename) ) 
            print(os.path.join(file_dir_path, filename)) # file names
            data_nummpy = img.get_fdata()
            plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
            plt.imshow(data_nummpy[:, :, data_nummpy.shape[2] // 2].T, cmap='Greys_r')
            i += 1 

file_dir_path = "FILE PATH"

list_tensor_volumes = prep_data_from_dir(file_dir_path, 100)
display_data_from_dir(file_dir_path, 20)

print(list_tensor_volumes[0].dtype) # dtype = float64 , type = torch.float64 tensor
print(len(list_tensor_volumes))

# split array 
train_data_initial = list_tensor_volumes[0:int(len(list_tensor_volumes)*(2/3))]
print(len(train_data_initial))
test_data_initial = list_tensor_volumes[int(len(list_tensor_volumes)*(2/3)):]
print(len(test_data_initial))

print(f"size of a volume {len(list_tensor_volumes[0])}" )
plt.imshow(list_tensor_volumes[0][128], cmap='Greys_r')
#%% 
"""
# WandB
# start a new wandb run to track this script
import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="DDPM initial-II",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "architecture": "DDPM",
    "dataset": "MRI dataset",
    "epochs": 100,
    }
) 
"""

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
    
# Data visualizer for datasets
def visualize_volume_dataset(dataset, num_images = 20): 
    fig = plt.figure(figsize=(20, 20))
    for idx in range(0,num_images): 
        data = dataset[idx]
        ax = fig.add_subplot(1, num_images, idx+1)
        # print(f"data shape {data.shape}") # Debug 
        ax.imshow(data[:, :, 0].T, cmap='Greys_r')
    plt.show()

# custom dataset for the niffty file MRI 
volume_custom_dataset = CustomDataset(list_tensor_volumes)
# visualize all volumes in half view
visualize_volume_dataset(volume_custom_dataset)

# Display the tensor images
def show_tensor_image(image): 
    if len(image.shape) == 4: 
        image = image[0,:,:,:]
    plt.imshow(image.permute(1,2,0), cmap='Greys_r')


# For all volumes 
 
# concat all volumes into single list
list_tensor_all_volumes = torch.stack(list_tensor_volumes) 
tensor_all_volumes = torch.DoubleTensor(list_tensor_all_volumes)
# reshape : get all images inside all voluems into single list [ Number of imgs, H, W ]
tensor_all_images = torch.reshape(tensor_all_volumes, [-1, 320, 320])
# add the color channel 
tensor_all_images = tensor_all_images.unsqueeze(1)
print(tensor_all_images.shape) # torch.Size([25600, 1, 320, 320])
# dataset
tensor_all_images_dataset = CustomDataset(tensor_all_images)
plt.imshow( tensor_all_images_dataset[0].permute(1,2,0) , cmap='Greys_r' )

"""
# concat all volumes into single list
list_tensor_littel_all_volumes = torch.stack( [ list_tensor_volumes[0], list_tensor_volumes[1], list_tensor_volumes[2], list_tensor_volumes[3], list_tensor_volumes[4], list_tensor_volumes[5] ]) 
tensor_littel_all_volumes = torch.DoubleTensor(list_tensor_littel_all_volumes)
# reshape : get all images inside all voluems into single list [ Number of imgs, H, W ]
tensor_littel_all_images = torch.reshape(tensor_littel_all_volumes, [-1, 320, 320])
# add the color channel 
tensor_littel_all_images = tensor_littel_all_images.unsqueeze(1)

print(tensor_littel_all_images.shape) # torch.Size([512, 1, 320, 320])
show_tensor_image(tensor_littel_all_images[0])
# dataset
tensor_littel_all_images_dataset = CustomDataset(tensor_littel_all_images)
plt.imshow( tensor_littel_all_images_dataset[0].permute(1,2,0) , cmap='Greys_r' )
"""

#%% 
# # Helper functions 
import json
import os
from collections import defaultdict
import torch
import torchvision.utils

# scale the colar chaneels int the format of [0, 1]
def scale_image(img): 
    return ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)

# add grid to output
def grid_output(img, row_size= -1):
    scaled_img = scale_image(img)
    return torchvision.utils.make_grid(scaled_img, nrow=row_size, pad_value=-1).cpu().data.permute(0, 2, 1).contiguous().permute(2, 1, 0)

def saveCheckPoint(state, filename = "my_checkpoint_anomaly.pth.tar"): 
    print("-- Checkpoint reached --")
    torch.save(state,filename)

def loadCheckPoint(state, model, optimizer ):
    print(" Checkpoint loading ")
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

# For Attention module 
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

#%% 
# Unet 
import math
from abc import abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# Standart positionel embedding 
class PositionalEmbedding(nn.Module): 
    def __init__(self, dim, scale = 1): 
        super().__init__()
        # dim must be mod2 = 0
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale
    def forward(self, x): 
        # device agnostic code
        device = x.device
        half_dim = self.dim // 2
        # use the formula of time embeddings 
        emb = np.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample(nn.Module): 
    def __init__(self, in_ch, use_conv, out_ch = None): 
        super().__init__()
        self.ch = in_ch
        out_ch = out_ch or in_ch # if out_ch =None, we want it to be in_ch 
        if use_conv: 
            # half the channels with classic half dimension conv layer
            self.down_sample = nn.Conv2d(in_ch, out_ch, 3, stride=2,padding=1)
        else : 
            # max-pooling 
            self.down_sample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, time_emb = None): 
        assert x.shape[1] == self.ch 
        return self.down_sample(x)

class Upsample(nn.Module): 
    def __init__(self, in_ch, use_conv, out_ch = None): 
        super().__init__()
        self.ch = in_ch
        self.use_conv = use_conv
        if use_conv : 
            self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x, time_emb = None): 
        assert x.shape[1] == self.ch 
        # interpolate the cordinates 
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv : 
            x = self.conv(x)
        return x
    
# This is the Abstract class for ResBlock
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """
# Black Box - Attentions 
class AttentionBlock(nn.Module):

    def __init__(self, in_channels, n_heads=1, n_head_channels=-1):
        super().__init__()
        self.in_channels = in_channels
        self.norm = GroupNorm32(32, self.in_channels)
        if n_head_channels == -1:
            self.num_heads = n_heads
        else:
            assert (
                    in_channels % n_head_channels == 0
            ), f"q,k,v channels {in_channels} is not divisible by num_head_channels {n_head_channels}"
            self.num_heads = in_channels // n_head_channels

        # query, key, value for attention
        self.to_qkv = nn.Conv1d(in_channels, in_channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(nn.Conv1d(in_channels, in_channels, 1))

    def forward(self, x, time=None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.to_qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)
class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, time=None):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
                "bct,bcs->bts", q * scale, k * scale
                )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)
    
# This ResBlock contain dropouts. It is a method to prevent the overfitting problem
# Each channel will be zeroed out independently on every forward call.
class ResBlock(TimestepBlock): 
    def __init__(
            self,
            in_ch,
            time_embed_dim,
            dropout,
            out_ch = None,
            use_conv = False,
            up=False,
            down=False ):
        super().__init__() # timestep inh. nn.Module 
        out_ch = out_ch or in_ch
        self.layers = nn.Sequential(
                        GroupNorm32(32, in_ch), 
                        nn.SiLU(), 
                        nn.Conv2d(in_ch, out_ch, 3, padding= 1)
                    )
        # indicates if Res network is used for up or down or just identity 
        self.updown = up or down 

        if up :
            self.h_upd = Upsample(in_ch, False)
            self.x_upd = Upsample(in_ch, False)
        elif down: 
            self.h_upd = Downsample(in_ch, False)
            self.x_upd = Downsample(in_ch, False)
        else: 
            self.h_upd = nn.Identity()

        self.embed_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, out_ch)
                )

        self.out_layers = nn.Sequential(
                GroupNorm32(32, out_ch),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                zero_module(nn.Conv2d(out_ch, out_ch, 3, padding=1))
                )

        # Skip connection 
        if out_ch == in_ch: 
            self.skip = nn.Identity() 
        elif use_conv: 
            self.skip = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        else: 
            self.skip = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, time_embed): 
        if self.updown: 
            # in self.layers, 
            #   we got conv layer in conv and ReLU and Norm in others
            in_others, in_conv = self.layers[:-1], self.layers[-1]
            h = in_others(x)
            h = self.h_upd(x)
            x = self.x_upd(x)
            h = in_conv(h)
        else: 
            h = self.layers(x)
        emb_out = self.embed_layers(time_embed).type(h.dtype)
        
#%% 
# Testing 
import matplotlib.pyplot as plt
import torch
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import auc, roc_curve

# print the heat map between real image and reconstr. image 
def heatmap(real : torch.Tensor, 
            reconst : torch.Tensor, 
            mask): 
    # mean square error 
    mse = ((reconst - real).square() * 2) - 1
    threshold = mse > 0 # get a boolean 0 or 1 
    threshold = (threshold.float()*2)-1 # use the boolean value to turn -1 or 1
    # show the image 
    # dim = 0, stack the layers vertically
    output = torch.cat((real, reconst.reshape(1, *reconst.shape), mse, threshold, mask))
    plt.imshow(grid_output(output, 5)[..., 0], cmap="gray")



#%% 
# We need beter noise types so we will use simplex noise 
import random

import matplotlib.pyplot as plt
import numpy as np

import evaluation
from helpers import *
from simplex import Simplex_CLASS
def beta_schedule(step, 
                  name = "cosine"): 
    betas = []
    if name == "cosine": 
        max_beta = 0.999
        # define lambda function expression 
        f = lambda t : 

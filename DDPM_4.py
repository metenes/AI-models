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
img = nb.load(filename = "<FILENAME>")
print(img.shape)

# shape of  the images is (256, 320, 320)
print(f"shows the type of the data on disk {img.get_data_dtype()}")
# get the numpy array with fdata()
data_nummpy = img.get_fdata()
# display the nii.gz 
plt.imshow(data_nummpy[:, :, data_nummpy.shape[2] // 2].T, cmap='Greys_r')
print(data_nummpy.shape)

# %%
import torchvision.transforms.functional as fn

def prep_data_from_dir(file_dir_path, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    i = 0 
    list_tensor_imgs = []
    # list_img_paths = []
    for filename in os.listdir(file_dir_path): 
        if i == num_samples : 
            break
        if filename.endswith('.nii.gz'): # niffy 
            img = nb.load( os.path.join(file_dir_path, filename)) 
            print(os.path.join(file_dir_path, filename)) # file names
            data_nummpy = img.get_fdata()
            list_tensor_imgs.append(torch.from_numpy(data_nummpy) ) # save the numpy to torch tensor
            # list_img_paths.append(os.path.join(file_dir_path, filename)) # save the file_path
            i += 1 
    return list_tensor_imgs
    
# Display all the nii.gz images in dataset
def display_data_from_dir(file_dir_path, num_samples=20, cols=4):
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
            print("\n")
            i += 1 
    
def normalize_data(list_imgs): 
    for idx, img_tensor in enumerate(list_imgs): 
       list_imgs[idx] = fn.normalize( img_tensor, mean=[0.5000], std=[.1000])
    return list_imgs

file_dir_path = "<FILENAME>"

# get tensor images
list_tensor_volumes = prep_data_from_dir(file_dir_path, 100)
# Normalize the list elemets (tensors in list)
list_tensor_volumes = normalize_data(list_tensor_volumes) 

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
# Helper functions
import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

# TODO 
# %%
# Unet 
import torch
import torch.nn as nn
import torch.nn.functional as F

# EMA is for updating model parameters - BLACK Box
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

# Self Attention - BLACK Box
class SelfAttention(nn.Module):
    def __init__(self, ch, size): 
        super(SelfAttention, self).__init__()
        # Init
        self.size = size
        self.ch = ch
        # multi head attention 
        self.mha = nn.MultiheadAttention(ch, 4, batch_first= True) 
        # linear layer Normalization 
        self.lin = nn.LayerNorm([ch])

        self.ff_self = nn.Sequential(
                    nn.LayerNorm([ch]),
                    nn.Linear(ch, ch),
                    nn.GELU(),
                    nn.Linear(ch, ch) )
        
    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)
    
class DoubleConv(nn.Module): 
    def __init__(self, in_ch, out_ch, mid_ch = None, residual = False): 
        super().__init__()
        # Init
        self.residual = residual
        if not mid_ch: 
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_ch) )
    def forward(self, x): 
        if self.residual: 
            # residual addition = skip connection 
            # GeLU(x + doubleconv(x))
            return F.gelu(x + self.double_conv(x))
        else: 
            self.double_conv(x)

# Down = maxpooling + 2 Conv 
# max pooling [B, C, H, W] -> [B, C, H_out, W_out] 
class Down(nn.Module): 
    def __init__(self, in_ch, out_ch, emb_dim = 256): 
        super().__init__()
        # for image 
        self.maxpool_conv = nn.Sequential(
                                nn.MaxPool2d(2), # Down 
                                # kernel size = 2
                                DoubleConv(in_ch, in_ch, residual=True),
                                DoubleConv(in_ch, out_ch),
                            )
        # for time embedding 
        self.emb_layer = nn.Sequential(
                            nn.SiLU(), 
                            nn.Linear(emb_dim, out_ch)
                        )
    
    def forward(self, x, t):
        x = self.maxpool_conv(x)
        # time emb
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        # the repeat is to increse the size fof time embeddig to cover all process in t 
        # repeat(1, 1, x.shape[-2], x.shape[-1]) is [1,1,320,320] means the t value will be assigned to every pixel 
        return x + emb
    
# Up = nn.Upsample + 2 Conv
class Up(nn.Module): 
    def __init__(self, in_ch, out_ch, emb_dim = 256):
        super().__init__()
        # for image
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_conv = nn.Sequential(
                            DoubleConv(in_ch, in_ch, residual=True),
                            DoubleConv(in_ch, out_ch, in_ch // 2),
                        )
        # for time embedding 
        self.emb_layer = nn.Sequential(
                            nn.SiLU(), 
                            nn.Linear(emb_dim, out_ch)
                        )
    def forward(self, x, skip_x, t): 
        # skip_x comes from the resudial blocks
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1) # combine x and resudial_x  to make a vector vertical
        x = self.up_conv(x)
        # time emb 
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return emb + x
    
class UNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
    
#%%
# DDPM model 
import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *

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
import torch.nn.functional as F

import wandb # weight and biases
# start a new experiment

class DDPM: 
    def __init__(self, noise_steps = 1000, beta_start=1e-4, beta_end=0.02, img_size=320, device="cuda"):
        # Init
        self.noise_steps = noise_steps 
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size        
        self.device = device 

        # pre-defined
        self.betas = self.beta_noise_schedule()
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alpha, dim=0)
        # dim = 0 means it is a vector horizantal = [ .. .. .. ]
        # dim = 1 means it is a vector vertical

    def beta_noise_schedule(self): 
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t): 
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - self.alphas_cumprod[t])[:, None, None, None]
        # gaussian noise = random 
        eps = torch.rand_like(x)
        # reparamzitaion trick
        # return noised image , noise
        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * eps , eps

    def sample_timesteps(self, n): 
        # return random timestep 
        # (n, ) is the vector shape 
        return torch.randint(low = 1, high= self.noise_steps, size= (n,))
    
    def sample(self, model, n, color_ch = 1): 
        # start evaluation 
        model.eval()
        with torch.no_grad(): # CUDA out of memory
            # produce random images in given format (batch, color, imgsize, imgsize) = x_T
            x = torch.randn( (n, color_ch, self.img_size, self.img_size ) ) 
            # go through the timesteps 
            for i in tqdm(reversed(range(1, self.noise_steps)), position= 0): 
                # produce t as batch number size for each elememt
                t = (torch.ones(n) * i).long().to(self.device)
                # Backward process = predict the noise 
                noise_pred = model(x, t)
                # use predicted noise to get the d-noised image 
                # [:, None, None, None] part is to take only unigue timestep of positionel embeddings
                alphas_t = self.alphas[t][:, None, None, None]
                alphas_cumprod_t = self.alphas_cumprod[t][:, None, None, None]
                betas_t = self.betas[t][:, None, None, None]
                if i > 1 : 
                    noise = torch.rand_like(x) 
                else :
                    # noise is zero at last step 
                    noise = torch.zeros_like(x) 
                x = 1 / torch.sqrt(alphas_t) * (x - ((1 - alphas_t) / (torch.sqrt(1 - alphas_cumprod_t))) * noise_pred) + torch.sqrt(betas_t) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
    
def train():
        # constants
        BATCH_SIZE = 8
        IMG_SIZE = 320
        epochs = 1000
        lr = 3e-4
        # device agnostic code
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # model 
        model = UNet().to(device)
        # dataloader and shuffel
        dataloader = DataLoader(tensor_all_images_dataset, batch_size= BATCH_SIZE, shuffle=True, drop_last=True)
        dataloader_size = len(dataloader)
        # optim 
        optimizer = optim.AdamW(model.parameters(), lr = lr)
        # loss 
        loss_fn = nn.MSELoss()
        # ddpm model
        ddpm = DDPM(img_size= IMG_SIZE, device= device)
        with torch.no_grad(): # CUDA out of memory
         for epoch in range(epochs): 
            for idx, (images, _) in enumerate(tqdm(dataloader)): 
                # device agnostic code
                images = images.to(device)
                # get random time value t
                t = ddpm.sample_timesteps(images.shape[0]).to(device)
                # forward 
                # use t to calculate the noised image at t and the amount of noise at t
                x_t, noise_real = ddpm.noise_images(images, t)
                # backward 
                # predict the noise by model
                noise_pred = model(x_t, t)
                # calcuate the difference between the noise_pred and noise_reel
                loss = loss_fn(noise_real, noise_pred)
#%% 
# start train
train()
# %%

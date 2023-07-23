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
img = nb.load(filename = "YOUR_FILE_NAME")
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

file_dir_path = "YOUR_FILE_NAME"

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
""" """
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
# DDPMs 
from torch.optim import Adam
from tqdm import tqdm

# DDPM class 
class DDPM: 
    def __init__( self, 
                  timestep = 1000,   
                  betas_start = 3e-4,
                  betas_end = 0.02,  
                  img_size = 64, 
                  device = "cpu"    ):
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
            for i in tgdm(reversed(range(1,self.timestep)), position = 0): 
              t = (torch.ones(n) * i).long().to(self.device)
              noise_pred = model(x_T,t)
              # parameters for prediction fucntion 
              alphas_cur = self.alphas[t][:, None, None, None]
              alphas_cumprod_cur = self.alphas_cumprod[t][:, None, None, None]
              betas_cur = self.betas[t][:, None, None, None]
              if i > 1 : 
                noise = torch.randn_like(x_T)
              else : 
                noise = torch.zeros_like(x_T) # in last step we dont need to add noise 
              # calculate the de noised image by 
              # mean + varaince
              x_T = 1 / torch.sqrt(alphas_cur) * (x - ((1 - alphas_cur) / (torch.sqrt(1 - alphas_cumprod_cur))) * noise_pred) + torch.sqrt(betas_cur) * noise

          model.train()
          x_T = (x_T.clamp(-1, 1) + 1) / 2
          x_T = (x_T * 255).type(torch.uint8)
          return x_T
          

# %%
# Unet Helper classes
import torch
import torch.nn as nn
import torch.nn.functional as F

# dont know need to look attenstion 
class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

# DoubleConv class
class DoubleConv(nn.Module): 
  def __init__(self, 
               in_ch, 
               out_ch, 
               mid_chs = None, 
               skip_con = False):
    super().__init__()
    # the skip connections 
    self.skip_con = skip_con
    if mid_chs == False: 
      mid_chs = out_ch # if there is no mid_chs dimensions must be conserved
    self.double_conv = nn.Sequential(
                        nn.Conv2d(in_ch, mid_chs, kernel_size= 3, padding= 1, bias= False), 
                        nn.GroupNorm(1, mid_chs), # Normaliztion 
                        nn.GELU(), # nn.ReLU would also work 
                        nn.Conv2d(mid_chs, out_ch, kernel_size= 3, padding= 1, bias= False),
                        nn.GroupNorm(1, out_ch), # Normaliztion 
                      )
  
  def forward(self, x): 
    # need to get the x value in up case if skip connection is presented
    if self.skip_con == True: 
      return F.gelu(x + self.double_conv(x))
    return self.double_conv(x)
  
# Down module 
class Down(nn.Module): 
  def __init__(self, in_ch, out_ch, time_emb_dim = 256):
    super().__init__() 
    self.down_conv = nn.Sequential(
                      nn.MaxPool2d(2), 
                      DoubleConv(in_ch, in_ch, skip_con=True), 
                      DoubleConv(in_ch, out_ch)
                    )
    # time_embbeding
    self.time_emb = nn.Sequential(
                      nn.SiLU(), 
                      nn.Linear(time_emb_dim, out_ch)
                    )

  def forward(self, x, t): 
    x = self.down_conv(x)
    emb = self.time_emb(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
    return x + emb

class Up(nn.Module): 
  def __init__(self, in_ch, out_ch, time_emb_dim = 256): 
    super().__init__()
    # up sampling 
    self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True) # directly used standadart libary
    self.up_conv = nn.Sequential(
                    DoubleConv(in_ch, in_ch, skip_con=True), 
                    DoubleConv(in_ch, out_ch)
                          )
#%% 
# Unet 
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
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
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

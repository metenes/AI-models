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
class Unet(nn.Module):
  def __init__(self, 
               in_dim = 3, 
               out_dim = 3, 
               time_dim = 256, 
               device ="cpu"):
    super().__init__()
    # init
    self.device = device 
    self.time_dim =  time_dim
    # Down layers
    self.in_layer = DoubleConv(in_dim, 64)
    self.down1 = Down(64,128)
    self.att1 = SelfAttention(128, 32)
    self.down2 = Down(128,256)
    self.att2 = SelfAttention(256, 16)
    self.down3 = Down(256,256)
    self.att3 = SelfAttention(256, 8)
    # Middel layers
    self.bot1 = DoubleConv(256, 512)
    self.bot2 = DoubleConv(256, 512)
    self.bot4 = DoubleConv(256, 512)
          
          
          
          

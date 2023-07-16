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
# DoubleConv class
class DoubleConv: 
  
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
    # layers
    self.in_layer = DoubleConv(in_dim, 64)

          
          
          
          

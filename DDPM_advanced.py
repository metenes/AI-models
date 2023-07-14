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
# DDPM methods
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
import torch.nn.functional as F

import wandb # weight and biases
# start a new experiment

BATCH_SIZE = 8
IMG_SIZE = 320

# dataloader and shuffel
tensor_littel_all_images_dataloader = DataLoader(tensor_all_images_dataset, batch_size= BATCH_SIZE, shuffle=True, drop_last=True)

# Helper methods 
def linear_beta_schedule(timestep, start = 0.0001, end = 0.02): 
    """" Linear schedule for noise to be added not cosine-schdule"""

    # linspace: Creates a one-dimensional tensor of size steps whose values 
    # # are evenly spaced from start to end, inclusive
    return torch.linspace(start , end, timestep)

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device = 'cpu'): 
    """" Forward step of DDPM 
         add a Gausian(random) noise to current state of the distrubution
         and return the noised state with respect to given timestep """

    # create random noise
    noise = torch.randn_like(x_0)
    # print(f"noise type {noise.dtype}") # noise is float.64
    # print(noise.shape) # shape is [256, 320, 320]

    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod,t,x_0.shape)
    # print(sqrt_alphas_cumprod_t.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod,t,x_0.shape)
    # print(sqrt_one_minus_alphas_cumprod_t.shape)
    # produce noise by multiplying the initial image tensor with cumulative mult. of alphas 
    # x_0
    # noised_image = sqrt_alphas_cumprod_t*x_0 + sqrt_one_minus_alphas_cumprod_t*noise
    return sqrt_alphas_cumprod_t.to(device=device)*x_0.to(device=device) + sqrt_one_minus_alphas_cumprod_t.to(device=device)*noise.to(device=device) , noise.to(device=device)

# Pre-calculated values 
T = 300 # last t value
betas = linear_beta_schedule(timestep=T, start=0.05, end= 0.2) 
alphas = 1 - betas   # alpha = 1 - beta
alphas_cumprod = torch.cumprod(alphas, axis= 0) # cumulative product

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

alphas_cumprod_prev = F.pad( alphas_cumprod[:-1],(1,0),value=1.0 ) # seperate the first value from others
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


# %% 
# Simulate forward diffusion
image = next(iter(tensor_littel_all_images_dataloader))[0]
print(image.dtype)

plt.figure(figsize=(15,15))
plt.axis('off')
num_images = 10
stepsize = int(T/num_images)

for idx in range(0, T, stepsize):
    t = torch.Tensor([idx]).type(torch.int64)
    plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
    img, noise = forward_diffusion_sample(image, t)
    show_tensor_image(img)


# %%
# U-net Arthitecture \ Backward

import torch
import math 
from torch import nn

# different than classical Unet, this one has time_emb_dim 
class Block(nn.Module): 
    def __init__(self, in_ch, out_ch, time_emb_dim, up = False):
        super().__init__() # nn super()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up : 
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding = 1) 
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4,2,1)
        else: 
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding = 1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4,2,1)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t):
        # print(f"CHECK block CONV1 { x.shape} ")
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # print(f"CHECK AFTER block CONV1 {  h.shape} ")
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        #print(f"CHECK BEFORE block END {  h.shape} ")
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        # print(f"CHECK AFTER block END {  h.shape} ")
        return self.transform(h)
    
class SinusoidalPositionEmbeddings(nn.Module): # about transformers need to look 
    # basically increase the rate of learning with a mathematical formula for placment of the words
    def __init__(self, dim): 
        super().__init__()
        self.dim = dim 

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SimpleUnet(nn.Module):
    """" simple, constant layerd unet artitecture """
    def __init__(self): 
        super().__init__()
        image_channels = 1 # RGB
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 1 # RGB
        # in_dim and out_dim must be same 
        # " need to give what you take"
        time_emb_dim = 32
        
        # Time embedding : about transformers 
        self.time_mlp = nn.Sequential(
                        SinusoidalPositionEmbeddings(time_emb_dim),
                        nn.Linear(time_emb_dim, time_emb_dim),
                        nn.ReLU()   
                    )
        
        # Initial layer
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding= 1)

        # Downsample 
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], time_emb_dim, False) for i in range(0, len(down_channels)-1)])

        # Upsample 
        self.ups = nn.ModuleList([Block(up_channels[i],up_channels[i+1], time_emb_dim, True) for i in range(0, len(up_channels)-1)])
        
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep): 
        # Embedd time --> transformator 
        t = self.time_mlp(timestep)
        # Initial conv
        # print(f"CHECK  { x.shape} ")
        x = self.conv0(x)
        # print(f"CHECK AFTER CONV0  { x.shape} ")
        # Unet with time embeddings
        residual_inputs = [] # store the result of every forward process / noised image
        for down in self.downs: # iterate block of unet from ModuleList
            x = down(x,t) # apply the time embadding and input and get the new image --> markov chain 
            residual_inputs.append(x)
        for up in self.ups : 
            residual_x = residual_inputs.pop()
             # Add residual x as additional channels --> copy and crop lines between two side of the U net layers
             # this is why there is mult with 2 in up convlations it double the size with cat
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        return self.output(x)

# Initialize the model 
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleUnet().to(device)

#%%

# loss function 
def get_loss(model, x_0, t): 
    # device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_noisy, noise = forward_diffusion_sample(x_0, t,device=device)
    x_noisy = x_noisy.float()
    # giving the noised version of the x_0 to the Neural Networlk to make predictions about the noise
    # print(f"CHECK  { x_noisy.shape} ")
    noise_pred = model(x_noisy, t) # predict the noise from noised version of x_0 at timestep t
    return F.l1_loss(noise, noise_pred) # variable loss function 

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

# %%
# DO NOT ITERATE IF YOU HAVE SAVED MODEL

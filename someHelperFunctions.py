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
    

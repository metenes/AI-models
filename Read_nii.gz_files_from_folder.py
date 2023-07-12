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

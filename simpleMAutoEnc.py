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
img = nb.load(filename = "<FILEPATH>")
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
        print(f"filename : {filename}")
        if filename[0] != "." and filename.endswith('.nii.gz'): # niffy 
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

file_dir_path = "<FILEPATH>"

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
"""
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
tensor_all_images_dataset = CustomDataset(tensor_littel_all_images)
plt.imshow( tensor_all_images_dataset[0].permute(1,2,0) , cmap='Greys_r' )

BATCH_SIZE = 8
IMG_SIZE = 320

# dataloader and shuffel
tensor_image_dataloader = DataLoader(tensor_all_images_dataset, batch_size= BATCH_SIZE, shuffle=True, drop_last=True)
#%% 
# Utility function 
import random
import torch
import numpy as np

# Reproducability of the code 
def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

#%% 
# Model
import torch
import timm
import numpy as np
from einops import repeat, rearrange
from einops.layers.torch import Rearrange

# timm libary is for ViT
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=320,
                 patch_size=8,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = torch.nn.Conv2d(1, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        print(f" E1 {img.shape}")
        patches = self.patchify(img)
        print(f" E2 {patches.shape}")
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        print(f" E3 {patches.shape}")
        patches = patches + self.pos_embedding
        patches, forward_indexes, backward_indexes = self.shuffle(patches)
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=320,
                 patch_size=8,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        print(f" D1 {features.shape}")
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        print(f" D2 {features.shape},  D2 {backward_indexes.shape}")
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding
        print(f" D3 {features.shape}")
        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature
        print(f" D4 {features.shape}")
        patches = self.head(features)
        print(f" D4 {patches.shape}")
        mask = torch.zeros_like(patches)
        print(f" D5 {mask.shape}")
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        print(f" D6 {img.shape}")
        mask = self.patch2img(mask)

        return img, mask

class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=320,
                 patch_size=8,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=1,
                 decoder_layer=4,
                 decoder_head=1,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features,  backward_indexes)
        return predicted_img, mask
#%% 
# Model Test 
""" 

shuffle = PatchShuffle(0.75)
a = torch.rand(16, 2, 10)
b, forward_indexes, backward_indexes = shuffle(a)

img = next(iter(tensor_image_dataloader))
plt.imshow(img[0].permute(1,2,0))
encoder = MAE_Encoder()
decoder = MAE_Decoder()
features, backward_indexes = encoder(img.float())
print(forward_indexes.shape)
predicted_img, mask = decoder(features, backward_indexes)
print(predicted_img.shape)
plt.imshow(predicted_img[0].detach().numpy().transpose(1,2,0))
loss = torch.mean((predicted_img - img) ** 2 * mask / 0.75)
print(loss) # non trained loss : 227887.5671 nice 

"""

#%% 
# Train 
""" """
import os
import argparse
import math
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

# pre-load var. 
load = False
checkpoint = None

# Save / load checkpoint 
def saveCheckPoint(state, filename = "my_checkpoint_MAE.pth.tar"): 
    print("-- Checkpoint reached --")
    torch.save(state,filename)

def loadCheckPoint(model, optimizer, state):
    print(" Checkpoint loading ")
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])


def main():
    # arguments  
    seed = 42
    batch_size = 4
    max_device_batch_size = 8
    base_learning_rate =1e-3
    weight_decay =0.05
    total_epoch =100
    warmup_epoch =5
   
    setup_seed(seed)

    batch_size = batch_size
    load_batch_size = min(max_device_batch_size, batch_size)
    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    # train dataset
    train_dataset = tensor_all_images_dataset
    train_dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    # device agnostic code 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Init 
    model = MAE_ViT()
    optim = torch.optim.AdamW(model.parameters(), lr=base_learning_rate * batch_size / 256, betas=(0.9, 0.999), weight_decay=weight_decay)
    # Load model, optim 
    if load:
        loadCheckPoint(model, optim, checkpoint)

    lr_func = lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    step_count = 0
    optim.zero_grad()
    # Train code 
    for e in range(total_epoch):
        model.train()
        losses = []
        acces = []
        for idx, img in tqdm(enumerate(train_dataloader)):
            step_count += 1
            img = img.to(device)
            pred_img, mask= model(img.float())
            loss = torch.mean((pred_img - img) ** 2 * mask / 0.75)
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()

        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = sum(acces) / len(acces)
        # Save checkpoint 
        checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optim.state_dict() }
        saveCheckPoint(model, optim, checkpoint)
        print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')

#%%
main()

# %%

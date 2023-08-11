#%% 
# Utility function 

import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

# timm libary is for ViT
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

def random_index(size): 
    # np.arange(size) : Return evenly spaced values within a given interval.
    forward_index = np.arange(size)
    # shuffle the index
    np.random.shuffle(forward_index)
    # np.argsort() : Returns the indices that would sort an array.
    backward_index = np.argsort(forward_index) 
    return forward_index, backward_index

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio):
        super().__init__()
        # Init
        self.ratio = ratio

    def forward(self, patches): 
        N, B, C = patches.shape()
        # N : number of batches
        # B : batch 
        # C : channel 

        # This is the N which will stay same = will not be masked
        remain_N = int(N * (1 - self.ratio))
        # For each batch, we calculate the both the forward_index and sorting index
        indexs = [ random_index(N) for _ in range(B) ]

        # Seperate the indexes 
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexs], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexs], axis=-1), dtype=torch.long).to(patches.device)
        
        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_N]

        return patches, forward_indexes, backward_indexes

class MAE_Encoder(torch.nn.Module): 
    def __init__(self, 
                 image_size= 320,
                 patch_size= 2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75 ): 
        super().__init__()
        self.cls_token 

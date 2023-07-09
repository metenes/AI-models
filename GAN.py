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
from torch.utils.tensorboard import SumaryWriter

class Discriminator(nn.Module): 
    def __init__(self,in_ch): 
        super().__init__()
        self.discr = nn.Sequential(nn.Linear(in_ch, 128), 
                                   nn.LeakyReLU(0.1), 
                                   nn.Linear(128,1),
                                   nn.Sigmoid())
        
    def forward(self, x): 
        return self.discr(x)
    
class Generator(nn.Module): 
    def __init__(self, noise_dim, img_dim): 
        super().__init__()
        self.gener = nn.Sequential(nn.Linear(noise_dim, 256), 
                                   nn.LeakyReLU(0.1), 
                                   nn.Linear(256, img_dim), 
                                   nn.Tanh())

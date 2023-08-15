## Standard libraries
import os
import numpy as np
import random
from PIL import Image
from types import SimpleNamespace
## Imports for plotting
import matplotlib.pyplot as plt
## PyTorch
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms


class InceptionBlock(nn.Module): 
    def __init__(self, c_in, c_red, c_out, act_fn): 
        """
        c_in - input dimension 
        c_red - residual dimensions dict
        c_out - out dimensions dict
        act_fn - activation funct. 
        """
        super().__init__()
        # sepeate the residual dimensions into self variables
        self.conv_1x1 = nn.Sequential(
            # select 1x1 residual dim. from dimension dict
            # in 1x1, c_out = c_red dimension 
            nn.Conv2d(c_in, c_out["1x1"], kernel_size=1 ), 
            nn.BatchNorm2d(c_out["1x1"]), 
            act_fn() 
        )

        # 3x3 convolution branch = 3x3 residual block
        self.conv_3x3 = nn.Sequential(
            # select 3x3 residual dim. from dimension dict
            nn.Conv2d(c_in, c_red["3x3"], kernel_size=1),
            nn.BatchNorm2d(c_red["3x3"]),
            act_fn(),
            nn.Conv2d(c_red["3x3"], c_out["3x3"], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out["3x3"]),
            act_fn()
        )

        # 5x5 convolution branch = 5x5 residual block
        self.conv_5x5 = nn.Sequential(
            # select 5x5 residual dim. from dimension dict
            nn.Conv2d(c_in, c_red["5x5"], kernel_size=1),
            nn.BatchNorm2d(c_red["5x5"]),
            act_fn(),
            nn.Conv2d(c_red["5x5"], c_out["5x5"], kernel_size=5, padding=2),
            nn.BatchNorm2d(c_out["5x5"]),
            act_fn()
        )

        # Max-pool branch
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(c_in, c_out["max"], kernel_size=1),
            nn.BatchNorm2d(c_out["max"]),
            act_fn()
        )  

    def forward(self, x):
        # all conv blocks are seperate be careful 
        x_1x1 = self.conv_1x1(x)
        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        x_max = self.max_pool(x)
        x_out = torch.cat([x_1x1, x_3x3, x_5x5, x_max], dim=1)
        return x_out

# Inception block of Google 
class GoogleNet(nn.Module): 
    def __init__(self, num_classes = 10, act_fn = nn.ReLU()):
        super().__init__()  
        # Init 
        self.act_fn = act_fn
        self.num_classes = num_classes

        self.create_network()
        self.init_params()

    def create_network(self): 
        # img_ch : number of color channel
        # img_size : image dimension 
        img_ch = 3, 
        img_size = 64
        
        # input conv which turn img to c_in dimension  
        self.input_net = nn.Sequential(
            nn.Conv2d(img_ch, img_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(img_size),
            self.act_fn()
        )
        # Stacking inception blocks
        # suitable for img_ch = 3, img_size = 64
        self.inception_blocks = nn.Sequential(
            InceptionBlock(img_size, c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 16, "3x3": 32, "5x5": 8, "max": 8}, act_fn=self.act_fn),
            InceptionBlock(img_size, c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12}, act_fn=self.act_fn),
            nn.MaxPool2d(img_ch, stride=2, padding=1),  # 32x32 => 16x16
            InceptionBlock(96, c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12}, act_fn=self.act_fn),
            InceptionBlock(96, c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16}, act_fn=self.act_fn),
            InceptionBlock(96, c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16}, act_fn=self.act_fn),
            InceptionBlock(96, c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 32, "3x3": 48, "5x5": 24, "max": 24}, act_fn=self.act_fn),
            nn.MaxPool2d(img_ch, stride=2, padding=1),  # 16x16 => 8x8
            InceptionBlock(128, c_red={"3x3": 48, "5x5": 16}, c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16}, act_fn=self.act_fn),
            InceptionBlock(128, c_red={"3x3": 48, "5x5": 16}, c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16}, act_fn=self.act_fn)
        )
        # output conv which turn c_red dimension to class number 
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, self.num_classes)
        )

    def _init_params(self):
        # Based on our discussion in Tutorial 4, we should initialize the convolutions according to the activation function
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.inception_blocks(x)
        x = self.output_net(x)
        return x
    
    

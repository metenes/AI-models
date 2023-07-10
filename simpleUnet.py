#%%

import torch
from torch import nn 

class DoubleConv(nn.Module): 
    def __init__(self, in_ch, out_ch): 
        super(DoubleConv, self).__init__() # do not know why ? 
        self.Conv = nn.Sequential(
            # first layer
            nn.Conv2d( in_ch, out_ch, 3, 1 ,1,bias=False ), # batchnorm will be used so bias = False
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace= True),
            # second layer
            nn.Conv2d( out_ch, out_ch, 3, 1 ,1,bias=False ), # batchnorm will be used so bias = False
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace= True)
        )

    def forward(self, x): 
        return self.Conv(x)
    
class UNET(nn.Module): 
    def __init__(self, 
                 in_ch = 3, 
                 out_ch = 1, 
                 features = [64,128,256,512] # Paper values
                 ): 
    
        super(UNET, self).__init__() 
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size= 2, stride=2)

        # Down conv. layers
        self.downs.append(DoubleConv(in_ch, features[0] ))
        for idx in range(1, len(features)-1): 
            self.downs.append(DoubleConv(in_ch= features[idx], out_ch=features[idx+1]))
        
        # Ups conv. layers
        for idx in range(1, len(features)-1): 
            self.ups.append(DoubleConv(in_ch= features[len(features) - idx], out_ch=features[len(features) - idx-1]))
        self.ups.append(DoubleConv(in_ch= 0, out_ch= out_ch))

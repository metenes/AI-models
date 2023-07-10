#%%

import torch
from torch import nn 

class DoubleConv(nn.Module): 
    def __init__(self, in_ch, out_ch ): 
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

        self.middel = DoubleConv(features[-1], features[-1]*2)

        # Ups conv. layers
        for idx in range(1, len(features)-1): 
            nn.ConvTranspose2d(features[len(features) - idx]*2, features[len(features) - idx], kernel_size= 2, stride=2)
            self.ups.append(DoubleConv(in_ch= features[len(features) - idx], out_ch=features[len(features) - idx-1]))

        self.ups.append(DoubleConv(in_ch= features[0]*2, out_ch= features[0]))

        self.out_conv = nn.Conv2d(features[0], out_ch)

    def forward(self, x): 
        skip_connections = [] # for linear skips 
        # downs
        for down in self.downs : 
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        # bottelneck
        x = self.middel(x)
        # ups 
        for up in reversed(self.ups): 
            skip = skip_connections.pop()
            x = torch.concat(x,skip, dim=1)
            x = up(x)
            
        return self.out_conv(x)
            


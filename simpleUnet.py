import torch
from torch import nn 

class DoubleConv(nn.Module): 
    def __init__(self, in_ch, out_ch, up): 
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
    def __init__(self): 
        

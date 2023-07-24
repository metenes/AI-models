#%% 
# Init models Black Box - Dont know what are these 
# Probabily pre-defined weights and biases by torch libary 
import torch
import torch.nn as nn
from torch.nn import init

def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

#%%
# Components of Unet
import torch
import torch.nn as nn
import torch.nn.functional as F

# Standart conv block 
class ConvBlock(nn.Module) :
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1 ): 
        super(ConvBlock, self).__init__()
        # Init
        self.n = n # number of ( conv + norm + relu )package in single conv block 
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding

        if is_batchnorm: 
            for i in range(1, n+1): 
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p), 
                                     nn.BatchNorm2d(out_size), 
                                     nn.ReLU(inplace=True) )
                # setattr = self.conv_name = conv 
                # setattr(object, name, value)
                setattr(self, f'conv{i}' , conv)
                in_size = out_size
        else: 
            for i in range(1, n+1): 
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p), 
                                     nn.ReLU(inplace=True) )
                # setattr = self.conv_name = conv 
                # setattr(object, name, value)
                setattr(self, f'conv{i}' , conv)
                in_size = out_size

        """ 
        # May be in de-commented 
        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')
        """
    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1): 
            # get conv layer
            conv = getattr(self, f'conv{i}' )
            x = conv(x)
        return x
        
class Up(nn.Module): 
    def __init__(self, in_ch, out_ch, is_deconv, n_concat = 2):
        super(Up, self).__init__()
        self.conv = ConvBlock(out_ch*2, out_ch, False)
        if is_deconv:
            # transpose = down conv
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        else: 
            # up sampling 
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        
        """ 
        # May be in de-commented 
        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')
        """

    def forward(self, inputs0, *input):
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)

class mergeConv(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(mergeConv, self).__init__()
        # self.conv = unetConv2(out_size*2, out_size, False)
        if is_deconv:
            self.conv = ConvBlock(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = ConvBlock(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        """ 
        # May be in de-commented 
        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')
        """

    def forward(self, inputs0, *input):
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)
#%% 
# Unet+2 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models

class Unet_plus2(nn.Module): 
    def __init__(self, 
                 in_ch = 1, 
                 out_ch = 1, 
                 feature_scale = 4, 
                 is_deconv = True,
                 is_batchnorm = True,
                 is_ds = True ): 
        super(Unet_plus2, self).__init__()
        # Init
        self.is_deconv = is_deconv
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds
        self.feature_scale = feature_scale

        # Filter channels
        filters = [64, 128, 256, 512, 1024]

        # Down Sample = Standart Unet
        self.conv00 = ConvBlock(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.conv10 = ConvBlock(filters[0], filters[1], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv20 = ConvBlock(filters[1], filters[2], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv30 = ConvBlock(filters[2], filters[3], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv40 = ConvBlock(filters[3], filters[4], self.is_batchnorm)

        # Up sampling 
        self.up_concat01 = mergeConv(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = mergeConv(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = mergeConv(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = mergeConv(filters[4], filters[3], self.is_deconv)
        self.up_concat02 = mergeConv(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = mergeConv(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = mergeConv(filters[3], filters[2], self.is_deconv, 3)
        self.up_concat03 = mergeConv(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = mergeConv(filters[2], filters[1], self.is_deconv, 4)
        self.up_concat04 = mergeConv(filters[1], filters[0], self.is_deconv, 5)

        # Final Convulution 
        self.final_1 = nn.Conv2d(filters[0], out_ch, 1)
        self.final_2 = nn.Conv2d(filters[0], out_ch, 1)
        self.final_3 = nn.Conv2d(filters[0], out_ch, 1)
        self.final_4 = nn.Conv2d(filters[0], out_ch, 1)

    def forward(self, inputs):
    
        """
            In naming section: 
                X_row_colm
                X_01 = row 0 and col 1 
        """

        # column : 0        
        X_00 = self.conv00(inputs)
        maxpool0 = self.maxpool0(X_00)
        X_10 = self.conv10(maxpool0)
        maxpool1 = self.maxpool1(X_10)
        X_20 = self.conv20(maxpool1)
        maxpool2 = self.maxpool2(X_20)
        X_30 = self.conv30(maxpool2)
        maxpool3 = self.maxpool3(X_30)
        X_40 = self.conv40(maxpool3)
        # column : 1
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40, X_30)
        # column : 2
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        X_22 = self.up_concat22(X_31, X_20, X_21)
        # column : 3
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)
        # column : 4
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)

        # Final layer
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)

        final = (final_1 + final_2 + final_3 + final_4) / 4

        if self.is_ds:
            return F.sigmoid(final)
        else:
            return F.sigmoid(final_4)
        
# Models output 
model = Unet_plus2()

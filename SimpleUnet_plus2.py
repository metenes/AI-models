#%% 
# Init models Black Box - Dont know what are these 
# Probabily pre-defined weights and biases by trorch libary 
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

class unetUp_origin(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp_origin, self).__init__()
        # self.conv = unetConv2(out_size*2, out_size, False)
        if is_deconv:
            self.conv = ConvBlock(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = ConvBlock(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        """ 
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

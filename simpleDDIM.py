#%% 
# Helper functions 

# Normaliztion 
def normalization(channels: int):
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)

# Swish  
def swish(x): 
    return x*torch.sigmoid(x)

class GaussianDistribution:
    def __init__(self, parameters): 
        self.mean, log_var = torch.chunk

# Resnet Block 
class ResnetBlock(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super().__init__()
        # first layer
        self.norm1 = normalization(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channelas, 3, stride=1, padding=1)
        # second layer
        self.norm2 = normalization(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)

        if in_channels != out_channels: 
            # this is for adjusting the skip connetions size, no kernel or padding 
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        else : 
            self.shortcut = nn.Identity()
    
    def forward(self, x): 
        # [batch_size, channels, height, width]
        h = x 
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        return self.shortcut(x) + h



class Encoder(nn.Module): 
    def __init__(self, *, channels, channel_multipliers, n_resnet_blocks, in_channels, z_channels):
      """
        in_channels is the number of channels in the image
        channels is the number of channels in the first convolution layer 
        channel_multipliers are the multiplicative factors for the number of channels in the subsequent blocks 
        z_channels is the number of channels in the embedding space
        n_resnet_blocks is the number of resnet layers at each resolution 
      """
    
      super().__init__()
      self.conv_in = nn.Conv2d(in_channels, channels, 3, stride=1, padding=1) # first conv layer img -> first channel of latent space
      # make channels in a list for next channels of latent space 
      # we are multipling with one because of the first conv
      channels_list = [m * channels for m in ([1]+channel_multipliers)]
      # Encoder 
      self.down = nn.ModuleList()
      # Number of blocks of different resolutions. The resolution is halfed at the end each top level block 
      n_resolution = len(channel_multipliers)
      for i in range(n_resolution): 
        # resnet block inside the down list 
        resnet_blocks = nn.ModuleList()
        for _ in range(n_resnet_blocks):
            resnet_blocks.append(Res(channels_list[i], channels_list[i+1]))


# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, decoder, encoder ):

#%% 
"""
    Latent diffusion models use an auto-encoder to map between image space and
    latent space. The diffusion model works on the latent space, which makes it
    a lot easier to train. 
"""
# We will first code the latent diffusion model but for that, we will need a Diffusion Unet structer
from labml_nn.diffusion.stable_diffusion.model.autoencoder import Autoencoder # Autocoder which will be used for, 
# Encoding the normal image to latent space
# Decode the output of Unet in latent space to the normal image
from labml_nn.diffusion.stable_diffusion.model.unet import UNetModel # will be Normal Unet which will work on latent space

class DiffusionWrapper(nn.Module): 
    def __init__(self,
                 diffusion_model: UNetModel): 
        # Init
        super().__init__()
        self.diffusion_model = diffusion_model

    def forward(self, x, time_step, context ): 
        # give latent variables to Unet Model 
        return self.diffusion_model(x, time_step, context)
    
class

#%% 
# Helper functions 
from typing import List
import torch
import torch.nn.functional as F
from torch import nn


# Normaliztion 
def normalization(channels: int):
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)

# Swish  
def swish(x): 
    return x*torch.sigmoid(x)

class GaussianDistribution:
    def __init__(self, parameters): 
        # torch.chunk(input, chunks, dim=0) = Attempts to split a tensor into the specified number of chunks. Each chunk is a view of the input tensor.
        self.mean, log_var = torch.chunk(parameters, 2, dim=1)
        # torch.clamp(input, min=None, max=None, ) = Clamps all elements in input into the range [ min, max ].
        self.log_var = torch.clamp(log_var, -30.0, 20.0)
        # reparameter trick 
        self.std = torch.exp(0.5 * self.log_var)

    def sample(self): 
        return self.mean + self.std * torch.randn_like(self.std)
    
# Resnet Block 
class ResnetBlock(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super().__init__()
        # first layer
        self.norm1 = normalization(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
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
    
# Upsamle Block 
class UpSample(nn.Module): 
    def __init__(self, channels ): 
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3 ,padding=1)
    
    def forward(self, x): 
        # Upsampling = interpolate 
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)

# Downsample Block 
class DownSample(nn.Module): 
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d( channels, channels, 3, stride=2, padding=0 )

    def forward(self, x): 
        # add pad firts becasuse stride = 2 so, size will be decreased 
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)
    
# Attention Block - Black Box
class AttnBlock(nn.Module):
    """
    ## Attention block
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # Group normalization
        self.norm = normalization(channels)
        # Query, key and value mappings
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        # Final $1 \times 1$ convolution layer
        self.proj_out = nn.Conv2d(channels, channels, 1)
        # Attention scaling factor
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor):
        """
        :param x: is the tensor of shape `[batch_size, channels, height, width]`
        """
        # Normalize `x`
        x_norm = self.norm(x)
        # Get query, key and vector embeddings
        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x_norm)

        # Reshape to query, key and vector embeedings from
        # `[batch_size, channels, height, width]` to
        # `[batch_size, channels, height * width]`
        b, c, h, w = q.shape
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)

        # Compute $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)$
        attn = torch.einsum('bci,bcj->bij', q, k) * self.scale
        attn = F.softmax(attn, dim=2)

        # Compute $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$
        out = torch.einsum('bij,bcj->bci', attn, v)

        # Reshape back to `[batch_size, channels, height, width]`
        out = out.view(b, c, h, w)
        # Final $1 \times 1$ convolution layer
        out = self.proj_out(out)

        # Add residual connection
        return x + out


#%% 
# AutoEncoder 
# Encoder will carry image into latent space 
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
        self.down = nn.ModuleList() # List of top-level blocks 
        # Number of blocks of different resolutions. The resolution is halfed at the end each top level block 
        n_resolution = len(channel_multipliers)

        for i in range(n_resolution): 
            # resnet block inside the down list 
            resnet_blocks = nn.ModuleList() # Each top level block consists of multiple ResNet Blocks and down-sampling 
            for _ in range(n_resnet_blocks):
                resnet_blocks.append(ResnetBlock(channels, channels_list[i + 1]))
                channels = channels_list[i + 1]

            down_top = nn.Module() # current top level block 
            down_top.block = resnet_blocks 

            # at the end of current top down block, expect the last one, we add downsample block
            if i != n_resolution - 1: 
                down_top.downsample = DownSample(channels)
            else: 
                down_top.downsample = nn.Identity()

            self.down.append(down_top) # add current top block to the Encoder list
    
        # Final ResNet blocks with attention
        self.mid = nn.Module() 
        self.mid.block_1 = ResnetBlock(channels, channels)
        self.mid.attn_1 = AttnBlock(channels)
        self.mid.block_2 = ResnetBlock(channels, channels)

        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv2d(channels, 2*z_channels, 3, stride=1, padding=1)

        def forward( self, img ): 
            # image will be in shape of [batch, channel, height, withd]
            x = self.conv_in(img)

            # Top-level block
            for down_block in self.down : 
                # Resnet Blocks
                for block in down_block :
                    # Resnet Block
                    x = block(x)
                # Block at the end of the Top level block 
                x = down_block.downsample(x)
            
            x = self.mid.block_1(x)
            x = self.mid.attn_1(x)
            x = self.mid.block_2(x)
                
            x = self.norm_out(x)
            x = swish(x)
            x = self.conv_out(x)
            return x

# Decoder will brin g the latent space elements into image again 
class Decoder(nn.Module): 
    def __init__(self, *, channels, channel_multipliers, n_resnet_blocks, out_channels, z_channels):
        """
        out_channels is the number of channels in the image
        channels is the number of channels in the first convolution layer
        channel_multipliers are the multiplicative factors for the number of channels in the subsequent blocks 
        z_channels is the number of channels in the embedding space
        n_resnet_blocks is the number of resnet layers at each resolution 
        """
        super().__init__()
        # Number of Residual blocks of different resolutions. The resolution is doubled at the end each top level block 
        n_resolutions = len(channel_multipliers)

        channels_list = [m * channels for m in channel_multipliers]
        # We set the first channel to be the largest as we are decoding 
        channels = channels_list[-1]

        # First convolution, we start with z_channels = latent space as we are decoding it. 
        self.conv_in = nn.Conv2d(z_channels, channels, 3, stride=1, padding=1)

        # As we are reversing the Encoder, the Attention blocks are first
        self.mid = nn.Module() 
        self.mid.block_1 = ResnetBlock(channels, channels)
        self.mid.attn_1 = AttnBlock(channels)
        self.mid.block_2 = ResnetBlock(channels, channels)

        # List of top-level blocks 
        self.up = nn.ModuleList()

        # top block 
        for i in reversed(range(n_resolutions)): 
            # resnet block inside a top block
            resnet_blocks = nn.ModuleList()
            for _ in range(n_resnet_blocks + 1): 
                resnet_blocks.append( ResnetBlock( channels, channels_list[i] ) )
                channels = channels_list[i]
            # current top up block 
            up_top = nn.Module()
            up_top.block = resnet_blocks
            
            # at the end of current top up block, expect the first one, we add upsampling block
            if i != 0 : 
                up_top.upsample = UpSample(channels)
            else :
                up_top.upsample = nn.Identity()

            # add the current block to top-level block 
            self.up.insert(0, up_top)

        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv2d(channels, out_channels, 3, stride=1, padding=1)

    def forward(self, z): 
        #  first conv. 
        h = self.conv_in(z)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # Top-level block
        for up_block in reversed(self.up): 
            # Resnet blocks
            for block in up_block: 
                # Resnet block
                h = block(h)
            # end of the top level up block
            h = up_block.upsample(h)

        # out layers 
        h = self.norm_out(h)
        h = swish(h)
        img = self.conv_out(h)

        return img 

# Autoencoder
class Autoencoder(nn.Module):
    def __init__( self, decoder, encoder, emb_channels, z_channels ):
        # emb_channels: the channel number after the quant_conv
        # z channels: the 

        super().__init__()
        # Init
        self.encoder = encoder
        self.decoder = decoder
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * emb_channels, 1)
        self.post_quant_conv = nn.Conv2d( emb_channels, z_channels, 1 )

    def encode(self, img ): 
        # img = [batch_size, img_channels, img_height, img_width]
        # This module will encode the img and return a gaussian representation of it. 
        # we want gaussian because we want the distrubution we know / can represent 

        z = self.encoder(img) # encoder is Encoder class object, we got latent representation
        moments = self.quant_conv(z) # why ? 
        return GaussianDistribution(moments)

    def decoder(self, z): 
        # z = [batch_size, emb_channels, z_height, z_height]
        # This module will return the decoder output 
        z = self.post_quant_conv(z)
        return self.decoder(z)
#%% 
# CLIP text embedder - Black Box
import torch
from torch import nn
from transformers import CLIPTokenizer, CLIPTextModel # This comes from the outer 

class CLIPTextEmbedder(nn.Module): 
    def __init__(self, version = "openai/clip-vit-large-patch14", device = "cuda:0", max_length = 77): 
        super().__init__()
        # Load the tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        # Load the CLIP transformer
        self.transformer = CLIPTextModel.from_pretrained(version).eval()

        self.device = device
        self.max_length = max_length 

    def forward(self, prompts ): 
        # Tokenize the prompts
        batch_encoding = self.tokenizer(prompts, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        # Get token ids
        tokens = batch_encoding["input_ids"].to(self.device)
        # Get CLIP embeddings
        return self.transformer(input_ids=tokens).last_hidden_state

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

class LatentDiffusion(nn.Module): 
    # We define these components 
    model: DiffusionWrapper
    first_stage_model : Autoencoder
    cond_stage_model: CLIPTextEmbedder

    def __init__(self, 
                 unet_model, 
                 autoencoder, 
                 clip_embedder, 
                 latent_scaling_factor, 
                 n_steps, 
                 linear_start, 
                 linear_end ): 
        
        # latent_scaling_factor is the scaling factor for the latent space. 
        # The encodings of the autoencoder are scaled by this before feeding into the U-Net. 
        # linear_start and end are for the beta schedule 
        super().__init__()
        # Init
        self.model = DiffusionWrapper(unet_model)
        self.first_stage_model = autoencoder
        self.latent_scaling_factor = latent_scaling_factor
        self.cond_stage_model = clip_embedder
        self.n_steps = n_steps

        # betas schedule 
        betas = torch.linspace(linear_start **0.5, linear_end **0.5, n_steps, dtype=torch.float64 ) ** 2
        # nn.Parameter = A kind of Tensor that is to be considered a module parameter when added to a nn.Module class 
        self.betas = nn.Parameter(betas.to(torch.float32), requires_grad = False)
        # alphas 
        alphas = 1. - betas
        # alphas_cumprod
        alphas_cumprod = torch.cumprod(alphas, dim = 0)
        # nn.Parameter
        self.alphas_cumprod = nn.Parameter(alphas_cumprod.to(torch.float32), requires_grad = False)
    

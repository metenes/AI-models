#%% 
# DDPM methods
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
import torch.nn.functional as F
import random
import wandb # weight and biases
# start a new experiment

BATCH_SIZE = 8
IMG_SIZE = 320

# dataloader and shuffel
tensor_all_images_dataloader = DataLoader(tensor_all_images_dataset, batch_size= BATCH_SIZE, shuffle=True, drop_last=True)

# Helper methods 
def linear_beta_schedule(timestep, start = 0.0001, end = 0.02): 
    """" Linear schedule for noise to be added not cosine-schdule"""

    # linspace: Creates a one-dimensional tensor of size steps whose values 
    # # are evenly spaced from start to end, inclusive
    return torch.linspace(start , end, timestep)

def quadratic_beta_schedule(timestep, start = 0.0007, end = 0.05 ):
    return torch.linspace(start**0.5, end**0.5, timestep) ** 2

def sigmoid_beta_schedule(timestep, start = 0.0001, end = 0.02):
    betas_in_fn = torch.linspace(-6, 6, timestep)
    return torch.sigmoid(betas_in_fn) * (end - start) + start


def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device = 'cpu'): 
    """" Forward step of DDPM 
         add a Gausian(random) noise to current state of the distrubution
         and return the noised state with respect to given timestep """

    # create random noise
    noise = torch.randn_like(x_0)
    # print(f"noise type {noise.dtype}") # noise is float.64
    # print(noise.shape) # shape is [1, 320, 320]

    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod,t,x_0.shape)
    # print(sqrt_alphas_cumprod_t.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod,t,x_0.shape)
    # print(sqrt_one_minus_alphas_cumprod_t.shape)
    # produce noise by multiplying the initial image tensor with cumulative mult. of alphas 
    # x_0
    # noised_image = sqrt_alphas_cumprod_t*x_0 + sqrt_one_minus_alphas_cumprod_t*noise
    return sqrt_alphas_cumprod_t.to(device=device)*x_0.to(device=device) + sqrt_one_minus_alphas_cumprod_t.to(device=device)*noise.to(device=device) , noise.to(device=device)

def forward_corruption_sample(x_0, t, top_left_x = 100, top_left_y= 100 ,height= 100 ,witdh= 100, device = 'cpu'): 
    # we will add a box with random size and cordinates which will be corrupted similar to forward diffusuion
    noise_tensor_like_x = torch.zeros_like(x_0)
    x_0_copy = torch.clone(x_0)
    t = T - t - 1
    # produce a noised image 
    x_noised, noise = forward_diffusion_sample(x_0, t, device)
    # get the specific part from noised image and integrage it to the image
    for idx, _ in enumerate(x_0): 
        x_0_copy[idx, 0, top_left_y:(top_left_y + height), top_left_x:(top_left_x + witdh)] -= x_noised[idx, 0, top_left_y:(top_left_y + height), top_left_x:(top_left_x + witdh)]
        # need to be similar size with the x_0 
        noise_tensor_like_x[idx, 0, top_left_y:(top_left_y + height), top_left_x:(top_left_x + witdh)] = noise[idx, 0, top_left_y:(top_left_y + height), top_left_x:(top_left_x + witdh)]
    return x_0_copy , noise_tensor_like_x

# Pre-calculated values 
T = 2000 # last t value
betas = quadratic_beta_schedule(timestep=T) 
alphas = 1 - betas   # alpha = 1 - beta
alphas_cumprod = torch.cumprod(alphas, axis= 0) # cumulative product

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

alphas_cumprod_prev = F.pad( alphas_cumprod[:-1],(1,0),value=1.0 ) # seperate the first value from others
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


# %% 
# Simulate forward diffusion
image = next(iter(tensor_all_images_dataloader))
print(image.dtype)

plt.figure(figsize=(15,15))
plt.axis('off')
num_images = 10
stepsize = int(T/num_images)

for idx in range(0, T, stepsize):
    t = torch.Tensor([idx]).type(torch.int64)
    plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
    img, noise = forward_diffusion_sample(image, t)
    # print(f" {img.shape} and {noise.shape}")
    show_tensor_image(img)


# %% 
# Simulate corrupt diffusion
image = next(iter(tensor_all_images_dataloader))
print(image.shape)

plt.figure(figsize=(15,15))
plt.axis('off')
num_images = 10
stepsize = int(T/num_images)

# select where to corrupt. 
top_left_x, top_left_y = 100, 100
height , witdh = 100, 100

for idx in range(0, T, stepsize):
    t = torch.Tensor([idx]).type(torch.int64)
    plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
    img, noise = forward_corruption_sample(image, t, top_left_x, top_left_y ,height ,witdh )
    show_tensor_image(img)

# %%
# U-net Arthitecture \ Backward

import torch
import math 
from torch import nn

# different than classical Unet, this one has time_emb_dim 
class Block(nn.Module): 
    def __init__(self, in_ch, out_ch, time_emb_dim, up = False):
        super().__init__() # nn super()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up : 
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding = 1) 
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4,2,1)
        else: 
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding = 1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4,2,1)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t):
        # print(f"CHECK block CONV1 { x.shape} ")
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # print(f"CHECK AFTER block CONV1 {  h.shape} ")
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        #print(f"CHECK BEFORE block END {  h.shape} ")
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        # print(f"CHECK AFTER block END {  h.shape} ")
        return self.transform(h)
    
class SinusoidalPositionEmbeddings(nn.Module): # about transformers need to look 
    # basically increase the rate of learning with a mathematical formula for placment of the words
    def __init__(self, dim): 
        super().__init__()
        self.dim = dim 

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SimpleUnet(nn.Module):
    """" simple, constant layerd unet artitecture """
    def __init__(self): 
        super().__init__()
        image_channels = 1 # RGB
        down_channels = (32, 64, 128, 256, 512, 1024, 2048)
        up_channels = (2048, 1024, 512, 256, 128, 64, 32)
        out_dim = 1 # RGB
        # in_dim and out_dim must be same 
        # " need to give what you take"
        time_emb_dim = 32
        
        # Time embedding : about transformers 
        self.time_mlp = nn.Sequential(
                        SinusoidalPositionEmbeddings(time_emb_dim),
                        nn.Linear(time_emb_dim, time_emb_dim),
                        nn.ReLU()   )
        
        # Initial layer
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding= 1)

        # Downsample 
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], time_emb_dim, False) for i in range(0, len(down_channels)-1)])

        # Upsample 
        self.ups = nn.ModuleList([Block(up_channels[i],up_channels[i+1], time_emb_dim, True) for i in range(0, len(up_channels)-1)])
        
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep): 
        # Embedd time --> transformator 
        t = self.time_mlp(timestep)
        # Initial conv
        # print(f"CHECK  { x.shape} ")
        x = self.conv0(x)
        # print(f"CHECK AFTER CONV0  { x.shape} ")
        # Unet with time embeddings
        residual_inputs = [] # store the result of every forward process / noised image
        for down in self.downs: # iterate block of unet from ModuleList
            x = down(x,t) # apply the time embadding and input and get the new image --> markov chain 
            residual_inputs.append(x)
        for up in self.ups : 
            residual_x = residual_inputs.pop()
             # Add residual x as additional channels --> copy and crop lines between two side of the U net layers
             # this is why there is mult with 2 in up convlations it double the size with cat
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        return self.output(x)

# Initialize the model 
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleUnet().to(device)

#%%
import random
# loss function 
def get_loss(model, x_0, t): 
    # device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # The sum of two Gaussian distributions is itself a Gaussian distribution. 
    # When you add two independent random variables that follow Gaussian distributions, 
    # their sum will also follow a Gaussian distribution.

    x_noisy, noise = forward_corruption_sample(x_0, t,device=device)
    x_noisy = x_noisy.float()
    # giving the noised version of the x_0 to the Neural Networlk to make predictions about the noise
    # print(f"CHECK  { x_noisy.shape} ")
    noise_pred = model(x_noisy, t) # predict the noise from noised version of x_0 at timestep t
    return F.l1_loss(noise, noise_pred) # variable loss function 

# %%
import random
# Sampling
@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    # print(x.shape)
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    x = x.float()
    # Call model (current image - noise prediction)
    # takes noise and convert it to denoised image 
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        # The t's are offset from the t's in the paper
        return model_mean 
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

@torch.no_grad()
def sample_plot_image():
    # Sample noise
    # img_size = IMG_SIZE
    # select where to corrupt. 
    top_left_x, top_left_y = 100, 100
    height , witdh = 100, 100
    # img = torch.randn((1, 1, img_size, img_size), device=device)
    # get a random image from dataset, and corrupt it to last T value = total gaussian distrubution 
    images = next(iter(tensor_all_images_dataloader))
    images, not_used = forward_corruption_sample(images, torch.Tensor([1999]).type(torch.int64), top_left_x, top_left_y ,height ,witdh )
    num_images = 21
    stepsize = int(T/num_images)
    # initial image 
    plt.figure(figsize=(15,15))
    plt.axis('off')
    plt.subplot(1, num_images, 1)
    show_tensor_image(img.detach().cpu())
    count = 2
    # Backward process 
    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        noise = sample_timestep(images, t)
        print(noise.shape)
        # Edit: This is to maintain the natural range of the distribution
        noise = torch.clamp(noise, -1.0, 1.0)
        # Recover previous step 
        images[:, 0, top_left_y:(top_left_y + height), top_left_x:(top_left_x + witdh)] += noise[:, 0, top_left_y:(top_left_y + height), top_left_x:(top_left_x + witdh)]
        if i % stepsize == 0:
            plt.subplot(1, num_images, count)
            count += 1
            show_tensor_image(images.detach().cpu())
    plt.show() 

# %%
# DO NOT ITERATE IF YOU HAVE SAVED MODEL
# Training / Save
from torch.optim import Adam
import random
def saveCheckPoint(state, filename = "my_checkpoint_corrupt_new_2.pth.tar"): 
    print("-- Checkpoint reached --")
    torch.save(state,filename)

def loadCheckPoint(state):
    print("-- Checkpoint loading --")
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])

# device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
epochs = 100

load_model = True
load_model_filename = "my_checkpoint_corrupt_new_1.pth.tar"

if(load_model == True): 
    loadCheckPoint(torch.load(load_model_filename))

for epoch in range(epochs):
    for step, batch in enumerate(tensor_all_images_dataloader):
      optimizer.zero_grad()
      t = torch.randint(0, T, (1,), device = device).long()
      loss = get_loss(model, batch.to(device), t)
      loss.backward()
      optimizer.step()

      if step % 64 == 0 :
        print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
        # sample_plot_image()
        # wandb.log({"epoch": epoch, "loss": loss}, step=step)
    # save the model
    checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict() }
    saveCheckPoint(checkpoint)

# %%
# DO NOT ITERATE IF YOU HAVE SAVED MODEL
# %%
# Test / Save
""" """
from torch.optim import Adam
def loadCheckPoint(state):
    print(" Checkpoint loading ")
    # TODO
    # create tensor from the dict ojects 
    # tensor_dict = {key: torch.tensor(value) for key, value in state.items()}
    # print(tensor_dict['state_dict'].shape) 
    # print(tensor_dict['optimizer'].shape) 

    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])

# device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
epochs = 100

load_model_filename = "my_checkpoint_corrupt_new_2.pth.tar"

loadCheckPoint(torch.load(load_model_filename, map_location=torch.device('cpu')))

for epoch in range(epochs):
    sample_plot_image()

# %%

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
""" 
# setup train data
train_data = datasets.FashionMNIST(root = "data",
                                   train = True,
                                   download =True,
                                   transform = torchvision.transforms.ToTensor(),
                                   target_transform = None
                                  )


# setup train data
test_data = datasets.FashionMNIST(root = "data",
                                   train = False,
                                   download =True,
                                   transform = torchvision.transforms.ToTensor(),
                                   target_transform = None
                                  )
"""

train_data = torchvision.datasets.FGVCAircraft( root = "data", 
                                                split = "train", 
                                                transform= torchvision.transforms.ToTensor(),
                                                target_transform= None,
                                                download= True )


test_data  = torchvision.datasets.FGVCAircraft( root = "data", 
                                                split = "test", 
                                                transform= torchvision.transforms.ToTensor(),
                                                target_transform= None, 
                                                download= True )


# visualize the data
image, label = train_data[1]
#plt.imshow(image.squeeze())
plt.imshow(image.permute(1,2,0))

print(image.shape)



# %%

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

def linear_beta_schedule(timestep, start = 0.0001, end = 0.02): 
    """" Linear schedule for noise to be added not cosine-schdule"""

    # linspace: Creates a one-dimensional tensor of size steps whose values 
    # # are evenly spaced from start to end, inclusive
    return torch.linspace(start , end, timestep)

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
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod,t,x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod,t,x_0.shape)

    # produce noise by multiplying the initial image tensor with cumulative mult. of alphas 
    # x_
    return sqrt_one_minus_alphas_cumprod_t.to(device=device)*x_0.to(device=device) + sqrt_one_minus_alphas_cumprod_t.to(device=device)*noise.to(device=device) , noise.to(device=device)

# Pre-calculated values 
T = 300 # last t value
betas = linear_beta_schedule(timestep=T) 
alphas = 1 - betas   # alpha = 1 - beta
alphas_cumprod = torch.cumprod(alphas, axis= 0) # cumulative product

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

alphas_cumprod_prev = F.pad( alphas_cumprod[:-1],(1,0),value=1.0 ) # seperate the first value from others
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# %%
# Data visualizer for datasets
def visualize_dataset(dataset, num_images = 20): 
    fig = plt.figure(figsize=(20, 20))
    for idx in range(0,num_images): 
        data = dataset[idx]
        ax = fig.add_subplot(1, num_images, idx+1)
        image,label = data
        print(f"data shape {image.shape}")
        ax.imshow( image.permute(1, 2, 0) )
    plt.show()

visualize_dataset(train_data)

# %%
import torch 
import numpy as np

IMG_SIZE = 64
BATCH_SIZE = 128

def load_transformed_dataset(): 
    # Transformer part
    data_transformers = [transforms.Resize((IMG_SIZE,IMG_SIZE)),    # Adjust the size of images to be same in all
                         transforms.RandomHorizontalFlip(),         # random flip 
                         transforms.ToTensor(),                     # turn image into tensor
                         transforms.Lambda(lambda t : (t*2)-1)]     # Apply a user-defined lambda as a transform. lambd (function) – Lambda/function to be used for transform.
    
    data_transformers = transforms.Compose(data_transformers)
    """" 
    # setup train data
    train = datasets.FashionMNIST(root = "data",
                                       train = True,
                                       download =True,
                                       transform = data_transformers,
                                       target_transform = None
                                      )

    # setup train data
    test = datasets.FashionMNIST(root = "data",
                                      train = False,
                                      download =True,
                                      transform = data_transformers,
                                      target_transform = None
                                    )
    """

    train = torchvision.datasets.FGVCAircraft( root = "data", 
                                                split = "train", 
                                                transform= data_transformers,
                                                target_transform= None,
                                                download= True )


    test = torchvision.datasets.FGVCAircraft( root = "data", 
                                                split = "test", 
                                                transform= data_transformers,
                                                target_transform= None, 
                                                download= True )

    print(test) 
    return torch.utils.data.ConcatDataset([train, test])

def show_tensor_image(image): 
    reverse_transforms = transforms.Compose([
                        transforms.Lambda(lambda t : (t+1)/2), 
                        transforms.Lambda(lambda t : t.permute(1,2,0)), # change the ordering of vectors  CHW to HWC 
                        transforms.Lambda(lambda t : t *255.), 
                        transforms.Lambda(lambda t : t.numpy().astype(np.uint8)),
                        transforms.ToPILImage(),
                                                ])
    # Take first image of batch
    if len(image.shape) == 4: 
        image = image[0,:,:,:]
    plt.imshow(reverse_transforms(image))

data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
visualize_dataset(data)


# %% 
# Simulate forward diffusion
image = next(iter(dataloader))[0]

plt.figure(figsize=(15,15))
plt.axis('off')
num_images = 10
stepsize = int(T/num_images)

for idx in range(0, T, stepsize):
    t = torch.Tensor([idx]).type(torch.int64)
    plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
    img, noise = forward_diffusion_sample(image, t)
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
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
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
        image_channels = 3 # RGB
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3 # RGB
        # in_dim and out_dim must be same 
        # " need to give what you take"
        time_emb_dim = 32
        
        # Time embedding : about transformers 
        self.time_mlp = nn.Sequential(
                        SinusoidalPositionEmbeddings(time_emb_dim),
                        nn.Linear(time_emb_dim, time_emb_dim),
                        nn.ReLU()   
                    )
        
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
        print(x.shape)
        x = self.conv0(x)
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

# loss function 
def get_loss(model, x_0, t): 
    # device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_noisy, noise = forward_diffusion_sample(x_0, t,device=device)
    # giving the noised version of the x_0 to the Neural Networlk to make predictions about the noise
    noise_pred = model(x_noisy, t) # predict the noise from noised version of x_0 at timestep t
    return F.l1_loss(noise, noise_pred) # variable loss function 

# %%
# Sampling
@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    print(x.shape)
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
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
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')

    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu())
    plt.show() 


# %%
# Training 
from torch.optim import Adam

# device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 100

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
      optimizer.zero_grad()

      t = torch.randint(0, T, (BATCH_SIZE,), device = device).long()
      loss = get_loss(model, batch[0].to(device), t)
      loss.backward()
      optimizer.step()

      if epoch % 5 == 0:
        print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
        sample_plot_image()

# %%

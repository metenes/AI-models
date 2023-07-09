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
from torch.utils.tensorboard import SumaryWriter
#%% 
# Discriminator
class Discriminator(nn.Module): 
    def __init__(self,in_ch): 
        super().__init__()
        self.discr = nn.Sequential(nn.Linear(in_ch, 128), 
                                   nn.LeakyReLU(0.1), 
                                   nn.Linear(128,1),
                                   nn.Sigmoid())
        
    def forward(self, x): 
        return self.discr(x)
#%% 
# Generator    
class Generator(nn.Module): 
    def __init__(self, noise_dim, img_dim): 
        super().__init__()
        self.gener = nn.Sequential(nn.Linear(noise_dim, 256), 
                                   nn.LeakyReLU(0.1), 
                                   nn.Linear(256, img_dim), 
                                   nn.Tanh())
    def forward(self, x):
       return self.gener(x) 
#%% 
# device agnostic code
if torch.cuda.is_available(): 
    device = "cuda"
else: 
    device = "cpu"

lr = 3e-4 # GANs are very sensetive to learning rate
z_dim = 64 
image_dim = 28 * 28 * 1
batch_size = 32
num_epochs = 50 
# init 
disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
# noise = random gaussian distr = randn
noise = torch.randn((batch_size, z_dim)).to(device)
# transforms.Compose is similar to torch.Sequential, compose different transforms into one 
transforms =transforms.Compose(
    [transforms.ToTensor() , 
     transforms.Normalize((0.5,), (0.5,))]
)

# Dataset
dataset =torchvision.datasets.MNIST(root = ".",
                                    transforms = transforms,
                                    download= True)
# Dataloader 
loader = DataLoader(dataset, batch_size, shuffle=True)
optim_disc = torch.optim.Adam(disc.parameters(), lr = lr)
optim_gen =torch.optim.Adam(gen.parameters(), lr = lr)
loss_fn = nn.BCELoss()

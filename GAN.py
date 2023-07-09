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
from torch.utils.tensorboard import summary 

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
# hyperparameter
epochs = 50 
# init 
disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)

# noise = random gaussian distr = randn
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
# transforms.Compose is similar to torch.Sequential, compose different transforms into one 
transforms = transforms.Compose(
                            [transforms.ToTensor() , 
                            transforms.Normalize((0.5,), (0.5,))]
                            )          

# Dataset
dataset =torchvision.datasets.MNIST(root = ".",
                                    transform= transforms,
                                    download= True)
# Dataloader 
loader = DataLoader(dataset, batch_size, shuffle=True)
# Optimaziers for gen and disc
optim_disc = torch.optim.Adam(disc.parameters(), lr = lr)
optim_gen =torch.optim.Adam(gen.parameters(), lr = lr)
loss_fn = nn.BCELoss()

# Traninig 
step = 0
for epoch in range(epochs):
    for idx,(real,_) in enumerate(loader): 
        # view(-1) : flatten the vector
        real = real.view(-1,784).to(device) # transform real image data for neural network input
        batch_size = real.shape[0]

    # Discriminator train max{ log(loss_fn(real)) + log(loss_fn(1 - Gen(noise)))}
        # noise
        noise = torch.randn(batch_size,z_dim).to(device)
        # Gen(noise)
        fake = gen(noise)

        # BCELoss = −wn​[yn​⋅logxn​+(1−yn​)⋅log(1−xn​)] we need to give ones to the yn to get only log part,
        # log(loss_fn(1 - Gen(noise))
        one_min_disc_fake = 1 - disc(fake).view(-1)
        lossD_fake =loss_fn(one_min_disc_fake, torch.ones_like(one_min_disc_fake)) # the torch.ones_like(disc_read) is due to the impl. of the lossBCE

        # log(loss_fn(real)) 
        disc_real = disc(real).view(-1)
        lossD_real =loss_fn(disc_real, torch.ones_like(disc_real)) # the torch.ones_like(disc_read) is due to the impl. of the lossBCE

        lossD = (lossD_fake + lossD_real)/2
        # zero gradient
        optim_disc.zero_grad()
        # loss backward
        lossD.backward(retain_graph = True) # retain_graph = True is for computational efficeny
        #  gradient descent
        optim_disc.step()

    # Train Generator max{ log(Gen(noise)) }
        output = disc(fake).view(-1)
        lossG = loss_fn(output, torch.ones_like(output))
        # zero gradient
        optim_gen.zero_grad()
        # loss backward
        lossG.backward()
        # gradient descent
        optim_gen.step()
        
        # for writter do not know what is it
        if idx == 0:
            print(
                f"Epoch [{epoch}/{epochs}] Batch {idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

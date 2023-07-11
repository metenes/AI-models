#%%
# Import Pytorch
import torch
from torch import nn
import torch.nn.functional as F
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
#%% 
# VAE 
class VAE(nn.Module): 
    def __init__(self, input_dim, hidden_dim = 200, z_dim = 20): 
        # input -> hidden layer -> mean, variance -> Parameterization trick -> Decoder -> output 
        super().__init__()
    # encoder 
        # first layer image to hidden 
        self.input_to_hidden = nn.Linear(input_dim , hidden_dim)
        # hidden layer to mean 
        self.hidden_to_mean  = nn.Linear(hidden_dim, z_dim)
        # hidden layer to variation 
        self.hidden_to_variance = nn.Linear(hidden_dim, z_dim)

    # decoder
        self.z_to_hidden = nn.Linear(z_dim, hidden_dim)
        self.hidden_to_output = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()

    def encoder(self, x): 
        h = self.relu(self.input_to_hidden(x))
        mean = self.hidden_to_mean(h)
        var = self.hidden_to_variance(h)
        return  mean, var 

    def decoder(self, z): 
        h = self.relu(self.z_to_hidden(z))
        return self.hidden_to_output(h)
    
    def forward(self, x): 
        mean, var = self.encoder(x)
        # reparam trick  
        noise = torch.randn_like(var)
        z = mean+ var*noise
        return self.decoder(z) ,mean , var

#%% 
# Training 
# device agnostic code
if torch.cuda.is_available :
    device = "cuda"
else: 
    device = "cpu"

input_dim = 784 
hidden_dim = 200 
z_dim = 20 
ephocs = 10 
batch = 32
lr = 3e-4
# dataset 
train_data = torchvision.datasets.MNIST(root = ".", 
                                        train = True, 
                                        transform= torchvision.transforms.ToTensor(), # This ensure that the image values are betweem 1 and 0
                                        download= True, 
                                        target_transform= None)

test_data = torchvision.datasets.MNIST( root = ".", 
                                        train = False, 
                                        transform= torchvision.transforms.ToTensor(), # This ensure that the image values are betweem 1 and 0
                                        download= True, 
                                        target_transform= None)

loader =  DataLoader(dataset= train_data, 
                     batch_size = batch, 
                     shuffle= True) 
# model 
model = VAE(input_dim).to(device)
# optim 
optim = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.BCELoss(reduction="sum")

for ephoc in range(ephocs):
    for batch_idx, (img, label) in enumerate(loader) :


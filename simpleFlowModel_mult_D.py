#%% 
# Utils 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class MaskedConv2d(nn.Module): 
    # *args are arguments in tuple 
    # **kwargs are arguments with keywords stored as dict
    def __init__(self, include_base_point, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        # create a object mask, self.mask , pre-init to zero
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask(include_base_point)

    # Dont know
    def create_mask(self, include_base_point):
        h_by_2 = self.kernel_size[0] // 2
        w_by_2 = self.kernel_size[1] // 2
        self.mask[:, :, :h_by_2] = 1
        self.mask[:, :, h_by_2, :w_by_2] = 1
        if include_base_point:
            self.mask[:, :, h_by_2, w_by_2] = 1

    def forward(self, x):
        # self.weight * self.mask
        return F.conv2d(x, self.weight * self.mask, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.LayerNorm):
    # *args are arguments in tuple 
    # **kwargs are arguments with keywords stored as dict
    def __init__(self, *args, **kwargs): 
        super(LayerNorm, self).__init__(*args, **kwargs)
    
    def forward(self, x): 
        x = x.permute(0,2,3,1).contiguous()
        x = super().forward(x)
        x = x.permute(0,3,1,2).contiguous()
        return x

class AutoRegressiveFlow(nn.Module): 
    def __init__(self, 
                 num_channels_input, 
                 num_layers= 5, 
                 num_channels_intermediate= 64, 
                 kernel_size= 7, 
                 n_components= 2, 
                 **kwargs):
        super(AutoRegressiveFlow, self).__init__()
        self.n_components = n_components
        # input layer 
        # input layer dont have include_base_point, it has no start point
        first_layer = MaskedConv2d(False, num_channels_input, num_channels_intermediate, kernel_size=kernel_size, padding=kernel_size//2, **kwargs)
        # model : array holds layers 
        model = [first_layer]
        # latent layers, num_channels_intermediate -> num_channels_intermediate
        block = lambda : MaskedConv2d(True, num_channels_intermediate, num_channels_intermediate, kernel_size=kernel_size, padding=kernel_size//2, **kwargs)

        for _ in range(num_layers):
            # in each element of the model, 
            # 1. Norm
            # 2. non-linear fn 
            # 3. conv in masked = block function 
            model.append( LayerNorm(num_channels_intermediate) )
            model.append( nn.ReLU )
            model.append( block() )

        second_last_layer = MaskedConv2d(True, num_channels_intermediate, num_channels_intermediate, 1, **kwargs)
        model.append(second_last_layer)
        last_layer = MaskedConv2d(True, num_channels_intermediate, n_components * 3 * num_channels_input, 1, **kwargs)
        model.append(last_layer)

        # unpack the array, make it sequential module 
        self.model = nn.Sequential(*model)

    def forward(self, x): 
        # x : [b, c, h, w]
        batch_size, c_in = x.size(0), x.size(1)
        # [h, w]
        h_and_w = x.size()[2:]
        out = self.model(x) # out : [b, c *3 *n_components , h, w] from last layer
        mus, log_sigmas, weight_logits = torch.chunk(out, 3, dim=1) # (b, n_components, c, h, w)
        # logits put in softmax 
        weights = F.softmax(weight_logits, dim=1)
        # sample distribution 
        distribution = Normal(mus, log_sigmas.exp())

        x = x.unsqueeze(1) # x.size() [b, 1, c, h, w]
        z = distribution.cdf(x) # z.size() [b, n_component, c, h, w]
        z = (z * weights).sum(1) # z.size() [b, c, h, w]

        log_dz_by_dx = (distribution.log_prob(x).exp() * weights).sum(1).log()

        return z, log_dz_by_dx

#%% 
# Train 
from torch.distributions.uniform import Uniform
import numpy as np

def saveCheckPoint(state, filename = "my_checkpoint_NFM_muld_1.pth.tar"): 
    print("-- Checkpoint reached --")
    torch.save(state,filename)

def loadCheckPoint(model, optimizer ,state):
    print(" Checkpoint loading ")
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])

# Device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def loss_funct(target_distribution, z, log_dz_by_dx):
    # minimize neg. log likelihood 
    log_likelihood = target_distribution.log_prob(z) + log_dz_by_dx
    return -log_likelihood.mean()

def train(model, train_loader, optimizer, target_distribution):
    # training
    model.train()
    for i, x in enumerate(train_loader):
        x = x.to(device)
        # add a littel noise
        x += torch.distributions.Uniform(0.0, 0.25).sample(x.shape).to(device)
        # model x 
        z, log_dz_by_dx = model(x)
        # loss 
        loss = loss_funct(target_distribution, z, log_dz_by_dx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 0: 
            # model save
            print( 'Loss at iteration {} is {}'.format(i, loss.cpu().item()) )
            checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict() }
            saveCheckPoint(checkpoint)

def main():
    BATCH_SIZE = 8
    IMG_SIZE = 320
    # Dataloader and shuffel
    tensor_littel_all_images_dataloader = DataLoader(tensor_all_images_dataset, batch_size= BATCH_SIZE, shuffle=True, drop_last=True) 
    # Flow based model
    flow = AutoRegressiveFlow(1, num_layers=5, n_components=10).to(device)
    # Optim.  
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
    # Training will be done to achive gaussain Noise
    target_distribution = Uniform(torch.tensor(0).float().to(device),torch.tensor(1).float().to(device))
    # Train 
    train(flow, tensor_littel_all_images_dataloader, optimizer, target_distribution)

#%% 
# start
main()
#%% 

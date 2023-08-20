#%% 
# Utils 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


class MaskedConv2d(nn.Conv2d):
    def __init__(self, include_base_point, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask(include_base_point)

    def forward(self, x):
        # print(self.weight * self.mask)
        return F.conv2d(x, self.weight * self.mask, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def create_mask(self, include_base_point):
        h_by_2 = self.kernel_size[0] // 2
        w_by_2 = self.kernel_size[1] // 2
        self.mask[:, :, :h_by_2] = 1
        self.mask[:, :, h_by_2, :w_by_2] = 1
        if include_base_point:
            self.mask[:, :, h_by_2, w_by_2] = 1

class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(LayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = x.permute(0,2,3,1).contiguous()
        x = super().forward(x)
        x = x.permute(0,3,1,2).contiguous()
        return x

class AutoRegressiveFlow(nn.Module):
    def __init__(self, num_channels_input, num_layers=5, num_channels_intermediate=64, kernel_size=7, n_components=2, **kwargs):
        super(AutoRegressiveFlow, self).__init__()
        first_layer = MaskedConv2d(False, num_channels_input, num_channels_intermediate, kernel_size=kernel_size, padding=kernel_size//2, **kwargs)
        model = [first_layer]
        block = lambda: MaskedConv2d(True, num_channels_intermediate, num_channels_intermediate, kernel_size=kernel_size, padding=kernel_size//2, **kwargs)

        for _ in range(num_layers):
            model.append(LayerNorm(num_channels_intermediate))
            model.append(nn.ReLU())
            model.append(block())

        second_last_layer = MaskedConv2d(True, num_channels_intermediate, num_channels_intermediate, 1, **kwargs)
        last_layer = MaskedConv2d(True, num_channels_intermediate, n_components * 3 * num_channels_input, 1, **kwargs)
        model.append(second_last_layer)
        model.append(last_layer)

        self.model = nn.Sequential(*model)
        self.n_components = n_components

    def forward(self, x):
        batch_size, c_in = x.size(0), x.size(1) # x.size() is (B, c_in, h, w)
        h_and_w = x.size()[2:]
        out = self.model(x) # out.size() is (B, c_in * 3 * n_components, h, w)
        out = out.view(batch_size, 3 * self.n_components, c_in, *h_and_w) # out.size() is (B, 3*n_components, c_in, h, w)
        mus, log_sigmas, weight_logits = torch.chunk(out, 3, dim=1) # (B, n_components, c_in, h, w)
        weights = F.softmax(weight_logits, dim=1)
        # Distribution in Normal is not adequent 
        distribution = torch.distributions.cauchy.Cauchy(mus, log_sigmas.exp())
        # distribution = torch.distributions.

        x = x.unsqueeze(1) # x.size() is (B, 1, c_in, h, w)
        z = distribution.cdf(x) # z.size() is (B, n_components, c_in, h, w)
        z = (z * weights).sum(1) # z.size() is (B, c_in, h, w)

        # problem log prob
        # print(f"distribution.log_prob(x).exp(): {distribution.log_prob(x).exp()}")
        # print(f"weights: {weights} ")
        log_dz_by_dx = (distribution.log_prob(x).exp() * weights).sum(1).log()
        return z, log_dz_by_dx

#%% 
# Train 
from torch.distributions.uniform import Uniform
import numpy as np
import sys

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
    # print(z.shape)
    # print(log_dz_by_dx)
    z = z - 0.01
    log_likelihood = target_distribution.log_prob(z) + log_dz_by_dx
    return -log_likelihood.mean()

def train(model, train_loader, optimizer, target_distribution):
    # training
    model.train()
    for i, x in enumerate(train_loader):
        # x = x[0]
        x = x.to(device)
        # add a littel noise
        x += torch.distributions.Uniform(0.0, 0.25).sample(x.shape).to(device)
        # model x 
        x = x.float()
        z, log_dz_by_dx = model(x)
        # loss 
        loss = loss_funct(target_distribution, z, log_dz_by_dx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0: 
            # model save
            print( 'Loss at iteration {} is {}'.format(i, loss.cpu().item()) )
            checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict() }
            saveCheckPoint(checkpoint)

def main():
    # Flow based model
    num_channel = 1
    flow = AutoRegressiveFlow(num_channel, num_layers=20, n_components=50).to(device)
    # Optim.  
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
    # Training will be done to achive gaussain Noise
    target_distribution = Uniform(torch.tensor(-0.1).double().to(device),torch.tensor(1.1).double().to(device))
    # Train 
    train(flow, tensor_littel_all_images_dataloader, optimizer, target_distribution)

#%% 
# start
main()
#%% 

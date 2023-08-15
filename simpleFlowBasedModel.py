#%% 
# Helper functions 
import os
import time
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torchvision.transforms 
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.optim

class ImageFlow(pl.LightningModule): # flow base libary for the torch 
    def __init__(self, flows, import_samples = 8): 
        """
        flows : list of nn.Module which will be applied to img
        flows are neural networks 
        import samples : number of samples will be taken in training
        """
        super().__init__()
        # Init
        self.flows = nn.ModuleList(flows)
        # Device agnostic code
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.import_samples = import_samples 
        # create prior distribution for final latent space
        # torch.distributions.normal.Normal(loc, scale) : creates a normal (also called Gaussian) distribution parameterized by loc and scale.
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
    
    def forward(self, imgs): 
        return self.get_likelihood(imgs)

    def encode(self, imgs):
        # Given a batch of images, return the latent representation z = applies flow to imgs 
        z, latent_distr = imgs, torch.zeros(imgs.shape[0], device=self.device)
        for flow in self.flows: 
            z, latent_distr = flow(z,latent_distr, reverse =False) # Forward flow x -> z
        return z, latent_distr

    def get_likelihood(self, imgs, type = False): 
        """
        Given a batch of images, return the likelihood of those.
        If type is True, this function returns the log likelihood of the input.
        Otherwise, return scaled negative log likelihood = bits per dimension
        """
        z, latent_distr = self.encode(imgs)
        # log_pz is the log likelihood of latent variable z
        log_pz = self.prior.log_prob(z).sum(dim=[1,2,3])
        # log_px is the log likelihood of data variable x
        log_px = latent_distr + log_pz
        # nll is neg. log likelihood
        nll = -log_px
        # bpd is scaled negative log likelihood = bits per dimension
        scale = np.log2(np.exp(1) / np.prod(imgs.shape[1:]))
        bpd = nll * scale
        return bpd.mean() if not type else log_px 

    @torch.no_grad()
    def sample(self, img_shape, z_init = None):
        """
        Sample a batch of images from the flow.
        """
        if z_init is None: 
            z = self.prior.sample(sample_shape=img_shape, device = self.device) # Gaussian distrubution 
        else : 
            z = z_init.to(self.device) # pre given z 

        # Transform z to x by inverting the flows
        latent_distr = torch.zeros(img_shape[0], device=self.device)
        for flow in reversed(self.flows): 
            z, latent_distr = flow(z, latent_distr, reversed =True)
        return z 
    
    def configure_optim(self): 
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]
    
    def test_step(self, batch, batch_idx):
        samples = [] # log likelihoods
        for _ in range(self.import_samples):
            # img present img's likelihood 
            img = self.get_likelihood(batch[0], type=True)
            samples.append(img)
        # stack all samples into one tensor 
        img = torch.stack(samples, dim=-1)
        # average if all likelihoods 

    def validation_step(self, batch, batch_idx):
        # same as trainin g_step 
        loss = self.get_likelihood(batch[0])
        return loss 
    
    def training_step(self, batch, batch_idx): 
        # Normalizing flows are trained by maximum likelihood 
        loss = self.get_likelihood(batch[0])
        self.log("train_bpd", loss)
        return loss 
# %% 
# Dequantization for discrt. values 

class Dequantization(nn.Module): 
    def __init__(self, alpha = 1e-5, quants = 256): 
        """
        alpha - small constant that is used to scale the original input.
                Prevents dealing with values very close to 0 and 1 when inverting the sigmoid
        quants - Number of possible discrete values 
        """
        super().__init__()
        # Init
        self.alpha = alpha
        self.quants = quants        

    def sigmoid(self, z, latent, reverse = False): 
        # Applies invertiable sigmoid transform 
        


    def forward(self, z, ldj, reverse= False): 
        if not 

#%%
# Sample Class for the DDIM
from typing import Optional, List
import torch
from labml_nn.diffusion.stable_diffusion.latent_diffusion import LatentDiffusion # same as Latent varaible model 

class DiffusionSampler: 
    model : LatentDiffusion
    def __init__(self, model): 
        super().__init__()
        # Init
        self.model = model
        self.n_steps = model.n_steps
    
    def get_eps(self, x, t, c, *, uncond_scale, uncond_cond):
        # x : [batch_size, channels, height, width]
        # t : [batch_size] 
        # c : [batch_size, embed_size] = condition in embed dim.

        if uncond_cond is None or uncond_scale == 1. : 
            # if cond_scale = 1, we cant seperate it by formula
            return self.model(x,t,c)
    
        # Duplicate the x ,c and t
        # Duplications are done to present all vectors in one vector
        x_in = torch.cat([x] * 2) # [x, x]
        t_in = torch.cat([t] * 2)
        c_in = torch.cat([uncond_cond, c]) # [uncond_cond, c]

        # As seperation done in same model
        # chunk seperate the vector into 2
        e_t_uncond, e_t_cond = self.model(x_in, t_in, c_in).chunk(2)

        # ϵθ​(xt​,c) = s*ϵcond​(xt​,c)+(s−1)ϵcond​(xt​,cu​)
        # might have a problem s-1 or 1-s
        e_t = e_t_uncond + uncond_scale * (e_t_cond - e_t_uncond)
        return e_t
    
    # These 3 classes will be overwriten by inherited class 
    def sample(self, 
               shape, 
               cond, 
               repeat_noise, 
               temperature, 
               x_T, 
               uncond_scale, 
               uncond_cond, 
               skip_steps ): 
        
        """
        shape : [batch_size, channels, height, width] = image shape
        temperature : will be constant in formula 
        skip_steps : the number of time steps to skip.
        """
        raise NotImplementedError()
    
    def paint(self, x, cond, t_start, *,
              orig = None,
              mask = None, orig_noise = None,
              uncond_scale = 1.,
              uncond_cond = None ):
        
        raise NotImplementedError()
    
    def q_sample(self, x0 , index, noise = None):
        raise NotImplementedError()

#%% 
# DDIM class extends DiffusionSampler
from typing import Optional, List
import numpy as np
import torch
from labml import monit
from labml_nn.diffusion.stable_diffusion.latent_diffusion import LatentDiffusion
from labml_nn.diffusion.stable_diffusion.sampler import DiffusionSampler

class DDIMSampler(DiffusionSampler): 
    # model is the model to predict noise ϵcond​(xt​,c) by convert the xt to Latent space, use Unet to predict, convert from Latent space 
    model: LatentDiffusion
    def __init__(self, model, n_steps, ddim_discretize = "uniform", ddim_eta = 0.):
        # ddim_discretize : specifies how to extract τ from [1,2,…,T]. It can be either uniform or quad.
        # ddim_eta : η used to calculate στ_i​​. η=0 makes the sampling process deterministic.
        # Init
        super().__init__(model)
        self.n_steps = model.n_steps
        
        if ddim_discretize == "uniform" :



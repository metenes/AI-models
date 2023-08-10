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

        # Formula : ϵθ​(xt​,c) = s*ϵcond​(xt​,c)+(s−1)ϵcond​(xt​,cu​)
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

        # Calculate t
        if ddim_discretize == "uniform" :
            c = self.n_steps // n_steps # calculate which timestep, model will be sampled 
            # np.asarray : Convert the input to an array.
            self.time_steps = np.asarray(list(range(0, self.n_steps, c))) + 1
        elif ddim_discretize == "quad" : 
            self.time_steps = ((np.linspace(0, np.sqrt(self.n_steps * .8), n_steps)) ** 2).astype(int) + 1

        # Pre-calculated variabels
        with torch.no_grad(): 
            # alpha_cumprod from the model 
            alpha_cumprod = self.model.alpha_bar
            # at_i​​
            self.ddim_alpha_cumprod = alpha_cumprod[self.time_steps].clone().to(torch.float32)
            self.ddim_alpha_cumprod_sqrt = torch.sqrt(self.ddim_alpha_cumprod)
            # at_i−1​​ # previous alpha_cumprod
            # as the x_0 will be never depend on itself, we automaitacly ignore the duplicate element in the start
            self.ddim_alpha_cumprod_prev = torch.cat([alpha_cumprod[0:1], alpha_cumprod[self.time_steps[:-1]]])
            self.ddim_one_minus_alpha_cumprod_sqrt = (1 - self.ddim_alpha_cumprod) ** .5
            #  σt_i​ will be calculated by, 
            self.ddim_sigma = ( ddim_eta * ( (1 - self.ddim_alpha_cumprod_prev) / (1 - self.ddim_alpha_cumprod) * (1 - self.ddim_alpha_cumprod / self.ddim_alpha_cumprod_prev) )** .5 )

    @torch.no_grad()
    def sample(self, 
               shape, 
               cond, 
               repeat_noise, 
               temperature, 
               x_T, 
               uncond_scale, 
               uncond_cond, 
               skip_steps ): 
        # Calculate x_0 with looping timesteps with p_sample method starting with x_T
        # shape: [batch_size, channels, height, width]
        # repeat_noise specified whether the noise should be same for all samples in the batch
        device = self.model.device 
        batch_size = shape[0]
        # if x_T is not given, it will be gaussian noise at last step 
        x = x_T if x_T is not None else torch.randn(shape, device=device)

        # np.flip() shape of the array is preserved, but the elements are reverse ordered.
        # Starts from the skip_steps to 1
        time_steps = np.flip(self.time_steps)[skip_steps:]

        for i, step in monit.enum('Sample', time_steps): # monit is just print the process
            index = len(time_steps) - i - 1
            # t values extended for all the batch elements 
            time_step_t = x.new_full((batch_size,), step, dtype=torch.long)
            x, pred_x0, e_t = self.p_sample(x,cond, time_step_t)
            
        return x 
    
    @torch.no_grad()
    def p_sample(self, x, c, t, step, index, *, repeat_noise, temperature, uncond_scale, uncond_cond): 
        # Calculates x_t-1 with x_t by Backward formula = get_x_prev_and_pred_x0() = Backward Process
        # Get eps 
        e_t = self.get_eps(x, t, c, uncond_scale, uncond_cond)
        x_prev, pred_x0 = self.get_x_prev_and_pred_x0(e_t, index, x, temperature, repeat_noise)
        
        return x_prev, pred_x0, e_t
    

    def get_x_prev_and_pred_x0(self, e_t, index, x, temperature, repeat_noise): 
        # Backward formula
        # index is similar to t 
        alpha_cumprod_index= self.ddim_alpha_cumprod[index]
        alpha_cumprod_prev_index = self.ddim_alpha_cumprod_prev[index]
        sigma_index = self.ddim_sigma[index]
        one_minus_alpha_cumprod_sqrt_index = self.ddim_one_minus_alpha_cumprod_sqrt[index]
        # x0 
        pred_x0 = (x - one_minus_alpha_cumprod_sqrt_index * e_t) / (alpha_cumprod_index ** 0.5)
        # Direction pointing to xt
        dir_xt = (1. - alpha_cumprod_prev_index - sigma_index ** 2).sqrt() * e_t

        # No noise will be added when x = x0
        if sigma_index == 0. : 
            noise = 0.
        # if repeat_noise, then noise will be same every time, else regenerate noise
        elif repeat_noise : 
            noise = torch.randn((1,*x.shape[1:]), device=x.device)
        else : 
            noise = torch.randn(x.shape, device=x.device)
        # Backward formula 
        x_prev = (alpha_cumprod_prev_index ** 0.5) * pred_x0 + dir_xt + sigma_index * noise

        return x_prev, pred_x0
    

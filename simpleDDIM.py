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
import os 
# Image settings
from nilearn import plotting
import pylab as plt
import numpy as np
import nibabel as nb

# Access data test directly from second folder
img = nb.load(filename = "/usr/bmicnas01/data-biwi-01/bmicdatasets/Sharing/Mete/data/datasets/T1w/181232_MPR1.nii.gz")
print(img.shape)
# shape of one the images is (256, 320, 320)
print(f"shows the type of the data on disk {img.get_data_dtype()}")
# get the numpy array with fdata()
data_nummpy = img.get_fdata()
# display the nii.gz 
plt.imshow(data_nummpy[:, :, data_nummpy.shape[2] // 2].T, cmap='Greys_r')
print(data_nummpy.shape)

# %%
# Display all the nii.gz images in dataset
def prep_data_from_dir(file_dir_path, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    i = 0 
    list_tensor_imgs = []
    # list_img_paths = []
    for filename in os.listdir(file_dir_path):
        if i == num_samples : 
            break
        if filename.endswith('.nii.gz'): # niffy 
            nb.load( os.path.join(file_dir_path, filename) ) 
            print(os.path.join(file_dir_path, filename)) # file names
            data_nummpy = img.get_fdata()
            list_tensor_imgs.append(torch.from_numpy(data_nummpy)) # save the numpy to torch tensor
            # list_img_paths.append(os.path.join(file_dir_path, filename)) # save the file_path
            i += 1 
    return list_tensor_imgs
    
# Display all the nii.gz images in dataset
def display_data_from_dir(file_dir_path, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15,15)) # figure size for display
    i = 0 
    for filename in os.listdir(file_dir_path):
        if i == num_samples : 
            break
        if filename.endswith('.nii.gz'): # niffy 
            nb.load( os.path.join(file_dir_path, filename) ) 
            print(os.path.join(file_dir_path, filename)) # file names
            data_nummpy = img.get_fdata()
            plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
            plt.imshow(data_nummpy[:, :, data_nummpy.shape[2] // 2].T, cmap='Greys_r')
            i += 1 

file_dir_path = "/usr/bmicnas01/data-biwi-01/bmicdatasets/Sharing/Mete/data/datasets/T1w"

list_tensor_volumes = prep_data_from_dir(file_dir_path, 100)
display_data_from_dir(file_dir_path, 20)

print(list_tensor_volumes[0].dtype) # dtype = float64 , type = torch.float64 tensor
print(len(list_tensor_volumes))

# split array 
train_data_initial = list_tensor_volumes[0:int(len(list_tensor_volumes)*(2/3))]
print(len(train_data_initial))
test_data_initial = list_tensor_volumes[int(len(list_tensor_volumes)*(2/3)):]
print(len(test_data_initial))

print(f"size of a volume {len(list_tensor_volumes[0])}" )
plt.imshow(list_tensor_volumes[0][128], cmap='Greys_r')
#%% 
"""
# WandB
# start a new wandb run to track this script
import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="DDPM initial-II",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "architecture": "DDPM",
    "dataset": "MRI dataset",
    "epochs": 100,
    }
) 
"""

# %%
# Custom Datasets 
import os
import pathlib
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Custom Dataset 
class CustomDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
    def __getitem__(self, idx):
        image = self.images[idx]      
        return image
    def __len__(self):
        return len(self.images)
    
# Data visualizer for datasets
def visualize_volume_dataset(dataset, num_images = 20): 
    fig = plt.figure(figsize=(20, 20))
    for idx in range(0,num_images): 
        data = dataset[idx]
        ax = fig.add_subplot(1, num_images, idx+1)
        # print(f"data shape {data.shape}") # Debug 
        ax.imshow(data[:, :, 0].T, cmap='Greys_r')
    plt.show()

# custom dataset for the niffty file MRI 
volume_custom_dataset = CustomDataset(list_tensor_volumes)
# visualize all volumes in half view
visualize_volume_dataset(volume_custom_dataset)

# Display the tensor images
def show_tensor_image(image): 
    if len(image.shape) == 4: 
        image = image[0,:,:,:]
    plt.imshow(image.permute(1,2,0), cmap='Greys_r')


# For all volumes 
 
# concat all volumes into single list
list_tensor_all_volumes = torch.stack(list_tensor_volumes) 
tensor_all_volumes = torch.DoubleTensor(list_tensor_all_volumes)
# reshape : get all images inside all voluems into single list [ Number of imgs, H, W ]
tensor_all_images = torch.reshape(tensor_all_volumes, [-1, 320, 320])
# add the color channel 
tensor_all_images = tensor_all_images.unsqueeze(1)
print(tensor_all_images.shape) # torch.Size([25600, 1, 320, 320])
# dataset
tensor_all_images_dataset = CustomDataset(tensor_all_images)
plt.imshow( tensor_all_images_dataset[0].permute(1,2,0) , cmap='Greys_r' )

"""
# concat all volumes into single list
list_tensor_littel_all_volumes = torch.stack( [ list_tensor_volumes[0], list_tensor_volumes[1], list_tensor_volumes[2], list_tensor_volumes[3], list_tensor_volumes[4], list_tensor_volumes[5] ]) 
tensor_littel_all_volumes = torch.DoubleTensor(list_tensor_littel_all_volumes)
# reshape : get all images inside all voluems into single list [ Number of imgs, H, W ]
tensor_littel_all_images = torch.reshape(tensor_littel_all_volumes, [-1, 320, 320])
# add the color channel 
tensor_littel_all_images = tensor_littel_all_images.unsqueeze(1)

print(tensor_littel_all_images.shape) # torch.Size([512, 1, 320, 320])
show_tensor_image(tensor_littel_all_images[0])
# dataset
tensor_littel_all_images_dataset = CustomDataset(tensor_littel_all_images)
plt.imshow( tensor_littel_all_images_dataset[0].permute(1,2,0) , cmap='Greys_r' )
"""
#%% 
# Helper functions 
from PIL import Image
from torch.autograd import Variable
import enum
import torch.nn.functional as F
from torchvision.utils import save_image
import torch
import math

def standardize(img):
    mean = torch.mean(img)
    std = torch.std(img)
    img = (img - mean) / std
    return img 

# tensor version 
def standardize_tensor(img):
    mean = img.mean()
    std = img.std()
    img = (img - mean) / std
    return img

def betas_for_alpha_cumprod_fn(timestep, alpha_cumprod_function, max_beta=0.999):
    """
        Create a beta schedule that discretizes the given alpha_t_bar function,
        which defines the cumulative product of (1-beta) over time from t = [0,1].

        alpha_cumprod_function is the lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.

        alpha_cumprod_function(t2)/alpha_cumprod_function(t1) is to obtain a_t at time t
        a_t * alpha_cumprod_function(t1) = alpha_cumprod_function(t2)

    """
    betas = []
    for i in range(timestep): 
        t1 = i / timestep # t
        t2 = (i + 1) / timestep # t + 1
        betas.append(min(1 - alpha_cumprod_function(t2)/alpha_cumprod_function(t1), max_beta ))
    return np.array(betas)


def get_beta_schedule(name, timestep, s): 
    if name == "linear" : 
        # linear beta schedule
        scale = 1000 / timestep
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace( beta_start, beta_end, timestep, dtype=np.float64)
    elif name == "cosine" : 
        return betas_for_alpha_cumprod_fn(timestep,  lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)
    else :
        return -1

def extract_into_tensor(arr, t, shape):
    """
    Get t th element from the arr, reshape/tensor 
    arr : 1D array                              
    t : timestep    
    """
    res = torch.from_numpy(arr).to(device=t.device)[t].float()
    while len(res.shape) < len(shape): 
        res = res[..., None]
    return res.expand(shape)



# These methods used to adjust the type of model - Black Box
class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """
    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon

    
class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()

class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion: 
    """
    rescale_timesteps: if True, pass floating point timesteps into the
                        model so that they are always scaled like in the
                        original paper (0 to 1000).

    """
    def __init__(self, *, 
                 betas, 
                 model_mean_type, 
                 model_var_type, 
                 loss_type, 
                 rescale_timestep = False): 
        # Init
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timestep = rescale_timestep
        self.betas = np.array(betas, dtype=np.float64) # to increase accuracy
        self.timestep_num = int(self.betas.shape[0])

        # pre-define alphas in np
        self.alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        # calculations for diffusion q(x_t | x_{t-1})  = Forward process in np
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0) = Backward process in np
        self.posterior_variance = ( betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod) )
        self.posterior_log_variance_clipped = np.log( np.append(self.posterior_variance[1], self.posterior_variance[1:]) )
        self.posterior_mean_coef1 = ( betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod) )
        self.posterior_mean_coef2 = ( (1.0 - self.alphas_cumprod_prev) * np.sqrt(self.alphas) / (1.0 - self.alphas_cumprod) )

    def q_mean_variance(self, x_0, t): 
        """ 
        Get the distribution elements q(x_t | x_0) = Forward step
            N( sqrt_alphas_cumprod[t] * x_0 , one_minus_alphas_cumprod )
            method will return the mean , variance and log variance for the reparam. method
        """

        mean = ( extract_into_tensor( self.sqrt_alphas_cumprod, t, x_0.shape ) ) * x_0
        variance = extract_into_tensor( 1.0 - self.alphas_cumprod, t, x_0.shape ) 
        log_variance = extract_into_tensor( self.log_one_minus_alphas_cumprod, t, x_0.shape )

        return mean, variance, log_variance

    def q_sample(self, x_0, t, noise= None): 
        """
        Get the noised image at t : q(x_t | x_0) = Forward step
            Using reparam. method
            sqrt_alphas_cumprod[t] * x_0 + one_minus_alphas_cumprod[t] * eps_noise
        """
        if noise is None: 
            noise = torch.randn_like(x_0)
        assert noise.shape == x_0.shape # check for shape validity 
        # reparam method 
        return extract_into_tensor(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise

    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance for q( x_{t-1} | x_t, x_0 ) = Backward diffusion 
        """
        assert x_0.shape == x_t.shape # both dependent variable must same size 

        post_mean = extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_0 + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        post_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        post_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        
        # Calcualtions must be done 
        assert ( post_mean.shape[0] == post_variance.shape[0] == post_log_variance_clipped.shape[0] == x_0.shape[0] )

    def scale_timesteps(self, t): 
        if self.rescale_timestep: 
            # means we t to be between 1000 and 0 as same as in the original paper
            return t.float() * (1000.0 / self.timestep_num)
        # else return same t
        return t 
    
    # Predicts x_0 from x_t and x_model which is models output for the noise prediction 
    # Will be usefull when we will change x_0s in main equation with x_t and predicted noise based formulas
    def predict_x_0_from_x_t_and_x_modelOut( self, x_t , t, x_modelOut ):
        assert x_t.shape == x_modelOut.shape
        # return (x_modelOut - coef2*x_t) / coef1
        return ( extract_into_tensor( 1.0 / self.posterior_mean_coef1, t, x_t.shape ) * x_modelOut 
                 - extract_into_tensor( self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape ) * x_t )

    def predict_x_0_from_eps(self, x_t, t, eps): 
        x_t = x_t[:, :4, ...] # why ? 
        assert x_t.shape == eps.shape 
        return ( extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps) 
        

    def p_mean_variance( self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None ):
        """
        Apply the model to get p( x_{t-1} | x_t ) which is estimation of q( x_{t-1} | x_t, x_0 ) and prediction of initial x_t , x0 
        x : the [N x C x ...] tensor at time t.
        t : a 1-D Tensor of timesteps.
        model_kwargs: if not None, a dict of extra keyword arguments to pass to the model. This can be used for conditioning.
            conditioning means stable diffusion type such as a keyword from NLPs 

        """
        if model_kwargs is None: 
            model_kwargs = {}
        # [ batch_num, channel_num, height, withd ] 
        batch_num, channel_num = x.shape[:2]

        # t must in shape of batch number as we are taking batch_num images
        assert t.shape ==(batch_num,)

        model_output = model(x, self.scale_timesteps(t), **model_kwargs )

        # This code block is to adjust the Model type, black box 
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (batch_num, channel_num * 2, *x.shape[2:])
            model_output, model_var_values = torch.split(model_output, channel_num, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = torch.exp(model_log_variance)
            else:
                min_log = extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = torch.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = extract_into_tensor(model_log_variance, t, x.shape)

        # get the x_0 with given function 
        def process_x_0(x): 
            # to predict x_0, we use our denoised function which is same as DDPM
            if denoised_fn is not None: 
                x = denoised_fn(x)
            # This is for clamps all elements in input x into the range [ min, max ]
            # in this case [-1, 1]
            if clip_denoised: 
                return x.clamp(-1,1)
            return x
        
        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_x_0 = process_x_0( self.predict_x_0_from_x_t_and_x_modelOut(x_t=x, t=t, xprev=model_output) )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_x_0(model_output)
            else:
                pred_xstart = process_x_0(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }











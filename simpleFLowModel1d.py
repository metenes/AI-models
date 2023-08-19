import torch
import torch.nn as nn
from torch.distributions.normal import Normal # Gaussian Distrubutions

class Flow1d(nn.Module): 
    def __init__(self, n_components): 
        super(Flow1d, self).__init__()
        # Mu, log_sigma and weight_logis for CDFs = weight * CDF(mu, log_sigma)
        # These values will adjusted by parameters 
        self.mu_values = nn.Parameter(torch.randn(n_components), requires_grad=True) 
        self.log_sigma = nn.Parameter(torch.zeros(n_components), requires_grad=True) # log sigma for effiecent comp.
        self.weight_logits = nn.Parameter(torch.ones(n_components), requires_grad=True)

    def forward(self, x): 
        x = x.view(-1,1)
        # As all weight_logits sum up to 1, we take softmax
        weights = self.weight_logits.softmax(dim=0).view(1,-1)
        # we will use Normal distrubutons to obtain the CDFs
        distribution = Normal(self.mus, self.log_sigmas.exp())
        # Z = weights * CDFs
        z = (distribution.cdf(x) * weights).sum(dim=1)
        # dz/dx = weights * dCDF/dx = weights * p(x) 
        # p(x) is PDF = dCDF/dx
        log_dz_by_dx = (distribution.log_prob(x).exp() * weights).sum(dim=1).log()
        return z, log_dz_by_dx

class LogitTransform(nn.Module): 
    def __init__(self, alpha): 
        super(LogitTransform, self).__init__()
        self.alpha = alpha 

    def forward(self, x):
        # affine definition
        x_affine = self.alpha/2 + (1-self.alpha)*x  
        # Logit transform
        z = torch.log(x_affine) - torch.log(1-x_affine)
        # deriative formula 
        log_dz_by_dx = torch.log(torch.FloatTensor([1-self.alpha])) - torch.log(x_affine) - torch.log(1-x_affine)
        return z, log_dz_by_dx

class FlowComposable1d(nn.Module): 
    def __init__(self, flow_models_list): 
        super(FlowComposable1d, self).__init__()
        self.flow_models_list = nn.ModuleList(flow_models_list)
    
    def forward(self, x): 
        # pre-init 
        # sum is hold due to maximization algorithm 
        z, sum_log_dz_by_dx = x, 0
        for flow in self.flow_models_list:
            z, log_dz_by_dx = flow(z)
            sum_log_dz_by_dx += log_dz_by_dx
        return z, sum_log_dz_by_dx

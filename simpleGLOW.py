#%% 
# Flow types 
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import types
import numpy as np
import scipy as sp
import scipy.linalg

def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output
    masks are the vectors which will be multiplied with weights vector
    their purpose is to re-define the connections between neurons
    initially, all neurons are connected to all neurons 
    after mult. with 0s some connections will be deleted

    this is needed to represent the dependencies between the inputs
    p(x1), p(x2 | x1), p(x3 | x1, x2) etc... order may different 

    the 0s are determined by the random ordering 
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    # return 0 or 1
    # element-wise compreseion between vectors 
    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()

class MaskedLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 mask,
                 cond_in_features=None,
                 bias=True):
        super(MaskedLinear, self).__init__()
        # nn.Linear(input, weight) y = Ax + b 
        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None: 
            self.cond_linear = nn.Linear(cond_in_features, out_features, bias=False)
        # register_buffer, takes parameter and name, count is as parameter of a register
        # self.mask 
        self.register_buffer("mask", mask)

    def forward(self, inputs, cond_inputs=None):
        # nn.Linear(input, weight) y = Ax + b 
        output = F.linear(inputs, self.linear.weight * self.mask, self.linear.bias)
        if cond_inputs is not None:
            output += self.cond_linear(cond_inputs)
        return output

nn.MaskedLinear = MaskedLinear

class MADESplit(nn.Module): 
    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 s_act='tanh', # activation for s(x)
                 t_act='relu', # activation for t(x)
                 pre_exp_tanh=False):
        super(MADESplit, self).__init__()
        # Init
        self.pre_exp_tanh = pre_exp_tanh
        # non-linear functions 
        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        # masks
        input_mask = get_mask(num_inputs, num_hidden, num_inputs,mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(num_hidden, num_inputs, num_inputs,mask_type='output')

        # activations 
        act_func = activations[s_act]
        self.s_joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask, num_cond_inputs)
        self.s_trunk = nn.Sequential(act_func(),
                                     nn.MaskedLinear(num_hidden, num_hidden,
                                                   hidden_mask), act_func(),
                                   nn.MaskedLinear(num_hidden, num_inputs,
                                                   output_mask))

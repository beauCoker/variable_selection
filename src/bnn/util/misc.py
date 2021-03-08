# standard library imports
import math

# package imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma

def softplus(x):
    return torch.log(1+torch.exp(x))
    
def softplus_inverse(y):
    return torch.log(torch.exp(y)-1)

def get_activation(activation_type='relu'):
    if activation_type=='relu':
        return F.relu
    elif activation_type=='tanh':
        return torch.tanh
    else:
        print('activation not recognized')


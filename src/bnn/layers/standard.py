# standard library imports
from math import pi, log, sqrt

# package imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import kl_divergence

# local imports
import bnn.util as util

class LinearLayer(nn.Module):
    """
    Linear model layer

    Assumes 1d outputs for now
    """
    def __init__(self, dim_in, prior_mu=0., prior_sig2=1., sig2_y=.1, **kwargs):
        super(LinearLayer, self).__init__()

        self.dim_in = dim_in
        self.prior_mu = prior_mu
        self.prior_sig2 = prior_sig2
        self.sig2_y = sig2_y

        self.register_buffer('mu', torch.empty(1, dim_in))
        self.register_buffer('sig2', torch.empty(dim_in, dim_in))

        self.init_parameters()
        self.sample_weights(store=True)

    def init_parameters(self):
        self.mu.normal_(0,1)
        self.sig2 = self.prior_sig2*torch.eye(self.dim_in)

    def sample_weights(self, store=False, prior=False):
        if prior:
            mu = self.prior_mu*torch.ones(1, self.dim_in)
            sig2 = self.prior_sig2*torch.eye(self.dim_in)
        else:
            mu = self.mu
            sig2 = self.sig2

        try:
            m = MultivariateNormal(mu, sig2)
            w = m.sample()
        except:
            print('Using np.random.multivariate_normal')
            w = torch.from_numpy(np.random.multivariate_normal(mu.reshape(-1).numpy(), sig2.numpy())).float()
        
        if store: self.w = w
        return w
    
    def fixed_point_updates(self, x, y):
        # conjugate updates

        prior_sig2inv_mat = 1/self.prior_sig2*torch.eye(self.dim_in)
        prior_mu_vec = torch.ones(self.dim_in,1)*self.prior_mu

        try:
            self.sig2 = torch.inverse(prior_sig2inv_mat + x.transpose(0,1)@x/self.sig2_y)
            self.mu = (self.sig2 @ (prior_sig2inv_mat@prior_mu_vec + x.transpose(0,1)@y/self.sig2_y)).transpose(0,1)
        except:
            print('Error: cannot update LinearLayer, skipping update')
            pass
        
    def forward(self, x, weights_type='sample_post'):
        '''
        weights_type = 'mean': 
        weights_type = 'sample': 
        weights_type = 'stored': 
        '''
        if weights_type == 'mean_prior':
            w = self.prior_mu

        elif weights_type == 'mean_post':
            w = self.mu

        elif weights_type == 'sample_prior':
            w = self.sample_weights(store=False, prior=True)

        elif weights_type == 'sample_post':
            w = self.sample_weights(store=False, prior=False)

        elif weights_type == 'stored':
            w = self.w

        return F.linear(x, w)


class RffLayer(nn.Module):
    """
    Random features layer
    """
    def __init__(self, dim_in, dim_out, lengthscale=1.0, **kwargs):
        super(RffLayer, self).__init__()

        ### architecture
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.lengthscale = lengthscale

        self.register_buffer('w', torch.empty(dim_out, dim_in))
        self.register_buffer('b', torch.empty(dim_out))

        self.sample_features()

        self.act = lambda z: sqrt(2/self.dim_out)*torch.cos(z)

    def sample_features(self):
        # sample random weights for RFF features
        self.w.normal_(0, 1 / self.lengthscale)
        self.b.uniform_(0, 2*pi)

    def forward(self, x):
        return self.act(F.linear(x, self.w, self.b))


def get_layer(name):
    if name == 'LinearLayer':
        return LinearLayer

    elif name == 'RffLayer':
        return RffLayer

    elif name == 'RffVarSelectHsLayer':
        return RffVarSelectHsLayer

    elif name == 'RffVarSelectBetaLayer':
        return RffVarSelectBetaLayer

    elif name == 'RffVarSelectLogitNormalLayer':
        return RffVarSelectLogitNormalLayer

    elif name == 'RffVarSelectSpikeSlabLogNormal':
        return RffVarSelectSpikeSlabLogNormal








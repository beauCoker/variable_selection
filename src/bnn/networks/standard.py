# standard library imports
import os
from math import sqrt, pi

# package imports
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions.multivariate_normal import MultivariateNormal

# local imports
import bnn.layers.standard as layers
import bnn.util as util

class NN(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, n_layers=1, act_name='relu'):
        super(NN, self).__init__()
        '''
        Simple BNN for regression
        n_layers: number of hidden layers (must be >= 1)
        '''
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.n_layers = n_layers

        self.layers = nn.ModuleList(
            [nn.Linear(dim_in, dim_hidden)] + \
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(n_layers-1)] + \
            [nn.Linear(dim_hidden, dim_out)] \
        )

        self.act = get_act(act_name)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        for l, layer in enumerate(self.layers):
            x = layer(x)
            if l < self.n_layers:
                x = self.act(x)
        return x

    def loss(self, x, y, return_metrics=True):
        '''
        '''
        f_pred = self.forward(x)
        loss = self.criterion(f_pred, y)

        if return_metrics:
            return loss, {'loss': loss.item()}
        else:
            return loss

    def init_parameters(self, seed):
        pass


class BNN(nn.Module):
    """
    Simple BNN for regression (1d output)

    n_layers: number of hidden layers (must be >= 1)
    noise_scale: standard deviation of observational noise
    w_scale_prior: standard deviation of prior over weights
    b_scale_prior: standard deviation of prior over biases
    scale_prior: if True, prior is scaled to ensure convergence as dim_hidden --> to infty
    """
    def __init__(self, dim_in, dim_hidden=50, noise_scale=1., n_layers=1, act_name='relu', layer_name='BBB', w_scale_prior=1., b_scale_prior=1., scale_prior=False, temperature_kl=1.0, **kwargs_layer):
        super(BNN, self).__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.n_layers = n_layers
        self.scale_prior = scale_prior
        self.noise_scale = noise_scale
        self.temperature_kl = temperature_kl

        self.act_name = act_name
        self.act = get_act(act_name)
        Layer = get_layer(layer_name)
        
        # scale prior for convergence as dim_hidden --> infty
        if scale_prior:
            w_scale_prior2 = w_scale_prior/np.sqrt(dim_hidden)
        else:
            w_scale_prior2 = w_scale_prior

        self.layers = nn.ModuleList(
            [Layer(dim_in, dim_hidden, w_scale_prior=w_scale_prior, b_scale_prior=b_scale_prior, **kwargs_layer)] + \
            [Layer(dim_hidden, dim_hidden, w_scale_prior=w_scale_prior2, b_scale_prior=w_scale_prior2, **kwargs_layer) for _ in range(n_layers-1)] + \
            [Layer(dim_hidden, 1, w_scale_prior=w_scale_prior2, b_scale_prior=w_scale_prior2, **kwargs_layer)] \
        )

    def forward(self, x, n_samp=1, prior=False):
        for l, layer in enumerate(self.layers):
            x = layer(x, n_samp=n_samp, prior=prior)
            if l < self.n_layers:
                x = self.act(x)
        return x

    def kl_divergence(self):
        return sum([layer.kl_divergence() for layer in self.layers])

    def log_prob(self, y_observed, f_pred):
        '''
        y_observed: (n_obs, dim_out)
        f_pred: (n_obs, n_pred, dim_out)

        averages over n_pred (e.g. could represent different samples), sums over n_obs
        '''
        lik = Normal(f_pred, self.noise_scale)
        return lik.log_prob(y_observed.unsqueeze(1)).mean(1).sum(0)

    def loss(self, x, y, return_metrics=True, n_samp=1):
        '''
        uses negative elbo as loss

        n_samp: number of samples from the variational distriution for computing the likelihood term
        '''

        f_pred = self.forward(x, n_samp=n_samp) # (n_obs, n_samp, dim_out)
        log_prob = self.log_prob(y, f_pred)
        kl = self.kl_divergence()
        loss_tempered = -log_prob + self.temperature_kl*kl # negative elbo

        if return_metrics:
            loss = -log_prob + kl # negative elbo
            metrics = {'loss': loss.item(), 'log_prob': log_prob.item(), 'kl': kl.item()}
            return loss_tempered, metrics
        else:
            return loss_tempered

    
    def init_parameters(self, seed=None, gain=None):
        '''
        initialize variational parameters
        '''
        if seed is not None:
            torch.manual_seed(seed)
        if gain is None:
            gain = nn.init.calculate_gain(self.act_name)
        for layer in self.layers:
            layer.init_parameters(gain)

    def set_temperature(self, temperature):
        self.temperature_kl = temperature


class Rff(nn.Module):
    """
    Single layer RFF model

    Variance of output layer scaled by width (see RFF activation function)
    """
    def __init__(self, dim_in, dim_out, dim_hidden, noise_sig2, prior_w2_sig2, lengthscale=1.0, **kwargs):
        super(Rff, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.noise_sig2 = noise_sig2
        self.prior_w2_sig2 = prior_w2_sig2
        self.lengthscale = lengthscale

        # layers
        self.layer_in = layers.RffLayer(self.dim_in, self.dim_hidden, lengthscale=lengthscale)
        self.layer_out = layers.LinearLayer(self.dim_hidden, prior_sig2=self.prior_w2_sig2, sig2_y=self.noise_sig2)

    def forward(self, x, x_linear=None, weights_type='sample_post'):
        h = self.layer_in(x)
        return self.layer_out(h, weights_type=weights_type)

    def fixed_point_updates(self, x, y):   
        h = self.layer_in(x) # hidden units
        self.layer_out.fixed_point_updates(h, y) # conjugate update of output weights 

    def init_parameters(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        self.layer_in.init_parameters()
        self.layer_out.init_parameters()

    def reinit_parameters(self, x, y, n_reinit=1):
        seeds = torch.zeros(n_reinit).long().random_(0, 1000)
        losses = torch.zeros(n_reinit)
        for i in range(n_reinit):
            self.init_parameters(seeds[i])
            losses[i] = self.loss(x, y)

        self.init_parameters(seeds[torch.argmin(losses).item()])
        


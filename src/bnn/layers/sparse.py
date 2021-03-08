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
from torch.distributions.half_cauchy import HalfCauchy
from torch.distributions.beta import Beta
from torch.distributions.log_normal import LogNormal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.uniform import Uniform
from torch.distributions.relaxed_bernoulli import LogitRelaxedBernoulli
from torch.distributions import kl_divergence

# local imports
from bnn.layers.standard import RffLayer
import bnn.util as util
from bnn.util.distributions import LogitNormal, ProductDistribution, InvGamma, PointMass

class _RffVarSelectLayer(RffLayer):
    '''
    '''
    def __init__(self, dim_in, dim_out, **kwargs):
        super().__init__(dim_in, dim_out)
        '''
        '''

        ### scale parameters
        self.s = torch.empty(self.dim_in)

    def init_parameters(self):
        raise NotImplementedError

    def get_prior(self):
        raise NotImplementedError

    def get_variational(self):
        raise NotImplementedError

    def sample_prior(self, shape=torch.Size([]), store=False):
        p = self.get_prior()
        s = p.sample(shape)
        if store: self.s = s
        return s

    def sample_variational(self, shape=torch.Size([]), store=False):
        # sample from the variational family or prior
        q = self.get_variational()
        s = q.rsample(shape)
        if store: self.s = s
        return s

    def log_prob_variational(self):
        # evaluates log prob of variational distribution at stored variational param values
        q = self.get_variational()
        return torch.sum(q.log_prob(self.s))

    def kl_divergence(self):
        p = self.get_prior()
        q = self.get_variational()
        return torch.sum(torch.distributions.kl_divergence(q, p))

    def fixed_point_updates(self):
        pass

    def forward(self, x, weights_type='sample_post', n_samp=None):
        '''
        if taking samples:
            if n_samp == None, output is (n_obs, dim_hidden)
            otherwise output is (n_obs, n_samp, dim_hidden)
        '''
        s_shape = torch.Size([]) if n_samp is None else (n_samp,)

        if weights_type == 'mean_prior':
            p = self.get_prior()
            s = p.mean()

        elif weights_type == 'mean_post':
            q = self.get_variational()
            s = q.mean()

        elif weights_type == 'sample_prior':
            s = self.sample_prior(s_shape)

        elif weights_type == 'sample_post':
            s = self.sample_variational(s_shape)

        elif weights_type == 'stored':
            s = self.s

        if weights_type == 'mean_prior' or weights_type == 'mean_prior' or n_samp is None:
            return self.act(F.linear(x, s*self.w, self.b)) # (n_obs, dim_hidden)
        else:
            xs = x.unsqueeze(1) * s.unsqueeze(0) # (n_obs, n_samp, dim_hidden)
            return self.act(F.linear(xs, self.w, self.b))

class RffVarSelectHsLayer(_RffVarSelectLayer):
    '''
    '''
    def __init__(self, dim_in, dim_out, b_g=1, b_0=1, infer_nu=True, nu=None):
        super().__init__(dim_in, dim_out)
        '''
        '''
        self.b_g = b_g
        self.b_0 = b_0

        self.infer_nu = infer_nu

        ### variational parameters

        # layer scales
        if self.infer_nu:
            self.nu_loc = nn.Parameter(torch.empty(1)) # of underlying normal
            self.nu_scale_untrans = nn.Parameter(torch.empty(1)) # of underlying normal

            self.register_buffer('vtheta_a', torch.empty(1))
            self.register_buffer('vtheta_b', torch.empty(1))
        
        else:
            self.register_buffer('nu', torch.tensor(nu))

        # input unit scales
        self.eta_loc = nn.Parameter(torch.empty(self.dim_in)) # of underlying normal
        self.eta_scale_untrans = nn.Parameter(torch.empty(self.dim_in)) # of underlying normal

        self.register_buffer('psi_a', torch.empty(self.dim_in))
        self.register_buffer('psi_b', torch.empty(self.dim_in))

        # input unit indicators
        self.s_loc = nn.Parameter(torch.empty(self.dim_in))
        self.s_scale_untrans = nn.Parameter(torch.empty(self.dim_in))

        ### priors

        # layer scales
        if self.infer_nu:
            self.nu_a_prior = torch.tensor(0.5)

            self.vtheta_a_prior = torch.tensor(0.5)
            self.vtheta_b_prior = torch.tensor(1/(b_g**2))

        # input unit scales
        self.eta_a_prior = torch.tensor(0.5) 

        self.psi_a_prior = torch.tensor(0.5)
        self.psi_b_prior = torch.tensor(1/(b_0**2))

        # input unit indicators
        self.s_loc_prior = torch.tensor(1.) # what do I want this to be? should it be 1?
        self.s_scale_prior = torch.tensor(1.)

        ### other stuff
        self.transform = nn.Softplus() # ensure correct range

        self.init_parameters()

    def init_parameters(self):
        # initialize variational parameters

        # layer scale
        if self.infer_nu:
            self.nu_loc.data.normal_(0, 1e-2)
            self.nu_scale_untrans.data.normal_(1e-4, 1e-2)

        # input unit scales
        self.eta_loc.data.normal_(0, 1e-2)
        self.eta_scale_untrans.data.normal_(1e-4, 1e-2)

        # input unit indicators
        self.s_loc.data.normal_(self.s_loc_prior, 1e-2)
        self.s_scale_untrans.data.normal_(1e-4, 1e-2)

        self.fixed_point_updates()

    def fixed_point_updates(self):

        # layer scale
        if self.infer_nu:
            self.vtheta_a = torch.tensor([1.]) # torch.ones((self.dim_out,)) # could do this in init
            self.vtheta_b = torch.exp(-self.nu_loc + 0.5*self.transform(self.nu_scale_untrans)) + 1/(self.b_g**2)

        # input unit scales
        self.psi_a = torch.ones((self.dim_in,)) # could do this in init
        self.psi_b = torch.exp(-self.eta_loc + 0.5*self.transform(self.eta_scale_untrans)) + 1/(self.b_0**2)

    def _get_prior_all(self):
        # returns all priors. includes aux variables
        s_dist = Normal(self.s_loc_prior*torch.ones(self.dim_in), self.s_scale_prior*torch.ones(self.dim_in))
        eta_dist = HalfCauchy(scale=self.b_0*torch.ones(self.dim_in)) # really should be inverse gamma, but then I'd need to sample conditional on psi
        psi_dist = InvGamma(self.psi_a_prior*torch.ones(self.dim_in), self.psi_b_prior*torch.ones(self.dim_in))

        if self.infer_nu:
            nu_dist = HalfCauchy(scale=self.b_g*torch.ones(1))
            vtheta_dist = InvGamma(self.vtheta_a_prior, self.vtheta_b_prior)
        else:
            nu_dist = PointMass(self.nu)
            vtheta_dist = None

        return s_dist, eta_dist, psi_dist, nu_dist, vtheta_dist

    def get_prior(self):
        s_dist, eta_dist, _, nu_dist, _ = self._get_prior_all()
        return ProductDistribution([s_dist, eta_dist, nu_dist])
        
    def _get_variational_all(self):
        # returns all variational distributions. includes aux variables
        s_dist = Normal(self.s_loc, self.transform(self.s_scale_untrans))
        eta_dist = LogNormal(loc=self.eta_loc, scale=self.transform(self.eta_scale_untrans))
        psi_dist = InvGamma(self.psi_a, self.psi_b)

        if self.infer_nu:
            nu_dist = LogNormal(loc=self.nu_loc, scale=self.transform(self.nu_scale_untrans))
            vtheta_dist = InvGamma(self.vtheta_a, self.vtheta_b)
        else:
            nu_dist = PointMass(self.nu)
            vtheta_dist = None

        return s_dist, eta_dist, psi_dist, nu_dist, vtheta_dist

    def get_variational(self):
        s_dist, eta_dist, _, nu_dist, _ = self._get_variational_all()
        return ProductDistribution([s_dist, eta_dist, nu_dist])

    def kl_divergence(self):
        '''
        overwrites parent class so aux variables included
        '''

        q_s, q_eta, q_psi, q_nu, q_vtheta = self._get_variational_all()
        p_s, p_eta, p_psi, p_nu, p_vtheta = self._get_variational_all()
        
        kl = 0.0

        # unit indicators (normal-normal)
        kl += torch.sum(kl_divergence(q_s, p_s))

        # eta
        kl += util.cross_entropy_cond_lognormal_invgamma_new(q_mu=self.eta_loc, 
                                                             q_sig2=self.transform(self.eta_scale_untrans).pow(2), 
                                                             q_alpha=self.psi_a, 
                                                             q_beta=self.psi_b, 
                                                             p_alpha=self.eta_a_prior) 
        kl += -self.q_eta.entropy()

        # psi
        kl += torch.sum(kl_divergence(q_psi, p_psi))


        if torch.infer_nu:
            # nu
            kl += util.cross_entropy_cond_lognormal_invgamma_new(q_mu=self.nu_loc, 
                                                                 q_sig2=self.transform(self.nu_scale_untrans).pow(2), 
                                                                 q_alpha=self.vtheta_a, 
                                                                 q_beta=self.vtheta_b, 
                                                                 p_alpha=self.nu_a_prior) 
            kl += -self.q_nu.entropy()

            # vtheta
            kl += torch.sum(kl_divergence(q_vtheta, p_vtheta))

        return kl

class RffVarSelectBetaLayer(_RffVarSelectLayer):
    '''
    '''
    def __init__(self, dim_in, dim_out, s_a_prior=1.0, s_b_prior=1.0):
        super().__init__(dim_in, dim_out)
        '''
        '''

        ### variational parameters

        # input unit indicators
        self.s_a_trans = nn.Parameter(torch.empty(self.dim_in))
        self.s_b_trans = nn.Parameter(torch.empty(self.dim_in))

        ### priors

        # input unit indicators
        self.s_a_prior = torch.tensor(s_a_prior)
        self.s_b_prior = torch.tensor(s_b_prior)
    
        ### other stuff
        self.untransform = nn.Softplus() # ensure correct range

        ### init params
        self.init_parameters()

    def init_parameters(self):
        # initialize variational parameters

        # input unit indicators
        self.s_a_trans.data.normal_(1, 1e-2)
        self.s_b_trans.data.normal_(1, 1e-2)

        self.sample_variational(store=True)

    def get_prior(self):
        return Beta(self.s_a_prior*torch.ones(self.dim_in), self.s_b_prior*torch.ones(self.dim_in))

    def get_variational(self):
        return Beta(self.untransform(self.s_a_trans), self.untransform(self.s_b_trans))

class RffVarSelectLogitNormalLayer(_RffVarSelectLayer):
    '''
    '''
    def __init__(self, dim_in, dim_out, s_loc_prior=0.0, s_scale_prior=1.0):
        super().__init__(dim_in, dim_out)
        '''
        '''
        ### variational parameters

        # input unit indicators
        self.s_loc = nn.Parameter(torch.empty(self.dim_in)) # of underlying normal
        self.s_scale_untrans = nn.Parameter(torch.empty(self.dim_in)) # of underlying normal

        ### priors

        # input unit indicators
        self.s_loc_prior = torch.tensor(s_loc_prior) # of underlying normal
        self.s_scale_prior = torch.tensor(s_scale_prior) # of underlying normal

        ### other stuff
        self.transform = nn.Softplus() # ensure correct range

        self.init_parameters()

    def init_parameters(self):
        self.s_loc.data.normal_(0, 1e-2)
        self.s_scale_untrans.data.normal_(1e-4, 1e-2)

        #self.s_loc.data.normal_(-10.0, 1e-2)
        #self.s_scale_untrans.data.normal_(10.0, 1e-2)
        #print('hey i changed this for an experiment')
        
        self.sample_variational(store=True)

    def get_prior(self):
        return LogitNormal(self.s_loc_prior*torch.ones(self.dim_in), self.s_scale_prior*torch.ones(self.dim_in))

    def get_variational(self):
        return LogitNormal(loc=self.s_loc, scale=self.transform(self.s_scale_untrans))

class RffVarSelectLogitNormalLayerHyper(_RffVarSelectLayer):
    '''
    '''
    def __init__(self, dim_in, dim_out, s_loc_prior=0.0, s_scale_prior=1.0, mu_loc_prior=0.0, mu_scale_prior=1.0):
        super().__init__(dim_in, dim_out)
        '''
        '''
        ### variational parameters

        # input unit indicators
        self.mu_loc = nn.Parameter(torch.empty(self.dim_in)) # of hyperprior
        self.mu_scale_untrans = nn.Parameter(torch.empty(self.dim_in)) # of hyperprior

        self.s_loc = nn.Parameter(torch.empty(self.dim_in)) # of underlying normal
        self.s_scale_untrans = nn.Parameter(torch.empty(self.dim_in)) # of underlying normal

        ### priors

        # input unit indicators
        self.mu_loc_prior = torch.tensor(mu_loc_prior) # of underlying normal
        self.mu_scale_prior = torch.tensor(mu_scale_prior) # of underlying normal

        self.s_scale_prior = torch.tensor(s_scale_prior) # of underlying normal

        ### other stuff
        self.transform = nn.Softplus() # ensure correct range

        self.init_parameters()

    def init_parameters(self):
        self.mu_loc.data.normal_(0, 1e-2)
        self.mu_scale_untrans.data.normal_(1e-4, 1e-2)

        self.s_loc.data.normal_(0, 1e-2)
        self.s_scale_untrans.data.normal_(1e-4, 1e-2)

        self.sample_variational(store=True)

    def get_prior(self):
        pass

    def get_variational(self):
        return LogitNormal(loc=self.s_loc, scale=self.transform(self.s_scale_untrans))

    def sample_prior(self, shape=torch.Size([]), store=False):
        p_mu = Normal(self.mu_loc_prior*torch.ones(self.dim_in), self.mu_scale_prior*torch.ones(self.dim_in))
        p_s = LogitNormal(loc=p_mu.rsample(shape), scale=self.s_scale_prior*torch.ones(self.dim_in))
        s = p_s.rsample(shape)
        if store: self.s = s
        return s

    def log_prob_variational(self):
        pass

    def kl_divergence(self):
        p_mu = Normal(self.mu_loc_prior*torch.ones(self.dim_in), self.mu_scale_prior*torch.ones(self.dim_in))
        q_mu = Normal(loc=self.mu_loc, scale=self.transform(self.mu_scale_untrans))
        q_s = get_variational()

        q_s.entropy()
        kl_divergence(q_mu, p_mu)

class RffVarSelectSpikeSlabLogNormal(_RffVarSelectLayer):
    '''
    '''
    def __init__(self, dim_in, dim_out, s_loc_prior=0.0, s_scale_prior=1.0, delta_pi_prior=0.5):
        super().__init__(dim_in, dim_out)
        '''
        # parameters:
        s: inverse lengthscale (called r in notes)
        delta: indicator variable

        # variational parameters:
        delta_pi: inclusion probability (called alpha in notes)
        s_loc
        s_scale

        # prior parameters:
        delta_pi_prior
        s_loc_prior
        s_scale_prior

        '''
        ### variational parameters

        # s: inverse lengthscale
        self.s_loc = nn.Parameter(torch.empty(self.dim_in)) # of underlying normal
        self.s_scale_untrans = nn.Parameter(torch.empty(self.dim_in)) # of underlying normal

        # pi: inclusion probability
        self.delta_pi = nn.Parameter(torch.empty(self.dim_in))

        ### priors

        # s: inverse lengthscale
        self.s_loc_prior = torch.tensor(s_loc_prior) # of underlying normal
        self.s_scale_prior = torch.tensor(s_scale_prior) # of underlying normal

        # pi: inclusion probability
        self.delta_pi_prior = torch.tensor(delta_pi_prior)

        ### other stuff
        self.transform = nn.Softplus() # ensure correct range
        self.transform_01 = lambda z: 1/(1+torch.exp(-z)) # logistic function

        self.init_parameters()

    def init_parameters(self):
        self.s_loc.data.normal_(1.0, 1e-2)
        self.s_scale_untrans.data.normal_(1e-1, 1e-2)
        #self.delta_pi.data.uniform_()
        self.delta_pi.data = .5*torch.ones(self.delta_pi.shape)
        self.sample_variational(store=True)

    def get_prior(self):
        # samples s from prior
        #p_delta = Bernoulli(self.delta_pi_prior).expand((self.dim_in,))
        p_delta = LogitRelaxedBernoulli(temperature=.2, probs=self.delta_pi_prior).expand((self.dim_in,))
        #p_s = Uniform(0, 5).expand((self.dim_in,))
        p_s = LogNormal(self.s_loc_prior, self.s_scale_prior).expand((self.dim_in,))
        return p_delta, p_s

    def get_variational(self):
        # samples s from prior
        #q_delta = Bernoulli(self.delta_pi)
        q_delta = LogitRelaxedBernoulli(temperature=.2, probs=self.transform_01(self.delta_pi))
        #q_s = Uniform(0, 5).expand((self.dim_in,))
        q_s = LogNormal(self.s_loc, self.transform(self.s_scale_untrans))
        return q_delta, q_s

    def sample_prior(self, shape=torch.Size([]), store=False):
        p_delta, p_s = self.get_prior()
        s = p_s.rsample(shape)*torch.sigmoid(p_delta.rsample(shape))
        if store: self.s = s 
        return s

    def sample_variational(self, shape=torch.Size([]), store=False):
        q_delta, q_s = self.get_variational()
        s = q_s.rsample(shape)*torch.sigmoid(q_delta.rsample(shape))
        if store: self.s = s
        return s

    def kl_divergence(self):
        _, p_s = self.get_prior()
        _, q_s = self.get_variational()
        delta_pi = self.transform_01(self.delta_pi)

        return torch.sum(delta_pi*(torch.log(delta_pi/self.delta_pi_prior) + kl_divergence(q_s, p_s)) \
            + (1-delta_pi)*torch.log((1-delta_pi)/(1-self.delta_pi_prior)))



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








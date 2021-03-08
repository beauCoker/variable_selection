# standard library imports
from functools import reduce
from numbers import Number

# package imports
import torch
from torch.distributions import constraints
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.distribution import Distribution
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all
from torch.distributions.kl import kl_divergence, register_kl

class LogitNormal(TransformedDistribution):
    r"""
    Creates a logit-normal distribution parameterized by
    :attr:`loc` and :attr:`scale` where::

        X ~ Normal(loc, scale)
        Y = Logistic(X) ~ LogitNormal(loc, scale)

    Example::

        >>> m = LogitNormal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # logit-normal distributed with mean=0 and stddev=1
        tensor([ 0.6798])

    Args:
        loc (float or Tensor): mean of logit of distribution
        scale (float or Tensor): standard deviation of logit of the distribution
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        base_dist = Normal(loc, scale)
        #super(LogitNormal, self).__init__(base_dist, SigmoidTransform(), validate_args=validate_args) # causes an error if using importlib.reload
        super().__init__(base_dist, SigmoidTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LogitNormal, _instance)
        return super(LogitNormal, self).expand(batch_shape, _instance=new)

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale

    @property
    def mean(self):
        raise NotImplementedError # No analytical solution

    @property
    def variance(self):
        raise NotImplementedError # No analytical solution

    def entropy(self):
        return self.base_dist.entropy() + self.loc # BC: is this correct?

@register_kl(LogitNormal, LogitNormal)
def _kl_logitnormal_logitnormal(p, q):
    return kl_divergence(p.base_dist, q.base_dist)



class ProductDistribution(Distribution):
    r"""
    """
    @property
    def mean(self):
        raise NotImplementedError

    @property
    def variance(self):
        raise NotImplementedError

    def __init__(self, dist_list):
        self.dist_list = dist_list
        batch_shape = dist_list[0].batch_shape
        #assert all(dist.batch_shape == batch_shape for dist in self.dist_list), 'batch shapes must match'
        super().__init__(batch_shape, validate_args=None)

    def expand(self, batch_shape, _instance=None):
        raise NotImplementedError

    def rsample(self, sample_shape=torch.Size()):
        return reduce(lambda x, y: x * y, [dist.rsample(sample_shape) for dist in self.dist_list])

    def sample(self, sample_shape=torch.Size()):
        return reduce(lambda x, y: x * y, [dist.sample(sample_shape) for dist in self.dist_list])

    def log_prob(self, value):
        return reduce(lambda x, y: x + y, [dist.log_prob(value) for dist in self.dist_list])

    def entropy(self):
        return reduce(lambda x, y: x + y, [dist.entropy() for dist in self.dist_list])


@register_kl(ProductDistribution, ProductDistribution)
def _kl_productdistribution_productdistribution(p, q):
    return reduce(lambda x, y: x + y, [kl_divergence(p_i, q_i) for p_i, q_i in zip(p.dist_list, q.dist_list)])


class PointMass(Distribution):
    r"""
    """
    @property
    def mean(self):
        raise self.loc

    @property
    def variance(self):
        raise torch.zeros(self.loc.shape)

    def __init__(self, loc):
        self.loc = loc
        if isinstance(loc, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()

        super().__init__(batch_shape, validate_args=None)

    def expand(self, batch_shape, _instance=None):
        raise NotImplementedError

    def rsample(self, sample_shape=torch.Size()):
        return self.loc.expand(batch_shape)

    def sample(self, sample_shape=torch.Size()):
        return self.loc.expand(batch_shape)

    def log_prob(self, value):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

@register_kl(PointMass, PointMass)
def _kl_pointmass_pointmass(p, q):
    return torch.tensor(0.)


def _standard_gamma(concentration):
    return torch._standard_gamma(concentration)

class InvGamma(ExponentialFamily):
    r"""
    concentration = alpha
    rate = beta
    """
    arg_constraints = {'concentration': constraints.positive, 'rate': constraints.positive}
    support = constraints.positive
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        assert self.concentration > 1, 'mean is not defined for concentration <= 1'
        return self.rate / (self.concentration - 1)

    @property
    def variance(self):
        assert self.concentration > 2, 'variance is not defined for concentration <= 2'
        return self.rate.pow(2) / ((self.concentration-1).pow(2)*(self.concentration-2))

    def __init__(self, concentration, rate, validate_args=None):
        self.concentration, self.rate = broadcast_all(concentration, rate)
        if isinstance(concentration, Number) and isinstance(rate, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.concentration.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(InvGamma, _instance)
        batch_shape = torch.Size(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        new.rate = self.rate.expand(batch_shape)
        super(InvGamma, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new


    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        value = _standard_gamma(self.concentration.expand(shape)) / self.rate.expand(shape)
        value.detach().clamp_(min=torch.finfo(value.dtype).tiny)  # do not record in autograd graph
        return 1.0/value


    def log_prob(self, value):
        value = torch.as_tensor(value, dtype=self.rate.dtype, device=self.rate.device)
        if self._validate_args:
            self._validate_sample(value)
        return (self.concentration * torch.log(self.rate) + \
                (-self.concentration - 1) * torch.log(value) - \
                value / self.rate - torch.lgamma(self.concentration))


    def entropy(self):
        return self.concentration + torch.log(self.rate) + torch.lgamma(self.concentration) - \
                (1.0 + self.concentration) * torch.digamma(self.concentration)





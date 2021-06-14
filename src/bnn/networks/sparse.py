# standard library imports
import os
import math
from math import sqrt, pi

# package imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions.multivariate_normal import MultivariateNormal
import tensorflow as tf
import tensorflow_probability as tfp

tf.keras.backend.set_floatx('float32')

# local imports
import bnn.layers.sparse as layers
import bnn.inference
import bnn.util as util

class BayesLinearLasso(nn.Module):
    """
    Linear regression with double expential prior
    """
    def __init__(self, dim_in, dim_out, prior_w2_sig2=1.0, noise_sig2=1.0, scale_global=1.0, groups=None, scale_groups=None):
        super(BayesLinearLasso, self).__init__()

        ### architecture
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.prior_w2_sig2 = prior_w2_sig2
        self.noise_sig2 = noise_sig2
        self.scale_global = scale_global
        self.groups = groups # list of lists with grouping (e.g. [[1,2,3], [4,5]])
        self.scale_groups = scale_groups

        
    def make_unnormalized_log_prob_tf(self, x, y):

        # Convert to tensors
        y_tf = tf.convert_to_tensor(y)
        x_tf = tf.convert_to_tensor(x)
        scale_global_tf = tf.dtypes.cast(tf.convert_to_tensor(self.scale_global), tf.float64)

        if self.groups is not None:
            groups_tf = [tf.convert_to_tensor(group) for group in self.groups]

        @tf.function
        def unnormalized_log_prob(w):
            resid = y_tf - x_tf@w

            # likelihood and L2 penalty
            log_prob = -1/(2*self.noise_sig2)*tf.transpose(resid)@(resid)
                       
            # L2 penalty?
            #log_prob += - 1/(2*prior_w2_sig2)*tf.transpose(w2)@w2

            # Within group
            log_prob -= tf.math.reduce_sum(scale_global_tf*tf.math.abs(w)) # L1 penalty
            #log_prob -= tf.math.reduce_sum(scale_global_tf*w**2) # L2 penalty
            

            # Group level
            if self.groups is not None:
                for scale_groups, group in zip(self.scale_groups, groups_tf):
                    log_prob -= scale_groups*tf.norm(tf.gather(w, group)) # L1 penalty
                    #log_prob -= scale_groups*tf.norm(tf.gather(w, group)**2) # L2 penalty

            return log_prob[0,0]

        return unnormalized_log_prob

    def train(self, x, y, num_results = int(10e3), num_burnin_steps = int(1e3)):
        '''
        Train with HMC
        '''
        unnormalized_log_prob_tf = self.make_unnormalized_log_prob_tf(x, y)
        init_values = .1*np.random.randn(self.dim_in,1)

        samples, accept = bnn.inference.mcmc.hmc_tf(unnormalized_log_prob_tf, 
            init_values, 
            num_results, 
            num_burnin_steps, 
            num_leapfrog_steps=3, 
            step_size=1.)

        return samples, accept

class RffGradPen(nn.Module):
    """
    Random features layer

    Variance of output layer scaled by width (see RFF activation function)
    """
    def __init__(self, dim_in, dim_hidden, dim_out, prior_w2_sig2=1.0, noise_sig2=1.0, scale_global=1.0, groups=None, scale_groups=None, lengthscale=1.0, penalty_type='l1'):
        super(RffGradPen, self).__init__()

        ### architecture
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.prior_w2_sig2 = prior_w2_sig2
        self.noise_sig2 = noise_sig2
        self.scale_global = scale_global
        self.groups = groups # list of lists with grouping (e.g. [[1,2,3], [4,5]])
        self.scale_groups = scale_groups
        self.lengthscale = lengthscale
        self.penalty_type = penalty_type

        self.register_buffer('w', torch.empty(dim_hidden, dim_in))
        self.register_buffer('b', torch.empty(dim_hidden))

        self.sample_features()

        self.act = lambda z: sqrt(2/self.dim_hidden)*torch.cos(z)

    def sample_features(self):
        # sample random weights for RFF features
        self.w.normal_(0, 1 / self.lengthscale)
        self.b.uniform_(0, 2*pi)

    def hidden_features(self, x):
        #return self.act(x@self.w.T + self.b.reshape(1,-1)) # (n, dim_hidden)
        return self.act(F.linear(x, self.w, self.b)) # (n, dim_hidden)

    def compute_jacobian(self, x):
        '''
        Compute jacobian of hidden units with respect to inputs. 
        Assumes inputs do not impact each other (i.e. input observation n only impacts hidden for observation n)

        Inputs:
            x: (n_obs, dim_in) tensor

        Outputs:
            jac: (n_obs, dim_out, dim_in) tensor of derivatives
        '''
        jac = []
        for n in range(x.shape[0]):
            jac_n = torch.autograd.functional.jacobian(self.hidden_features, x[n,:].reshape(1,-1)).squeeze() # dim_hidden x dim_in
            jac.append(jac_n)
        return torch.stack(jac) # n_obs x dim_out x dim_in


    def compute_Ax(self, x):
        '''
        Computes A matrix
        '''
        n = x.shape[0]
        J = self.compute_jacobian(x) # N x K x D

        #Ja = -sqrt(2/self.dim_hidden) * self.w.unsqueeze(0) * torch.sin(F.linear(x, self.w, self.b)).unsqueeze(-1) #analytical jacobian

        # all inputs
        A_d = [1/n*J[:,:,d].T@J[:,:,d] for d in range(self.dim_in)]

        # groups of inputs
        if self.groups is not None:
            A_groups = [torch.sum(torch.stack([A_d[i] for i in group]),0) for group in self.groups]
        else:
            A_groups = None

        return A_d, A_groups

    def make_unnormalized_log_prob_tf(self, x, y):

        # Set prior (since based on data)
        Ax_d, Ax_groups = self.compute_Ax(x)

        # Convert to tensors
        y_tf = tf.convert_to_tensor(y)
        h_tf = tf.convert_to_tensor(self.hidden_features(x))
        Ax_d_tf = [tf.convert_to_tensor(A) for A in Ax_d]
        
        if Ax_groups is not None:
            Ax_groups_tf = [tf.convert_to_tensor(A) for A in Ax_groups]

        @tf.function
        def unnormalized_log_prob(w):
            resid = y_tf - h_tf@w

            # likelihood
            log_prob = -1/(2*self.noise_sig2)*tf.transpose(resid)@(resid)

            # L2 penalty
            log_prob += - 1/(2*self.prior_w2_sig2)*tf.transpose(w)@w 

            ## likelihood and L2 penalty
            #log_prob = -1/self.noise_sig2*tf.transpose(resid)@(resid) \
            #           - tf.transpose(w)@(1/self.prior_w2_sig2*tf.eye(self.dim_hidden, dtype=tf.float64))@w 

            # Within group gradient penalty
            for scale_global, A in zip(self.scale_global, Ax_d_tf):
                grad_f_sq = tf.transpose(w)@(A)@w
                if self.penalty_type == 'l1':
                    log_prob += - scale_global*tf.math.sqrt(grad_f_sq)
                elif self.penalty_type == 'l2':
                    log_prob += - scale_global*grad_f_sq

            # Group level gradient penalty
            if Ax_groups is not None:
                for scale_groups, A in zip(self.scale_groups, Ax_groups_tf):
                    log_prob -= scale_groups*tf.math.sqrt(tf.transpose(w)@(A)@w)

            return log_prob[0,0]

        return unnormalized_log_prob


    def train(self, x, y, num_results = int(10e3), num_burnin_steps = int(1e3)):
        '''
        Train with HMC
        '''
        unnormalized_log_prob_tf = self.make_unnormalized_log_prob_tf(x, y)

        init_values = .1*np.random.randn(self.dim_hidden,1) #tf.constant(.01, shape=(self.dim_hidden,1), dtype=tf.float64) 

        samples, accept = bnn.inference.mcmc.hmc_tf(unnormalized_log_prob_tf, 
            init_values, 
            num_results, 
            num_burnin_steps, 
            num_leapfrog_steps=3, 
            step_size=1.)

        return samples, accept


class RffGradPenHyper(nn.Module):
    """
    Random features layer
    INCLUDES PRIOR ON lengthscale AND HYPERPRIOR ON prior_w2_sig2

    Variance of output layer scaled by width (see RFF activation function)

    Inputs:
    -   dim_in: dimension of inputs (int)
    -   dim_hidden: number of hidden units (int)
    -   dim_out: output dimension (int)
    -   prior_w2_sig2: prior variance of output weights. Corresponds to amplitude variance of RBF kernel. (scalar)
    -   noise_sig2: observational noise (scalar)
    -   scale_global: NOT IMPLEMENTED
    -   groups: NOT IMPLEMENTED
    -   scale_groups: NOT IMPLEMENTED
    -   lengthscale: Corresponds to lengthscale of RBF kernel. (scalar)
    -   penalty_type: select 'l1' for lasso penalty, 'l2' for ridge penalty (str)
    """
    def __init__(self, dim_in, dim_hidden, dim_out, prior_w2_sig2=1.0, noise_sig2=1.0, scale_global=1.0, groups=None, scale_groups=None, lengthscale=1.0, penalty_type='l1'):
        super(RffGradPenHyper, self).__init__()

        ### architecture
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.prior_w2_sig2 = prior_w2_sig2
        self.noise_sig2 = noise_sig2
        self.scale_global = scale_global
        self.groups = groups # list of lists with grouping (e.g. [[1,2,3], [4,5]])
        self.scale_groups = scale_groups
        self.lengthscale = lengthscale
        self.penalty_type = penalty_type

        self.register_buffer('w', torch.empty(dim_hidden, dim_in))
        self.register_buffer('b', torch.empty(dim_hidden))

        self.sample_features()

        self.act = lambda z: sqrt(2/self.dim_hidden)*torch.cos(z)
        self.act_tf = lambda z: sqrt(2/self.dim_hidden)*tf.math.cos(z)

    def sample_features(self):
        # sample random weights for RFF features
        self.w.normal_(0, 1)
        self.b.uniform_(0, 2*pi)

        self.w_tf = tf.convert_to_tensor(self.w)
        self.b_tf = tf.convert_to_tensor(self.b)
        
    def hidden_features(self, x, lengthscale=1.0):
        #return self.act(x@self.w.T + self.b.reshape(1,-1)) # (n, dim_hidden)
        return self.act(F.linear(x, self.w / lengthscale, self.b)) # (n, dim_hidden)

    def hidden_features_tf(self, x, lengthscale=1.0):
        #return self.act(x@self.w.T + self.b.reshape(1,-1)) # (n, dim_hidden)
        return self.act_tf(x @ tf.transpose(self.w_tf) / lengthscale + tf.reshape(self.b_tf, (1,-1))) # (n, dim_hidden)

    def hidden_features_tf_precompute(self, x_w_tf, lengthscale=1.0):
        return self.act_tf(x_w_tf / lengthscale + tf.reshape(self.b_tf, (1,-1))) # (n, dim_hidden)

    def make_unnormalized_log_prob_tf(self, x, y):

        # Convert to tensors
        x_tf = tf.convert_to_tensor(x)
        y_tf = tf.convert_to_tensor(y)
        n = x.shape[0]

        # for lengthscale prior and prior_w2_sig2 hyperprior
        l_alpha = tf.convert_to_tensor(1.0, dtype=tf.float64)
        l_beta = tf.convert_to_tensor(1.0, dtype=tf.float64)

        prior_w2_sig2_alpha = tf.convert_to_tensor(1.0, dtype=tf.float64)
        prior_w2_sig2_beta = tf.convert_to_tensor(1.0, dtype=tf.float64)

        def log_prob_invgamma(x, alpha, beta):
            unnormalized_prob = -(1. + alpha) * tf.math.log(x) - beta / x
            normalization = (
            tf.math.lgamma(alpha) - alpha * tf.math.log(beta))
            return unnormalized_prob - normalization

        # precompute
        x_w_tf = x @ tf.transpose(self.w_tf)

        @tf.function
        def unnormalized_log_prob(w, l, prior_w2_sig2):
            '''
            w: output layer weights
            l: lengthscale
            '''

            h_tf = self.hidden_features_tf_precompute(x_w_tf, l)
            resid = y_tf - h_tf@w

            # Jacobian of hidden layer (N x K x D)
            J = -sqrt(2/self.dim_hidden) * tf.expand_dims(self.w_tf,0) / l * tf.expand_dims(tf.math.sin(x_w_tf / l + tf.reshape(self.b_tf, (1,-1))), -1) # analytical jacobian
            
            # gradient penalties for each input dimension
            Ax_d_tf = [1/n*tf.transpose(J[:,:,d])@J[:,:,d] for d in range(self.dim_in)]

            # likelihood
            log_prob = -1/(2*self.noise_sig2)*tf.transpose(resid)@(resid)

            # L2 penalty
            log_prob += - 1/(2*prior_w2_sig2)*tf.transpose(w)@w 

            # prior_w2_sig2 hyperprior
            log_prob += log_prob_invgamma(prior_w2_sig2, prior_w2_sig2_alpha, prior_w2_sig2_beta)

            # lengthscale prior
            log_prob += log_prob_invgamma(l, l_alpha, l_beta)
            
            # Within group gradient penalty
            for scale_global, A in zip(self.scale_global, Ax_d_tf):
                grad_f_sq = tf.transpose(w)@(A)@w
                if self.penalty_type == 'l1':
                    log_prob += - scale_global*tf.math.sqrt(grad_f_sq)
                elif self.penalty_type == 'l2':
                    log_prob += - scale_global*grad_f_sq

            '''
            # Group level gradient penalty
            if Ax_groups is not None:
                for scale_groups, A in zip(self.scale_groups, Ax_groups_tf):
                    log_prob -= scale_groups*tf.math.sqrt(tf.transpose(w)@(A)@w)
            '''

            return log_prob[0,0]

        return unnormalized_log_prob


    def train(self, x, y, num_results = int(10e3), num_burnin_steps = int(1e3)):
        '''
        Train with HMC
        '''
        unnormalized_log_prob_tf = self.make_unnormalized_log_prob_tf(x, y)
        init_values = [.1*np.random.randn(self.dim_hidden,1), tf.constant(1.0, dtype=tf.float64), tf.constant(1.0, dtype=tf.float64)]

        samples, accept = bnn.inference.mcmc.hmc_tf(unnormalized_log_prob_tf, 
            init_values, 
            num_results, 
            num_burnin_steps, 
            num_leapfrog_steps=3, 
            step_size=1.)

        return samples, accept


class RffGradPenHyper_v2(object):
    """
    Random features layer
    INCLUDES PRIOR ON lengthscale AND HYPERPRIOR ON prior_w2_sig2

    Variance of output layer scaled by width (see RFF activation function)

    Inputs:
    -   dim_in: dimension of inputs (int)
    -   dim_hidden: number of hidden units (int)
    -   dim_out: output dimension (int)
    -   prior_w2_sig2: prior variance of output weights. Corresponds to amplitude variance of RBF kernel. (scalar)
    -   noise_sig2: observational noise (scalar)
    -   scale_global: NOT IMPLEMENTED
    -   groups: NOT IMPLEMENTED
    -   scale_groups: NOT IMPLEMENTED
    -   lengthscale: Corresponds to lengthscale of RBF kernel. (scalar)
    -   penalty_type: select 'l1' for lasso penalty, 'l2' for ridge penalty (str)
    """
    def __init__(self, dim_in, dim_hidden, dim_out, prior_w2_sig2=1.0, noise_sig2=1.0, scale_global=1.0, groups=None, scale_groups=None, lengthscale=1.0, penalty_type='l1', family='gaussian'):
        super(RffGradPenHyper_v2, self).__init__()

        ### architecture
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.prior_w2_sig2 = prior_w2_sig2
        self.noise_sig2 = noise_sig2
        self.scale_global = scale_global
        self.groups = groups # list of lists with grouping (e.g. [[1,2,3], [4,5]])
        self.scale_groups = scale_groups
        self.lengthscale = lengthscale
        self.penalty_type = penalty_type
        self.family = family

        self.sample_features()
        self.act = lambda z: sqrt(2/self.dim_hidden)*tf.math.cos(z)

    def sample_features(self):
        # sample random weights for RFF features
        self.w1 = tf.cast(tf.convert_to_tensor(np.random.normal(0,1,(self.dim_hidden, self.dim_in))), dtype=tf.float32)
        self.b1 = tf.cast(tf.convert_to_tensor(np.random.uniform(0,2*pi,(self.dim_hidden,))), dtype=tf.float32)

    def compute_xw1(self, x):
        if not tf.is_tensor(x):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
        return x @ tf.transpose(self.w1)

    def hidden_features(self, x=None, xw1=None, lengthscale=None):
        if xw1 is None:
            xw1 = self.compute_xw1(x)
        if lengthscale is None:
            lengthscale = self.lengthscale

        return self.act(xw1 / lengthscale + tf.reshape(self.b1, (1,-1))) # (n, dim_hidden)

    def forward(self, w2, x=None, xw1=None, lengthscale=None, h=None):
        if h is None:
            if xw1 is None:
                xw1 = self.compute_xw1(x)
            if lengthscale is None:
                lengthscale = self.lengthscale
            h = self.hidden_features(x, xw1, lengthscale)
        
        return h@tf.reshape(w2,(-1,1))

    def jacobian_hidden_features(self, x=None, xw1=None, lengthscale=None):
        if xw1 is None:
            xw1 = self.compute_xw1(x)
        if lengthscale is None:
            lengthscale = self.lengthscale
        return -sqrt(2/self.dim_hidden) * tf.expand_dims(self.w1,0) / lengthscale * tf.expand_dims(tf.math.sin(xw1 / lengthscale + tf.reshape(self.b1, (1,-1))), -1) # analytical jacobian

    def grad_norm(self, x=None, xw1=None, lengthscale=None):
        J = self.jacobian_hidden_features(x=x, xw1=xw1, lengthscale=lengthscale)
        Ax_d = [1/J.shape[0]*tf.transpose(J[:,:,d])@J[:,:,d] for d in range(self.dim_in)]
        return Ax_d

    def log_marginal_likelihood(self, x, y):


        ## GPy for comparison
        import GPy
        kernel = GPy.kern.RBF(input_dim=x.shape[1], lengthscale=self.lengthscale, variance=self.prior_w2_sig2)
        K_gp = kernel.K(x,x) # (n, n)

        gp = GPy.models.GPRegression(x,y,kernel)
        gp.Gaussian_noise.variance = self.noise_sig2

        log_prob_gp = gp.log_likelihood()


        ## regular log marginal likelihood
        n = x.shape[0]
        x = tf.cast(tf.convert_to_tensor(x), tf.float32)
        y = tf.cast(tf.convert_to_tensor(y), tf.float32)

        h = self.hidden_features(x)
        K = h @ tf.transpose(h) * self.prior_w2_sig2

        A = K + self.noise_sig2*tf.eye(n)

        log_prob = -0.5*n*np.log(2*np.pi) - 0.5*tf.linalg.logdet(A) - 0.5*tf.transpose(y) @ tf.linalg.inv(A) @ y


        ## faster way
        m = int(self.dim_hidden / 2)
        phi = h * np.sqrt(self.dim_hidden/2)


        A = tf.transpose(phi)@phi + m*self.noise_sig2/self.prior_w2_sig2*tf.eye(2*m)

        R = tf.linalg.cholesky(A)
        alpha1 = tf.linalg.solve(R, tf.transpose(phi)@y)

        log_prob2 = -1/(2*self.noise_sig2)*(tf.norm(y)**2 - tf.norm(alpha1)**2) - 0.5*tf.reduce_sum(tf.math.log(tf.linalg.diag_part(R)**2)) + m*np.log(m*self.noise_sig2/self.prior_w2_sig2) - n/2*np.log(2*np.pi*self.noise_sig2)

        breakpoint()

    def make_log_marginal_likelihood(self, x, y):

        xw1 = self.compute_xw1(x)

        @tf.function
        def unnormalized_log_prob(lengthscale, xw1=xw1):

            n = x.shape[0]
            h = self.hidden_features(x=None, xw1=xw1, lengthscale=lengthscale)
            m = int(self.dim_hidden / 2)
            phi = h * np.sqrt(self.dim_hidden/2)
            A = tf.transpose(phi)@phi + m*self.noise_sig2/self.prior_w2_sig2*tf.eye(2*m)
            R = tf.linalg.cholesky(A)
            alpha1 = tf.linalg.solve(R, tf.transpose(phi)@y)
            log_prob = -1/(2*self.noise_sig2)*(tf.norm(y)**2 - tf.norm(alpha1)**2) - 0.5*tf.reduce_sum(tf.math.log(tf.linalg.diag_part(R)**2)) + m*tf.math.log(m*self.noise_sig2/self.prior_w2_sig2) - n/2*np.log(2*np.pi*self.noise_sig2)

            return log_prob

        return unnormalized_log_prob



    def train_log_marginal_likelihood(self, x, y, n_epochs, learning_rate=0.001, early_stopping=False, tol=1e-4, patience=3, clipvalue=100, batch_size=None):
        
        x = tf.cast(tf.convert_to_tensor(x), tf.float32)
        y = tf.cast(tf.convert_to_tensor(y), tf.float32)

        hyperparam_hist = {}
        

        lengthscale_map = tfp.math.softplus_inverse(tf.constant(self.lengthscale, dtype=tf.float32)) # note: _map is untransformed by softplus
        
        lengthscale_map = tf.Variable(lengthscale_map, dtype=np.float32)
        hyperparam_hist['lengthscale'] = [tf.math.softplus(lengthscale_map).numpy()]
        print('lengthscale init: ', hyperparam_hist['lengthscale'][0])

        #prior_w2_sig2_map = tfp.math.softplus_inverse(tf.constant(self.prior_w2_sig2, dtype=tf.float32)) # note: _map is untransformed by softplus
        #prior_w2_sig2_map = tf.Variable(prior_w2_sig2_map, dtype=np.float32)
        #hyperparam_hist['prior_w2_sig2'] = [tf.math.softplus(prior_w2_sig2_map).numpy()]
        #print('prior_w2_sig2 init: ', hyperparam_hist['prior_w2_sig2'][0])

        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, clipvalue=clipvalue)

        train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        if batch_size is None:
            batch_size = x.shape[0]# full batch
        train_dataset = train_dataset.batch(batch_size)

        n_no_improvement = 0 # for early stopping

        neg_log_marginal_likelihoods = [] # one for each batch
        for xbatch, ybatch in train_dataset:
            log_marginal_likelihood_ = self.make_log_marginal_likelihood(xbatch, ybatch)

            log_marginal_likelihood = lambda lengthscale: log_marginal_likelihood_(tf.math.softplus(lengthscale)) # use softpluses
            var_list = [lengthscale_map]

            neg_log_marginal_likelihoods.append(lambda: -log_marginal_likelihood(*var_list)) # evaluate on Variables

        for epoch in range(n_epochs):
            for neg_log_marginal_likelihood in neg_log_marginal_likelihoods:

                opt.minimize(neg_log_marginal_likelihood, var_list=var_list)

                ### if you want to processes the gradients
                #with tf.GradientTape() as tape:
                #    loss = neg_log_marginal_likelihood()
                #grads = tape.gradient(loss, var_list)
                #grads = [tf.clip_by_norm(g, 500.) for g in grads]
                #opt.apply_gradients(zip(grads, var_list))
                #print(tf.math.softplus(lengthscale_map))
                
                ###

            hyperparam_hist['lengthscale'].append(tf.math.softplus(lengthscale_map).numpy().item())
            #hyperparam_hist['prior_w2_sig2'].append(tf.math.softplus(prior_w2_sig2_map).numpy().item())

            if early_stopping:
                if np.all([np.abs(val[-1] - val[-2]) < tol for val in hyperparam_hist.values()]):
                    n_no_improvement += 1
                else:
                    n_no_improvement = 0
                if n_no_improvement==patience:
                    break

        # unpack
        self.lengthscale = tf.convert_to_tensor(tf.math.softplus(lengthscale_map)).numpy().item()
        #self.prior_w2_sig2 = tf.convert_to_tensor(tf.math.softplus(prior_w2_sig2_map)).numpy().item()

        print('lengthscale final: ', self.lengthscale)
        #print('prior_w2_sig2 final: ', self.prior_w2_sig2)
        
        return hyperparam_hist













    def make_unnormalized_log_prob(self, x, y, infer_lengthscale=False):

        # for lengthscale prior and prior_w2_sig2 hyperprior (should move this to init...)
        lengthscale_alpha = tf.convert_to_tensor(1.0, dtype=tf.float32)
        lengthscale_beta = tf.convert_to_tensor(1.0, dtype=tf.float32)

        prior_w2_sig2_alpha = tf.convert_to_tensor(1.0, dtype=tf.float32)
        prior_w2_sig2_beta = tf.convert_to_tensor(1.0, dtype=tf.float32)

        def log_prob_invgamma(x, alpha, beta):
            unnormalized_prob = -(1. + alpha) * tf.math.log(x) - beta / x
            normalization = (tf.math.lgamma(alpha) - alpha * tf.math.log(beta))
            return unnormalized_prob - normalization

        # precompute
        xw1 = self.compute_xw1(x)
        h = self.hidden_features(x=None, xw1=xw1, lengthscale=self.lengthscale)
        Ax_d = self.grad_norm(x=None, xw1=xw1, lengthscale=self.lengthscale)

        @tf.function
        def unnormalized_log_prob(w2, lengthscale, prior_w2_sig2, infer_lengthscale=infer_lengthscale, xw1=xw1, h=h, Ax_d=Ax_d, family=self.family):
            '''
            w2: output layer weights
            lengthscale: lengthscale
            prior_w2_sig2: prior variance of output layer weights

            lengthscale and prior_w2_sig2 are only used if infer_hyper is True
            '''

            if infer_lengthscale:
                # recompute hidden features and gradient penalty (because they depend on lengthscale)
                h = self.hidden_features(x=None, xw1=xw1, lengthscale=lengthscale)
                Ax_d = self.grad_norm(x=None, xw1=xw1, lengthscale=lengthscale)

            f_pred = self.forward(w2, h=h)
            resid = y - f_pred

            # likelihood
            if family == 'gaussian':
                log_prob = -1/(2*self.noise_sig2)*tf.transpose(resid)@(resid)
            elif family == 'poisson':
                log_prob = tf.reduce_sum(y * f_pred - tf.math.exp(f_pred))
            elif family == 'binomial':
                #p_pred = tf.math.sigmoid(f_pred)
                #log_prob = tf.reduce_sum(y*tf.math.log(p_pred) + (1-y)*tf.math.log(p_pred))
                log_prob = tf.reduce_sum(y*f_pred - tf.math.log(1+tf.math.exp(f_pred)))

            # L2 penalty
            log_prob += - 1/(2*prior_w2_sig2)*tf.transpose(w2)@w2
            
            # prior_w2_sig2 hyperprior
            log_prob += log_prob_invgamma(prior_w2_sig2, prior_w2_sig2_alpha, prior_w2_sig2_beta) 

            # lengthscale prior
            log_prob += log_prob_invgamma(lengthscale**2, lengthscale_alpha, lengthscale_beta) 
            
            # Within group gradient penalty
            for scale_global, A in zip(self.scale_global, Ax_d):
                grad_f_sq = tf.transpose(w2)@(A)@w2
                if self.penalty_type == 'l1':
                    log_prob += - scale_global*tf.math.sqrt(grad_f_sq)
                elif self.penalty_type == 'l2':
                    log_prob += - scale_global*grad_f_sq

            '''
            # Group level gradient penalty
            if Ax_groups is not None:
                for scale_groups, A in zip(self.scale_groups, Ax_groups_tf):
                    log_prob -= scale_groups*tf.math.sqrt(tf.transpose(w)@(A)@w)
            '''

            return log_prob[0,0]

        #@tf.function
        def unnormalized_log_prob_vec(params):
            return unnormalized_log_prob(w2=tf.reshape(params[:-2],(-1,1)), lengthscale=params[-2], prior_w2_sig2=params[-1])


        return unnormalized_log_prob, unnormalized_log_prob_vec



    def train_map(self, x, y, n_epochs, learning_rate=0.001, early_stopping=False, tol=1e-4, patience=3, clipvalue=100, batch_size=None, infer_lengthscale=True, infer_prior_w2_sig2=True):
        x = tf.cast(tf.convert_to_tensor(x), tf.float32)
        y = tf.cast(tf.convert_to_tensor(y), tf.float32)

        hyperparam_hist = {}
        
        # starting values
        w2_map = tf.Variable(np.random.randn(self.dim_hidden,1)/self.dim_hidden, dtype=np.float32)


        lengthscale_map = tfp.math.softplus_inverse(tf.constant(self.lengthscale, dtype=tf.float32)) # note: _map is untransformed by softplus
        if infer_lengthscale:
            lengthscale_map = tf.Variable(lengthscale_map, dtype=np.float32)
            hyperparam_hist['lengthscale'] = [tf.math.softplus(lengthscale_map).numpy()]
            print('lengthscale init: ', hyperparam_hist['lengthscale'][0])

        prior_w2_sig2_map = tfp.math.softplus_inverse(tf.constant(self.prior_w2_sig2, dtype=tf.float32)) # note: _map is untransformed by softplus
        if infer_prior_w2_sig2:
            prior_w2_sig2_map = tf.Variable(prior_w2_sig2_map, dtype=np.float32)
            hyperparam_hist['prior_w2_sig2'] = [tf.math.softplus(prior_w2_sig2_map).numpy()]
            print('lengthscale init: ', hyperparam_hist['prior_w2_sig2'][0])

        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, clipvalue=clipvalue)

        train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        if batch_size is None:
            batch_size = x.shape[0]# full batch
        train_dataset = train_dataset.batch(batch_size)

        n_no_improvement = 0 # for early stopping

        unnormalized_neg_log_probs = [] # one for each batch
        for xbatch, ybatch in train_dataset:
            unnormalized_log_prob_, _ = self.make_unnormalized_log_prob(xbatch, ybatch, infer_lengthscale=infer_lengthscale)

            if infer_lengthscale and infer_prior_w2_sig2:
                unnormalized_log_prob = lambda w2, lengthscale, prior_w2_sig2: unnormalized_log_prob_(w2, tf.math.softplus(lengthscale), tf.math.softplus(prior_w2_sig2)) # use softpluses
                var_list = [w2_map, lengthscale_map, prior_w2_sig2_map]

            elif infer_lengthscale and (not infer_prior_w2_sig2):
                unnormalized_log_prob = lambda w2, lengthscale: unnormalized_log_prob_(w2, tf.math.softplus(lengthscale), tf.math.softplus(prior_w2_sig2_map)) # use softpluses
                var_list = [w2_map, lengthscale_map]

            elif (not infer_lengthscale) and infer_prior_w2_sig2:
                unnormalized_log_prob = lambda w2, prior_w2_sig2: unnormalized_log_prob_(w2, tf.math.softplus(lengthscale_map), tf.math.softplus(prior_w2_sig2)) # use softpluses
                var_list = [w2_map, prior_w2_sig2_map]

            else:
                unnormalized_log_prob = lambda w2: unnormalized_log_prob_(w2, tf.math.softplus(lengthscale_map), tf.math.softplus(prior_w2_sig2_map)) # use softpluses
                var_list = [w2_map]

            unnormalized_neg_log_probs.append(lambda: -unnormalized_log_prob(*var_list)) # evaluate on Variables

        for epoch in range(n_epochs):
            for unnormalized_neg_log_prob in unnormalized_neg_log_probs:
                #opt.minimize(unnormalized_neg_log_prob, var_list=var_list)

                ### if you want to processes the gradients
                with tf.GradientTape() as tape:
                    loss = unnormalized_neg_log_prob()
                grads = tape.gradient(loss, var_list)
                grads = [tf.clip_by_norm(g, 500.) for g in grads]
                opt.apply_gradients(zip(grads, var_list))
                print(tf.math.softplus(lengthscale_map))
                ###

            if infer_lengthscale:
                hyperparam_hist['lengthscale'].append(tf.math.softplus(lengthscale_map).numpy().item())

            if infer_prior_w2_sig2:
                hyperparam_hist['prior_w2_sig2'].append(tf.math.softplus(prior_w2_sig2_map).numpy().item())

            if early_stopping:
                if np.all([np.abs(val[-1] - val[-2]) < tol for val in hyperparam_hist.values()]):
                    n_no_improvement += 1
                else:
                    n_no_improvement = 0
                if n_no_improvement==patience:
                    break

        # unpack
        w2 = tf.convert_to_tensor(w2_map)
        if infer_lengthscale:
            self.lengthscale = tf.convert_to_tensor(tf.math.softplus(lengthscale_map)).numpy().item()

        if infer_prior_w2_sig2:
            self.prior_w2_sig2 = tf.convert_to_tensor(tf.math.softplus(prior_w2_sig2_map)).numpy().item()

        print('lengthscale final: ', self.lengthscale)
        print('prior_w2_sig2 final: ', self.prior_w2_sig2)
        
        return w2, hyperparam_hist


    def train(self, x, y, num_results = int(10e3), num_burnin_steps = int(1e3), infer_lengthscale=False, infer_prior_w2_sig2=False, w2_init=None):
        '''
        '''
        x = tf.cast(tf.convert_to_tensor(x), tf.float32)
        y = tf.cast(tf.convert_to_tensor(y), tf.float32)

        # initialize w2 randomly or to MAP
        if w2_init is None:
            w2_init = tf.cast(tf.convert_to_tensor(np.random.randn(self.dim_hidden,1)/self.dim_hidden), tf.float32)

        # set up objective and initialization depending if variational parameters inferred
        unnormalized_log_prob_, _ = self.make_unnormalized_log_prob(x, y, infer_lengthscale=infer_lengthscale)

        if infer_lengthscale and infer_prior_w2_sig2:
            unnormalized_log_prob = lambda w2, lengthscale, prior_w2_sig2: unnormalized_log_prob_(w2, lengthscale, prior_w2_sig2) 
            init_values = [w2_init, tf.constant(self.lengthscale, dtype=tf.float32), tf.constant(self.prior_w2_sig2, dtype=tf.float32)]

        elif infer_lengthscale and (not infer_prior_w2_sig2):
            unnormalized_log_prob = lambda w2, lengthscale: unnormalized_log_prob_(w2, lengthscale, self.prior_w2_sig2)
            init_values = [w2_init, tf.constant(self.lengthscale, dtype=tf.float32)]

        elif (not infer_lengthscale) and infer_prior_w2_sig2:
            unnormalized_log_prob = lambda w2, prior_w2_sig2: unnormalized_log_prob_(w2, self.lengthscale, prior_w2_sig2)
            init_values = [w2_init, tf.constant(self.prior_w2_sig2, dtype=tf.float32)]

        else:
            unnormalized_log_prob = lambda w2: unnormalized_log_prob_(w2, self.lengthscale, self.prior_w2_sig2)
            init_values = w2_init

        ###
        #breakpoint()
        #unnormalized_log_prob(w2_init)
        ###

        samples, accept = bnn.inference.mcmc.hmc_tf(unnormalized_log_prob, 
            init_values, 
            num_results, 
            num_burnin_steps, 
            num_leapfrog_steps=3, 
            step_size=1.)

        return samples, accept


    def fit(self, x, y):
        '''
        Computes conjugate posterior
        '''
        assert self.penalty_type == 'l2'
        #assert np.all([s==self.scale_global[0] for s in self.scale_global]) # only works if all scales are the same (easy to adapt if not though)
        
        x = tf.cast(tf.convert_to_tensor(x), tf.float32)
        y = tf.cast(tf.convert_to_tensor(y), tf.float32)

        h = self.hidden_features(x, lengthscale=self.lengthscale)
        Ax_d = self.grad_norm(x, lengthscale=self.lengthscale)
        Ax_d = [s*A for s,A in zip(self.scale_global, Ax_d)] # multiply by scale
        Ax = tf.reduce_sum(tf.stack(Ax_d),0) # sum over input dimension

        prior_sig2inv_mat = 1/self.prior_w2_sig2*tf.eye(self.dim_hidden) + Ax # prior includes gradient penalty
        sig2 = tf.linalg.inv(prior_sig2inv_mat + tf.transpose(h)@(h)/self.noise_sig2) # Should replace with cholesky
        mu = sig2 @ tf.transpose(h)@y/self.noise_sig2

        return mu, sig2

    

class RffGradPenHyper_v3(object):
    """
    - L2 penalty only
    - closed-form solution
    - hyperparameters: lengthscale, prior_w2_sig2 (amplitude variance), and scale_global (regularization strength)
    - lengthscale and scale_global hyperparameters can be input-specific 

    Inputs:
    -   dim_in: dimension of inputs (int)
    -   dim_hidden: number of hidden units (int)
    -   dim_out: output dimension (int)
    -   prior_w2_sig2: prior variance of output weights. Corresponds to amplitude variance of RBF kernel. (scalar)
    -   noise_sig2: observational noise (scalar)
    -   scale_global: NOT IMPLEMENTED
    -   groups: NOT IMPLEMENTED
    -   scale_groups: NOT IMPLEMENTED
    -   lengthscale: Corresponds to lengthscale of RBF kernel. (scalar)
    -   penalty_type: select 'l1' for lasso penalty, 'l2' for ridge penalty (str)
    """
    def __init__(self, dim_in, dim_hidden, dim_out, prior_w2_sig2=1.0, noise_sig2=1.0, scale_global=1.0, groups=None, scale_groups=None, lengthscale=1.0, penalty_type='l2', family='gaussian', dtype=tf.float32):
        super(RffGradPenHyper_v3, self).__init__()

        assert scale_groups is None
        assert penalty_type=='l2'
        assert family=='gaussian'

        self.dtype = dtype

        ### architecture
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.noise_sig2 = noise_sig2
        self.groups = groups # list of lists with grouping (e.g. [[1,2,3], [4,5]])
        self.scale_groups = scale_groups
        self.penalty_type = penalty_type
        self.family = family

        # hyperparameters
        self.lengthscale = tf.reshape(tf.cast(tf.convert_to_tensor(lengthscale), self.dtype), -1)
        self.prior_w2_sig2 = tf.reshape(tf.cast(tf.convert_to_tensor(prior_w2_sig2), self.dtype), -1)
        self.scale_global = tf.reshape(tf.cast(tf.convert_to_tensor(scale_global), self.dtype), -1)

        self.sample_features()
        self.act = lambda z: sqrt(2/self.dim_hidden)*tf.math.cos(z)

    def sample_features(self):
        # sample random weights for RFF features
        self.w1 = tf.cast(tf.convert_to_tensor(np.random.normal(0,1,(self.dim_hidden, self.dim_in))), dtype=tf.float32)
        self.b1 = tf.cast(tf.convert_to_tensor(np.random.uniform(0,2*pi,(self.dim_hidden,))), dtype=tf.float32)

    def compute_xw1(self, x, lengthscale=None):
        if not tf.is_tensor(x):
            x = tf.convert_to_tensor(x, dtype=tf.float32) # (n, dim_in)
        if lengthscale is None:
            lengthscale = self.lengthscale # (dim_in, )
        return (x / tf.expand_dims(lengthscale, 0)) @ tf.transpose(self.w1) # (n, dim_hidden)

    def hidden_features(self, x=None, xw1=None, lengthscale=None):
        if xw1 is None:
            xw1 = self.compute_xw1(x, lengthscale=lengthscale)
        if lengthscale is None:
            lengthscale = self.lengthscale

        return self.act(xw1 + tf.reshape(self.b1, (1,-1))) # (n, dim_hidden)

    def forward(self, w2, x=None, xw1=None, lengthscale=None, h=None):
        if h is None:
            if xw1 is None:
                xw1 = self.compute_xw1(x, lengthscale=lengthscale)
            if lengthscale is None:
                lengthscale = self.lengthscale
            h = self.hidden_features(x, xw1, lengthscale)
        
        return h@tf.reshape(w2,(-1,1))

    def jacobian_hidden_features(self, x=None, xw1=None, lengthscale=None):
        if xw1 is None:
            xw1 = self.compute_xw1(x, lengthscale=None)
        if lengthscale is None:
            lengthscale = self.lengthscale
        return -sqrt(2/self.dim_hidden) * tf.expand_dims(self.w1 / tf.expand_dims(lengthscale, 0), 0) * tf.expand_dims(tf.math.sin(xw1 + tf.reshape(self.b1, (1,-1))), -1) # analytical jacobian

    def grad_norm(self, x=None, xw1=None, lengthscale=None):
        J = self.jacobian_hidden_features(x=x, xw1=xw1, lengthscale=lengthscale)
        Ax_d = [1/J.shape[0]*tf.transpose(J[:,:,d])@J[:,:,d] for d in range(self.dim_in)]
        return Ax_d # list of length D, each element is K x K

    def make_log_marginal_likelihood(self):

        @tf.function
        def log_marginal_likelihood(x, y, lengthscale, prior_w2_sig2, scale_global):

            # precompute
            N = x.shape[0]
            K = self.dim_hidden
            yy = tf.transpose(y) @ y
            h = self.hidden_features(x=x, xw1=None, lengthscale=lengthscale)

            # gradients
            Ax_d = self.grad_norm(x, lengthscale=lengthscale) # list of length D, each element is K x K
            Ax_d = tf.stack(Ax_d) * tf.reshape(scale_global,(-1,1,1)) # (D, K, K)
            Ax = tf.reduce_sum(tf.stack(Ax_d),0) # sum over input dimension

            # inverse of prior covariance of w2
            if self.prior_w2_sig2.shape[0]==1:
                prior_cov_inv = tf.eye(self.dim_hidden)/prior_w2_sig2
            else:
                prior_cov_inv = tf.linalg.diag(1/prior_w2_sig2)
            prior_cov_inv += Ax
            #prior_cov = tf.linalg.inv(prior_cov_inv) # not sure if there's a way around this...

            # cholesky stuff
            A = prior_cov_inv + tf.transpose(h)@h/self.noise_sig2
            L = tf.linalg.cholesky(A)
            alpha = tf.linalg.solve(L, tf.transpose(h)@y)
            R = tf.linalg.cholesky(prior_cov_inv)

            # log marginal likelihood
            log_prob = \
                -N/2*tf.math.log(2*np.pi*self.noise_sig2) \
                -1/(2*self.noise_sig2)*yy \
                +tf.reduce_sum(tf.math.log(tf.linalg.diag_part(R))) \
                -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L))) \
                +1/(2*self.noise_sig2**2) * tf.transpose(alpha)@alpha
            return log_prob

        return log_marginal_likelihood

    def train_log_marginal_likelihood(self, x, y, n_epochs, learning_rate=0.001, clipvalue=100, batch_size=None, opt_lengthscale=True, opt_prior_w2_sig2=True, opt_scale_global=True):
        
        x = tf.cast(tf.convert_to_tensor(x), tf.float32)
        y = tf.cast(tf.convert_to_tensor(y), tf.float32)
        
        # initialize
        lengthscale_raw = tf.Variable(tfp.math.softplus_inverse(tf.cast(tf.convert_to_tensor(self.lengthscale), tf.float32)), name='lengthscale') # note: _raw is untransformed by softplus
        prior_w2_sig2_raw = tf.Variable(tfp.math.softplus_inverse(tf.cast(tf.convert_to_tensor(self.prior_w2_sig2), tf.float32)), name='prior_w2_sig2')
        scale_global_raw = tf.Variable(tfp.math.softplus_inverse(tf.cast(tf.convert_to_tensor(self.scale_global), tf.float32)), name='scale_global')
        
        # initialize 
        hyperparam_hist = {}
        hyperparam_hist['lengthscale'] = [tf.math.softplus(lengthscale_raw).numpy()]
        hyperparam_hist['prior_w2_sig2'] = [tf.math.softplus(prior_w2_sig2_raw).numpy()]
        hyperparam_hist['scale_global'] = [tf.math.softplus(scale_global_raw).numpy()]
        hyperparam_hist['loss'] = [None]

        # batch dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        if batch_size is None:
            batch_size = x.shape[0]# full batch
        train_dataset = train_dataset.batch(batch_size)

        # set up optimization
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        log_marginal_likelihood = self.make_log_marginal_likelihood()
        log_marginal_likelihood_raw = lambda x, y, lengthscale_raw, prior_w2_sig2_raw, scale_global_raw: log_marginal_likelihood(x, y, tf.math.softplus(lengthscale_raw), tf.math.softplus(prior_w2_sig2_raw), tf.math.softplus(scale_global_raw)) # takes raw values

        # decide which hyperparameters to optimize
        var_list = []
        lr_mult = []
        if opt_lengthscale:
            var_list += [lengthscale_raw]
            lr_mult += [1.0]
        if opt_prior_w2_sig2:
            var_list += [prior_w2_sig2_raw]
            lr_mult += [1.0]
        if opt_scale_global:
            var_list += [scale_global_raw]
            lr_mult += [100.0]

        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                loss = -log_marginal_likelihood_raw(x, y, lengthscale_raw, prior_w2_sig2_raw, scale_global_raw)
            grads = tape.gradient(loss, var_list)
            grads = [g*m for g, m in zip(grads,lr_mult)]
            #grads = [tf.clip_by_norm(g, clipvalue) for g in grads] # gradient clipping
            opt.apply_gradients(zip(grads, var_list))
            return loss

        # optimize
        for epoch in range(n_epochs):
            for xbatch, ybatch in train_dataset:
                loss = train_step(xbatch, ybatch)
  
            hyperparam_hist['lengthscale'].append(tf.math.softplus(lengthscale_raw).numpy())
            hyperparam_hist['prior_w2_sig2'].append(tf.math.softplus(prior_w2_sig2_raw).numpy())
            hyperparam_hist['scale_global'].append(tf.math.softplus(scale_global_raw).numpy())
            hyperparam_hist['loss'].append(loss.numpy().item())
            
        # unpack
        self.lengthscale = tf.math.softplus(lengthscale_raw)
        self.prior_w2_sig2 = tf.math.softplus(prior_w2_sig2_raw)
        self.scale_global = tf.math.softplus(scale_global_raw)
        
        return hyperparam_hist


    def fit(self, x, y):
        '''
        Computes conjugate posterior
        '''
        
        x = tf.cast(tf.convert_to_tensor(x), tf.float32)
        y = tf.cast(tf.convert_to_tensor(y), tf.float32)

        h = self.hidden_features(x, lengthscale=self.lengthscale)
        Ax_d = self.grad_norm(x, lengthscale=self.lengthscale) # (N, K, D)
        #Ax_d = Ax_d * tf.expand_dims(self.scale_global,(0,1)) # multiply by scale (N, K, D)
        Ax_d = [s*A for s,A in zip(self.scale_global, Ax_d)] # multiply by scale (N, K, D)
        Ax = tf.reduce_sum(tf.stack(Ax_d),0) # sum over input dimension

        if self.prior_w2_sig2.shape[0]==1:
            prior_sig2inv_mat = tf.eye(self.dim_hidden)/self.prior_w2_sig2
        else:
            prior_sig2inv_mat = tf.linalg.diag(self.prior_w2_sig2)
        prior_sig2inv_mat += Ax # prior includes gradient penalty


        sig2 = tf.linalg.inv(prior_sig2inv_mat + tf.transpose(h)@(h)/self.noise_sig2) # Should replace with cholesky
        mu = sig2 @ tf.transpose(h)@y/self.noise_sig2

        return mu, sig2

        






class RffHs(nn.Module):
    """
    RFF model with horseshoe

    Currently only single layer supported
    """
    def __init__(self, 
        dim_in, \
        dim_out, \
        dim_hidden=50, \
        infer_noise=False, sig2_inv=None, sig2_inv_alpha_prior=None, sig2_inv_beta_prior=None, \
        linear_term=False, linear_dim_in=None,
        layer_in_name='RffVarSelectLogitNormalLayer',
        **kwargs):
        super(RffHs, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.infer_noise=infer_noise
        self.linear_term=linear_term
        self.linear_dim_in=linear_dim_in

        # noise
        if self.infer_noise:
            self.sig2_inv_alpha_prior=torch.tensor(sig2_inv_alpha_prior)
            self.sig2_inv_beta_prior=torch.tensor(sig2_inv_beta_prior)
            self.sig2_inv = None

            self.register_buffer('sig2_inv_alpha', torch.empty(1, requires_grad=False))  # For now each output gets same noise
            self.register_buffer('sig2_inv_beta', torch.empty(1, requires_grad=False)) 
        else:
            self.sig2_inv_alpha_prior=None
            self.sig2_inv_beta_prior=None

            self.register_buffer('sig2_inv', torch.tensor(sig2_inv).clone().detach())

        # layers
        #self.layer_in = layers.RffHsLayer2(self.dim_in, self.dim_hidden, **kwargs)
        #self.layer_in = layers.RffLogitNormalLayer(self.dim_in, self.dim_hidden, **kwargs)
        self.layer_in = layers.get_layer(layer_in_name)(self.dim_in, self.dim_hidden, **kwargs)
        self.layer_out = layers.LinearLayer(self.dim_hidden, sig2_y=1/sig2_inv, **kwargs)

    def forward(self, x, x_linear=None, weights_type_layer_in='sample_post', weights_type_layer_out='sample_post', n_samp_layer_in=None):
        '''
        n_samp is number of samples from variational distribution (first layer)
        '''

        # network
        h = self.layer_in(x, weights_type=weights_type_layer_in, n_samp=n_samp_layer_in)
        y = self.layer_out(h, weights_type=weights_type_layer_out)

        # add linear term if specified
        if self.linear_term and x_linear is not None:
            return y + self.blm(x_linear, sample=sample)
        else:
            return y

    def sample_posterior_predictive(self, x_test, x_train, y_train):
        '''
        Need training data in order to get sample from non-variational full conditional distribution (output layer)

        Code duplicates some of forward, not ideal
        '''

        # 1: sample from variational distribution
        self.layer_in.sample_variational(store=True)

        # 2: forward pass of training data with sample from 1
        h = self.layer_in(x_train, weights_type='stored')

        # 3: sample output weights from conjugate (depends on ouput from 2)
        self.layer_out.fixed_point_updates(h, y_train) # conjugate update of output weights 
        self.layer_out.sample_weights(store=True)

        # 4: forward pass of test data using samples from 1 and 3
        return self.forward(x_test, weights_type_layer_in='stored', weights_type_layer_out='stored')

    def kl_divergence(self):
        return self.layer_in.kl_divergence()

    def log_prob(self, y_observed, y_pred):
        '''
        y_observed: (n_obs, dim_out)
        y_pred: (n_obs, n_pred, dim_out)

        averages over n_pred (e.g. could represent different samples), sums over n_obs
        '''
        lik = Normal(y_pred, torch.sqrt(1/self.sig2_inv))
        return lik.log_prob(y_observed.unsqueeze(1)).mean(1).sum(0)

    def loss_original(self, x, y, x_linear=None, temperature=1, n_samp=1):
        '''negative elbo'''
        y_pred = self.forward(x, x_linear, weights_type_layer_in='sample_post', weights_type_layer_out='stored', n_samp_layer_in=n_samp)

        kl_divergence = self.kl_divergence()
        #kl_divergence = 0

        log_prob = self.log_prob(y, y_pred)
        #log_prob = 0
        return -log_prob + temperature*kl_divergence

    def loss(self, x, y, x_linear=None, temperature=1, n_samp=1):
        '''
        Uses sample of weights from full conditional *based on samples of s* to compute likelihood
        '''

        kl_divergence = self.kl_divergence()
        #breakpoint()

        # 1: sample from variational distribution
        self.layer_in.sample_variational(store=True)

        # 2: forward pass of training data with sample from 1
        h = self.layer_in(x, weights_type='stored')

        # 3: sample output weights from conjugate (depends on ouput from 2)
        self.layer_out.fixed_point_updates(h, y) # conjugate update of output weights 
        self.layer_out.sample_weights(store=True)

        # 4: forward pass of test data using samples from 1 and 3
        y_pred = self.forward(x, weights_type_layer_in='stored', weights_type_layer_out='stored', n_samp_layer_in=1)

        log_prob = self.log_prob(y, y_pred)

        return -log_prob + temperature*kl_divergence


    def fixed_point_updates(self, x, y, x_linear=None, temperature=1): 
        self.layer_in.fixed_point_updates() # update horseshoe aux variables

        #### COMMENTING OUT OUTPUT LAYER UPDATES SINCE NOW PART OF LOSS FUNCTION ####
        """
        h = self.layer_in(x, weights_type='sample_post') # hidden units based on sample from variational dist
        self.layer_out.fixed_point_updates(h, y) # conjugate update of output weights 
        self.layer_out.sample_weights(store=True) # sample output weights from full conditional
        """
        ####

        if self.linear_term:
            if self.infer_noise:
                self.blm.sig2_inv = self.sig2_inv_alpha/self.sig2_inv_beta # Shouldnt this be a samplle?
            
            self.blm.fixed_point_updates(y - self.forward(x, x_linear=None, sample=True)) # Subtract off just the bnn

        if self.infer_noise and temperature > 0: 
            
            sample_y_bnn = self.forward(x, x_linear=None, sample=True) # Sample
            if self.linear_term:
                E_y_linear = F.linear(x_linear, self.blm.beta_mu)
                SSR = torch.sum((y-sample_y_bnn-E_y_linear)**2) + torch.sum(self.blm.xx_inv * self.blm.beta_sig2).sum()
            else:
                SSR = torch.sum((y - sample_y_bnn)**2)

            self.sig2_inv_alpha = self.sig2_inv_alpha_prior + temperature*0.5*x.shape[0] # Can be precomputed
            self.sig2_inv_beta = self.sig2_inv_beta_prior + temperature*0.5*SSR

    def init_parameters(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        self.layer_in.init_parameters()
        self.layer_out.init_parameters()

        if self.infer_noise:
            self.sig2_inv_alpha = self.sig2_inv_alpha_prior
            self.sig2_inv_beta = self.sig2_inv_beta_prior

        if self.linear_term:
            self.blm.init_parameters()

    def reinit_parameters(self, x, y, n_reinit=1):
        seeds = torch.zeros(n_reinit).long().random_(0, 1000)
        losses = torch.zeros(n_reinit)
        for i in range(n_reinit):
            self.init_parameters(seeds[i])
            losses[i] = self.loss(x, y)

        self.init_parameters(seeds[torch.argmin(losses).item()])

    def precompute(self, x=None, x_linear=None):
        # Needs to be run before training
        if self.linear_term:
            self.blm.precompute(x_linear)

    def get_n_parameters(self):
        n_param=0
        for p in self.parameters():
            n_param+=np.prod(p.shape)
        return n_param

    def print_state(self, x, y, epoch=0, n_epochs=0):
        '''
        prints things like training loss, test loss, etc
        '''
        print('Epoch[{}/{}], kl: {:.6f}, likelihood: {:.6f}, elbo: {:.6f}'\
                        .format(epoch, n_epochs, self.kl_divergence().item(), -self.loss(x,y,temperature=0).item(), -self.loss(x,y).item()))


class RffBeta(nn.Module):
    """
    RFF model beta prior on indicators

    Currently only single layer supported
    """
    def __init__(self, 
        dim_in, \
        dim_out, \
        dim_hidden=50, \
        infer_noise=False, sig2_inv=None, sig2_inv_alpha_prior=None, sig2_inv_beta_prior=None, \
        linear_term=False, linear_dim_in=None,
        **kwargs):
        super(RffBeta, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.infer_noise=infer_noise
        self.linear_term=linear_term
        self.linear_dim_in=linear_dim_in

        # noise
        if self.infer_noise:
            self.sig2_inv_alpha_prior=torch.tensor(sig2_inv_alpha_prior)
            self.sig2_inv_beta_prior=torch.tensor(sig2_inv_beta_prior)
            self.sig2_inv = None

            self.register_buffer('sig2_inv_alpha', torch.empty(1, requires_grad=False))  # For now each output gets same noise
            self.register_buffer('sig2_inv_beta', torch.empty(1, requires_grad=False)) 
        else:
            self.sig2_inv_alpha_prior=None
            self.sig2_inv_beta_prior=None

            self.register_buffer('sig2_inv', torch.tensor(sig2_inv).clone().detach())

        # layers
        self.layer_in = layers.RffBetaLayer(self.dim_in, self.dim_hidden, **kwargs)
        self.layer_out = layers.LinearLayer(self.dim_hidden, sig2_y=1/sig2_inv, **kwargs)

    def forward(self, x, x_linear=None, weights_type_layer_in='sample_post', weights_type_layer_out='sample_post'):

        # network
        h = self.layer_in(x, weights_type=weights_type_layer_in)
        y = self.layer_out(h, weights_type=weights_type_layer_out)

        # add linear term if specified
        if self.linear_term and x_linear is not None:
            return y + self.blm(x_linear, sample=sample)
        else:
            return y

    def kl_divergence(self):
        return self.layer_in.kl_divergence()

    def compute_loss_gradients(self, x, y, x_linear=None, temperature=1.):

        # sample from variational dist
        self.layer_in.sample_variational(store=True)

        # compute log likelihood
        y_pred = self.forward(x, x_linear, weights_type_layer_in='stored', weights_type_layer_out='stored')
        log_lik = -self.neg_log_prob(y, y_pred)

        # gradients of score function
        for p in self.layer_in.parameters(): 
            if p.grad is not None:
                p.grad.zero_()

        log_q = self.layer_in.log_prob_variational()
        log_q.backward()

        self.layer_in.s_a_trans_grad_q = self.layer_in.s_a_trans.grad.clone()
        self.layer_in.s_b_trans_grad_q = self.layer_in.s_b_trans.grad.clone()

        # gradients of kl
        for p in self.layer_in.parameters(): p.grad.zero_()

        kl = self.kl_divergence()
        kl.backward()

        self.layer_in.s_a_trans_grad_kl = self.layer_in.s_a_trans.grad.clone()
        self.layer_in.s_b_trans_grad_kl = self.layer_in.s_b_trans.grad.clone()

        # gradients of loss=-elbo
        with torch.no_grad():
            self.layer_in.s_a_trans.grad = -log_lik*self.layer_in.s_a_trans_grad_q + temperature*self.layer_in.s_a_trans_grad_kl
            self.layer_in.s_b_trans.grad = -log_lik*self.layer_in.s_b_trans_grad_q + temperature*self.layer_in.s_b_trans_grad_kl

    def loss(self, x, y, x_linear=None, temperature=1):
        '''negative elbo
        NON DIFFERENTIABLE BECAUSE OF SCORE METHOD
        '''
        y_pred = self.forward(x, x_linear, weights_type_layer_in='sample_post', weights_type_layer_out='stored')

        kl_divergence = self.kl_divergence()
        #kl_divergence = 0

        neg_log_prob = self.neg_log_prob(y, y_pred)
        #neg_log_prob = 0

        return neg_log_prob + temperature*kl_divergence

    def neg_log_prob(self, y_observed, y_pred):
        N = y_observed.shape[0]
        if self.infer_noise:
            sig2_inv = self.sig2_inv_alpha/self.sig2_inv_beta # Is this right? i.e. IG vs G
        else:
            sig2_inv = self.sig2_inv
        log_prob = -0.5 * N * math.log(2 * math.pi) + 0.5 * N * torch.log(sig2_inv) - 0.5 * torch.sum((y_observed - y_pred)**2) * sig2_inv
        return -log_prob

    def fixed_point_updates(self, x, y, x_linear=None, temperature=1): 

        h = self.layer_in(x, weights_type='sample_post') # hidden units based on sample from variational dist
        self.layer_out.fixed_point_updates(h, y) # conjugate update of output weights 

        self.layer_out.sample_weights(store=True) # sample output weights from full conditional

        if self.linear_term:
            if self.infer_noise:
                self.blm.sig2_inv = self.sig2_inv_alpha/self.sig2_inv_beta # Shouldnt this be a samplle?
            
            self.blm.fixed_point_updates(y - self.forward(x, x_linear=None, sample=True)) # Subtract off just the bnn

        if self.infer_noise and temperature > 0: 
            
            sample_y_bnn = self.forward(x, x_linear=None, sample=True) # Sample
            if self.linear_term:
                E_y_linear = F.linear(x_linear, self.blm.beta_mu)
                SSR = torch.sum((y-sample_y_bnn-E_y_linear)**2) + torch.sum(self.blm.xx_inv * self.blm.beta_sig2).sum()
            else:
                SSR = torch.sum((y - sample_y_bnn)**2)

            self.sig2_inv_alpha = self.sig2_inv_alpha_prior + temperature*0.5*x.shape[0] # Can be precomputed
            self.sig2_inv_beta = self.sig2_inv_beta_prior + temperature*0.5*SSR

    def init_parameters(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        self.layer_in.init_parameters()
        self.layer_out.init_parameters()

        if self.infer_noise:
            self.sig2_inv_alpha = self.sig2_inv_alpha_prior
            self.sig2_inv_beta = self.sig2_inv_beta_prior

        if self.linear_term:
            self.blm.init_parameters()

    def reinit_parameters(self, x, y, n_reinit=1):
        seeds = torch.zeros(n_reinit).long().random_(0, 1000)
        losses = torch.zeros(n_reinit)
        for i in range(n_reinit):
            self.init_parameters(seeds[i])
            losses[i] = self.loss(x, y)

        self.init_parameters(seeds[torch.argmin(losses).item()])

    def precompute(self, x=None, x_linear=None):
        # Needs to be run before training
        if self.linear_term:
            self.blm.precompute(x_linear)

    def get_n_parameters(self):
        n_param=0
        for p in self.parameters():
            n_param+=np.prod(p.shape)
        return n_param

    def print_state(self, x, y, epoch=0, n_epochs=0):
        '''
        prints things like training loss, test loss, etc
        '''
        print('Epoch[{}/{}], kl: {:.6f}, likelihood: {:.6f}, elbo: {:.6f}'\
                        .format(epoch, n_epochs, self.kl_divergence().item(), -self.loss(x,y,temperature=0).item(), -self.loss(x,y).item()))




def train(model, optimizer, x, y, n_epochs, x_linear=None, n_warmup = 0, n_rep_opt=10, print_freq=None, frac_start_save=1, frac_lookback=0.5, path_checkpoint='./'):
    '''
    frac_lookback will only result in reloading early stopped model if frac_lookback < 1 - frac_start_save
    '''

    loss = torch.zeros(n_epochs)
    loss_best = torch.tensor(float('inf'))
    loss_best_saved = torch.tensor(float('inf'))
    saved_model = False
    model.precompute(x, x_linear)

    for epoch in range(n_epochs):

        # TEMPERATURE HARDECODED, NEED TO FIX
        #temperature_kl = 0. if epoch < n_epochs/2 else 1.0
        #temperature_kl = epoch / (n_epochs/2) if epoch < n_epochs/2 else 1.0
        temperature_kl = epoch / (n_epochs/10) if epoch < n_epochs/10 else 1.0
        #temperature_kl = 0. # SET TO ZERO TO IGNORE KL

        for i in range(n_rep_opt):

            l = model.loss(x, y, x_linear=x_linear, temperature=temperature_kl)

            # backward
            optimizer.zero_grad()
            l.backward(retain_graph=True)
            optimizer.step()

            ##
            #print('------------- %d -------------' % epoch)
            #print('s     :', model.layer_in.s_loc.data)
            #print('grad  :', model.layer_in.s_loc.grad)

            #model.layer_in.s_loc.grad.zero_()
            #kl = model.layer_in.kl_divergence()
            #kl.backward()
            #if epoch > 500:
            #    breakpoint()
            #print('grad kl:', model.layer_in.s_loc.grad)
            ##

        loss[epoch] = l.item()

        with torch.no_grad():
            model.fixed_point_updates(x, y, x_linear=x_linear, temperature=1)

        # print state
        if print_freq is not None:
            if (epoch + 1) % print_freq == 0:
                model.print_state(x, y, epoch+1, n_epochs)

        # see if improvement made (only used if KL isn't tempered)
        if loss[epoch] < loss_best and temperature_kl==1.0:
            loss_best = loss[epoch]

        # save model
        if epoch > frac_start_save*n_epochs and loss[epoch] < loss_best_saved:
            print('saving mode at epoch = %d' % epoch)
            saved_model = True
            loss_best_saved = loss[epoch]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss[epoch],
            },  os.path.join(path_checkpoint, 'checkpoint.tar'))

        # end training if no improvement made in a while and more than half way done
        epoch_lookback = np.maximum(1, int(epoch - .25*n_epochs)) # lookback is 25% of samples by default
        if epoch_lookback > frac_start_save*n_epochs+1:
            loss_best_lookback = torch.min(loss[epoch_lookback:epoch+1])
            percent_improvement = (loss_best - loss_best_lookback)/torch.abs(loss_best) # positive is better
            if percent_improvement < 0.0:
                print('stopping early at epoch = %d' % epoch)
                break

    # reload best model if saving
    if saved_model:
        checkpoint = torch.load(os.path.join(path_checkpoint, 'checkpoint.tar'))
        model.load_state_dict(checkpoint['model_state_dict'])
        print('reloading best model from epoch = %d' % checkpoint['epoch'])
        model.eval()


    return loss[:epoch]


def train_score(model, optimizer, x, y, n_epochs, x_linear=None, n_warmup = 0, n_rep_opt=10, print_freq=None, frac_start_save=1):
    loss = torch.zeros(n_epochs)
    loss_best = 1e9 # Need better way of initializing to make sure it's big enough
    model.precompute(x, x_linear)

    for epoch in range(n_epochs):

        # TEMPERATURE HARDECODED, NEED TO FIX
        #temperature_kl = 0. if epoch < n_epochs/2 else 1
        #temperature_kl = epoch / (n_epochs/2) if epoch < n_epochs/2 else 1
        temperature_kl = 0. # SET TO ZERO TO IGNORE KL

        for i in range(n_rep_opt):

            optimizer.zero_grad()

            model.compute_loss_gradients(x, y, x_linear=x_linear, temperature=temperature_kl)

            # backward
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            optimizer.step()

        with torch.no_grad():
            model.fixed_point_updates(x, y, x_linear=x_linear, temperature=1)

        if epoch > frac_start_save*n_epochs and loss[epoch] < loss_best: 
            print('saving...')
            loss_best = loss[epoch]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss[epoch],
            }, 'checkpoint.tar')

        if print_freq is not None:
            if (epoch + 1) % print_freq == 0:
                model.print_state(x, y, epoch+1, n_epochs)

    return loss

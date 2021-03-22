# standard library imports
import os
import sys
from math import pi, log, sqrt

# package imports
import GPy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
#import tensorflow as tf
#from tensorflow import keras
#import rpy2.robjects as robjects
#from rpy2.robjects.packages import importr

# local imports
import util
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # add directory with bnn package to search path
import bnn.networks as networks

torch.set_default_dtype(torch.float64)

class VarImportance(object):
    '''
    Parent class for variable importance models. All methods need to be overwritten.
    '''
    def __init__(self):
        pass
    def train(self):
        pass

    def estimate_psi(self, x, n_samp):
        '''
        Returns mean and variance of variable importance evaluated at inputs X
        '''
        mean = np.zeros(X.shape[0])
        var = np.zeros(X.shape[0])
        return mean, var

    def sample_f_post(self, x):
        '''
        Returns a single sample from the posterior predictive evaluated at inputs x
        '''


class GPyVarImportance(object):
    def __init__(self, X, Y, sig2, opt_sig2=True, opt_kernel_hyperparam=True, lengthscale=1.0, variance=1.0):
        super().__init__()

        self.dim_in = X.shape[1]
        self.kernel = GPy.kern.RBF(input_dim=self.dim_in, lengthscale=lengthscale, variance=variance)
        self.model = GPy.models.GPRegression(X,Y,self.kernel)
        self.model.Gaussian_noise.variance = sig2

        self.opt_sig2 = opt_sig2
        self.opt_kernel_hyperparam = opt_kernel_hyperparam
        
        if not opt_sig2:
            self.model.Gaussian_noise.fix()

        if not opt_kernel_hyperparam:
            self.model.kern.lengthscale.fix()
            self.model.kern.variance.fix()
        
    def train(self):
        self.model.optimize_restarts(num_restarts = 1, verbose=False)
    
    def estimate_psi(self, X, n_samp=1000):
        '''
        estimates mean and variance of variable importance psi
        X:  inputs to evaluate gradient
        n_samp:  number of MC samples
        '''

        grad_mu, grad_var = self.model.predict_jacobian(X, full_cov=True) # mean and variance of derivative, (N*, d)
        #psi = np.mean(grad_mu[:,:,0]**2, axis=0)
        psi_mean = np.zeros(self.dim_in)
        psi_var = np.zeros(self.dim_in)
        
        for l in range(self.dim_in):
            try:
                grad_samp = np.random.multivariate_normal(grad_mu[:,l,0], grad_var[:,:,l,l], size=n_samp) # (n_samp, N*)
            except:
                try:
                    print('error with multivariate normal, trying adding noise to diagonal')
                    grad_samp = np.random.multivariate_normal(grad_mu[:,l,0], grad_var[:,:,l,l]+1e-6*np.eye(X.shape[0]), size=n_samp) # (n_samp, N*)
                except:
                    print('error with multivariate normal, unable to sample')
                    grad_samp = np.zeros((n_samp, X.shape[0]))

            psi_samp = np.mean(grad_samp**2,1)
            psi_mean[l] = np.mean(psi_samp)
            psi_var[l] = np.var(psi_samp)
            
        return psi_mean, psi_var

    def sample_f_post(self, x):
        # inputs and outputs are numpy arrays
        return self.model.posterior_samples_f(x, size=1)

    def sample_f_prior(self, x):
        mu = np.zeros(x.shape[0])
        C = self.kernel.K(x,x)
        return np.random.multivariate_normal(mu,C,1)

class RffVarImportance(object):
    def __init__(self, X):
        super().__init__()
        self.dim_in = X.shape[1]


    def train(self, X, Y, sig2, rff_dim=1200, batch_size=16, epochs=16):

        model_graph = tf.Graph()
        model_sess = tf.Session(graph=model_graph)

        with model_graph.as_default():
            X_tr = tf.placeholder(dtype=tf.float64, shape=[None, self.dim_in])
            Y_true = tf.placeholder(dtype=tf.float64, shape=[None, 1])
            H_inv = tf.placeholder(dtype=tf.float64, shape=[rff_dim, rff_dim])
            Phi_y = tf.placeholder(dtype=tf.float64, shape=[rff_dim, 1])

            rff_layer = kernel_layers.RandomFourierFeatures(output_dim=rff_dim,
                                                            kernel_initializer='gaussian',
                                                            trainable=True)

            ## define model
            rff_output = tf.cast(rff_layer(X_tr) * np.sqrt(2. / rff_dim), dtype=tf.float64)

            weight_cov = util.minibatch_woodbury_update(rff_output, H_inv)

            covl_xy = util.minibatch_interaction_update(Phi_y, rff_output, Y_true)

            random_feature_weight = rff_layer.kernel

            random_feature_bias = rff_layer.bias

        ### Training and Evaluation ###
        X_batches = util.split_into_batches(X, batch_size) * epochs
        Y_batches = util.split_into_batches(Y, batch_size) * epochs

        num_steps = X_batches.__len__()
        num_batch = int(num_steps / epochs)

        with model_sess as sess:
            sess.run(tf.global_variables_initializer())

            rff_1 = sess.run(rff_output, feed_dict={X_tr: X_batches[0]})
            weight_cov_val = util.compute_inverse(rff_1, sig_sq=sig2**2)
            covl_xy_val = np.matmul(rff_1.T, Y_batches[0])

            rff_weight, rff_bias = sess.run([random_feature_weight, random_feature_bias])

            for batch_id in range(1, num_batch):
                X_batch = X_batches[batch_id]
                Y_batch = Y_batches[batch_id]

                ## update posterior mean/covariance
                try:
                    weight_cov_val, covl_xy_val = sess.run([weight_cov, covl_xy],
                                                           feed_dict={X_tr: X_batch,
                                                                      Y_true: Y_batch,
                                                                      H_inv: weight_cov_val,
                                                                      Phi_y: covl_xy_val})
                except:
                    print("\n================================\n"
                          "Problem occurred at Step {}\n"
                          "================================".format(batch_id))

        self.beta = np.matmul(weight_cov_val, covl_xy_val)[:,0]

        self.Sigma_beta = weight_cov_val * sig2**2

        self.RFF_weight = rff_weight  # (d, D)

        self.RFF_bias = rff_bias  # (D, )



    def estimate_psi(self, X, n_samp=1000):
        '''
        estimates mean and variance of variable importance psi
        X:  inputs to evaluate gradient
        n_samp:  number of MC samples
        '''

        nD_mat = np.sin(np.matmul(X, self.RFF_weight) + self.RFF_bias)
        n, d = X.shape
        D = self.RFF_weight.shape[1]
        der_array = np.zeros((n, d, n_samp))
        beta_samp = np.random.multivariate_normal(self.beta, self.Sigma_beta, size=n_samp).T
        # (D, n_samp)
        for r in range(n):
            cur_mat = np.diag(nD_mat[r,:])
            cur_mat_W = np.matmul(self.RFF_weight, cur_mat)  # (d, D)
            cur_W_beta = np.matmul(cur_mat_W, beta_samp)  # (d, n_samp)
            der_array[r,:,:] = cur_W_beta

        der_array = der_array * np.sqrt(2. / D)
        psi_mean = np.zeros(self.dim_in)
        psi_var = np.zeros(self.dim_in)

        for l in range(self.dim_in):
            grad_samp = der_array[:,l,:].T  # (n_samp, n)
            psi_samp = np.mean(grad_samp ** 2, 1)
            psi_mean[l] = np.mean(psi_samp)
            psi_var[l] = np.var(psi_samp)

        return psi_mean, psi_var

    def sample_f_post(self, x):
        # inputs and outputs are numpy arrays
        D = self.RFF_weight.shape[1]
        phi = np.sqrt(2. / D) * np.cos(np.matmul(x, self.RFF_weight) + self.RFF_bias)
        beta_samp = np.random.multivariate_normal(self.beta, self.Sigma_beta, size=1).T
        return np.matmul(phi, beta_samp)


class RffVarImportancePytorch(object):
    def __init__(self, X, Y, noise_sig2, prior_w2_sig2, dim_hidden=50, lengthscale=1.0):
        super().__init__()

        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

        self.model = networks.standard.Rff(dim_in=X.shape[1], dim_out=Y.shape[1], dim_hidden=dim_hidden, noise_sig2=noise_sig2, prior_w2_sig2=prior_w2_sig2, lengthscale=lengthscale)

    def train(self):
        self.model.fixed_point_updates(self.X, self.Y)

    def estimate_psi(self, X=None, n_samp=1000):
        '''
        Use automatic gradients

        estimates mean and variance of variable importance psi
        X:  inputs to evaluate gradient
        n_samp:  number of MC samples
        '''

        X = torch.from_numpy(X)
        X.requires_grad = True

        psi_mean = np.zeros(self.model.dim_in)
        psi_var = np.zeros(self.model.dim_in)

        psi_samp = torch.zeros((n_samp, self.model.dim_in))
        for i in range(n_samp):

            f = self.model(X, weights_type='sample_post')
            torch.sum(f).backward()
            psi_samp[i,:] = torch.mean(X.grad**2,0)
            X.grad.zero_()

        psi_mean = torch.mean(psi_samp, 0)
        psi_var = torch.var(psi_samp, 0)

        return psi_mean.numpy(), psi_var.numpy()


    def sample_f_post(self, x):
        # inputs and outputs are numpy arrays
        return self.model(torch.from_numpy(x), weights_type='sample_post').detach().numpy()

    def sample_f_prior(self, x):
        # inputs and outputs are numpy arrays
        return self.model(torch.from_numpy(x), weights_type='sample_prior').detach().numpy()


class RffHsVarImportance(object):
    def __init__(self, 
        X, Y, \
        sig2_inv, \
        dim_in=1, dim_out=1, dim_hidden=50, \
        infer_noise=False, sig2_inv_alpha_prior=None, sig2_inv_beta_prior=None, \
        linear_term=False, linear_dim_in=None,\
        **model_kwargs):
        super().__init__()

        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

        self.model = RffHs(dim_in=X.shape[1], dim_out=Y.shape[1], dim_hidden=dim_hidden, \
            infer_noise=infer_noise, sig2_inv=sig2_inv, sig2_inv_alpha_prior=sig2_inv_alpha_prior, sig2_inv_beta_prior=sig2_inv_beta_prior, \
            linear_term=linear_term, linear_dim_in=linear_dim_in, **model_kwargs)

        
    def train(self, lr=.001, n_epochs=100, path_checkpoint='./'):
        # returns loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.reinit_parameters(self.X, self.Y, n_reinit=10) 

        ### TEMPORARY MANUALLY INITIALIZE
        #self.model.layer_in.s_loc.data[1] = torch.tensor(1.0)
        #self.model.layer_in.s_scale_untrans.data[1] = torch.tensor(0.)
        #print('MANUALLY INITIALIZED')
        #print('s_loc:', self.model.layer_in.s_loc.data)
        #print('s_scale:', self.model.layer_in.transform(self.model.layer_in.s_scale_untrans.data))
        ###

        return train_rffhs(self.model, optimizer, self.X, self.Y, n_epochs=n_epochs, n_rep_opt=10, print_freq=1, frac_start_save=.5, frac_lookback=.2, path_checkpoint=path_checkpoint)


    def estimate_psi(self, X=None, n_samp=200):
        '''
        Use automatic gradients

        estimates mean and variance of variable importance psi
        X:  inputs to evaluate gradient
        n_samp:  number of MC samples
        '''

        X = torch.from_numpy(X)
        X.requires_grad = True

        psi_mean = np.zeros(self.model.dim_in)
        psi_var = np.zeros(self.model.dim_in)

        psi_samp = torch.zeros((n_samp, self.model.dim_in))
        for i in range(n_samp):

            #f = self.model(X, weights_type_layer_in='sample_post', weights_type_layer_out='sample_post')
            f = self.model.sample_posterior_predictive(x_test=X, x_train=self.X, y_train=self.Y)
            
            torch.sum(f).backward()
            psi_samp[i,:] = torch.mean(X.grad**2,0)
            X.grad.zero_()

        psi_mean = torch.mean(psi_samp, 0)
        psi_var = torch.var(psi_samp, 0)

        return psi_mean.numpy(), psi_var.numpy()


    def estimate_psi2(self, X=None, n_samp=1000):
        '''
        estimates mean and variance of variable importance psi
        X:  inputs to evaluate gradient
        n_samp:  number of MC samples
        '''
        with torch.no_grad():

            #breakpoint()

            dist_nu = torch.distributions.log_normal.LogNormal(loc=self.model.layer_in.lognu_mu, 
                                                               scale=self.model.layer_in.lognu_logsig2.exp().sqrt())
            
            dist_eta = torch.distributions.log_normal.LogNormal(loc=self.model.layer_in.logeta_mu, 
                                                                scale=self.model.layer_in.logeta_logsig2.exp().sqrt())
            
            dist_beta = torch.distributions.multivariate_normal.MultivariateNormal(loc=self.model.layer_out.mu, 
                                                                                   covariance_matrix=self.model.layer_out.sig2)

            psi_mean = np.zeros(self.model.dim_in)
            psi_var = np.zeros(self.model.dim_in)
            for l in range(self.model.dim_in):

                # TO DO: replace loop for efficiency
                grad_samp = torch.zeros((n_samp, X.shape[0]))
                for i in range(n_samp):

                    samp_nu = dist_nu.sample()
                    samp_eta = dist_eta.sample()
                    samp_beta = dist_beta.sample()

                    nu_eta_w = samp_nu*samp_eta*self.model.layer_in.w

                    grad_samp[i,:] = (-sqrt(2/self.model.dim_out) \
                                     *torch.sin(F.linear(torch.from_numpy(X), nu_eta_w, self.model.layer_in.b)) \
                                     @ torch.diag(nu_eta_w[:,l]) \
                                     @ samp_beta.T).reshape(-1)

                psi_samp = torch.mean(grad_samp**2,1)
                psi_mean[l] = torch.mean(psi_samp)
                psi_var[l] = torch.var(psi_samp)

        return psi_mean.numpy(), psi_var.numpy()

    def dist_scales(self):
        '''
        returns mean and variance parameters of input-specific scale eta (not log eta)
        '''

        logeta_mu = self.model.layer_in.logeta_mu.detach()
        logeta_sig2 = self.model.layer_in.logeta_logsig2.exp().detach()

        eta_mu = torch.exp(logeta_mu + logeta_sig2/2)
        eta_sig2 = (torch.exp(logeta_sig2)-1)*torch.exp(2*logeta_mu+logeta_sig2)

        return eta_mu.numpy(), eta_sig2.numpy()

    #def sample_f_post(self, x):
    #    # inputs and outputs are numpy arrays
    #    return self.model(torch.from_numpy(x), weights_type_layer_in='sample_post', weights_type_layer_out='sample_post').detach().numpy()

    def sample_f_post(self, x_test):
        # inputs and outputs are numpy arrays
        with torch.no_grad():
            return self.model.sample_posterior_predictive(x_test=torch.from_numpy(x_test), x_train=self.X, y_train=self.Y).numpy().reshape(-1)



class BKMRVarImportance(object):
    def __init__(self, Z, Y2, sig2):
        super().__init__()

        self.bkmr = importr('bkmr') 
        self.base = importr('base') 
        self.sigsq_true = robjects.FloatVector([sig2])

        Zvec = robjects.FloatVector(Z.reshape(-1))
        self.Z = robjects.r.matrix(Zvec, nrow=Z.shape[0], ncol=Z.shape[1], byrow=True)

        Yvec = robjects.FloatVector(Y2.reshape(-1))
        self.Y2 = robjects.r.matrix(Yvec, nrow=Y2.shape[0], ncol=Y2.shape[1], byrow=True)

        self.have_stored_samples = False
        
    def train(self, n_samp=1000):

        self.n_samp = n_samp
        self.base.set_seed(robjects.FloatVector([1]))
        
        '''
        ### debugging
        print('id(Y3): ', id(Y3))
        robjects.Y3 = Y3
        print('id(robjects.Y3): ', id(robjects.Y3))
        
        import rpy2.robjects.vectors
        import rpy2.robjects.functions
        import rpy2.rinterface_lib.conversion
        import rpy2.rinterface_lib.callbacks
        import rpy2.rinterface
        rpy2.robjects.vectors.Y3 = Y3
        rpy2.robjects.functions.Y3 = Y3
        rpy2.rinterface_lib.conversion.Y3 = Y3
        rpy2.rinterface_lib.callbacks.Y3 = Y3
        rpy2.rinterface.Y3 = Y3

        breakpoint()
        ###
        '''

        self.fitkm = self.bkmr.kmbayes(y = self.Y2, Z = self.Z, iter = robjects.IntVector([n_samp]), verbose = robjects.vectors.BoolVector([False]), varsel = robjects.vectors.BoolVector([True]))

    def estimate_psi(self, X=None, n_samp=None):
        '''
        RETURNS POSTERIOR INCLUSION PROBABILITIES (PIPs) NOT VARIABLE IMPORTANCES
        '''
        out = self.bkmr.ExtractPIPs(self.fitkm)
        
        pip = np.ascontiguousarray(out.rx2('PIP'))
        pip_var = np.zeros_like(pip)
        print('pip:', pip)
        return pip, pip_var


    """
    def sample_f_post(self, x_test, use_saved_x_test=True):
        # inputs and outputs are numpy arrays
        sel = np.random.choice(range(int(self.n_samp/2), self.n_samp)) # randomly samples from second half of samples
        breakpoint()
        # this is kind of a hack to save time
        if (use_saved_x_test and not self.have_saved_x_test) or not use_saved_x_test:
            self.Znewvec = robjects.FloatVector(x_test.reshape(-1))
            self.Znew = robjects.r.matrix(self.Znewvec, nrow=x_test.shape[0], ncol=x_test.shape[1], byrow=True)
            self.have_saved_x_test = True

        return np.ascontiguousarray(self.base.t(self.bkmr.SamplePred(self.fitkm, Znew = self.Znew, Xnew = self.base.cbind(0), sel=sel.item()))) # (n, 1)
    """

    def sample_f_post(self, x_test):

        # first check if x_test has changed
        if self.have_stored_samples:
            if not np.array_equal(self.x_test, x_test):
                self.have_stored_samples = False
        '''
        if not self.have_stored_samples:
            self.have_stored_samples = True
            self.x_test = x_test
            
            self.Znewvec = robjects.FloatVector(x_test.reshape(-1))
            self.Znew = robjects.r.matrix(self.Znewvec, nrow=x_test.shape[0], ncol=x_test.shape[1], byrow=True)
            
            self.samples = np.ascontiguousarray(self.bkmr.SamplePred(self.fitkm, Znew = self.Znew, Xnew = self.base.cbind(0), sel=self.base.seq(int(self.n_samp/2), self.n_samp))) # (samps, inputs)
            
            self.sample_idx = np.random.permutation(np.arange(self.samples.shape[0])) # posterior samples for inference
            self.sample_iter = 0

        if self.sample_iter == self.samples.shape[0]:
            self.sample_iter = 0

        f = self.samples[self.sample_iter, :].reshape(-1,1)
        self.sample_iter += 1
        return f
        '''

        if not self.have_stored_samples:
            self.have_stored_samples = True
            self.x_test = x_test
            
            self.Znewvec = robjects.FloatVector(x_test.reshape(-1))
            self.Znew = robjects.r.matrix(self.Znewvec, nrow=x_test.shape[0], ncol=x_test.shape[1], byrow=True)
            
            self.samples = np.ascontiguousarray(self.bkmr.SamplePred(self.fitkm, Znew = self.Znew, Xnew = self.base.cbind(0), sel=self.base.seq(int(self.n_samp/2), self.n_samp))) # (samps, inputs)
            
        return self.samples[np.random.choice(self.samples.shape[0]), :].reshape(-1,1)

class BayesLinearLassoVarImportance(object):
    def __init__(self, X, Y, prior_w2_sig2=1.0, noise_sig2=1.0, scale_global=1.0, groups=None, scale_groups=None):
        super().__init__()

        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
        self.model = networks.sparse.BayesLinearLasso(dim_in=X.shape[1], dim_out=Y.shape[1], prior_w2_sig2=prior_w2_sig2, noise_sig2=noise_sig2, scale_global=scale_global, groups=groups, scale_groups=scale_groups)

    def train(self, num_results = int(10e3), num_burnin_steps = int(1e3)):
        '''
        Train with HMC
        '''
        self.samples, self.accept = self.model.train(self.X, self.Y, num_results = num_results, num_burnin_steps = num_burnin_steps)
        return self.samples, self.accept

    def estimate_psi(self, X=None, n_samp=1000):
        '''
        Uses closed form

        estimates mean and variance of variable importance psi
        X:  inputs to evaluate gradient
        n_samp:  number of MC samples
        '''
        return np.mean(self.samples**2,0), np.var(self.samples**2,0)

    def sample_f_post(self, x):
        i = np.random.choice(self.samples.shape[0])
        w = self.samples[i, :].reshape(-1,1)
        return (x@w.reshape(-1,1)).reshape(-1)


class RffGradPenVarImportance(object):
    def __init__(self, X, Y, dim_hidden=50, prior_w2_sig2=1.0, noise_sig2=1.0, scale_global=1.0, groups=None, scale_groups=None, lengthscale=1.0, penalty_type='l1'):
        super().__init__()

        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
        self.model = networks.sparse.RffGradPen(dim_in=X.shape[1], dim_hidden=dim_hidden, dim_out=Y.shape[1], prior_w2_sig2=prior_w2_sig2, noise_sig2=noise_sig2, scale_global=scale_global, groups=groups, scale_groups=scale_groups, lengthscale=lengthscale, penalty_type=penalty_type)

    def train(self, num_results = int(10e3), num_burnin_steps = int(1e3)):
        '''
        Train with HMC
        '''
        self.samples, self.accept = self.model.train(self.X, self.Y, num_results = num_results, num_burnin_steps = num_burnin_steps)
        return self.samples, self.accept

    def estimate_psi(self, X=None, n_samp=1000):
        '''
        Uses closed form

        estimates mean and variance of variable importance psi
        X:  inputs to evaluate gradient
        n_samp:  number of MC samples
        '''

        Ax_d, _ = self.model.compute_Ax(torch.from_numpy(X))
        Ax_d = [A.detach().numpy() for A in Ax_d]
        psi = np.zeros((self.samples.shape[0], X.shape[1]))
        for d in range(X.shape[1]):
            for s in range(self.samples.shape[0]):
                psi[s,d] = self.samples[s,:].reshape(1,-1)@Ax_d[d]@self.samples[s,:].reshape(-1,1) 
        return np.mean(psi,0), np.var(psi,0)

    def sample_f_post(self, x):
        x = torch.from_numpy(x)
        h = self.model.hidden_features(x)
        i = np.random.choice(self.samples.shape[0])
        w2 = self.samples[i, :].reshape(-1,1)
        return (h@w2.reshape(-1,1)).reshape(-1)


class RffGradPenVarImportanceHyper(object):
    def __init__(self, X, Y, dim_hidden=50, prior_w2_sig2=1.0, noise_sig2=1.0, scale_global=1.0, groups=None, scale_groups=None, lengthscale=1.0, penalty_type='l1'):
        super().__init__()

        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
        self.model = networks.sparse.RffGradPenHyper(dim_in=X.shape[1], dim_hidden=dim_hidden, dim_out=Y.shape[1], prior_w2_sig2=prior_w2_sig2, noise_sig2=noise_sig2, scale_global=scale_global, groups=groups, scale_groups=scale_groups, lengthscale=lengthscale, penalty_type=penalty_type)

    def train(self, num_results = int(10e3), num_burnin_steps = int(1e3)):
        '''
        Train with HMC
        '''
        self.samples, self.accept = self.model.train(self.X, self.Y, num_results = num_results, num_burnin_steps = num_burnin_steps)
        return self.samples, self.accept

    def estimate_psi(self, X=None, n_samp=1000):
        '''
        Uses closed form

        estimates mean and variance of variable importance psi
        X:  inputs to evaluate gradient
        n_samp:  number of MC samples
        '''

        psi = np.zeros((self.samples[0].shape[0], X.shape[1]))
        n = X.shape[0]

        w1 = self.model.w.unsqueeze(0).numpy()
        b1 = self.model.b.reshape(1,-1).numpy()
        X_w1 = X @ self.model.w.numpy().T

        for s in range(self.samples[0].shape[0]):
            w = self.samples[0][s,:] # output layer weights
            l = self.samples[1][s] # lengthscale

            J = -sqrt(2/self.model.dim_hidden) * w1 / l * np.expand_dims(np.sin(X_w1 / l + b1), -1) #analytical jacobian
            #J = -sqrt(2/self.model.dim_hidden) * self.model.w.unsqueeze(0) / l * tf.expand_dims(tf.math.sin(x_w_tf / l + tf.reshape(self.b_tf, (1,-1))), -1) # analytical jacobian
            #J = J.numpy()

            Ax_d = [1/n*J[:,:,d].T@J[:,:,d] for d in range(self.model.dim_in)]

            for d in range(X.shape[1]):
                psi[s,d] = w.reshape(1,-1)@Ax_d[d]@w.reshape(-1,1) 

        return np.mean(psi,0), np.var(psi,0)

    def sample_f_post(self, x):
        i = np.random.choice(self.samples[0].shape[0])

        x = torch.from_numpy(x)
        h = self.model.hidden_features(x, lengthscale = self.samples[1][i].item()).numpy()
        w2 = self.samples[0][i, :].reshape(-1,1)
        return (h@w2.reshape(-1,1)).reshape(-1)



class RffGradPenVarImportanceHyper_v2(object):
    def __init__(self, X, Y, dim_hidden=50, prior_w2_sig2=1.0, noise_sig2=1.0, scale_global=1.0, groups=None, scale_groups=None, lengthscale=1.0, penalty_type='l1'):
        super().__init__()

        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
        self.model = networks.sparse.RffGradPenHyper_v2(dim_in=X.shape[1], dim_hidden=dim_hidden, dim_out=Y.shape[1], prior_w2_sig2=prior_w2_sig2, noise_sig2=noise_sig2, scale_global=scale_global, groups=groups, scale_groups=scale_groups, lengthscale=lengthscale, penalty_type=penalty_type)

    
    def train_map(self):
        '''
        Train with HMC
        '''
        w2 = self.model.train_map(self.X, self.Y)
        return w2


    def train(self, num_results = int(10e3), num_burnin_steps = int(1e3), infer_hyper=False, optimize_hyper=False):
        '''
        Train with HMC
        '''
        self.samples, self.accept = self.model.train(self.X, self.Y, num_results = num_results, num_burnin_steps = num_burnin_steps, infer_hyper=infer_hyper, optimize_hyper=optimize_hyper)
        
        self.num_results = num_results
        self.infer_hyper=infer_hyper
        self.optimize_hyper=optimize_hyper
        return self.samples, self.accept

    def estimate_psi(self, X=None, n_samp=1000):
        '''
        Uses closed form

        estimates mean and variance of variable importance psi
        X:  inputs to evaluate gradient
        n_samp:  number of MC samples
        '''

        # allocate space
        psi = np.zeros((n_samp, X.shape[1]))
        
        # precompute
        xw1 = self.model.compute_xw1(X)

        if not self.infer_hyper:
            Ax_d = self.model.grad_norm(x=None, xw1=xw1) # only need to compute once
        
        for i, s in enumerate(np.random.choice(self.num_results, size=min(n_samp, self.num_results), replace=False)):
            if self.infer_hyper:
                w2 = self.samples[0][s,:] # output layer weights
                l = self.samples[1][s] # lengthscale
                Ax_d = self.model.grad_norm(x=None, lengthscale=l, xw1=xw1) # recompute because depends on lengthscale
            else:
                w2 = self.samples[s,:]

            for d in range(X.shape[1]):
                psi[i,d] = w2.reshape(1,-1)@Ax_d[d]@w2.reshape(-1,1) 

        return np.mean(psi,0), np.var(psi,0)

    def sample_f_post(self, x):
        s = np.random.choice(self.num_results) # should make sure same sample isn't selected more than once...
        xw1 = self.model.compute_xw1(x) # is there a way to not keep rerunning this?

        if self.infer_hyper:
            w2 = self.samples[0][s,:] # output layer weights
            l = self.samples[1][s] # lengthscale
        else:
            w2 = self.samples[s,:]
            l = self.model.lengthscale

        return self.model.forward(w2, xw1=xw1, lengthscale=l).numpy()








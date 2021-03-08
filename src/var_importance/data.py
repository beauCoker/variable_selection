# standard library imports
import os
import sys

# package imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import GPy
except:
    print("Unable to import GPy, but only needed for RBF toy")

try:
    import autograd.numpy as np
    from autograd import grad, jacobian
except:
    print("Unable to import autograd, so variable importance can't be calculated")
    import numpy as np

class Dataset:
    x_train = None # inputs
    z_train = None # covariates
    f_train = None # ground truth function
    y_train = None # observed outcomes
    psi_train = None # variable importance

    x_test = None
    z_test = None
    f_test = None
    y_test = None
    psi_test = None


class Toy(Dataset):
    def __init__(self, 
                 f,
                 x_train,
                 x_test=None,
                 noise_sig2=1.0,
                 seed=0,
                 standardize=True):

        self.f = lambda x: f(x).reshape(-1,1) # makes sure output is (n,1)
        self.noise_sig2 = noise_sig2
        self.dim_in = x_train.shape[1]

        # train
        self.x_train = x_train
        self.n_train = x_train.shape[0]
    
        # test
        self.x_test = x_test
        self.n_test = None if x_test is None else x_test.shape[0]

        self.evaluate_f()
        self.sample_y(seed)

        self.standardized = False
        if standardize:
            self.standardize()
        try:
            self.evaluate_psi() # note: psi always based on whether data originally standardized
        except:
            print('Unable to compute variable importance, possible autograd not imported')

    def evaluate_f(self):
        # train
        self.f_train = self.f(self.x_train).reshape(-1,1)

        # test
        if self.x_test is not None:
            self.f_test = self.f(self.x_test).reshape(-1,1)

    def sample_y(self, seed=0):
        r_noise = np.random.RandomState(seed)
        
        # train
        noise = r_noise.randn(self.n_train,1) * np.sqrt(self.noise_sig2)
        self.y_train = self.f_train + noise

        # test
        if self.x_test is not None:
            noise_test = r_noise.randn(self.n_test,1) * np.sqrt(self.noise_sig2)
            self.y_test = self.f_test + noise_test

    def evaluate_psi(self):
        grad_f = grad(self.f)

        jac_f_train = np.zeros((self.n_train, self.dim_in))
        for i in range(self.n_train):
            jac_f_train[i,:] = grad_f(self.x_train[i,:].reshape(1,-1))
        self.psi_train = np.mean(jac_f_train**2, 0)

        if self.x_test is not None:
            jac_f_test = np.zeros((self.n_test, self.dim_in))
            for i in range(self.n_test):
                jac_f_test[i,:] = grad_f(self.x_test[i,:].reshape(1,-1))
            self.psi_test = np.mean(jac_f_test**2, 0)

    def standardize(self):
        zscore = lambda x, mu, sigma: (x - mu.reshape(1,-1)) / sigma.reshape(1,-1)
        un_zscore = lambda x, mu, sigma: x * sigma.reshape(1,-1) + mu.reshape(1,-1)
            
        if not self.standardized:
            
            self.mu_x = np.mean(self.x_train, axis=0)
            self.sigma_x = np.std(self.x_train, axis=0)

            #self.mu_f = np.mean(self.f_train, axis=0)
            #self.sigma_f = np.std(self.f_train, axis=0)

            self.mu_y = np.mean(self.y_train, axis=0)
            self.sigma_y = np.std(self.y_train, axis=0)

            self.x_train = zscore(self.x_train, self.mu_x, self.sigma_x)
            if self.x_test is not None:
                self.x_test = zscore(self.x_test, self.mu_x, self.sigma_x)

            self.f_train = zscore(self.f_train, self.mu_y, self.sigma_y)
            if self.f_test is not None:
                self.f_test = zscore(self.f_test, self.mu_y, self.sigma_y)

            self.y_train = zscore(self.y_train, self.mu_y, self.sigma_y)
            if self.y_test is not None:
                self.y_test = zscore(self.y_test, self.mu_y, self.sigma_y)

            self.f_orig = self.f
            self.f = lambda x: zscore(self.f_orig(un_zscore(x, self.mu_x, self.sigma_x)), self.mu_y, self.sigma_y)
            self.standardized = True

    def unstandardize(self):
        zscore = lambda x, mu, sigma: (x - mu.reshape(1,-1)) / sigma.reshape(1,-1)
        un_zscore = lambda x, mu, sigma: x * sigma.reshape(1,-1) + mu.reshape(1,-1)    

        if self.standardized:
            
            self.x_train = un_zscore(self.x_train, self.mu_x, self.sigma_x)
            if self.x_test is not None:
                self.x_test = un_zscore(self.x_test, self.mu_x, self.sigma_x)

            self.f_train = un_zscore(self.f_train, self.mu_y, self.sigma_y)
            if self.f_test is not None:
                self.f_test = un_zscore(self.f_test, self.mu_y, self.sigma_y)

            self.y_train = un_zscore(self.y_train, self.mu_y, self.sigma_y)
            if self.y_test is not None:
                self.y_test = un_zscore(self.y_test, self.mu_y, self.sigma_y)

            self.f = self.f_orig
            self.standardized = False


'''
class SinToy(Toy):
    def __init__(self, dim_in, noise_sig2, n_train, n_test=100, seed_x=0, seed_noise=0, n_nonzero=1):
        assert dim_in>=n_nonzero

        # sample x
        r_x = np.random.RandomState(seed_x)
        x_train = r_x.uniform(-5,5,(n_train, dim_in))
        x_test = r_x.uniform(-5,5,(n_test, dim_in))

        # ground truth function
        f = lambda x: np.sum(np.sin(x[:,:n_nonzero]),-1).reshape(-1,1)

        super().__init__(f, x_train, x_test, noise_sig2, seed_noise, standardize=True)
'''

def sin_toy(dim_in, noise_sig2, n_train, n_test=100, seed_x=0, seed_noise=0, n_nonzero=1):
    assert dim_in>=n_nonzero

    # sample x
    r_x = np.random.RandomState(seed_x)
    x_train = r_x.uniform(-5,5,(n_train, dim_in))
    x_test = r_x.uniform(-5,5,(n_test, dim_in))

    # ground truth function
    f = lambda x: np.sum(np.sin(x[:,:n_nonzero]),-1).reshape(-1,1)

    return Toy(f, x_train, x_test, noise_sig2, seed_noise)

def rff_toy(dim_in, noise_sig2, n_train, n_test=100, dim_hidden=50, seed_x=0, seed_w=0, seed_noise=0, n_nonzero=1):

    # sample x
    r_x = np.random.RandomState(seed_x)
    x_train = r_x.randn(n_train, dim_in)
    x_test = r_x.randn(n_test, dim_in)

    # ground truth function
    r_w = np.random.RandomState(seed_w)
    w1 = r_w.randn(dim_hidden, n_nonzero)
    b1 = r_w.uniform(0, 2*np.pi, (dim_hidden,1))
    w2 = r_w.randn(1, dim_hidden)
    act = lambda z: np.sqrt(2/dim_hidden)*np.cos(z)
    f = lambda z: act(z[:, :n_nonzero]@w1.T + b1.T) @ w2.T

    return Toy(f, x_train, x_test, noise_sig2, seed_noise)


def mixselect_toy(dim_in, noise_sig2, n_train, n_test=100, seed_x=0, seed_noise=0, version=1):

    # sample x
    r_x = np.random.RandomState(seed_x)
    x_train = r_x.randn(n_train, dim_in)
    x_test = r_x.randn(n_test, dim_in)

    if version==1:
        assert dim_in >= 5
        f = lambda x: x[:,0] - x[:,1] + x[:,2] + 0.5*x[:,3]**2 + 4/(np.exp(-2*x[:,4]) + 1)

    elif version==2:
        assert dim_in >= 3
        f = lambda x: x[:,0] + x[:,1] - x[:,2]

    elif version==3:
        assert dim_in >= 3
        f = lambda x: np.sin(x[:,0] + 3*x[:,2]) - 0.5*x[:,2]**2 + np.exp(-0.1*x[:,0])

    return Toy(f, x_train, x_test, noise_sig2, seed_noise)

def load_dataset(name, dim_in, noise_sig2, n_train, n_test=100, signal_scale=1.0, n_nonzero=1, subtract_covariates=False):
    '''
    inputs:

    returns:
    '''

    if name == 'sin':
        dataset = sin_toy(dim_in, noise_sig2, n_train, n_test=n_test, seed_x=0, seed_noise=0, n_nonzero=n_nonzero)

    elif name == 'rff':
        dataset = rff_toy(dim_in, noise_sig2, n_train, n_test=n_test, seed_x=0, seed_noise=0, n_nonzero=n_nonzero)

    elif name == 'mixselect1':
        dataset = mixselect_toy(dim_in, noise_sig2, n_train, version=1)

    elif name == 'mixselect2':
        dataset = mixselect_toy(dim_in, noise_sig2, n_train, version=2)

    elif name == 'mixselect3':
        dataset = mixselect_toy(dim_in, noise_sig2, n_train, version=3)

    return dataset
    
if __name__ == "__main__":

    dir_out = './datasets'
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    names = ['sin', 'rff', 'mixselect1', 'mixselect2', 'mixselect3']
    dim_in = 3

    for name in names:

        try:
            data = load_dataset(name, dim_in=dim_in, noise_sig2=.01, n_train=100, n_test=100, n_nonzero=2)
        except:
            print('Unable to load dataset %s' % name)
            continue

        data.unstandardize()
        
        # variable importance
        if data.psi_train is not None:
            fig, ax = plt.subplots()
            variable = np.arange(dim_in).tolist()
            psi = data.psi_train.tolist()
            split = ['train']*dim_in

            if data.psi_test is not None:
                variable += np.arange(dim_in).tolist()
                psi += data.psi_test.tolist()
                split += ['test']*dim_in

            psi_df = pd.DataFrame({
                'variable': variable,
                'psi': psi,
                'split': split
                })
            fig, ax = plt.subplots()
            ax.set_title(r'variable importance $\psi$')
            sns.barplot(x="variable", y="psi", hue="split", data=psi_df, ax=ax)
            fig.savefig(os.path.join(dir_out, 'dataset=%s_var_import.png' % name))
            plt.close()

        # data
        fig, ax = plt.subplots(1,dim_in, figsize=(12,4))
        for i in range(dim_in):
            # train
            ax[i].scatter(data.x_train[:,i], data.y_train, label='train', color='blue')
            
            # test
            if data.x_test is not None and data.y_test is not None:
                ax[i].scatter(data.x_test[:,i], data.y_test, label='test', color='red')            

            if hasattr(data, 'f'):
                n_grid = 100
                #x_grid = np.ones((n_grid,dim_in)) * np.median(data.x_train,0).reshape(1,-1) # hold other variables at median
                x_grid = np.zeros((n_grid,dim_in)) # hold other variables at zero


                x_grid_i = np.linspace(
                    min(data.x_train[:,i].min(), data.x_test[:,i].min()),
                    max(data.x_train[:,i].max(), data.x_test[:,i].max()),
                    n_grid
                    )
                x_grid[:,i] = x_grid_i
                f_grid_i = data.f(x_grid)
                ax[i].plot(x_grid_i, f_grid_i, label='f', color='black')
                
        ax[0].legend()
    
        fig.savefig(os.path.join(dir_out, 'dataset=%s.png' % name))






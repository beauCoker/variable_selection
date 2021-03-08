# standard library imports
import os
import sys

# package imports
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import GPy
import seaborn as sns

## --------------- plotting --------------- 

def rmse(f, f_pred):
    # predictive root mean squared error (RMSE)
    return np.sqrt(np.mean((f - f_pred) ** 2))


def picp(f, f_pred_lb, f_pred_ub):
    # prediction interval coverage (PICP)
    return np.mean(np.logical_and(f >= f_pred_lb, f <= f_pred_ub))


def mpiw(f_pred_lb, f_pred_ub):
    # mean prediction interval width (MPIW)
    return np.mean(f_pred_ub - f_pred_lb)


def test_log_likelihood(mean, cov, test_y):
    return multivariate_normal.logpdf(test_y.reshape(-1), mean.reshape(-1), cov)


## --------------- RFF --------------- 

def minibatch_woodbury_update(X, H_inv):
    """Minibatch update of linear regression posterior covariance
    using Woodbury matrix identity.

    inv(H + X^T X) = H_inv - H_inv X^T inv(I + X H_inv X^T) X H_inv

    Args:
        X: (tf.Tensor) A M x K matrix of batched observation.
        H_inv: (tf.Tensor) A K x K matrix of posterior covariance of rff coefficients.


    Returns:
        H_new: (tf.Tensor) K x K covariance matrix after update.
    """
    batch_size = tf.shape(X)[0]

    M0 = tf.eye(batch_size, dtype=tf.float64) + tf.matmul(X, tf.matmul(H_inv, X, transpose_b=True))
    M = tf.matrix_inverse(M0)
    B = tf.matmul(X, H_inv)
    H_new = H_inv - tf.matmul(B, tf.matmul(M, B), transpose_a=True)

    return H_new


def minibatch_interaction_update(Phi_y, rff_output, Y_batch):
    return Phi_y + tf.matmul(rff_output, Y_batch, transpose_a=True)


def compute_inverse(X, sig_sq=1):
    return np.linalg.inv(np.matmul(X.T, X) + sig_sq * np.identity(X.shape[1]))


def split_into_batches(X, batch_size):
    return [X[i:i + batch_size] for i in range(0, len(X), batch_size)]

## --------------- plotting --------------- 

def plot_results_grid(data, x_vals, y_vals, x_lab, y_lab, fig=None, ax=None):
    if fig is None and ax is None:
        fig, ax = plt.subplots(1,data.shape[2], figsize=(16,4), sharex=True, sharey=True)
    
    vmax = np.nanmax(data)
    ax[0].set_ylabel(y_lab)
    for i in range(data.shape[2]):
        pcm=ax[i].imshow(data[:,:,i],vmin=0, vmax=vmax)
        ax[i].set_title('X_%d'%i)
        ax[i].set_xlabel(x_lab)

    plt.xticks(np.arange(len(x_vals)), labels=x_vals)

    plt.yticks(np.arange(len(y_vals)), labels=y_vals)

    fig.colorbar(pcm, ax=ax[:], shrink=0.6)

    return fig, ax 


def plot_results_dist(data, dim_in, n_obs_list, data_true=None, fig=None, ax=None):
    '''
    data:   (obs x rep x input) <-- data for models with same input dimension
    data_true:   (input)

    For each combination of n_obs and variable_idx, plots distribution of psi_mean over reps.

    '''

    if fig is None and ax is None:
        fig,ax = plt.subplots(len(n_obs_list),1,figsize=(dim_in*2,16),sharex=True,sharey=True)
    if len(n_obs_list)==1: ax = [ax]

    ax[-1].set_xlabel('variable')
    
    if data_true is not None:
        ax[0].legend([plt.Line2D([0], [0], color='red')],['"truth"'])

    n_rep = data.shape[1]
    for j, n_obs in enumerate(n_obs_list):
        if n_rep>1:
            sns.violinplot(data=data[j,:,:], ax=ax[j])
        else:
            ax[j].bar(np.arange(dim_in), data[j,0,:dim_in])
        for i in range(dim_in):
            ax[j].set_ylabel('%d obs.' % n_obs)
            if data_true is not None:
                ax[j].plot(np.array([i-.25,i+.25]),np.array([data_true[i],data_true[i]]),'red')
        
    ax[0].set_xlim(-.5,dim_in-.5)
    return fig, ax


def plot_slice(f_sampler, x, y, quantile=.5, dim=0, n_samp=500, f_true=None, ax=None):
    '''

    x: (N,D) training inputs
    y: (N,1) or (N,) training outputs
    quantile: Quantile of fixed x variables to use in plot
    dim: dimension of x to plot on x-axis

    Everything should be numpy
    '''

    if ax is None:
        fig, ax = plt.subplots()

    # x-axis
    midx = (x[:,dim].min() + x[:,dim].max())/2
    dx = x[:,dim].max() - x[:,dim].min()
    x_plot = np.linspace(midx - .75*dx, midx + .75*dx, 100)

    #x_plot_all = np.quantile(x, q=quantile, axis=0)*np.ones((x_plot.shape[0], x.shape[1])) # use quantile
    x_plot_all = np.zeros((x_plot.shape[0], x.shape[1])) # use zeros
    x_plot_all[:, dim] = x_plot

    # sample from model
    f_samp_plot = np.zeros((n_samp, x_plot.shape[0]))
    for i in range(n_samp):
        f_samp_plot[i,:] = f_sampler(x_plot_all).reshape(-1)

    # plot
    ax.scatter(x[:,dim], y) # training data
    ax.plot(x_plot, np.mean(f_samp_plot, 0), color='blue', label='post mean') # posterior mean
    for q in [.025, .05, .1]:
        ci = np.quantile(f_samp_plot, [q, 1-q], axis=0)
        ax.fill_between(x_plot_all[:,dim].reshape(-1), ci[0,:], ci[1,:], alpha=.1, color='blue')

    if f_true is not None:
        ax.plot(x_plot, f_true(x_plot_all), color='orange', label='truth') # posterior mean

    # plot a few samples
    n_samp_plot = max(3, n_samp)
    ax.plot(x_plot_all[:,dim].reshape(-1), f_samp_plot[:n_samp_plot,:].T, alpha=.1, color='blue')


def plot_slices(f_sampler, x, y, quantile=.5, n_samp=500, f_true=None, figsize=(4,4)):  
    dim_in = x.shape[1]  
    fig, ax = plt.subplots(1, dim_in, figsize=figsize, sharey=True)

    fig.suptitle("1d slices")
    for dim in range(dim_in):
        ax_dim = ax[dim] if dim_in>1 else ax
        plot_slice(f_sampler, x, y, quantile=quantile, dim=dim, n_samp=n_samp, f_true=f_true, ax=ax_dim)
        ax_dim.set_xlabel('x'+str(dim))

    if dim_in>1:
        ax[0].set_ylabel('y')
        ax[0].legend()
    else:
        ax.set_ylabel('y')
        ax.legend()

    return fig, ax

## --------------- miscellaneous --------------- 

def arrange_full(start, stop, step): 
    # so, e.g., np.arrange(1,10,1) returns [1,2,...,10] instead of [1,2,...,9]
    return np.arange(start, stop+((stop-start)%step==0), step) 





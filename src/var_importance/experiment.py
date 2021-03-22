# standard library imports
import os
import sys
import argparse
import time

# package imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# local imports
import util as util
from data import load_dataset
import models as models

def get_parser():

    parser = argparse.ArgumentParser()

    # general experiment arguments
    parser.add_argument('--dir_out', type=str, default='output/')
    parser.add_argument('--compute_risk', action='store_true', help='compute bias, variance, risk')

    # dataset arguments
    parser.add_argument('--dataset', type=str, default='rbf')
    parser.add_argument('--n_obs', type=int, default=10)
    parser.add_argument('--dim_in', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1, help='seed for dataset')
    parser.add_argument('--subtract_covariates', action='store_true', help='use Y - beta*X as outcome')
    parser.add_argument('--beta_scale', type=float, default=1.0)
    parser.add_argument('--n_nonzero', type=int, default=1)
    
    # general model argument 
    parser.add_argument('--model', type=str, default='GP', help='select e.g. "GP" or "RFF"')
    parser.add_argument('--sig2', type=float, default=.01, help='observational noise')
    
    # general inference arguments
    parser.add_argument('--n_burnin_hmc', type=int, default=5e3)
    parser.add_argument('--n_sample_hmc', type=int, default=10e3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)

    # general RFF options
    parser.add_argument('--n_rff', type=int, default=None, help='set to sqrt(n)log(n) if set to None')

    # RffVarSelect options
    parser.add_argument('--layer_in_name', type=str, default='RffVarSelectLogitNormalLayer')
    parser.add_argument('--s_loc_prior', type=float, default=0.0)
    parser.add_argument('--s_scale_prior', type=float, default=1.0)

    # BKMR options
    parser.add_argument('--bkmr_n_samp', type=int, default=1000)

    # GradPen / BayesLinearLasso arguments:
    parser.add_argument('--prior_w2_sig2', type=float, default=1.0)
    parser.add_argument('--lengthscale', type=float, default=1.0)
    parser.add_argument('--scale_global', type=float, default=1.0)
    parser.add_argument('--scale_groups', type=float, default=1.0)
    parser.add_argument('--penalty_type', type=str, default='l1', help='Choose "l1" for lasso or "l2" for ridge')
    parser.add_argument('--infer_hyper', action='store_true')
    parser.add_argument('--optimize_hyper', action='store_true')

    # GP options
    parser.add_argument('--opt_likelihood_variance', action='store_true')
    parser.add_argument('--opt_kernel_hyperparam', action='store_true')
    parser.add_argument('--kernel_lengthscale', type=float, default=1.0)
    parser.add_argument('--kernel_variance', type=float, default=1.0)

    return parser

def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args)

    if not os.path.exists(args.dir_out):
        os.makedirs(args.dir_out)

    with open(os.path.join(args.dir_out, 'program_info.txt'), 'w') as f:
        f.write('Call:\n%s\n\n' % ' '.join(sys.argv[:]))

    ## allocate space for results
    res = {}

    # set default number of RFF
    if args.n_rff is None:
        n_rff = int(np.sqrt(args.n_obs) * np.log(args.n_obs)) # based on https://papers.nips.cc/paper/2018/file/7ae11af20803185120e83d3ce4fb4ed7-Paper.pdf
    else:
        n_rff = args.n_rff

    # --------- Load data -----------
    data = load_dataset(args.dataset, dim_in=args.dim_in, noise_sig2=args.sig2, n_train=args.n_obs, signal_scale=args.beta_scale, n_nonzero=args.n_nonzero, subtract_covariates=args.subtract_covariates)

    res['psi_train_true'] = data.psi_train
    res['psi_test_true'] = data.psi_test

    # --------- Train model -----------
    start_time = time.time()
    if args.model=='GP':
        m = models.GPyVarImportance(data.x_train, data.y_train, sig2=args.sig2, \
            opt_kernel_hyperparam=args.opt_kernel_hyperparam, \
            opt_sig2=args.opt_likelihood_variance,\
            lengthscale=args.kernel_lengthscale, variance=args.kernel_variance)

        m.train()
        res['kernel_lengthscale'] = m.model.kern.lengthscale.item()
        res['kernel_variance'] = m.model.kern.variance.item()
        print(m.model) 

    elif args.model=='BAYESLINEARLASSO':
        m = models.BayesLinearLassoVarImportance(data.x_train, data.y_train, prior_w2_sig2=args.prior_w2_sig2, noise_sig2=args.sig2, scale_global=[args.scale_global]*args.dim_in)          
        samples, accept = m.train(num_results = args.n_sample_hmc, num_burnin_steps = args.n_burnin_hmc)

    elif args.model=='RFFGRADPEN':
        m = models.RffGradPenVarImportance(data.x_train, data.y_train, n_rff, prior_w2_sig2=args.prior_w2_sig2, noise_sig2=args.sig2, scale_global=[args.scale_global]*args.dim_in, lengthscale=args.lengthscale, penalty_type=args.penalty_type)                        
        samples, accept = m.train(num_results = args.n_sample_hmc, num_burnin_steps = args.n_burnin_hmc)

    elif args.model=='RFFGRADPENHYPER':
        m = models.RffGradPenVarImportanceHyper(data.x_train, data.y_train, n_rff, prior_w2_sig2=args.prior_w2_sig2, noise_sig2=args.sig2, scale_global=[args.scale_global]*args.dim_in, lengthscale=args.lengthscale, penalty_type=args.penalty_type)                        
        samples, accept = m.train(num_results = args.n_sample_hmc, num_burnin_steps = args.n_burnin_hmc)

    elif args.model=='RFFGRADPENHYPER_v2':
        m = models.RffGradPenVarImportanceHyper_v2(data.x_train, data.y_train, n_rff, prior_w2_sig2=args.prior_w2_sig2, noise_sig2=args.sig2, scale_global=[args.scale_global]*args.dim_in, lengthscale=args.lengthscale, penalty_type=args.penalty_type)                        
        samples, accept = m.train(num_results = args.n_sample_hmc, num_burnin_steps = args.n_burnin_hmc, infer_hyper=args.infer_hyper, optimize_hyper=args.optimize_hyper)

        if args.optimize_hyper:
            res['opt_lengthscale'] = m.model.lengthscale.numpy()
            res['opt_prior_w2_sig2'] = m.model.prior_w2_sig2.numpy()

    elif args.model=='RFF':
        m = models.RffVarImportance(Z)
        m.train(data.x_train, data.y_train, args.sig2, rff_dim=n_rff, batch_size=args.batch_size, epochs=args.epochs)

    elif args.model=='RFF-PYTORCH':
        m = models.RffVarImportancePytorch(data.x_train, data.y_train, noise_sig2=args.sig2, prior_w2_sig2=args.prior_w2_sig2, dim_hidden=n_rff, lengthscale=args.lengthscale)
        m.train()

    elif args.model=='RFFHS':
        m = models.RffHsVarImportance(data.x_train, data.y_train, dim_hidden=n_rff, sig2_inv=1/sig2,
            layer_in_name=args.layer_in_name, 
            s_loc_prior=args.s_loc_prior,
            s_scale_prior=args.s_scale_prior)

        loss = m.train(n_epochs=args.epochs, lr=args.lr, path_checkpoint=args.dir_out)

    elif args.model=='BKMR':
        m = models.BKMRVarImportance(data.x_train, data.y_train, args.sig2)    
        m.train(n_samp=args.bkmr_n_samp)

    res['runtime_train'] = time.time() - start_time

    # --------- Analyze results -----------

    # variable importance
    #try:
    psi_est_train = m.estimate_psi(data.x_train)
    res['psi_mean_train'] = psi_est_train[0]
    res['psi_var_train'] = psi_est_train[1]

    if data.x_test is not None:
        psi_est_test = m.estimate_psi(data.x_test)
        res['psi_mean_test'] = psi_est_test[0]
        res['psi_var_test'] = psi_est_test[1]
    #except:
    #    print('Unable to compute variable importance')


    # barplot of variable importance
    try:
        df = pd.DataFrame({
            'variable': np.arange(args.dim_in),
            'estimated': psi_est_train[0],
        })
        if data.psi_train is not None:
            df['true'] =  data.psi_train
        df = pd.melt(df, id_vars=['variable'], var_name='type', value_name='psi')
        fig, ax = plt.subplots()
        sns.barplot(x='variable', y='psi', hue='type', data=df, ax=ax)
        ax.set_title('variable importance on training data')
        fig.savefig(os.path.join(args.dir_out, 'psi_train.png'))
        plt.close()

        if data.x_test is not None:
            df = pd.DataFrame({
            'variable': np.arange(args.dim_in),
            'estimated': psi_est_test[0],
            })
            if data.psi_test is not None:
                df['true'] =  data.psi_test
            df = pd.melt(df, id_vars=['variable'], var_name='type', value_name='psi')
            fig, ax = plt.subplots()
            sns.barplot(x='variable', y='psi', hue='type', data=df, ax=ax)
            ax.set_title('variable importance on test data')
            fig.savefig(os.path.join(args.dir_out, 'psi_test.png'))
            plt.close()

    except:
        print('Unable to plot variable importance')



    # slices of prior predicive
    try:
        if hasattr(m, 'sample_f_prior'):
            fig, ax = util.plot_slices(m.sample_f_prior, data.x_train, data.y_train, quantile=.5, n_samp=100, f_true=data.f, figsize=(4*args.dim_in,4))
            fig.savefig(os.path.join(args.dir_out,'slices_prior.png'))
            plt.close('all')
    except:
        print('Unable to plot prior predictive')

    # slices of posterior predicive
    try:
        if hasattr(m, 'sample_f_post'):
            fig, ax = util.plot_slices(m.sample_f_post, data.x_train, data.y_train, quantile=.5, n_samp=100, f_true=data.f, figsize=(4*args.dim_in,4))
            fig.savefig(os.path.join(args.dir_out,'slices_post.png'))
            plt.close('all')
    except:
        print('Unable to plot posterior predictive')


    # RMSE
    try:
        # could clean up code
        n_samp_risk = 200
        y_hat = np.zeros((n_samp_risk, data.x_train.shape[0]))
        y_hat_test = np.zeros((n_samp_risk, data.x_test.shape[0]))

        for ii in range(n_samp_risk):
            y_hat[ii,:] = m.sample_f_post(data.x_train).reshape(1,-1)
            y_hat_test[ii,:] = m.sample_f_post(data.x_test).reshape(1,-1)

        # posterior predictive mean
        y_hat = np.mean(y_hat, 0).reshape(-1,1)
        y_hat_test = np.mean(y_hat_test, 0).reshape(-1,1)

        # risk 
        res['rmse'] = np.sqrt(np.mean((data.y_train - y_hat)**2))
        res['rmse_test'] = np.sqrt(np.mean((data.y_test - y_hat_test)**2))
    except:
        print('Unable to compute risk')


    try:
        # plot loss if available
        if 'loss' in locals():
            fig, ax = plt.subplots()
            ax.plot(loss.numpy())
            ax.set_xlabel('iterations')
            ax.set_ylabel('loss')
            fig.savefig(os.path.join(args.dir_out, 'loss.png'))
            plt.close('all')


        # plot samples and acceptance if available
        if 'samples' in locals() and 'accept' in locals():

            def plot_samples(samples, accept, param_idx=0):
                fig, ax = plt.subplots()
                ax.plot(samples)
                ax.set_xlabel('iterations')
                ax.set_ylabel('samples (param set %d)' % param_idx)
                ax.set_title('mean = %.3f, accept = %.3f' % (samples.mean(), accept.mean()))
                fig.savefig(os.path.join(args.dir_out, 'trace_%d.png' % param_idx))
                plt.close('all')

            if isinstance(samples, list): 
                for i, s in enumerate(samples):
                    plot_samples(s, accept, i)
            else:
                plot_samples(samples, accept)

        # plot prior if HS
        if args.model=='RFFHS':
            logit = lambda x: np.log(x/(1-x))

            def logitnormal_pdf(x, mu=0, sig2=1):
                return 1/np.sqrt(2*np.pi*sig2)/(x*(1-x))*np.exp(-(logit(x)-mu)**2/(2*sig2))

            eps = .0001
            xx = np.linspace(0+eps,1-eps,1000)
            fig, ax = plt.subplots()
            ax.plot(xx, logitnormal_pdf(xx, args.s_loc_prior, args.s_scale_prior**2))
            ax.set_title('loc = %.3f, scale = %.3f' % (args.s_loc_prior, args.s_scale_prior))
            fig.savefig(os.path.join(args.dir_out,'logitnormal_prior.png'))
            plt.close('all')

        # posterior of s if HS (PLOT CODE SHOULD BE IMPROVED)
        if args.model=='RFFHS' and args.layer_in_name=='RffVarSelectLogitNormalLayer':
            # save 
            res['s_loc_post'] = m.model.layer_in.s_loc.detach().numpy()
            res['s_scale_post'] = m.model.layer_in.transform(m.model.layer_in.s_scale_untrans).detach().numpy()

            # plot
            eps = .0001
            x = np.linspace(0+eps,1-eps,100)
            logit = lambda x: np.log(x/(1-x))
            def logitnormal_pdf(x, mu=0, sig2=1):
                return 1/np.sqrt(2*np.pi*sig2)/(x*(1-x))*np.exp(-(logit(x)-mu)**2/(2*sig2))

            fig, ax = plt.subplots()
            for d in range(m.model.dim_in):
                loc = m.model.layer_in.s_loc[d].detach().numpy()
                scale = m.model.layer_in.transform(m.model.layer_in.s_scale_untrans[d]).detach().numpy()
                ax.plot(x, logitnormal_pdf(x,loc,scale**2), label='s%d: loc=%.5f, scale=%.5f'%(d,loc,scale))
            fig.legend()
            fig.savefig(os.path.join(args.dir_out, 'sposterior.png'))
            plt.close('all')
    except:
        print('Error with miscellaneous plots/metrics')
    
    # save results                     
    np.save(os.path.join(args.dir_out, 'results.npy'), res)

    return res

if __name__ == '__main__':
    main()
# standard library imports
import os
import sys
import argparse

# package imports
import numpy as np
from data import load_dataset

def get_parser():

    parser = argparse.ArgumentParser()

    # dataset arguments
    parser.add_argument('--dataset', type=str, default='rbf')
    parser.add_argument('--n_obs', type=int, default=10)
    parser.add_argument('--dim_in', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1, help='seed for dataset')
    parser.add_argument('--subtract_covariates', action='store_true', help='use Y - beta*X as outcome')
    parser.add_argument('--beta_scale', type=float, default=1.0)
    parser.add_argument('--sig2', type=float, default=.01, help='observational noise')
    parser.add_argument('--dir_out', type=str, default='./data')
    parser.add_argument('--n_nonzero', type=int, default=1)
    return parser

def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args)

    ### load
    data = load_dataset(args.dataset, dim_in=args.dim_in, noise_sig2=args.sig2, n_train=args.n_obs, signal_scale=args.beta_scale, n_nonzero=args.n_nonzero, subtract_covariates=args.subtract_covariates)

    ### save
    if not os.path.exists(args.dir_out):
        os.makedirs(args.dir_out)

    if data.x_train is not None:
        np.savetxt(os.path.join(args.dir_out, 'Z.csv'), data.x_train, delimiter=',')
    #if X is not None:
    #    np.savetxt(os.path.join(args.dir_out, 'X.csv'), X, delimiter=',')
    if data.y_train is not None:
        np.savetxt(os.path.join(args.dir_out, 'Y.csv'), data.y_train, delimiter=',')
    if data.x_test is not None:
        np.savetxt(os.path.join(args.dir_out, 'Z_test.csv'), data.x_test, delimiter=',')
    #if X_test is not None:
    #    np.savetxt(os.path.join(args.dir_out, 'X_test.csv'), X_test, delimiter=',')
    if data.y_test is not None:
        np.savetxt(os.path.join(args.dir_out, 'Y_test.csv'), data.y_test, delimiter=',')




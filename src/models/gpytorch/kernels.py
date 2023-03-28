# standard library imports
import os
import sys
import pdb
import inspect

# package imports
import numpy as np
import gpytorch
from gpytorch.priors import LogNormalPrior, NormalPrior, UniformPrior

# local imports


def make_kernel(**kwargs):

    # priors
    if 'ls_prior' in kwargs:
        ls_prior = UniformPrior(1, 2)
    else:
        ls_prior = None

    if 'var_prior' in kwargs:
        var_prior =  UniformPrior(1, 2)
    else:
        var_prior = None

    # kernel
    if kwargs['name'] == 'rbf':
        kernel = gpytorch.kernels.RBFKernel(lengthscale_prior = ls_prior)

    elif kwargs['name'] == 'matern12':
        kernel = gpytorch.kernels.MaternKernel(nu=1/2, lengthscale_prior = ls_prior)

    elif kwargs['name'] == 'matern32':
        kernel = gpytorch.kernels.MaternKernel(nu=3/2, lengthscale_prior = ls_prior)

    elif kwargs['name'] == 'matern52':
        kernel = gpytorch.kernels.MaternKernel(nu=5/2, lengthscale_prior = ls_prior)

    # always add variance
    kernel = gpytorch.kernels.ScaleKernel(kernel,  outputscale_prior=var_prior)

    # initialize
    if 'ls' in kwargs:
        kernel.base_kernel.lengthscale = kwargs['ls']

    if 'var' in kwargs:
        kernel.outputscale = kwargs['var']

    return kernel


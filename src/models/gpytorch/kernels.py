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
        ls_prior, ls_constraint = make_prior(kwargs['ls_prior'])
    else:
        ls_prior, ls_constraint = None, None

    if 'var_prior' in kwargs:
        var_prior, var_constraint = make_prior(kwargs['var_prior'])
    else:
        var_prior, var_constraint = None, None

    # kernel
    if kwargs['name'] == 'rbf':
        kernel = gpytorch.kernels.RBFKernel(lengthscale_prior = ls_prior, lengthscale_constraint = ls_constraint)

    elif kwargs['name'] == 'matern12':
        kernel = gpytorch.kernels.MaternKernel(nu=1/2, lengthscale_prior = ls_prior, lengthscale_constraint = ls_constraint)

    elif kwargs['name'] == 'matern32':
        kernel = gpytorch.kernels.MaternKernel(nu=3/2, lengthscale_prior = ls_prior, lengthscale_constraint = ls_constraint)

    elif kwargs['name'] == 'matern52':
        kernel = gpytorch.kernels.MaternKernel(nu=5/2, lengthscale_prior = ls_prior, lengthscale_constraint = ls_constraint)

    # always add variance
    kernel = gpytorch.kernels.ScaleKernel(kernel,  outputscale_prior=var_prior, outputscale_constraint=var_constraint)

    # initialize
    if 'ls' in kwargs:
        kernel.base_kernel.lengthscale = kwargs['ls']

    if 'var' in kwargs:
        kernel.outputscale = kwargs['var']

    return kernel

def make_prior(name):
    if name == 'uniform':
        prior = gpytorch.priors.UniformPrior(.1, 10)
        constraint = gpytorch.constraints.Interval(.1, 10)

    elif name == 'gamma':
        prior = gpytorch.priors.GammaPrior(.1, .1)
        constraint = gpytorch.constraints.Positive()

    else:
        raise ValueError('Prior not found')

    return prior, constraint


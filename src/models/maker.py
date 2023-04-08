# standard library imports
import os
import sys
import pdb
import inspect

# package imports
import numpy as np
from .pyro import models as models_pyro
from .numpyro import models as models_numpyro

# local imports
from .gpytorch import models as gp_models
from .gpytorch import kernels
from .gpytorch.std_gp import StandardGP
from .gpytorch.mkl_gp import MKLGP
from .gpytorch.dkl_gp import DKLGP
from .gpytorch.filtered_gp import FilteredGP
from .gpytorch.st_gp import STGP_LML, STGP_MCMC
from .gpytorch.util import make_gaussian_likelihood

from ..util import *

def make_model(ds, **kwargs):
    x = np_to_torch(ds['train']['x'])
    y = np_to_torch(ds['train']['y'])
    likelihood = make_gaussian_likelihood(kwargs['noise_std'])


    if kwargs['name'] == 'std_gp':

        kwargs_kern, kwargs = parse_config(kwargs, 'kern_')
        kernel = kernels.make_kernel(**kwargs_kern)
        gp = StandardGP(x, y, kernel, likelihood)
        
    elif kwargs['name'] == 'mkl_gp':
        kernel0 = kernels.make_kernel(**{'name': 'rbf', 'var_prior': 'gamma'})
        kernel1 = kernels.make_kernel(**{'name': 'matern12', 'var_prior': 'gamma', 'ls_prior': 'gamma'})
        kernel_list = [kernel0, kernel1]

        gp = MKLGP(x, y, kernel_list, likelihood)


    elif kwargs['name'] == 'dkl_gp':
        kwargs_kern, kwargs = parse_config(kwargs, 'kern_')
        kernel = kernels.make_kernel(**kwargs_kern)

        gp = DKLGP(x, y, kernel, likelihood)

    elif kwargs['name'] == 'filtered_gp':
        kwargs_kern, kwargs = parse_config(kwargs, 'kern_')
        kernel = kernels.make_kernel(**kwargs_kern)

        gp = FilteredGP(ds, kernel, likelihood)

    elif kwargs['name'] == 'st_gp':
        kwargs_kern, kwargs = parse_config(kwargs, 'kern_')
        kernel = kernels.make_kernel(**kwargs_kern)
        kernel = kernel.base_kernel

        if kwargs['inference'] == 'lml':
            gp = STGP_LML(x, y, nu=5, rho=3, kernel=kernel, noise_std=kwargs['noise_std'])
        elif kwargs['inference'] == 'mcmc':
            gp = STGP_MCMC(x, y, nu=5, rho=3, kernel=kernel, likelihood=likelihood)

    elif kwargs['name'] == 'bnn':
        Model = models_pyro.BNN

        model = Model(architecture=[1,10,1], noise_std=kwargs['noise_std'])


    elif kwargs['name'] == 'numpyro_bnn':
        Model = models_numpyro.BNN

        model = Model(architecture=[1,10,1], output_var=kwargs['noise_std'], w_prior_var=1.0)

    else:
        raise ValueError('Model not found')



    if 'gp' in kwargs['name']:
        if kwargs['inference'] == 'lml':
            model = gp_models.LMLGP(gp)
        elif kwargs['inference'] == 'mcmc':
            model = gp_models.MCMCGP(gp)


    #a = {k:v for k,v in kwargs.items() if k in inspect.getfullargspec(Model)[0]}

    return model


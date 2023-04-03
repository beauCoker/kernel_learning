# standard library imports
import os
import sys
import pdb
import inspect

# package imports
import numpy as np

# local imports
from .gpytorch import models as models_gpytorch
from .gpytorch import models_DKL as models_gpytorch_dkl
from .gpytorch import models_MKL as models_gpytorch_mkl
from .gpytorch import kernels

from .pyro import models as models_pyro
from .numpyro import models as models_numpyro
from ..util import *

def make_model(ds, **kwargs):

    kwargs_kern, kwargs = parse_config(kwargs, 'kern_')

    if kwargs['name'] == 'exact_gp':

        kernel = kernels.make_kernel(**kwargs_kern)

        Model = models_gpytorch.ExactGP
        a = {k:v for k,v in kwargs.items() if k in inspect.getfullargspec(Model)[0]}
        a['kernel'] = kernel

        model = Model(x=ds['train']['x'], y=ds['train']['y'], **a)
        
    elif kwargs['name'] == 'mcmc_gp':

        kernel = kernels.make_kernel(**kwargs_kern)

        Model = models_gpytorch.MCMCGP
        a = {k:v for k,v in kwargs.items() if k in inspect.getfullargspec(Model)[0]}
        a['kernel'] = kernel

        model = Model(x=ds['train']['x'], y=ds['train']['y'], **a)

    elif kwargs['name'] == 'filtered_gp':

        kernel = kernels.make_kernel(**kwargs_kern)

        Model = models_gpytorch.FilteredGPGrid
        a = {k:v for k,v in kwargs.items() if k in inspect.getfullargspec(Model)[0]}
        a['kernel'] = kernel

        model = Model(ds=ds, **a)

    elif kwargs['name'] == 'dkl_gp':
        kernel = kernels.make_kernel(**kwargs_kern)

        Model = models_gpytorch_dkl.MCMCDKLGP
        a = {k:v for k,v in kwargs.items() if k in inspect.getfullargspec(Model)[0]}
        a['kernel'] = kernel

        model = Model(x=ds['train']['x'], y=ds['train']['y'], **a)


    elif kwargs['name'] == 'mkl_gp':
        #kernel = kernels.make_kernel(**kwargs_kern)
        kernel0 = kernels.make_kernel(**{'name': 'rbf'})
        kernel1 = kernels.make_kernel(**{'name': 'matern12'})

        Model = models_gpytorch_mkl.MKLGP
        a = {k:v for k,v in kwargs.items() if k in inspect.getfullargspec(Model)[0]}
        a['kernels'] = [kernel0, kernel1]

        model = Model(x=ds['train']['x'], y=ds['train']['y'], **a)

    elif kwargs['name'] == 'bnn':
        Model = models_pyro.BNN

        model = Model(architecture=[1,10,1], noise_std=kwargs['noise_std'])


    elif kwargs['name'] == 'numpyro_bnn':
        Model = models_numpyro.BNN

        model = Model(architecture=[1,10,1], output_var=kwargs['noise_std'], w_prior_var=1.0)


    return model


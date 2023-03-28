# standard library imports
import os
import sys
import pdb
import inspect

# package imports
import numpy as np

# local imports
from .gpytorch import models as models_gpytorch
from .gpytorch import kernels
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

    return model


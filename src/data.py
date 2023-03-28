# standard library imports
import os
import sys
import pdb

# package imports
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

# local imports
from .models.gpytorch.kernels import make_kernel
from .models.gpytorch.models import ExactGP
from .util import *

#BASEDIR = '../../data/'


def load_dataset(**kwargs):

    kwargs_kern, kwargs = parse_config(kwargs, 'kern_')

    if kwargs['name'] == 'GP':
        ds = gen_gp_dataset(
            n_train=kwargs['n_train'],
            n_test=100,
            dim_in=kwargs['dim_in'], 
            noise_std=kwargs['noise_std'], 
            seed=kwargs['seed'], 
            seed_split=kwargs['seed_split'],
            kwargs_kern=kwargs_kern,
        )

    elif kwargs['name'] == 'GPgrid':
        ds = gen_gp_grid_dataset(
            n_train=kwargs['n_train'], 
            dim_in=kwargs['dim_in'], 
            noise_std=kwargs['noise_std'], 
            seed=kwargs['seed'], 
            seed_split=kwargs['seed_split'],
            kwargs_kern=kwargs_kern,
        )

    else:
        raise ValueError('Dataset not found')

    return ds

def train_test_split_dataset(variables, test_size=0.1, shuffle=True, seed=0):
    '''
    e.g. variables = {'x': np.zeros(10), 'y': np.zeros(10)}
    '''
    ds = {'train': {}, 'test': {}, 'info': {}}
    for varname, var in variables.items():
        var_split = train_test_split(var, test_size=test_size, shuffle=shuffle, random_state=seed)
        ds['train'][varname] = var_split[0]
        ds['test'][varname] = var_split[1]
    ds['info'] = {}
    return ds


def gen_gp_dataset(n_train, n_test, dim_in, noise_std, n_grid=100, seed=None, seed_split=None, kwargs_kern={}):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # inputs
    n_obs = n_train + n_test
    x = generate_x(n_obs, dim_in, dist='uniform', rng=rng)
    if dim_in == 1:
        x = np.concatenate([x, np.linspace(-1.5, 1.5, n_grid).reshape(-1,1)], axis=0)

    kernel = make_kernel(**kwargs_kern)
    model = ExactGP(kernel, x=torch.tensor([]), y=torch.tensor([]), noise_std=1.0)
    model.model.double()

    # function
    f = model.sample_f(x, n_samp=1, prior=True)[0, ...]

    # observed values
    noise = rng.normal(0, noise_std, x.shape[0])
    y = f + noise

    ds = train_test_split_dataset(
        {'x': x[:n_obs, :],
        'f': f[:n_obs],
        'y': y[:n_obs],
        },
        test_size=n_test/n_obs,
        seed=seed_split
    )

    if ds['train']['x'].shape[0] != n_train:
        print('WARNING: n_train is %d but dataset has %d training observations' % (n_train, ds['train']['x'].shape[0]))

    # add other stuff to dataset (not split by)
    ds['train']['k'] = model.predict_k(ds['train']['x'])
    ds['test']['k'] = model.predict_k(ds['test']['x'])

    ds['info']['noise_std'] = noise_std

    if dim_in == 1:
        ds['grid'] = {}
        ds['grid']['x'] = x[n_obs:, :]
        ds['grid']['y'] = y[n_obs:]
        ds['grid']['f'] = f[n_obs:]
        ds['grid']['k'] = model.predict_k(x[n_obs:, :])

    return ds

def generate_x(n_obs, dim_in, dist='uniform', rng=None):
    if rng is None:
        rng = np.random.default_rng()

    if dist=='uniform':
        x = rng.uniform(-1, 1, (n_obs, dim_in))

    elif dist=='normal':
        x = rng.normal(0, 1, (n_obs, dim_in))

    return x




def gen_gp_grid_dataset(n_train, dim_in, noise_std, n_grid=90, seed=None, seed_split=None, kwargs_kern={}):
    assert dim_in==1
    assert n_train < n_grid
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # inputs
    x_train = generate_x(n_train, dim_in, dist='uniform', rng=rng)
    x = np.linspace(-1.5, 1.5, n_grid).reshape(-1,1)

    # map to grid
    diff = np.expand_dims(x_train,-1)-np.expand_dims(x,0)
    idx_train = np.argmin(np.linalg.norm(diff, axis=-1), axis=-1)
    idx_test = np.setdiff1d(np.arange(n_grid), idx_train)

    x_train = x[idx_train]
    x_test = x[idx_test]

    kernel = make_kernel(**kwargs_kern)
    model = ExactGP(kernel, x=torch.tensor([]), y=torch.tensor([]), noise_std=1.0)
    model.model.double()

    # function
    f = model.sample_f(x, n_samp=1, prior=True)[0, ...]

    # observed values
    noise = rng.normal(0, noise_std, x.shape[0])
    y = f + noise

    # store
    ds = {'train': {}, 'test': {}, 'grid': {}, 'info': {}}
    ds['train']['x'] = x_train
    ds['train']['y'] = y[idx_train]
    ds['train']['f'] = f[idx_train]
    ds['train']['k'] = model.predict_k(x_train)

    ds['test']['x'] = x_test
    ds['test']['y'] = y[idx_test]
    ds['test']['f'] = f[idx_test]
    ds['test']['k'] = model.predict_k(x_test)

    ds['grid']['x'] = x
    ds['grid']['y'] = y
    ds['grid']['f'] = f
    ds['grid']['k'] = model.predict_k(x)

    ds['info']['idx_train'] = idx_train
    ds['info']['idx_test'] = idx_test
    ds['info']['idx_grid'] = np.arange(n_grid)

    ds['info']['noise_std'] = noise_std


    return ds
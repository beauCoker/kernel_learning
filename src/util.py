# standard library imports
import os
import sys
import pdb
import inspect

# package imports
import torch
import numpy as np
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

# local imports


def to_jnp(*arr):
    arr_jnp = [jnp.asarray(a) if not isinstance(a, jnp.ndarray) else a for a in arr]
    return arr_jnp if len(arr_jnp)>1 else arr_jnp[0]

def to_np(*arr):
    arr_np = [np.asarray(a) if not isinstance(a, np.ndarray) else a for a in arr]
    return arr_np if len(arr_np)>1 else arr_np[0]

def torch_to_np(*arr):
    arr_numpy = [a.numpy() if isinstance(a, torch.Tensor) else a for a in arr]
    return arr_numpy if len(arr_numpy)>1 else arr_numpy[0]

def np_to_torch(*arr):
    arr_torch = [torch.from_numpy(a) if not isinstance(a, torch.Tensor) else a for a in arr]
    return arr_torch if len(arr_torch)>1 else arr_torch[0]

def to_flat_dict(d, d_add, prefix=''):
    for key, val in d_add.items():
        if isinstance(val, dict):
            new_prefix = prefix+'_'+key if prefix != '' else key
            d = to_flat_dict(d, val, prefix = new_prefix)
        else:
            d[prefix + '_' + key] = val
    return d

def ds_astype(ds, dtype=np.float32):
    for split_name, split_vars in ds.items():
        if split_name not in ['train','test','val','grid']:
            continue
        for varname, var in split_vars.items():
            if isinstance(var, np.ndarray):
                ds[split_name][varname] = var.astype(dtype)
            elif isinstance(var, torch.Tensor):
                ds[split_name][varname] = var.type(dtype)
    return ds

def parse_config(config, *prefixes):
    '''
    Separates dictionary (config) by string prefixes in key
    Returns N+1 length tuple of dictionaries where N is number of prefixes 
    (last element is entries that don't match a prefix)
    '''
    config_remain = dict(config).copy()
    configs = []
    for prefix in prefixes: 
        configs.append({})
        l = len(prefix)
        for key, val in config.items():
            if isinstance(key,str) and key[:l] == prefix:
                configs[-1][key[l:]] = val
                config_remain.pop(key)
    return *configs, config_remain

def transfer_args(args_list, config_base, *configs):
    '''
    transfers args from config_base to any other configs
    '''
    for arg in args_list:
        for config in configs:
            config[arg] = config_base[arg]    
# standard library imports
import os
import sys
import pdb
import inspect

# package imports
import numpy as np
import jax.numpy as jnp
from jax import random, vmap
from jax.scipy.stats import norm, multivariate_normal
from jax.config import config
config.update("jax_enable_x64", True)

# local imports
from .util import *


# these real-valued metrics should take objects over all observations
# last argument should be true value
sq_err = lambda x, y: jnp.mean(jnp.power(x - y, 2))
fro_err = lambda A, B: jnp.linalg.norm(A - B, ord='fro')
align_err = lambda A, B: jnp.sum(A * B) / jnp.sqrt(np.sum(A**2) * jnp.sum(B**2))
def nlpd_err(mean, cov, obs):
    #z = multivariate_normal(mean = mean, cov=cov)
    #return -z.logpdf(test_y)/test_y.shape[0]
    return -multivariate_normal.logpdf(obs, mean=mean, cov=cov)/obs.shape[0]

def diagnlpd_err(mean, cov, obs):
    return -jnp.mean(norm.logpdf(obs, loc=mean, scale=jnp.sqrt(jnp.diag(cov)).reshape(-1)))


def compute_error(metric, true, *pred):
    '''
    pred: (n_obs, ...)
    true: (n_obs, ...)
    f: scalar-valued function of two (...) dimensional inputs
    '''
    true, *pred = to_jnp(true, *pred)
    error = metric(*pred, true)
    return to_np(error)

def compute_risk(metric, true, *samples):
    true, *samples = to_jnp(true, *samples)
    metric_vec = vmap(lambda x: metric(*x, true)) # vectorize over samples
    risk = jnp.mean(metric_vec(samples)) # average over samples
    return to_np(risk)

def compute_error_risk(metric, true, *samples):

    true, *samples = to_jnp(true, *samples)
    samples_mean = [jnp.mean(s, 0) for s in samples]
    
    metric_vec = vmap(lambda x: metric(*x, true)) # vectorize over samples

    error = metric(*samples_mean, true)
    risk = jnp.mean(metric_vec(samples)) # average over samples

    return to_np(error, risk)
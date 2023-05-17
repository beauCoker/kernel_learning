# standard library imports
import os
import sys
import pdb

# package imports
import torch
import gpytorch
from gpytorch.priors import LogNormalPrior, NormalPrior, UniformPrior
import pyro
from pyro.infer.mcmc import NUTS, MCMC, HMC
from torch.autograd.functional import hessian

# local imports
from src.util import *

def expected_upcrossings(kernel, u=0):
    '''
    See Eq 4.3 in RW
    '''
    if not kernel.is_stationary:
        print('WARNING: kernel should be stationary to compute expected upcrossings')
    r = torch.tensor([[0.0]])
    iso_kernel = lambda r: kernel(r, r).to_dense()

    kpp0 = hessian(iso_kernel, r)
    k0 = iso_kernel(r)

    E_N = 1/(2*torch.pi) * torch.sqrt(-kpp0 / k0) * torch.exp(-u**2/(2*k0))
    return E_N.item()

def estimate_eigenvalue_decay(kernel=None, K=None, n=1000):
    '''

    '''
    range_fit = range(1, 100)
    index = np.arange(n)
    lam = estimate_eigenvalues(kernel=kernel, K=K, n=n)
    a, b = estimate_power_law_lm(index[range_fit], lam[range_fit])
    return b

from sklearn.linear_model import LinearRegression
def estimate_power_law_lm(x, y):
    '''
    y = a exp(-b*x) <--> log(y) = log(a) - b*log(x)
    '''
    np.log(x)
    m = LinearRegression().fit(np.log(x).reshape(-1,1), np.log(y))
    a = np.exp(m.intercept_)
    b = -m.coef_
    return a, b[0]

def estimate_eigenvalues(kernel=None, K=None, n=1000):
    if kernel is not None and K is none:
        dist = torch.distributions.uniform.Uniform(0, 1)
        #dist = torch.distributions.normal.Normal(0, 1)
        x = dist.sample(torch.Size((n,))).reshape(-1, 1)
        K = kernel(x).to_dense()

    K = torch_to_np(K)
    with torch.no_grad():
        evals, evecs = np.linalg.eigh(K)    
    lam = np.flip(evals, (0,))/K.shape[0]
    
    return lam

def make_gaussian_likelihood(noise_std, train=False):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = noise_std**2
    likelihood.noise_covar.raw_noise.requires_grad_(train)  
    return likelihood

def training_loop(model, loss, x, y, n_epochs=10, lr=0.1, verbose=True, **kwargs):

    # Find optimal model hyperparameters
    model.train()
    #likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = torch.zeros(n_epochs)

    for i in range(n_epochs):

        # Zero gradients from previous iteration
        optimizer.zero_grad()

        # Output from model
        output = model(x)

        # Calc loss and backprop gradients
        loss_value = -loss(output, y)
        loss_value.backward()

        if verbose:
            print('Iter %d/%d - Loss: %.3f' % (i + 1, n_epochs, loss_value.item()))
        optimizer.step()

        losses[i] = loss_value.item()

    #print('final loss:', -loss(model(x), y).item())

    return {'loss': losses.numpy()}

def get_prior_hypers(model, n_samp):
    '''
    samples from all priors
    '''
    out = {}
    for prior in model.named_priors():
        name = prior[0]
        dist = prior[2]
        out[name] = dist.sample(torch.Size((n_samp,))).numpy()
    return out

def get_hyperparams(model):
    '''
    Return all hyperparameters of model
    NOTE: flattens all arrays
    '''
    out = {}
    for name, param, constraint in model.named_parameters_and_constraints():
        value = constraint.transform(param) if constraint is not None else param
        out[name] = value.detach().reshape(-1).numpy()
    return out 
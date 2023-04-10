# standard library imports
import os
import sys
import pdb
import copy

# package imports
import torch
import gpytorch
import pyro
from pyro.infer.mcmc import NUTS, MCMC, HMC

# local imports
from src.util import *
from .util import training_loop


class Base:
    def __init__(self):
        pass

    def predict_fy_dist(self, x, prior=False):
        raise NotImplementedError

    def sample_fy_dist(self, x, prior=False):
        raise NotImplementedError

    def sample_f(self, x, n_samp, prior=False):
        raise NotImplementedError

    def sample_k(self, x, n_samp, prior=False):
        raise NotImplementedError

    def sample_k_hypers(self):
        raise NotImplementedError


class LMLGP(object):
    def __init__(self, gp):
        self.gp = gp
        self.gp.double()

        self.covar_module_init = copy.deepcopy(self.gp.covar_module)

    def fit(self, x, y, **kwargs):
        x, y = np_to_torch(x, y)
        self.gp.train()

        if hasattr(self.gp, 'ExactMarginalLogLikelihood'):
            mll = self.gp.ExactMarginalLogLikelihood
        else:
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)

        out = training_loop(model=self.gp, loss=mll, x=x, y=y, **kwargs)
        self.gp.eval()
        return out

    @torch.no_grad()
    def predict_f(self, x, prior=False):
        fdist, _ = self.predict_fy_dist(x, prior=prior)
        return torch_to_numpy(fdist.mean)

    @torch.no_grad()
    def sample_f(self, x, n_samp, prior=False):
        fdist, _ = self.predict_fy_dist(x, prior=prior)
        return torch_to_np(fdist.sample(sample_shape=torch.Size((n_samp,))))

    @torch.no_grad()
    def predict_fy_dist(self, x, prior=False):
        '''
        Returns n-dimensional multivariate normal
        '''
        x = np_to_torch(x)
        self.gp.eval()
        #with gpytorch.settings.fast_pred_var(), gpytorch.settings.prior_mode(prior):
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.prior_mode(prior), gpytorch.settings.debug(False):
            fdist = self.gp(x)
            ydist = self.gp.likelihood(fdist)
        return fdist, ydist

    @torch.no_grad()
    def predict_k(self, x, prior=False):
        x = np_to_torch(x)
        self.gp.eval()

        if prior:
            with gpytorch.settings.prior_mode(True): 
                covar_module_current = copy.deepcopy(self.gp.covar_module)
                self.gp.covar_module = self.covar_module_init
                fdist = self.gp(x)
                self.gp.covar_module = covar_module_current

        else:
            with gpytorch.settings.prior_mode(True): # want prior_mode=True because want the kernel that was learned
                fdist = self.gp(x)

        return torch_to_np(fdist.covariance_matrix.detach())


class MCMCGP(object):
    def __init__(self, gp):
        self.gp = gp
        self.gp.double()

    def fit(self, x, y, n_samp=100, n_warmup=10, verbose=False, **kwargs):
        x, y = np_to_torch(x, y)
        self.gp.train()

        def pyro_model(x, y):
            with gpytorch.settings.fast_computations(False, False, False):
                sampled_model = self.gp.pyro_sample_from_prior()
                output = sampled_model.likelihood(sampled_model(x))
                pyro.sample("obs", output, obs=y)
            return y

        nuts_kernel = NUTS(pyro_model)
        mcmc_run = MCMC(nuts_kernel, num_samples=n_samp, warmup_steps=n_warmup, disable_progbar=False)
        mcmc_run.run(x, y)

        self.gp.eval()
        self.samples = mcmc_run.get_samples()
        self.gp.pyro_load_from_samples(self.samples)
        self.n_samp_stored = n_samp
        return self.samples

    @torch.no_grad()
    def sample_f(self, x, n_samp, prior=False):
        fdist, _ = self.predict_fdist(x, prior=prior)
        return torch_to_np(fdist.sample(sample_shape=torch.Size((n_samp,))))

    @torch.no_grad()
    def predict_k(self, x, prior=False):
        # would just be mean of samples
        pass

    @torch.no_grad()
    def sample_k(self, x, n_samp, prior=False):
        '''
        prior refers to the kernel hyperparameters. This is not the kernel of the fuction-space posterior.
        '''
        if not prior:
            assert n_samp == self.n_samp_stored
        x = np_to_torch(x)
        self.gp.eval()

        if prior:
            with gpytorch.settings.prior_mode(True): 
                sampled_model = self.gp.pyro_sample_from_prior()
                fdist = sampled_model(x)

        else:
            with gpytorch.settings.fast_pred_var(), gpytorch.settings.prior_mode(True): # want prior_mode=True because want the kernel that was learned
                fdist = self.gp(x)

        return torch_to_np(fdist.covariance_matrix.detach())

    @torch.no_grad()
    def sample_fy_dist(self, x, n_samp, prior=False):
        if not prior:
            assert n_samp == self.n_samp_stored
        x = np_to_torch(x)
        #expanded_x = x.unsqueeze(0).repeat(n_samp, 1, 1) # I don't think this is needed after pyro_load_from_samples called

        self.gp.eval()

        if prior:
            with gpytorch.settings.prior_mode(True):
                sampled_model = self.gp.pyro_sample_from_prior()
                fdist = sampled_model(x)
                
                # hack to get shape right. I think this happens because 
                if len(fdist.mean.shape)==1:
                    fdist.mean.data = fdist.mean.unsqueeze(0).repeat(n_samp, 1) # HACK. This could fail if there is prior over mean function.
                
                ydist = sampled_model.likelihood(fdist)

        else:
            with gpytorch.settings.fast_pred_var(), gpytorch.settings.prior_mode(False):
                fdist = self.gp(x)
                ydist = self.gp.likelihood(fdist)

        return fdist, ydist


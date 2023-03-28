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

# local imports
from src.util import *

class ExactGP(object):
    def __init__(self, kernel, x, y, noise_std, train_likelihood=False):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise = noise_std**2
        self.train_likelihood = train_likelihood

        if not self.train_likelihood:
            self.likelihood.noise_covar.raw_noise.requires_grad_(False)  

        x, y = np_to_torch(x, y)
        self.model = ExactGPModel(x, y, kernel, self.likelihood)

        self.model.eval()
        self.likelihood.eval()

    def fit(self, x, y, **kwargs):
        x, y = np_to_torch(x, y)

        self.model.train()
        if self.train_likelihood:
            self.likelihood.train()

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        self.model.eval()
        self.likelihood.eval()

        return training_loop(model=self.model, loss=mll, x=x, y=y, **kwargs)

    def predict_f(self, x, prior=False):
        fdist, _ = self.predict_fdist(x, prior=prior)
        return torch_to_numpy(fdist.mean)

    def sample_f(self, x, n_samp, prior=False):
        fdist, _ = self.predict_fdist(x, prior=prior)
        return torch_to_np(fdist.sample(sample_shape=torch.Size((n_samp,))))

    def predict_fdist(self, x, prior=False):
        x = np_to_torch(x)
        with torch.no_grad(),  gpytorch.settings.fast_pred_var(), gpytorch.settings.prior_mode(prior):
            self.model.eval()
            self.likelihood.eval()
            fdist = self.model(x)
            ydist = self.likelihood(fdist)
        return fdist, ydist

    def sample_fdist(self, x, n_samp, prior=False):
        x = np_to_torch(x)
        expanded_x = x.unsqueeze(0).repeat(n_samp, 1, 1)
        return self.predict_fdist(expanded_x, prior=prior)

    def predict_k(self, x, prior=False):
        # what should this do if prior=True?
        if prior:
            print('WARNING: unclear how prior should perform... should it sample from prior?')
        x = np_to_torch(x)
        return torch_to_np(self.model.covar_module(x).to_dense().detach())

    def sample_k(self, x, n_samp, prior=False):
        # just repeats prediction (since hypers fixed)
        if prior:
            print('WARNING: unclear how prior should perform... should it sample from prior?')
        x = np_to_torch(x)
        
        '''
        expanded_x = x.unsqueeze(0).repeat(n_samp, 1, 1)
        with torch.no_grad(),  gpytorch.settings.fast_pred_var(), gpytorch.settings.prior_mode(prior):
            self.model.eval()
            self.likelihood.eval()
            k_samp = self.model.covar_module(expanded_x).to_dense().detach()

        return torch_to_np(k_samp)
        '''
        return torch_to_np(self.model.covar_module(x).to_dense().detach().unsqueeze(0).repeat(n_samp, 1, 1))


class MCMCGP(object):
    def __init__(self, kernel, x, y, noise_std, train_likelihood=False):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise = noise_std**2
        self.train_likelihood = train_likelihood

        if not self.train_likelihood:
            self.likelihood.noise_covar.raw_noise.requires_grad_(False)  

        x, y = np_to_torch(x, y)
        self.model = ExactGPModel(x, y, kernel, self.likelihood)

    def fit(self, x, y, n_samp=100, n_warmup=10, verbose=False, **kwargs):
        x, y = np_to_torch(x, y)

        self.model.train()
        if self.train_likelihood:
            self.likelihood.train()

        def pyro_model(x, y):
            with gpytorch.settings.fast_computations(False, False, False):
                sampled_model = self.model.pyro_sample_from_prior()
                output = sampled_model.likelihood(sampled_model(x))
                pyro.sample("obs", output, obs=y)
            return y

        nuts_kernel = NUTS(pyro_model)
        mcmc_run = MCMC(nuts_kernel, num_samples=n_samp, warmup_steps=n_warmup, disable_progbar=False)
        mcmc_run.run(x, y)

        self.model.eval()
        self.likelihood.eval()

        samples = mcmc_run.get_samples()
        self.model.pyro_load_from_samples(samples)
        self.n_samp_stored = n_samp
        return samples

    def predict_f(self, x, prior=False):
        # based on samples
        #n_samp = 1000 if prior else self.mcmc_run.num_samples
        #samples = self.sample(n_samp=n_samp, prior=prior)
        pass

    def sample_f(self, x, n_samp, prior=False):
        fdist, _ = self.predict_fdist(x, prior=prior)
        return torch_to_np(fdist.sample(sample_shape=torch.Size((n_samp,))))

    def predict_k(self, x, prior=False):
        # would just be mean of samples
        pass

    def sample_k(self, x, n_samp, prior=False):
        if not prior:
            assert n_samp == self.n_samp_stored
        x = np_to_torch(x)
        expanded_x = x.unsqueeze(0).repeat(n_samp, 1, 1)
        with torch.no_grad(),  gpytorch.settings.fast_pred_var(), gpytorch.settings.prior_mode(prior):
            self.model.eval()
            self.likelihood.eval()
            k_samp = self.model.covar_module(expanded_x).to_dense().detach()

        return torch_to_np(k_samp)

    def sample_fdist(self, x, n_samp, prior=False):
        if not prior:
            assert n_samp == self.n_samp_stored
        x = np_to_torch(x)
        expanded_x = x.unsqueeze(0).repeat(n_samp, 1, 1)

        with torch.no_grad(),  gpytorch.settings.fast_pred_var(), gpytorch.settings.prior_mode(prior):
            self.model.eval()
            self.likelihood.eval()
            fdist = self.model(expanded_x)
            ydist = self.likelihood(fdist)

        return fdist, ydist


class ExactGPModel(gpytorch.models.ExactGP):
    '''
    example: kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    '''
    def __init__(self, train_x, train_y, kernel, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


###########
import numpy as np
def ones(*args, **kwargs):
    t = torch.ones(*args, **kwargs)
    #if cuda_available():
    #    t = t.cuda()
    return t

def dct_matrix(N):
    '''
    Returns an (N,N) matrix A corresponding to the orthnormal DCT-II transformation
    Inverse transformation is the transpose of this matrix
    
    In other words:
    - A @ f.T = dct(f, type=2, norm='ortho', axis=1).T
    - f.T = A.T @ dct(f, type=2, norm='ortho', axis=1)
    where f is an (n_samp, N) array
    
    N: number of function evaluations
    '''
    k = torch.arange(N).unsqueeze(-1).type(torch.float64)
    n = torch.arange(N).type(torch.float64)
    A = 2.0 * torch.cos(np.pi * k * (2.0 * n + 1.0) / (2.0 * N))
    
    # normalization stuff (so A is orthonormal)
    A[0,:] = A[0,:] / np.sqrt(2.0) 
    A = A / np.sqrt(2.0 * N)
    
    #if cuda_available():
    #    A = A.cuda()

    return A

def lowpass_reconstructor_matrix(N, K, c=None):
    '''
    Returns matrix corresponding to the lowpass_reconstructor
    
    Example: 
        Let f be an (n_samp, N) array and let A = lowpass_reconstructor_matrix(N, K)
        Then f @ A is the reconstructed function samples without the top K highest frequencies
    
    N: number of function evaluations
    K: number of frequencies considered "high" (and so zeroed out), 0 <= K <= N
    '''    
    A = dct_matrix(N)

    if c is None:
        c = ones(N).type(torch.float64)
        if K > 0:
            c[-K:] = 0.0

    return A.t() @ torch.diag(c) @ A

class FilteredGPModel(gpytorch.models.ExactGP):
    '''
    example: kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    '''
    def __init__(self, train_x, test_x, grid_x, train_y, kernel, likelihood):
        super(FilteredGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

        self.train_x = train_x
        self.test_x = test_x
        self.grid_x = grid_x
        self.x_all = torch.concat([train_x, test_x, grid_x], 0)

        self.n_train = train_x.shape[0]
        self.n_test = test_x.shape[0]
        self.n_grid = grid_x.shape[0]
        self.n_all = self.n_train + self.n_test + self.n_grid

        self.idx_train = torch.arange(self.n_train)
        self.idx_test = torch.arange(self.n_train, self.n_train+self.n_test)
        self.idx_grid = torch.arange(self.n_train+self.n_test, self.n_train+self.n_test+self.n_grid)

        self.T = lowpass_reconstructor_matrix(self.n_all, 99).type(train_x.dtype)
    
    def forward(self, x):
        if x.shape[0]==self.n_train and torch.all(self.train_x == x):
            idx = self.idx_train

        elif x.shape[0]==self.n_test and torch.all(self.test_x == x):
            idx = self.idx_test

        elif x.shape[0]==self.n_grid and torch.all(self.grid_x == x):
            idx = self.idx_grid

        elif x.shape[0]==self.n_train+self.n_train and torch.all(torch.concat([self.train_x, self.train_x], 0) == x):
            idx = torch.concat([self.idx_train, self.idx_train])

        elif x.shape[0]==self.n_train+self.n_test and torch.all(torch.concat([self.train_x, self.test_x], 0) == x):
            idx = torch.concat([self.idx_train, self.idx_test])

        elif x.shape[0]==self.n_train+self.n_grid and torch.all(torch.concat([self.train_x, self.grid_x], 0) == x):
            idx = torch.concat([self.idx_train, self.idx_grid])


        else:
            breakpoint()

        #mean_x = self.mean_module(self.x_all[idx])
        #covar_x = self.covar_module(self.x_all[idx])


        mean_x_all = self.mean_module(self.x_all)
        covar_x_all = self.covar_module(self.x_all)

        # apply filtering
        mean_x_all = self.T @ mean_x_all
        covar_x_all = self.T @ covar_x_all @ self.T.t()

        #covar_x_all = covar_x_all + torch.eye(self.n_all).type(x.dtype)*1e-6

        # select by index
        mean_x = mean_x_all[idx]
        covar_x = covar_x_all[idx][:, idx]

        covar_x = covar_x + torch.eye(self.x_all[idx].shape[0]).type(x.dtype)*1e-6

        print('------------', covar_x.shape)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class FilteredGP(ExactGP):
    def __init__(self, kernel, train_x, test_x, grid_x, train_y, noise_std):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise = noise_std**2
        self.likelihood.noise_covar.raw_noise.requires_grad_(False)  

        train_x, test_x, grid_x, train_y = np_to_torch(train_x, test_x, grid_x, train_y)
        self.model = FilteredGPModel(train_x=train_x, test_x=test_x, grid_x=grid_x, train_y=train_y, kernel=kernel, likelihood=self.likelihood)







##########


def training_loop(model, loss, x, y, n_epochs=10, lr=0.1, verbose=True, **kwargs):

    # Find optimal model hyperparameters
    model.train()
    #likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)    

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



###

class FilteredGPGridModel(gpytorch.models.ExactGP):
    '''
    example: kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    '''
    def __init__(self, ds, kernel, likelihood):
        super(FilteredGPGridModel, self).__init__(torch.from_numpy(ds['train']['x']), torch.from_numpy(ds['train']['y']), likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

        self.train_x = torch.from_numpy(ds['train']['x'])
        self.test_x = torch.from_numpy(ds['test']['x'])
        self.grid_x = torch.from_numpy(ds['grid']['x'])

        self.n_train = self.train_x.shape[0]
        self.n_test = self.test_x.shape[0]
        self.n_grid = self.grid_x.shape[0]

        self.idx_train = torch.from_numpy(ds['info']['idx_train'])
        self.idx_test = torch.from_numpy(ds['info']['idx_test'])
        self.idx_grid = torch.from_numpy(ds['info']['idx_grid'])

        self.c = torch.nn.Parameter(torch.randn(self.grid_x.shape[0]))

        #self.T = lowpass_reconstructor_matrix(self.grid_x.shape[0], 80, self.c).type(self.train_x.dtype).t()
        #self.T = torch.eye(self.grid_x.shape[0]).type(self.train_x.dtype)

    def forward(self, x):
        n_x = x.shape[-2]
        if len(x.shape)>2:
            x0 = x[0,:,:]
        else:
            x0 = x

        if n_x==self.n_train and torch.all(self.train_x == x0):
            idx = self.idx_train

        elif n_x==self.n_test and torch.all(self.test_x == x0):
            idx = self.idx_test

        elif n_x==self.n_grid and torch.all(self.grid_x == x0):
            idx = self.idx_grid

        elif n_x==self.n_train+self.n_train and torch.all(torch.concat([self.train_x, self.train_x], 0) == x0):
            idx = torch.concat([self.idx_train, self.idx_train])

        elif n_x==self.n_train+self.n_test and torch.all(torch.concat([self.train_x, self.test_x], 0) == x0):
            idx = torch.concat([self.idx_train, self.idx_test])

        elif n_x==self.n_train+self.n_grid and torch.all(torch.concat([self.train_x, self.grid_x], 0) == x0):
            idx = torch.concat([self.idx_train, self.idx_grid])
            
        else:
            breakpoint()

        self.T = lowpass_reconstructor_matrix(self.grid_x.shape[0], 80, self.c).type(self.train_x.dtype).t() # need to recompute if using self.c
        #self.T = lowpass_reconstructor_matrix(self.grid_x.shape[0], 80).type(self.train_x.dtype).t() # without c
        #self.T = torch.eye(self.grid_x.shape[0]).type(self.train_x.dtype) # without any filtering

        #mean_x = self.mean_module(self.grid_x
        #covar_x = self.covar_module(self.grid_x

        # expand grid_x as necessary
        if len(x.shape)>2:
            grid_x = self.grid_x.unsqueeze(0).repeat(x.shape[0], 1, 1)
        else:
            grid_x = self.grid_x

        mean_x_all = self.mean_module(grid_x)
        covar_x_all = self.covar_module(grid_x)

        # apply filtering
        mean_x_all = torch.matmul(mean_x_all, self.T.t())
        covar_x_all = torch.matmul(self.T, torch.matmul(covar_x_all, self.T.t()))

        #covar_x_all = covar_x_all + torch.eye(self.n_all).type(x.dtype)*1e-6

        # select by index
        if len(x.shape)>2:
            mean_x = mean_x_all[:,idx]
            covar_x = covar_x_all[:,idx][:, :, idx]
            covar_x = covar_x + torch.eye(idx.shape[0]).unsqueeze(0)*1e-8

        else:
            mean_x = mean_x_all[idx]
            covar_x = covar_x_all[idx][:, idx]
            covar_x = covar_x + torch.eye(idx.shape[0])*1e-8

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class FilteredGPGrid(ExactGP):
    def __init__(self, kernel, ds, noise_std):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise = noise_std**2
        self.likelihood.noise_covar.raw_noise.requires_grad_(False)  

        self.model = FilteredGPGridModel(ds=ds, kernel=kernel, likelihood=self.likelihood)

    def sample_fdist(self, x, n_samp, prior=False):
        x = np_to_torch(x)
        expanded_x = x.unsqueeze(0).repeat(n_samp, 1, 1)

        with torch.no_grad(),  gpytorch.settings.fast_pred_var(), gpytorch.settings.prior_mode(prior):
            self.model.eval()
            fdist = self.model(expanded_x)
            ydist = self.likelihood(fdist)

        return fdist, ydist
# standard library imports
import os
import sys
import pdb

# package imports
import torch
import gpytorch

# local imports
from src.util import *

class StandardGP(gpytorch.models.ExactGP):
    '''
    '''
    def __init__(self, x, y, kernel, likelihood):
        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


    def ExactMarginalLogLikelihoodTest(self, p_f, y):
        '''
        Just testing out for comparison
        '''
        # input is a multivariate gaussian and the data
        p_y = self.likelihood(p_f)
        m = p_y.mean
        K = p_y.covariance_matrix
        y = (y - m).reshape(-1,1)
        n = y.shape[0]

        L = torch.linalg.cholesky(K, upper=False) # (N, N)
        alpha = torch.linalg.solve_triangular(L.t(), torch.linalg.solve_triangular(L, y, upper=False), upper=True) # (N, 1)

        LML = -0.5 * y.t() @ alpha - torch.sum(torch.log(torch.diag(L))) - n/2*torch.log(torch.tensor(2*torch.pi))
        return LML[0,0] / n # consistent with gpytorch implementation

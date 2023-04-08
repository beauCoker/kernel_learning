# standard library imports
import os
import sys
import pdb
from functools import reduce

# package imports
import torch
import gpytorch

# local imports
from src.util import *

class DKLGP(gpytorch.models.ExactGP):
    def __init__(self, x, y, kernel, likelihood, n_hidden=20):
        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.feature_extractor = LargeFeatureExtractor(dim_in=x.shape[-1], n_hidden=n_hidden)

        self.register_prior(
            'linear1_weight_prior', 
            gpytorch.priors.NormalPrior(torch.zeros(n_hidden,1), torch.ones(n_hidden,1)),
            lambda m: m.feature_extractor.linear1.weight,
            self.closure_weight1)

        self.register_prior(
            'linear1_bias_prior', 
            gpytorch.priors.NormalPrior(torch.zeros(n_hidden), torch.ones(n_hidden)),
            lambda m: m.feature_extractor.linear1.bias,
            self.closure_bias1)

        self.register_prior(
            'linear2_weight_prior', 
            gpytorch.priors.NormalPrior(torch.zeros(2,n_hidden), torch.ones(2,n_hidden)),
            lambda m: m.feature_extractor.linear2.weight,
            self.closure_weight2)

        self.register_prior(
            'linear2_bias_prior', 
            gpytorch.priors.NormalPrior(torch.zeros(2), torch.ones(2)),
            lambda m: m.feature_extractor.linear2.bias,
            self.closure_bias2)

    def closure_weight1(self, m, v):
        # needs to take an instance of self (m) and assign v to the corresponding parameter
        m.feature_extractor.linear1.weight.data = v
        
    def closure_bias1(self, m, v):
        m.feature_extractor.linear1.bias.data = v

    def closure_weight2(self, m, v):
        m.feature_extractor.linear2.weight.data = v
        
    def closure_bias2(self, m, v):
        m.feature_extractor.linear2.bias.data = v

    def forward(self, x):
        projected_x = self.feature_extractor(x)

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


from torch import Tensor
import torch.nn.functional as F
class Linear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, input: Tensor) -> Tensor:
        '''
        if input.dim()==2:
            return F.linear(input, self.weight, self.bias)
        elif input.dim()==3:
            return torch.sum(input.unsqueeze(2) * self.weight.unsqueeze(1), -1) + self.bias.unsqueeze(1)
        '''
        if self.weight.dim() == 2:
            return F.linear(input, self.weight, self.bias)
        elif self.weight.dim() == 3:
            if input.dim()==2:
                input = input.unsqueeze(0)
            return torch.sum(input.unsqueeze(2) * self.weight.unsqueeze(1), -1) + self.bias.unsqueeze(1)



class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, dim_in, n_hidden):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', Linear(dim_in, n_hidden))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', Linear(n_hidden, 2))
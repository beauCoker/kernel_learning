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

class MKLGP(gpytorch.models.ExactGP):
    '''
    '''
    def __init__(self, x, y, kernels, likelihood):
        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = reduce(lambda a, b: a+b, kernels)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
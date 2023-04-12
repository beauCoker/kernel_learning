# standard library imports
import os
import sys
import pdb

# package imports
import torch
import gpytorch

# local imports
from src.util import *

class FilteredGP(gpytorch.models.ExactGP):
    '''
    example: kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    '''
    def __init__(self, ds, kernel, likelihood, c_init=0.0):
        super().__init__(np_to_torch(ds['train']['x']), np_to_torch(ds['train']['y']), likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

        self.train_x = np_to_torch(ds['train']['x'])
        self.test_x = np_to_torch(ds['test']['x'])
        self.ood_x = np_to_torch(ds['ood']['x'])
        self.grid_x = np_to_torch(ds['grid']['x'])

        self.n_train = self.train_x.shape[0]
        self.n_test = self.test_x.shape[0]
        self.n_ood = self.ood_x.shape[0]
        self.n_grid = self.grid_x.shape[0]

        self.idx_train = np_to_torch(ds['info']['idx_train'])
        self.idx_test = np_to_torch(ds['info']['idx_test'])
        self.idx_ood = np_to_torch(ds['info']['idx_ood'])
        self.idx_grid = np_to_torch(ds['info']['idx_grid'])


        self.version = 1
        if self.version in [1,2]:
            c_shape = (1,)
        elif self.version == 3:
            c_shape = (self.grid_x.shape[0],)

        self.register_parameter(name='raw_c', parameter=torch.nn.Parameter(c_init * torch.ones(c_shape)))
        #if c_constraint is None:
        #c_constraint = gpytorch.constraints.Positive()
        c_constraint = gpytorch.constraints.GreaterThan(lower_bound=-10.0)
        #c_constraint = None
        self.register_constraint("raw_c", c_constraint)

        #c_prior = None
        c_prior = gpytorch.priors.NormalPrior(0, 4)
        if c_prior is not None:
            self.register_prior(
                "c_prior",
                c_prior,
                lambda m: m.c,
                lambda m, v : m._set_c(v),
            )

        self.c = c_init

        self.A = dct_matrix(self.grid_x.shape[0])

    @property
    def c(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_c_constraint.transform(self.raw_c)

    @c.setter
    def c(self, value):
        return self._set_c(value)

    def _set_c(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_c)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_c=self.raw_c_constraint.inverse_transform(value))

    def forward(self, x):
        assert len(x.shape)==2 # batching not implemented
        n_x = x.shape[-2]
        if len(x.shape)>2:
            x0 = x[0,:,:]
        else:
            x0 = x

        if n_x==self.n_train and torch.all(self.train_x == x0):
            idx = self.idx_train

        elif n_x==self.n_test and torch.all(self.test_x == x0):
            idx = self.idx_test

        elif n_x==self.n_ood and torch.all(self.ood_x == x0):
            idx = self.idx_ood

        elif n_x==self.n_grid and torch.all(self.grid_x == x0):
            idx = self.idx_grid

        elif n_x==self.n_train+self.n_train and torch.all(torch.concat([self.train_x, self.train_x], 0) == x0):
            idx = torch.concat([self.idx_train, self.idx_train])

        elif n_x==self.n_train+self.n_test and torch.all(torch.concat([self.train_x, self.test_x], 0) == x0):
            idx = torch.concat([self.idx_train, self.idx_test])

        elif n_x==self.n_train+self.n_ood and torch.all(torch.concat([self.train_x, self.ood_x], 0) == x0):
            idx = torch.concat([self.idx_train, self.idx_ood])

        elif n_x==self.n_train+self.n_grid and torch.all(torch.concat([self.train_x, self.grid_x], 0) == x0):
            idx = torch.concat([self.idx_train, self.idx_grid])
            
        else:
            breakpoint()

        #try:
        if len(x.shape)>2:
            C = self.c.unsqueeze(1)*torch.eye(self.c.shape[1])
            T = self.A.t() @ C @ self.A
        else:
            if self.version==1:
                #a = a * n**(-b) * n**(-c) = a * n**(-(b+c)) <-- so larger c is faster decay, i.e. smoother functions
                c_diag = torch.arange(1,self.n_grid+1)**(-self.c) * torch.ones(self.n_grid)

            elif self.version==2:
                pass

            elif self.version==3:
                c_diag = self.c 

            T = self.A.t() @ torch.diag(c_diag) @ self.A
        #except:
            #print('Trying again...')
            #C = self.c.unsqueeze(1)*torch.eye(self.c.shape[1])
            #T = self.A.t() @ C @ self.A

        # expand grid_x as necessary
        if len(x.shape)>2:
            grid_x = self.grid_x.unsqueeze(0).repeat(x.shape[0], 1, 1)
        else:
            grid_x = self.grid_x

        mean_x_all = self.mean_module(grid_x)
        covar_x_all = self.covar_module(grid_x)

        # apply filtering
        mean_x_all = torch.matmul(mean_x_all, T.t())
        covar_x_all = torch.matmul(T, torch.matmul(covar_x_all, T.t()))

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


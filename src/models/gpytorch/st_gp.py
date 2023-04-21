# standard library imports
import os
import sys
import pdb

# package imports
import torch
import gpytorch
from pyro.distributions.multivariate_studentt import MultivariateStudentT

# local imports
from src.util import *

class GaussianLikelihoodForStudentT(gpytorch.likelihoods.GaussianLikelihood):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, m):
        assert(isinstance(m, MultivariateStudentT))
        noise = torch.sqrt(self.noise).type(m.loc.dtype) * torch.eye(m.loc.shape[0], dtype=m.loc.dtype)
        return MultivariateStudentT(m.df, m.loc, m.scale_tril + noise)

class STGP_LML(gpytorch.models.ExactGP):
    '''
    '''
    def __init__(self, x, y, nu, kernel, noise_std):
        likelihood = GaussianLikelihoodForStudentT()
        #likelihood = gpytorch.likelihoods.GaussianLikelihood() # TEMP
        likelihood.noise = noise_std**2
        likelihood.noise_covar.raw_noise.requires_grad_(False)  

        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

        nu_constraint = gpytorch.constraints.GreaterThan(lower_bound=2.0)
        nu_init = nu_constraint.inverse_transform(nu * torch.ones(1))

        self.register_parameter(name='raw_nu', parameter=torch.nn.Parameter(nu_init))
        self.register_constraint("raw_nu", nu_constraint)
    

    @property
    def nu(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_nu_constraint.transform(self.raw_nu)

    @nu.setter
    def nu(self, value):
        return self._set_nu(value)

    def _set_nu(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_nu)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_nu=self.raw_nu_constraint.inverse_transform(value))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x):
        if self.training:
            return self.forward_gaussian(x)
        else:
            return self.forward_studentt(x)

    def forward_gaussian(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def forward_studentt(self, x):
        n = x.shape[0]
        if gpytorch.settings.prior_mode.on():
            m = self.mean_module(x)
            K = self.covar_module(x)
            L = torch.linalg.cholesky(K, upper=False)
            out = MultivariateStudentT(self.nu.item(), m.type(torch.float32), L.type(torch.float32))
            #out = gpytorch.distributions.MultivariateNormal(m, K) # TEMP
        else:
            # for testing
            #mean_gp_ = self.mean_module(x)
            #cov_gp_ = self.covar_module(x)

            x1 = self.train_inputs[0]
            n1 = x1.shape[0]
            n2 = x.shape[0]
            noise = self.likelihood.noise * torch.eye(n1)

            # 1: observed, 2: testing
            with gpytorch.settings.prior_mode(True):
                m1 = self.mean_module(x1)
                K11 = self.covar_module(x1).to_dense() + noise

                K21 = self.covar_module(x, x1).to_dense()

                m2 = self.mean_module(x)
                K22 = self.covar_module(x).to_dense()

            y1 = (self.train_targets - m1).reshape(-1,1)
            nu = self.nu
            L = torch.linalg.cholesky(K11, upper=False)

            alpha = torch.linalg.solve_triangular(L.t(), torch.linalg.solve_triangular(L, y1, upper=False), upper=True) # (N, 1)
            
            v = torch.linalg.solve_triangular(L, K21.t().to_dense(), upper=False) # tranpose of k?

            mean_gp = K21 @ alpha + m2.reshape(-1,1)
            cov_gp = K22 - v.t() @ v

            # without cholesky
            #mean_gp = K21 @ torch.linalg.inv(K11.to_dense()) @ y1 + m2.reshape(-1,1)
            #cov_gp = K22 - K21 @ torch.linalg.inv(K11.to_dense()) @ K21.t()

            # add jitter
            jitter_mat = 1e-8 * torch.eye(n2)
            cov_gp = cov_gp + jitter_mat

            beta1 = y1.t() @ alpha
            cov_out = (nu + beta1 - 2) / (nu + n1 - 2) * cov_gp
            
            #breakpoint()
            scale_tril_out = torch.linalg.cholesky(cov_out, upper=False)
            out = MultivariateStudentT((nu + n1).item(), mean_gp.reshape(-1).type(torch.float32), scale_tril_out.type(torch.float32))
            #out = gpytorch.distributions.MultivariateNormal(mean_gp.reshape(-1), cov_gp) # TEMP

        return out


        #covar_x = covar_x * (self.nu-2) / (self.rho * self.covar_module.r_inv)
        #return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def ExactMarginalLogLikelihood(self, p_f, y):
        '''
        p_f should just be the GP part, so you can get m and K
        Could probably use log_prob from MultivariateStudentT...
        '''
        # input is a multivariate gaussian and the data
        #p_y = self.likelihood(p_f)
        m = p_f.mean
        K = p_f.covariance_matrix + self.likelihood.noise * torch.eye(y.shape[0])
        y = (y - m).reshape(-1,1)
        n = y.shape[0]
        nu = self.nu

        L = torch.linalg.cholesky(K, upper=False) # (N, N)
        alpha = torch.linalg.solve_triangular(L.t(), torch.linalg.solve_triangular(L, y, upper=False), upper=True) # (N, 1)

        LML = - torch.lgamma((nu + n) / 2) \
              - n/2 * torch.log((nu-2)*torch.pi) \
              - torch.lgamma(nu/2) \
              - torch.sum(torch.log(torch.diag(L))) \
              - (nu+n)/2 * torch.log(1 + y.t() @ alpha / (nu-2))

        return LML[0,0] / n # consistent with gpytorch implementation

class STGP_MCMC(gpytorch.models.ExactGP):
    '''
    '''
    def __init__(self, x, y, nu, rho, kernel, likelihood):
        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

        self.nu = torch.tensor(nu)
        self.rho = torch.tensor(rho)
        self.covar_module.r_inv = torch.nn.Parameter(torch.tensor(1.0))

        def closure_r_inv(m, v): 
            m.covar_module.r_inv.data = v

        self.register_prior(
            'r_inv_prior', 
            gpytorch.priors.GammaPrior(torch.tensor(nu/2), torch.tensor(rho/2)),
            lambda m: m.covar_module.r_inv,
            closure_r_inv)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        try:
            covar_x = covar_x * (self.nu-2) / (self.rho * self.covar_module.r_inv)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        except:
            breakpoint()

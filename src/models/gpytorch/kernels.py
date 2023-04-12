# standard library imports
import os
import sys
import pdb
import inspect

# package imports
import numpy as np
import torch
import gpytorch
from gpytorch.priors import LogNormalPrior, NormalPrior, UniformPrior
from gpytorch.constraints import Positive

# local imports


def make_kernel(**kwargs):

    # priors
    if 'ls_prior' in kwargs:
        ls_prior, ls_constraint = make_prior(kwargs['ls_prior'])
    else:
        ls_prior, ls_constraint = None, None

    if 'var_prior' in kwargs:
        var_prior, var_constraint = make_prior(kwargs['var_prior'])
    else:
        var_prior, var_constraint = None, None

    # kernel
    if kwargs['name'] == 'rbf':
        kernel = gpytorch.kernels.RBFKernel(lengthscale_prior = ls_prior, lengthscale_constraint = ls_constraint)

    elif kwargs['name'] == 'matern12':
        kernel = gpytorch.kernels.MaternKernel(nu=1/2, lengthscale_prior = ls_prior, lengthscale_constraint = ls_constraint)

    elif kwargs['name'] == 'matern32':
        kernel = gpytorch.kernels.MaternKernel(nu=3/2, lengthscale_prior = ls_prior, lengthscale_constraint = ls_constraint)

    elif kwargs['name'] == 'matern52':
        kernel = gpytorch.kernels.MaternKernel(nu=5/2, lengthscale_prior = ls_prior, lengthscale_constraint = ls_constraint)

    elif 'poly' in kwargs['name']:
        q = int(kwargs['name'][-1]) # last character assumed to be q parameter
        kernel = gpytorch.kernels.PiecewisePolynomialKernel(q=q)

    elif kwargs['name'] == 'rq':
        kernel = gpytorch.kernels.RQKernel()

    elif kwargs['name'] == 'arccos':
        kernel = ArcCosine()

    elif kwargs['name'] == 'arccosls':
        kernel = ArcCosineLS()

    # always add variance
    kernel = gpytorch.kernels.ScaleKernel(kernel,  outputscale_prior=var_prior, outputscale_constraint=var_constraint)

    # initialize
    if 'ls' in kwargs and kernel.base_kernel.has_lengthscale:
        kernel.base_kernel.lengthscale = kwargs['ls']

    if 'var' in kwargs:
        kernel.outputscale = kwargs['var']

    if 'alpha' in kwargs:
        kernel.base_kernel.lengthscale = kwargs['alpha'] # for RQKernel

    return kernel

def make_prior(name):
    if name == 'uniform':
        prior = gpytorch.priors.UniformPrior(1,2)
        constraint = gpytorch.constraints.Interval(1,2)

    elif name == 'gamma':
        prior = gpytorch.priors.GammaPrior(.1, .1)
        constraint = gpytorch.constraints.Positive()

    else:
        raise ValueError('Prior not found')

    return prior, constraint



class ArcCosineLS(gpytorch.kernels.Kernel):
    """
    The Arc-cosine family of kernels which mimics the computation in neural
    networks. The order parameter specifies the assumed activation function.
    The Multi Layer Perceptron (MLP) kernel is closely related to the ArcCosine
    kernel of order 0.
    The key reference is :cite:t:`NIPS2009_3628`.
    """

    implemented_orders = {0, 1, 2}
    is_stationary = False
    has_lengthscale = True

    def __init__(
        self,
        order = 1,
        has_lengthscale = True, 
        weight_variance_prior=None, weight_variance_constraint=None,
        bias_variance_prior=None, bias_variance_constraint=None,
        **kwargs
    ):
        """
        :param order: specifies the activation function of the neural network
          the function is a rectified monomial of the chosen order
        :param variance: the (initial) value for the variance parameter
        :param weight_variance: the (initial) value for the weight_variance parameter,
            to induce ARD behaviour this must be initialised as an array the same
            length as the the number of active dimensions e.g. [1., 1., 1.]
        :param bias_variance: the (initial) value for the bias_variance parameter
            defaults to 1.0
        :param active_dims: a slice or list specifying which columns of X are used
        """
        self.order = order
        self.has_lengthscale = has_lengthscale
        super().__init__(**kwargs)

        # register the raw parameter
        self.register_parameter(name='raw_bias_variance', parameter=torch.nn.Parameter(torch.zeros(1)))

        # set the parameter constraint to be positive, when nothing is specified
        if bias_variance_constraint is None:
            bias_variance_constraint = Positive()

        # register the constraint
        self.register_constraint("raw_bias_variance", bias_variance_constraint)

        # set the parameter prior, see
        # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
        if bias_variance_prior is not None:
            self.register_prior(
                "bias_variance_prior",
                bias_variance_prior,
                lambda m: m.bias_variance,
                lambda m, v : m._set_bias_variance(v),
            )

    # now set up the 'actual' paramter
    @property
    def bias_variance(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_bias_variance_constraint.transform(self.raw_bias_variance)

    @bias_variance.setter
    def bias_variance(self, value):
        return self._set_bias_variance(value)

    def _set_bias_variance(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_bias_variance)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_bias_variance=self.raw_bias_variance_constraint.inverse_transform(value))


    def _diag_weighted_product(self, X, weight_variance=None):
        #return tf.reduce_sum(self.weight_variance * tf.square(X), axis=-1) + self.bias_variance
        return torch.sum(weight_variance * X**2, axis=-1) + self.bias_variance


    def _full_weighted_product(self, X, X2=None, weight_variance=None):
        if X2 is None:
            #return tf.linalg.matmul((self.weight_variance * X), X, transpose_b=True) + self.bias_variance
            return torch.matmul((weight_variance * X), torch.transpose(X, -1,-2)) + self.bias_variance
        else:
            #return tf.linalg.matmul((self.weight_variance * X), X2, transpose_b=True) + self.bias_variance
            return torch.matmul((weight_variance * X), torch.transpose(X2, -1,-2)) + self.bias_variance

    def _J(self, theta):
        """
        Implements the order dependent family of functions defined in equations
        4 to 7 in the reference paper.
        """
        if self.order == 0:
            return np.pi - theta
        elif self.order == 1:
            return torch.sin(theta) + (np.pi - theta) * torch.cos(theta)
        else:
            assert self.order == 2, f"Don't know how to handle order {self.order}."
            return 3.0 * torch.sin(theta) * torch.cos(theta) + (np.pi - theta) * (
                1.0 + 2.0 * torch.cos(theta) ** 2
            )

    def forward(self, X, X2, **params):
        if self.has_lengthscale:
            weight_variance = 1/self.lengthscale**2
        else:
            weight_variance = self.weight_variance

        X_denominator = torch.sqrt(self._diag_weighted_product(X, weight_variance))
        if X2 is None:
            X2_denominator = X_denominator
            numerator = self._full_weighted_product(X, None, weight_variance)
        else:
            X2_denominator = torch.sqrt(self._diag_weighted_product(X2, weight_variance))
            numerator = self._full_weighted_product(X, X2, weight_variance)
        X_denominator = X_denominator.unsqueeze(-1)
        X2_denominator = X2_denominator.unsqueeze(-2)

        cos_theta = numerator / X_denominator / X2_denominator
        jitter = 1e-15
        theta = torch.acos(jitter + (1 - 2 * jitter) * cos_theta)
        
        out= (
            (1.0 / torch.pi)
            * self._J(theta)
            * X_denominator ** self.order
            * X2_denominator ** self.order
        )
        '''
        # Workshop version
        out = (
            self.variance
            * (0.5 / torch.pi)
            * self._J(theta)
            * X_denominator ** self.order
            * X2_denominator ** self.order
            + self.bias_variance
        )
        '''
        return out

class arc_cos:
    def __init__(self, weight_variance = 1., bias_variance = 1.):
        self.weight_variance = weight_variance
        self.bias_variance = bias_variance

    def _weighted_product(self, x, y = None):
        if x.ndim == 1: x = x.reshape(-1, 1)
        if y is None:
            return np.diag(self.weight_variance * x @ x.T +  self.bias_variance)
        if y.ndim == 1: y = y.reshape(1,-1)
        return self.weight_variance * x @ y.T + self.bias_variance

    def k(self, x, y = None):
        if x.ndim == 1: x = x.reshape(-1, 1)
        x_norm = np.sqrt(self._weighted_product(x, None))[:,np.newaxis]
        if y is None:
            y = np.copy(x)
            y_norm = np.copy(x_norm).T
        else:
            if y.ndim == 1: y = y.reshape(-1,1)
            y_norm = np.sqrt(self._weighted_product(y, None))[np.newaxis,:]
        cos_theta = self._weighted_product(x, y)/x_norm/y_norm
        jitter = 1e-15
        theta = np.arccos(jitter + (1 - 2 * jitter) * cos_theta)
        
        # workshop version:
        #return self.weight_variance*0.5/np.pi * x_norm * y_norm * (np.sin(theta) + (np.pi - theta) * np.cos(theta))+self.bias_variance

        # gptorch version:
        return 1.0/np.pi * x_norm * y_norm * (np.sin(theta) + (np.pi - theta) * np.cos(theta))



class ArcCosine(ArcCosineLS):
    """
    The Arc-cosine family of kernels which mimics the computation in neural
    networks. The order parameter specifies the assumed activation function.
    The Multi Layer Perceptron (MLP) kernel is closely related to the ArcCosine
    kernel of order 0.
    The key reference is :cite:t:`NIPS2009_3628`.
    """

    implemented_orders = {0, 1, 2}
    is_stationary = False

    def __init__(
        self,
        weight_variance_prior=None, weight_variance_constraint=None,
        **kwargs
    ):
        """
        :param order: specifies the activation function of the neural network
          the function is a rectified monomial of the chosen order
        :param variance: the (initial) value for the variance parameter
        :param weight_variance: the (initial) value for the weight_variance parameter,
            to induce ARD behaviour this must be initialised as an array the same
            length as the the number of active dimensions e.g. [1., 1., 1.]
        :param bias_variance: the (initial) value for the bias_variance parameter
            defaults to 1.0
        :param active_dims: a slice or list specifying which columns of X are used
        """
        super().__init__(has_lengthscale=False, **kwargs)

        # register the raw parameter
        self.register_parameter(name='raw_weight_variance', parameter=torch.nn.Parameter(torch.zeros(1)))

        # set the parameter constraint to be positive, when nothing is specified
        if weight_variance_constraint is None:
            weight_variance_constraint = Positive()

        # register the constraint
        self.register_constraint("raw_weight_variance", weight_variance_constraint)

        # set the parameter prior, see
        # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
        if weight_variance_prior is not None:
            self.register_prior(
                "weight_variance_prior",
                weight_variance_prior,
                lambda m: m.weight_variance,
                lambda m, v : m._set_weight_variance(v),
            )

    # now set up the 'actual' paramter
    @property
    def weight_variance(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_weight_variance_constraint.transform(self.raw_weight_variance)

    @weight_variance.setter
    def weight_variance(self, value):
        return self._set_weight_variance(value)

    def _set_weight_variance(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_weight_variance)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_weight_variance=self.raw_weight_variance_constraint.inverse_transform(value))



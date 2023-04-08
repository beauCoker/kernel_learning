import math

import numpy as np
import numpyro
import jax.numpy as jnp
from jax import random, vmap
from jax.scipy.special import logsumexp
from numpyro import handlers
from numpyro.handlers import scope
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
from numpyro.infer.util import Predictive

#from lowpass_filter import lowpass_reconstructor_matrix
#from util import *

from src.util import *

class GaussianLayer():
    def __init__(self, input_dim, output_dim, prior_var):
        super().__init__()

        self.prior_var = prior_var
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        
    def forward(self, x):
        assert(len(x.shape) >= 2)

        W = numpyro.sample('W', dist.Normal(
            jnp.zeros((self.input_dim, self.output_dim)), 
            jnp.ones((self.input_dim, self.output_dim)) * math.sqrt(self.prior_var),
        ))

        b = numpyro.sample('b', dist.Normal(
            jnp.zeros((self.output_dim)), 
            jnp.ones((self.output_dim)) * math.sqrt(self.prior_var),
        ))

        h = x @ W + b
        assert(h.shape[-2:] == (x.shape[-2], self.output_dim))

        return h


class BNN():
    def __init__(
            self, 
            architecture, 
            output_var, 
            w_prior_var,
            activation=lambda h: jnp.maximum(h, 0.0),
            scale_last_hidden_by_sqrt_width=True,
    ):
        super().__init__()

        assert(len(architecture) > 2)
        assert(output_var > 0.0)

        self.architecture = architecture
        self.activation = activation
        self.output_var = output_var
        self.w_prior_var = w_prior_var
        self.scale_last_hidden_by_sqrt_width = scale_last_hidden_by_sqrt_width

        self.hidden_layers = []
        for idx, (input_sz, output_sz) in enumerate(zip(architecture[:-1], architecture[1:])):
            self.hidden_layers.append(
                GaussianLayer(input_sz, output_sz, w_prior_var),
            )

    def sample_mean(self, x):
        assert(len(x.shape) >= 2)
        
        h = x
        for idx, layer in enumerate(self.hidden_layers):
            with scope(prefix='layer_{}'.format(idx)):
                h = layer(h)

            if idx < len(self.hidden_layers) - 1:
                h = self.activation(h)
            
            if idx == len(self.hidden_layers) - 2 and self.scale_last_hidden_by_sqrt_width:
                h = h / math.sqrt(layer.output_dim)
        
        return h        
    
    def forward(self, x, y=None):
        assert(len(x.shape) >= 2)
        assert(y is None or len(y.shape) >= 2)

        mu = numpyro.deterministic('f', self.sample_mean(x))
        p_y_given_everything = dist.Normal(mu, math.sqrt(self.output_var)).to_event(1)
        y_samples = numpyro.sample('obs', p_y_given_everything, obs=y)

        return p_y_given_everything

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


    def fit(self, x, y, n_samp, **kwargs):
        x, y = to_jnp(x, y)
        self.samples = sample_posterior(
            self, 
            x, 
            y.reshape(-1,1), 
            num_samples=n_samp, 
            warmup_steps=100,
            num_chains=1,
            verbose=True, 
            seed=0)

    def sample_f(self, x, n_samp, prior=False):
        if prior:
            y = to_np(weight_space_to_function_space(self, x, n_samp_prior=n_samp, seed=0))
        else:
            y = to_np(weight_space_to_function_space(self, x, samples=self.samples, seed=0))
        return y[:, :, 0], y[:, :, 0]

    def sample_k(self, x, n_samp, prior=False):
        return np.ones((n_samp, x.shape[0], x.shape[0]))
            

class LowPassBNN(BNN):
    def __init__(self, domain, x_train_indices, x_test_indices, *args, threshold=0.0, **kwargs):
        super(LowPassBNN, self).__init__(*args, **kwargs)

        assert(len(domain.shape) == 2)
        assert(domain.shape[-1] == 1)

        self.x_train_indices = x_train_indices
        self.x_test_indices = x_test_indices
        
        self.threshold = threshold
        self.domain = domain
        N = domain.shape[-2]
        
        self.x_train = self.domain[self.x_train_indices]
        self.x_test = self.domain[self.x_test_indices]
        
        self.lp_matrix = lowpass_reconstructor_matrix(N, int(threshold * N))
        
    def sample_mean(self, x):                
        if memoized_tensor_eq(x, self.x_train):
            indices = self.x_train_indices
        elif memoized_tensor_eq(x, self.domain):
            indices = jnp.arange(self.domain.shape[0])
        else:
            assert(memoized_tensor_eq(x, self.x_test))
            indices = self.x_test_indices
            
        mu_domain = super(LowPassBNN, self).sample_mean(self.domain)       
        assert(mu_domain.shape[-2] == self.domain.shape[-2])
        
        mu_domain_smooth = (mu_domain.squeeze() @ self.lp_matrix)
        mu_domain_smooth = mu_domain_smooth.reshape(*mu_domain_smooth.shape + (1,))
        assert(mu_domain.shape == mu_domain_smooth.shape)
        
        mu = mu_domain_smooth[indices]
        assert(mu.shape[-2] == x.shape[-2])
        
        return mu


def sample_prior_functions(model, x, num_samples=500, seed=0):
    with handlers.seed(rng_seed=seed):
        return jnp.hstack([model(x).mean for _ in range(num_samples)]).T

    
def sample_posterior(
        model, 
        x_train, 
        y_train, 
        num_samples=500, 
        warmup_steps=500,
        num_chains=1,
        verbose=True, 
        seed=0,
):
    nuts_kernel = NUTS(model.forward)
    mcmc = MCMC(
        nuts_kernel, 
        num_samples=num_samples,
        num_warmup=warmup_steps,
        num_chains=num_chains,
        progress_bar=verbose,
        chain_method='parallel',
    )
    
    mcmc.run(random.PRNGKey(seed), x_train, y=y_train)
    samples = mcmc.get_samples()

    return samples


def num_samples_in(samples):
    return samples[list(samples.keys())[0]].shape[0]


def weight_space_to_function_space(model, x, samples=None, n_samp_prior=None, seed=0):
    #num_samples = num_samples_in(samples)

    if samples is not None and n_samp_prior is None:
        predictive = Predictive(model, samples)
    elif samples is None and n_samp_prior is not None:
        predictive = Predictive(model, num_samples=n_samp_prior)
    else:
         raise ValueError('Need to specify only one of samples or n_samp_prior')
    
    y_pred = predictive(random.PRNGKey(seed), x=x)['f']

    #assert(y_pred.shape == (num_samples, x.shape[0], model.architecture[-1]))
    return y_pred



def evaluate_conditional_ll(model, samples, x, y_true, seed=0):
    num_samples = num_samples_in(samples)
    
    ll = []
    with handlers.seed(rng_seed=seed):    
        for i in range(num_samples):
            sample = {k:v[i] for k, v in samples.items()}
            
            ll.append(jnp.expand_dims(
                handlers.condition(model.forward, data=sample)(x).log_prob(y_true),
            axis=0))

    ll = jnp.vstack(ll)
    assert(ll.shape == (num_samples, x.shape[0]))

    return ll

    

def evaluate_ll(model, samples, x, y_true):
    conditional_ll = evaluate_conditional_ll(model, samples, x, y_true)
    
    ll = logsumexp(conditional_ll, axis=0) - math.log(conditional_ll.shape[0])
    assert(ll.shape == (x.shape[0],))
    
    return jnp.mean(ll).tolist()


def evaluate_mse(f_true, f_pred):
    assert(f_true.shape == f_pred.shape[1:])
    assert(len(f_true.shape) == 2)
    assert(f_true.shape[-1] == 1)

    f_pred_mean = jnp.mean(f_pred, axis=0)
    assert(f_pred_mean.shape == f_true.shape)

    mse = jnp.mean(jnp.power(f_pred_mean - f_true, 2.0))

    return mse.tolist()
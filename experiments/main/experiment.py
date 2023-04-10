# standard library imports
import time
import os
import sys
import pdb

# package imports 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wandb

# local imports
sys.path.append('../../')

#from src.util.data import load_dataset
from src.models.maker import make_model
from src.data import load_dataset
import src.plot as plot
from src.wandb import fig2img
from src.util import *
from src.metrics import *
import src.models.gpytorch.util as util_gpytorch


CONFIG_DEFAULT = {
    'ds_name': 'GP',
    'ds_n_train': 20,
    'ds_kern_name': 'matern32',
    'ds_kern_ls': 1.0,
    'ds_kern_var': 1.0,
    'ds_seed': 0,
    'ds_seed_split': 1,
    'm_name': 'std_gp',
    'm_kern_name': 'arccos',
    'm_kern_ls': 1.0,
    'm_kern_var': 1.0,
    'm_kern_var_prior': 'gamma',
    'm_kern_ls_prior': 'gamma',
    'm_inference': 'lml',
    'opt_n_samp': 10, # prior and posterior samples (for predictives and mcmc)
    'opt_n_epochs': 100,
    'opt_lr': .01,
    'noise_std': .01,
    'dim_in': 1,
    'train': False,
}
#'noise_std': 'true'
TESTRUN = False
ARGS ={
    'dir_out': './output/',
    'f_metrics': {'sqerr': sq_err},
    'k_metrics': {'fro': fro_err, 'align': align_err},
    'fdist_metrics': {'nlpd': nlpd_err, 'diagnlpd': diagnlpd_err, 'trace': trace_of_cov} # applyed to f and y distributions
}

torch.set_default_dtype(torch.float64)

def main():
    print('hey')
    if not TESTRUN:
        wandb.init(project="", config=CONFIG_DEFAULT)
        config = wandb.config
    else:
        config = CONFIG_DEFAULT

    if not os.path.exists(ARGS['dir_out']):
        os.makedirs(ARGS['dir_out'])

    res = {}

    # parse config by use
    config_m, config_ds, config_opt, config_exp = parse_config(config, 'm_', 'ds_', 'opt_')
    transfer_args(['dim_in','noise_std'], config_exp, config_m, config_ds)

    # load data
    ds = load_dataset(**config_ds)
    ds = ds_astype(ds, np.float64)
    splits = [split for split in ds.keys() if split != 'info'] # i.e. train, test, grid

    # define model
    model = make_model(ds, **config_m)

    # train
    if config_exp['train']:
        model.fit(x=ds['train']['x'], y=ds['train']['y'], **config_opt)

    res = {}
    for cond in ['prior', 'post']:
        res[cond] = {}

        # things that don't depend on split
        if config_m['inference'] == 'lml':

            # lengthscale
            if config_m['name'] == 'dkl_gp':
                print('WARNING: EXPECTED UPCROSS NOT RECOMMENDED FOR DKL')
                # to fix for DKL, need it to use forward sample
            res[cond]['upcross'] = util_gpytorch.expected_upcrossings(model.gp.covar_module)

            # smoothness
            dist = torch.distributions.uniform.Uniform(0, 1)
            x = dist.sample(torch.Size((1000,config['dim_in'])))
            k_samp, k_mean = get_k_samples(model, x, n_samp=config_opt['n_samp'], prior=cond=='prior')
            res[cond]['smooth'] = util_gpytorch.estimate_eigenvalue_decay(K=k_mean,n=1000)
        else:
            print('ONLY WORKS FOR LML FOR NOW')

        # things that do depend on split
        for split in splits:
            res_ = {}

            # f and y samples
            (f_samp, f_mean, f_cov, f_samp_mean, f_samp_cov), (y_samp, y_mean, y_cov, y_samp_mean, y_samp_cov) = get_fy_samples(model, ds[split]['x'], config_opt['n_samp'], prior=cond=='prior')

            # k samples
            k_samp, k_mean = get_k_samples(model, ds[split]['x'], n_samp=config_opt['n_samp'], prior=cond=='prior')

            # metrics

            ## f
            for metric_name, metric in ARGS['f_metrics'].items():
                res_['f_error_' + metric_name] = compute_error(metric, ds[split]['f'], f_mean).item()
                res_['f_risk_' + metric_name] = compute_risk(metric, ds[split]['f'], f_samp).item()

            for metric_name, metric in ARGS['fdist_metrics'].items():
                jitter_matrix = np.eye(ds[split]['f'].shape[0])*1e-8
                res_['fdist_error_' + metric_name] = compute_error(metric, ds[split]['f'], f_mean, f_cov + jitter_matrix).item()
                if f_samp_mean is not None and f_samp_cov is not None:
                    res_['fdist_risk_' + metric_name] = compute_risk(metric, ds[split]['f'], f_samp_mean, f_samp_cov + np.expand_dims(jitter_matrix,0)).item()

            ## y
            for metric_name, metric in ARGS['f_metrics'].items():
                res_['y_error_' + metric_name] = compute_error(metric, ds[split]['y'], y_mean).item()
                res_['y_risk_' + metric_name] = compute_risk(metric, ds[split]['y'], y_samp).item()

            for metric_name, metric in ARGS['fdist_metrics'].items():
                res_['ydist_error_' + metric_name] = compute_error(metric, ds[split]['y'], y_mean, y_cov).item()
                if y_samp_mean is not None and y_samp_cov is not None:
                    res_['ydist_risk_' + metric_name] = compute_risk(metric, ds[split]['y'], y_samp_mean, y_samp_cov).item()

            ## k
            for metric_name, metric in ARGS['k_metrics'].items():
                res_['k_error_' + metric_name] = compute_error(metric, ds[split]['k'], k_mean).item()
                if k_samp is not None:
                    res_['k_risk_' + metric_name] = compute_risk(metric, ds[split]['k'], k_samp).item()

            res[cond][split] = res_

            # 1d plots
            if split=='grid' and config_exp['dim_in'] == 1:

                ## function space
                fig, ax = plt.subplots()
                plot.plot_f(x=ds['grid']['x'], f_samp=np.expand_dims(f_samp, -1), n_samp_plot=5, ax=ax)
                ax.scatter(ds['train']['x'], ds['train']['y'], label='train')
                ax.scatter(ds['test']['x'], ds['test']['y'], label='test', alpha=.2)
                if 'ood' in ds.keys():
                    ax.scatter(ds['ood']['x'], ds['ood']['y'], label='test_OOD', alpha=.2)
                file_name = 'f_%s_%s' % (cond, split)
                
                if TESTRUN:
                    fig.savefig(os.path.join(ARGS['dir_out'], file_name + '.png'))
                else:
                    wandb.log({file_name: wandb.Image(fig2img(fig))})
    

    # metrics that depend on prior AND posterior
    for cond in ['prior', 'post']:
        for split in splits:
            for dist in ['fdist','ydist']:
                for avg in ['error', 'risk']:
                    if 'trace' in ARGS['fdist_metrics']:
                        prefix = '_'.join([dist,avg,'']) 

                        if avg=='risk' and f_samp_mean is not None and f_samp_cov is not None:
                            res[cond][split][prefix + 'contract'] = res['prior'][split][prefix + 'trace'] - res[cond][split][prefix + 'trace']
           
    # metrics that don't depend on x
    '''
    for cond in ['prior', 'post']:

        try:
            samples = model.sample_k_hypers(n_samp=config_opt['n_samp'], prior=cond=='prior')
            for key, val in samples.items():
                res[cond][key+'_err'] = compute_error(sq_err, ds['info'][key], np.mean(val)).item()
                res[cond][key+'_err'] = compute_risk(sq_err, ds['info'][key], val).item()
        except:
            print('WARNING: unable to access kernel hyperparameters')
    '''


    hyperparams = util_gpytorch.get_hyperparams(model.gp.covar_module)
    for key,val in hyperparams.items():
        res[key+'_mean'] = np.mean(val)
        res[key+'_var'] = np.var(val)


    # to flat for wandb
    if not TESTRUN:
        res_flat = to_flat_dict({}, res)
        for key, val in res_flat.items():
            wandb.summary[key] = val

    print(res)
    print('Posterior:')
    print(pd.DataFrame({k:v for k,v in res['post'].items() if k in splits}))


def get_k_samples(model, x, n_samp, prior=False):
    if hasattr(model, 'sample_k'):
        k_samp = model.sample_k(x, n_samp=n_samp, prior=prior)
        k_mean = np.mean(k_samp, 0)

    elif hasattr(model, 'predict_k'):
        k_samp = None
        k_mean = model.predict_k(x, prior=prior)

    # check
    n_obs = x.shape[0]
    check(k_mean, (n_obs, n_obs))
    if k_samp is not None:
        check(k_samp, (n_samp, n_obs, n_obs))

    return k_samp, k_mean


def get_fy_samples(model, x, n_samp, prior=False):

    # only returned for methods with sample_fy_dist
    f_samp_mean, f_samp_cov = None, None
    y_samp_mean, y_samp_cov = None, None

    # f and y samples
    if hasattr(model, 'sample_fy_dist'):
        fdist, ydist = model.sample_fy_dist(x, n_samp=n_samp, prior=prior)
        
        # f
        f_samp = fdist.sample().detach().numpy() # SxN
        f_samp_mean = fdist.mean.detach().numpy() # SxN
        f_samp_cov = fdist.covariance_matrix.detach().numpy() # SxNxN

        f_mean = np.mean(f_samp_mean, 0) # N
        f_cov = np.mean(f_samp_cov, 0) # NxN

        # y
        y_samp = ydist.sample().detach().numpy() # SxN
        y_samp_mean = ydist.mean.detach().numpy() # SxN
        y_samp_cov = ydist.covariance_matrix.detach().numpy() # SxNxN

        y_mean = np.mean(y_samp_mean, 0) # N
        y_cov = np.mean(y_samp_cov, 0) # NxN

    elif hasattr(model, 'predict_fy_dist'):
        fdist, ydist = model.predict_fy_dist(x, prior=prior)
        f_mean = fdist.mean.detach().numpy() # N
        f_cov = fdist.covariance_matrix.detach().numpy() # NxN
        f_samp = fdist.sample(torch.Size((n_samp,))).numpy() # SxN

        y_mean = ydist.mean.detach().numpy() # N
        y_cov = ydist.covariance_matrix.detach().numpy() # NxN
        y_samp = ydist.sample(torch.Size((n_samp,))).numpy()


    elif hasattr(model, 'sample_fy'):
        fdist, f_samp_mean, f_samp_cov = None, None, None
        ydist, y_samp_mean, y_samp_cov = None, None, None

        f_samp, y_samp = model.sample_f(x, n_samp=n_samp, prior=prior) # SxN
        
        # f
        f_mean = np.mean(f_samp, 0) # N
        f_cov = np.cov(f_samp.T) # NxN

        # y
        y_mean = np.mean(y_samp, 0) # N
        y_cov = np.cov(y_samp.T) # NxN

    # check shapes
    n_obs = x.shape[0]

    ## f
    check(f_samp, (n_samp, n_obs))
    check(f_mean, (n_obs,))
    check(f_cov, (n_obs, n_obs))
    if f_samp_mean is not None:
        check(f_samp_mean, (n_samp, n_obs))
    if f_samp_cov is not None:
        check(f_samp_cov, (n_samp, n_obs, n_obs))

    ## y
    check(y_samp, (n_samp, n_obs))
    check(y_mean, (n_obs,))
    check(y_cov, (n_obs, n_obs))
    if y_samp_mean is not None:
        check(y_samp_mean, (n_samp, n_obs))
    if y_samp_cov is not None:
        check(y_samp_cov, (n_samp, n_obs, n_obs))

    return (f_samp, f_mean, f_cov, f_samp_mean, f_samp_cov), (y_samp, y_mean, y_cov, y_samp_mean, y_samp_cov)

def check(arr, shape, instance=np.ndarray):
    assert arr.shape == shape
    assert isinstance(arr, instance)


if __name__ == '__main__':
    main()


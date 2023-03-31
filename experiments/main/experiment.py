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


CONFIG_DEFAULT = {
    'ds_name': 'GP',
    'ds_n_train': 80,
    'ds_kern_name': 'matern32',
    'ds_kern_ls': 1.0,
    'ds_kern_var': 1.0,
    'ds_seed': 0,
    'ds_seed_split': 1,
    'm_name': 'exact_gp',
    'm_kern_name': 'matern12',
    'm_kern_ls': 1.0,
    'm_kern_var': 1.0,
    'm_kern_var_prior': 'gamma',
    'm_kern_ls_prior': 'gamma',
    'opt_n_samp': 100, # prior and posterior samples (for predictives and mcmc)
    'opt_n_epochs': 100,
    'noise_std': .01,
    'dim_in': 1,
    'train': True,
}
#'noise_std': 'true'
TESTRUN = False
ARGS ={
    'dir_out': './output/',
    'f_metrics': {'sqerr': sq_err},
    'k_metrics': {'fro': fro_err, 'align': align_err},
    'fdist_metrics': {'nlpd': nlpd_err, 'diagnlpd': diagnlpd_err, 'trace': trace_of_cov} # applyed to f and y distributions
}

def main():
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
    model.model.double()

    # train
    if config_exp['train']:
        model.fit(x=ds['train']['x'], y=ds['train']['y'], **config_opt)

    res = {}
    for cond in ['prior', 'post']:
        res[cond] = {}

        for split in splits:
            res_ = {}

            # sample

            # f and y samples
            if hasattr(model, 'sample_fdist'):
                fdist, ydist = model.sample_fdist(ds[split]['x'], n_samp=config_opt['n_samp'], prior=cond=='prior')
                
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

            else:
                fdist, f_samp_mean, f_samp_cov = None, None, None
                ydist, y_samp_mean, y_samp_cov = None, None, None

                f_samp, y_samp = model.sample_f(ds[split]['x'], n_samp=n_samp, prior=cond=='prior').detach().numpy() # SxN

                # f
                f_mean = np.mean(f_samp, 0) # N
                f_cov = np.cov(f_samp, 0) # NxN

                # y
                y_mean = np.mean(y_samp, 0) # N
                y_cov = np.cov(y_samp, 0) # NxN

            # k samples
            k_samp = model.sample_k(ds[split]['x'], n_samp=config_opt['n_samp'], prior=cond=='prior')
            k_mean = np.mean(k_samp, 0)

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
    

    # metrics that depend on prior and posterior
    for cond in ['prior', 'post']:
        for split in splits:
            for dist in ['fdist','ydist']:
                for avg in ['error', 'risk']:
                    if 'trace' in ARGS['fdist_metrics']:
                        prefix = '_'.join([dist,avg,'']) 
                        res[cond][split][prefix + 'contract'] = res['prior'][split][prefix + 'trace'] - res[cond][split][prefix + 'trace']
           
    # metrics that don't depend on x
    for cond in ['prior', 'post']:
        samples = model.sample_k_hypers(n_samp=config_opt['n_samp'], prior=cond=='prior')
        for key, val in samples.items():
            res[cond][key+'_err'] = compute_error(sq_err, ds['info'][key], np.mean(val)).item()
            res[cond][key+'_err'] = compute_risk(sq_err, ds['info'][key], val).item()
   

    # to flat for wandb
    if not TESTRUN:
        res_flat = to_flat_dict({}, res)
        for key, val in res_flat.items():
            wandb.summary[key] = val

    print(res)
    print('Posterior:')
    print(pd.DataFrame({k:v for k,v in res['post'].items() if k in splits}))

if __name__ == '__main__':
    main()


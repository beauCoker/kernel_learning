# standard library imports
import os
import sys
import pdb
import json
import yaml
import io
from PIL import Image

# package imports
import numpy as np
import pandas as pd
import wandb

def api_get_run(entity='beaucoker', project='', run=''):
    '''
    use this to test run.history(), e.g.
    '''
    api = wandb.Api()
    run = api.run('/'.join([entity,project,run]))
    return run

def api_download_files_run(
    run,
    tables=[],
    dir_out='./wandb_downloads/'):

    dir_out_run = os.path.join(dir_out, run.name)
    if not os.path.exists(dir_out_run):
        os.makedirs(dir_out_run)

    # config
    file = run.file('config.yaml')
    file.download(root=dir_out_run, replace=True)
    
    # summary
    file = run.file('wandb-summary.json')
    file.name = 'summary.json'
    file.download(root=dir_out_run, replace=True)

    def parse_filename(name):
        name_last = name.split('/')[-1] # exclude path
        return '_'.join(name_last.split('_')[:-2]) # remove extra characters added by wandb

    '''
    # tables
    for table in tables:
        # I don't think history is the most reliable way to find files. Try run.files() instead.
        history = run.history()[table].dropna()
        path = history.to_list()[0]['path']
        file = run.file(path)
        file.name = table + '.json'
        file.download(root=dir_out_run, replace=True)
    '''

    # tables
    for file in run.files():
        filename_parsed = parse_filename(file.name)
        if filename_parsed in tables:
            file = run.file(file.name)
            file.name = filename_parsed + '.json'
            file.download(root=dir_out_run, replace=True)

def api_download_files(
    entity='beaucoker', 
    project='', 
    tables=[], 
    dir_out='./wandb_downloads/', 
    skip_already_downloaded=False, 
    max_downloads=None):

    api = wandb.Api()

    # make directory
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    # get runs 
    runs = api.runs(entity + '/' + project)

    n_downloaded = 0
    for i, run in enumerate(runs):
        
        # make directory for run
        dir_out_run = os.path.join(dir_out, run.name)
        if os.path.exists(dir_out_run) and skip_already_downloaded:
            print('skipping run %s because already downloaded' % run.name)
            continue
            
        api_download_files_run(run, tables, dir_out)
        
        n_downloaded += 1
        print('Done with run %s' % run.name)

        if max_downloads is not None:
            if n_downloaded>=max_downloads:
                print('max downloads reached')
                break


def combine_downloaded_files(dir_files='./wandb_downloads/', tables=[]):

    def parse_config(config):
        config.pop('wandb_version')
        config.pop('_wandb')
        parsed = {}
        for key, val in config.items():
            parsed[key] = val['value']
        return parsed

    def parse_summary(summary):
        summary.pop('_timestamp')
        summary.pop('_runtime')
        summary.pop('_step')
        summary.pop('_wandb')
        parsed = {}
        for key, val in summary.items():
            if not isinstance(val, dict):
                parsed[key] = val
        return parsed

    # get all folders
    folders = [name for name in os.listdir(dir_files) if os.path.isdir(os.path.join(dir_files, name))]

    # allocate space (lists of files)
    d_config = []
    d_tables = {}
    for table in tables:
        d_tables[table] = []
    d_sum = []

    for folder in folders:
        # config
        f = open(os.path.join(dir_files, folder,'config.yaml'))
        d = yaml.safe_load(f.read())
        f.close()
        
        d = parse_config(d)
        d['run'] = folder
        
        d_config.append(d)
        
        # summary
        f = open(os.path.join(dir_files, folder, 'summary.json'))
        d = json.loads(f.read())
        f.close()
        d = parse_summary(d)
        d['run'] = folder
        
        d_sum.append(d)
        
        # tables
        for table in tables:
            f = open(os.path.join(dir_files, folder, table + '.json'))
            d = json.loads(f.read())
            f.close()

            d = pd.DataFrame(d['data'], columns = d['columns'])
            d['run'] = folder
            d_tables[table].append(d)
        
    # convert to dataframes
    df_config = pd.DataFrame(d_config)
    df_sum = pd.DataFrame(d_sum)
    df_tables = {}
    for table in tables:
        df_tables[table] = pd.concat(d_tables[table]).reset_index(drop=True)

    # merge dataframes
    df_sum = df_config.merge(df_sum, on='run', suffixes=('_config', '_summary'))

    for table in tables:
        df_tables[table] = df_config.merge(df_tables[table], on='run', suffixes=('_config', '_table'))

    return df_config, df_sum, df_tables

####################
# other #
####################

def fig2img(fig):
    '''
    Convert a Matplotlib figure to a PIL Image
    '''
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

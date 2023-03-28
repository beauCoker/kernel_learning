# standard library imports
import os
import sys
import pdb

# package imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

######
# Code that isn't ready yet / is only temporarly
#####



def plot_f(x, f_samp, n_samp_plot=None, ax=None):
    '''
    x: (n, 1)
    f_samp: (n_samp, n, 1)
    '''
    if ax is None:
        fig, ax = plt.subplots()
    
    f_mean = np.mean(f_samp, 0)
    f_std = np.std(f_samp, 0)
        
    ax.plot(x, f_mean, color='tab:blue', label='model')
    ax.fill_between(x.reshape(-1), 
            f_mean.reshape(-1) - 2*f_std.reshape(-1),
            f_mean.reshape(-1) + 2*f_std.reshape(-1), color='tab:blue', alpha=.4)
    
    if n_samp_plot is not None:
        ax.plot(x, f_samp[:n_samp_plot,:,0].T, color='tab:blue', alpha=.4)
        
    return ax 
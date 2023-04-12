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



def plot_f(x, f_samp, n_samp_plot=None, f_mean=None, f_std=None, color='tab:blue', ax=None):
    '''
    x: (n, 1)
    f_samp: (n_samp, n, 1)
    '''
    if ax is None:
        fig, ax = plt.subplots()

    if f_mean is None:
        f_mean = np.mean(f_samp, 0)

    if f_std is None:
        f_std = np.std(f_samp, 0)
    
    ax.plot(x, f_mean, color=color, label='model')
    ax.fill_between(x.reshape(-1), 
            f_mean.reshape(-1) - 2*f_std.reshape(-1),
            f_mean.reshape(-1) + 2*f_std.reshape(-1), color=color, alpha=.25, linewidth=0)
    
    if n_samp_plot is not None:
        ax.plot(x, f_samp[:n_samp_plot,:,0].T, color=color, alpha=.75, linewidth=.75)
        
    return ax 
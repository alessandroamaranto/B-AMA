# -*- coding: utf-8 -*-
"""
Created on Mon May 30 17:14:52 2022

@author: AMARANTO
"""

import numpy as np
import os
import glob
from scipy.stats import norm
import matplotlib.pyplot as plt
import configparser
import pandas as pd
import seaborn as sns

def read_model_data(case_study, period):
    
    ext = '*.csv' 
    fpth = os.path.join('input', case_study, ext)
    
    fls = glob.glob(fpth)
    
    X = np.ndarray(shape = (len(fls), 1), dtype = 'object')
    
    
    
    for i in range(0, len(fls)):
        
        X[i, 0] = np.genfromtxt(fls[i], delimiter = ',', skip_header = 1)
        
    
    return(X)

def powerset(s):
    
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]
        
def exhaustive_set(cn):
    
    nCol = np.arange(cn.shape[1]-1)
    r = [x for x in powerset(nCol)]
    r.sort()

    return(r[1:])

def concat_ciclo(u, nY, ex):

    u_ex = u[:ex]
    ui = np.tile(u, nY)

    ui = np.hstack([ui, u_ex])

    return(ui)

def normalize_vector(x):
     
    xm = min(x)
    xM = max(x)
    
    y = (x - xm)/(xM - xm)
    
    return(y)


def residuals_stats(r):

    
    mu, std = norm.fit(r)
    
    return(mu, std)

def plot_ivs_forward(pg, i_s, perf, case_study):
    
    Config = configparser.ConfigParser()
    Config.read('configuration_settings.txt')
    
    # Output parameters
    output_location = 'output'                                     # Output folder
    output_name = str(Config.get('Case_study','name'))             # Case study name
    
    pg[pg<0] = 0
    fig, ax = plt.subplots(1, 2, figsize = (18,7), gridspec_kw={'width_ratios': [2.5, 1]}) 

    sc = ax[0].imshow(pg, cmap='viridis', interpolation='nearest')
    #cax = fig.add_axes([ax[0].get_position().x1+0.01,ax[0].get_position().y0,0.02,ax[0].get_position().height])
    cb = fig.colorbar(sc, ax = ax[0], location = 'bottom', shrink = 0.6)
    ax[0].set_xlabel('Input column', fontsize = 18)
    ax[0].set_ylabel('Iteration', fontsize = 18)
    ax[0].set_yticks(np.arange(pg.shape[0]), np.arange(pg.shape[0]) )

    cb.set_label('NSE [-]')
    ax[0].set_title('Forward input selection results', fontsize = 20)
    
    plot_improvement_bar(i_s, perf, ax[1])
    
    # Save
    fn = output_name + '_ivs_forward'   + '.png'
    fs = os.path.join(output_location, case_study, fn)
    
    plt.subplots_adjust(wspace = 0.15)
    
    
    
    
    fig.savefig(fs)
    
def hyper_param_plot(hyper_param, error, case_study):
    
    Config = configparser.ConfigParser()
    Config.read('configuration_settings.txt')
    
    # Output parameters
    output_location = 'output'                                     # Output folder
    output_name = str(Config.get('Case_study','name'))             # Case study name
    
    hp =  np.asarray(hyper_param)
    e1 = np.mean(error, axis = 1)
    
    fig, ax = plt.subplots(1, hp.shape[1], constrained_layout=True, figsize = (18, 5))
    
    for i in range(0, hp.shape[1]):
        
        hp_plot = np.transpose(np.vstack([hp[:, i], e1.astype(float)]))
        
        cnam = 'p_' + str(i)
        hp_plot = pd.DataFrame(hp_plot, columns = [cnam, 'RMSE_p'])
        hp_plot.iloc[:, 1] = pd.to_numeric(hp_plot["RMSE_p"])
        sss = sns.boxplot(ax = ax[i], x=cnam, y='RMSE_p', data=hp_plot)
        sss.set_xlabel(cnam, fontsize = 18)
        sss.set_ylabel('RMSE_p', fontsize = 18)
    
    # Save
    fn = output_name + '_hyper_param'   + '.png'
    fs = os.path.join(output_location, case_study, fn)
    
    fig.savefig(fs)

def plot_improvement(i_s, perf, case_study):
    
    Config = configparser.ConfigParser()
    Config.read('configuration_settings.txt')
    
    # Output parameters
    output_location = 'output'                                     # Output folder
    output_name = str(Config.get('Case_study','name'))             # Case study name
    
    fig, ax = plt.subplots()
    
    plot_improvement_bar(i_s, perf, ax)
    
    fn = output_name + '_ivs_improvements'   + '.png'
    fs = os.path.join(output_location, case_study, fn)
    
    fig.savefig(fs)
    

def plot_improvement_bar(i_s, perf, ax):
    
    is_l = i_s
    i_s = ['{:.2f}'.format(x) for x in i_s]
    
    ax.bar(i_s, perf, width = 0.4)
    ax.plot(i_s, perf, color = 'black',  linewidth=2)
    
    ax.set_xlabel('Input column', fontsize = 18)
    ax.set_xticks(i_s, np.array(is_l).astype(int).astype(str))
    ax.set_ylabel('NSE [-]', fontsize = 18)
    #ax.set_title('Cross-validation performance improvements ', fontsize = 18)
    

def plot_box(performance, case_study):
    
    Config = configparser.ConfigParser()
    Config.read('configuration_settings.txt')
    
    # Output parameters
    output_location = 'output'                                     # Output folder
    output_name = str(Config.get('Case_study','name'))             # Case study name
    
    fig, ax = plt.subplots()
    ax.boxplot(performance)
    ax.set_xlabel('Input combinations', fontsize = 18)
    ax.set_ylabel('RMSEp', fontsize = 18)
    ax.set_xticks([])
    ax.set_title('Performance variability across input space', fontsize = 20)
    
    fn = output_name + '_ivs_exaustive'   + '.png'
    fs = os.path.join(output_location, case_study, fn)
    
    
    
    
    fig.savefig(fs)
    
    
        
    
    
    

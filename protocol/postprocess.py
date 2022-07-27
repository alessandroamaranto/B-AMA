# -*- coding: utf-8 -*-
"""
Created on Mon May 30 17:30:04 2022

@author: AMARANTO
"""

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import configparser
import os
import pandas as pd
from hydroeval import evaluator, nse

class postprocess():
    
    """
    save and plot the results
    """

    def __init__(self):

        Config = configparser.ConfigParser()
        Config.read('configuration_settings.txt')
        
        self.output_name = str(Config.get('Case_study','name'))      # Case study name

    def save_forecasts(self, yc_rec, yv_rec, c, v, si, i, n_v, period, model, case_study):
        
        """
        save_forecasts(self, yc_rec, yv_rec, c, v, si, i, n_v, period, model, case_study)
        
        Save the modelling results.
        
        Input:
            - yc_rec = reconstructed training set
            - yv_rec = reconstructed test set
            - c = training set
            - v = test set
            - si = split index
            - i = file index (only if multiple files are in a single folder)
            - n_v = length of test set
            - period = time-series periodicity
            - model = selected modelling techniquqe
            - case_study = case study name
        
        Returns:
            - yr = predicted y
            - yo = observed y
            - cs, vs = training and test set index
        """
        
        # Allocate memory for observed and predicted TS
        yr = np.ndarray(shape = (yc_rec.shape[0] + yv_rec.shape[0], ))
        yo = np.ndarray(shape = (yc_rec.shape[0] + yv_rec.shape[0], ))
        
        # Index training and test set
        vs = np.arange(si*period, (si+n_v)*period)
        cs = np.delete(np.arange(0, yr.shape[0]), vs.astype(int))
        
        # Insert predicted data according to the index
        yr[vs.astype(int)] = yv_rec
        yr[cs.astype(int)] = yc_rec
        
        yo[vs.astype(int)] = v[:, v.shape[1]-1]
        yo[cs.astype(int)] = c[:, c.shape[1]-1]
        
        
        d = {'Observed': yo, 'Predicted' : yr} 
        df = pd.DataFrame(data = d)
        
        # Save output
        fn = self.output_name + '_' + str(i) + model + 'csv'
        fs = os.path.join('output', case_study, fn)
        
        df.to_csv(fs, index = False)
        
        
        return(yr, yo, vs, cs)
    
    
    def plot_forecasts(self, yr, yo, vs, cs, i, period, model, case_study, start, stop):
       
        """
        plot_forecasts(self, yr, yo, vs, cs, i, period, model, case_study, start, stop):
        
        Plot modelling results.
        
        Input:
            - yr = forecasts
            - yo = observation
            - cs = training set index
            - vs = test set index
            - i = file index (only if multiple files are in a single folder)
            - period = time-series periodicity
            - model = selected modelling techniquqe
            - case_study = case study name
            - start = initial year
            - stop = final year
        """ 
        
        self.plot_ts(yr, yo, vs, cs, i, period, model, case_study, start, stop)
        self.plot_scatter(yr, yo, vs, cs, model, case_study)
       
        
    def plot_ts(self, yr, yo, vs, cs, i, period, model, case_study, start, stop):
        
        # Define time step
        if period == 12:
            step = 'months'
        elif period == 365:
            step = 'days'
        elif period == 8760:
            step = 'hours'
        else:
            raise ValueError("system periodiciti not included")
        
        # Initialize the plot
        fig, ax = plt.subplots(constrained_layout=True, figsize=(8,6))
        
        xlb = 'Time (' + step + ')'
        
        x = np.arange(0, len(yr))
        
        # Line plot
        ax.plot(x, yo, label = 'Observed', color = '#3E065F')
        ax.plot(x, yr, label = 'Predicted', color = '#FF0075')
        
        # Scatter
        ax.scatter(x[cs.astype(int)], yr[cs.astype(int)], label = 'Calibration', color = '#69DADB')
        ax.scatter(x[vs.astype(int)], yr[vs.astype(int)], label = 'Validation', color = '#FEE440')
        
        # Legend
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=4, mode="expand", borderaxespad=0.)
        
        # Error stats
        tx = 'NSE = ' + str(np.round(evaluator(nse, yr[vs.astype(int)], yo[vs.astype(int)])[0], 2))
        tx_p = max(np.max(yo), np.max(yr))
        ax.text(0.1, tx_p, tx, fontsize = 16, weight = 'bold')
        
        # Adjust graphics
        ax.set_xlabel(xlb, fontsize = 20)
        ax.set_ylabel('y', fontsize = 20)
        
        ax.autoscale(enable=True, axis='x', tight=True)
        
        tck = np.arange(min(x), max(x), period)
        tck_l = np.arange(start, stop + 1)
        
        if len(tck) < 10:
        
            ax.set_xticks(tck)
            ax.set_xticklabels(tck_l)
        else:
            
            pl_stop = np.linspace(0, len(tck), num = 10).astype(int)
            ax.set_xticks(tck[pl_stop[:len(pl_stop)-1]])
            ax.set_xticklabels(tck_l[pl_stop[:len(pl_stop)-1]])
        
        # Save
        fn = self.output_name + '_' + str(i) + model + '.png'
        fs = os.path.join('output', case_study, fn)
        
        fig.savefig(fs)
        
    def plot_scatter(self, yr, yo, vs, cs, model, case_study):
        
        fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(12,6))
        
        X = np.vstack([yr, yo])
        lb = np.min(X)
        ub = np.max(X)
        
        ax[0].scatter(yo[cs.astype(int)], yr[cs.astype(int)], color = '#69DADB', alpha = 0.8)
        ax[0].plot(np.arange(lb, ub + 1), np.arange(lb, ub + 1))
        # Adjust graphics
        ax[0].set_xlabel('Observed', fontsize = 18)
        ax[0].set_ylabel('Predicted', fontsize = 18)
        ax[0].set_title('Training set', fontsize = 20)
        ax[0].set_xlim(lb, ub)
        ax[0].set_ylim(lb, ub)
        tx = 'NSE = ' + str(np.round(evaluator(nse, yr[cs.astype(int)], yo[cs.astype(int)])[0], 1))
        ax[0].text(lb, ub-0.03*ub, tx, fontsize = 16, weight = 'bold')
        
        ax[1].scatter(yo[vs.astype(int)], yr[vs.astype(int)], color = '#FEE440', alpha = 0.8)
        ax[1].plot(np.arange(lb, ub + 1), np.arange(lb, ub + 1))
        ax[1].set_xlabel('Observed', fontsize = 18)
        ax[1].set_ylabel('Predicted', fontsize = 18)
        ax[1].set_title('Test set', fontsize = 20)
        ax[1].set_xlim(lb, ub)
        ax[1].set_ylim(lb, ub)
        
        tx = 'NSE = ' + str(np.round(evaluator(nse, yr[vs.astype(int)], yo[vs.astype(int)])[0], 1))
        ax[1].text(lb, ub-0.03*ub, tx, fontsize = 16, weight = 'bold')
        
        
        fn = self.output_name + '_scatter_'  + model + '.png'
        fs = os.path.join('output', case_study, fn)

        fig.savefig(fs)


        


        
        

        
        
        
        
        
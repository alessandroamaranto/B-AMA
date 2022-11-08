# -*- coding: utf-8 -*-
"""
Created on Mon May 30 16:58:05 2022

@author: AMARANTO
"""

import numpy as np
import numpy.matlib
import configparser
import os
import math
from hydroeval import evaluator, nse
import protocol.utils as ut

class model_testing():
    
    """
    test the optimal models (defined by model_training) and evaluate their 
    performance in the test set
    """

    def __init__(self):
        
        Config = configparser.ConfigParser()
        Config.read(os.path.join('protocol', 'advanced_configurations.txt'))
                
        
        self.mode = str(Config.get('Data_Transformation','mode'))
    
    def de_normalize(self, xn, a, b, mn, mX):
        
        """
        de_normalize(self, xn, a, b, mn, mX)
        
        invert normalization
        
        Input: 
            - xn = normalized data
            - a = upper bound
            - b = lower bound
            - mn = minimum
            - mX = maximum
        
        Returns:
            - x = de-normalized data
        """
        
        x = mn + (((xn-b)/(a-b))*(mX-mn))
        
        return(x)
    
    def reconstruct_season(self, y,  mn, mX, period):
        
        """
        reconstruct_season(self, y,  mn, mX, period)
        
        add seasonality
        
        Input: 
            - y = normalized data
            - mn = ciclostationary average
            - mX = ciclostationary variance
            - period = periodicity of the time series
        
        Returns:
            - yr = reconstrunced data
        """
        
        # Season reconstruction parameters
        ln = y.shape[0]
        nY = math.floor(ln/period)
        ex = ln - nY*period
        
        # Ciclostationary mean and variance 
        uv = ut.concat_ciclo(mn, nY, ex)
        var_v = ut.concat_ciclo(mX, nY, ex)
         
        # Reconstruct data
        yr = y*var_v + uv
         
        return(yr)
    
        
    def reconstruct_pattern(self, y_ext_c, y_ext_v, mn, mX, period):
        
        """
        reconstruct_pattern(self, y_ext_c, y_ext_v, mn, mX, period)
        
        Reconstruct the original time-series to re-convert from data-transformation
        
        Input:
            - y_ext_c = predicted output in the training set
            - y_ext_v = predicted output in the test set
            - mn, mX = reconstruction parameters
            - period = seasonality of the time-series
        
        Output:
            - yc_theta = reconstructed training set
            - yv_theta = reconstructed test set
        """
        
        if self.mode == 'seasonal':
            yc_theta = self.reconstruct_season(y_ext_c, mn, mX, period)
            yv_theta = self.reconstruct_season(y_ext_v, mn, mX, period)
        elif self.mode == 'min_max':
            yc_theta = self.de_normalize(y_ext_c, 0.9, 0.1, mn[len(mn)-1], mX[len(mX)-1])
            yv_theta = self.de_normalize(y_ext_v, 0.9, 0.1, mn[len(mn)-1], mX[len(mX)-1])
        else:
            raise ValueError("normalization method should be either min_max or seasonal")
        
        
        return(yc_theta, yv_theta)
    
        
    def test_model(self, ms, cn, vn, column_index, mn, mX, model, period):
        
        """
        test_model(self, ms, cn, vn, column_index, mn, mX, period)
        
        Evaluate modelling performance in the test set, and returns the 
        reconstructed predicted time-series
        
        Input:
            - ms = the modelling ensamble selected through model_training
            - cn = the normalized calibration set
            - vn = the normalized validation set
            - column_index = the selected input via IVS
            - mn, mX = time series reconstruction paramaeters
            - period = periodicity or seasonality of the time-series
            
        Returns:
            - yc_rec = reconstructed training set
            - yv_rec = reconstructed test set
            - eps_c = NSE in the training set
            - eps_v = NSE in the test set
            - res = residuals
        """
        
        xc = cn[:, column_index]        # Training input
        yc = cn[:, cn.shape[1]-1]       # Training output
        
        xv = vn[:, column_index]        # Test input
        yv = vn[:, vn.shape[1]-1]       # Test output
        
        # Allocate memory for training and test output
        y_theta_v = np.ndarray(shape = (yv.shape[0], len(ms)))
        y_theta_c = np.ndarray(shape = (yc.shape[0], len(ms)))
        
        # Perform ensamble prediction
        for j in range(0, len(ms)):
            
            if model == 'ann' or model == 'svm': 
            
                y_theta_v[:, j] = ms[j].predict(xv)
                y_theta_c[:, j] = ms[j].predict(xc)
                
            else:
                try:
                    modulename = model + '_module'
                    new_module = __import__(modulename)
                    
                    y_theta_v[:, j] = new_module.test_module(ms[j], xv)
                    y_theta_c[:, j] = new_module.test_module(ms[j], xc)
                    
                except:
                    
                    raise ValueError('No module for model' + model + 'specified')
                
            
        # Aggregate the forecast
        y_ext_v = np.mean(y_theta_v, axis = 1)
        y_ext_c = np.mean(y_theta_c, axis = 1)
        
        # Conpute the residuals
        res = yc - y_ext_c
        
        # Reconstruct time-series
        yc_rec, yv_rec = self.reconstruct_pattern(y_ext_c, y_ext_v, mn, mX, period)
        yc_o_rec, yv_o_rec = self.reconstruct_pattern(yc, yv, mn, mX, period)
        
        # Compute the error statistics
        eps_v = evaluator(nse, yv_rec, yv_o_rec)[0]
        eps_c = evaluator(nse, yc_rec, yc_o_rec)[0]
        
        return(yc_rec, yv_rec, eps_c, eps_v, res)
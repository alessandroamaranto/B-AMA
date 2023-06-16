# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:39:01 2022

@author: AMARANTO
"""

import numpy as np
import os
import math
import configparser
import protocol.utils as ut

class data_transformation():
    
    """
    data_tranformation prepares the data for the following step by either removing
    the seasonal component or normalizing them in the range 0-1.
    User-defined options, to be specified in the norm_config.txt:
        - min_max to normalize in the range 0-1 (default)
        - seasonal to remove the seasonal component
    """
    
    def __init__(self):

        Config = configparser.ConfigParser()
        Config.read(os.path.join('protocol', 'advanced_configurations.txt'))
                
        self.mode = str(Config.get('Data_Transformation','mode')) # user-defined transformation 
        self.ma_f = int(Config.get('Data_Transformation','ma_f')) # user-defined transformation

    def normalize(self, c, v):
        
        """
        normalize(self, c, v)
        
        Normalize the data in the range 0.1-0.9.
        
        Input:
            - c = training set
            - v = test set
        Returns:
            - cn = normalized training set
            - vn = normalized test set
            - mn = minimum (for time series reconstruction)
            - mX = maximum (for time series reconstruction)
        
        """
        
        # compute minimum and maximum of each column
        mn = np.min(c, axis = 0)
        mX = np.max(c, axis = 0)
        
        # allocate memory for output data-sets
        cn = np.empty(c.shape)
        vn = np.empty(v.shape)
        
        # Normalize
        for i in range(0, cn.shape[1]):
            
            cn[:, i] = ((0.8*(c[:, i]-mn[i]))/(mX[i]-mn[i]))+0.1
            
            try:
                vn[:, i] = ((0.8*(v[:, i]-mn[i]))/(mX[i]-mn[i]))+0.1
            except:
                print('no' + str(i) + 'th column, check if this operation is carried in forecast mode')
        
        return(cn, vn, mn , mX)
    
    
    def moving_average(self, ci, nY, f, period):
        
        """
        moving_average(self, ci, nY, f, period)
        
        Compute the moving average of a time series to remove noise.
        
        Input:
            - ci = ith column of the training set
            - nY = number of periods
            - f = moving window length
            - period = time-series periodicity
        Returns:
            - mi = moving average (size = len(period))
            - m = moving average (size = len(ci))
        """
        
        # reshape the columns for it to be period, nY
        ci = np.reshape(ci, newshape = (period, nY), order = 'F')
        shp = ci.shape 
            
        # Build the matrix to compute the moving average
        Y_up = np.c_[
            ci[shp[0]-f:shp[0], shp[1]-1],
            ci[shp[0]-f:shp[0], :shp[1]-1]
                          ]
        
        Y_d = np.c_[
            ci[:f, 1:shp[1]],
            ci[:f, 0]
            ]
        
        Y = np.concatenate(
            (Y_up, ci, Y_d)
            )
        
        # allocate memory for moving average
        mi = np.empty(shape = period)
        
        # compute moving average
        for k in range(0, period):
            
            mi[k] = np.mean(Y[k:k+2*f, :])
        
        # repeat data for the same length of the training set.
        mm = np.tile(mi, nY )
        
        return(mi, mm)
            
    
    
    def remove_season(self, c, v, period):
        
        """
        remove_season(self, c, v, period)
        
        Remove seasonality from data, such that z = (x - ut)/sigmat, 
        where ut is the ciclostationary average with moving windowm and
        sigmat is the ciclostationary variance.
        
        Input:
            - c = training set
            - v = test set
            - period = time-series periodicity
        Returns:
            - c_s = de-seasonalized training set
            - v_s = de-seasonalized test set
            - uc = ciclostationary mean
            - var_c = ciclostationary variance
        """
        
        
        # Get the integers of the number of years in the training set
        nY = math.floor(c.shape[0]/period)
        ex = c.shape[0] - nY*period
        
        # Truncate training set and allocate memory for de-sesonalized training
        c_trunc = c[:c.shape[0]-ex,:]
        c_s = np.ndarray(shape = c.shape)
        
        # Get the integers of the number of years in the training set
        nY_v = math.floor(v.shape[0]/period)
        ex_v = v.shape[0] - nY_v*period
        
        # Allocate memory for de-seasonalized test set
        v_s = np.ndarray(shape = v.shape)
        
        # De-seasonalize each column
        for i in range(0, c_trunc.shape[1]):
            
            ci = c_trunc[:, i]
            
            # Compute ciclostazionary moving average and variance
            uc, uci = self.moving_average(ci, nY, self.ma_f, period)
            var_c, var_ci = self.moving_average( np.power(ci-uci, 2), nY, self.ma_f, period)
            
            var_c = np.sqrt(var_c)
            
            # Concatenate for the extra days
            uc_i = ut.concat_ciclo(uc, nY, ex)
            var_ci = ut.concat_ciclo(var_c, nY, ex)
            
            # Normalize data
            c_s[:, i] = (c[:, i] - uc_i)/var_ci
            
            # Concatenate for the extra days in the test set (using cyclo mean 
            # and variance computed in the training set)
            uv = ut.concat_ciclo(uc, nY_v, ex_v)
            var_v = ut.concat_ciclo(var_c, nY_v, ex_v)
            
            # Transform test set
            v_s[:, i] = (v[:, i] - uv)/var_v
        
        return(c_s, v_s, uc, var_c)
        
   
    
    def transform_data(self, c, v, period):
        
        """
        transform_data(self, c, v, period)
        
        Tranform the data accordig to the user's choice
        
        Input:
            - c = training set
            - v = test set
            - period = time-series periodicity
        Returns:
            - cn = normalized training set
            - vn = normalized test set
            - mn = minimum (for time series reconstruction)
            - mX = maximum (for time series reconstruction)
        """
        
        # Switch and normalize
        if self.mode == 'seasonal':
            cn, vn, mn, mX = self.remove_season(c, v, period)
        elif self.mode == 'min_max':
            cn, vn, mn, mX = self.normalize(c, v)
        else:
            raise ValueError("normalization method should be either min_max or seasonal")
        
        return(cn, vn, mn, mX)

        
    
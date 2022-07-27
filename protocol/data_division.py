# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:41:27 2022

@author: AMARANTO
"""

import numpy as np
import os
import configparser

import protocol.utils as ut

class data_division():
    
    """
    data_division performs the split between training and test set.
    At the current state, two methods are employed:
        - Optimal split (default): as described by Amaranto et al; (2020)
        - User defined split 
    To change the default configurations, switch the split_custom parameter in the 
    split_config.txt configuration file and specify the index of the first period in
    the test set.
    """

    def __init__(self):

        Config = configparser.ConfigParser()
        Config.read(os.path.join('protocol', 'advanced_configurations.txt'))
                
        # Split between calibration and validation parameters
        self.custom_split = int(Config.get('Data_Division','split_custom')) # Default: 0, optimal split is performed
        
        # If custom split == 1, then specify the firts period to be included in the validation set
        self.split_index = np.array([e.strip() for e in Config.get('Data_Division',
                                                                   'split_period').split(',')]).astype(
                                                                       int)    
    def compute_split_stats(self, yc, yv):
        
        """
        compute_split_stats(self, yc, yv)
        
        Compute the objective function value for the t-th split.
        The objective function is defined as s_mean + s_sd, where:
            - s_mean is the squared difference between the training and the test
                set mean
            - s_sd is the quared difference between the training and the test
                set standard deviation.
        
        Input:
            - yc: training set - output
            - yv: test set - output
        Returns:
            - d: objective function value (to be minimized)
        """
        
        s_mean = np.power(np.mean(yc) - np.mean(yv), 2)  
        s_sd = np.power(np.power(np.var(yc) - np.var(yv), 2), 0.5)
        
        d = s_mean + s_sd
        
        return(d) 
    
    def split_data(self, x, nY, n_v, period):
        
        """
        split_data(self, x, nY, n_v, period)
        
        Iteratively split the dataset into training and validation, and extracts
        the split minimizing the objective function d, defined as the sum between the
        normalized average difference and the normalized standard deviation difference
        computed between the training and test set.
        For further information, consult Amaranto et al., 2020 (https://doi.org/10.1016/j.jhydrol.2020.124957)
        
        Input:
            - x: the whole dataset
            - nY: number of periods
            - n_v: number of periods in the test set
            - period: time series periodicity
        Returns:
            - yS: optimal starting period in the test set
        """
        
        # Extract dependent variable and normalize
        xi = x[:, x.shape[1]-1]
        y = ut.normalize_vector(xi)
        
        # Initialize the objective function
        dummy_d = np.Inf
        
        # Iterate across all possible splits
        for t in range(0, nY - n_v + 1):
            
            # Define iteration-specific training and test set
            cut = np.arange(t*period, (t + n_v)*period)
            yv = y[cut]
            yc = np.delete(y, cut)
            
            # Compute objective function
            d = self.compute_split_stats(yc, yv)
            
            # Extract optimal split
            if d < dummy_d:
                
                dummy_d = d
                yS = t
                
        return(yS)
            
    def optimal_split(self, x, period, i):
        
        """
        optimal_split(self, x, period, i)
        
        split data between training and test set according to the user's choice.
        
        Input: 
            - x = the whole dataset
            - period =  periodicity of the time series.
            - self.split_index = if custom split == 1, the first period in the
                test set
            - i = time series index (only if multiple files are in the 'input/
                                     case-study folder, otherwise always 0')
        Returns:
            - oS = index of the firt period to be included in the test set
            - n_v = test set length
        """
        
        nD = x.shape[0] # number of time steps
        nY = int(nD/period) # number of periods
        n_v = round((x.shape[0]/period)*0.3, 0) # test set length
        n_v = int(n_v) # convert from float to int
        
        # Compute optimal or custom-defined split
        if self.custom_split:
            oS = self.split_index[i]
            return(oS, n_v)
        else:
            oS = self.split_data(x, nY, int(n_v), period)
            
            return(oS, n_v)
        
    def split(self, xi, si, period, n_v):
        
        
        """
        split(self, xi, si, period, n_v)
        
        split data between training and test set.
        Input: 
            - xi = the whole dataset
            - si = index of the first period to be included in the test set
            - period =  periodicity of the time series.
            - n_v = number of periods to be included in the test set
        Returns:
            - c = training set
            - v = test set
        """
        
        # Create index of test set data
        cut = np.arange(si*period, (si + n_v)*period)
        
        # Extract index-corresponding data for test, delete for training
        v = xi[cut.astype(int), :]
        c = np.delete(xi, cut.astype(int), axis = 0)
        
        return(c, v)
    
    
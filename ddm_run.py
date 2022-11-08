# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 10:29:34 2021

@author: AMARANTO
"""

import numpy as np
import configparser
import dill
import os
import protocol.ivs as ivs
import protocol.data_division as dd
import protocol.data_transformation as dt
import protocol.model_training as mt
import protocol.model_testing as mte
import protocol.utils as ut
import protocol.postprocess as pp

class B_AMA():
    
    """
    B_AMA employes all the fundamental blocks to develop data driven models for
    hydrological time-series.
    In particular:
        - Data division and transformation;
        - Input variable selection
        - Model calibration and k-fold cross-validation,
            with optimization of model architecture and hyper-parameters
        - Model testing and performance assessment
        - Visual analytics for results presentation
    """
    

    def __init__(self):
        
        
        # Import configuration settings file
        Config = configparser.ConfigParser()
        
        Config.read('configuration_settings.txt')
        
        # Case study
        self.case_study = str(Config.get('Case_study','name'))

        # Input periodicity
        self.period = int(Config.get('Data_prop','period')) 
        self.start = int(Config.get('Data_prop','start'))
        self.end = int(Config.get('Data_prop','end'))
        
        # Model type
        self.model = str(Config.get('ddm','model'))
        
    
    def protocol_run(self):
        
        """
        protocol_run(self)
        
        Develops the step for implementing the DDM.
        
        Input:
            self:
                - case_study: name of the input folder where data are stored
                - period: system periodicity
                - start: initial year
                - end: final year
                - model: ddm to be trained
        
        Returns:
            results:
                 - eps_c = Nash-Sutcliffe efficiency index in the training set
                 - eps_v = Nash-Sutcliffe efficiency index in the test set
                 - columns_selected = input variables selected in the final model
                     architecture
                 - ms = the models
        """
        
        # Initialize the 'result' class
        class Object(object):
            pass
        
        results = Object()

        # Import data
        X = ut.read_model_data(self.case_study, self.period)
        
        # Allocate memory for the results
        eps_v = np.empty(X.shape[0]) # validation error
        eps_c = np.empty(X.shape[0]) # calibration error
        column_index = np.ndarray(shape = (X.shape[0], ), dtype = 'object') # variable selected
        ms = np.ndarray(shape = (X.shape[0], ), dtype = 'object') # model optimal architectures
        
        # Protocol start: iterate along the number of time series for which the forecast is necessary
        for i in range(0, X.shape[0]):
            
            xi = X[i, 0]
            
            # Data division
            si, n_v = dd.data_division().optimal_split(xi, self.period, i)
            c, v = dd.data_division().split(xi, si, self.period, n_v)
            
            # Data normalization
            cn, vn, mn, mX = dt.data_transformation().transform_data(c, v,
                                                                     self.period)
            
            # Input variable selection
            column_index[i] = ivs.input_variable_selection().select_input(cn, self.case_study)
            
            # Training and testing
            ms[i] = mt.model_training().train_model(cn, column_index[i],
                                                    self.model, self.case_study)
            
            yc_rec, yv_rec, eps_c[i], eps_v[i], res = mte.model_testing().test_model(ms[i],
                                                                                     cn,
                                                                                     vn,
                                                                                     column_index[i],
                                                                                     mn,
                                                                                     mX,
                                                                                     self.model,
                                                                                     self.period)
            
            # Save the forecasts and plot the results
            yr, yo, vs, cs = pp.postprocess().save_forecasts(yc_rec, yv_rec, c, v,
                                                             si, i, n_v, self.period,
                                                             self.model, self.case_study)
            
            pp.postprocess().plot_forecasts(yr, yo, vs, cs, i, self.period, self.model,
                                            self.case_study,
                                            self.start,
                                            self.end)
        
        # Fill the 'results' class
        results.calibration_error = eps_c
        results.validation_error = eps_v
        results.columns_selected = column_index
        results.models = ms
        
        try:
            # Save as pkl element
            fpt = os.path.join('output', self.case_study, 'results.pkl')
            with open(fpt, 'wb') as f:
                dill.dump(results, f)
        except:
            print('The selected module does not allow to save as pkl')

        # Return
        return(results)

results = B_AMA().protocol_run()




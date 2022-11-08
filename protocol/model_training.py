# -*- coding: utf-8 -*-
"""
Created on Mon May 30 16:09:53 2022

@author: AMARANTO
"""

import numpy as np
import math
import numpy.matlib
import configparser
import os
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import itertools

import protocol.utils as ut

class model_training():
    
    """
    model_training select the optimal model architecture and the optimal hyper-
    parameters value based on a k-fold cross-validation:
    """

    def __init__(self):

        Config = configparser.ConfigParser()
        Config.read(os.path.join('protocol', 'advanced_configurations.txt'))
        
        self.n_folds = int(Config.get('IVS','n_split'))
        
        
    def gen_hyperparam_space(self, model):
        
        """
        gen_hyperparam_space(self, model)
        
        Generate the hyper-parameter space for the selected model.
        
        Input:
            model = the selected model
        
        Returns:
            hyper param = the models candidate hyper parameters:
                ANN {activation function, number of neurons, number of iterations, 
                     alfa and learning rate}
                SVM {kernel type, eps, C and tolerance}
        """
        Config = configparser.ConfigParser()
        Config.read(os.path.join('protocol', 'advanced_configurations.txt'))
        
        # Read the hyper parameters set from the configuration settings file
        if  model == 'svm':
            krn = np.array([e.strip() for e in Config.get('Model_Training', 'krn').split(',')]).tolist()
            eps = np.array([e.strip() for e in Config.get('Model_Training', 'n_eps').split(',')]).tolist()
            nodes = np.array([e.strip() for e in Config.get('Model_Training', 'n_C').split(',')]).tolist()
            tol = np.array([e.strip() for e in Config.get('Model_Training', 'n_tol').split(',')]).tolist()
            
            # Generate hyperparameters cartesian product
            ls = [krn, eps, nodes, tol]
            hyper_param = []
            
            for element in itertools.product(*ls):
                hyper_param.append(element)
                
        
        elif model == 'ann':
            af = np.array([e.strip() for e in Config.get('Model_Training', 'activation').split(',')]).tolist()
            n_neu = np.array([e.strip() for e in Config.get('Model_Training', 'neurons').split(',')]).tolist()
            n_it = np.array([e.strip() for e in Config.get('Model_Training', 'iter').split(',')]).tolist()
            alpha = np.array([e.strip() for e in Config.get('Model_Training', 'alfa').split(',')]).tolist()
            learn = np.array([e.strip() for e in Config.get('Model_Training', 'learning').split(',')]).tolist()
            
            # Generate hyperparameters cartesian product
            ls = [af, n_neu, n_it, alpha, learn]
            hyper_param = []
            
            for element in itertools.product(*ls):
                hyper_param.append(element)
        
        else:
            
            try:
                Config.read(os.path.join('protocol', model + '_module_config.txt'))
                n_param = int(Config.get('dimensionality', 'n_param'))
                
                ls = []
                for c in range(n_param):
                    p_name = 'p' + str(c)
                    ls.append(np.array([e.strip() for e in Config.get('param_values', p_name).split(',')]).tolist())
                
                hyper_param = []
                
                for element in itertools.product(*ls):
                    hyper_param.append(element)
                    
                    
                
            except:
                print('warning, no model-specific configuration settings file specified')
        
        return(hyper_param)
            
        
    def train_model_node(self, xc, yc, hp, model):
        
        """
        train_model_node(self, model)
        
        Given a hyper-parameter configuration, trains k-models (one per each fold)
        
        Input:
            - xc = input subset
            - yc = output
            - hp = the hyper-parameters combination
            - model = the selected model
            
        Returns:
            - models_node = the trained model
            - error_node = the error computed in each fold 
        """
        
        # Allocate memory for the outputs
        models_node = np.ndarray(shape = (self.n_folds, ), dtype = 'object')    # Trained models
        errors_node = np.ndarray(shape = (self.n_folds, ))                      # Corresponding error
        
        # Compute the length of each fold
        l_fold = int(xc.shape[0]/self.n_folds)

        
        for f in range(0, self.n_folds):
            
            cut = np.arange(f*l_fold, (f+1)*l_fold)
            
            # Prepare training and cv data
            x_tr = np.delete(xc, cut.astype(int), axis = 0)
            y_tr = np.delete(yc, cut.astype(int), axis = 0)
            
            if xc.shape[1] > 1:
                x_cv = xc[cut.astype(int), :]
            else:
                x_cv = xc[cut.astype(int)]
                
            y_cv = yc[cut.astype(int)]
            
            # Train the models
            if model == 'svm':
            
                ddm = SVR(kernel=hp[0],
                        gamma='auto',
                        coef0=0.0,
                        tol=float(hp[3]),
                        C=float(hp[2]),
                        epsilon = float(hp[1]),
                        shrinking=True,
                        cache_size=200,
                        verbose=False,
                        max_iter=-1)
                models_node[f] = ddm.fit(x_tr, y_tr)
                y_theta = models_node[f].predict(x_cv)
            
            elif model == 'ann':
                
                ddm = MLPRegressor(hidden_layer_sizes = (int(hp[1])),  
                                   activation = hp[0],
                                  learning_rate = hp[4], 
                                  max_iter = int(hp[2]),
                                  alpha = float(hp[3])
                                )
                models_node[f] = ddm.fit(x_tr, y_tr)
                y_theta = models_node[f].predict(x_cv)
            else:
                try:
                    modulename = model + '_module'
                    new_module = __import__(modulename)
                    
                    models_node[f], y_theta = new_module.train_module(x_tr, x_cv, y_tr, hp)
                    
                except:
                    
                    raise ValueError('No module for model' + model + 'specified')
                
            # Compute the error in each fold
            
            errors_node[f] = math.sqrt(mean_squared_error(y_theta, y_cv))
            
        
        return(models_node, errors_node)
        
    
    def train_model(self, cn, column_index, model, case_study):
        
        """
        train_model(self, cn, column_index, model)
        
        Iterate across the hyper parameters space and models architectures, to select
        the optimal one.
        
        Input:
            - cn = calibration set
            - column_index = selected input from IVS
            - model = the selected modelling techinique
            
        Returns:
            - ms = the selected optimal modelling ensamble
        """
        
        # Generate hyper parameters space
        hyper_param = self.gen_hyperparam_space(model)
        
        # Allocate memory for models and errors
        models = np.ndarray(shape = (len(hyper_param), self.n_folds), dtype = 'object')
        error = np.ndarray(shape = (len(hyper_param), self.n_folds))
        
        # Divide input and output
        xc = cn[:, column_index]
        yc = cn[:, cn.shape[1]-1]
        
        # Iterate across the hyper parameter space
        for i in range(0, len(hyper_param)):
            
            models[i, :], error[i, :] = self.train_model_node(xc, yc, hyper_param[i], model)
        
        # Get the index of the minimum error for each fold
        idx = np.argmin(error, axis = 0)
        
        # Extract the optima model and corresponding error
        ms = np.ndarray(shape = (len(idx) ,), dtype = 'object' )
        eps = np.ndarray(shape = (len(idx) ,))
        
        ut.hyper_param_plot(hyper_param, error, case_study)
        for i in range(0, models.shape[1]):
    
            ms[i] = models[idx[i], i]
            eps[i] = error[idx[i], i]
        
        
        return(ms)
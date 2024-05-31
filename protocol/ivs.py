# -*- coding: utf-8 -*-
"""
Created on Mon May 30 12:32:18 2022

@author: AMARANTO
"""

import numpy as np
import os
import math
import numpy.matlib
import configparser
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from hydroeval import evaluator, nse
from sklearn.svm import SVR

import protocol.utils as ut

class input_variable_selection():
    
    """
    input_variable_selection select the predictors for the data-driven models
    Three IVS methods are currently employed (below the method and the keyword 
                                              to be specified in ivs_config.txt):
        - forward selection (forward_selection) - default;
        - model-based correlation (correlation);
        - exhaustive search (exhaustive)
    """
    
    
    def __init__(self):

        Config = configparser.ConfigParser()
        Config.read(os.path.join('protocol', 'advanced_configurations.txt'))
                
        # Ivs param
        self.ivs_mode = str(Config.get('IVS','ivs_method'))    # IVS method
        self.np_max = int(Config.get('IVS','max_predictor'))   # Maximum number of predictors
        self.tol_max = float(Config.get('IVS','max_tol'))      # Tolerance stopping criteria
        self.n_folds = int(Config.get('IVS','n_split'))        # Number of folds in cross-validation
        
    
    def run_model_fold(self, xc, yc):
        
        """
        run_model_fold(xc, yc)
        
        Trains k (where k = number of folds) SVM models, and returns the average
        RMSE and NSE across the folds.
        
        Input:
            - xc = input subset
            - yc = output
            - n.folds = number of folds in the cross-validation
            
        Returns:
            - err = average RMSE across the folds
            - err_NS = average NSE across the folds 
        """
        
        err = 0                                     # RMSE
        err_NS = 0                                  # NSE
        l_fold = int(xc.shape[0]/self.n_folds)      # Length of each fold
        
        # Iterate across folds
        for f in range(0, self.n_folds):
            
            # Extract input and output training data for the k-th fold
            cut = np.arange(f*l_fold, (f+1)*l_fold)
            
            x_tr = np.delete(xc, cut.astype(int), axis = 0)
            y_tr = np.delete(yc, cut.astype(int), axis = 0)
            
            # Extract input and output cv data for the k-th fold
            if xc.shape[1] > 1:
                x_cv = xc[cut.astype(int), :]
            else:
                x_cv = xc[cut.astype(int)]
                
            y_cv = yc[cut.astype(int)]
            
            # Train the SVM model
            ddm = SVR(kernel='rbf',
                    gamma='auto',
                    coef0=0.0,
                    tol=0.01,
                    C=3,
                    shrinking=True,
                    cache_size=200,
                    verbose=False,
                    max_iter=-1)
            
            mod = ddm.fit(x_tr, y_tr)
            y_theta = mod.predict(x_cv)
            
            # Compute the error metrics
            err =  err + math.sqrt(mean_squared_error(y_theta, y_cv))
            err_NS = err_NS + evaluator(nse, y_theta, y_cv)[0]

        # Extract the average    
        err = err/self.n_folds
        err_NS = err_NS/self.n_folds
        
        return(err, err_NS)
    
    def ivs_correlation(self, cn, case_study):
        
        """
        ivs_coorelation(self, cn)
        
        Iterative procedure, model-based. Input candidates are ranked based on the 
        cross-correlation with the output. Each predictor is iteratively added 
        to the input subset according with the ranking. 
        The procedure stops when either no further improvements are achieved or 
        the maximum number of predictors is reached.
        
        Input:
            - cn = training set
            - tol_max = objective function improvement for stopping criteria
            - np_max = maximum number of predictors
            
        Returns:
            - column index(es) of the selected predictor(s)
        """
        
        performance = []
        
        # Compute cross-correlation and sort input candidates
        c_mat = np.corrcoef(cn, rowvar = False)[cn.shape[1]-1, :cn.shape[1]-1]
        prev = c_mat.argsort()[::-1]
        
        n_pr = 0
        
        # Iterate through predictors
        while n_pr < self.np_max:
            
            yc = cn[:, cn.shape[1]-1]
            xc = cn[:, prev[0:n_pr+1]]
            
            performance.append(self.run_model_fold(xc, yc)[1])
            
            # Check for stopping criteria
            if n_pr > 0:
                tol = performance[n_pr] - performance[n_pr-1]
            
                if tol < self.tol_max:
                    
                    n_pr -= 1
                    break
            
            n_pr += 1
        
        ut.plot_improvement(prev[0:n_pr+1], np.array(performance[0:n_pr+1]), case_study)
        return(prev[0:n_pr+1])
    
    def ivs_forward_selection(self, cn, case_study):
        
        """
        ivs_forward_selection(self, cn)
        
        Iterative procedure. Trains n = cn.shape[1] -1 SISO models, and extracts
        the predictor maximising some performance criteria. It then trains n-1 model,
        combining the selected predictor with each of the remaining candidates.
        The procedure stops when no further improvements are achieved.
        
        Input:
            - cn = training set
            - tol_max = objective function improvement for stopping criteria
            
        Returns:
            - column index(es) of the selected predictor(s)
        """
        
        # forward_selection parameters
        k = 0                       # counter     
        n_max = cn.shape[1] -1      # maximum number of predictor
        i_s = []                    # selected columns
        p_box = []                  # improvement vector
        
        # Iterate across all predictors
        while k < cn.shape[1] -1:
            
            # Train n SISO models, and select the best predictor
            if k == 0:
                
                performance = []

                for i in range(0, n_max):
                    
                    yc = cn[:, cn.shape[1]-1]
                    xc = cn[:, [i]]
                    
                    performance.append(self.run_model_fold(xc, yc)[1])
                
                # Add to the selected input set the best predictor, save the performance
                i_s.append(np.argmax(performance))
                p_max = np.max(performance)
                p_box.append(p_max)
                p_graph = np.array(performance)
                k = k + 1
            
            else:
                
                
                # Train n-len(i_s) MISO models ultis stopping criteria is met
                performance = []
                
                var_range = np.arange(n_max)
                var_range = np.delete(var_range, i_s)
                
                # Iterate along the remaining candidates
                for i in var_range:
                    
                    yc = cn[:, cn.shape[1]-1]       # Output
                    e_c = np.hstack([i, i_s])       # Candidate input index
                    xc = cn[:, e_c]                 # Candidate input set
                    
                    # Train the model and save input candidate performance
                    performance.append(self.run_model_fold(xc, yc)[1])
                
                k = k + 1
                
                # Extract the MISO best performance and performance index
                m_i = np.argmax(performance)
                p_max_new = np.max(performance)
                
                # Add new line to performance matrix
                pn = np.empty(n_max)
                pn[i_s] = float('nan')
                
                m = 0
                for j in range(0, n_max):
                    
                    if j not in i_s:
                        pn[j] = performance[m]
                        m = m + 1
                
                p_graph = np.vstack([p_graph, pn])
                
                # Check if stopping criteia is met
                if p_max_new - p_max > self.tol_max:
                    
                    
                    p_max = p_max_new
                    p_box.append(p_max)
                    i_s.append(var_range[m_i])        
                
                else:
                    break
        
        # Plot IVS results
        ut.plot_ivs_forward(p_graph, i_s, p_box, case_study)
        
        return(i_s)
                    
    
    def ivs_exhaustive_search(self, cn, case_study):
        
        """
        ivs_exhaustive_search(self, cn)
        
        Tries all the possible input combination, extracts the one minimizing the
        RMSE in the cross-validation set. 
        
        Input:
            - cn = training set
            
        Returns:
            - column index(es) of the selected predictor(s)
        """
        
        # Generate set of all possible input combination
        i_sp = ut.exhaustive_set(cn)
        performance = np.empty(len(i_sp))
        
        # Iterate through combinations
        for j in range(0, len(i_sp)):
                
            i_ss = i_sp[j]
            
            xc = cn[:, i_ss]
            yc = cn[:, cn.shape[1]-1]
            
            # Test the candidate input and return the RMSE
            performance[j] = self.run_model_fold(xc, yc)[0]
        
        # Extract the best performance
        idx = np.argmin(performance).astype(int)
        ut.plot_box(performance, case_study)
        
        return(i_sp[idx])
            
    
    def select_input(self, cn, case_study):
        
        """
        
        select_input(self, cn)
        
        Selects the predictors for the data-driven models
        
        Input:
            - self.ivs_mode = input variable selection method
            - cn = training set
        
        Returns:
            - i_s = column index(es) of the selected predictor(s)
        """
        
        # Inspect the method and run the ivs procedure
        if self.ivs_mode == 'correlation':
            i_s = self.ivs_correlation(cn, case_study)
        
        elif self.ivs_mode == 'forward_selection':
            i_s = self.ivs_forward_selection(cn, case_study)
        
        elif self.ivs_mode == 'exhaustive':    
            i_s = self.ivs_exhaustive_search(cn, case_study)
        
        else:
            i_s = np.arange(cn.shape[1]-1)
            
        return(i_s)
    
    

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 16:42:20 2022

@author: AMARANTO
"""

# Import the required libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def train_module(x_tr, x_cv, y_tr, hp):
    
    """
    train_module(x_tr, x_cv, y_tr, hp)
    
    training function for an additional model module in the B-AMA protocol
    
    Input: 
        - x_tr = normalized training set input [n_training_instances, n_features]
        - x_cv = normalized cross validation set input [n_cv_instances, n_features]
        - y_tr = dependent variable in the training set [n_training_instances]
        - hp = hyper-parameters values [problem dimensionality]
        
    
    Returns:
        - new_model = the model
        - yh = prediction on the cross-validation set [n_cv_instances]
    """
    
    
    # Reshaping the data in the LSTM input format (n_instances, 1, n_features)
    x_tr = x_tr.reshape((x_tr.shape[0], 1, x_tr.shape[1]))
    x_cv = x_cv.reshape((x_cv.shape[0], 1, x_cv.shape[1]))

    # Define the new model
    new_model = Sequential()
    new_model.add(LSTM(int(hp[0]), input_shape=(x_tr.shape[1], x_tr.shape[2])))
    new_model.add(Dense(1))
    new_model.compile(loss='mae', optimizer='adam')
    # Fit the model
    new_model.fit(x_tr, y_tr, epochs=int(hp[1]))
    
    # Predict
    yh = new_model.predict(x_cv)
    return(new_model, yh)

def test_module(m, x):
    
    """
    test_module(m, x):
    
    test function for an additional model module in the B-AMA protocol
    
    Input: 
        - m = the model 
        - x = normalized test set input [n_test_instances, n_features]
        
    
    Returns:
        - y = normalized test set output [n_test_instances, ]
    """
    
    # Reshaping the data in the LSTM input format (n_instances, 1, n_features)
    x = x.reshape((x.shape[0], 1, x.shape[1]))
    # Perform the forecast
    y = m.predict(x)
    
    # Reshape the data in (n_instances, )
    y = y.reshape((y.shape[0], ))
    
    return(y)
    
    
    
    
    
    
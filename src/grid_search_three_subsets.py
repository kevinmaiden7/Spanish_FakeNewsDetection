#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from io import BytesIO

def grid_search(X, Y, models, epochs, batch_size, iterations, test_size):
    
    Y = dataset.label.values
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = test_size, shuffle = True)

    mean_dev_accuracy = np.zeros(np.size(models))
    
    i = 0
    for model in models:
    
        iterations = iterations
        dev_accuracy_shuffle_split = np.zeros(iterations)
        shuffle = ShuffleSplit(n_splits = iterations, test_size = test_size)
        
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy']) # Compile model
        w = BytesIO()
        model.save_weights(w) # Save initial weights
        
        j = 0
        for train, dev in shuffle.split(X_train, y_train):
            Xtrain = X_train[train]
            Ytrain = y_train[train]
            Xdev = X_train[dev]
            Ydev = y_train[dev]

            model.fit(Xtrain, Ytrain, epochs = epochs, batch_size = batch_size, shuffle = True) # Fit model

            loss, accuracy_val = model.evaluate(Xdev, Ydev) # Validate model
            dev_accuracy_shuffle_split[j] = accuracy_val

            model.load_weights(w) # Restore initial weights
            
            j += 1
              
        mean_dev_accuracy[i] = round(np.mean(dev_accuracy_shuffle_split), 3)
        std = round(np.std(dev_accuracy_shuffle_split), 3)
        print('Model ' + str(i) +' --> dev_acc: ' + str(mean_dev_accuracy[i]) + ' +- ' + str(std))
        i += 1
    
    # test
    best_model_index = np.argmax(mean_dev_accuracy)
    best_model = models[best_model_index]
    best_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy']) # Compile best model
    best_model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, shuffle = True) # Fit best model
    y_pred = best_model.predict(X_test) # Test best model
    y_pred = y_pred > 0.5 # Sigmoid activation function
    accuracy_test = accuracy_score(y_test, y_pred)
    
    return best_model_index, round(accuracy_test, 3)

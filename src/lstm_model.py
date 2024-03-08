
import time
import os

import numpy as np
import pandas as pd
import mysql.connector

from datetime import datetime
from database_manager import DatabaseConnector

# TENSORFLOW
import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.losses import MeanSquaredError
from tensorflow.python.keras.metrics import RootMeanSquaredError
from tensorflow.python.keras.models import load_model

# SCIKIT 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

'''
CROP YIELD: https://www.nature.com/articles/s41598-021-97221-7
G4G: https://www.geeksforgeeks.org/multiple-linear-regression-with-scikit-learn/
SCALING: https://betterdatascience.com/data-scaling-for-machine-learning/
Using Multivariate Linear Regression for Biochemical Oxygen Demand Prediction in Waste Water: https://arxiv.org/pdf/2209.14297.pdf  **GREAT RESOURCE**
'''

class LSTMmodel:
    def __init__(self, features_train, targets_train, features_test, targets_test):
        self.model = None

        self.x_train = features_train
        self.y_train = targets_train
        self.x_test = features_test
        self.y_test = targets_test

        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    def build_model(self, input_shape):
        model = Sequential([
        InputLayer(input_shape),
        LSTM(64, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=[RootMeanSquaredError()])
        return model
    
    def train_model(self, epochs=10, batch_size=32, validation_split=0.2):
        checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'best_model.h5')
        cp = ModelCheckpoint(checkpoint_path, save_best_only=True)
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[cp])

    def evaluate_model_performance(self): 
        # perform predictions on features (setpoints)
        predictions = self.model.predict(self.x_test).flatten()
        print('mean_squared_error : ', mean_squared_error(self.y_test, predictions))
        print('mean_absolute_error : ', mean_absolute_error(self.y_test, predictions))


    @staticmethod
    def reshape_data(df, features, targets, cycle_length=1):
        X, Y = [], []
        for i in range(len(df) - cycle_length):
            X.append(df[features].iloc[i:i+cycle_length].values)
            Y.append(df[targets].iloc[i+cycle_length-1])
        return np.array(X), np.array(Y).reshape(-1, 1)
    
    @staticmethod
    def predict(model, data):
        return model.predict(data)
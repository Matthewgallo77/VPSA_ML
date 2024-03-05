
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
    def __init__(self, feature_variables, target, train_df, test_df, scaler_inputs, scaler_target):
        self.feature_names = list(feature_variables.keys()) # list holding features
        self.target_name = list(target.keys())[0] # string holding name of target
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

        self.train_df = train_df
        self.test_df = test_df

        self.scaler_inputs = scaler_inputs
        self.scaler_target = scaler_target

        


    def evaluate_model_performance(y_test, predictions): 
        # Evaluate performance of model
        print('mean_squared_error : ', mean_squared_error(y_test, predictions))
        print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))

    def predict_purity_using_LSTM(trained_LSTM_model, test_matrix, scaler): 
        # Predict purity for the trained model 
        prediction_scaled = trained_LSTM_model.predict(test_matrix)
        prediction_2d = np.array(prediction_scaled).reshape(-1,1)

        prediction_inverse_scaled = scaler.inverse_transform(prediction_2d)

        return prediction_inverse_scaled

def reshape_training_data(df):
    # Reshape training data
    x=[]
    y=[]

    cycle_length = 1 # 30 data points per cycle 
    for i in range(0, len(df)-cycle_length):
        x.append(df.iloc[i:i+cycle_length, :-1].values) # CAPTURE 30 DATA POINTS ON EACH LOOP (approx 30 minutes of plant data)
        y.append(df.iloc[i+cycle_length, -1]) # CAPTURE 30 DATA POINTS ON EACH LOOP (Purity only)

    x = np.array(x)
    y = np.array(y)
    return x, y


def reshape_test_data(test_df):
    test_data = test_df.drop(targetPurity, axis=1).to_numpy() # Drop target purity from the data frame and convert to numpy array (1,5)
    test_data = test_data.reshape(1,1,5)

    return test_data


def trainLSTM_Model(x, y, cycle_length=1):  
    # Train long short term memory model
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) # SET RANDOM STATE FOR REPRODUCIBLE RESULTS
    print(x_train.shape, y_train.shape, x_test.shape,  y_test.shape)

    LSTM_model = Sequential() # Stack of layers with input tensor and output tensor
    LSTM_model.add(InputLayer((cycle_length,5))) # Each input sample is array of 30 sequences, each containing 5 features
    LSTM_model.add(LSTM(64)) # Add LSTM  layer. 64 neurons
    LSTM_model.add(Dense(8, 'relu')) # Add fully connected layer. 8 specifies number of neurons in layer. Rectified Linear Unit
    LSTM_model.add(Dense(1, 'linear')) # The predicted purity. Adds another dense layer. 1 neuron. Linear activation function
    LSTM_model.summary()

    cp = ModelCheckpoint('model/', save_best_only=True) # THIS SAVES THE BEST MODEL, LOWEST LOSS

    LSTM_model.compile(optimizer='adam', loss=MeanSquaredError() , metrics=[RootMeanSquaredError()])
    LSTM_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), callbacks=[cp])

    train_predictions = LSTM_model.predict(x_train).flatten()
    train_results = pd.DataFrame(data={'Train Predictions':train_predictions, ' Actuals': y_train})

    return LSTM_model

if __name__ == '__main__':
    db = DatabaseConnector() # INSTANCE OF DATABASE
    
    colNames = ['TCYCLEREAL','F401','TI143', 'PmaxAvg', 'PminAvg', 'AI401'] # LIST OF SENSORS WE WANT DATA FROM
    targetPurity = colNames[-1]
    test_setpoints = [27000, 2.5, 95.13, 5.8, -6.81, 78] # NEW SETPOINTS YOU WANT TO TEST


    scaler_inputs = MinMaxScaler() # Scaler used for input
    scaler_purity = MinMaxScaler() # Scalar used for purity
    # PmaxAvg, PminAvg, TCYCLEREAL, F401_1C, TI143, AI401


    testraw_df = convert_test_data_to_dataframe(test_setpoints)
    testprocessed_df = preprocessData(testraw_df, scaler_inputs, scaler_purity, targetPurity)
    reshaped_test_data = reshape_test_data(testprocessed_df)
    print(reshaped_test_data)
    
    inputdata_matrix, purity_matrix = reshape_training_data(processed_df)
    testdata_matrix = reshape_test_data(testprocessed_df)

    print(testdata_matrix)
    model = trainLSTM_Model(inputdata_matrix, purity_matrix, cycle_length=1)
    model = load_model('model') # LOAD THE BEST MODEL
    prediction = predict_purity_using_LSTM(model, testdata_matrix, scaler_purity)
    print(prediction)

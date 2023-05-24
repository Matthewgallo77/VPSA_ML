import mysql.connector
from datetime import datetime
from database_collect import DB
import time
import numpy as np
import pandas as pd

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
 
def getPandasFrame(colNames):
    sensorDict = {} # {'sensor1': sensor_data... 'sensorn': sensor_data}

    for i in range(len(colNames)):
        colName = colNames[i]
        data = db.getColumnValues(colName)
        sensorDict.update({colName: [value for value in data]})  # GET SENSOR DATA, (time is also available but not added to df since already organized by index)
        
    df = pd.DataFrame.from_dict(sensorDict, orient='columns')  # DICTIONARY TO DATA FRAME
    df = df[20::] # USE DATA AFTER 20 MINUTES FOR STABILIZATION OF PLANT

    return df

def getTestPandasFrame(test_setpoints): 
    test_sensorDict ={} # {'sensor1': test_setpoint1... 'sensorn': test_setpointn}

    for i in range(len(colNames)):
        colName = colNames[i]
        data = test_setpoints[i]
        test_sensorDict.update({colName: data})
        test_sensorList = [test_sensorDict]

    test_df = pd.DataFrame(test_sensorList)

    return test_df

def preprocessData(df, scaler):
    # IMPLEMENT IMPUTATION... if necessasry
    df.dropna(inplace=True)
    df[colNames] = scaler.transform(df[colNames]) # SCALES VALUES 

    # reuse scaler for future use
    return df

def train_PolyRegressionModel(df):

    x = df.drop(targetPurity, axis=1) # CREATES FRAME OF INPUT VARIABLES
    y = df[targetPurity] # CREATES FRAME OF PURITY
    
    # https://www.geeksforgeeks.org/how-to-split-a-dataset-into-train-and-test-sets-using-python/   
    # Explore different test_sizes to see what works best
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=None, shuffle=None, stratify=None )

    model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
    model.fit(x_train, y_train) # THE MODEL LEARNS THE TRAINING DATA
    return model
 
def evaluatePerformance(y_test, predictions): # AIM FOR LOW MSE AND MAE
    print('mean_squared_error : ', mean_squared_error(y_test, predictions))
    print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))

def predictPurity(trainedModel, setpoints_test): # INPUT DESIRED SETPOINTS, OUTPUTS ARRAY OF PURITY 
    setpoints_test = setpoints_test.drop(targetPurity, axis=1)
    predictions = trainedModel.predict(setpoints_test)
    return predictions

def reshapeTestData(df):
    x=[]
    y=[]

    cycle_length = 30  # 1-min data collection frequency ~30mins of plant runtime
    for i in range(0, len(df)-cycle_length):
        x.append(df.iloc[i:i+cycle_length, :-1].values) # CAPTURE 30 DATA POINTS ON EACH LOOP (approx 30 minutes of plant data)
        y.append(df.iloc[i+cycle_length, -1]) # CAPTURE 30 DATA POINTS ON EACH LOOP (Purity only)

    x = np.array(x)
    y = np.array(y)

    return x, y

def trainLSTM_Model(x, y, cycle_length=30):  

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) # SET RANDOM STATE FOR REPRODUCIBLE RESULTS
    print(x_train.shape, y_train.shape, x_test.shape,  y_test.shape)

    '''
    units = 64neurons (power of 2 usually)
    Dense is what you are predicting (purity)
    '''
    model = Sequential() # DEFINE LSTM MODEL 
    model.add(InputLayer((30,5)))
    model.add(LSTM(64))
    model.add(Dense(8, 'relu'))
    model.add(Dense(1, 'linear'))
    model.summary()

    cp = ModelCheckpoint('model/', save_best_only=True) # THIS SAVES THE BEST MODEL, LOWEST LOSS

    model.compile(optimizer='adam', loss=MeanSquaredError() , metrics=[RootMeanSquaredError()])

    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), callbacks=[cp])

    train_predictions = model.predict(x_train).flatten()
    train_results = pd.DataFrame(data={'Train Predictions':train_predictions, ' Actuals': y_train})
    print(train_results)

    return model

if __name__ == '__main__':
    db = DB() # INSTANCE OF DATABASE
    scaler = MinMaxScaler() # GLOBAL SCALAR
    colNames = ['TCYCLEREAL','F401','TI143', 'PmaxAvg', 'PminAvg', 'AI401'] # LIST OF SENSORS WE WANT DATA FROM
    targetPurity = colNames[-1]
    test_setpoints = [31000, 2.3, 78.13, 5.8, -7.01, 80] # NEW SETPOINTS YOU WANT TO TEST

    # PmaxAvg, PminAvg, TCYCLEREAL, F401_1C, TI143, AI401_1C_OLD
    raw_df = getPandasFrame(colNames) # DICTIONARY CONTAINING {MONTH-DAY:DF, MONTH-DAY2: DF2 ...}

    scaler.fit(raw_df[colNames])
    processed_df = preprocessData(raw_df, scaler)
    testraw_df = getTestPandasFrame(test_setpoints)
    testprocessed_df = preprocessData(testraw_df, scaler)

    inputdata_matrix, purity_matrix = reshapeTestData(processed_df)

    model = trainLSTM_Model(inputdata_matrix, purity_matrix, cycle_length=30)
    model = load_model('model') # LOAD THE BEST MODEL

    




    


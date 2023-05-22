import mysql.connector
from datetime import datetime
from database_collect import DB
import time
import numpy as np
import pandas as pd

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
    # timestamps = db.getTimestamps()
    # sensorDict.update({'Timestamp': [date for date in timestamps]})
    for i in range(len(colNames)):
        colName = colNames[i]
        data = db.getColumnValues(colName)
        sensorDict.update({colName: [value for value in data]})  # GET SENSOR DATA, (time is also available but not added to df since already organized by index)
        
    df = pd.DataFrame.from_dict(sensorDict, orient='columns')  # DICTIONARY TO DATA FRAME
    print(df)
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
    # IMPLEMENT IMPUTATION, repacing outliers with mean
    df.dropna(inplace=True)
    df[colNames] = scaler.transform(df[colNames]) # SCALES VALUES 

    # reuse scaler for future use
    return df

def trainModel(df):

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

if __name__ == '__main__':
    db = DB() # INSTANCE OF DATABASE
    scaler = MinMaxScaler() # GLOBAL SCALAR

    # PmaxAvg, PminAvg, TCYCLEREAL, F401_1C, TI143, AI401_1C_OLD
    PmaxAvg, PminAvg = 0, 0 # dummy values get from tag table
    colNames = ['TCYCLEREAL', 'F401_1C', 'TI143', 'AI401_1C_OLD'] # LIST OF SENSORS WE WANT DATA FROM
    targetPurity = colNames[-1]

    raw_df = getPandasFrame(colNames) # DICTIONARY CONTAINING {MONTH-DAY:DF, MONTH-DAY2: DF2 ...}
    scaler.fit(raw_df[colNames])
    processed_df = preprocessData(raw_df, scaler)
    trainedModel = trainModel(processed_df) # TRAIN THE MODEL FROM PAST DATA

    test_setpoints = [31000, 2.33, 89.6, 89.77] # NEW SETPOINTS YOU WANT TO TEST
    testraw_df = getTestPandasFrame(test_setpoints)

    testprocessed_df = preprocessData(testraw_df, scaler)

    purityPredictions = predictPurity(trainedModel, testprocessed_df) 
    print(purityPredictions)

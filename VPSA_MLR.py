import mysql.connector
from datetime import datetime
from database_collect import DB
import time
import numpy as np
import pandas as pd

# SCIKIT 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

    model = LinearRegression()
    model.fit(x_train, y_train) # THE MODEL LEARNS THE TRAINING DATA
    return model
 
def evaluatePerformance(y_test, predictions):
    print('mean_squared_error : ', mean_squared_error(y_test, predictions))
    print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))

def predictPurity(trainedModel, setpoints_test):
    setpoints_test = setpoints_test.drop(targetPurity, axis=1)
    predictions = trainedModel.predict(setpoints_test)
    print(predictions)
    return None

if __name__ == '__main__':
    db = DB()
    scaler = MinMaxScaler() # GLOBAL SCALAR

    # PmaxAvg, PminAvg, TCYCLEREAL, F401_1C, AI401_1C_OLD
    PmaxAvg, PminAvg = 10, 5 # dummy values
    colNames = ['TCYCLEREAL', 'F401_1C', 'AI401_1C_OLD'] # LIST OF SENSORS WE WANT DATA FROM
    targetPurity = colNames[-1]

    rawdf = getPandasFrame(colNames) # DICTIONARY CONTAINING {MONTH-DAY:DF, MONTH-DAY2: DF2 ...}
    print(rawdf)
    scaler.fit(rawdf[colNames])

    processed_df = preprocessData(rawdf, scaler)
    trainedModel = trainModel(processed_df) # TRAIN THE MODEL FROM PAST DATA
    test_setpoints = [27000, 2.61, 89.7] # NEW SETPOINTS YOU WANT TO TEST
    testraw_df = getTestPandasFrame(test_setpoints)
    testprocessed_df = preprocessData(testraw_df, scaler)
    # print(testprocessed_df) 
    purityPredictions = predictPurity(trainedModel, testprocessed_df) 
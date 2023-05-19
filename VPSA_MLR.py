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
    timestamps = db.getTimestamps()
    sensorDict.update({'Timestamp': [date for date in timestamps]})
    for i in range(len(colNames)):
        colName = colNames[i]
        data = db.getColumnValues(colName)
        sensorDict.update({colName: [value for value in data]})  # GET SENSOR DATA, (time is also available but not added to df since already organized by index)
    
    df = pd.DataFrame.from_dict(sensorDict, orient='columns')  # DICTIONARY TO DATA FRAME

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%m-%d %H:%M:%S')
    df.set_index('Timestamp', inplace=True)

    dictdata_bydate = {date.strftime('%m-%d'): df_group for date, df_group in df.groupby(df.index.date)} # DICTIONARY CONTAINING {MONTH-DAY:DF, MONTH-DAY2: DF2 ...}
    
    return dictdata_bydate

    # THE DATA FRAME IS ORGANIZED WHERE EACH ROW IS CORRESPONDING SENSOR VALUE. ALLOWING FOR EASY DATA MANIPULATION 
    #     PT205A-(1)  PT205B-(1)  PT401-(1)  PURITY-1
    # 0          3.0         4.0        5.0      11.0
    # 1         12.0        13.0       14.0      20.0

def preprocessData(df_dict):
    # IMPLEMENT IMPUTATION, repacing outliers with mean

    processed_data = {}
    for date, df in df_dict.items():
        df.dropna(inplace=True)
        scaler = MinMaxScaler()
        df[colNames] = scaler.fit_transform(df[colNames]) # SCALES VALUES 
        processed_data[date] = df

    return processed_data

def trainModel(df):
    x = df.drop(targetPurity, axis=1) # CREATES FRAME OF INPUT VARIABLES
    y = df[targetPurity] # CREATES FRAME OF PURITY
    
    # https://www.geeksforgeeks.org/how-to-split-a-dataset-into-train-and-test-sets-using-python/

    # Explore different test_sizes to see what works besst
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=None, shuffle=None, stratify=None )

    model = LinearRegression()
    model.fit(x_train, y_train) # THE MODEL LEARNS THE TRAINING DATA

    predictions = model.predict(x_test)

    evaluatePerformance(y_test, predictions)
 
def evaluatePerformance(y_test, predictions):
    print('mean_squared_error : ', mean_squared_error(y_test, predictions))
    print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))


def getCycleTime():
    x=5
    return x

if __name__ == '__main__':
    db = DB()

    colNames = ['PT205A-(1)', 'PT205B-(1)', 'PT401-(1)', 'PURITY-1'] # LIST OF SENSORS WE WANT DATA FROM
    targetPurity = colNames[3]
    df_dict = getPandasFrame(colNames) # DICTIONARY CONTAINING {MONTH-DAY:DF, MONTH-DAY2: DF2 ...}
    processed_data = preprocessData(df_dict)
    print(processed_data)
    
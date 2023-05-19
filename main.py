from database_collect import DB 
import time

def dataDict(colNames):
    sensorDict = {} # {'sensor1': (sensor_data, time)... 'sensorn': (sensor_data, time)}
    for i in range(1, len(colNames)):
        colName = colNames[i]
        data = db.getColumnValues(colName)
        sensorDict.update({colName: data})
        time.sleep(1)

    return sensorDict

def pandaFrame()

if __name__ == '__main__':
    db = DB()
    
    colNames = ['Timestamp', 'PT205A-(1)', 'PT205B-(1)', 'PT401-(1)'] # LIST OF SENSORS WE WANT DATA FROM





    
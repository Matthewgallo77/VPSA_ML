import mysql.connector
from datetime import datetime
import time
import numpy as np
import pandas as pd
import sys
import threading

class DB:    

    def __init__(self):

        while True:
            self.databaseName = 'pilot_test'
            self.host = 'localhost'
            self.user = 'root'
            self.password = 'root'


            self.animate()
            # Try to connect to the database
            try:
                
                self.db = mysql.connector.connect(
                    host=self.host,
                    user=self.user,
                    passwd=self.password,
                    database=self.databaseName
                )

                self.table_past = 'pastValues'
                self.table_live = 'liveValues'
                break

            except mysql.connector.Error as err:
                print("\nFailed to connect to database:", err)
                print("\nPlease check your connection information and try again.")

    def animate(self):
        animation = ["[■□□□□□□□□□]","[■■□□□□□□□□]", "[■■■□□□□□□□]", "[■■■■□□□□□□]", "[■■■■■□□□□□]", "[■■■■■■□□□□]", "[■■■■■■■□□□]", "[■■■■■■■■□□]", "[■■■■■■■■■□]", "[■■■■■■■■■■]"]
        for i in range(len(animation)):
            time.sleep(0.001)
            sys.stdout.write("\r" + animation[i % len(animation)])
            sys.stdout.flush()     


    def getColumnValues(self, colName): # GET VALUE OF COLUMN AND CORRESPONDING TIME
        mycursor = self.db.cursor()
        mycursor.execute(f"SELECT `{colName}` FROM {self.table_past}")
        data = [float(row[0]) for row in mycursor.fetchall()]

        return data
    
    
    def getTimestamps(self): # LIST OF TIMESTAMPS
        mycursor = self.db.cursor()
        mycursor.execute(f"SELECT Timestamp FROM {self.table_past}")
        timestamps = [row[0].strftime('%m-%d %H:%M:%S') for row in mycursor.fetchall()]

        return timestamps


            


import time
import math
import threading
import os

import pandas as pd
from datetime import datetime

import mysql.connector
import mysql.connector.errors

class DatabaseConnector:    

    def __init__(self, database_name=None, host=None, user=None, password=None):

        self.db = None
        self.database_name = 'pilotplant'
        self.host = 'localhost'
        self.user = 'root'
        self.password = 'root'
        self.sql_config_path = 'configs/mysql_config.txt' # this will be the database_name
        self.is_connected = False
        
        self.data_frame = None # store data frame for operations 

        self.table_live = 'liveValues'
        self.table_past = 'pastValues'

    def update_live_values(self, values, names): # UPDATES VALUE AND TIME IN ALL TABLES
        mycursor = self.db.cursor()
        update_queries = []
        for value, name in zip(values, names):
            processed_value = self.process_value(value)
            update_queries.append((datetime.now(), processed_value, name))
        
        update_query = f"UPDATE {self.table_live} SET Timestamp = %s, Value = %s WHERE Name = %s"
        mycursor.executemany(update_query, update_queries)
        self.db.commit()
        return

    def add_past_values(self, values, names):
        mycursor = self.db.cursor()
        insertNewValue_Query= f"INSERT INTO {self.table_past} (Timestamp, "
        for i, name in enumerate(names):
            if i==len(names)-1:
                insertNewValue_Query += f"`{name}`)"
            else:
                insertNewValue_Query += f"`{name}`, "
        insertNewValue_Query += f" VALUES ('{datetime.now()}',"
        for i, value in enumerate(values):
            try:
                if value is None or math.isnan(float(value)) or (float(value) < -99999.99999 or float(value) > 99999.99999):
                    value = 0
            except (ValueError, TypeError, mysql.connector.errors.DataError) as e:
                value = 0
            if i==len(values)-1:
                insertNewValue_Query += f" {value})"
            else:
                insertNewValue_Query += f" {value}, "
        
        mycursor.execute(insertNewValue_Query)
        self.db.commit()     

        return                  
 
    def build_insert_past_values_query(self, Name, Datatype, Address):
        mycursor = self.db.cursor()
        mycursor.execute(f"INSERT IGNORE INTO {self.table_live} (Name, Datatype, Address, Timestamp, Value) VALUES (%s, %s, %s, %s, NULL)", (Name, Datatype, Address, datetime.now()))
        self.db.commit()
        return
 
    def create_live_table(self):
        mycursor = self.db.cursor()
        mycursor.execute(f"CREATE TABLE IF NOT EXISTS {self.table_live} (Name VARCHAR(50) PRIMARY KEY, Datatype VARCHAR(10) NOT NULL, Address VARCHAR(10) NOT NULL, Timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL, Value DECIMAL(10,5))")
        self.db.commit()
        return

    def create_past_sensor_data_table(self, names):
        # INPUT: NAMES
        # FUNCTION: FORMATS TABLE WITH TIMESTAMP, SENSORNAME & SENSORNAME_VALUE
        mycursor = self.db.cursor()
        query = f"CREATE TABLE IF NOT EXISTS {self.table_past} (Timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL ,"
        for i, name in enumerate(names):
            if i==len(names)-1:
                query += f"`{name}` DECIMAL(10,5))"
            else:
                query += f"`{name}` DECIMAL(10,5), "
        mycursor.execute(query)
        self.db.commit()
        return

    def get_column_headers(self): # GET ALL THE HEADERS OF A COLUMN
        mycursor = self.db.cursor()
        mycursor.execute("DESCRIBE pastvalues")
        tagNames = [row[0] for row in mycursor.fetchall()]
        return tagNames

    def get_column_value(self, col_name): # GET VALUE OF COLUMN AND CORRESPONDING TIME
        mycursor = self.db.cursor()
        mycursor.execute(f"SELECT `{col_name}`, Timestamp FROM {self.table_past}")
        data = []
        for row in mycursor.fetchall():
            data.append((float(row[0]), row[1].strftime('%H:%M:%S')))

        return data

    def get_column_values(self, col_name):
        mycursor = self.db.cursor()
        data = []
        mycursor.execute(f"SELECT `{col_name}` FROM {self.table_past}")
        data = [float(row[0]) if row[0] is not None else None for row in mycursor.fetchall()]
        return data
    
    def get_live_value_and_time(self, sensorName):
        mycursor = self.db.cursor()
        mycursor.execute(f"SELECT Value, Timestamp FROM {self.table_live} WHERE Name = '{sensorName}'")
        result = mycursor.fetchall()
        
        return (float(result[0][0]), result[0][1].strftime('%H:%M:%S'))
    
    def get_live_value(self, tagName):
        mycursor = self.db.cursor()
        mycursor.execute(f"SELECT Value FROM {self.table_live} WHERE Name = '{tagName}'")
        result = mycursor.fetchone()        
        return float(result[0])
    
    def get_address_by_name(self, tag_name):
        mycursor = self.db.cursor()
        mycursor.execute(f"SELECT Address FROM {self.table_live} WHERE Name = %s", (tag_name,))
        result = mycursor.fetchone()
        return result[0] if result else None
    
    def get_datatype_by_name(self, tag_name):
        mycursor = self.db.cursor()
        mycursor.execute(f"SELECT Datatype FROM {self.table_live} WHERE Name = %s", (tag_name,))
        result = mycursor.fetchone()
        return result[0] if result else None


    def write_db_config(self):
        os.makedirs(os.path.dirname(self.sql_config_path), exist_ok=True)  # Create the directory if it doesn't exist
        with open(self.sql_config_path, 'w') as file:
            file.write(f"{self.database_name},{self.host},{self.user},{self.password}")
        return
    
    def get_timestamps(self): # LIST OF TIMESTAMPS
        mycursor = self.db.cursor()
        mycursor.execute(f"SELECT Timestamp FROM {self.table_past}")
        timestamps = [row[0].strftime('%m-%d %H:%M:%S') for row in mycursor.fetchall()]

        return timestamps

    def load_db_config(self):
        if os.path.exists(self.sql_config_dir):
            for file in os.listdir(self.sql_config_dir):
                if file.endswith("config.txt"):
                    self.sql_config_path = os.path.join(self.sql_config_dir, file)
                    with open(self.sql_config_path, 'r') as file:
                        db_name, host, user, password = file.read().strip().split(',')
                        self.database_name = db_name
                        self.host = host
                        self.user = user
                        self.password = password
                        return True
        return False

    def connect_database(self):
        try:
            self.db = mysql.connector.connect(
                host=self.host,
                user=self.user,
                passwd=self.password,
                database=self.database_name
            )
            if self.db.is_connected():
                self.is_connected = True 
                self.write_db_config()
                return True
        except mysql.connector.Error as err:
            return False
        

    def process_value(self, value):
            try:
                if value is None or math.isnan(float(value)) or (float(value) < -99999.99999 or float(value) > 99999.99999):
                    return 0
                else:
                    return float(value)
            except (ValueError, TypeError, mysql.connector.DataError) as e:
                return 0
            
    




    

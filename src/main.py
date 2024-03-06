import sys

from database_manager import DatabaseConnector 
from PLC_manager import PLCManager
from data_prep import DataPreparationPipeline
from lstm_model import LSTMmodel


import pandas as pd

def main():
    # connect to database
    database_manager = DatabaseConnector() #
    database_manager.connect_database()
    PLC_manager = PLCManager()
    PLC_manager.connect_plc()
    # selected features and their corresponding setpoints

    # fetch and prepare the data for training
    data_pipeline = DataPreparationPipeline(database_manager, PLC_manager)
    

    # use data to train the model
    # lstm_model = LSTMmodel(feature_variables, target_variable, data_pipeline.train_df, data_pipeline.test_df, data_pipeline.scaler_inputs, data_pipeline.scaler_target)


if __name__ == '__main__':
    main()


    
import sys

from database_manager import DatabaseConnector 
from PLC_manager import PLCManager
from data_prep import DataPreparationPipeline
from lstm_model import LSTMmodel


import utilities
import pandas as pd

def main():
    # connect to database
    database_manager = DatabaseConnector()
    database_manager.connect_database()

    PLC_manager = PLCManager()
    feature_variables = PLC_manager.feature_variables
    target_variables = PLC_manager.target_variables
    # selected features and their corresponding setpoints

    # fetch and prepare the data for training
    data_pipeline = DataPreparationPipeline(database_manager, PLC_manager, feature_variables, target_variable)

    # prepare test data
    

    # use data to train the model
    # lstm_model = LSTMmodel(feature_variables, target_variable, data_pipeline.train_df, data_pipeline.test_df, data_pipeline.scaler_inputs, data_pipeline.scaler_target)


if __name__ == '__main__':
    main()


    
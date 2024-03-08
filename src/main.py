import sys

from database_manager import DatabaseConnector 
from PLC_manager import PLCManager
from data_prep import DataPreparationPipeline
from lstm_model import LSTMmodel

import pandas as pd

def main():

    database_manager = DatabaseConnector() # connect to database 
    PLC_manager = PLCManager() # connect to PLC

    data_pipeline = DataPreparationPipeline(database_manager, PLC_manager)
    processed_train_df, processed_test_df = data_pipeline.processed_train_data, data_pipeline.processed_test_data
    


    # use data to train the model
    # lstm_model = LSTMmodel(feature_variables, target_variable, data_pipeline.train_df, data_pipeline.test_df, data_pipeline.scaler_inputs, data_pipeline.scaler_target)


if __name__ == '__main__':
    main()


    
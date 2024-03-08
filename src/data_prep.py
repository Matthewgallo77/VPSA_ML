import os 
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

class DataPreparationPipeline:

    def __init__(self, database, plc):

        self.database = database # database object
        self.plc = plc # plc object

        self.feature_names = ['Pmax_avg', 'Pmin_avg', 'Cycle_Time', 'Purity_Avg', 'Temp_Avg'] # input
        self.feature_min_max_values = {
            'Pmax_avg': (14, 24),  # Example min and max values for features
            'Pmin_avg': (3, 15),
            'Cycle_Time': (15, 125),
            'Purity_Avg': (60, 100),
            'Temp_Avg': (-20, 150)
        }
        
        self.target_names = ['FIC12_Avg', 'FIC31_Avg'] # output
        self.target_min_max_values = {
            'FIC12_Avg': (0, 50), 
            'FIC31_Avg': (0, 5)
        }

        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data') # dir path for storing training and testing data

        # set setpoints for feature and targets
        '''
        Goal: automate this by reading directly from PLC memory (for now we have manual entry)
        '''

        self.feature_variables, self.target_variables = self.set_feature_setpoints(), self.set_target_setpoints()

        self.raw_train_data = self.train_data_to_dataframe() 
        self.processed_train_data = self.preprocess_data(self.raw_train_data) # preprocess train data
        self.features_train, self.targets_train = self.processed_train_data[self.feature_variables], self.processed_train_data[self.target_variables]

        self.raw_test_data = self.test_data_to_dataframe()
        self.processed_test_data = self.preprocess_data(self.raw_test_data) # preprocess test data
        self.features_test, self.targets_test = self.processed_test_data[self.feature_variables], self.processed_test_data[self.target_variables]
        
    def train_data_to_dataframe(self):
        # read values from database
        # create training data dataframe
        sensor_dict = {}
        for col_name in self.feature_names + self.target_names:
            data = self.database.get_column_values(col_name)
            sensor_dict[col_name] = data
        
        train_df = pd.DataFrame(sensor_dict)
                
        # save raw data to csv
        raw_data_path = os.path.join(self.data_dir, 'raw_train_data.csv')
        train_df.to_csv(raw_data_path, index=False)

        return train_df
    
    def test_data_to_dataframe(self):
        # test data containing initial values of features and target
        test_sensor_dict = {}
        for col_name in self.feature_names + self.target_names:
            if col_name in self.feature_variables:
                test_sensor_dict[col_name] = [self.feature_variables[col_name]]
            elif col_name in self.target_variables:
                test_sensor_dict[col_name] = [self.target_variables[col_name]]

        test_df = pd.DataFrame(test_sensor_dict)

        raw_data_path = os.path.join(self.data_dir, 'raw_test_data.csv')
        test_df.to_csv(raw_data_path, index=False)

        return test_df
        
    def set_feature_setpoints(self):
        feature_variables = {}

        for feature_name in self.feature_names:
            try:
                value = float(input(f"Please input {feature_name} setpoint: "))
                feature_variables[feature_name] = value
            except ValueError:
                value = 0 
                feature_variables[feature_name] = value
                return None  # or you might want to retry the input

        return feature_variables
    
    def set_target_setpoints(self):
        target_variables = {}

        for target_name in self.target_names:
            try:
                target_variables[target_name] = float(input(f"Please input {target_name} setpoint: "))
            except ValueError:
                value = 0
                target_variables[target_name] = value

        return target_variables

    
    def preprocess_data(self, df):
        df_cleaned = df.dropna()
        features_scaled = self.preset_scaler(df_cleaned[self.feature_names], self.feature_min_max_values)
        targets_scaled = self.preset_scaler(df_cleaned[self.target_names], self.target_min_max_values)
        return pd.concat([features_scaled, targets_scaled], axis=1)

    def preset_scaler(self, data, min_max_values):
        data_scaled = pd.DataFrame()
        for col in data.columns:
            min_val, max_val = min_max_values[col]
            data_scaled[col] = (data[col] - min_val) / (max_val - min_val)
        return data_scaled

    
    
    
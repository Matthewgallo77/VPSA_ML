import os 
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

class DataPreparationPipeline:

    def __init__(self, database, plc):
        self.database = database # database object
        self.plc = plc

        self.feature_names = ['Pmax_avg', 'Pmin_avg', 'Cycle_Time', 'Purity_Avg', 'Temp_Avg']
        self.target_names = ['FIC12_Avg', 'FIC31_Avg']

        self.scaler_inputs = MinMaxScaler() # scaler for input features
        self.scaler_target = MinMaxScaler() # scaler for target feature

        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data') # dir for data

        # get setpoints for feature and targets
        self.feature_variables = self.set_feature_setpoints() 
        self.target_variables = self.set_target_setpoints()

        # get training data from sql database
        self.raw_train_data = self.train_data_to_dataframe() 
        self.processed_train_data = self.preprocess_data(self.raw_train_data) # preprocess train data

        # get test data from feature and target variables
        self.raw_test_data = self.test_data_to_dataframe()
        self.processed_test_data = self.preprocess_data(self.raw_test_data) # preprocess test data

    def train_data_to_dataframe(self):
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

        df_cleaned = df.dropna() # drop nan values (can occur from issues reading from memory)

        # split into features and target
        features = df_cleaned[self.feature_names] 
        target = df_cleaned[self.target_names]

        self.scaler_inputs.fit(features)
        features_scaled = self.scaler_inputs.transform(features)

        self.scaler_target.fit(target)
        targets_scaled = self.scaler_target.transform(target)

        df_scaled = pd.DataFrame(features_scaled, columns=self.feature_names)

        targets_scaled_df = pd.DataFrame(targets_scaled, columns=self.target_names)
        df_scaled = pd.concat([df_scaled, targets_scaled_df], axis=1)
        print(df_scaled)
        return df_scaled
    
    
    
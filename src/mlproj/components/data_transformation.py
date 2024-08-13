import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.mlproj.logger import logging
from src.mlproj.exceptions import CustomException
from src.mlproj.utils import save_object_file_path


@dataclass
class DataTransformationConfig:
    processor_obj_file_path = os.path.join('artifact', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def data_transformer(self):
        try:
            data = pd.read_csv(os.path.join('notebook/Data', 'raw.csv'))
            df = data.drop(['customerID', 'TotalCharges'], axis=1)

            numerical_features = [
                feature for feature in df.columns if df[feature].dtype != 'O']
            categorical_features = [
                feature for feature in df.columns if df[feature].dtype == 'O']
            
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical Columns: {categorical_features}")
            logging.info(f"Numerical Columns: {numerical_features}")

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_features),
                ('cat_pipeline', cat_pipeline, categorical_features)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info('Reading the train and test data files')

            preprocessor_obj = self.data_transformer()

            target_col_name = 'Churn'

            '''Train data splitting'''
            inp_train_data = train_df.drop(columns=[target_col_name], axis=1)
            target_train_data = train_df[target_col_name]

            '''Test data splitting'''
            inp_test_data = test_df.drop(columns=[target_col_name], axis=1)
            target_test_data = test_df[target_col_name]

            logging.info("Preprocessing train and test data")

            procs_train_arr = preprocessor_obj.fit_transform(inp_train_data)
            procs_test_arr = preprocessor_obj.transform(inp_test_data)

            train_arr = np.c_[procs_train_arr, np.array(target_train_data)]
            test_arr = np.c_[procs_test_arr, np.array(target_test_data)]

            save_object_file_path(
                file_path=self.data_transformation_config.
                processor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.processor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler,TargetEncoder,OrdinalEncoder

from src.exception import CustomException 
from src.logger import logging
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils import save_object
import os


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Convert to datetime
        X['policy_bind_date'] = pd.to_datetime(X['policy_bind_date'])
        X['incident_date'] = pd.to_datetime(X['incident_date'])

        # Extract features
        X['policy_year'] = X['policy_bind_date'].dt.year
        X['policy_month'] = X['policy_bind_date'].dt.month
        X['policy_day'] = X['policy_bind_date'].dt.day
        X['policy_dayofweek'] = X['policy_bind_date'].dt.dayofweek

        # Days between policy and incident
        X['days_to_incident'] = (X['incident_date'] - X['policy_bind_date']).dt.days

        # Drop original date columns
        X.drop(columns=['policy_bind_date', 'incident_date'], inplace=True)

        X['injury_ratio'] = X['injury_claim'] / X['total_claim_amount']
        X['property_ratio'] = X['property_claim'] / X['total_claim_amount']
        X['vehicle_ratio'] = X['vehicle_claim'] / X['total_claim_amount']

        return X


class DataTransfomation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            # Define column groups
            target_encoding_cols = [
                'incident_state',
                'insured_occupation',
                'auto_make',
                'insured_hobbies',
                'incident_city',
                'auto_model'
            ]

            onehot_cols = [
                'insured_sex',
                'policy_state',
                'policy_csl',
                'insured_relationship',
                'incident_type',
                'collision_type',
                'authorities_contacted',
                'insured_education_level',
                'property_damage',
                'police_report_available'
            ]

            numerical_cols = [
                'months_as_customer', 'age', 'policy_deductable',
                'umbrella_limit', 'capital-gains', 'capital-loss',
                'incident_hour_of_the_day', 'number_of_vehicles_involved',
                'bodily_injuries', 'witnesses', 'total_claim_amount', 
                'injury_claim', 'property_claim', 'vehicle_claim',
                'auto_year', 'policy_year', 'policy_month',
                'policy_day', 'policy_dayofweek', 'days_to_incident','injury_ratio','property_ratio','vehicle_ratio'
            ]

            ordinal_cols = ['incident_severity']

            severity_order =['Trivial Damage','Minor Damage','Major Damage','Total Loss']


            # 1. Numerical Pipeline
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # 2. Target Guided Encoding Pipeline (For High Cardinality)
            target_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                # TargetEncoder automatically handles high cardinality 
                # by shrinking estimates toward the global mean
                ('target_encoder', TargetEncoder()), 
                ('scaler', StandardScaler())
            ])

            # 3. One Hot Encoding Pipeline (For Low Cardinality)
            oh_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False)) # with_mean=False for sparse matrices
            ])

            #4. Ordinal Encoding Pipeline 
            ordinal_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinal_encoder',OrdinalEncoder(categories=[severity_order])),
                ('scaler',StandardScaler())
                ]
            )

            logging.info("Pipelines defined for Numerical, Target, and OneHot features.")

            # Combine all via ColumnTransformer
            preprocessor_step = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_cols),
                    ('target_pipeline', target_pipeline, target_encoding_cols),
                    ('oh_pipeline', oh_pipeline, onehot_cols),
                    ('ordinal_pipeline',ordinal_pipeline, ordinal_cols)
                ],
                remainder='drop' # Drop columns not specified
            )

            final_pipeline = Pipeline(steps=[
                ('feature_engineering', FeatureEngineeringTransformer()),
                ('preprocessor', preprocessor_step)
            ])

            return final_pipeline

        except Exception as e:
            raise CustomException(e, sys)


    
    def initate_data_transformation(self,train_path,test_path):

        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            target_column_name = 'fraud_reported'
            
            logging.info("Mapping target column values to 0 and 1")
            target_map = {'Y': 1, 'N': 0} # Check if your data uses 'Y'/'N' or 'Yes'/'No'

            train_df[target_column_name] = train_df[target_column_name].map(target_map)
            test_df[target_column_name] = test_df[target_column_name].map(target_map)

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformer_object()

            drop_columns = [target_column_name,'_c39','incident_location','insured_zip']

            input_feature_train_df = train_df.drop(columns=drop_columns)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing datasets.")
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df,target_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            logging.info('Exception occured in initiate_data_transformation function')
            raise CustomException(e,sys)


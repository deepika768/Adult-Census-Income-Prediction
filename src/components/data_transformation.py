import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os 

from src.utils import save_object


class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
             numerical_col=["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week",]
             categorical_col=["education","workclass","marital-status","occupation","relationship","race","sex","country","salary"]


             num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

             cat_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]
             )
             logging.info(f"Categorical columns: {categorical_col}")
             logging.info(f"Numerical columns: {numerical_col}")


             preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_col),
                ("cat_pipelines",cat_pipeline,categorical_col)

                ]


            )

             return preprocessor

        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            
            logging.info(f"Train DataFrame columns: {train_df.columns}")
            logging.info(f"Train DataFrame head:\n{train_df.head()}")
            logging.info(f"Test DataFrame columns: {test_df.columns}")
            logging.info(f"Test DataFrame head:\n{test_df.head()}")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            #target_column_name="salary"
            numerical_columns = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]

            input_feature_train_df=train_df.drop(columns="salary",axis=1)
            target_feature_train_df=train_df["salary"]

            input_feature_test_df=test_df.drop(columns="salary",axis=1)
            target_feature_test_df=test_df["salary"]

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)



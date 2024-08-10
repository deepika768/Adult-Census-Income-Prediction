import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object


class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_col = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
            categorical_col = ["education", "workclass", "marital-status", "occupation", "relationship", "race", "sex", "country"]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_col}")
            logging.info(f"Numerical columns: {numerical_col}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_col),
                    ("cat_pipelines", cat_pipeline, categorical_col)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            

            # Log train and test DataFrame information
            logging.info(f"Train DataFrame columns: {train_df.columns}")
            logging.info(f"Train DataFrame head:\n{train_df.head()}")
            logging.info(f"Test DataFrame columns: {test_df.columns}")
            logging.info(f"Test DataFrame head:\n{test_df.head()}")

            # Verify that 'salary' column is in both DataFrames
            if 'salary' not in train_df.columns:
                raise KeyError(f"'salary' column not found in train dataframe. Columns: {train_df.columns}")
            if 'salary' not in test_df.columns:
                raise KeyError(f"'salary' column not found in test dataframe. Columns: {test_df.columns}")

            # Ensure no leading/trailing spaces in column names
            train_df.columns = train_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()

            # Check data types
            logging.info(f"Train DataFrame dtypes:\n{train_df.dtypes}")
            logging.info(f"Test DataFrame dtypes:\n{test_df.dtypes}")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # Debug: Print the first few rows of the train and test DataFrames
            logging.info(f"Train DataFrame head after strip:\n{train_df.head()}")
            logging.info(f"Test DataFrame head after strip:\n{test_df.head()}")

            # Separate input features and target feature
            input_feature_train_df = train_df.drop(columns="salary", axis=1)
            target_feature_train_df = train_df["salary"]

            input_feature_test_df = test_df.drop(columns="salary", axis=1)
            target_feature_test_df = test_df["salary"]

            # Debug: Print the input features and target features before transformation
            logging.info(f"Input features (train):\n{input_feature_train_df.head()}")
            logging.info(f"Target features (train):\n{target_feature_train_df.head()}")
            logging.info(f"Input features (test):\n{input_feature_test_df.head()}")
            logging.info(f"Target features (test):\n{target_feature_test_df.head()}")

            # Check for any missing columns in the transformation process
            missing_columns = [col for col in ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"] if col not in input_feature_train_df.columns]
            if missing_columns:
                raise KeyError(f"Missing columns in the train dataframe: {missing_columns}")

            missing_columns_test = [col for col in ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"] if col not in input_feature_test_df.columns]
            if missing_columns_test:
                raise KeyError(f"Missing columns in the test dataframe: {missing_columns_test}")

            # Apply preprocessing transformations
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Label encode the target feature
            label_encoder = LabelEncoder()
            target_feature_train_arr = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)

            # Debug: Print the shapes of the arrays before concatenation
            logging.info(f"Shape of input_feature_train_arr: {input_feature_train_arr.shape}")
            logging.info(f"Shape of target_feature_train_arr: {target_feature_train_arr.shape}")
            logging.info(f"Shape of input_feature_test_arr: {input_feature_test_arr.shape}")
            logging.info(f"Shape of target_feature_test_arr: {target_feature_test_arr.shape}")

            # Ensure the target arrays are 2D for concatenation
            target_feature_train_arr = target_feature_train_arr.reshape(-1, 1)
            target_feature_test_arr = target_feature_test_arr.reshape(-1, 1)

            # Check dimensions before concatenation
            assert input_feature_train_arr.shape[0] == target_feature_train_arr.shape[0], \
            "Number of rows in input_feature_train_arr and target_feature_train_arr must match."
            assert input_feature_test_arr.shape[0] == target_feature_test_arr.shape[0], \
            "Number of rows in input_feature_test_arr and target_feature_test_arr must match."


            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_arr)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_arr)]

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
            raise CustomException(e, sys)



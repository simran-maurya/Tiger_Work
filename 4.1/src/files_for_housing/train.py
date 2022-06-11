'''This script is used for training the model(s).
   Input Arguments for this script are-
     1) Folder Path where train dataset is stored.
     2) Folder Path where model will be saved '''


import os
import pickle
import tarfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint
from six.moves import urllib
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor

from files_for_housing import dict_Config_logger_implementation_file as dc


def load_dataset(path):

      df=pd.read_csv(path)
      return df

def X_y_creation(df):
      y = df["median_house_value"].copy(deep=True)
      X = df.drop(["median_house_value"], axis=1)
      return X,y

def create_and_save_LR_model():

      #Creating Model
      lin_reg = LinearRegression()
      lin_reg.fit(X,y)

      #Saving Linear Regression Model
      model_name = type(lin_reg).__name__
      path = os.path.join(args.save_model, f"{model_name}.pkl")
      with open(path, "wb") as file:
        pickle.dump(lin_reg, file)

def create_and_save_DTR_model():
      # Training Decision Tree Regressor
      tree_reg = DecisionTreeRegressor(random_state=42)
      tree_reg.fit(X, y)

      #Saving Decision Tree Regressor
      model_name = type(tree_reg).__name__
      path = os.path.join(args.save_model, f"{model_name}.pkl")
      with open(path, "wb") as file:
        pickle.dump(tree_reg, file)


def create_and_save_RFR_model():

      #Training RandomForestRegressor
      random_forest = RandomForestRegressor()
      param_grid =[
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        {"bootstrap": [False],"n_estimators": [3, 10],"max_features": [2, 3, 4]},
      ]
      grid_search = GridSearchCV(
            random_forest,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=5,
            return_train_score=True,
      )
      grid_search.fit(X, y)

      #Saving RandomForestRegressor
      model_name = type(grid_search.best_estimator_).__name__
      path = os.path.join(args.save_model, f"{model_name}.pkl")
      with open(path, "wb") as file:
        pickle.dump(grid_search.best_estimator_, file)



if __name__=='__main__':

  #Taking Input Arguments

  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("-td","--train_dataset",help='Path to training dataset csv file.',default='data/processed_dataset/housing_train.csv')
  parser.add_argument("-sm","--save_model",help='Specify the path where you want to save the model',default='models/')
  parser.add_argument("--log-level", type=str, default="DEBUG")
  parser.add_argument("--no-console-log", action="store_true")
  parser.add_argument("--log-path", type=str, default="")
  args = parser.parse_args()

  # Creating the logger
  logger = dc.configure_logger(log_level=args.log_level,log_file=args.log_path,console=not args.no_console_log,name="train.py")

  # Loading the Test Dataset

  df = load_dataset(args.train_dataset)
  logger.debug("Loading the Test Dataset")


  # Creating X and y Variables

  X,y=X_y_creation(df)
  logger.debug("Creating X and y Variables")

  #Creating the Directory to save the Mode
  os.makedirs(args.save_model, exist_ok=True)
  logger.debug("Creating the Directory to save the Mode")

  # Training Linear Regressor Model and Saving it
  create_and_save_LR_model()
  logger.debug("Training Linear Regressor Model and Saving it")

  # Training Decision Tree Regressor Model and Saving it
  create_and_save_DTR_model()
  logger.debug("Training Decision Tree Regressor Model and Saving it")

  # Training Random Forest Regressor Model and Saving it
  create_and_save_RFR_model()
  logger.debug("Training Random Forest Regressor Model and Saving it")

  # Training And Saving Of Different Models done Successfully.
  logger.debug("Training And Saving Of Different Models done Successfully.")

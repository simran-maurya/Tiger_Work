'''This script is used to score the model(s) we have trained in train.py file.
   Input Arguments taken by this script are-
   1) Folder where models are stored.
   2) Folder where test dataset is stored.
   3) Folder where output will be stored.
'''

import os
import pickle
from argparse import ArgumentParser, Namespace
from glob import glob
from logging import Logger

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error

from files_for_housing import dict_Config_logger_implementation_file as dc


def load_dataset(path):
    df=pd.read_csv(path)
    return df

def X_y_creation(df):
    y = df["median_house_value"].copy(deep=True)
    X = df.drop(["median_house_value"], axis=1)
    return X,y

def models_list(model_folder):
    paths = glob(f"{model_folder}/*.pkl")
    paths = sorted(paths)
    models = []

    for path in paths:
        if os.path.isfile(path):
            model = pickle.load(open(path, "rb"))
            models.append(model)
    return models

def model_scoring(models_list):
    for model in models_list:
        model_name = type(model).__name__
        # Printing Model Names
        print(f"Model: {model_name}")

        # Creating scores dictionary having R2 score, RSME and MAE
        scores = {}
        scores["R2 score"] = model.score(X, y)
        y_hat = model.predict(X)

        if args.rmse:
            rmse = np.sqrt(mean_squared_error(y, y_hat))
            scores["RMSE"] = rmse

        if args.mae:
            mae = mean_absolute_error(y, y_hat)
            scores["MAE"] = mae

        # Printinf Score Dictionary

        print(scores)


if __name__ == "__main__":

    # Taking the Input Arguments

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--models",help='Specify the path where our models are stored',default='models/')
    parser.add_argument("-td","--test_dataset",help='Specify the path our test dataset is stored',default='data/processed_dataset/housing_test.csv')
    parser.add_argument("--rmse", action="store_true", help="Show RMSE.")
    parser.add_argument("--mae", action="store_true", help="Show MAE.")
    parser.add_argument("--log-level", type=str, default="DEBUG")
    parser.add_argument("--no-console-log", action="store_true")
    parser.add_argument("--log-path", type=str, default="")
    args = parser.parse_args()

    # Creating the logger
    logger = dc.configure_logger(log_level=args.log_level,log_file=args.log_path,console=not args.no_console_log,name="score.py")

    # Loading the Test Dataset

    df = load_dataset(args.test_dataset)
    logger.debug("Loading the Test Dataset")

    # Creating X and y Variables

    X,y=X_y_creation(df)
    logger.debug("Creating X and y Variables")

    # Loading models in a form of a list

    models=models_list(args.models)

    logger.debug("Loading models in a form of a list")


    #Scoring different Models
    model_scoring(models)
    logger.debug("Scoring different Models")





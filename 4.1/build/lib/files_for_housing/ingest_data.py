'''A script to download and create training and validation datasets.
   Different input arguments this script will take are-
    1) folder where data(raw and processed will be stored)
    2) Arguments for logging
'''
import logging
import logging.config
import os
import tarfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

from files_for_housing import dict_Config_logger_implementation_file as dc


def fetch_housing_data(housing_url, housing_path):
    os.makedirs(housing_path, exist_ok = True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def pre_process_data(df,imputer=None):
    df = pd.get_dummies(df, columns=["ocean_proximity"])

    if imputer is None:
        imputer = SimpleImputer(strategy="median")
        imputer.fit(df)

    data = imputer.transform(df)
    df = pd.DataFrame(data, columns=df.columns, index=df.index)

    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]

    return (df, imputer)


if __name__ == "__main__":

    # Taking input arguments using 'argparse' module

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-ad","--actualdataset",help='Specify the path where you want to store your actual downloaded dataset',default='data/actual_dataset/')
    parser.add_argument("-pd","--processeddataset",help='Specify the path where you want to store your processed dataset',default='data/processed_dataset/')
    parser.add_argument("--log-level", type=str, default="DEBUG")
    parser.add_argument("--no-console-log", action="store_true")
    parser.add_argument("--log-path", type=str, default="")
    args = parser.parse_args()

    # Creating the logger
    logger = dc.configure_logger(log_level=args.log_level,log_file=args.log_path,console=not args.no_console_log,name='ingest_data.py')



    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_PATH = args.actualdataset
    # actual_dataset is folder where our actual dataset is stored
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    #Fetching the dataset

    fetch_housing_data(HOUSING_URL, HOUSING_PATH)
    logger.debug("Fetched housing data.")

    #Loading the dataste
    housing = load_housing_data(HOUSING_PATH)
    logger.debug("Loaded housing data into a csv file.")

    #Creating train_set and test_test
    #Using Stratified shuffling approach with 'median_income' parameter
    #Converting 'median_income' to categorial form
    housing["income_cat"] = pd.cut(housing["median_income"], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    logger.debug("Splitted data into train and test set using Startisifed Shuffling.")

    #Preprocessing dataset
    logger.debug("Preprocessing...")
    train_set, imputer = pre_process_data(strat_train_set)
    test_set, _ = pre_process_data(strat_test_set, imputer)
    logger.debug("Preprocessing finished.")
    # Preprocessing finished.

    # Saving datasets
    logger.debug("Saving datasets.")
    os.makedirs(args.processeddataset, exist_ok = True)

    train_path = os.path.join(args.processeddataset, "housing_train.csv")
    train_set.to_csv(train_path)
    logger.debug(f"Preprocessed train datasets stored at {train_path}.")

    test_path = os.path.join(args.processeddataset, "housing_test.csv")
    test_set.to_csv(test_path)
    logger.debug(f"Preprocessed test datasets stored at {test_path}.")

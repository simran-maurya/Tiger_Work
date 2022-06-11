import os
import tarfile

from files_for_housing import ingest_data as id
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from housing_price_shreya_kanodia import ingest_data as id
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit

# Testing 1st function block of ingest_data.py

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = 'data/actual_dataset/'
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def test_fetch_housing_data():
    id.fetch_housing_data(HOUSING_URL, HOUSING_PATH)
    assert os.path.isfile("data/actual_dataset/housing.tgz")
    assert os.path.isfile("data/actual_dataset/housing.csv")

# Testing 2nd function block of ingest_data.py

def test_load_housing_data():
    obtained=id.load_housing_data(HOUSING_PATH)
    assert type(obtained) is pd.DataFrame

# Testing 3rd function block of ingest_data.py

def test_pre_process_data():
    housing_df = pd.read_csv("data/actual_dataset/housing.csv")
    housing_df["income_cat"] = pd.cut(housing_df["median_income"], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing_df, housing_df["income_cat"]):
        strat_train_set = housing_df.loc[train_index]
        strat_test_set = housing_df.loc[test_index]
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    train_set, imputer = id.pre_process_data(strat_train_set)
    test_set, _ = id.pre_process_data(strat_test_set)


    assert not train_set.isna().sum().sum()
    assert "ocean_proximity" not in train_set.columns
    assert "ocean_proximity" not in test_set.columns
    assert "rooms_per_household" in train_set.columns
    assert "rooms_per_household" in test_set.columns
    assert "population_per_household" in train_set.columns
    assert "population_per_household" in test_set.columns
    assert "bedrooms_per_room" in train_set.columns
    assert "bedrooms_per_room" in test_set.columns


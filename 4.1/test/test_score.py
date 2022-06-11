from re import A

import pandas as pd
from files_for_housing import score as s

HOUSING_PATH='data/processed_dataset/housing_test.csv'
model_folder='models/'

def test_load_dataset():
    obtained=s.load_dataset(HOUSING_PATH)
    # actual=pd.read_csv('data/processed_dataset/housing_test.csv')
    assert type(obtained) is pd.DataFrame

def test_X_y_creation():
    X, y = s.X_y_creation(pd.read_csv('data/processed_dataset/housing_train.csv'))
    assert len(X) == len(y)
    assert "median_house_value" not in X.columns
    assert not X.isna().sum().sum()
    assert len(y.shape) == 1

def test_models_list():
    models=s.models_list(model_folder)
    assert len(models) == 3

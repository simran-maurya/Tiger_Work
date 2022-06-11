# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

The following techniques have been used:

 - Linear regression
 - Decision Tree
 - Random Forest

## Installation:
### Prerequisites:
Prerequisite dependencies are stored in `enviornment_file/env.yml`. To setup the conda environment:

`$ conda env create --file enviornment_file/env.yml`

`$ conda activate mle-dev`

### Setup:
For editable install:
`$ pip install -e .`

For normal install:
`$ pip install .`

## Run code:
### To download and process data:
`$ python src/files_for_housing/ingest_data.py -r data/actual_dataset/ -p data/processed_dataset/`
### To train the models:
`$ python src/files_for_housing/train.py -d data/processed_dataset/housing_train.csv -m models/`
### To score trained models:
`$ python src/files_for_housing/score.py -d data/processed_dataset/housing_test.csv -m models/`
### Note:
You can get information on command line arguments for each of the above scripts using `-h` or `--help`. For example:

`$ python src/files_for_housing/train.py --help`
## Steps performed:
 - We prepared and cleaned the data.
 - We checked and imputed missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

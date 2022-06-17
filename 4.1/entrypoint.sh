#!/usr/bin/env bash

conda run -n mle-dev python -m files_for_housing.ingest_data -ad data/actual_dataset/ -pd data/processed_dataset/ --log-path logs/log.txt
conda run -n mle-dev python -m files_for_housing.train -td data/processed_dataset/housing_train.csv -sm models/ --log-path logs/log.txt
conda run -n mle-dev python -m files_for_housing.score -td data/processed_dataset/housing_test.csv -m models/ --log-path logs/log.txt --rmse --mae
conda run -n mle-dev python -m pytest /home/files_for_housing-0.1/test/unit_tests

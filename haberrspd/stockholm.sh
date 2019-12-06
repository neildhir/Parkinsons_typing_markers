#!/bin/bash

# MJFF

upper_count=300
lower_count=100
spanish_data=SpanishData-preprocessed.csv

## char
python hyperopt.py --which_dataset MJFF --which_information char --unique_ID english_mjff_attempt_1 --csv_file EnglishData-preprocessed_attempt_1.csv --round_limit $upper_count --save_model y

python hyperopt.py --which_dataset MJFF --which_information char --unique_ID english_mjff_attempt_2 --csv_file EnglishData-preprocessed_attempt_2.csv --round_limit $upper_count --save_model y

python hyperopt.py --which_dataset MJFF --which_information char --unique_ID spanish_mjff --csv_file $spanish_data --round_limit $upper_count --save_model y

## char_time
python hyperopt.py --which_dataset MJFF --which_information char_time --unique_ID english_mjff_attempt_1 --csv_file EnglishData-preprocessed_attempt_1.csv --round_limit $lower_count --save_model y

python hyperopt.py --which_dataset MJFF --which_information char_time --unique_ID english_mjff_attempt_2 --csv_file EnglishData-preprocessed_attempt_2.csv --round_limit $lower_count --save_model y

python hyperopt.py --which_dataset MJFF --which_information char_time --unique_ID spanish_mjff --csv_file $spanish_data --round_limit $lower_count --save_model y

## char_time_space
python hyperopt.py --which_dataset MJFF --which_information char_time_space --unique_ID english_mjff_attempt_1 --csv_file EnglishData-preprocessed_attempt_1.csv --round_limit $lower_count --save_model y

python hyperopt.py --which_dataset MJFF --which_information char_time_space --unique_ID english_mjff_attempt_2 --csv_file EnglishData-preprocessed_attempt_2.csv --round_limit $lower_count --save_model y

python hyperopt.py --which_dataset MJFF --which_information char_time_space --unique_ID spanish_mjff --csv_file $spanish_data --round_limit $lower_count --save_model y

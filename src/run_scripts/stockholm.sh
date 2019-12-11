#!/bin/bash

# MJFF

upper_count=100
lower_count=100
spanish_data=SpanishData-preprocessed.csv
attempt_1=EnglishData-preprocessed_attempt_1.csv
attempt_2=EnglishData-preprocessed_attempt_2.csv

## char
python hyperopt.py --which_dataset MJFF --which_information char --unique_ID english_mjff_attempt_1 --csv_file $attempt_1 --round_limit $upper_count --save_model y

python hyperopt.py --which_dataset MJFF --which_information char --unique_ID english_mjff_attempt_2 --csv_file $attempt_2 --round_limit $upper_count --save_model y

python hyperopt.py --which_dataset MJFF --which_information char --unique_ID spanish_mjff --csv_file $spanish_data --round_limit $upper_count --save_model y

## char_time
python hyperopt.py --which_dataset MJFF --which_information char_time --unique_ID english_mjff_attempt_1 --csv_file $attempt_1 --round_limit $lower_count --save_model y

python hyperopt.py --which_dataset MJFF --which_information char_time --unique_ID english_mjff_attempt_2 --csv_file $attempt_2 --round_limit $lower_count --save_model y

python hyperopt.py --which_dataset MJFF --which_information char_time --unique_ID spanish_mjff --csv_file $spanish_data --round_limit $lower_count --save_model y

## char_time_space
python hyperopt.py --which_dataset MJFF --which_information char_time_space --unique_ID english_mjff_attempt_1 --csv_file $attempt_1 --round_limit $lower_count --save_model y

python hyperopt.py --which_dataset MJFF --which_information char_time_space --unique_ID english_mjff_attempt_2 --csv_file $attempt_2 --round_limit $lower_count --save_model y

python hyperopt.py --which_dataset MJFF --which_information char_time_space --unique_ID spanish_mjff --csv_file $spanish_data --round_limit $lower_count --save_model y

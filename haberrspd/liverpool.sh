#!/bin/bash

# MRC

upper_count=300
lower_count=150
attempt_1=EnglishData-preprocessed_attempt_1.csv
attempt_2=EnglishData-preprocessed_attempt_2.csv

## char
python hyperopt.py --which_dataset MRC --which_information char --unique_ID english_mrc_attempt_1 --csv_file $attempt_1 --round_limit $upper_count --save_model y

python hyperopt.py --which_dataset MRC --which_information char --unique_ID english_mrc_attempt_2 --csv_file $attempt_2 --round_limit $upper_count --save_model y

## char_time
python hyperopt.py --which_dataset MRC --which_information char_time --unique_ID english_mrc_attempt_1 --csv_file $attempt_1 --round_limit $lower_count --save_model y

python hyperopt.py --which_dataset MRC --which_information char_time --unique_ID english_mrc_attempt_2 --csv_file $attempt_2 --round_limit $lower_count --save_model y

## char_time_space
python hyperopt.py --which_dataset MRC --which_information char_time_space --unique_ID english_mrc_attempt_1 --csv_file $attempt_1 --round_limit $lower_count --save_model y

python hyperopt.py --which_dataset MRC --which_information char_time_space --unique_ID english_mrc_attempt_2 --csv_file $attempt_2 --round_limit $lower_count --save_model y


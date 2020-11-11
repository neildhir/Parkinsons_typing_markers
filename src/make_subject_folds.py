#!/usr/bin/env python

"""
<Name of module>
Date created:  2020-03-09

Authors: Mathias Edman mathias@kamin.ai
 
Description:
    <This is a story about a man that wanted a big journey.>
 
Usage: <python script.py input_data output_data or ./script.py input_data output_data>
 
"""


import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import argparse


def main(data_path, num_folds, output_name):
    df = pd.read_csv(data_path)
    grouped_df = df.groupby(['Participant_ID', 'Sentence_ID']).size().reset_index()

    assert grouped_df[0].max() == 1, 'Duplicate sentences!'

    pd_patients = df[df['Diagnosis']==1]['Participant_ID'].unique()
    healthy_controls = df[df['Diagnosis']==0]['Participant_ID'].unique()

    fold_df = grouped_df[['Participant_ID', 'Sentence_ID']].copy()
    fold_df['fold'] = -1 * np.ones(len(fold_df), dtype=int)

    kf = KFold(n_splits=num_folds, shuffle=True)
    fold_idx = 0
    for train, test in kf.split(pd_patients):
        fold_df.loc[fold_df['Participant_ID'].isin(pd_patients[test]), 'fold'] = fold_idx
        fold_idx += 1


    fold_idx = 0
    for train, test in kf.split(healthy_controls):
        fold_df.loc[fold_df['Participant_ID'].isin(healthy_controls[test]), 'fold'] = fold_idx
        fold_idx += 1


    fold_df.to_csv(output_name, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path',
                        metavar='',
                        type=str,
                        required=True,
                        help='path to data')
    parser.add_argument('-n', '--num_folds',
                        metavar='',
                        type=int,
                        required=True,
                        help='path to data')
    parser.add_argument('-o', '--output_name',
                        metavar='',
                        type=str,
                        required=True,
                        help='name of output file')
    args = parser.parse_args()

    main(data_path=args.data_path,
         num_folds=args.num_folds,
         output_name=args.output_name)

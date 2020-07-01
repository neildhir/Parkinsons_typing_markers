#!/usr/bin/env python
import pandas as pd
import numpy as np
from scipy.stats import iqr
from pathlib import Path
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter


def split_sentence_in_word_pairs(x, label, fold, char2idx):
    '''
    Helper that divides preprocessed sentence matrix in to word_pair matrices for
    pretraining CNN filters
    '''

    space_locations = np.where(x[:, char2idx[' ']])[0]
    if space_locations[0] != 0:
        space_locations = np.insert(space_locations, 0, 0)

    word_pair_seq = []
    for i in range(len(space_locations) - 2):
        word_pair_seq.append(x[space_locations[i]:space_locations[i + 2]])

    y = [label] * len(word_pair_seq)
    fld = [fold] * len(word_pair_seq)
    return word_pair_seq, y, fld


def mk_wordpair_data(X, Y, folds, char2idx):
    word_pairs, labels, fold_array = [], [], []
    for x, y, fold in zip(X, Y, folds):
        wp, lbl, fld = split_sentence_in_word_pairs(x, y, fold, char2idx)
        word_pairs.extend(wp)
        labels.extend(lbl)
        fold_array.extend(fld)

    return np.array(word_pairs), np.array(labels), np.array(fold_array)


def participant_normalise(df, how='divmean'):
    how_allowed = ['divmean', 'standard', 'robust']

    assert how in how_allowed, '{} not in allowed values: {}'.format(how, how_allowed)
    df.IKI_timings = df.IKI_timings.apply(lambda x: np.asarray(x))

    for grp, data in df.groupby(by='Participant_ID'):

        cnt_data = np.concatenate(data.IKI_timings.values)

        if how == 'divmean':
            mean = cnt_data.mean()
            df.loc[data.index, 'IKI_timings'] = (data.IKI_timings / mean)
        elif how == 'standard':
            mean = cnt_data.mean()
            std = cnt_data.std()
            df.loc[data.index, 'IKI_timings'] = (data.IKI_timings - mean) / std
        elif how == 'robust':
            median = np.median(cnt_data)
            iq_range = iqr(cnt_data)
            df.loc[data.index, 'IKI_timings'] = (data.IKI_timings - median) / iq_range

        else:
            raise ValueError

def sentence_normalise(df, how='divmean'):
    how_allowed = ['divmean', 'standard', 'robust']

    assert how in how_allowed, '{} not in allowed values: {}'.format(how, how_allowed)
    df.IKI_timings = df.IKI_timings.apply(lambda x: np.asarray(x))
    norm_data = []
    for index, data in df.iterrows():

        cnt_data = data.IKI_timings

        if how == 'divmean':
            mean = cnt_data.mean()
            iki_normalised = (data.IKI_timings / mean)
        elif how == 'standard':
            mean = cnt_data.mean()
            std = cnt_data.std()
            iki_normalised = (data.IKI_timings - mean) / std
        elif how == 'robust':
            median = np.median(cnt_data)
            iq_range = iqr(cnt_data)
            iki_normalised = (data.IKI_timings - median) / iq_range

        else:
            raise ValueError

        norm_data.append(iki_normalised)
    df['IKI_timings'] = norm_data




def mk_standard_dataset(df, char2idx):
    channels = len(char2idx) + 1
    X = []
    y = df.Diagnosis.values
    for idx, row in df.iterrows():

        PPTS_list = row.PPTS_list
        IKI_timings = row.IKI_timings
        x = np.zeros((len(PPTS_list), channels))
        for i, char in enumerate(PPTS_list):
            x[i, char2idx[char]] = 1

        x[1:, -1] = IKI_timings
        X.append(x)
    return np.asarray(X), y


def adjust_range(df, how='minmax'):
    how_allowed = ['minmax', 'robust']
    assert how in how_allowed, '{} not in allowed values: {}'.format(how, how_allowed)
    all_timings = np.concatenate(df.IKI_timings.values)

    if how == 'minmax':
        mini = all_timings.min()
        maxi = all_timings.max()
        df.IKI_timings = (df.IKI_timings - mini) / (maxi - mini)

    elif how == 'robust':
        median = np.median(all_timings)
        iq_range = iqr(all_timings)
        df.IKI_timings = (df.IKI_timings - median) / (iq_range)




    else:
        raise ValueError


def extract_folds(X_dict, y_dict, test_fold):
    x_test = X_dict[test_fold]
    y_test = y_dict[test_fold]

    x_train = []
    y_train = []
    for key, item in X_dict.items():
        if key != test_fold:
            x_train.extend(item)
            y_train.extend(y_dict[key])

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    return x_train, x_test, y_train, y_test


def make_experiment_dataset(data_path, fold_path, participant_norm, global_norm,sentence_norm = False):
    # read from file
    df = pd.read_csv(data_path)
    print(df.head())
    df['IKI_timings'] = df.IKI_timings.apply(lambda x: eval(x))
    df['IKI_timings_original'] = df['IKI_timings']
    df['PPTS_list'] = df.PPTS_list.apply(lambda x: eval(x))

    folds = pd.read_csv(fold_path)

    # merge data and fold
    only_subjects = folds[['Participant_ID', 'fold']].drop_duplicates()
    df = df.merge(only_subjects, on=['Participant_ID'])

    # make char2idx dict and store
    unique_chars = Counter(np.concatenate(df.PPTS_list.values)).most_common()
    char2idx = {char_count[0]: i for i, char_count in enumerate(unique_chars)}
    char2idx_path = data_path.parent / (data_path.stem + '_char2idx.json')
    with open(char2idx_path, 'w') as json_file:
        json.dump(char2idx, json_file)

    # normalise on subject level
    if sentence_norm:
        sentence_normalise(df, how=participant_norm)
    else:
        participant_normalise(df, how=participant_norm)

    # adjust range
    adjust_range(df, how=global_norm)

    # make wordpair dataset for training

    # pad


    # make sentence dataset for training
    X_sentence, y_sentence = mk_standard_dataset(df, char2idx)

    X_wordpair, y_wordpair, fold_wordpair = mk_wordpair_data(X_sentence,y_sentence,df.fold.values,char2idx)


    # print('WARNING MAXLEN 700')
    X_sentence = pad_sequences(X_sentence, maxlen=None, dtype='float16')
    X_wordpair = pad_sequences(X_wordpair, maxlen=None, dtype='float16')
    return (X_wordpair, y_wordpair, fold_wordpair), (X_sentence, y_sentence), df


if __name__ == "__main__":
    root = Path(r'C:\Users\Mathias\repos\habitual_errors_NLP\data\MJFF\preproc\char_time')

    data_path = root / 'EnglishData-preprocessed_attempt_1.csv'
    fold_path = root / 'mjff_english_5fold.csv'
    # char2idx_path = root / 'EnglishData-preprocessed_attempt_1_char2idx.json'

    wordpair_data, sentence_data, df = make_experiment_dataset(data_path, fold_path, participant_norm='robust',
                                                               global_norm='robust',sentence_norm = True)

    print('Done')

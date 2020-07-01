#!/usr/bin/env python
import pandas as pd
import numpy as np
from scipy.stats import iqr
from pathlib import Path
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter

def reduce_input(x):
    '''
    Refactor this method.. very very fast hack.
    '''
    single_char = []
    pre = '*'
    segment_list = []
    segment = []
    for i in range(0, len(x)):
        current = x[i]
        if current != pre:
            single_char.append(current)
            segment_list.append(segment)
            segment = []
        pre = current
        segment.append(current)
    segment_list.append(segment)
    del segment_list[0]

    count_list = [len(s) for s in segment_list]

    return single_char, count_list


def proc_frame(df):
    SINGLE_CHAR = []
    COUNT_LIST = []
    for idx, row in df.iterrows():
        single_char, count_list = reduce_input(row.Preprocessed_typed_sentence)

        SINGLE_CHAR.append(single_char)
        COUNT_LIST.append(count_list)

    df['single_char'] = SINGLE_CHAR
    df['count_list'] = COUNT_LIST


def participant_normalise(df, how='divmean'):
    how_allowed = ['divmean', 'standard', 'robust']

    assert how in how_allowed, '{} not in allowed values: {}'.format(how, how_allowed)
    df.count_list = df.count_list.apply(lambda x: eval(x))
    df.count_list = df.count_list.apply(lambda x: np.asarray(x))

    for grp, data in df.groupby(by='Participant_ID'):

        cnt_data = np.concatenate(data.count_list.values)

        if how == 'divmean':
            mean = cnt_data.mean()
            df.loc[data.index, 'count_list'] = (data.count_list / mean)
        elif how == 'standard':
            mean = cnt_data.mean()
            std = cnt_data.std()
            df.loc[data.index, 'count_list'] = (data.count_list - mean) / std
        elif how == 'robust':
            median = np.median(cnt_data)
            iq_range = iqr(cnt_data)
            df.loc[data.index, 'count_list'] = (data.count_list - median) / iq_range

        else:
            raise ValueError


def mk_dataset(df, char2idx):
    pad_len = df.single_char.apply(lambda x: len(x)).max()
    channels = len(char2idx) + 1

    X = {fld: [] for fld in df.fold.unique()}
    y = {fld: [] for fld in df.fold.unique()}
    for idx, row in df.iterrows():

        single_char = ''.join(row.single_char)
        count_list = row.count_list
        fld = row.fold

        idx_list = np.asarray([char2idx[char] for char in single_char])

        ## Get indices of spaces and add to start and end
        sl = np.where(idx_list == 0)[0]  # space locations
        if sl[0] != 0:
            sl = np.insert(sl, 0, 0)
        if sl[-1] != 0:
            sl = np.insert(sl, len(sl), len(single_char) - 1)

        space_ind = 0
        word_pairs = []
        word_pair_times = []
        while space_ind + 2 < len(sl):
            word_pairs.append(single_char[sl[space_ind]: sl[space_ind + 2] + 1])
            word_pair_times.append(count_list[sl[space_ind]: sl[space_ind + 2] + 1])
            space_ind += 1

        for word_pair, word_pair_time in zip(word_pairs, word_pair_times):

            x = np.zeros((len(word_pair), channels))
            for i, char_iki in enumerate(zip(word_pair, word_pair_time)):
                char, iki = char_iki

                x[i, char2idx[char]] = 1
                x[i, -1] = iki

            X[fld].append(x)
            y[fld].append(row.Diagnosis)
    return X, y, pad_len


def mk_standard_dataset(df, char2idx):
    pad_len = df.single_char.apply(lambda x: len(x)).max()
    channels = len(char2idx) + 1
    X = []
    y = df.Diagnosis.values
    for idx, row in df.iterrows():

        single_char = row.single_char
        count_list = row.count_list
        x = np.zeros((len(single_char), channels))
        for i, char in enumerate(single_char):
            x[i, char2idx[char]] = 1

        x[:, -1] = count_list
        X.append(x)
    return np.asarray(X), y


def adjust_range(df, how='minmax'):
    how_allowed = ['minmax', 'robust']
    assert how in how_allowed, '{} not in allowed values: {}'.format(how, how_allowed)
    all_timings = np.concatenate(df.count_list.values)

    if how == 'minmax':
        mini = all_timings.min()
        maxi = all_timings.max()
        df.count_list = (df.count_list - mini) / (maxi - mini)

    elif how == 'robust':
        median = np.median(all_timings)
        iq_range = iqr(all_timings)
        df.count_list = (df.count_list - median) / (iq_range)




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


def make_experiment_dataset(data_path, fold_path, char2idx_path, participant_norm, global_norm):
    # read from file
    with open(char2idx_path) as json_file:
        char2idx = json.load(json_file)

    df = pd.read_csv(data_path)
    folds = pd.read_csv(fold_path)

    # merge data and fold
    df = df.merge(folds, on=['Participant_ID', 'Sentence_ID'])

    # process frame to obtain character and timing sequences
    proc_frame(df)


    # normalise on subject level
    participant_normalise(df, how=participant_norm)


    # adjust range
    adjust_range(df, how=global_norm)

    # make wordpair dataset for training
    X_wordpair, y_wordpair, pad_len = mk_dataset(df, char2idx)
    # pad
    for key, data in X_wordpair.items():
        X_wordpair[key] = pad_sequences(data, maxlen=30, dtype='float16')

    # make sentence dataset for training
    X_sentence, y_sentence = mk_standard_dataset(df, char2idx)
    #print('WARNING MAXLEN 700')
    X_sentence = pad_sequences(X_sentence, maxlen=None, dtype='float16')

    return (X_wordpair, y_wordpair), (X_sentence, y_sentence), df



def make_experiment_datasetV2(data_path, fold_path, char2idx_path, participant_norm, global_norm):
    # read from file
    with open(char2idx_path) as json_file:
        char2idx = json.load(json_file)

    df = pd.read_csv(data_path)
    folds = pd.read_csv(fold_path)

    # merge data and fold
    df = df.merge(folds, on=['Participant_ID', 'Sentence_ID'])


    # normalise on subject level
    participant_normalise(df, how=participant_norm)

    # adjust range
    adjust_range(df, how=global_norm)

    # make wordpair dataset for training
    X_wordpair, y_wordpair, pad_len = mk_dataset(df, char2idx)
    # pad
    for key, data in X_wordpair.items():
        X_wordpair[key] = pad_sequences(data, maxlen=120, dtype='float16')
        #print(X_wordpair[key].shape)
    # make sentence dataset for training
    X_sentence, y_sentence = mk_standard_dataset(df, char2idx)
    #print('WARNING MAXLEN 700')
    X_sentence = pad_sequences(X_sentence, maxlen=None, dtype='float16')

    return (X_wordpair, y_wordpair), (X_sentence, y_sentence), df




if __name__ == "__main__":
    root = Path(r'C:\Users\Mathias\repos\habitual_errors_NLP\data\MJFF\preproc\char_time')

    data_path = root / 'EnglishData-preprocessed_attempt_1.csv'
    fold_path = root / 'EnglishData-preprocessed_attempt_1_fold.csv'
    char2idx_path = root / 'EnglishData-preprocessed_attempt_1_char2idx.json'

    wordpair_data, sentence_data, df = make_experiment_dataset(data_path, fold_path, char2idx_path)

    print('Done')

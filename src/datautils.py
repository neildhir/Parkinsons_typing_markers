#!/usr/bin/env python
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import iqr
from tensorflow.keras.preprocessing.sequence import pad_sequences

from gensim.models import KeyedVectors


def split_sentence_in_word_pairs(x, label, fold, char2idx, space_locations=None):
    """
    Helper that divides preprocessed sentence matrix in to word_pair matrices for
    pretraining CNN filters
    """

    if space_locations is None:
        space_locations = np.where(x[:, char2idx[" "]])[0]

    if space_locations[0] != 0:
        space_locations = np.insert(np.array(space_locations), 0, 0)

    word_pair_seq = []
    for i in range(len(space_locations) - 2):
        word_pair_seq.append(x[space_locations[i] : space_locations[i + 2]])

    y = [label] * len(word_pair_seq)
    fld = [fold] * len(word_pair_seq)
    return word_pair_seq, y, fld


def mk_char2vec_wordpair_data(X, Y, folds, space_locations):
    word_pairs, labels, fold_array = [], [], []
    i = 0
    for x, y, fold, sl in zip(X, Y, folds, space_locations):
        wp, lbl, fld = split_sentence_in_word_pairs(x, y, fold, char2idx=None, space_locations=sl)
        word_pairs.extend(wp)
        labels.extend(lbl)
        fold_array.extend(fld)
        i += 1

    return np.array(word_pairs), np.array(labels), np.array(fold_array)


def mk_wordpair_data(X, Y, folds, char2idx):
    word_pairs, labels, fold_array = [], [], []
    for x, y, fold in zip(X, Y, folds):
        wp, lbl, fld = split_sentence_in_word_pairs(x, y, fold, char2idx)
        word_pairs.extend(wp)
        labels.extend(lbl)
        fold_array.extend(fld)

    return np.array(word_pairs), np.array(labels), np.array(fold_array)


def participant_normalise(df, how="divmean"):
    how_allowed = ["divmean", "standard", "robust"]
    assert how in how_allowed, "{} not in allowed values: {}".format(how, how_allowed)

    cols2normalise = ["IKI_timings"]
    if "hold_time" in df.columns:
        cols2normalise = cols2normalise + ["hold_time"]
    if "pause_time" in df.columns:
        cols2normalise = cols2normalise + ["pause_time"]

    print("Normalising: {}".format(cols2normalise))

    for normcol in cols2normalise:
        df[normcol] = df[normcol].apply(lambda x: np.asarray(x))

        for grp, data in df.groupby(by="Participant_ID"):

            cnt_data = np.concatenate(data[normcol].values)

            if how == "divmean":
                mean = cnt_data.mean()
                df.loc[data.index, normcol] = data[normcol] / mean
            elif how == "standard":
                mean = cnt_data.mean()
                std = cnt_data.std()
                df.loc[data.index, normcol] = (data[normcol] - mean) / std
            elif how == "robust":
                median = np.median(cnt_data)
                iq_range = iqr(cnt_data)
                df.loc[data.index, normcol] = (data[normcol] - median) / iq_range

            else:
                raise ValueError


def sentence_normalise(df, how="divmean"):
    how_allowed = ["divmean", "standard", "robust"]
    assert how in how_allowed, "{} not in allowed values: {}".format(how, how_allowed)

    cols2normalise = ["IKI_timings"]
    if "hold_time" in df.columns:
        cols2normalise = cols2normalise + ["hold_time"]

    for normcol in cols2normalise:
        df[normcol] = df[normcol].apply(lambda x: np.asarray(x))
        norm_data = []
        for index, data in df.iterrows():

            cnt_data = data[normcol]

            if how == "divmean":
                mean = cnt_data.mean()
                normalised = data[normcol] / mean
            elif how == "standard":
                mean = cnt_data.mean()
                std = cnt_data.std()
                normalised = (data[normcol] - mean) / std
            elif how == "robust":
                median = np.median(cnt_data)
                iq_range = iqr(cnt_data)
                normalised = (data[normcol] - median) / iq_range

            else:
                raise ValueError

            norm_data.append(normalised)
        df[normcol] = norm_data


def mk_char2vec_dataset(df: pd.DataFrame, hold_time: bool):
    char2vec = KeyedVectors.load("./misc/cbow10_w3.wv")
    extra_channels = 1
    if hold_time:
        extra_channels += 2
    cv_size = len(char2vec["a"])
    channels = cv_size + extra_channels
    X = []
    y = df.Diagnosis.values
    space_locations = []
    for idx, row in df.iterrows():

        PPTS_list = row.PPTS_list
        IKI_timings = row.IKI_timings
        x = np.zeros((len(PPTS_list), channels))
        space_loc = []
        for i, char in enumerate(PPTS_list):
            if char == " ":
                space_loc.append(i)

            try:
                x[i, :cv_size] = char2vec[char]
            except KeyError:
                if char == "ω":
                    print("Could not find {} in vocabulary, filling with ones".format(char))
                    x[i, :cv_size] = np.ones(cv_size)
                else:
                    print("Could not find {} in vocabulary, filling with zeros".format(char))
                    x[i, :cv_size] = np.zeros(cv_size)
        x[1:, -1] = IKI_timings

        if hold_time:
            x[:, -2] = row.hold_time
            x[1:, -3] = row.pause_time

        X.append(x)
        space_locations.append(space_loc)
    return np.asarray(X), y, space_locations


def mk_standard_dataset(df: pd.DataFrame, char2idx: dict, hold_time: bool):
    extra_channels = 1
    if hold_time:
        extra_channels += 2
    channels = len(char2idx) + extra_channels
    X = []
    y = df.Diagnosis.values
    for idx, row in df.iterrows():

        PPTS_list = row.PPTS_list
        IKI_timings = row.IKI_timings
        x = np.zeros((len(PPTS_list), channels))
        for i, char in enumerate(PPTS_list):
            try:
                x[i, char2idx[char]] = 1
            except KeyError:
                print("Could not find {} in vocabulary, filling with unkchar = £".format(char))
                x[i, char2idx["£"]] = 1
        x[1:, -1] = IKI_timings

        if hold_time:
            x[:, -2] = row.hold_time
            x[1:, -3] = row.pause_time

        X.append(x)
    return np.asarray(X), y


def mk_timeonly_dataset(df: pd.DataFrame, hold_time: bool):
    channels = 1
    if hold_time:
        channels += 2

    X = []
    space_locations = []
    y = df.Diagnosis.values
    for idx, row in df.iterrows():
        PPTS_list = np.array(row.PPTS_list)
        IKI_timings = row.IKI_timings
        x = np.zeros((len(PPTS_list), channels))

        x[1:, 0] = IKI_timings

        if hold_time:
            x[:, 1] = row.hold_time
            x[1:, 2] = row.pause_time
        space_locations.append(np.where(PPTS_list == " ")[0])
        X.append(x)
    return np.asarray(X), y, space_locations


def adjust_range(df, how="minmax"):
    how_allowed = ["minmax", "robust"]
    assert how in how_allowed, "{} not in allowed values: {}".format(how, how_allowed)

    cols2normalise = ["IKI_timings"]
    if "hold_time" in df.columns:
        cols2normalise = cols2normalise + ["hold_time"]

    if "pause_time" in df.columns:
        cols2normalise = cols2normalise + ["pause_time"]

    print("Adjusting range for: {}".format(cols2normalise))

    for normcol in cols2normalise:
        df[normcol] = df[normcol].apply(lambda x: np.asarray(x))
        all_timings = np.concatenate(df[normcol].values)
        if how == "minmax":
            mini = all_timings.min()
            maxi = all_timings.max()
            df[normcol] = (df[normcol] - mini) / (maxi - mini)

        elif how == "robust":
            median = np.median(all_timings)
            iq_range = iqr(all_timings)
            df[normcol] = (df[normcol] - median) / (iq_range)

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


def make_experiment_dataset(
    data_path, fold_path, participant_norm, global_norm, sentence_norm=False, hold_time=False, feature_type="standard"
):
    # read from file
    df = pd.read_csv(data_path)
    df["IKI_timings"] = df.IKI_timings.apply(lambda x: eval(x))
    df["IKI_timings_original"] = df["IKI_timings"]
    df["PPTS_list"] = df.PPTS_list.apply(lambda x: eval(x))
    if "hold_time" in df.columns:
        df["hold_time"] = df.hold_time.apply(lambda x: eval(x))
        df["hold_time_original"] = df["hold_time"]

    if "pause_time" in df.columns:
        df["pause_time"] = df.pause_time.apply(lambda x: eval(x))
        df["pause_time_original"] = df["pause_time"]

    folds = pd.read_csv(fold_path)

    # merge data and fold
    only_subjects = folds[["Participant_ID", "fold"]].drop_duplicates()
    df = df.merge(only_subjects, on=["Participant_ID"])

    # make char2idx dict and store
    unique_chars = Counter(np.concatenate(df.PPTS_list.values)).most_common()
    char2idx = {char_count[0]: i for i, char_count in enumerate(unique_chars)}

    # normalise on subject level
    if sentence_norm:
        sentence_normalise(df, how=participant_norm)
    else:
        if participant_norm == "NONE":
            print("NO PARTICIPANT NORMALISATION")
        else:
            participant_normalise(df, how=participant_norm)

    # adjust range
    adjust_range(df, how=global_norm)

    # make wordpair dataset for training

    # pad

    # make sentence dataset for training

    if feature_type == "char2vec":
        X_sentence, y_sentence, space_locations = mk_char2vec_dataset(df, hold_time)
        X_wordpair, y_wordpair, fold_wordpair = mk_char2vec_wordpair_data(
            X_sentence, y_sentence, df.fold.values, space_locations
        )

    elif feature_type == "standard":
        X_sentence, y_sentence = mk_standard_dataset(df, char2idx, hold_time)
        X_wordpair, y_wordpair, fold_wordpair = mk_wordpair_data(X_sentence, y_sentence, df.fold.values, char2idx)
    elif feature_type == "timeonly":
        X_sentence, y_sentence, space_locations = mk_timeonly_dataset(df, hold_time)
        X_wordpair, y_wordpair, fold_wordpair = mk_char2vec_wordpair_data(
            X_sentence, y_sentence, df.fold.values, space_locations
        )

    # print('WARNING MAXLEN 700')
    X_sentence = pad_sequences(X_sentence, maxlen=None, dtype="float16")
    X_wordpair = pad_sequences(X_wordpair, maxlen=None, dtype="float16")
    return (X_wordpair, y_wordpair, fold_wordpair), (X_sentence, y_sentence), df, char2idx


if __name__ == "__main__":
    root = Path(r"C:\Users\Mathias\repos\habitual_errors_NLP\data\MJFF\preproc\char_time")

    data_path = root / "EnglishData-preprocessed_attempt_1.csv"
    fold_path = root / "mjff_english_5fold.csv"
    # char2idx_path = root / 'EnglishData-preprocessed_attempt_1_char2idx.json'

    wordpair_data, sentence_data, df = make_experiment_dataset(
        data_path, fold_path, participant_norm="robust", global_norm="robust", sentence_norm=True
    )

    print("Done")

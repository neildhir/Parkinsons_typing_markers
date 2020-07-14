#!/usr/bin/env python
from pathlib import Path
from datautilsV2 import make_experiment_dataset, extract_folds
from models import mk_cnn_model, mk_composite_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from copy import copy
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pandas as pd
import json


def run_experiment(data_path, fold_path, prefix, participant_norm, global_norm, sentence_norm=False, hold_time=False,
                   feature_type='standard'):
    res_cols = ['Participant_ID', 'Diagnosis', 'Sentence_ID', 'fold', 'PPTS_list',
                'IKI_timings_original',]# 'Attempt']
    if hold_time:
        res_cols = res_cols + ['hold_time_original']

    save_path = Path('../results')
    print(prefix)
    if not os.path.exists(save_path / prefix):
        os.makedirs(save_path / prefix)
        os.makedirs(save_path / prefix / 'logs')



    wordpair_data, sentence_data, df, char2idx = make_experiment_dataset(data_path, fold_path, participant_norm,
                                                                         global_norm, sentence_norm, hold_time, feature_type = feature_type)

    char2idx_path = save_path / prefix / 'char2idx.json'
    with open(char2idx_path, 'w') as json_file:
        json.dump(char2idx, json_file)

    print(df.shape)


    X_wordpair, y_wordpair, fold_wordpair = wordpair_data
    X_sentence, y_sentence = sentence_data

    df.to_csv(save_path / prefix / 'processed_data.csv', index=False)
    df.to_pickle(save_path / prefix / 'processed_data.pkl')

    class_weight = {0: 1, 1: 1.5}

    callbacks = [EarlyStopping(verbose=1, patience=16, restore_best_weights=True, monitor='val_loss'),
                 ReduceLROnPlateau(
                     monitor='val_loss', factor=0.5, patience=10, verbose=1, mode='auto',
                     min_delta=0.0001, cooldown=0, min_lr=0),
                 # TensorBoard(log_dir = Path('../logs'))
                 ]
    for test_fold in df.fold.unique():
        val_fold = test_fold + 1 if test_fold + 1 <= 4 else 0
        print('EVAL FOLD {}, WP valfold: {}'.format(test_fold, val_fold))
        res_df = copy(df[df['fold'] == test_fold][res_cols])


        test_mask = fold_wordpair == test_fold
        val_mask = fold_wordpair == val_fold
        train_mask = ~(test_mask | val_mask)

        x_train = X_wordpair[train_mask]
        y_train = y_wordpair[train_mask]
        x_val = X_wordpair[val_mask]
        y_val = y_wordpair[val_mask]
        #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

        input_shape = x_train.shape[1:]

        wordpair_model = mk_cnn_model(input_shape)
        wordpair_model.compile(Adam(1e-3), 'sparse_categorical_crossentropy', metrics=['accuracy'])
        history = wordpair_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=200, verbose=1,
                                     batch_size=16, shuffle=True,
                                     class_weight=class_weight, callbacks=callbacks)
        #bz = 64
        hist_df = pd.DataFrame(history.history)
        save_to = save_path / prefix / 'logs' / 'pretrain_f{}.csv'.format(test_fold)
        hist_df.to_csv(save_to, index=False)

        # Train sentence model
        # split data
        test_mask = df.fold == test_fold

        x_test = X_sentence[test_mask]
        y_test = y_sentence[test_mask]

        x_train = X_sentence[~test_mask]
        y_train = y_sentence[~test_mask]

        x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.2,shuffle = True)


        input_shape = x_train.shape[1:]
        sentence_model = mk_composite_model(input_shape)


        # Extract filter weights from worpair_model and freeze
        sentence_model.layers[0].set_weights(wordpair_model.layers[0].get_weights())
        sentence_model.layers[0].trainable = False

        sentence_model.layers[2].set_weights(wordpair_model.layers[2].get_weights())
        sentence_model.layers[2].trainable = False

        sentence_model.compile(Adam(1e-3), 'sparse_categorical_crossentropy', metrics=['accuracy'])
        history = sentence_model.fit(x_train, y_train, callbacks=callbacks, validation_data=(x_val,y_val), epochs=200, verbose=1,
                                     batch_size=16, shuffle=True, class_weight=class_weight)
        #bz = 128

        hist_df = pd.DataFrame(history.history)
        save_to = save_path / prefix / 'logs' / 'rough_f{}.csv'.format(test_fold)
        hist_df.to_csv(save_to, index=False)

        save_to = save_path / prefix / 'rough_{}.h5'.format(test_fold)
        sentence_model.save(str(save_to))
        pred = sentence_model.predict(x_test)
        res_df['rough'] = pred[:, 1]


        # unlock weights
        sentence_model.layers[0].trainable = True
        sentence_model.layers[2].trainable = True

        sentence_model.compile(Adam(1e-4), 'sparse_categorical_crossentropy', metrics=['accuracy'])
        history = sentence_model.fit(x_train, y_train, callbacks=callbacks, validation_data=(x_val,y_val), epochs=200, verbose=1,
                                     batch_size=32, shuffle=True, class_weight=class_weight)
        hist_df = pd.DataFrame(history.history)
        save_to = save_path / prefix / 'logs' / 'tuned_f{}.csv'.format(test_fold)
        hist_df.to_csv(save_to, index=False)

        save_to = save_path / prefix / 'tuned_{}.h5'.format(test_fold)
        sentence_model.save(str(save_to))
        pred = sentence_model.predict(x_test)
        res_df['tuned'] = pred[:, 1]
        res_df['check'] = y_test



        save_to = save_path / prefix / 'fold_{}.csv'.format(test_fold)
        res_df.to_csv(str(save_to))

        np.save(save_path / prefix / 'x_fold_{}.npy'.format(test_fold), x_test)
        np.save(save_path / prefix / 'y_fold_{}.npy'.format(test_fold), y_test)
        del wordpair_model
        del sentence_model

        K.clear_session()


if __name__ == '__main__':

    ds = 'MRC'

    if ds == 'MJFFENG':
        ### MJFF ENGLISH PATHS ###
        root = Path(r'C:\Users\Mathias\repos\habitual_errors_NLP\data\MJFF\preproc\char_time')

        data_path = root / 'EnglishData-preprocessed_attempt_1and2.csv'
        fold_path = root / 'mjff_english_5fold.csv'
        name = 'MJFFENG'
        ##########################
    elif ds == 'MJFFSPAN':

        ### MJFF SPANISH PATHS ###
        root = Path(r'C:\Users\Mathias\repos\habitual_errors_NLP\data\MJFF\preproc\char_time')

        data_path = root / 'SpanishData-preprocessed.csv'
        fold_path = root / 'mjff_spanish_5fold.csv'
        # char2idx_path = root / 'SpanishData-preprocessed_char2idx.json'
        name = 'MJFFSPAN'
        ##########################
    if ds == 'MRC':
        ### MJFF ENGLISH PATHS ###
        root = Path(r'C:\Users\Mathias\repos\habitual_errors_NLP\data\MRC\preproc\char_time')

        data_path = root / 'EnglishData-preprocessed_ALL.csv'
        fold_path = root / 'mrc_fold_all.csv'
        # char2idx_path = root / 'EnglishData-preprocessed_attempt_1_char2idx.json'
        name = 'MRC'
        ##########################

    ###PARAMS###
    participant_norm = 'robust'
    global_norm = 'robust'
    sentence_norm = False
    hold_time = True
    features = 'standard'

    assert features in ['standard', 'char2vec', 'timeonly']

    ############

    prefix = 'ALL_{}_P-{}_G-{}_S-{}_{}'.format(name, participant_norm, global_norm, int(sentence_norm),features)
    run_experiment(data_path, fold_path, prefix, participant_norm, global_norm, sentence_norm, hold_time = hold_time, feature_type = features)
    print('Done')

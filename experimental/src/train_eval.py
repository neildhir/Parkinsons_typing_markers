#!/usr/bin/env python
from pathlib import Path
from datautils import make_experiment_dataset, extract_folds
from models import mk_cnn_model, mk_composite_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from copy import copy
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def run_experiment(data_path,fold_path,char2idx_path,prefix,participant_norm,global_norm):
    res_cols = ['Participant_ID', 'Diagnosis', 'Sentence_ID', 'fold']

    save_path = Path('../results')
    print(prefix)
    if not os.path.exists(save_path / prefix):
        os.makedirs(save_path / prefix)


    wordpair_data, sentence_data, df = make_experiment_dataset(data_path, fold_path, char2idx_path, participant_norm,
                                                               global_norm)

    #lens = df['single_char'].apply(lambda x: len(x))
    #plt.hist(lens)
    #plt.show()
    #sys.exit()


    X_wordpair, y_wordpair = wordpair_data
    X_sentence, y_sentence = sentence_data
    df.to_csv(save_path / prefix / 'processed_data.csv',index = False)
    class_weight = {0: 1, 1: 2}








    callbacks = [EarlyStopping(verbose=1, patience=10, restore_best_weights=True)]
    for test_fold in df.fold.unique():
        print('EVAL FOLD {}'.format(test_fold))
        res_df = copy(df[df['fold']==test_fold][res_cols])

        # Train wordpair model
        x_train, x_test, y_train, y_test = extract_folds(X_wordpair,y_wordpair, test_fold)


        wordpair_model = mk_cnn_model()
        wordpair_model.compile(Adam(1e-3), 'sparse_categorical_crossentropy', metrics=['accuracy'])
        wordpair_model.fit(x_train, y_train, validation_split = 0.2, epochs=100, verbose=1, batch_size=64, shuffle=True,
                  class_weight=class_weight, callbacks=callbacks)



        # Train sentence model
        sentence_model = mk_composite_model()

        # Extract filter weights from worpair_model and freeze
        sentence_model.layers[0].set_weights(wordpair_model.layers[0].get_weights())
        sentence_model.layers[0].trainable = False

        sentence_model.layers[2].set_weights(wordpair_model.layers[2].get_weights())
        sentence_model.layers[2].trainable = False

        #split data
        test_mask = df.fold == test_fold

        x_test = X_sentence[test_mask]
        y_test = y_sentence[test_mask]

        x_train = X_sentence[~test_mask]
        y_train = y_sentence[~test_mask]

        sentence_model.compile(Adam(1e-3), 'sparse_categorical_crossentropy', metrics=['accuracy'])
        sentence_model.fit(x_train, y_train, callbacks=callbacks, validation_split=0.2, epochs=100, verbose=1,
                            batch_size=128, shuffle=True, class_weight=class_weight)

        save_to = save_path / prefix / 'rough_{}.h5'.format(test_fold)
        sentence_model.save(str(save_to))
        pred = sentence_model.predict(x_test)
        res_df['rough']=pred[:, 1]


        #unlock weights
        sentence_model.layers[0].trainable = True
        sentence_model.layers[2].trainable = True

        sentence_model.compile(Adam(1e-4), 'sparse_categorical_crossentropy', metrics=['accuracy'])
        sentence_model.fit(x_train, y_train, callbacks=callbacks, validation_split=0.2, epochs=100, verbose=1,
                            batch_size=128, shuffle=True, class_weight=class_weight)


        save_to = save_path / prefix / 'tuned_{}.h5'.format(test_fold)
        sentence_model.save(str(save_to))
        pred = sentence_model.predict(x_test)
        res_df['tuned'] = pred[:, 1]
        res_df['check'] = y_test

        save_to = save_path / prefix / 'fold_{}.csv'.format(test_fold)
        res_df.to_csv(str(save_to))





if __name__ == '__main__':


    ds = 'MJFFENG'

    if ds == 'MJFFENG':
        ### MJFF ENGLISH PATHS ###
        root = Path(r'C:\Users\Mathias\repos\habitual_errors_NLP\data\MJFF\preproc\char_time')

        data_path = root / 'EnglishData-preprocessed_attempt_1.csv'
        fold_path = root / 'mjff_english_5fold.csv'
        char2idx_path = root / 'EnglishData-preprocessed_attempt_1_char2idx.json'
        name = 'MJFFENG'
        ##########################
    elif ds == 'MJFFSPAN':

        ### MJFF SPANISH PATHS ###
        root = Path(r'C:\Users\Mathias\repos\habitual_errors_NLP\data\MJFF\preproc\char_time')

        data_path = root / 'SpanishData-preprocessed.csv'
        fold_path = root / 'mjff_spanish_5fold.csv'
        char2idx_path = root / 'SpanishData-preprocessed_char2idx.json'
        name = 'MJFFSPAN'
        ##########################
    if ds == 'MRC':
        ### MJFF ENGLISH PATHS ###
        root = Path(r'C:\Users\Mathias\repos\habitual_errors_NLP\data\MRC\preproc\char_time')

        data_path = root / 'EnglishData-preprocessed_attempt_1.csv'
        fold_path = root / 'mrc_5fold.csv'
        char2idx_path = root / 'EnglishData-preprocessed_attempt_1_char2idx.json'
        name = 'MRC'
        ##########################






    ###PARAMS###
    participant_norm='robust'
    global_norm='robust'

    ############

    prefix = 'newpp_{}_P-{}_G-{}'.format(name,participant_norm,global_norm)
    run_experiment(data_path,fold_path,char2idx_path,prefix,participant_norm,global_norm)
    print('Done')
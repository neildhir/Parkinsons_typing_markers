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
sys.path.append('../../src/gradcam')

from gradcam import compute_saliency


def run_experiment(data_path,fold_path,char2idx_path,prefix,participant_norm,global_norm):
    res_cols = ['Participant_ID', 'Diagnosis', 'Sentence_ID', 'fold']

    save_path = Path('../results')
    print(prefix)
    if not os.path.exists(save_path / prefix):
        os.makedirs(save_path / prefix)

    save_to = save_path / prefix

    print(list(save_to.glob('tuned_*')))


    wordpair_data, sentence_data, df = make_experiment_dataset(data_path, fold_path, char2idx_path, participant_norm,
                                                               global_norm)

    #lens = df['single_char'].apply(lambda x: len(x))
    #plt.hist(lens)
    #plt.show()
    #sys.exit()



    X_sentence, y_sentence = sentence_data
    #df.to_csv(save_path / prefix / 'processed_data.csv',index = False)








    model = mk_composite_model()
    df['conv1d_0'] = np.nan
    df['conv1d_1'] = np.nan
    for test_fold in df.fold.unique():
        model_file = save_to / 'tuned_{}.h5'.format(test_fold)
        model.load_weights(str(model_file))
        print(model.summary())

        test_mask = df.fold == test_fold

        x_test = X_sentence[test_mask]
        y_test = y_sentence[test_mask]

        x_train = X_sentence[~test_mask]
        y_train = y_sentence[~test_mask]

        l0_gcam = []
        l1_gcam = []
        jj= 0
        for x_input in x_test:
            out0 = compute_saliency(model, x_input, layer_name='conv1d', cls=1)
            l0_gcam.append(out0[0])
            out1 = compute_saliency(model, x_input, layer_name='conv1d_1', cls = 1)
            l1_gcam.append(out1[1])

            print(jj, '/', len(x_test))
            jj+=1

        df[test_mask,'conv1d_0'] = l0_gcam
        df[test_mask,'conv1d_1'] = l1_gcam



    df.to_csv(save_to / 'gcam.csv', index = False)







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

    prefix = 'xx_{}_P-{}_G-{}'.format(name,participant_norm,global_norm)
    run_experiment(data_path,fold_path,char2idx_path,prefix,participant_norm,global_norm)
    print('Done')
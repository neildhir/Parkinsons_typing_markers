#!/usr/bin/env python

"""
<Name of module>
Date created:  2020-03-03

Authors: Mathias Edman mathias@kamin.ai
 
Description:
    <This is a story about a man that wanted a big journey.>
 
Usage: <python script.py input_data output_data or ./script.py input_data output_data>
 
"""
import sys

sys.path.append("..")
#sys.path.append('src/')
#from utils.utils import encode_sentence, process_sentences
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pandas as pd
import json
import numpy as np
import argparse
from pathlib import Path
import os

from src.charCNN.data_utilities import create_training_data_keras


def main(data_path, model_file, fold_idx):
    model_file = Path(model_file)
    data_path = Path(data_path)

    save_dir = model_file.parents[0] / data_path.stem

    #DATA_ROOT = Path("../data/") / args.which_dataset.upper() / "preproc"
    DATA_ROOT = data_path.parents[1]
    which_information = data_path.parents[0].stem


    #read FIT params
    hyperopt_df = pd.read_csv(model_file.parents[0] / model_file.name.replace('model.json','results.csv'))
    best_model_idx = hyperopt_df['val_loss'].argmin()
    batch_size = hyperopt_df.loc[best_model_idx,'batch_size']
    learning_rate = hyperopt_df.loc[best_model_idx,'lr']
    control_class_weight = hyperopt_df.loc[best_model_idx,'control_class_weight']
    pd_class_weight = hyperopt_df.loc[best_model_idx,'pd_class_weight']


    X_train, X_val, X_test, y_train, y_val, y_test,test_subject_id,test_sentence_id, max_sentence_length, alphabet_size = create_training_data_keras(
        DATA_ROOT, which_information, args.data_path, test_fold_idx = fold_idx)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    with open(model_file, 'r') as json_file:
        architecture = json.load(json_file)
        model = model_from_json(json.dumps(architecture))




    '''
    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)
    model_json = model.to_json()
    model = model_from_json(model_json)
    '''

    model.compile(optimizer=tf.keras.optimizers.Adam(lr= learning_rate * 1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])


    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='auto',
                                  #min_delta=0.0001, cooldown=0, min_lr=0)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto',
                                   baseline=None, restore_best_weights=True)
    callbacks = [early_stopping]

    class_weights = {0: control_class_weight, 1: pd_class_weight}
    print(X_train.shape)
    print(X_val.shape)
    print(fold_idx)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=500, verbose=1, callbacks=callbacks,
              validation_data=(X_val, y_val), class_weight=class_weights)

    model_name = 'fold_{}.h5'.format(fold_idx)
    model.save(save_dir / model_name)

    p_test = model.predict(X_test)
    out_data = {'Participant_ID': test_subject_id, 'Sentence_ID': test_sentence_id, 'Diagnosis': y_test, 'Prediction': p_test[:,1]}
    out_df = pd.DataFrame(out_data)

    log_name = 'fold_{}.csv'.format(fold_idx)
    out_df.to_csv(save_dir / log_name, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path',
                        metavar='',
                        type=str,
                        required=True,
                        help='path to data')
    parser.add_argument('-n', '--n_folds',
                        metavar='',
                        type=int,
                        required=True,
                        help='path to fold file')
    parser.add_argument('-m', '--model_file',
                        metavar='',
                        type=str,
                        required=True,
                        help='path to model file')
    args = parser.parse_args()

    for i in range(0,args.n_folds):
        print('EVALUATING FOLD: ', i)
        main(data_path=args.data_path,
             fold_idx = i,
             model_file=args.model_file)

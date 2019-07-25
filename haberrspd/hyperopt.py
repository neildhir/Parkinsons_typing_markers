import argparse
import datetime
import os
import sys
from pathlib import Path
import tensorflow as tf

sys.path.append("..")


import numpy as np
import talos as ta
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam, Nadam  # Which optimisers to consider
from numpy import asarray, vstack
from sklearn.utils import class_weight
from talos import Deploy, Predict, Restore

# This block is important if we want the memory to grow on the GPU, and not block allocate the whole thing
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import haberrspd.charCNN.globals  # Global params live here
from haberrspd.charCNN.data_utils_tf import create_training_data_keras
from haberrspd.charCNN.models_tf import char_cnn_model_talos



# --- PARSE ADDITIONAL USER SETTINGS

parser = argparse.ArgumentParser(description='CNN text classifier.')
# dataset
parser.add_argument('-dataset',
                    type=str,
                    default="EnglishData-preprocessed.csv",
                    help='Dataset to use for hyperparam optimisation [default is EnglishData-preprocessed.csv i.e. time + char information.]')
args = parser.parse_args()


# --- LOAD DATA

DATA_ROOT = Path("../data/") / "MJFF" / "preproc"  # Note the relative path
X_train, X_test, y_train, y_test, max_sentence_length, alphabet_size = \
    create_training_data_keras(DATA_ROOT, args.dataset)

# Class weights are dynamic as the data-loader is stochastic and changes with each run.
class_weights = dict(zip([0, 1],
                         class_weight.compute_class_weight('balanced', list(set(y_train)), y_train)))


# --- HYPERPARMETERS TO OPTIMISE

"""
Dynamic optimisation algoritns to choose from in keras.

'sgd': SGD,
'rmsprop': RMSprop,
'adagrad': Adagrad,
'adadelta': Adadelta,
'adam': Adam,
'adamax': Adamax,
'nadam': Nadam
"""


optimisation_parameters = {
    'lr': (0.1, 10, 5),  # Learning rate
    'conv_output_space': [4, 8, 16],  # ,8],
    'number_of_large_filters': [1, 2, 4],
    'number_of_small_filters': [1, 2, 4],
    'large_filter_length': [20, 40, 80, 160],
    'small_filter_length': [5, 10, 20],
    'pool_length': [2, 4],
    'dense_units_layer_3': [32],
    'dense_units_layer_2': [16],
    'batch_size': [32],
    'epochs': [100],
    'dropout': [0.05, 0.1, 0.2],  # ,0.1,0.2],
    'conv_kernel_initializer': ['uniform'],
    'conv_bias_initializer': ['uniform'],
    'dense_kernel_initializer': ['uniform'],
    'dense_bias_initializer': ['uniform'],
    'optimizer': [Adam],  # If used this way, these have to explicitly imported
    'loss': ['binary_crossentropy'],  # Loss functions
    'conv_activation': ['relu'],
    'dense_activation': ['relu'],
    'last_activation': ['sigmoid'],

    # Stationary parameters, i.e. do not get optimised
    'max_sentence_length': [max_sentence_length],
    'alphabet_size': [alphabet_size],
    'control_class_weight': [class_weights[0]],
    'pd_class_weight': [class_weights[1]],
}


# --- RUN OPTIMISATION

scanner = ta.Scan(x=X_train,
                  y=asarray(y_train).reshape(-1, 1),
                  x_val=X_test,
                  y_val=asarray(y_test).reshape(-1, 1),
                  model=char_cnn_model_talos,
                  disable_progress_bar=False,
                  params=optimisation_parameters)
# grid_downsample=0.01,  # Randomly samples 1% of the grid

# --- STORE RESULTS FOR FUTURE USE

# Query best learned model
probs = Predict(scanner)  # XXX: set options for the method properly

# Extract class-probabilities from best learned model
labels_and_label_probs = np.zeros((len(X_test), 2))
for i, (y, x) in enumerate(zip(y_test, X_test)):
    # Note that keras takes a 3D array and not the standard 2D, hence extra axis
    labels_and_label_probs[i, :] = [y, float(probs.predict(x[np.newaxis, :, :]))]

time_and_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Save array for later use
np.savetxt("label_and_label_probs_" + time_and_date + ".csv",
           labels_and_label_probs,
           fmt='%.15f',
           delimiter=",")

# Save whole model for later use
Deploy(scan_object=scanner, model_name='talos_best_model_' + time_and_date, metric='val_acc')

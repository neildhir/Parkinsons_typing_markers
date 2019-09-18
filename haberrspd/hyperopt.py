import argparse
import datetime
import os
import sys
from pathlib import Path

sys.path.append("..")
import numpy as np
import talos as ta
import tensorflow as tf
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
from haberrspd.charCNN.data_utils_tf import (create_training_data_keras,
                                             size_of_optimisation_space)
from haberrspd.charCNN.models_tf import char_cnn_model_talos




# --- PARSE ADDITIONAL USER SETTINGS

parser = argparse.ArgumentParser(description='CNN text classifier.')
parser.add_argument('-which_information',
                    type=str,
                    default="char_time",
                    help='Tells the model which type of data to use for the optimisation.')
parser.add_argument('-dataset',
                    type=str,
                    default="EnglishData-preprocessed.csv",
                    help='Dataset to use for hyperparam optimisation [default is EnglishData-preprocessed.csv i.e. time + char information.]')
parser.add_argument('-round_limit',
                    type=int,
                    default=100,
                    help='Puts a hard limit on the number of parameter permutations that will be entertained.')
parser.add_argument('-fraction_limit',
                    type=float,
                    default=None,
                    help=' The fraction of `params` that will be tested (Default is 5%).')
args = parser.parse_args()

# Fraction limit takes priority over the hard limit
if args.fraction_limit:
    args.round_limit = None

# --- LOAD DATA

DATA_ROOT = Path("../data/") / "MJFF" / "preproc"
X_train, X_test, y_train, y_test, max_sentence_length, alphabet_size = \
    create_training_data_keras(DATA_ROOT,
                               args.which_information,
                               args.dataset)

# Class weights are dynamic as the data-loader is stochastic and changes with each run.
class_weights = dict(zip([0, 1], class_weight.compute_class_weight('balanced', list(set(y_train)), y_train)))


# --- HYPERPARMETERS TO OPTIMISE

if args.which_information == 'char_time_space':
    # Add the coordinate size to alphabet size so that convolutions adapt
    alphabet_size += 2


# Note that the more parameters we have in here, the longer this is going to take.
optimisation_parameters = {
    'lr': (0.1, 10, 5),
    'conv_output_space': [8, 16, 32],  # ,8],
    'number_of_large_filters': [1, 2, 4],
    'number_of_small_filters': [1, 2, 4],
    'large_filter_length': [8,16,32], # When time is included [20,40,80,160], when not: [10,20,40,80]
    'small_filter_length': [2,4,8],#[5, 10, 20],
    'pool_length': [2],
    'dense_units_layer_3': [32, 64],
    'dense_units_layer_2': [16, 32],
    'batch_size': [16, 32, 64],
    'epochs': [200],
    'dropout': (0, 0.5, 5),
    'conv_padding': ['same'],
    'conv_kernel_initializer': ['uniform'],
    'conv_bias_initializer': ['uniform'],
    'dense_kernel_initializer': ['uniform'],
    'dense_bias_initializer': ['uniform'],
    'optimizer': [Adam, Nadam],  # If used this way, these have to explicitly imported
    'loss': ['logcosh', 'binary_crossentropy'],  # Loss functions
    'conv_activation': ['relu'],
    'dense_activation': ['relu'],
    'last_activation': ['sigmoid'],

    # Stationary parameters, i.e. do not get optimised
    'max_sentence_length': [max_sentence_length],
    'alphabet_size': [alphabet_size],
    'control_class_weight': [class_weights[0]],
    'pd_class_weight': [class_weights[1]],
}

space_size = size_of_optimisation_space(optimisation_parameters)
print('\nThe _RAW_ (i.e. not-yet-reduced) parameter permutation space is: {}\n'.format(space_size))
if args.fraction_limit:
    print('The reduced permutation optimisation space is of size: {}\n'.format(int(space_size * args.fraction_limit)))

# --- RUN OPTIMISATION

scanner = ta.Scan(x=X_train,
                  y=asarray(y_train).reshape(-1, 1),
                  x_val=X_test,
                  y_val=asarray(y_test).reshape(-1, 1),
                  round_limit=args.round_limit,  # Hard limit on the number of permutations we test
                  fraction_limit=args.fraction_limit,  # Percentage of permutation space to explore
                  model=char_cnn_model_talos,
                  disable_progress_bar=False,
                  params=optimisation_parameters)

# --- STORE RESULTS FOR FUTURE USE

my_dir ='../results/' + args.which_information
if not os.path.exists(my_dir):
    os.makedirs(my_dir)

# Query best learned model
probs = Predict(scanner)  # XXX: set options for the method properly

# Extract class-probabilities from best learned model
labels_and_label_probs = np.zeros((len(X_test), 2))
for i, (y, x) in enumerate(zip(y_test, X_test)):
    # Note that keras takes a 3D array and not the standard 2D, hence extra axis
    labels_and_label_probs[i, :] = [y, float(probs.predict(x[np.newaxis, :, :]))]

time_and_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Save array for later use
np.savetxt('../results/' + args.which_information + '/' + "label_and_label_probs_" + time_and_date + str(args.fraction_limit) + ".csv",
           labels_and_label_probs,
           fmt='%.15f',
           delimiter=",")

# Save whole model for later use
best_model_file = 'talos_best_model_' + time_and_date
Deploy(scan_object=scanner,
       model_name=best_model_file,
       metric='val_acc')

# Deploy is stupid, hence just move the file
os.rename(best_model_file + '.zip', my_dir + '/' + best_model_file + '.zip')

# Move talos' history file too
os.system('mv *.csv ../results/' + args.which_information)
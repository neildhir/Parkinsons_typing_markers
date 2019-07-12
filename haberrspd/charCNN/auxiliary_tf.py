import numpy as np
import keras
import matplotlib.pyplot as plt
import keras.backend as K
from keras.initializers import RandomNormal
from keras import callbacks
from keras.layers import Conv1D, Dense, GlobalMaxPool1D, MaxPooling1D, Dropout
from numpy import array, int64, ones
from pandas import read_csv
from sklearn.model_selection import train_test_split
from tensorflow import cast, float32, one_hot
import matplotlib


def binarize(x):
    """
    # XXX: change this to use a smaller dimensional representation of each character, see footnote 3 of paper `Character-aware neural language model`

    # Example from Torch: torch.nn.Embedding(big number, much smaller number)

    Parameters
    ----------
    x : string
        A character string from a sentence
    Returns
    -------
    Tensor
        A one-hot encoded tensor-representation of a character
    """
    # TODO: double-check how this encoding actually prints out [ 0 0 0 0 1 ...] etc
    return cast(one_hot(x,
                        K.alphabet_size,
                        on_value=1,
                        off_value=0,
                        axis=-1),
                float32)  # TODO: check precision


def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], K.alphabet_size


def binarize_outshape_sentence(in_shape):
    return in_shape[1],  K.alphabet_size


class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))


def create_mjff_data_objects(df):
    """
    Note that the interpretation here is that each document is comensurate with a subject
    in the dataset.
    """
    subject_documents = []  # Contains on the index all sentences typed by a particular subject
    subject_diagnoses = []  # Contains on the index, the PD diagnosis of a particular subject

    for i in df.Patient_ID.drop_duplicates():
        # Ensure that all sentences are lower-case (this improves inference further down the pipe)
        subject_documents.append(df.loc[(df.Patient_ID == i)].Preprocessed_typed_sentence.str.lower().tolist())

        # XXX: This returns one diagnosis per patient, but we may want one diagnosis per sentence
        subject_diagnoses.append(df.loc[(df.Patient_ID == i)].Diagnosis.drop_duplicates().tolist()[0])

    # Get the unique set of characters in the alphabet
    alphabet = set(''.join([item for sublist in subject_documents for item in sublist]))

    return subject_documents, subject_diagnoses, alphabet


def create_training_data(DATA_ROOT, data_string, which_level='sentence'):
    """
    This function creats one-hot encoded character -data for the document (=subject)
    classification model, as well as the sentence classification model.

    Parameters
    ----------
    DATA_ROOT : str
        Location of the MJFF data folder
    data_string : str
        The .csv file that we want to analyse
    which_level : str, optional
        Option sets if we are working on documents on sentence -level

    Returns
    -------
    tuple
        Contains the training and test data as well as some parameters

    Raises
    ------
    ValueError
        If we have passed a level option which doesn't exist.
    """
    assert type(data_string) is str
    assert which_level in ['sentence', 'document']

    df = read_csv(DATA_ROOT / data_string, header=0)  # MJFF data
    subject_documents, subjects_diagnoses, alphabet = create_mjff_data_objects(df)

    # Store alphabet size
    alphabet_size = len(alphabet)
    # Make the size available to the binarize functions
    setattr(K, 'alphabet_size', alphabet_size)

    print('Total number of characters:', alphabet_size)
    alphabet_indices = dict((c, i) for i, c in enumerate(alphabet))
    indices_alphabet = dict((i, c) for i, c in enumerate(alphabet))

    # Rounds (up) to nearest thousand
    max_sentence_length = round(df.Preprocessed_typed_sentence.apply(lambda x: len(x)).max(), -3)

    # Populate the training array
    if which_level == 'document':
        # Note here that the first MJFF data has each subject on 15 written sentences
        max_sentences_per_subject = 30
        # Make training data array
        X = ones((len(subject_documents), max_sentences_per_subject, max_sentence_length), dtype=int64) * -1
        # Make a target array from binary diagnoses
        y = array(subjects_diagnoses)
        for i, doc in enumerate(subject_documents):
            for j, sentence in enumerate(doc):
                if j < max_sentences_per_subject:
                    for t, char in enumerate(sentence[-max_sentence_length:]):
                        # XXX: this is in reverse order
                        X[i, j, (max_sentence_length - 1 - t)] = alphabet_indices[char]

        print('Sample X (encoded sentence): {}'.format(X[13, 2]))
        print('Target y (1: PD; 0: control): {}'.format(y[13]))

        # Chop up data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        return X_train, X_test, y_train, y_test, max_sentences_per_subject, max_sentence_length

    elif which_level == 'sentence':
        # Make training data array
        all_sentences = [item for sublist in subject_documents for item in sublist]
        X = ones((len(all_sentences), max_sentence_length), dtype=int64) * -1
        y = df.Diagnosis.tolist()
        for j, sentence in enumerate(all_sentences):
            for t, char in enumerate(sentence[-max_sentence_length:]):
                # This gets binarised on the fly, instead of storing the whole thing in memory
                # X[j, (max_sentence_length - 1 - t)] = alphabet_indices[char] # Characters in reversed order
                X[j, t] = alphabet_indices[char]

        # Chop up data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
        return X_train, X_test, y_train, y_test, max_sentence_length

    else:
        raise ValueError


# =============
# MODEL BLOCKS
# =============


def character_dense_dropout_block(flattened, units, dropout_rates):
    """
    To be used with char_cnn_model() from Zhang et al.'s paper.

    Parameters
    ----------
    flattened : [type]
        [description]
    units : [type]
        [description]
    rates : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    assert len(units) == len(dropout_rates)

    # Create multiple filters on the fly
    j = 0
    while units:
        unit = units.pop(0)

        # Assign appropriate activation function for dense layers
        if units:
            # List is not empty
            activation_func = 'relu'
        elif not units:
            # List is empty, alas we have reached the end of it and switch activation
            activation_func = 'sigmoid'

        # Dense
        flattened = Dense(unit,
                          kernel_initializer=RandomNormal(mean=0.0, stddev=0.05),
                          bias_initializer=RandomNormal(mean=0.0, stddev=0.05),
                          activation=activation_func)(flattened)

        # Dropout
        if dropout_rates[j]:
            # Only enters this logic if the entry is != None
            flattened = Dropout(dropout_rates[j])(flattened)

        # Increment index counter
        j += 1

    return flattened


def character_1D_convolution_maxpool_block_v2(embedded,
                                              nb_filters: list,
                                              filter_lengths: list,
                                              pool_lengths: list,
                                              **params: dict):
    """
    To be used with char_cnn_model() from Zhang et al.'s paper.

    Parameters
    ----------
    embedded : [type]
        A sentencen which has been one-hot encoded (on character-level)
    nb_filters : tuple, optional
        [description]
    filter_lengths : tuple, optional
        [description]
    pool_length : tuple, optional
        The pooling sizes, we use None if a layers is not meant to have pooling

    Returns
    -------
    [type]
        [description]
    """

    assert len(nb_filters) == len(filter_lengths) == len(pool_lengths)

    # Create multiple filters on the fly
    for i in range(len(nb_filters)):

        # Convolution
        embedded = Conv1D(filters=nb_filters[i],
                          kernel_size=filter_lengths[i],
                          kernel_initializer=params['kernel_initializer'],
                          bias_initializer=params['bias_initializer'],
                          activation=params['activation'])(embedded)

        # Max pooling
        if pool_lengths[i]:
            embedded = MaxPooling1D(pool_size=pool_lengths[i])(embedded)

    return embedded


def character_1D_convolution_block(embedded,
                                   nb_filter=(32, 64),
                                   filter_length=(3, 3),
                                   subsample=(2, 1),
                                   pool_length=(2, 2)):

    assert len(nb_filter) == len(filter_length) == len(subsample) == len(pool_length)

    # Create multiple filters on the fly
    for i in range(len(nb_filter)):
        # convolution
        embedded = Conv1D(filters=nb_filter[i],
                          kernel_size=filter_length[i],
                          padding='valid',
                          activation='relu',  # TODO: may be a more suitable activation func. here
                          kernel_initializer='glorot_normal',
                          strides=subsample[i])(embedded)
        # pooling
        if pool_length[i]:
            embedded = Dropout(0.1)(embedded)
            embedded = MaxPooling1D(pool_size=pool_length[i])(embedded)

    return embedded

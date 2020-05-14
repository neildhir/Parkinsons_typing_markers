"""
This file only hosts _complete models_ all other other functions are found in auxiliary_tf.py
"""

from keras.backend import int_shape, ndim
from keras.callbacks import EarlyStopping
from keras.initializers import RandomNormal
from keras.layers import (
    LSTM,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Lambda,
    MaxPooling1D,
    TimeDistributed,
    concatenate,
)
from keras.models import Model
from talos.model.normalizers import lr_normalizer

from .auxiliary_tf import (
    character_1D_convolution_block,
    character_1D_convolution_maxpool_block_v2,
    character_dense_dropout_block,
)
from .data_utilities import binarize, binarize_outshape, binarize_outshape_sentence


# from talos import live
# from talos.metrics.keras_metrics import fmeasure_acc


def char_lstm_cnn_model(max_sentences_per_subject, max_sentence_length):
    """
    Model from: “Exploring the Limits of Language Modeling”
    """

    # <<< sentence encoding >>>
    """
    This model starts from reading characters and forming concepts of “words”, then uses a bi-directional LSTM to read “words” as a sequence and account for their position.
    """

    # Set the sentence input
    input_sentence = Input(shape=(max_sentence_length,), dtype="int64")

    # Binarize the sentence's character on the fly, don't store in memory
    # char indices to one hot matrix, 1D sequence to 2D
    embedded = Lambda(binarize, output_shape=binarize_outshape)(input_sentence)
    # 1D convolutions
    embedded = character_1D_convolution_block(
        embedded, nb_filter=(32, 64), filter_length=(5, 5), subsample=(1, 1), pool_length=(2, 2)
    )
    # Sentence bi-directional LSTM
    bi_lstm_sent = Bidirectional(
        LSTM(32, return_sequences=False, dropout=0.15, recurrent_dropout=0.15, implementation=1)
    )(embedded)
    sent_encode = Dropout(0.3)(bi_lstm_sent)
    encoder = Model(inputs=input_sentence, outputs=sent_encode)
    encoder.summary()  # Model summary

    # <<< document encoding >>>
    """
    After that each sentence encoding is being passed through a second bi-directional LSTM that does the final document encoding.
    """

    # Set the document input
    document = Input(shape=(max_sentences_per_subject, max_sentence_length), dtype="int64")
    encoded = TimeDistributed(encoder)(document)
    bi_lstm_doc = Bidirectional(
        LSTM(32, return_sequences=False, dropout=0.15, recurrent_dropout=0.15, implementation=1)
    )(encoded)
    output = Dropout(0.3)(bi_lstm_doc)
    output = Dense(128, activation="relu")(output)
    output = Dropout(0.3)(output)
    output = Dense(2, activation="softmax")(output) #change to 2 and softmax

    return Model(outputs=output, inputs=document)


def char_cnn_model(max_sentence_length):
    """
    Model from: "Character-level Convolutional Networks for Text Classification"
                    / "Text Understanding from Scratch"
    """

    # Set the sentence input
    input_sentence = Input(shape=(max_sentence_length,), dtype="int64")

    # Lambda layer that will create a one-hot encoding of a sequence of characters on the fly. Holding one-hot encodings in memory is very inefficient.
    embedded = Lambda(binarize, output_shape=binarize_outshape)(input_sentence)

    # Convolutions and MaxPooling
    dim_output_space = 32  # Original from paper: 256
    number_of_filters = 2  # Orginal from paper: 6

    nb_filters = [dim_output_space] * number_of_filters
    filter_lengths = [10, 10]  # Original from paper: [7, 7, 3, 3, 3, 3]
    pool_lengths = [3, 3]  # Original from paper [3, 3, None, None, None, 3]

    # TODO: fix this, is currently relying on the talos function
    embedded = character_1D_convolution_maxpool_block_v2(embedded, nb_filters, filter_lengths, pool_lengths)
    # Reshaping to 1D array for further layers
    flattened = Flatten()(embedded)

    # Fully connected layers with (some) dropout
    dense_units = [16, 8, 1]  # Original from paper: [1024, 1024, num_classes]
    dropout_rates = [0.5, 0.5, None]
    final = character_dense_dropout_block(flattened, dense_units, dropout_rates)

    return Model(inputs=input_sentence, outputs=final)


def char_cnn_model_talos(X_train, y_train, X_test, y_test, params):
    """
    The same as "Character-level Convolutional Networks for Text Classification"
                    / "Text Understanding from Scratch"

    ...but with hyperparameter optimisation.
    """

    # Set the sentence input, which is a sentence which has been one-hot encoded
    input_sentence = Input(shape=(params["max_sentence_length"], params["alphabet_size"]), dtype="float32")

    # # Lambda layer that will create a one-hot encoding of a sequence of characters on the fly. Holding one-hot encodings in memory is very inefficient.
    # embedded = Lambda(binarize, output_shape=binarize_outshape)(input_sentence)

    # Convolutions and MaxPooling
    total_filter_count = params["number_of_large_filters"] + params["number_of_small_filters"]
    nb_filters = [params["conv_output_space"]] * total_filter_count

    # Large filters
    large_filter_lengths = [params["large_filter_length"]] * params["number_of_large_filters"]

    # Small filters
    small_filter_lengths = [params["small_filter_length"]] * params["number_of_small_filters"]

    # Pooling
    if total_filter_count == 2:
        pool_lengths = [params["pool_length"]] * total_filter_count
    elif total_filter_count > 2:
        # Follow the paper adopted in the original paper, but with dynamic assignment on size
        pool_lengths = [None] * total_filter_count  # Basically we do not want to pool too much
        pool_lengths[0], pool_lengths[1], pool_lengths[-1] = [params["pool_length"]] * 3
    else:
        raise ValueError

    embedded = character_1D_convolution_maxpool_block_v2(
        input_sentence,  # used to be: embedded
        nb_filters,
        large_filter_lengths + small_filter_lengths,
        pool_lengths,
        **params
    )

    # Reshaping to 1D array for further layers
    flattened = Flatten()(embedded)

    # Fully connected layers with (some) dropout
    dense_units = [params["dense_units_layer_3"], params["dense_units_layer_2"], 2] # NEED TO CHANGE 2 - > 1 to rollback to scalar output
    dropout_rates = [params["dropout"], params["dropout"], None]
    final = character_dense_dropout_block(flattened, dense_units, dropout_rates, **params)

    model = Model(inputs=input_sentence, outputs=final)

    # > Compile
    model.compile(
        loss=params["loss"],
        optimizer=params["optimizer"](lr=lr_normalizer(params["lr"], params["optimizer"])),
        metrics=["accuracy"],
    )  # , fmeasure_acc])
    # > Fit model


    out = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        verbose=0,  # Set to zero if using live plotting of losses
        class_weight={0: params["control_class_weight"], 1: params["pd_class_weight"]},
        # Monitor the loss with early stopping
        callbacks=[EarlyStopping(patience=15, min_delta=0.0001, monitor='val_loss', restore_best_weights=True, verbose=1)],
        batch_size=params["batch_size"],
        epochs=params["epochs"],
    )

    return out, model

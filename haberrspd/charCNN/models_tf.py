"""
This file only hosts _complete models_ all other other functions are found in auxiliary_tf.py
"""
from keras.layers import (Conv1D,
                          LSTM,
                          Bidirectional,
                          Flatten,
                          MaxPooling1D,
                          Dense,
                          Dropout,
                          Input,
                          Lambda,
                          TimeDistributed,
                          concatenate)
from keras.models import Model
from keras.initializers import RandomNormal
from keras.backend import int_shape, ndim
from haberrspd.charCNN.auxiliary_tf import (binarize, binarize_outshape, binarize_outshape_sentence,
                                            character_1D_convolution_maxpool_block_v2,
                                            character_dense_dropout_block,
                                            character_1D_convolution_block)


def char_lstm_cnn_model(max_sentences_per_subject,
                        max_sentence_length):
    """
    Model from: “Exploring the Limits of Language Modeling”
    """

    # <<< sentence encoding >>>
    """
    This model starts from reading characters and forming concepts of “words”, then uses a bi-directional LSTM to read “words” as a sequence and account for their position.
    """

    # Set the sentence input
    input_sentence = Input(shape=(max_sentence_length,), dtype='int64')

    # Binarize the sentence's character on the fly, don't store in memory
    # char indices to one hot matrix, 1D sequence to 2D
    embedded = Lambda(binarize, output_shape=binarize_outshape)(input_sentence)
    # 1D convolutions
    embedded = character_1D_convolution_block(embedded,
                                              nb_filter=(32, 64),
                                              filter_length=(5, 5),
                                              subsample=(1, 1),
                                              pool_length=(2, 2))
    # Sentence bi-directional LSTM
    bi_lstm_sent = Bidirectional(LSTM(32,
                                      return_sequences=False,
                                      dropout=0.15,
                                      recurrent_dropout=0.15,
                                      implementation=1))(embedded)
    sent_encode = Dropout(0.3)(bi_lstm_sent)
    encoder = Model(inputs=input_sentence, outputs=sent_encode)
    encoder.summary()  # Model summary

    # <<< document encoding >>>
    """
    After that each sentence encoding is being passed through a second bi-directional LSTM that does the final document encoding.
    """

    # Set the document input
    document = Input(shape=(max_sentences_per_subject, max_sentence_length), dtype='int64')
    encoded = TimeDistributed(encoder)(document)
    bi_lstm_doc = Bidirectional(LSTM(32,
                                     return_sequences=False,
                                     dropout=0.15,
                                     recurrent_dropout=0.15,
                                     implementation=1))(encoded)
    output = Dropout(0.3)(bi_lstm_doc)
    output = Dense(128, activation='relu')(output)
    output = Dropout(0.3)(output)
    output = Dense(1, activation='sigmoid')(output)

    return Model(outputs=output, inputs=document)


def char_cnn_model(max_sentence_length):
    """
    Model from: "Character-level Convolutional Networks for Text Classification"
                    / "Text Understanding from Scratch"
    """

    # Set the sentence input, which is a sentence which has been one-hot encoded
    input_sentence = Input(shape=(max_sentence_length,), dtype='int64')

    # Lambda layer that will create a one-hot encoding of a sequence of characters on the fly. Holding one-hot encodings in memory is very inefficient.
    embedded = Lambda(binarize, output_shape=binarize_outshape)(input_sentence)

    # Convolutions and MaxPooling
    dim_output_space = 32  # Original from paper: 256
    number_of_filters = 2  # Orginal from paper: 6

    nb_filters = [dim_output_space] * number_of_filters
    filter_lengths = [10, 10, ]  # Original from paper: [7, 7, 3, 3, 3, 3]
    pool_lengths = [3, 3, ]  # Original from paper [3, 3, None, None, None, 3]

    # TODO: fix this, is currently relying on the talos function
    embedded = character_1D_convolution_maxpool_block_v2(embedded,
                                                         nb_filters,
                                                         filter_lengths,
                                                         pool_lengths)
    # Reshaping to 1D array for further layers
    flattened = Flatten()(embedded)

    # Fully connected layers with (some) dropout
    dense_units = [16, 8, 1]  # Original from paper: [1024, 1024, num_classes]
    dropout_rates = [0.5, 0.5, None]
    final = character_dense_dropout_block(flattened, dense_units, dropout_rates)

    return Model(inputs=input_sentence, outputs=final)


def char_cnn_model_talos(X_train,
                         y_train,
                         X_test,
                         y_test,
                         params: dict,
                         non_opt_params: dict):
    """
    The same as "Character-level Convolutional Networks for Text Classification"
                    / "Text Understanding from Scratch"

    ...but with hyperparameter optimisation.
    """

    # Set the sentence input, which is a sentence which has been one-hot encoded
    input_sentence = Input(shape=(params['max_sentence_length'],), dtype='int64')

    # Lambda layer that will create a one-hot encoding of a sequence of characters on the fly. Holding one-hot encodings in memory is very inefficient.
    embedded = Lambda(binarize, output_shape=binarize_outshape)(input_sentence)

    # Block-creation of convolution layers
    embedded = character_1D_convolution_maxpool_block_v2(embedded, **params)

    # Reshaping to 1D array for further layers
    flattened = Flatten()(embedded)

    # Fully connected layers with (some) dropout
    dense_units = [params['dense_units_layer_3'], params['dense_units_layer_2'], 1]
    dropout_rates = [params['dropout'], params['dropout'], None]
    final = character_dense_dropout_block(flattened,
                                          dense_units,
                                          dropout_rates,
                                          **params)

    model = Model(inputs=input_sentence, outputs=final)

    # > Compile
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
    # > Fit model
    out = model.fit(X_train,
                    y_train,
                    validation_data=(X_test, y_test),
                    verbose=0,  # Set to zero if using live plotting of losses
                    class_weight=params['class_weights'],
                    batch_size=params['batch_size'],
                    epochs=params['epochs'])

    return out, model

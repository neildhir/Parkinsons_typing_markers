"""
This file only hosts _complete models_ all other other functions are found in auxiliary_tf.py
"""
from keras.layers import (Input,
                          Lambda,
                          Dense,
                          Bidirectional,
                          LSTM,
                          TimeDistributed,
                          concatenate)
from keras.models import Model
from haberrspd.charCNN.auxiliary_tf import (binarize,
                                            binarize_outshape,
                                            character_1D_convolution_block)


def lstm_cnn_model(max_sentences_per_subject,
                   max_sentence_length):

    document = Input(shape=(max_sentences_per_subject,
                            max_sentence_length),
                     dtype='int64')
    in_sentence = Input(shape=(max_sentence_length,),
                        dtype='int64')
    # Binarize the sentence's character on the fly, don't store in memory
    embedded = Lambda(binarize,
                      output_shape=binarize_outshape)(in_sentence)

    block2 = character_1D_convolution_block(embedded,
                                            (32, 64),
                                            filter_length=(5, 5),
                                            subsample=(1, 1),
                                            pool_length=(2, 2))

    block3 = character_1D_convolution_block(embedded,
                                            (32, 64),
                                            filter_length=(7, 5),
                                            subsample=(1, 1),
                                            pool_length=(2, 2))

    sent_encode = concatenate([block2, block3], axis=-1)
    encoder = Model(inputs=in_sentence, outputs=sent_encode)
    encoder.summary()  # Model summary

    encoded = TimeDistributed(encoder)(document)
    lstm_h = 16
    bidirectional_lstm = Bidirectional(LSTM(lstm_h,
                                            return_sequences=False,
                                            dropout=0.1,
                                            recurrent_dropout=0.1,
                                            implementation=1))(encoded)

    output = Dense(1, activation='sigmoid')(bidirectional_lstm)
    return Model(outputs=output,
                 inputs=document)

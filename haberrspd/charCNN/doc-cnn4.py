import os
import re
import sys

import keras.backend as K
import keras.callbacks
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import (LSTM, BatchNormalization, Bidirectional, Conv1D,
                          Dense, Dropout, GlobalMaxPool1D, Input, Lambda,
                          MaxPooling1D, TimeDistributed, concatenate)
from keras.models import Model
from keras.optimizers import Adam

# Extra options to make GPU work as required
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.tensorflow_backend.set_session(sess)


# TODO: change this to use a smaller dimensional representation of each character, see footnote 3 of paper `Character-aware neural language model`
# Example: torch.nn.Embedding(big number, much smaller number)
def binarize(x, sz=71):
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))


def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], 71


total = len(sys.argv)
cmdargs = str(sys.argv)

print("Script name: %s" % str(sys.argv[0]))
checkpoint = None
# Check that local path is correct
assert os.path.exists("../../data/test_data/labeledTrainData.tsv")
if len(sys.argv) == 2:
    if os.path.exists(str(sys.argv[1])):
        print("Checkpoint : %s" % str(sys.argv[1]))
        checkpoint = str(sys.argv[1])

# Load MJFF data
data = pd.read_csv("../data/MJFF/preproc/EnglishData-preprocessed.csv", header=0)
docs = []
diagnoses = []
# Note that the interpretation here is that each document is comensurate with a suject
# in the dataset.
for i in data.Patient_ID:
    docs.append(data.loc[(data.Patient_ID == i)].Preprocessed_typed_sentence.str.lower())
    diagnoses.append(data.loc[(data.Patient_ID == i)])

# Get the character alphabet
chars = set(''.join(docs))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print('Sample doc{}'.format(docs[1200]))

maxlen = 512
max_sentences = 15

X = np.ones((len(docs), max_sentences, maxlen), dtype=np.int64) * -1
y = np.array(diagnoses)

for i, doc in enumerate(docs):
    for j, sentence in enumerate(doc):
        if j < max_sentences:
            for t, char in enumerate(sentence[-maxlen:]):
                X[i, j, (maxlen - 1 - t)] = char_indices[char]

print('Sample X:{}'.format(X[1200, 2]))
print('y:{}'.format(y[1200]))

ids = np.arange(len(X))
np.random.shuffle(ids)

# shuffle
X = X[ids]
y = y[ids]

X_train = X[:20000]
X_test = X[22500:]

y_train = y[:20000]
y_test = y[22500:]


def char_block(in_layer,
               nb_filter=(64, 100),
               filter_length=(3, 3),
               subsample=(2, 1),
               pool_length=(2, 2)):
    block = in_layer
    for i in range(len(nb_filter)):

        block = Conv1D(filters=nb_filter[i],
                       kernel_size=filter_length[i],
                       padding='valid',
                       activation='tanh',
                       strides=subsample[i])(block)

        # block = BatchNormalization()(block)
        # block = Dropout(0.1)(block)
        if pool_length[i]:
            block = MaxPooling1D(pool_size=pool_length[i])(block)

    # block = Lambda(max_1d, output_shape=(nb_filter[-1],))(block)
    block = GlobalMaxPool1D()(block)
    block = Dense(128, activation='relu')(block)
    return block


max_features = len(chars) + 1
char_embedding = 40

document = Input(shape=(max_sentences, maxlen), dtype='int64')
in_sentence = Input(shape=(maxlen,), dtype='int64')

embedded = Lambda(binarize, output_shape=binarize_outshape)(in_sentence)

block2 = char_block(embedded, (128, 256), filter_length=(5, 5), subsample=(1, 1), pool_length=(2, 2))
block3 = char_block(embedded, (192, 320), filter_length=(7, 5), subsample=(1, 1), pool_length=(2, 2))

sent_encode = concatenate([block2, block3], axis=-1)
# sent_encode = Dropout(0.2)(sent_encode)

encoder = Model(inputs=in_sentence, outputs=sent_encode)
encoder.summary()

encoded = TimeDistributed(encoder)(document)

lstm_h = 92

lstm_layer = LSTM(lstm_h, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, implementation=0)(encoded)
lstm_layer2 = LSTM(lstm_h, return_sequences=False, dropout=0.1, recurrent_dropout=0.1, implementation=0)(lstm_layer)

# output = Dropout(0.2)(bi_lstm)
output = Dense(1, activation='sigmoid')(lstm_layer2)

model = Model(outputs=output, inputs=document)

model.summary()

if checkpoint:
    model.load_weights(checkpoint)

file_name = os.path.basename(sys.argv[0]).split('.')[0]

# Check if checkpoints dir exists, if not make it
if not os.path.exists('../../keras_checkpoints'):
    os.makedirs('../../keras_checkpoints')
check_cb = keras.callbacks.ModelCheckpoint('../../keras_checkpoints/' + file_name + '.{epoch:02d}-{val_loss:.2f}.hdf5',
                                           monitor='val_loss',
                                           verbose=0,
                                           save_best_only=True,
                                           mode='min')
earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=10,
                                             verbose=0,
                                             mode='auto')
optimizer = 'adam'
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(X_train,
          y_train,
          validation_data=(X_test, y_test),
          batch_size=20,
          epochs=30,
          shuffle=True,
          callbacks=[check_cb, earlystop_cb])

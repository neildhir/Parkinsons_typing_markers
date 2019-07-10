import keras.backend as K
from keras import callbacks
from keras.layers import Conv1D, Dense, GlobalMaxPool1D, MaxPooling1D
from numpy import array, int64, ones
from pandas import read_csv
from sklearn.model_selection import train_test_split
from tensorflow import cast, float32, one_hot


def binarize(x):
    """
    # XXX: change this to use a smaller dimensional representation of each character, see footnote 3 of paper `Character-aware neural language model`

    # Example from Torch: torch.nn.Embedding(big number, much smaller number)

    Parameters
    ----------
    x : [type]
        [description]
    Returns
    -------
    Tensor
        A one-hot encoded tensor-representation of a character
    """
    return cast(one_hot(x,
                        K.alphabet_size,
                        on_value=1,
                        off_value=0,
                        axis=-1),
                float32)  # TODO: check precision


def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], K.alphabet_size


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


def create_training_data(DATA_ROOT, data_string):
    print("This function is currently only designed for long-format data.")
    assert type(data_string) is str

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
    # Note here that the first MJFF data has each subject on 15 written sentences
    max_sentences_per_subject = 30

    # Make training data array
    X = ones((len(subject_documents), max_sentences_per_subject, max_sentence_length), dtype=int64) * -1
    # Make a target array from binary diagnoses
    y = array(subjects_diagnoses)

    # Populate the training array
    for i, doc in enumerate(subject_documents):
        for j, sentence in enumerate(doc):
            if j < max_sentences_per_subject:
                for t, char in enumerate(sentence[-max_sentence_length:]):
                    X[i, j, (max_sentence_length - 1 - t)] = alphabet_indices[char]

    print('Sample X:{}'.format(X[13, 2]))
    print('Target y:{}'.format(y[13]))

    # Chop up data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    return X_train, X_test, y_train, y_test, max_sentences_per_subject, max_sentence_length

# =============
# MODEL BLOCKS
# =============


def character_1D_convolution_block(in_layer,
                                   nb_filter=(32, 64),
                                   filter_length=(3, 3),
                                   subsample=(2, 1),
                                   pool_length=(2, 2)):

    assert len(nb_filter) == len(filter_length) == len(subsample) == len(pool_length)

    block = in_layer
    # Create multiple filters on the fly
    for i in range(len(nb_filter)):
        # convolution
        block = Conv1D(filters=nb_filter[i],
                       kernel_size=filter_length[i],
                       padding='valid',
                       activation='tanh',  # TODO: check if relu might be more appropriate here
                       strides=subsample[i])(block)
        # pooling
        if pool_length[i]:
            block = MaxPooling1D(pool_size=pool_length[i])(block)
    block = GlobalMaxPool1D()(block)
    block = Dense(64, activation='relu')(block)
    return block

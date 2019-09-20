import keras.backend as K
from keras import callbacks
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from numpy import array, int64, ones, hstack, pad, einsum, dstack
from pandas import read_csv
from sklearn.model_selection import train_test_split
from tensorflow import cast, float32, one_hot
import itertools


def size_of_optimisation_space(params):
    space = 1
    for attribute in params.keys():
        if type(attribute) == tuple:
            space *= params[attribute][-1]
        else:
            space *= len(params[attribute])

    return space


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
    return cast(
        one_hot(x, K.alphabet_size, on_value=1, off_value=0, axis=-1), float32
    )  # TODO: check precision


def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], K.alphabet_size


def binarize_outshape_sentence(in_shape):
    return in_shape[1], K.alphabet_size


class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get("loss"))
        self.accuracies.append(logs.get("acc"))


def create_mjff_data_objects(df):
    """
    Note that the interpretation here is that each document is comensurate with a subject
    in the dataset.
    """
    subject_documents = (
        []
    )  # Contains on the index all sentences typed by a particular subject
    subject_diagnoses = (
        []
    )  # Contains on the index, the PD diagnosis of a particular subject

    for i in df.Patient_ID.drop_duplicates():
        # Ensure that all sentences are lower-case (this improves inference further down the pipe)
        subject_documents.append(
            df.loc[(df.Patient_ID == i)]
            .Preprocessed_typed_sentence.str.lower()
            .tolist()
        )

        # XXX: This returns one diagnosis per patient, but we may want one diagnosis per sentence
        subject_diagnoses.append(
            df.loc[(df.Patient_ID == i)].Diagnosis.drop_duplicates().tolist()[0]
        )

    # Get the unique set of characters in the alphabet
    alphabet = set("".join([item for sublist in subject_documents for item in sublist]))

    return subject_documents, subject_diagnoses, alphabet


def create_training_data(DATA_ROOT, data_string, which_level="sentence"):
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
    assert which_level in ["sentence", "document"]

    df = read_csv(DATA_ROOT / data_string, header=0)  # MJFF data
    subject_documents, subjects_diagnoses, alphabet = create_mjff_data_objects(df)

    # Store alphabet size
    alphabet_size = len(alphabet)
    # Make the size available to the binarize functions
    setattr(K, "alphabet_size", alphabet_size)

    print("Total number of characters:", alphabet_size)
    alphabet_indices = dict((c, i) for i, c in enumerate(alphabet))
    indices_alphabet = dict((i, c) for i, c in enumerate(alphabet))

    # Rounds (up) to nearest thousand
    max_sentence_length = round(
        df.Preprocessed_typed_sentence.apply(lambda x: len(x)).max(), -3
    )

    # Populate the training array
    if which_level == "document":
        # Note here that the first MJFF data has each subject on 15 written sentences
        max_sentences_per_subject = 30
        # Make training data array
        X = (
            ones(
                (
                    len(subject_documents),
                    max_sentences_per_subject,
                    max_sentence_length,
                ),
                dtype=int64,
            )
            * -1
        )
        # Make a target array from binary diagnoses
        y = array(subjects_diagnoses)
        for i, doc in enumerate(subject_documents):
            for j, sentence in enumerate(doc):
                if j < max_sentences_per_subject:
                    for t, char in enumerate(sentence[-max_sentence_length:]):
                        # XXX: this is in reverse order
                        X[i, j, (max_sentence_length - 1 - t)] = alphabet_indices[char]

        print("Sample X (encoded sentence): {}".format(X[13, 2]))
        print("Target y (1: PD; 0: control): {}".format(y[13]))

        # Chop up data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True
        )
        return (
            X_train,
            X_test,
            y_train,
            y_test,
            max_sentences_per_subject,
            max_sentence_length,
        )

    elif which_level == "sentence":
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
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, shuffle=True
        )
        return X_train, X_test, y_train, y_test, max_sentence_length

    else:
        raise ValueError


def create_training_data_keras(DATA_ROOT, which_information, data_string):
    """
    This function creats one-hot encoded character -data for the document (=subject)
    classification model, as well as the sentence classification model. The functionality
    within is keras specific.

    Parameters
    ----------
    DATA_ROOT : str
        Location of the MJFF data folder
    data_string : str
        The .csv file that we want to analyse

    Returns
    -------
    tuple
        Contains the training and test data as well as some parameters
    """
    assert type(data_string) is str

    if which_information == "char_time_space":
        # Get relevant long-format data
        df = read_csv(DATA_ROOT / "char_time" / data_string, header=0)  # MJFF data
    else:
        df = read_csv(
            DATA_ROOT / which_information / data_string, header=0
        )  # MJFF data

    subject_documents, subjects_diagnoses, alphabet = create_mjff_data_objects(df)

    # Store alphabet size
    alphabet_size = len(alphabet)

    print("Total number of characters:", alphabet_size)
    alphabet_indices = dict((c, i) for i, c in enumerate(alphabet))

    if which_information == "char_time" or which_information == "char_time_space":
        # Rounds (up) to nearest thousand
        max_sentence_length = round(
            df.Preprocessed_typed_sentence.apply(lambda x: len(x)).max(), -3
        )
    if which_information == "char":
        # Rounds (up) to nearest hundred
        max_sentence_length = round(
            df.Preprocessed_typed_sentence.apply(lambda x: len(x)).max(), -2
        )

    # Make training data array
    all_sentences = [item for sublist in subject_documents for item in sublist]

    # Initialise tokenizer which maps characters to integers
    tk = Tokenizer(num_words=None, char_level=True)

    # Fit to text: convert all chars to ints
    tk.fit_on_texts(all_sentences)

    # Update alphabet
    tk.word_index = alphabet_indices

    if feat_type == "doc":
        raise NotImplementedError
        # If we are using document features
        X = []
        for doc in subject_documents:
            # Create integer representations of subject's written sentences
            tmp_int_docs = tk.texts_to_sequences(doc)
            # Pad sequences so that they all have the same length and then one-hot encode
            X.append(
                to_categorical(
                    pad_sequences(
                        tmp_int_docs, maxlen=max_sentence_length, padding="post"
                    )
                )
            )

        if which_information == "char_time_space":
            raise NotImplementedError

    # Get integer sequences: converts sequences of chars to sequences of ints
    int_sequences = tk.texts_to_sequences(all_sentences)

    # Pad sequences so that they all have the same length and then one-hot encode
    X = to_categorical(
        pad_sequences(int_sequences, maxlen=max_sentence_length, padding="post")
    )

    if which_information == "char_time_space":
        # Load relevant keyboard
        keyboard = us_standard_layout_keyboard()  # OBS: nested list
        # Check that all chars are in fact in our "keyboard" -- if not, we cannot map a coordinate
        assert alphabet.issubset(set(list(itertools.chain.from_iterable(keyboard))))
        space = [
            english_keys_to_2d_coordinates(sentence, keyboard)
            for sentence in all_sentences
        ]
        space_padded = [
            pad(s, [(0, max_sentence_length - len(s)), (0, 0)], mode="constant")
            for s in space
        ]
        # Append coordinates to one-hot encoded sentences
        X = einsum(
            "ijk->kij", dstack([hstack((x, s)) for (x, s) in zip(X, space_padded)])
        )

    # Get labels (diagnoses)
    y = df.Diagnosis.tolist()

    # Chop up data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=True
    )
    return X_train, X_test, y_train, y_test, max_sentence_length, alphabet_size


def us_standard_layout_keyboard():
    """
    Keyboard layout used for the MJFF data.

    For details see: https://www.nature.com/articles/s41598-019-39294-z/figures/5

    Parameters
    ----------
    typed_sentence : [type]
        [description]
    keyboard : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # Lower caps
    kb_row_0 = ["`", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "=", ""]
    kb_row_1 = ["", "q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "[", "]", ""]
    kb_row_2 = ["", "a", "s", "d", "f", "g", "h", "j", "k", "l", ";", "'", "", ""]
    kb_row_3 = ["\\", "z", "x", "c", "v", "b", "n", "m", ",", ".", "/", "", "", ""]
    kb_row_4 = [ "", "", "", " ", " ", " ", " ", " ", " ", "", "", "", "", "", ]  # Space bar
    # Upper caps
    kb_row_0_u = ["~", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "_", "+", ""]
    kb_row_1_u = ["", "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "{", "}", "|"]
    kb_row_2_u = ["", "A", "S", "D", "F", "G", "H", "J", "K", "L", ":", "", "", ""]
    kb_row_3_u = ["", "Z", "X", "C", "V", "B", "N", "M", "<", ">", "?", "", "", ""]
    keyboard = [
        kb_row_0,
        kb_row_1,
        kb_row_2,
        kb_row_3,
        kb_row_4,
        kb_row_0_u,
        kb_row_1_u,
        kb_row_2_u,
        kb_row_3_u,
    ]

    return keyboard


def english_keys_to_2d_coordinates(typed_sentence, keyboard):
    """
    Function returns the 2D coordinates of the characters in the sentence.

    Parameters
    ----------
    typed_sentence : [type]
        [description]
    keyboard : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    # Store individual key coordinates here
    coordinates = []
    for c in typed_sentence:
        for i, r in enumerate(keyboard):
            try:
                # Check if character is in given key-map
                coordinates.append((i, r.index(c)))
            except:
                # If not, store the UNK coordinate (center of keyboard)
                pass
    return array(coordinates)

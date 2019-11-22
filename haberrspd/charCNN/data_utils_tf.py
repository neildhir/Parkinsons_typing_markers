import itertools

import keras.backend as K
from keras import callbacks
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from numpy import argwhere, array, dstack, einsum, hstack, int64, ones, pad, matrix
from pandas import read_csv
from sklearn.model_selection import train_test_split
from tensorflow import cast, float32, one_hot


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
    return cast(one_hot(x, K.alphabet_size, on_value=1, off_value=0, axis=-1), float32)  # TODO: check precision


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
    subject_documents = []  # Contains on the index all sentences typed by a particular subject
    subject_diagnoses = []  # Contains on the index, the PD diagnosis of a particular subject

    for i in df.Patient_ID.drop_duplicates():
        # Ensure that all sentences are lower-case (this improves inference further down the pipe)
        subject_documents.append(df.loc[(df.Patient_ID == i)].Preprocessed_typed_sentence.str.lower().tolist())

        # XXX: This returns one diagnosis per patient, but we may want one diagnosis per sentence
        subject_diagnoses.append(df.loc[(df.Patient_ID == i)].Diagnosis.drop_duplicates().tolist()[0])

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
    max_sentence_length = round(df.Preprocessed_typed_sentence.apply(lambda x: len(x)).max(), -3)

    # Populate the training array
    if which_level == "document":
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

        print("Sample X (encoded sentence): {}".format(X[13, 2]))
        print("Target y (1: PD; 0: control): {}".format(y[13]))

        # Chop up data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        return (X_train, X_test, y_train, y_test, max_sentences_per_subject, max_sentence_length)

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
        return X_train, X_test, y_train, y_test, max_sentence_length

    else:
        raise ValueError


def create_training_data_keras(DATA_ROOT, which_information, data_file):
    """
    This function creats one-hot encoded character -data for the document (=subject)
    classification model, as well as the sentence classification model. The functionality
    within is keras specific.

    Parameters
    ----------
    DATA_ROOT : str
        Location of the [MJFF/MRC] data folder, this implicitly sets which dataset is under investigation
    which_information : str
        If we are looking at [char, char_time, char_time_space]
    data_string : str
        The .csv file that we want to analyse

    Returns
    -------
    tuple
        Contains the training and test data as well as some parameters
    """
    assert type(data_file) is str

    if which_information == "char_time_space":
        # Get relevant long-format data
        df = read_csv(DATA_ROOT / "char_time" / data_file, header=0)  # MJFF data
    else:
        df = read_csv(DATA_ROOT / which_information / data_file, header=0)  # MJFF data

    subject_documents, subjects_diagnoses, alphabet = create_mjff_data_objects(df)

    # Store alphabet size
    alphabet_size = len(alphabet)

    print("Total number of characters:", alphabet_size)
    alphabet_indices = dict((c, i) for i, c in enumerate(alphabet))

    if which_information == "char_time" or which_information == "char_time_space":
        # Rounds (up) to nearest thousand, finding the maximum sentence length over all sentences
        max_sentence_length = round(df.Preprocessed_typed_sentence.apply(lambda x: len(x)).max(), -3)
    if which_information == "char":
        # Rounds (up) to nearest hundred
        max_sentence_length = round(df.Preprocessed_typed_sentence.apply(lambda x: len(x)).max(), -2)

    # Make training data array
    all_sentences = [item for sublist in subject_documents for item in sublist]

    # Initialise tokenizer which maps characters to integers
    tk = Tokenizer(num_words=None, char_level=True)

    # Fit to text: convert all chars to ints
    tk.fit_on_texts(all_sentences)

    # Update alphabet
    tk.word_index = alphabet_indices

    # Classification by "document" i.e. all sentences, per candidate, are collated into a bag called a document
    if feat_type == "doc":
        raise NotImplementedError
        # If we are using document features
        X = []
        for doc in subject_documents:
            # Create integer representations of subject's written sentences
            tmp_int_docs = tk.texts_to_sequences(doc)
            # Pad sequences so that they all have the same length and then one-hot encode
            X.append(to_categorical(pad_sequences(tmp_int_docs, maxlen=max_sentence_length, padding="post")))

        if which_information == "char_time_space":
            raise NotImplementedError

    # Get integer sequences: converts sequences of chars to sequences of ints
    int_sequences = tk.texts_to_sequences(all_sentences)

    # Pad sequences so that they all have the same length and then one-hot encode
    X = to_categorical(pad_sequences(int_sequences, maxlen=max_sentence_length, padding="post"))

    if which_information == "char_time_space":
        # Load relevant keyboard
        # TODO: fix this to reflect the difference in keyboard for MJFF and MRC
        keyboard = us_standard_layout_keyboard()  # OBS: nested list
        # Check that all chars are in fact in our "keyboard" -- if not, we cannot map a coordinate
        assert alphabet.issubset(set(list(itertools.chain.from_iterable(keyboard))))
        # TODO: fix this to reflect the difference in keyboard for MJFF and MRC
        space = [english_keys_to_2d_coordinates_mjff(sentence, keyboard) for sentence in all_sentences]
        space_padded = [pad(s, [(0, max_sentence_length - len(s)), (0, 0)], mode="constant") for s in space]
        # Append coordinates to one-hot encoded sentences
        X = einsum("ijk->kij", dstack([hstack((x, s)) for (x, s) in zip(X, space_padded)]))

    # Get labels (diagnoses)
    y = df.Diagnosis.tolist()

    # Chop up data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)

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

    # TODO: This needs to be changed to reflect the same keyboard as the MRC one (See below)
    raise ValueError

    # Lower caps
    kb_row_0 = ["`", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "=", ""]
    kb_row_1 = ["", "q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "[", "]", ""]
    kb_row_2 = ["", "a", "s", "d", "f", "g", "h", "j", "k", "l", ";", "'", "", ""]
    kb_row_3 = ["\\", "z", "x", "c", "v", "b", "n", "m", ",", ".", "/", "", "", ""]
    kb_row_4 = ["", "", "", " ", " ", " ", " ", " ", " ", "", "", "", "", ""]  # Space bar
    # Upper caps
    kb_row_0_u = ["~", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "_", "+", ""]
    kb_row_1_u = ["", "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "{", "}", "|"]
    kb_row_2_u = ["", "A", "S", "D", "F", "G", "H", "J", "K", "L", ":", "", "", ""]
    kb_row_3_u = ["", "Z", "X", "C", "V", "B", "N", "M", "<", ">", "?", "", "", ""]
    keyboard = [kb_row_0, kb_row_1, kb_row_2, kb_row_3, kb_row_4, kb_row_0_u, kb_row_1_u, kb_row_2_u, kb_row_3_u]

    return keyboard


def english_keys_to_2d_coordinates_mjff(typed_sentence, keyboard):
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
                pass

    return array(coordinates)


def english_keys_to_2d_coordinates_mrc(typed_sentence, typed_key_locations, english_lower, english_upper) -> array:
    """
    Function returns the 2D coordinates of the characters in the sentence.

    OBS: UNK characters are mapped to the centre of the keyboard, as they are confounding.

    Parameters
    ----------
    typed_sentence : list
        list of characters containing the typed sentence by the user
    typed_key_location : list
        list of key locations (e.g. Shift appears twice on the keyboard)
    english_lower : array

    english_upper : array


    Returns
    -------
    array-like
        Returns an area of the coordinates used to produce the passed sentence
    """

    assert "Backspace" not in english_lower or english_upper
    assert len(typed_sentence) > 1
    assert len(typed_key_locations) > 1

    modifier_keys = ["Shift", "Control", "Meta", "Alt"]
    cords = []
    for char, char_loc in zip(typed_sentence, typed_key_locations):
        if char in english_lower:
            # Lower caps

            # Special characters which appear twice on the keyboard
            if char in modifier_keys:
                cords.append(
                    tuple(
                        [argwhere(english_lower == char)[0] if char_loc <= 1 else argwhere(english_lower == char)[1]][0]
                    )
                )
            else:
                # Normal characters
                cords.append(tuple(argwhere(english_lower == char)[0]))

        elif char in english_upper:
            # Upper caps characters
            cords.append(tuple(argwhere(english_upper == char)[0]))

        else:
            # all chars not in the keyboard are mapped to the UNK character
            cords.append((3, 7))

    assert None not in cords

    return array(cords)


def us_english_keyboard_mrc():
    """
    QWERTY keyboard used for the MRC data collection.

    Note that this keyboard contains modifier keys (such as 'Shift').
    """

    # Lower case
    english_lower = matrix(
        [
            [u"`", u"1", u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9", u"0", u"-", u"=", None],
            [u"Tab", u"q", u"w", u"e", u"r", u"t", u"y", u"u", u"i", u"o", u"p", "[", "]", "\\"],
            [u"CapsLock", u"a", u"s", u"d", u"f", u"g", u"h", u"j", u"k", u"l", u";", u"'", None, None],
            [u"Shift", u"z", u"x", u"c", u"v", u"b", u"n", u"m", u",", u".", u"/", u"Shift", None, None],
            [u"Control", u"Meta", u"Alt", u" ", u" ", u" ", u" ", u" ", u"Alt", u"Meta", None, u"Control", None, None],
        ],
        dtype="U",
    )
    # Upper case
    english_upper = matrix(
        [
            [u"~", u"!", u"@", u"#", u"$", u"%", u"^", u"&", u"*", u"(", u")", u"_", u"+", None],
            [None, u"Q", u"W", u"E", u"R", u"T", u"Y", u"U", u"I", u"O", u"P", u"{", u"}", u"|"],
            [None, u"A", u"S", u"D", u"F", u"G", u"H", u"J", u"K", u"L", u":", u'"', None, None],
            [None, u"Z", u"X", u"C", u"V", u"B", u"N", u"M", u"<", u">", u"?", None, None, None],
            [None, None, None, " ", " ", " ", " ", " ", None, None, None, None, None, None],
        ],
        dtype="U",
    )

    return english_lower, english_upper

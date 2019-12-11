import itertools
from typing import Tuple

import keras.backend as K
from keras import callbacks
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from numpy import argwhere, array, array_equal, concatenate, dstack, einsum, empty, hstack, int64, matrix, ones, pad
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow import cast, float32, one_hot
from collections import defaultdict
from math import ceil

from src.preprocess import modifier_key_replacements


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


def create_data_objects(df):
    """
    Note that the interpretation here is that each document is comensurate with a subject
    in the dataset.
    """
    subject_documents = []  # Contains on the index all sentences typed by a particular subject
    subject_locations = []  # Contains on the index all key locations used by a subject [MRC only]
    subject_diagnoses = defaultdict(dict)  # Contains on the index, the PD diagnosis of a particular subject

    for i in df.Patient_ID.unique():
        # Ensure that all sentences are lower-case (this improves inference further down the pipe)
        subject_documents.append(df.loc[(df.Patient_ID == i)].Preprocessed_typed_sentence.tolist())

        # This returns one diagnosis per patient
        # subject_diagnoses.append(df.loc[(df.Patient_ID == i)].Diagnosis.drop_duplicates().tolist()[0])
        # This returns one diagnosis per patient
        subject_diagnoses[i] = df[(df.Patient_ID == i)].Diagnosis.tolist()

        if "Preprocessed_locations" in df.columns:
            # We are dealing with MRC data if this is true
            subject_locations.append(df.loc[(df.Patient_ID == i)].Preprocessed_locations.tolist())

    # Get the unique set of characters in the alphabet
    alphabet = set("".join([item for sublist in subject_documents for item in sublist]))

    return subject_documents, subject_locations, subject_diagnoses, alphabet


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

    df = read_csv(DATA_ROOT / data_string, header=0)
    subject_documents, subject_locations, subjects_diagnoses, alphabet = create_data_objects(df)

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


def roundup(x):
    return int(ceil(x / 100.0)) * 100


def create_training_data_keras(
    DATA_ROOT,
    which_information,
    csv_file,
    feat_type=None,
    indicator_character="ω",
    mrc_unk_symbol="£",
    for_plotting_results=False,
):
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
    assert type(csv_file) is str

    if which_information == "char_time_space":
        # Get relevant long-format data
        df = read_csv(DATA_ROOT / "char_time" / csv_file, header=0)
    else:
        df = read_csv(DATA_ROOT / which_information / csv_file, header=0)

    # Get relevant data objects
    subject_documents, subject_locations, subjects_diagnoses, alphabet = create_data_objects(df)
    # Make training data array
    all_sentences = [item for sublist in subject_documents for item in sublist]
    all_locations = [item for sublist in subject_locations for item in sublist]
    # Get max sentence length
    max_sentence_length = roundup(max([len(s) for s in all_sentences]))
    # Store alphabet size
    alphabet_size = len(alphabet)
    print("Total number of characters used in all typed sentences:", alphabet_size)
    alphabet_indices = dict((c, i) for i, c in enumerate(alphabet))

    # Initialise tokenizer which maps characters to integers
    tk = Tokenizer(num_words=None, char_level=True)
    # Fit to text: convert all chars to ints
    tk.fit_on_texts(all_sentences)
    # Update alphabet
    tk.word_index = alphabet_indices

    # Classification by "document" i.e. all sentences, per candidate, are collated into a bag called a document
    if feat_type:
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
        if "MJFF" in str(DATA_ROOT):
            if "spanish" in csv_file.lower():
                layout = "spanish"
            else:
                layout = "uk"
            keyboard_lower, keyboard_upper = english_language_qwerty_keyboard(layout=layout)
            # Check that all chars are in fact in our "keyboard" -- if not, we cannot map a coordinate
            if layout == "uk":
                # Only relevant for UK keyboard
                character_set = set(itertools.chain.from_iterable(concatenate([keyboard_lower, keyboard_upper]))).union(
                    set([indicator_character])
                )
                assert alphabet.issubset(character_set), (alphabet - character_set, alphabet, character_set)
            # Set the coordinate space for all typed sentences
            space = [
                uk_and_spanish_keyboard_keys_to_2d_coordinates_mjff(sentence, keyboard_lower, keyboard_upper)
                for sentence in all_sentences
            ]

        elif "MRC" in str(DATA_ROOT):
            keyboard_lower, keyboard_upper = english_language_qwerty_keyboard(layout="us")
            # Check that all chars are in fact in our "keyboard" -- if not, we cannot map a coordinate
            character_set = set(itertools.chain.from_iterable(concatenate([keyboard_lower, keyboard_upper]))).union(
                set([indicator_character, mrc_unk_symbol])
            )
            assert alphabet.issubset(character_set), (alphabet - character_set, alphabet, character_set)
            # Set the coordinate space for all typed sentences
            space = [
                us_keyboard_keys_to_2d_coordinates_mrc(sentence, locations, keyboard_lower, keyboard_upper)
                for sentence, locations in zip(all_sentences, all_locations)
            ]

        else:
            raise ValueError

        space_padded = [pad(s, [(0, max_sentence_length - len(s)), (0, 0)], mode="constant") for s in space]
        # Append coordinates to one-hot encoded sentences
        X = einsum("ijk->kij", dstack([hstack((x, s)) for (x, s) in zip(X, space_padded)]))

    # Get labels (diagnoses)
    y = list(itertools.chain.from_iterable(subjects_diagnoses.values()))  # df.Diagnosis.tolist()

    if for_plotting_results is False:
        # [stratified] chop-up into train and test sets
        # X_train, X_test, y_train, y_test = train_test_split(X, array(y), test_size=0.1, stratify=y)
        # return X_train, X_test, y_train, y_test, max_sentence_length, alphabet_size

        # train/val/test == 80/10/10
        X_train, X_test, y_train, y_test = train_test_split(X, array(y), test_size=0.1, random_state=1)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=(1 / 9), random_state=1)
        return X_train, X_test, X_val, y_train, y_test, y_val, max_sentence_length, alphabet_size
    else:
        # When we plot results we just want the processed full dataset.
        return X, array(y)


def uk_and_spanish_keyboard_keys_to_2d_coordinates_mjff(typed_sentence, lower_keyboard, upper_keyboard) -> array:
    """
    Function returns the 2D coordinates of the characters in the sentence. The MJFF does not
    have any locator keys so this function is different from the parallel MRC function.

    Parameters
    ----------
    typed_sentence : list
        list of characters containing the typed sentence by the user
    english_lower : array

    english_upper : array


    Returns
    -------
    array-like
        Returns an area of the coordinates used to produce the passed sentence
    """
    assert isinstance(typed_sentence, str)
    assert ~array_equal(lower_keyboard, upper_keyboard)
    assert len(typed_sentence) > 1
    coordinates = []
    for char in typed_sentence:
        if char in lower_keyboard:
            # Lower caps characters
            coordinates.append(tuple(argwhere(lower_keyboard == char)[0]))
        elif char in upper_keyboard:
            # Upper caps characters (also includes special keys such as '#')
            coordinates.append(tuple(argwhere(upper_keyboard == char)[0]))
        else:
            # Error indicator character is mapped to the UNK coordinate ("middle" of keyboard)
            coordinates.append((3, 7))

    assert None not in coordinates

    return array(coordinates)


def us_keyboard_keys_to_2d_coordinates_mrc(
    typed_sentence, typed_key_locations, lower_keyboard, upper_keyboard
) -> array:
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

    modifier_keys = modifier_key_replacements()  # A dictionary
    assert ~array_equal(lower_keyboard, upper_keyboard)
    assert len(typed_sentence) > 1
    assert len(typed_key_locations) > 1
    assert len(typed_sentence) == len(typed_key_locations)

    all_coordinates = []
    typed_key_locations = list(map(int, typed_key_locations))
    for char, char_location in zip(typed_sentence, typed_key_locations):
        # Lower caps
        if char in lower_keyboard:
            # Special characters which appear twice on the keyboard
            if char in modifier_keys.values():
                all_coordinates.append(
                    tuple(
                        [
                            argwhere(lower_keyboard == char)[0]
                            if char_location <= 1
                            else argwhere(lower_keyboard == char)[1]
                        ][0]
                    )
                )
            else:
                # Normal characters
                all_coordinates.append(tuple(argwhere(lower_keyboard == char)[0]))
        elif char in upper_keyboard:
            # Upper caps characters (also includes special keys such as '#')
            all_coordinates.append(tuple(argwhere(upper_keyboard == char)[0]))
        else:
            # all chars not in the keyboard are mapped to the UNK character
            # this includes the error-indicator ω (omega) as well as UNK symbol £
            all_coordinates.append((3, 7))

    assert None not in all_coordinates

    return array(all_coordinates)


def english_language_qwerty_keyboard(
    layout="us", use_replacement_modifier_symbols=True, indicator_character="ω"
) -> Tuple[array, array]:
    """
    [lower-case] QWERTY keyboard used for the MRC data collection.

    For details see: https://www.nature.com/articles/s41598-019-39294-z/figures/5

    Note that this US keyboard contains modifier keys (such as 'Shift'), the UK does not.
    """

    if layout == "uk":

        # UK and US keyboard layouts are slightly different.

        # Lower case
        lower = array(
            [
                [u"`", u"1", u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9", u"0", u"-", u"=", indicator_character],
                [None, u"q", u"w", u"e", u"r", u"t", u"y", u"u", u"i", u"o", u"p", "[", "]", None],
                [None, u"a", u"s", u"d", u"f", u"g", u"h", u"j", u"k", u"l", u";", u"'", "#", None],
                ["\\", u"z", u"x", u"c", u"v", u"b", u"n", u"m", u",", u".", u"/", None, None, None],
                [None, None, None, u" ", u" ", u" ", u" ", u" ", None, None, None, None, None, None],
            ],
            dtype="U",
        )
        # Upper case
        upper = array(
            [
                [u"¬", u"!", u'"', u"£", u"$", u"%", u"^", u"&", u"*", u"(", u")", u"_", u"+", None],
                [None, u"Q", u"W", u"E", u"R", u"T", u"Y", u"U", u"I", u"O", u"P", u"{", u"}", None],
                [None, u"A", u"S", u"D", u"F", u"G", u"H", u"J", u"K", u"L", u":", u"@", "~", None],
                ["|", u"Z", u"X", u"C", u"V", u"B", u"N", u"M", u"<", u">", u"?", None, None, None],
                [None, None, None, u" ", u" ", u" ", u" ", u" ", None, None, None, None, None, None],
            ],
            dtype="U",
        )

    elif layout == "spanish":

        # Note this keyboard has been slightly modified to accommodate the indicator_character which signifies and erroful typing.

        # Lower case
        lower = array(
            [
                [u"º", u"1", u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9", u"0", u"'", "¡", indicator_character],
                [None, u"q", u"w", u"e", u"r", u"t", u"y", u"u", u"i", u"o", u"p", u"`", u"+", None],
                [None, u"a", u"s", u"d", u"f", u"g", u"h", u"j", u"k", u"l", u"ñ", u"‘", u"ç", None],
                [u"<", u"z", u"x", u"c", u"v", u"b", u"n", u"m", u",", u".", u"-", None, None, None],
                [None, None, None, u" ", u" ", u" ", u" ", u" ", None, None, None, None, None, None],
            ],
            dtype="U",
        )
        upper = array(
            [
                [u"ª", u"!", u'"', u"·", u"$", u"%", u"&", u"/", u"(", u")", u"=", u"?", u"¿", None],
                [None, u"Q", u"W", u"E", u"R", u"T", u"Y", u"U", u"I", u"O", u"P", u"^", u"*", None],
                [None, u"A", u"S", u"D", u"F", u"G", u"H", u"J", u"K", u"L", u"Ñ", u"¨", u"Ç", None],
                [u">", u"Z", u"X", u"C", u"V", u"B", u"N", u"M", u";", u":", u"_", None, None, None],
                [None, None, None, u" ", u" ", u" ", u" ", u" ", None, None, None, None, None, None],
            ],
            dtype="U",
        )
    elif layout == "us":

        # Lower case
        lower = array(
            [
                [u"`", u"1", u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9", u"0", u"-", u"=", indicator_character],
                [u"tab", u"q", u"w", u"e", u"r", u"t", u"y", u"u", u"i", u"o", u"p", "[", "]", "\\"],
                [u"capslock", u"a", u"s", u"d", u"f", u"g", u"h", u"j", u"k", u"l", u";", u"'", None, None],
                [u"shift", u"z", u"x", u"c", u"v", u"b", u"n", u"m", u",", u".", u"/", u"shift", None, None],
                [
                    u"control",
                    u"meta",
                    u"alt",
                    u" ",
                    u" ",
                    u" ",
                    u" ",
                    u" ",
                    u"alt",
                    u"meta",
                    None,
                    u"control",
                    None,
                    None,
                ],
            ],
            dtype="U",
        )
        # Upper case
        upper = array(
            [
                [u"~", u"!", u"@", u"#", u"$", u"%", u"^", u"&", u"*", u"(", u")", u"_", u"+", None],
                [None, u"Q", u"W", u"E", u"R", u"T", u"Y", u"U", u"I", u"O", u"P", u"{", u"}", u"|"],
                [None, u"A", u"S", u"D", u"F", u"G", u"H", u"J", u"K", u"L", u":", u'"', None, None],
                [None, u"Z", u"X", u"C", u"V", u"B", u"N", u"M", u"<", u">", u"?", None, None, None],
                [None, None, None, u" ", u" ", u" ", u" ", u" ", None, None, None, None, None, None],
            ],
            dtype="U",
        )

        # We only use modifier symbols in the MRC dataset
        if use_replacement_modifier_symbols:
            # Get global dict
            mod_dict = modifier_key_replacements()
            for i in range(lower.shape[0]):
                if any([key in lower[i, :] for key in mod_dict.keys()]):
                    for j, item in enumerate(lower[i, :]):
                        if item in mod_dict.keys():
                            lower[i, j] = mod_dict[item]
    else:
        raise ValueError

    assert lower.shape == upper.shape, "Lower shape: {}; upper shape: {}".format(lower.shape, upper.shape)

    return lower, upper


def getval_array(d):
    v = array(list(d.values()))
    k = array(list(d.keys()))
    maxv = k.max()
    minv = k.min()
    n = maxv - minv + 1
    val = empty(n, dtype=v.dtype)
    val[k] = v
    return val


def bmatrix_from_str(list_of_chars, list_of_ikis=None):
    """
    Returns a LaTeX bmatrix

    Usage:
        print(bmatrix_from_str(['c','a','t','t'],[2,2,2,2]))
    """
    if list_of_ikis is None:
        list_of_ikis = [1] * len(list_of_chars)

    cat = OneHotEncoder()
    C = "".join([x * y for x, y in zip(list_of_chars, list_of_ikis)])
    X = array(list(C), dtype=object).T
    out = cat.fit_transform(X.reshape(-1, 1)).astype(int).toarray().T
    if len(out.shape) > 2:
        raise ValueError("bmatrix can at most display two dimensions")
    lines = str(out).replace("[", "").replace("]", "").splitlines()
    rv = [r"\begin{bmatrix}"]
    rv += ["  " + " & ".join(l.split()) + r"\\" for l in lines]
    rv += [r"\end{bmatrix}"]
    return "\n".join(rv)

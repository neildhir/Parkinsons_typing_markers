import copy
import re
import socket
import warnings
from collections import Counter, defaultdict
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Tuple
from nltk.metrics import edit_distance  # Levenshtein
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .__init_paths import data_root


def create_char_compression_time_mjff_data(df: pd.DataFrame,
                                           char_count_response_threshold=40) -> Tuple[dict, list]:
    assert 'sentence_text' in df.columns
    assert 'participant_id' in df.columns
    # Here we filter out responses where the number of characters per typed
    # response, is below a threshold value (40 by default)
    assert "response_id" in df.columns
    df = df.groupby('response_id').filter(lambda x: x['response_id'].count() > char_count_response_threshold)

    # Get the unique number of subjects
    subjects = sorted(set(df.participant_id))  # NOTE: set() is weakly random
    # sent_ids = sorted(set(df.sentence_id))

    # All sentences will be stored here, indexed by their type
    char_compression_sentences = defaultdict(dict)

    # Loop over subjects
    for subj_idx in subjects:
        # Not all subjects have typed all sentences hence we have to do it this way
        for sent_idx in df.loc[(df.participant_id == subj_idx)].sentence_id.unique():

            print("subject: {} -- sentence: {}".format(subj_idx, sent_idx))

            # Locate df segment to extract
            coordinates = (df.participant_id == subj_idx) & (df.sentence_id == sent_idx)

            # "correct" the sentence by operating on user backspaces
            corrected_sentence, removed_chars_indx = backspace_corrector(df.loc[coordinates, "key"].tolist())

            # Update the compression times for each user given the above operation
            tmp_timestamps = df.loc[coordinates, "timestamp"].reset_index(drop=True)
            assert set(removed_chars_indx).issubset(range(len(tmp_timestamps))
                                                    ), "Indices to remove: {} -- total length of timestamp vector: {}".format(removed_chars_indx, len(tmp_timestamps))
            timestamps = tmp_timestamps.drop(index=removed_chars_indx)

            # Make long-format version of each typed, corrected, sentence
            char_compression_sentences[subj_idx][sent_idx] = \
                make_character_compression_time_sentence(timestamps,
                                                         corrected_sentence)

    # No one likes an empty list so we remove them here
    for subj_idx in subjects:
        # Not all subjects have typed all sentences hence we have to do it this way
        for sent_idx in df.loc[(df.participant_id == subj_idx)].sentence_id.unique():
            # Combines sentences to contiguous sequences (if not empty)
            # if not char_compression_sentences[subj_idx][sent_idx]:
            char_compression_sentences[subj_idx][sent_idx] = ''.join(char_compression_sentences[subj_idx][sent_idx])

    return char_compression_sentences


def make_character_compression_time_sentence(compression_times: pd.Series,
                                             characters: pd.Series,
                                             time_redux_fact=10) -> str:
    """
    Function creates a long-format sentence, where each character is repeated for a discrete
    number of steps, commensurate with how long that character was compressed for, when
    the sentence was being typed by the participants.

    By design, we obviously cannot repeat the last character for a number of steps.

    Parameters
    ----------
    compression_times : pd.Series
        Compression times in milliseconds
    characters : pd.Series
        Individual characters in the typed sentence.
    indices_removed_backspakces: list
        List of integer locations at which a backstop was removed from the original sentence
    time_redux_fact : int, optional
        Time reduction factor, to go from milliseconds to something else, by default 10
        A millisecond is 1/1000 of a second. Convert this to centisecond (1/100s).

    Returns
    -------
    list
        Returns a list in which each character has been repeated a number of times.
    """

    # Function to flatten a list of lists
    def flatten(l): return [item for sublist in l for item in sublist]

    assert len(compression_times) == len(characters), "Lengths are: {} and {}".format(
        len(compression_times), len(characters))
    char_times = compression_times.diff().values.astype(int) // time_redux_fact
    return flatten([[c]*n for c, n in zip(characters[:-1], char_times[1:])])


def measure_levensthein_for_lang8_data(data_address: str,
                                       ld_threshold: int = 2) -> pd.DataFrame:
    """
    Measures the Levensthein (edit distance) between typed sentences and their
    (human) corrected counter-parts. We only take the first correction --
    some sentences have multiple human-corrected examples.

    Parameters
    ----------
    data_address : str
        Location of the lang8 data file.
    ld_threshold : int, optional
        The threshold cut-off for the Levensthein distance.

    Returns
    -------
    pd.DataFrame
        Return the a dataframe with header:
        ['typed sentence', 'corrected sentence', 'Levensthein distance']
    """
    # Load
    df = pd.read_csv(data_address,
                     sep='\t',
                     names=["A", "B", "C", "D", "written", "corrected"])
    print("Pre-drop entries count: %s" % df.shape[0])

    # Filter out rows which do not have a correction (i.e. A  > 0) and get only raw data
    df = df.loc[df['A'] > 0].filter(items=["written", "corrected"])

    print("Post-drop entries count: %s" % df.shape[0])

    # Calculate the Levenshtein distance
    df["distance"] = df.loc[:, ["written", "corrected"]].apply(lambda x: edit_distance(*x), axis=1)

    # Only return sentence pairs of a certain LD
    return df.loc[df.distance.isin([1, ld_threshold])]


def create_mjff_training_data(df: pd.DataFrame) -> Tuple[dict,
                                                         list]:
    """
    Function takes a dataframe which contains training data from the English&Spanish typing dataset,
    extracts the typed sentences and stores them in a dictionary, indexed by the true sentence.

    Parameters
    ----------
    df : pandas dataframe
        Pandas dataframe which contains the typed sentence data

    Returns
    -------
    dict
        Dictionary indexed by the true sentence, with the typed sentences on the values
    """

    assert 'sentence_text' in df.columns
    assert 'participant_id' in df.columns

    # Get the unique number of subjects
    subjects = sorted(set(df.participant_id))  # NOTE: set() is weakly random
    sent_ids = sorted(set(df.sentence_id))
    # All typed sentences will be stored here, indexed by their type
    typed_keys = defaultdict(dict)
    # A deconstructed dataframe by sentence ID and text only
    df_sent_id = df.groupby(['sentence_id', 'sentence_text']).size().reset_index()

    for sub_id in subjects:
        for sent_id in sent_ids:
            typed_keys[sub_id][sent_id] = df.loc[(df.participant_id == sub_id) &
                                                 (df.sentence_id == sent_id), "key"].tolist()

    # No one likes an empty list so we remove them here
    for sub_id in subjects:
        for sent_id in sent_ids:
            typed_keys[sub_id][sent_id] = [x for x in typed_keys[sub_id][sent_id] if x != []]

    return typed_keys, sent_ids, df_sent_id


def create_mjff_iki_training_data(df: pd.DataFrame) -> dict:
    """
    Function takes a dataframe which contains training data from the English&Spanish typing dataset,
    extracts the interkey-interval (IKI) for each typed character. No error correctionis made.

    Parameters
    ----------
    df : pandas dataframe
        Pandas dataframe which contains the typed IKI sentence data

    Returns
    -------
    dict
        Dictionary indexed by the true sentence, with the typed sentences on the values
    """

    assert 'sentence_text' in df.columns
    assert 'timestamp' in df.columns
    assert 'participant_id' in df.columns

    # Convert target sentences to integer IDs instead, easier to work with
    sentences = list(set(df.sentence_text))

    # Get the unique number of subjects
    subjects = set(df.participant_id)

    # The IKI time-series, per sentence, per subject is stored in this dict
    typed_sentence_IKI_ts_by_subject = defaultdict(list)

    # Loop over all the participants
    for subject in subjects:

        # Get all the necessary typed info for particular subject
        info_per_subject = df.loc[df['participant_id'] == subject][['key', 'timestamp', 'sentence_text']]

        # Get all sentences of a particular type
        for sentence in sentences:

            # Append the IKI to the reference sentence store, at that sentence ID
            ts = info_per_subject.loc[info_per_subject['sentence_text'] == sentence].timestamp.values
            # Append to subject specifics
            typed_sentence_IKI_ts_by_subject[subject].extend([ts])

    # TODO: want to add some more checks here to ensure that we have not missed anything

    # No one likes an empty list so we remove them here
    for subject in subjects:
        # Remove empty arrays that may have snuck in
        typed_sentence_IKI_ts_by_subject[subject] = [
            x for x in typed_sentence_IKI_ts_by_subject[subject] if x.size != 0]

    # Re-base each array so that it starts at zero.
    for subject in subjects:
        # Remove empty arrays that may have snuck in
        typed_sentence_IKI_ts_by_subject[subject] = [
            x - x.min() for x in typed_sentence_IKI_ts_by_subject[subject]]

    return typed_sentence_IKI_ts_by_subject


def count_backspaces_per_subject_per_sentence(dictionary: dict) -> dict:
    """
    This function counts the number of backspaces used per sentence, per subject. This is so that this
    information can be used as a feature for the model later.

    OBS: this functions does not take into account _where_ the backspace was invoked in the sentence.
    That is left for further work.

    Parameters
    ----------
    sub_dict : dict
        This is the dictionary calculated by create_mjff_training_data()

    Returns
    -------
    tuple
        Dictionary, per subject, with counts of backspaces per typed sentence
    """

    # Store the backspace counts in a dictionary indexed by subject
    backspace_count_per_subject_per_sentence = defaultdict(list)
    # Loop over subjects and their sentences and counts the backspaces
    for subject in dictionary.keys():
        # Loop over all their typed sentences
        for sentence in dictionary[subject]:
            # Here we count the number of backspaces found in this sentence
            backspace_count_per_subject_per_sentence[subject].extend(Counter(sentence)['backspace'])

    return backspace_count_per_subject_per_sentence


def combine_characters_to_form_words_at_space(typed_keys: dict,
                                              sent_ids: list,
                                              correct: bool = True) -> dict:
    """
    This function takes the typed_keys constructed in 'create_mjff_training_data' and
    combines it at the space character. This typed_keys has had the backspaces removed.

    There are some assumptions with this construction which are dealt with inside the function.

    Parameters
    ----------
    typed_keys : dict
        Dictionary containing the empirical sentences typed by subjects

    correct : int
        Issues which type of correction we wish to make before we pass anything convert anything into
        an embedding

    Returns
    -------
    dict
        Dictionary with target sentences on keys, and constructed sentences on the values
    """

    # Store whichever result here
    completed_sentence_per_subject_per_sentence = defaultdict(dict)
    if correct == True or correct == False:
        for sub_id in typed_keys.keys():
            for sent_id in sent_ids:
                if correct:
                    # A fairly simple correction algorithm applied to invoke the subject's correction
                    completed_sentence_per_subject_per_sentence[sub_id][sent_id] = ''.join(
                        backspace_corrector(typed_keys[sub_id][sent_id]))
                elif correct is False:
                    # Here we remove those same backspaces from the sentence so that we
                    # can construct words. This is an in-place operation.
                    completed_sentence_per_subject_per_sentence[sub_id][sent_id] = typed_keys[sub_id][sent_id].remove(
                        'backspace')
    elif correct == -1:
        # We enter here if we do not want any correction to our sentences, implicitly this means that we
        # keep all the backspaces in the sentence as characters.
        for sub_id in typed_keys.keys():
            for sent_id in sent_ids:
                completed_sentence_per_subject_per_sentence[sub_id][sent_id] = ''.join(
                    typed_keys[sub_id][sent_id])

    return completed_sentence_per_subject_per_sentence


def backspace_corrector(sentence: list,
                        removal_character='backspace',
                        invokation_type=1,
                        verbose: bool = False) -> list:

    # Want to pass things as list because it easier to work with w.r.t. to strings
    assert isinstance(sentence, list)
    assert invokation_type in [-1, 0, 1]
    original_sentence = copy.copy(sentence)

    # Check that we are not passing a sentence which only contains backspaces
    if [removal_character] * len(sentence) == sentence:
        # Return an empty list which will get filtered out at the next stage
        return []

    # SPECIAL CASE: WE KEEP THE BACKSPACE AN ENCODE IT AS A CHARACTER FOR USE IN E.G. A charCNN MODEL

    if invokation_type == -1:
        # In place of 'backspace' we use a pound-sign
        return ['£' if x == 'backspace' else x for x in sentence]

    # Need to assert that this is given a sequentially ordered array
    def range_extend(x): return list(np.array(x) - len(x)) + x

    # Recursive method to remove the leading backspaces
    def remove_leading_backspaces(x):
        # Function recursively removes the leading backspace(s) if present
        if x[0] == removal_character:
            return remove_leading_backspaces(x[1:])
        else:
            return x

    # Apply to passed sentence
    pre_removal_length = len(sentence)
    sentence = remove_leading_backspaces(sentence)
    post_removal_length = len(sentence)
    nr_leading_chars_removed = (pre_removal_length - post_removal_length)

    # Generate coordinates of all items to remove
    remove_cords = []

    # Find the indices of all the reamaining backspace occurences
    backspace_indices = np.where(np.asarray(sentence) == removal_character)[0]

    # Find all singular and contiguous appearances of backspace
    backspace_groups = []
    for k, g in groupby(enumerate(backspace_indices), lambda ix: ix[0] - ix[1]):
        backspace_groups.append(list(map(itemgetter(1), g)))

    if invokation_type == 0:
        # Remove all characters indicated by 'backspace'

        for group in backspace_groups:
            # A singular backspace
            if len(group) == 1:
                remove_cords.extend([group[0]-1, group[0]])

            else:
                remove_cords.extend(range_extend(group))

    elif invokation_type == 1:
        # Remove all characters indicated by 'backspace' _except_ the last character which is kept

        for group in backspace_groups:

            # Solitary backspace removel proceedure
            if len(group) == 1:
                remove_cords.extend([group[0]])  # Remove _just_ the backspace and nothing else, don't invoke
            else:
                # XXX: this _may_ introduce negative indices at the start of a sentence
                # these are filtered out further down
                remove_cords.extend(range_extend(group[:-1]))  # This invokes the n-1 backspaces
                # This remove the nth backspace and the immediately following character
                remove_cords.extend([group[-1], group[-1]+1])

    else:
        raise ValueError

    # Filter out negative indices which are non-sensical for deletion (arises when more backspaces than characters in beginning of sentence)
    remove_cords = list(filter(lambda x: x >= 0, remove_cords))
    # Filter out deletion indices which appear at the end of the sentence as part of a contiguous group of backspaces
    remove_cords = list(filter(lambda x: x < len(original_sentence), remove_cords))
    # Do actual deletion
    invoked_sentence = np.delete(sentence, remove_cords).tolist()

    if verbose:
        print("Original sentence: {}\n".format(original_sentence))
        print("Edited sentence: {} \n -----".format(invoked_sentence))

    if nr_leading_chars_removed == 0:
        # No leading backspaces found, so we do not have to update the index list
        return invoked_sentence, remove_cords
    else:
        # Update indices to match the originals
        remove_cords = [i + nr_leading_chars_removed for i in remove_cords]
        return invoked_sentence, list(range(nr_leading_chars_removed)) + remove_cords


def create_dataframe_from_processed_data(my_dict: dict,
                                         sentence_ids: list,
                                         df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Function creates a pandas DataFrame which will be used by the NLP model
    downstream.

    Parameters
    ----------
    my_dict : dict
        Dictionary containing all the preprocessed typed sentences, indexed by subject
    sentence_id : list
        List containing the sentence_IDs per subject
    df_meta : Pandas DataFrame
        Contains the mete information on each patient

    Returns
    -------
    Pandas DataFrame
        Returns the compiled dataframe from all subjects.
    """

    final_out = []
    for patient_id in my_dict.keys():
        final_out.append(
            pd.DataFrame(
                [
                    [patient_id,
                     # This row selects the diagnosis of the patient
                     int(df_meta.loc[df_meta['index'] == patient_id, df_meta.columns[-1]]),  # ['diagnosis']),
                     sent_id,
                     my_dict[patient_id][sent_id]] for sent_id in my_dict[patient_id].keys()
                ]
            )
        )
    df = pd.concat(final_out, axis=0)
    df.columns = ['Patient_ID', 'Diagnosis', 'Sentence_ID', 'Preprocessed_typed_sentence']

    # Final check for empty values
    df['Preprocessed_typed_sentence'].replace('', np.nan, inplace=True)
    # Remove all such rows
    df.dropna(subset=['Preprocessed_typed_sentence'], inplace=True)

    return df


def create_long_form_NLP_datasets_from_MJFF_English_data(use_mechanical_turk=False):
    """
    End-to-end creation of raw data to NLP readable train and test sets.

    Returns
    -------
    pandas dataframe
        Processed dataframe
    """

    # Pre-select columns to use
    meta_cols = ['index', 'diagnosis']
    data_cols = ['timestamp', 'key', 'response_id', 'response_created', 'participant_id',
                 'sentence_id', 'sentence_text']

    # Raw data
    df = pd.read_csv(data_root / 'EnglishData.csv', usecols=data_cols)
    df_meta = pd.read_csv(data_root / "EnglishParticipantKey.csv", usecols=meta_cols)

    # If we want to use MT as well
    if use_mechanical_turk:
        df_mt = pd.read_csv(data_root / 'MechanicalTurkCombinedEnglishData.csv',
                            dtype={'participant_id': str})
        df_meta_mt = pd.read_csv(data_root / "MechanicalTurkEnglishParticipantKey.csv")
        # Change type to match other meta data
        assert all(df.columns == df_mt.columns)
        assert all(df_meta.columns == df_meta_mt.columns)
        # Combine
        df = pd.concat([df, df_mt]).reset_index(drop=True)
        df_meta = pd.concat([df_meta, df_meta_mt]).reset_index(drop=True)

    # Creates long sequences with characters repeated for IKI number of steps
    out = create_char_compression_time_mjff_data(df)

    # Extracts all the characters per typed sentence, per subject
    _, numerical_sentence_ids, reference_sentences = create_mjff_training_data(df)

    # Return the empirical data and the reference sentences for downstream tasks
    return (create_dataframe_from_processed_data(out,
                                                 numerical_sentence_ids,
                                                 df_meta).reset_index(drop=True),
            reference_sentences.loc[:, ['sentence_id', 'sentence_text']])


def create_NLP_datasets_from_MJFF_English_data(use_mechanical_turk=False):
    """
    End-to-end creation of raw data to NLP readable train and test sets.

    Parameters
    ----------
    use_mechanical_turk: bool
        To add mechanical turk data or not to the training set

    Returns
    -------
    pandas dataframe
        Processed dataframe
    """

    if socket.gethostname() == 'pax':
        # Monster machine
        data_root = '../data/MJFF/'  # My local path
        data_root = Path(data_root)
    else:
        # Laptop
        data_root = '/home/nd/data/liverpool/MJFF'  # My local path
        data_root = Path(data_root)

    # Raw data
    df = pd.read_csv(data_root / 'EnglishData.csv')
    df_meta = pd.read_csv(data_root / "EnglishParticipantKey.csv")

    if use_mechanical_turk:
        df_mt = pd.read_csv(data_root / 'MechanicalTurkCombinedEnglishData.csv')
        df_meta_mt = pd.read_csv(data_root / "MechanicalTurkEnglishParticipantKey.csv")
        # Drop columns from main data to facilitate concatenation
        df.drop(columns=['parameters_workerId', 'parameters_consent'], inplace=True)
        assert all(df.columns == df_mt.columns)
        # Combine
        df = pd.concat([df, df_mt]).reset_index(drop=True)
        df_meta = pd.concat([df_meta, df_meta_mt]).reset_index(drop=True)

    # Extracts all the characters per typed sentence, per subject
    out, numerical_sentence_ids, reference_sentences = create_mjff_training_data(df)

    # Make proper sentences from characters, where default is to invoke backspaces
    out = combine_characters_to_form_words_at_space(out, numerical_sentence_ids)

    # Return the empirical data and the reference sentences for downstream tasks
    return (create_dataframe_from_processed_data(out,
                                                 numerical_sentence_ids,
                                                 df_meta).reset_index(drop=True),
            reference_sentences.loc[:, ['sentence_id', 'sentence_text']])


def create_NLP_datasets_from_MJFF_Spanish_data() -> pd.DataFrame:
    """
    Creatae NLP-readable dataset from Spanish MJFF data.
    """

    # Monster machine
    data_root = '../data/MJFF/'  # Relative path
    data_root = Path(data_root)

    # Meta
    df_meta = pd.read_csv(data_root / "SpanishParticipantKey.csv")

    # Text
    df = pd.read_csv(data_root / 'SpanishData.csv')

    # There is no label for subject 167, so we remove her here.
    df = df.query('participant_id != 167')

    # Extracts all the characters per typed sentence, per subject
    out = create_mjff_training_data(df)

    # No one likes an empty list so we remove them here
    for subject in out.keys():
        # Remove empty lists that may have snuck in
        out[subject] = [x for x in out[subject] if x != []]

    # Make proper sentences from characters, where default is to invoke backspaces
    out = combine_characters_to_form_words_at_space(out)

    # No one likes an empty list so we remove them here
    for subject in out.keys():
        # Remove empty lists that may have snuck in
        out[subject] = [x for x in out[subject] if x != []]

    out = create_dataframe_from_processed_data(out, df_meta).reset_index(drop=True)
    # Because characters are typed sequentially special Spanish characters
    # are not invoked in the dataset. We fix this here.
    return create_proper_spanish_letters(out)


def create_proper_spanish_letters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Because characters are typed sequentially special Spanish characters
    are not invoked in the dataset. We fix this here.

    Parameters
    ----------
    df : pandas dataframe
        Containes sentences with improper spanish letters

    Returns
    -------
    pandas dataframe
        Corrected characters used in sentences
    """

    special_spanish_substrings = ["´", "~", '"']
    char_unicodes = [u'\u0301', u'\u0303', u'\u0308']
    unicode_dict = dict(zip(special_spanish_substrings, char_unicodes))

    for sent_idx in df.index:
        sentence = df.loc[sent_idx, 'Preprocessed_typed_sentence']
        # Check if there are any special characters present
        if any(substring in sentence for substring in special_spanish_substrings):
            # Special chars are found

            # Check which substring is present
            for substring in special_spanish_substrings:
                if substring in sentence:
                    # Get coordinates of all substrings like this
                    cords = [sentence.start() for sentence in re.finditer(substring, sentence)]
                    # Create the proper Spanish letters at these coordinates
                    # assumption of this operation is that the special character comes before
                    # the concatenating character.

                    # Convert str to a mutable object so we can operate upon it
                    lst_str = list(sentence)
                    # Update the 'string'
                    for i in cords:
                        lst_str[i] = lst_str[i+1] + unicode_dict[substring]
                    # Delete superfluos character and Convert back to immutable object i.e. a string
                    df.loc[sent_idx, 'Preprocessed_typed_sentence'] = ''.join(np.delete(lst_str, np.array(cords)+1))
    return df


def mjff_dataset_stats(df: pd.DataFrame):
    print("Total number of participants: %d" % (len(set(df.participant_id))))
    print("Total number of target sentences: %d" % (len(set(df.sentence_text))))
    print("Total number of sentence IDs: %d" % (len(set(df.sentence_id))))
    print("Total number of keys used: %d" % (len(set(df.key))))

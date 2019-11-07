import copy
import re
import socket
import warnings
from collections import Counter, defaultdict
from itertools import count, groupby
from operator import itemgetter
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from nltk import edit_distance as ed
from nltk.metrics import edit_distance  # Levenshtein
from scipy.stats import gamma, gengamma, lognorm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Support vector classifier
from sklearn.metrics import roc_curve, auc

from .__init_paths import data_root

# ------------------------------------------ MRC------------------------------------------ #


class processMRC:
    """
    Governing class with which the user will interface.
    All the heavy lifting happens under the hood.

    1. Cleaning takes place
    2. Preprocessing takes place
    """

    def __init__(self):
        print("\tMedical Research Council funded PD copy-typing data.\n")

    def __call__(self, long_format=True) -> pd.DataFrame:

        # Location on Neil's big machine in Sweden
        data_root = Path("../data/MRC/")

        # Read data
        raw = pd.read_csv(data_root / "CombinedTypingDataSept27.csv", header=0)

        # Clean
        df = clean_mrc(raw)

        # Preprocess: create sentences to be used in NLP model
        sentences, _ = create_sentences_from_raw_typing_mrc(df)

        # Convert into NLP-readable format
        df = create_dataframe_from_processed_data(sentences, raw)

        # Print summary stats of what we have loaded.
        dataset_summary_statistics(df)

        return df


def clean_mrc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function provides the heavy lifting in cleaning the MRC dataset.

    Parameters
    ----------
    df : pandas dataframe
        Raw MRC dataset.

    Returns
    -------
    pandas dataframe
        The cleaned dataset
    """

    remove_typed_sentences_with_high_edit_distance(df)
    remove_sentences_with_arrow_keys(df)

    # Replace following keys with an UNK symbol (in this case "£")
    df.key.replace(
        [
            "ArrowDown",
            "ArrowUp",
            "ContextMenu",
            "Delete",
            "End",
            "Enter",
            "F11",
            "F16",
            "\n",
            "Home",
            "Insert",
            "MediaPreviousTrack",
            "None",
            "NumLock",
            "PageDown",
            "Process",
            "Unidentified",
        ],
        "£",  # We use a pound-sign because Americans don't have this symbol on their keyboards.
        inplace=True,
    )

    # Make all keys lower-case
    df.key = df.key.str.lower()

    # For keys of char length > 1, we replace them with a special symbols with len() == 1
    char_replace_dict = {
        "backspace": "α",
        "shift": "β",
        "control": "γ",
        "capslock": "δ",
        "meta": "ε",
        "tab": "ζ",
        "alt": "η",
    }
    df.replace({"key": char_replace_dict}, inplace=True)

    # As well as all columns just to make life easier
    df.columns = df.columns.str.lower()

    # Return only relevant columns
    return df[["key", "type", "location", "timestamp", "participant_id", "sentence_id", "diagnosis"]].reset_index(
        drop=True
    )


def create_sentences_from_raw_typing_mrc(
    df: pd.DataFrame, make_long_format=True, time_redux_fact=10
) -> Tuple[dict, list]:

    fail = 0
    success = 0
    corrected_sentences = defaultdict(dict)
    broken_sentences = defaultdict(dict)
    char_compression_sentences = defaultdict(dict)
    for subj_idx in df.participant_id.unique():
        # Not all subjects have typed all sentences hence we have to do it this way
        for sent_idx in df.loc[(df.participant_id == subj_idx)].sentence_id.unique():

            # Locate df segment to extract
            coordinates = (df.participant_id == subj_idx) & (df.sentence_id == sent_idx)

            # Store temporary dataframe because why not
            tmp_df = df.loc[coordinates, ("key", "timestamp", "type")].reset_index(drop=True)  # Reset index

            # Action order:
            #     0. Sort dataset
            #     1. Implement backspaces
            #     2. Remove contiguous shifts
            #     3. Remove solitary keys

            # Get correctly ordered sentences and total compression times
            tmp_df = move_to_strict_striped_type_order(tmp_df)

            # Method to 'implement' the users' backspace actions
            backspace_implementer_mrc(tmp_df)

            # Removes contiguous shift presses
            combine_contiguous_shift_keydowns_without_matching_keyup(tmp_df)

            # Remove solitary key-presses which do not have a matching keyup or keydown
            remove_solitary_key_presses(tmp_df)

            # Check what we managed to achieve
            if assess_repeating_key_compression_pattern(tmp_df.type.tolist()):

                # Condition succeeds: data-collection is fixed
                corrected_sentences[subj_idx][sent_idx] = tmp_df
                success += 1

            else:

                # Condition fails: data-collection is broken
                broken_sentences[subj_idx][sent_idx] = tmp_df
                fail += 1
                print("[broken sentence] Participant: {}, Sentence: {}".format(subj_idx, sent_idx))

    for subj_idx in corrected_sentences.keys():
        # Not all subjects have typed all sentences hence we have to do it this way
        for sent_idx in corrected_sentences[subj_idx].keys():
            if make_long_format:
                # Final long-format sentences stored here
                char_compression_sentences[subj_idx][sent_idx] = "".join(
                    make_character_compression_time_sentence_mrc(
                        corrected_sentences[subj_idx][sent_idx], time_redux_fact=time_redux_fact
                    )
                )
            else:
                # We do not use the time-dimension and look only at the spatial component
                # Final long-format sentences stored here
                char_compression_sentences[subj_idx][sent_idx] = "".join(
                    corrected_sentences[subj_idx][sent_idx].key[::2]
                )  # [::2] takes into account that we only want one of the keydown-keyup pair.

    print("Percentage failed: {}".format(round(100 * (fail / (success + fail)), 2)))
    print(fail, success)

    return char_compression_sentences


def remove_solitary_key_presses(df):

    suspect_keys = []
    for key, value in Counter(df.key.tolist()).items():
        if value % 2 != 0:
            # Find all keys which appear an unequal number of times
            suspect_keys.append(key)

    # Do not remove "correction identifier key" i.e. €
    suspect_keys = [key for key in suspect_keys if key not in {"€"}]

    # Find all instances of suspect keys in df
    if len(suspect_keys) != 0:
        indices_to_keep = []
        all_idxs = []
        for key in suspect_keys:
            idxs = df.loc[df.key == key].index
            all_idxs.extend(idxs)
            # If there is more than one such key
            for pair in list(zip(idxs, idxs[1:]))[::2]:
                if pair[1] - pair[0] == 1:
                    indices_to_keep.extend(pair)

        # Take set difference to find what's left
        indices_to_remove = list(set(all_idxs) - set(indices_to_keep))

        # In-place operation, no need to return anything. Cannot reset index at this point.
        df.drop(df.index[indices_to_remove], inplace=True)
        # Reset index so that we can sort it properly in the next step
        df.reset_index(drop=True, inplace=True)


def move_to_strict_striped_type_order(df):

    df_2 = pd.DataFrame(columns=["key", "timestamp", "type"])
    indexes = []
    for i in range(len(df)):
        if i not in indexes:
            df_2 = df_2.append(df.loc[i, :])
            letter = df.loc[i, "key"]
            indexes.append(i)

            for j in range(i + 1, len(df)):
                if (df.loc[j, "key"] == df.loc[i, "key"]) and (j not in indexes):

                    df_2 = df_2.append(df.loc[j, :])
                    indexes.append(j)
                    break

    return df_2.reset_index(drop=True)


def assess_repeating_key_compression_pattern(lst, pattern=("keydown", "keyup")):

    assert set(pattern).issubset(set(lst))
    pat_len = len(pattern)
    if ("keydown" == lst[0]) and (len(lst) % pat_len == 0) and (list(pattern) * (len(lst) // pat_len) == lst):
        return True
    else:
        return False


def combine_contiguous_shift_keydowns_without_matching_keyup(df, shift_char="β"):
    """
    Function assumes that df has been sorted before getting this far.
    """

    # Get the index of all shift keydowns (these are the ones causing the registration problems)
    idxs_down = df.index[(df["key"] == shift_char) & (df["type"] == "keydown")].tolist()

    # Locate all contiguous sub-sequences
    keydown_groups = []
    for k, g in groupby(enumerate(idxs_down), lambda ix: ix[0] - ix[1]):
        keydown_groups.append(list(map(itemgetter(1), g)))

    # Check what is inside shift groups (if they only contain 'keydown' or 'keyup' there is a problem)
    removal_keydown_coordinates = []
    for g in keydown_groups:
        # Contiguous groups of shifts
        if len(g) > 1:
            ii = None
            for j in range(1, 6):
                if (df.loc[g[-1] + j, "type"] == shift_char) and (df.loc[g[-1] + j, "key"] == shift_char):
                    ii = j
            if ii:
                # Do this if the immediate key after each group is a "keyup"
                removal_keydown_coordinates.extend(g[1:])
            else:
                # Do this if there is no immediately preceeding "keyup"
                removal_keydown_coordinates.extend(g)

    # In-place operation, no need to return anything. Cannot reset index at this point.
    df.drop(df.index[removal_keydown_coordinates], inplace=True)
    # Reset index so that we can sort it properly in the next step
    df.reset_index(drop=True, inplace=True)


def range_extend_mrc(x):
    # Need to assert that this is given a sequentially ordered array
    out = list(range(x[0] - 2 * len(x), x[0] - len(x))) + list(range(x[0] - len(x), x[0])) + x
    assert np.diff(out).sum() == len(out) - 1
    return out


def make_character_compression_time_sentence_mrc(df: pd.DataFrame, time_redux_fact=10) -> str:
    long_form_sentence = []
    for i in list(df.index)[::2]:
        # Total key compression time
        comp_time = abs(df.timestamp[i + 1] - df.timestamp[i])
        # Character compressed
        long_form_sentence.append([df.key[i]] * int(comp_time // time_redux_fact))

    return flatten(long_form_sentence)


def backspace_implementer_mrc(df: pd.DataFrame, backspace_char="α"):

    # 0) Remove any singular backspaces that appear bc. data-reading problems
    idxs = df.index[(df.key == backspace_char)].tolist()
    groups = []
    remove = []
    for k, g in groupby(enumerate(sorted(idxs)), lambda ix: ix[1] - ix[0]):
        groups.append(list(map(itemgetter(1), g)))
    # Only remove ones which are actually only of list length 1
    for g in groups:
        # Data-reading error
        if len(g) == 1:
            remove.extend(g)
        # We replace these inline so we don't have to do it later
        elif len(g) == 2:
            # Place indicators [keydown]
            df.loc[g[0], "key"] = "€"
            # Place indicators [keyup]
            df.loc[g[1], "key"] = "€"

    if remove:
        # In-place droppping of rows with only one backspace
        df.drop(df.index[remove], inplace=True)
        # Reset index so that we can sort it properly in the next step
        df.reset_index(drop=True, inplace=True)

    # 1) Delete all backspace+keyups to start with
    idxs_up = df.index[(df.key == backspace_char) & (df.type == "keyup")].tolist()
    # Copy these rows for later use
    df_keyup = copy.copy(df.iloc[idxs_up])
    # In-place dropping of these rows
    df.drop(df.index[idxs_up], inplace=True)
    # Reset index so that we can sort it properly in the next step
    df.reset_index(drop=True, inplace=True)

    # 2) Find all remaining backspace+keydowns
    idxs = df.index[(df.key == backspace_char) & (df.type == "keydown")].tolist()
    contiguous_groups = []
    for k, g in groupby(enumerate(sorted(idxs)), lambda ix: ix[1] - ix[0]):
        contiguous_groups.append(list(map(itemgetter(1), g)))

    indices_to_remove = []
    if idxs:
        for g in contiguous_groups:

            gg = range_extend_mrc(g)
            # If any negative indices, correct and move indicator characters
            if any(i < 0 for i in gg):
                gg = list(filter(lambda x: x >= 0, gg))
                indices_to_remove.extend(gg[1:-1])
                # Place indicators [keydown]
                df.loc[gg[0], "key"] = "€"
            else:
                indices_to_remove.extend(gg[3:-1])
                # Place indicators [keydown]
                df.loc[gg[2], "key"] = "€"

            # Place indicators [keyup]
            # Given a value of keydown timestamp (z), select a row in the keyup df
            # where timestamp is closest to z.
            keyup_timestamp = df_keyup.loc[(df_keyup["timestamp"] >= df.loc[gg[-1], "timestamp"])].timestamp.values[0]
            df.loc[gg[-1], ("key", "timestamp", "type")] = ["€", keyup_timestamp, "keyup"]

        # In-place operation, no need to return anything. Cannot reset index at this point.
        df.drop(df.index[indices_to_remove], inplace=True)

        # Reset index so that we can sort it properly in the next step
        df.reset_index(drop=True, inplace=True)

        # Check that the indicators appear in the right places
        indicator_indices = df.index[(df.key == "€")].tolist()
        for pair in list(zip(indicator_indices, indicator_indices[1:]))[::2]:
            assert pair[1] - pair[0] == 1, indicator_indices
        assert backspace_char not in df.key.tolist()


def drop_sentences_with_faulty_data_collection(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function removes about 10% of the collected MRC data because certain sentences
    lack matching up and down key-strokes. These are necessary to calculate the total
    compression time of a key. Consequently, without them, no such calculation can be made.
    Hence they are removed. Note that this is an in-place operation

    Parameters
    ----------
    df : pd.DataFrame
        The raw MRC dataset
    """

    print("\nRemoval of sentences with faulty data collection...\n")
    print("Size of dataframe before row pruning: {}".format(df.shape))

    subjects = sorted(set(df.participant_id))
    for subj_idx in subjects:
        # Not all subjects have typed all sentences hence we have to do it this way
        for sent_idx in df.loc[(df.participant_id == subj_idx)].sentence_id.unique():
            if len(df.loc[(df.participant_id == subj_idx) & (df.sentence_id == sent_idx)]) % 2 != 0:
                # Drop in-place
                df.drop(df[(df.participant_id == subj_idx) & (df.sentence_id == sent_idx)].index, inplace=True)

    print("Size of dataframe after row pruning: {}".format(df.shape))


def get_typed_sentence_and_edit_distance(df: pd.DataFrame, edit_distances_df=None):

    # If edit distances have not been passed
    if edit_distances_df is None:
        edit_distances_df = calculate_edit_distance_between_response_and_target_MRC(df)

    subjects = sorted(set(df.participant_id))  # NOTE: set() is weakly random# Store edit distances here
    data = []
    # Loop over subjects
    for subj_idx in subjects:
        # Not all subjects have typed all sentences hence we have to do it this way
        for sent_idx in df.loc[(df.participant_id == subj_idx)].sentence_id.unique():
            if sent_idx < 16:
                coordinates = (df.participant_id == subj_idx) & (df.sentence_id == sent_idx)
                # Assign typed sentence and its corresponding edit distance
                data.append(
                    [
                        subj_idx,
                        sent_idx,
                        df.loc[coordinates, "response_content"].unique().tolist()[0],
                        edit_distances_df.loc[subj_idx, sent_idx],
                    ]
                )

    out = pd.DataFrame.from_records(data)
    out.columns = ["Subject", "Sentence ID", "Typed sentence", "Edit distance"]

    return out


def remove_sentences_with_arrow_keys(df: pd.DataFrame):

    print("\nRemoval of sentences with left/right arrows keys...\n")
    print("Size of dataframe before row pruning: {}".format(df.shape))
    # Specify arrow movements
    arrow_corpus = ["ArrowRight", "ArrowLeft"]
    # Storage
    data = []
    for arrow in arrow_corpus:
        data.append(df.loc[df["key"] == arrow, ("key", "participant_id", "sentence_id")].values)

    dff = pd.DataFrame.from_records(np.concatenate(data))
    dff.columns = ["key", "participant_id", "sentence_id"]

    left_index = sorted(dff.loc[dff.key == "ArrowLeft", "participant_id"].unique())
    right_index = sorted(dff.loc[dff.key == "ArrowRight", "participant_id"].unique())
    arrowleft_matrix = pd.DataFrame(index=left_index, columns=range(1, 16))  # 15 unique sentences
    arrowright_matrix = pd.DataFrame(index=right_index, columns=range(1, 16))  # 15 unique sentences

    # Dataframe contains the number of arrow keys used by subject and by sentence ID
    for subj_idx in left_index:
        for sent_idx in range(1, 16):
            arrowleft_matrix.loc[subj_idx, sent_idx] = dff.loc[
                (dff.participant_id == subj_idx) & (dff.sentence_id == sent_idx) & (dff.key == "ArrowLeft"), "key"
            ].count()

    # Populate
    for subj_idx in right_index:
        for sent_idx in range(1, 16):
            arrowright_matrix.loc[subj_idx, sent_idx] = dff.loc[
                (dff.participant_id == subj_idx) & (dff.sentence_id == sent_idx) & (dff.key == "ArrowRight"), "key"
            ].count()

    # Get coordinates of all rows and then drop
    for subj_idx, sent_idx in arrowleft_matrix[arrowleft_matrix != 0].stack().index:
        df.drop(df[(df["participant_id"] == subj_idx) & (df["sentence_id"] == sent_idx)].index, inplace=True)
    for subj_idx, sent_idx in arrowright_matrix[arrowright_matrix != 0].stack().index:
        df.drop(df[(df["participant_id"] == subj_idx) & (df["sentence_id"] == sent_idx)].index, inplace=True)
    print("Size of dataframe after row pruning: {}".format(df.shape))

    # Replace "Spacebar" with a blank space for homegeneity
    df.key.replace("Spacebar", " ", inplace=True)


def remove_typed_sentences_with_high_edit_distance(df: pd.DataFrame, edit_distances_df=None, threshold=75):

    print("Removal of sentences with 'high' Levenshtein distance...\n")

    # If edit distances have not been passed
    if edit_distances_df is None:
        edit_distances_df = calculate_edit_distance_between_response_and_target_MRC(df)

    assert threshold is not None, "You need to provide a cut-off value for the edit distance threshold"

    print("Size of dataframe before row pruning: {}".format(df.shape))

    subjects = sorted(set(df.participant_id))  # NOTE: set() is weakly random# Store edit distances here
    # Loop over subjects
    for subj_idx in subjects:
        # Not all subjects have typed all sentences hence we have to do it this way
        for sent_idx in df.loc[(df.participant_id == subj_idx)].sentence_id.unique():
            if edit_distances_df.loc[subj_idx, sent_idx] > threshold:
                # Remove sentence from dataframe
                df.drop(df[(df["participant_id"] == subj_idx) & (df["sentence_id"] == sent_idx)].index, inplace=True)

    print("Size of dataframe after row pruning: {}".format(df.shape))


def calculate_edit_distance_between_response_and_target_MRC(df: pd.DataFrame):
    subjects = sorted(set(df.participant_id))  # NOTE: set() is weakly random# Store edit distances here
    edit_distances_df = pd.DataFrame(index=subjects, columns=range(1, 16))  # 15 unique sentences
    # Loop over subjects
    for subj_idx in subjects:
        # Not all subjects have typed all sentences hence we have to do it this way
        for sent_idx in df.loc[(df.participant_id == subj_idx)].sentence_id.unique():
            # Locate df segment to extract
            coordinates = (df.participant_id == subj_idx) & (df.sentence_id == sent_idx)
            # Calculate the edit distance
            response = df.loc[coordinates, "response_content"].unique()[0]
            target = df.loc[coordinates, "sentence_content"].unique()[0]
            if type(response) is str and type(target) is str:
                edit_distances_df.loc[subj_idx, sent_idx] = edit_distance(response, target)
            else:
                # To handle sentences where nothing was written
                edit_distances_df.loc[subj_idx, sent_idx] = 9999

    # Replace all NaNs with 9999 as well (NaNs are sentences which were not typed)
    edit_distances_df.fillna(9999, inplace=True)

    return edit_distances_df


# ------------------------------------------- MJFF ------------------------------------------- #


class preprocessMJFF:
    """
    Governing class with which the user will interface.
    All the heavy lifting happens under the hood.
    """

    def __init__(self):
        print("\tMichael J. Fox Foundation PD copy-typing data.\n")

    def __call__(self, get_language: str = "english", include_time=True) -> pd.DataFrame:

        assert get_language in ["english", "spanish", "all"], "You must pass a valid option."

        if get_language == "all":
            # Load English MJFF data
            df_english, _ = create_MJFF_dataset("english", include_time)
            # Load Spanish MJFF data
            df_spanish, _ = create_MJFF_dataset("spanish", include_time)
            # Merge datasets
            assert all(df_english.columns == df_spanish.columns)
            df = pd.concat([df_english, df_spanish], ignore_index=True)
            # Print summary stats of what we have loaded.
            dataset_summary_statistics(df)
            return df
        else:
            df, _ = create_MJFF_dataset(get_language, include_time)
            # Print summary stats of what we have loaded.
            dataset_summary_statistics(df)
            return df


def select_attempt(df, df_meta, attempt):
    """
    Function that filters the main data based on the subjects' attempt at the task.

    Parameters
    ----------
    df : pd.DataFrame
        Contains the main data
    df_meta : pd.DataFrame
        Contains the meta information such as diagnosis
    attempt : int
        Selects either attemp 1 or attempt 2.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """

    # Add blank column
    df["attempt"] = ""
    assert set(df_meta.participant_id.unique()) == set(df.participant_id.unique())
    for idx in df.participant_id.unique():
        df.loc[(df.participant_id == idx), "attempt"] = int(df_meta.loc[(df_meta.participant_id == idx), "attempt"])

    # Select only rows with a specific attempt
    return df.loc[(df.attempt == attempt)]


def dataset_summary_statistics(df: pd.DataFrame):
    """
    Some summary statistics of the preprocessed dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed pandas dataframe.
    """
    sentence_lengths = np.stack([np.char.str_len(i) for i in df.Preprocessed_typed_sentence.unique()])
    print("Total number of study subjects: %d" % (len(set(df.Patient_ID))))
    print("Number of sentences typed by PD patients: %d" % (len(df.loc[df.Diagnosis == 1])))
    print("Number of sentences typed by controls: %d" % (len(df.loc[df.Diagnosis == 0])))
    print("Average sentence length: %05.2f" % sentence_lengths.mean())
    print("Minimum sentence length: %d" % sentence_lengths.min())
    print("Maximum sentence length: %d" % sentence_lengths.max())


def sentence_level_pause_correction_mjff(
    df, char_count_response_threshold=40, cut_off_percentile=0.99, correction_model="gengamma"
) -> Tuple[dict, list]:

    assert set(["participant_id", "key", "timestamp", "sentence_id"]).issubset(df.columns)
    # Filter out responses where the number of characters per typed
    # response, is below a threshold value (40 by default)
    df = df.groupby("sentence_id").filter(lambda x: x["sentence_id"].count() > char_count_response_threshold)
    assert not df.empty

    # Sentence identifiers
    sentences = sorted(set(df.sentence_id))  # NOTE: set() is weakly random

    # Store corrected sentences here
    corrected_timestamp_diff = defaultdict(dict)

    # Response time modelling
    pause_funcs = {"gengamma": gengamma.fit, "lognorm": lognorm.fit, "gamma": gamma.fit}
    pause_funcs_cut_off_quantile = {"gengamma": gengamma.ppf, "lognorm": lognorm.ppf, "gamma": gamma.ppf}
    pause_first_moment = {"gengamma": gengamma.mean, "lognorm": lognorm.mean, "gamma": gamma.mean}

    # Storage for critical values
    pause_replacement_stats = {}

    # Loop over all sentences
    for sent in sentences:
        timestamp_diffs = []
        # Loop over all subjects which have typed this sentence
        for sub in df.loc[(df.sentence_id == sent)].participant_id.unique():

            # Get all delta timestamps for this sentence, across all subjects
            tmp = df.loc[(df.sentence_id == sent) & (df.participant_id == sub)].timestamp.diff().tolist()
            # Store for later
            corrected_timestamp_diff[sent][sub] = np.array(tmp)
            # Append to get statistics over all participants
            timestamp_diffs.extend(tmp)

        # Move to numpy array for easier computation
        x = np.array(timestamp_diffs)

        # Remove all NANs and remove all NEGATIVE values
        x = x[~np.isnan(x)]
        # Have to do in two operations because of NaN presence
        x = x[x > 0.0]

        # Fit suitable density for modelling correct replacement value
        params_MLE = pause_funcs[correction_model](x)

        # Set cut off value
        cut_off_value = pause_funcs_cut_off_quantile[correction_model](*((cut_off_percentile,) + params_MLE))
        assert cut_off_value > 0, "INFO:\n\t value: {} \n\t sentence ID: {}".format(cut_off_value, sent)

        # Set replacement value
        replacement_value = pause_first_moment[correction_model](*params_MLE)
        assert replacement_value > 0, "INFO:\n\t value: {} \n\t sentence ID: {}".format(replacement_value, sent)

        # Store for replacement operation in next loop
        pause_replacement_stats[sent] = (cut_off_value, replacement_value)

    # Search all delta timestamps and replace which exeed cut_off_value
    for sent in sentences:
        # Loop over all subjects which have typed this sentence
        for sub in df.loc[(df.sentence_id == sent)].participant_id.unique():

            # Make temporary conversion to numpy array
            x = corrected_timestamp_diff[sent][sub][1:]  # (remove the first entry as it is a NaN)
            corrected_timestamp_diff[sent][sub] = pd.Series(
                # np.concatenate is faster than np.insert
                np.concatenate(  # Add back NaN to maintain index order
                    (
                        [np.nan],
                        # Two conditions are used here
                        np.where(
                            np.logical_or(x > pause_replacement_stats[sent][0], x < 0),
                            pause_replacement_stats[sent][1],
                            x,
                        ),
                    )
                )
            )

    return corrected_timestamp_diff


def create_char_compression_time_mjff_data(
    df: pd.DataFrame, char_count_response_threshold=40, time_redux_fact=10
) -> Tuple[dict, list]:

    assert set(["participant_id", "key", "timestamp", "sentence_id"]).issubset(df.columns)
    # Filter out responses where the number of characters per typed
    # response, is below a threshold value (40 by default)
    df = df[df.groupby(["participant_id", "sentence_id"]).key.transform("count") > char_count_response_threshold]
    assert not df.empty

    # Get the unique number of subjects
    subjects = sorted(set(df.participant_id))  # NOTE: set() is weakly random

    # All sentences will be stored here, indexed by their type
    char_compression_sentences = defaultdict(dict)

    # Get the updated compression times
    corrected_compression_times = sentence_level_pause_correction_mjff(
        df, char_count_response_threshold=char_count_response_threshold
    )

    # Loop over subjects
    for subj_idx in subjects:
        # Not all subjects have typed all sentences hence we have to do it this way
        for sent_idx in df.loc[(df.participant_id == subj_idx)].sentence_id.unique():

            # print("subject: {} -- sentence: {}".format(subj_idx, sent_idx))

            # Locate df segment to extract
            coordinates = (df.participant_id == subj_idx) & (df.sentence_id == sent_idx)

            # "correct" the sentence by operating on user backspaces
            corrected_char_sentence, removed_chars_indx = backspace_corrector(df.loc[coordinates, "key"].tolist())

            L = len(corrected_compression_times[sent_idx][subj_idx])
            assert set(removed_chars_indx).issubset(
                range(L)
            ), "Indices to remove: {} -- total length of timestamp vector: {}".format(removed_chars_indx, L)
            compression_times = corrected_compression_times[sent_idx][subj_idx].drop(index=removed_chars_indx)

            # Make long-format version of each typed, corrected, sentence
            # Note that we remove the last character to make the calculation correct.
            char_compression_sentences[subj_idx][sent_idx] = make_character_compression_time_sentence(
                compression_times, corrected_char_sentence, time_redux_fact
            )

    # No one likes an empty list so we remove them here
    for subj_idx in subjects:
        # Not all subjects have typed all sentences hence we have to do it this way
        for sent_idx in df.loc[(df.participant_id == subj_idx)].sentence_id.unique():
            # Combines sentences to contiguous sequences (if not empty)
            # if not char_compression_sentences[subj_idx][sent_idx]:
            char_compression_sentences[subj_idx][sent_idx] = "".join(char_compression_sentences[subj_idx][sent_idx])

    return char_compression_sentences


def create_char_mjff_data(df: pd.DataFrame, char_count_response_threshold: int = 40, invokation_type: int = 1) -> dict:
    """
    Function combines the typed characters, per sentence, per subject, to form proper sentences using various forms of error invokation.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data is stored in this dataframe
    char_count_response_threshold : int, optional
        Some sentences are too short hence need to be filered out, by default 40
    invokation_type : int, optional
        Selects which type of backstop correction we invoke, by default 1

    Returns
    -------
    dict
        Dictionary indexed by subjects, containing all sentences per subject
    """

    assert set(["participant_id", "key", "timestamp", "sentence_id"]).issubset(df.columns)

    # Filter out responses where the number of characters per typed
    # response, is below a threshold value (40 by default)
    df = df[df.groupby(["participant_id", "sentence_id"]).key.transform("count") > char_count_response_threshold]
    assert not df.empty

    # Get the unique number of subjects
    subjects = sorted(set(df.participant_id))  # NOTE: set() is weakly random

    # All sentences will be stored here, indexed by their type
    char_sentences = defaultdict(dict)

    # Loop over subjects
    for subj_idx in subjects:
        # Not all subjects have typed all sentences hence we have to do it this way
        for sent_idx in df.loc[(df.participant_id == subj_idx)].sentence_id.unique():

            # Locate df segment to extract
            coordinates = (df.participant_id == subj_idx) & (df.sentence_id == sent_idx)

            # "correct" the sentence by operating on user backspaces
            corrected_char_sentence, _ = backspace_corrector(
                df.loc[coordinates, "key"].tolist(), invokation_type=invokation_type
            )

            # Note that we remove the last character to make the calculation correct.
            char_sentences[subj_idx][sent_idx] = corrected_char_sentence

    # Combines the string
    for subj_idx in subjects:
        # Not all subjects have typed all sentences hence we have to do it this way
        for sent_idx in df.loc[(df.participant_id == subj_idx)].sentence_id.unique():
            # Combines sentences to contiguous sequences (if not empty)
            # if not char_compression_sentences[subj_idx][sent_idx]:
            char_sentences[subj_idx][sent_idx] = "".join(char_sentences[subj_idx][sent_idx])

    return char_sentences


def flatten(my_list):
    # Function to flatten a list of lists
    return [item for sublist in my_list for item in sublist]


def make_character_compression_time_sentence(
    compression_times: pd.Series, characters: pd.Series, time_redux_fact=10
) -> str:
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
    time_redux_fact : int, optional
        Time reduction factor, to go from milliseconds to something else, by default 10
        A millisecond is 1/1000 of a second. Convert this to centisecond (1/100s).

    Returns
    -------
    list
        Returns a list in which each character has been repeated a number of times.
    """

    assert len(compression_times) == len(characters), "Lengths are: {} and {}".format(
        len(compression_times), len(characters)
    )

    if type(compression_times) is not pd.Series:
        compression_times = pd.Series(compression_times)

    char_times = compression_times // time_redux_fact
    return flatten([[c] * int(n) for c, n in zip(characters[:-1], char_times[1:])])


def measure_levensthein_for_lang8_data(data_address: str, ld_threshold: int = 2) -> pd.DataFrame:
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
    df = pd.read_csv(data_address, sep="\t", names=["A", "B", "C", "D", "written", "corrected"])
    print("Pre-drop entries count: %s" % df.shape[0])

    # Filter out rows which do not have a correction (i.e. A  > 0) and get only raw data
    df = df.loc[df["A"] > 0].filter(items=["written", "corrected"])

    print("Post-drop entries count: %s" % df.shape[0])

    # Calculate the Levenshtein distance
    df["distance"] = df.loc[:, ["written", "corrected"]].apply(lambda x: edit_distance(*x), axis=1)

    # Only return sentence pairs of a certain LD
    return df.loc[df.distance.isin([1, ld_threshold])]


def create_mjff_training_data(df: pd.DataFrame) -> Tuple[dict, list]:
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

    assert "sentence_text" in df.columns
    assert "participant_id" in df.columns

    # Get the unique number of subjects
    subjects = sorted(set(df.participant_id))  # NOTE: set() is weakly random
    sent_ids = sorted(set(df.sentence_id))
    # All typed sentences will be stored here, indexed by their type
    typed_keys = defaultdict(dict)
    # A deconstructed dataframe by sentence ID and text only
    # df_sent_id = df.groupby(['sentence_id', 'sentence_text']).size().reset_index(drop=True)
    df_sent_id = df.loc[:, ["sentence_id", "sentence_text"]].drop_duplicates().reset_index(drop=True)

    for sub_id in subjects:
        for sent_id in sent_ids:
            typed_keys[sub_id][sent_id] = df.loc[
                (df.participant_id == sub_id) & (df.sentence_id == sent_id), "key"
            ].tolist()

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

    assert "sentence_text" in df.columns
    assert "timestamp" in df.columns
    assert "participant_id" in df.columns

    # Convert target sentences to integer IDs instead, easier to work with
    sentences = list(set(df.sentence_text))

    # Get the unique number of subjects
    subjects = set(df.participant_id)

    # The IKI time-series, per sentence, per subject is stored in this dict
    typed_sentence_IKI_ts_by_subject = defaultdict(list)

    # Loop over all the participants
    for subject in subjects:

        # Get all the necessary typed info for particular subject
        info_per_subject = df.loc[df["participant_id"] == subject][["key", "timestamp", "sentence_text"]]

        # Get all sentences of a particular type
        for sentence in sentences:

            # Append the IKI to the reference sentence store, at that sentence ID
            ts = info_per_subject.loc[info_per_subject["sentence_text"] == sentence].timestamp.values
            # Append to subject specifics
            typed_sentence_IKI_ts_by_subject[subject].extend([ts])

    # TODO: want to add some more checks here to ensure that we have not missed anything

    # No one likes an empty list so we remove them here
    for subject in subjects:
        # Remove empty arrays that may have snuck in
        typed_sentence_IKI_ts_by_subject[subject] = [
            x for x in typed_sentence_IKI_ts_by_subject[subject] if x.size != 0
        ]

    # Re-base each array so that it starts at zero.
    for subject in subjects:
        # Remove empty arrays that may have snuck in
        typed_sentence_IKI_ts_by_subject[subject] = [x - x.min() for x in typed_sentence_IKI_ts_by_subject[subject]]

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
            backspace_count_per_subject_per_sentence[subject].extend(Counter(sentence)["backspace"])

    return backspace_count_per_subject_per_sentence


def combine_characters_to_form_words_at_space(typed_keys: dict, sent_ids: list, correct: bool = True) -> dict:
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
                    completed_sentence_per_subject_per_sentence[sub_id][sent_id] = "".join(
                        backspace_corrector(typed_keys[sub_id][sent_id])
                    )
                elif correct is False:
                    # Here we remove those same backspaces from the sentence so that we
                    # can construct words. This is an in-place operation.
                    completed_sentence_per_subject_per_sentence[sub_id][sent_id] = typed_keys[sub_id][sent_id].remove(
                        "backspace"
                    )
    elif correct == -1:
        # We enter here if we do not want any correction to our sentences, implicitly this means that we
        # keep all the backspaces in the sentence as characters.
        for sub_id in typed_keys.keys():
            for sent_id in sent_ids:
                completed_sentence_per_subject_per_sentence[sub_id][sent_id] = "".join(typed_keys[sub_id][sent_id])

    return completed_sentence_per_subject_per_sentence


def range_extend(x):
    # Need to assert that this is given a sequentially ordered array
    return list(np.array(x) - len(x)) + x


def remove_leading_backspaces(x, removal_character):
    # Recursive method to remove the leading backspaces
    # Function recursively removes the leading backspace(s) if present
    if x[0] == removal_character:
        return remove_leading_backspaces(x[1:], removal_character)
    else:
        return x


def backspace_corrector(
    sentence: list, removal_character="backspace", invokation_type=1, verbose: bool = False
) -> Tuple[list, list]:
    """
    Method corrects the sentence by logically acting on the backspaces invoked by the
    subject.

    Parameters
    ----------
    sentence : list
        List of characters which form a sentence when combined
    removal_character : str, optional
        Which character is encoded by as a removal action, by default "backspace"
    invokation_type : int, optional
        Selects which type of invokation we employ, by default 1
    verbose : bool, optional
        Prints out the edits being made, by default False

    Returns
    -------
    Tuple[list, list]
        The corrected sentence and coordinates removed
    """

    # TODO: remember that this proceedure has changed, needs to mimic the behaviour of the MRC functionality: i.e. the error is kept in the sentence.

    # Want to pass things as list because it easier to work with w.r.t. to strings
    assert isinstance(sentence, list)
    assert invokation_type in [-1, 0, 1]
    original_sentence = copy.copy(sentence)

    # Check that we are not passing a sentence which only contains backspaces
    if [removal_character] * len(sentence) == sentence:
        # Return an empty list which will get filtered out at the next stage
        return []

    # Special case: the backspace action is kept in the sentence but replaced with a single
    # character to map the action to a single unit.
    if invokation_type == -1:
        # In place of 'backspace' we use a pound-sign
        return ["#" if x == removal_character else x for x in sentence], None

    # Apply to passed sentence
    pre_removal_length = len(sentence)
    sentence = remove_leading_backspaces(sentence, removal_character)
    post_removal_length = len(sentence)
    nr_leading_chars_removed = pre_removal_length - post_removal_length

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
                remove_cords.extend([group[0] - 1, group[0]])

            else:
                remove_cords.extend(range_extend(group))

    elif invokation_type == 1:
        # Remove all characters indicated by 'backspace' _except_ the last character which is kept

        for group in backspace_groups:

            # Solitary backspace removal proceedure
            if len(group) == 1:
                # We do it this way because: because we want to capture these error slips
                remove_cords.extend([group[0]])
                # Remove _just_ the backspace and nothing else, don't invoke
            else:
                # XXX: this _may_ introduce negative indices at the start of a sentence
                # these are filtered out further down
                remove_cords.extend(range_extend(group[:-1]))  # This invokes the n-1 backspaces
                # This removes the nth backspace and the immediately following character
                remove_cords.extend([group[-1], group[-1] + 1])

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


def create_dataframe_from_processed_data(my_dict: dict, df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Function creates a pandas DataFrame which will be used by the NLP model
    downstream.

    Parameters
    ----------
    my_dict : dict
        Dictionary containing all the preprocessed typed sentences, indexed by subject
    df_meta : Pandas DataFrame
        Contains the mete information on each patient

    Returns
    -------
    Pandas DataFrame
        Returns the compiled dataframe from all subjects.
    """

    assert "diagnosis" in df_meta.columns

    final_out = []
    for participant_id in my_dict.keys():
        final_out.append(
            pd.DataFrame(
                [
                    [
                        participant_id,
                        int(df_meta.loc[df_meta.participant_id == participant_id, "diagnosis"]),
                        str(sent_id),
                        my_dict[participant_id][sent_id],
                    ]
                    for sent_id in my_dict[participant_id].keys()
                ]
            )
        )
    df = pd.concat(final_out, axis=0)
    df.columns = ["Patient_ID", "Diagnosis", "Sentence_ID", "Preprocessed_typed_sentence"]

    # Final check for empty values
    df["Preprocessed_typed_sentence"].replace("", np.nan, inplace=True)
    # Remove all such rows
    df.dropna(subset=["Preprocessed_typed_sentence"], inplace=True)
    return df


def remap_English_MJFF_participant_ids(df):
    replacement_ids = {}
    for the_str in set(df.Patient_ID):
        base = str("".join(map(str, [int(s) for s in the_str.split()[0] if s.isdigit()])))
        replacement_ids[the_str] = base
    return df.replace({"Patient_ID": replacement_ids})


def create_MJFF_dataset(
    language="english", include_time=True, attempt=1, invokation_type=1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end creation of raw data to NLP readable train and test sets.

    Parameters
    ----------
    langauge : str
        Select which language should be preprocessed.
    attempt: int
        Select which attempt we are intersted in.

    Returns
    -------
    pandas dataframe
        Processed dataframe
    """

    # Load raw text and meta data
    if language == "english":
        df_meta = pd.read_csv(
            data_root / "MJFF" / "raw" / "EnglishParticipantKey.csv",
            header=0,
            names=["participant_id", "ID", "attempt", "diagnosis"],
        )
        df = pd.read_csv(data_root / "MJFF" / "raw" / "EnglishData-duplicateeventsremoved.csv")

        # Select which attempt we are interested in.
        df = select_attempt(df, df_meta, attempt=attempt)

    elif language == "spanish":
        df = pd.read_csv(data_root / "MJFF" / "raw" / "SpanishData-duplicateeventsremoved.csv")
        df_meta = pd.read_csv(
            data_root / "MJFF" / "raw" / "SpanishParticipantKey.csv",
            header=0,
            names=["participant_id", "ID", "attempt", "diagnosis"],
        )
        df_meta.index = df_meta.index.astype(str)

        # Post-processing of the data could have lead to corrupted entries
        uncorrupted_participants = [i for i in set(df.participant_id) if i.isdigit()]

        # There is no label for subject 167, so we remove her here.
        uncorrupted_participants.remove("167")
        df = df[df["participant_id"].isin(uncorrupted_participants)]

        # 'correct' Spanish characters
        df = create_proper_spanish_letters(df)

        # Select which attempt we are interested in.
        df = select_attempt(df, df_meta, attempt)

    else:
        raise ValueError

    # Get the tuple (sentence ID, reference sentence) as a dataframe
    reference_sentences = df.loc[:, ["sentence_id", "sentence_text"]].drop_duplicates().reset_index(drop=True)

    if include_time:
        # This option includes information on: character and timing

        # Creates long sequences with characters repeated for IKI number of steps
        out = create_char_compression_time_mjff_data(df)
    else:
        # This option _only_ includes the characters.
        out = create_char_mjff_data(df, invokation_type=invokation_type)

    # Final formatting of typing data
    df = create_dataframe_from_processed_data(out, df_meta).reset_index(drop=True)

    if language == "english":
        # Remap participant identifiers so that that e.g. 10a -> 10 and 10b -> 10.
        return remap_English_MJFF_participant_ids(df), reference_sentences

    # Return the empirical data and the reference sentences for downstream tasks
    return df, reference_sentences


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

    if socket.gethostname() == "pax":
        # Monster machine
        data_root = "../data/MJFF/"  # My local path
        data_root = Path(data_root)
    else:
        # Laptop
        data_root = "/home/nd/data/liverpool/MJFF"  # My local path
        data_root = Path(data_root)

    # Raw data
    df = pd.read_csv(data_root / "EnglishData.csv")
    df_meta = pd.read_csv(data_root / "EnglishParticipantKey.csv")

    if use_mechanical_turk:
        df_mt = pd.read_csv(data_root / "MechanicalTurkCombinedEnglishData.csv")
        df_meta_mt = pd.read_csv(data_root / "MechanicalTurkEnglishParticipantKey.csv")
        # Drop columns from main data to facilitate concatenation
        df.drop(columns=["parameters_workerId", "parameters_consent"], inplace=True)
        assert all(df.columns == df_mt.columns)
        # Combine
        df = pd.concat([df, df_mt]).reset_index(drop=True)
        df_meta = pd.concat([df_meta, df_meta_mt]).reset_index(drop=True)

    # Extracts all the characters per typed sentence, per subject
    out, numerical_sentence_ids, reference_sentences = create_mjff_training_data(df)

    # Make proper sentences from characters, where default is to invoke backspaces
    out = combine_characters_to_form_words_at_space(out, numerical_sentence_ids)

    # Return the empirical data and the reference sentences for downstream tasks
    return (
        create_dataframe_from_processed_data(out, numerical_sentence_ids, df_meta).reset_index(drop=True),
        reference_sentences.loc[:, ["sentence_id", "sentence_text"]],
    )


def create_NLP_datasets_from_MJFF_Spanish_data() -> pd.DataFrame:
    """
    Creatae NLP-readable dataset from Spanish MJFF data.
    """

    # Monster machine
    data_root = "../data/MJFF/"  # Relative path
    data_root = Path(data_root)

    # Meta
    df_meta = pd.read_csv(data_root / "SpanishParticipantKey.csv")

    # Text
    df = pd.read_csv(data_root / "SpanishData.csv")

    # There is no label for subject 167, so we remove her here.
    df = df.query("participant_id != 167")

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

    assert set(["participant_id", "key", "timestamp", "sentence_id"]).issubset(df.columns)
    special_spanish_characters = ["´", "~", '"']
    char_unicodes = [u"\u0301", u"\u0303", u"\u0308"]
    unicode_dict = dict(zip(special_spanish_characters, char_unicodes))

    # Get the unique number of participants (control AND pd)
    subjects = sorted(set(df.participant_id))  # NOTE: set() is weakly random
    # Sentence identifiers
    sentences = sorted(set(df.sentence_id))  # NOTE: set() is weakly random
    # Loop over all sentences
    for sent in sentences:
        # Loop over all subjects
        for sub in subjects:
            # Get typed sentence
            typed_characters = df.loc[(df.participant_id == sub) & (df.sentence_id == sent)].key.values
            typed_chars_index = df.loc[(df.participant_id == sub) & (df.sentence_id == sent)].index
            if any(c in special_spanish_characters for c in typed_characters):
                # Check which character is present in the typed sentence
                for char in special_spanish_characters:
                    if char in typed_characters:
                        # At these coordinates the _singular_ special chars live
                        coordinates_to_remove = np.where(typed_characters == char)[0]
                        # Assign the proper Spanish character in the dataframe
                        for i in coordinates_to_remove:
                            df.loc[typed_chars_index[i + 1], "key"] = typed_characters[i + 1] + unicode_dict[char]
                        # Drop the singular characters in place
                        df.drop(typed_chars_index[coordinates_to_remove], inplace=True)

    return df


# Basline calculation stuff


def create_mjff_baselines(df, df_meta, attempt=1, invokation_type=1):

    # Select which attempt we are considering
    df = select_attempt(df, df_meta, attempt = attempt)

    # First calculate the IKI for each sentence
    # TODO: does this need to be moved after the edit-distance?
    df_iki = get_iki_baseline_df(df, df_meta, invokation_type = invokation_type)

    # Second calculate the edit-distance
    df, reference_sentences = create_MJFF_dataset("english", False, attempt, invokation_type)
    df_edit = get_edit_distance_df(df, reference_sentences)

    # Combine all measures
    df = combine_iki_and_edit_distance(df_edit, df_iki)
    assert not df.isnull().values.any()

    return df


def get_iki_baseline_df(df, df_meta, invokation_type):

    assert len(df.attempt.unique()) == 1

    # Corrected compression times (i.e. timestamp difference / delta)
    # TODO: this needs to be updated to match the MRC format
    corrected_compression_times = sentence_level_pause_correction_mjff(df)

    data = []
    # Loop over sentence IDs
    for sentence in corrected_compression_times.keys():
        # Loop over participant IDs
        for participant in corrected_compression_times[sentence]:

            if invokation_type == 1:
                # Find the correction to the IKI

                # Locate df segment to extract
                coordinates = (df.participant_id == participant) & (df.sentence_id == sentence)

                # Not all participants typed all sentences, this conditions check that
                if len(df[coordinates]) != 0:

                    # "correct" the sentence by operating on user backspaces
                    _, removed_chars_indx = backspace_corrector(
                        df.loc[coordinates, "key"].tolist(), invokation_type=invokation_type
                    )

                    L = len(corrected_compression_times[sentence][participant])
                    assert set(removed_chars_indx).issubset(
                        range(L)
                    ), "Indices to remove: {} -- total length of timestamp vector: {}".format(removed_chars_indx, L)

                    # Adjust actual compression times
                    iki = corrected_compression_times[sentence][participant].drop(index=removed_chars_indx)

            elif invokation_type == -1:
                # Uncorrected IKI extracted here
                iki = corrected_compression_times[sentence][participant][1:]

            else:
                # TODO: clean this up for other options
                raise ValueError

            # Append to list which we'll pass to a dataframe in subsequent cells
            data.append(
                (
                    participant,
                    sentence,
                    int(df_meta.loc[(df_meta.participant_id == participant), "diagnosis"]),
                    iki.mean(),
                    iki.var(),
                )
            )

    col_names = ["Patient_ID", "Sentence_ID", "Diagnosis", "Mean_IKI", "Var_IKI"]
    df_iki = remap_English_MJFF_participant_ids(pd.DataFrame(data, columns=col_names))
    df_iki.dropna(inplace=True)
    df_iki.reset_index(drop=True, inplace=True)
    assert not df_iki.isnull().values.any()

    return df_iki


def calculate_all_baseline_ROC_curves(df, test_size=0.25):
    measures = ["edit_distance", "Mean_IKI", "Diagnosis"]
    assert set(measures).issubset(df.columns)
    all_combinations = ["Mean_IKI", ["edit_distance", "Mean_IKI"]]
    # Store all results in a dict which will be passed to plotting
    results = {"I": None, "II": None}
    assert len(results) == len(all_combinations)
    for i, j in zip(all_combinations, results.keys()):
        if isinstance(i, list):
            X = df[i].to_numpy()
        else:
            X = df[i].to_numpy().reshape(-1, 1)
        # targets
        y = df.Diagnosis.to_numpy()

        # We use 25% of our data for test [sklearn default]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Classify
        clf = SVC(class_weight="balanced", gamma="auto", probability=True)
        clf.fit(X_train, y_train)
        y_probas = clf.predict_proba(X_test)

        # Store
        results[j] = (y_test, y_probas[:, 1])

    return results


def combine_iki_and_edit_distance(df_edit, df_iki):
    df_edit["Mean_IKI"] = ""
    for idx in df_edit.Patient_ID.unique():
        for sent_idx in df_edit.loc[df_edit.Patient_ID == idx].Sentence_ID:
            a = float(df_iki[(df_iki.Patient_ID == idx) & (df_iki.Sentence_ID == int(sent_idx))]["Mean_IKI"])
            df_edit.loc[(df_edit.Patient_ID == idx) & (df_edit.Sentence_ID == sent_idx), "Mean_IKI"] = a

    return df_edit


def get_edit_distance_df(df, ref):
    assert len(df.attempt.unique()) == 1
    df["edit_distance"] = ""
    for idx in df.Patient_ID.unique():
        for sent_idx in df.loc[df.Patient_ID == idx].Sentence_ID:
            a = df[(df.Patient_ID == idx) & (df.Sentence_ID == sent_idx)]["Preprocessed_typed_sentence"].values[0]
            b = ref[ref.sentence_id == int(sent_idx)]["sentence_text"].values[0]
            df.loc[(df.Patient_ID == idx) & (df.Sentence_ID == sent_idx), "edit_distance"] = ed(a, b)
    return df

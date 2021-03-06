import copy
import re
import socket
import warnings
from collections import Counter, defaultdict
from itertools import chain, count, groupby
from operator import itemgetter
from os import path
from pathlib import Path
from statistics import median
from typing import Tuple

import numpy as np
import pandas as pd
from nltk.metrics import edit_distance  # Levenshtein
from scipy.stats import gamma, gengamma, iqr, lognorm

# ------------------------------------------ MRC------------------------------------------ #


def remove_superfluous_reponse_id_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate rows for certain (subject, sentence) indices where the data-storing process became corrupted, causing multiple versions of the same sentence to be stored under the same aforementioned index.

    This function removes the duplicates entries, and ensures that only one response_id is recorded per typed sentence.

    Parameters
    ----------
    df : pd.DataFrame
        Raw pandas dataframe (file loaded from "CombinedTypingDataSept27.csv")

    Returns
    -------
    pd.DataFrame
        Dataframe without any duplicates per row.
    """
    indices_to_remove = []
    for subj_idx in df.participant_id.unique():
        # Not all subjects have typed all sentences hence we have to do it this way
        for sent_idx in df.loc[(df.participant_id == subj_idx)].sentence_id.unique():
            # Locate df segment to extract
            coordinates = (df.participant_id == subj_idx) & (df.sentence_id == sent_idx)
            # In practise each sentence should have only _one_ response_id,
            # more than one is evidence of a corrupted data-reading process.
            re_ID = df.loc[coordinates].response_id.unique().tolist()
            if len(re_ID) != 1:
                # Response IDs to remove, get indices of these
                indices_to_remove.extend(df.loc[coordinates & (df.response_id.isin(re_ID[1:]))].index)

    # Once we have located all superfluous response IDs we drop them in-place and reset the index
    df.drop(df.index[indices_to_remove], inplace=True)
    # Reset index so that we can sort it properly in the next step
    df.reset_index(drop=True, inplace=True)


class preprocessMRC:
    def __init__(self):
        """
        This class is somewhat different to the MJFF loading class. See options below.

        Parameters
        ----------
        which_attempt : int, optional
            There are two options here:
                1) which_attempt=1 corresponds to controls vs. unmedicated patients
                1) which_attempt=2 corresponds to controls vs. medicated patients
        include_time : bool, optional
            If we are using long-format or not, by default True

        Returns
        -------
        pd.DataFrame
            Dataframe which headers: subject_id, sentence_id, medication, processed_typed_sentence
        """
        print("\tMedical Research Council funded PD copy-typing data.\n")

    def __call__(self, robustly_scale=False) -> pd.DataFrame:
        if robustly_scale:
            df, df_scaled = create_MRC_dataset(robustly_scale=robustly_scale)
            dataset_summary_statistics(df)
            return df, df_scaled
        else:
            df = create_MRC_dataset(robustly_scale=robustly_scale)
            # Print summary stats of what we have loaded.
            dataset_summary_statistics(df)
            return df


def create_MRC_dataset(
    robustly_scale=False, include_time=False, attempt=None, invokation_type=1, drop_shift=True
) -> pd.DataFrame:
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
    data_root = Path("data/MRC/")  # My local path
    backspace_char = "α"
    shift_char = "β"
    df = process_mrc(pd.read_csv(data_root / "MRCData-processed-interpolated.csv"))

    # Select which attempt we are interested in
    if attempt == 1:
        # Controls vs unmedicated patients
        df = df.loc[(df.medication == 0)]
    elif attempt == 2:
        # Controls vs medicated patients
        df = pd.concat([df[(df.diagnosis == 1) & (df.medication == 1)], df[df.diagnosis == 0]], ignore_index=True)
    elif not attempt:
        #  Use all data
        pass
    else:
        raise ValueError

    # Remove shift characters or not
    if drop_shift:
        # In-place dropping of these rows
        df.drop(df.index[(df.key == shift_char)], inplace=True)
        # Reset index so that we can sort it properly in the next step
        df.reset_index(drop=True, inplace=True)

    assert all(df.groupby(["participant_id", "sentence_id"]).key.transform("count") > 40)

    if include_time:
        #  FYI: this option is not used anymore

        # Creates long sequences with characters repeated for IKI number of steps
        sentence_dictionary, location_dictionary = create_char_iki_extended_data(
            df, backspace_char=backspace_char, invokation_type=1
        )
        iki_timings_dictionary, sentence_list_dictionary, hold_down_dict = None, None, None

    else:
        # This option _only_ includes the characters.
        (
            sentence_dictionary,
            location_dictionary,
            iki_timings_dictionary,
            sentence_list_dictionary,
            hold_down_dict,
            pause_dict,
        ) = create_char_data(df, backspace_char=backspace_char, invokation_type=invokation_type)

    if robustly_scale:
        # Final formatting of typing data (note that df contains the diagnosis here)
        final = create_dataframe_from_processed_data_mrc(
            sentence_dictionary,
            location_dictionary,
            iki_timings_dictionary,
            sentence_list_dictionary,
            hold_down_dict,
            pause_dict,
            df,
        ).reset_index(drop=True)

        #  Re-map the sentence-ID columns, got messed up during logging
        df_normal = remap_sentence_ids_for_control_subjects_mrc(final)

        #  Robustly scale the results
        iki_timings_dictionary = robustly_scale_timings(iki_timings_dictionary)
        hold_down_dict = robustly_scale_timings(hold_down_dict)
        pause_dict = robustly_scale_timings(pause_dict)

        # Final formatting of typing data (note that df contains the diagnosis here)
        final = create_dataframe_from_processed_data_mrc(
            sentence_dictionary,
            location_dictionary,
            iki_timings_dictionary,
            sentence_list_dictionary,
            hold_down_dict,
            pause_dict,
            df,
        ).reset_index(drop=True)

        #  Re-map the sentence-ID columns, got messed up during logging
        df_scaled = remap_sentence_ids_for_control_subjects_mrc(final)

        return df_normal, df_scaled

    # Final formatting of typing data (note that df contains the diagnosis here)
    final = create_dataframe_from_processed_data_mrc(
        sentence_dictionary,
        #location_dictionary,
        iki_timings_dictionary,
        sentence_list_dictionary,
        hold_down_dict,
        pause_dict,
        df,
    ).reset_index(drop=True)

    #  Re-map the sentence-ID columns, got messed up during logging
    df = remap_sentence_ids_for_control_subjects_mrc(final)

    # Return the empirical data and the reference sentences for downstream tasks
    return df


def robustly_scale_parameters(timings):
    arr = np.hstack([timings[i] for i in timings.keys()])
    return np.median(arr), iqr(arr)


def robustly_scale_timings(my_dict):
    # Loop over subjects
    for participant_id in my_dict.keys():
        # Get scaling paramers
        median, iqr = robustly_scale_parameters(my_dict[participant_id])
        # Not all subjects have typed all sentences hence we have to do it this way
        for sent_id in my_dict[participant_id].keys():
            my_dict[participant_id][sent_id] = [(x - median) / iqr for x in my_dict[participant_id][sent_id]]

    return my_dict


def process_mrc(df: pd.DataFrame, unk_symbol="£") -> pd.DataFrame:
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

    # Remove subject 1039 and 138, their responses are faulty
    for i in [1039, 138]:
        df.drop(df.index[(df.participant_id == i)], inplace=True)
    # Reset index so that we can sort it properly in the next step
    df.reset_index(drop=True, inplace=True)

    # Remove duplicate response IDs -- only _one_ response ID per subject
    remove_superfluous_reponse_id_rows(df)
    # Some sentences are too corrupted to be useuful so are dropped
    remove_typed_sentences_with_high_edit_distance(df)
    # Some subjects have used left/right arrow keys which is not allowed
    remove_sentences_with_arrow_keys(df)

    # Replace following keys with an UNK symbol (in this case "£")
    keys2replace = [
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
    ]
    keys2replace = keys2replace + [key.lower() for key in keys2replace]

    df.key.replace(
        keys2replace,
        unk_symbol,  # We use a pound-sign because Americans don't have this symbol on their keyboards.
        inplace=True,
    )

    # Make all keys lower-case
    df.key = df.key.str.lower()

    # For keys of char length > 1, we replace them with a special symbols with len() == 1
    modifier_key_replacement_dict = modifier_key_replacements()
    df.replace({"key": modifier_key_replacement_dict}, inplace=True)

    # Add a medication column to tell if they are on medication or not
    assign_medication_column(df)

    # Make column names lower case as well, to make life easier
    df.columns = df.columns.str.lower()

    # Return only relevant columns
    return df[
        ["key", "participant_id", "sentence_id", "diagnosis", "medication", "keydown", "keyup"]
    ].reset_index(drop=True)


def modifier_key_replacements() -> dict:
    return {"backspace": "α", "shift": "β", "control": "γ", "capslock": "δ", "meta": "ε", "tab": "ζ", "alt": "η"}


def remap_sentence_ids_for_control_subjects_mrc(df):
    # PD subjects have ID numbers which are all less than 1000
    control_subjects = [i for i in df.Participant_ID.unique() if i > 1000]
    target_maps = [str(i) for i in [2, 4, 5, 3, 1, 10, 9, 7, 8, 14, 6, 15, 12, 11, 13]]
    key_maps = [str(i) for i in range(1, 16)]
    correct_sentence_id_mapping = dict(zip(key_maps, target_maps))
    A = df.loc[(df.Participant_ID.isin(control_subjects))]["Sentence_ID"].map(correct_sentence_id_mapping).tolist()
    df.loc[(df.Participant_ID.isin(control_subjects)), "Sentence_ID"] = A
    return df


def assign_medication_column(df):
    # Load master reference for medication
    meds = pd.read_csv(Path("data/MRC") / "MedicationInfo.csv")
    # Pre-assign a column of zeros
    N, _ = df.shape
    df["medication"] = np.zeros(N, dtype=int)
    for i in meds.TypingID:
        df.loc[df.participant_id == i, "medication"] = int(meds[(meds.TypingID == i)].MEDICATED)


def make_char_compression_sentences_from_raw_typing_mrc(
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


def remove_solitary_key_presses(df, verbose=False):

    suspect_keys = []
    for key, value in Counter(df.key.tolist()).items():
        if value % 2 != 0:
            # Find all keys which appear an unequal number of times
            suspect_keys.append(key)

    # Do not remove "correction identifier key" i.e. ω
    suspect_keys = [key for key in suspect_keys if key not in {"ω", "α"}]

    if verbose:
        print(suspect_keys)

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

    # Get the index of all shift keydowns (these are the ones causing the registration problems)
    idxs_down = df.index[(df["key"] == shift_char) & (df["type"] == "keydown")].tolist()

    # Locate all contiguous sub-sequences
    keydown_groups = []
    for k, g in groupby(enumerate(idxs_down), lambda ix: ix[0] - ix[1]):
        keydown_groups.append(list(map(itemgetter(1), g)))

    # Select only proper groups not singular shift keys
    keydown_groups = [i for i in keydown_groups if len(i) > 1]
    # If groups exist
    if keydown_groups:
        # Check what is inside shift groups (if they only contain 'keydown' or 'keyup' there is a problem)
        removal_indices = []
        # Only look at groups which are longer than 1
        for g in keydown_groups:
            # Contiguous groups of shifts
            shift_keyup_index = None
            # Looking at a range(1,6) is somewhat arbitrary and no proper condition
            # for finding a more optimal range has been considered. This, however,
            # can easily be accommodated for.
            for j in range(1, 6):
                # This is true if we find an immediately preceeding "keyup"
                if (df.loc[g[-1] + j, "type"] == "keyup") and (df.loc[g[-1] + j, "key"] == shift_char):
                    shift_keyup_index = j + g[-1]
                    # The break prevents the loop from continuing should we be at the end
                    # if an array for example.
                    break
            if shift_keyup_index:
                # Do this if the immediate key after each group is a "keyup"
                removal_indices.extend(g[1:])
                # Move they shift+keyup index so it preceds the shift+keydown block
                df = insert_row_in_dataframe(df, shift_keyup_index, g[-1])
            else:
                # Do this if there is no immediately preceeding "keyup"
                removal_indices.extend(g)
        # Final drop and index reset
        df.drop(df.index[removal_indices], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    else:
        return df


def insert_row_in_dataframe(df, index_to_insert, location_to_insert):
    """Function to properly order shift up and shift down.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe which contains the raw data
    index_to_insert : int
        Index of the row we wish to move to the correct position
    location_to_insert : int
        The index after which we wish to insert a row.

    Returns
    -------
    pd.DataFrame
        A correctly ordered dataframe which consecutive keydown and keyups
    """

    assert set([index_to_insert, location_to_insert]).issubset(set(df.index))

    # Row to insert
    row = pd.DataFrame(df.loc[index_to_insert].to_dict(), index=[location_to_insert])
    # New dataframe
    df_new = pd.concat([df.loc[:location_to_insert], row, df.loc[(location_to_insert + 1) :]])
    # Drop the old row at its old location
    df_new.drop(index_to_insert, inplace=True)
    # Reset index
    df_new.reset_index(drop=True, inplace=True)

    return df_new


def make_character_compression_time_sentence_mrc(df: pd.DataFrame, time_redux_fact=10) -> str:
    long_form_sentence = []
    for i in list(df.index)[::2]:
        # Total key compression time
        comp_time = abs(df.timestamp[i + 1] - df.timestamp[i])
        # Character compressed
        long_form_sentence.append([df.key[i]] * int(comp_time // time_redux_fact))

    return flatten(long_form_sentence)


def range_extend_mrc(x):
    # Need to assert that this is given a sequentially ordered array
    out = list(range(x[0] - 2 * len(x), x[0] - len(x))) + list(range(x[0] - len(x), x[0])) + x
    assert np.diff(out).sum() == len(out) - 1
    return out


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
            df.loc[g[0], "key"] = "ω"
            # Place indicators [keyup]
            df.loc[g[1], "key"] = "ω"

    if remove:
        # In-place droppping of rows with only one backspace
        df.drop(df.index[remove], inplace=True)
        # Reset index so that we can sort it properly in the next step
        df.reset_index(drop=True, inplace=True)

    # 1) Delete all backspace+keyups to start with
    idxs_up = df.index[(df.key == backspace_char) & (df.type == "keyup")].tolist()
    # Copy these rows for later use
    df_keyup = df.iloc[idxs_up].copy()
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
                df.loc[gg[0], "key"] = "ω"
            else:
                indices_to_remove.extend(gg[3:-1])
                # Place indicators [keydown]
                df.loc[gg[2], "key"] = "ω"

            # Place indicators [keyup]
            # Given a value of keydown timestamp (z), select a row in the keyup df
            # where timestamp is closest to z.
            keyup_timestamp = df_keyup.loc[(df_keyup["timestamp"] >= df.loc[gg[-1], "timestamp"])].timestamp.values[0]
            df.loc[gg[-1], ("key", "timestamp", "type")] = ["ω", keyup_timestamp, "keyup"]

        # In-place operation, no need to return anything. Cannot reset index at this point.
        df.drop(df.index[indices_to_remove], inplace=True)

        # Reset index so that we can sort it properly in the next step
        df.reset_index(drop=True, inplace=True)

        # Check that the indicators appear in the right places
        indicator_indices = df.index[(df.key == "ω")].tolist()
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
    arrow_corpus = ["arrowright", "arrowleft"]
    # Storage
    data = []
    for arrow in arrow_corpus:
        data.append(df.loc[df["key"].str.lower() == arrow, ("key", "participant_id", "sentence_id")].values)

    dff = pd.DataFrame.from_records(np.concatenate(data))
    dff.columns = ["key", "participant_id", "sentence_id"]

    left_index = sorted(dff.loc[dff.key == "arrowleft", "participant_id"].unique())
    right_index = sorted(dff.loc[dff.key == "arrowright", "participant_id"].unique())
    arrowleft_matrix = pd.DataFrame(index=left_index, columns=range(1, 16))  # 15 unique sentences
    arrowright_matrix = pd.DataFrame(index=right_index, columns=range(1, 16))  # 15 unique sentences

    # Dataframe contains the number of arrow keys used by subject and by sentence ID
    for subj_idx in left_index:
        for sent_idx in range(1, 16):
            arrowleft_matrix.loc[subj_idx, sent_idx] = dff.loc[
                (dff.participant_id == subj_idx) & (dff.sentence_id == sent_idx) & (dff.key == "arrowleft"), "key"
            ].count()

    # Populate
    for subj_idx in right_index:
        for sent_idx in range(1, 16):
            arrowright_matrix.loc[subj_idx, sent_idx] = dff.loc[
                (dff.participant_id == subj_idx) & (dff.sentence_id == sent_idx) & (dff.key == "arrowright"), "key"
            ].count()

    # Get coordinates of all rows and then drop
    for subj_idx, sent_idx in arrowleft_matrix[arrowleft_matrix != 0].stack().index:
        df.drop(df[(df["participant_id"] == subj_idx) & (df["sentence_id"] == sent_idx)].index, inplace=True)
    for subj_idx, sent_idx in arrowright_matrix[arrowright_matrix != 0].stack().index:
        df.drop(df[(df["participant_id"] == subj_idx) & (df["sentence_id"] == sent_idx)].index, inplace=True)
    print("Size of dataframe after row pruning: {}".format(df.shape))

    # Replace "Spacebar" with a blank space for homegeneity
    df.key.replace("spacebar", " ", inplace=True)


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

    def __call__(self, which_language: str = "english", which_attempt=1, include_time=True) -> pd.DataFrame:

        assert which_language in ["english", "spanish", "all"], "You must pass a valid option."

        if which_language == "all":
            # Load English MJFF data
            df_english, _ = create_MJFF_dataset("english", include_time, attempt=which_attempt)
            # Load Spanish MJFF data
            df_spanish, _ = create_MJFF_dataset("spanish", include_time, attempt=which_attempt)
            # Merge datasets
            assert all(df_english.columns == df_spanish.columns)
            df = pd.concat([df_english, df_spanish], ignore_index=True)
            # Print summary stats of what we have loaded.
            dataset_summary_statistics(df)
            return df
        else:
            df, _ = create_MJFF_dataset(which_language, include_time, attempt=which_attempt)
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
    print("\nTotal number of study subjects: %d" % (len(df.Participant_ID.unique())))
    print("Number of sentences typed by PD patients: %d" % (len(df.loc[df.Diagnosis == 1])))
    print("Number of sentences typed by controls: %d" % (len(df.loc[df.Diagnosis == 0])))
    print("Average sentence length: %05.2f" % sentence_lengths.mean())
    print("Minimum sentence length: %d" % sentence_lengths.min())
    print("Maximum sentence length: %d" % sentence_lengths.max())


def typing_dynamics_editor(
    df: pd.DataFrame,
    what_to_correct="hold_time",
    char_count_response_threshold: int = 40,
    cut_off_percentile: int = 99,
    correction_model: str = "gengamma",
) -> Tuple[dict, dict]:
    """
    Function is used to correct the typing dynamics, to attend to anomalies like subjects stopping mid-typing
    to attend to other matters, thus causing faulty temporal dynamics.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe which contains the raw typed sentences (MJFF or MRC dataset)
    char_count_response_threshold : int, optional
        The minimum number of characters required for an entry to be considered a valid attempt
    cut_off_percentile : float, optional
        IKI values above this threshold are replace by the first moment of correction_model, by default 99
    correction_model : str, optional
        Generative model used to do adjusments, by default "gengamma"

    Returns
    -------
    Tuple[dict, list]
        Dictionary of sentences indexed by subject
    """

    assert what_to_correct in ["pause", "iki", "hold_time"]
    assert set(["participant_id", "key", "timestamp", "sentence_id", "keydown", "keyup"]).issubset(df.columns)
    # Filter out responses where the number of characters per typed
    # response, is below a threshold value (40 by default)
    df = df.groupby("sentence_id").filter(lambda x: x["sentence_id"].count() > char_count_response_threshold)
    assert not df.empty

    # Sentence identifiers
    sentences = sorted(set(df.sentence_id))  # NOTE: set() is weakly random

    # Store corrected sentences here
    corrected = defaultdict(dict)

    # Dynamics modelling
    dynamics_funcs = {"gengamma": gengamma.fit, "lognorm": lognorm.fit, "gamma": gamma.fit}
    dynamics_funcs_cut_off_quantile = {"gengamma": gengamma.ppf, "lognorm": lognorm.ppf, "gamma": gamma.ppf}
    estimated_dynamics_first_moment = {"gengamma": gengamma.mean, "lognorm": lognorm.mean, "gamma": gamma.mean}

    # Storage for critical values
    dynamics_replacement_statistics = {}

    # Loop over all sentences
    for sentence in sentences:
        per_sentence_statistics = []
        # Loop over all subjects which have typed this sentence
        for subject in df.loc[(df.sentence_id == sentence)].participant_id.unique():
            coordinates = (df.participant_id == subject) & (df.sentence_id == sentence)
            # Get all delta timestamps for this sentence, across all subjects (this is IKI)
            if what_to_correct == "iki":
                #  IKI = (key-down at time T) - (key-down at T-1)
                # corrected[sentence][subject] = df.loc[coordinates].timestamp.diff().values
                corrected[sentence][subject] = (
                    df.loc[coordinates].keydown[1:].values - df.loc[coordinates].keydown[:-1].values
                )
            elif what_to_correct == "hold_time":
                #  Hold time = (key-up) - (key-down)
                corrected[sentence][subject] = df.loc[coordinates].keyup.values - df.loc[coordinates].keydown.values
            elif what_to_correct == "pause":
                # Pause = IKI - (hold time)
                iki = df.loc[coordinates].keydown[1:].values - df.loc[coordinates].keydown[:-1].values
                hold_time = df.loc[coordinates].keyup.values - df.loc[coordinates].keydown.values
                pause = iki - hold_time[:-1]
                # Store for later
                corrected[sentence][subject] = np.array(pause)

            # Append to get statistics over all participants
            per_sentence_statistics.extend(list(corrected[sentence][subject]))

        # Move to numpy array for easier computation
        x = np.array(per_sentence_statistics)  # TODO: this is stupid fix this.
        # Remove all NANs and remove all NEGATIVE values (this removes the first NaN value too)
        x = x[~np.isnan(x)]
        # Have to do in two operations because of NaN presence
        x = x[x > 0.0]
        # Fit suitable density for modelling correct replacement value
        params_MLE = dynamics_funcs[correction_model](x)

        # Set cut off value
        cut_off_value = np.percentile(x, cut_off_percentile, interpolation="lower")

        assert cut_off_value > 0, "INFO:\n\t value: {} \n\t sentence ID: {}".format(cut_off_value, sentence)

        # Set replacement value
        replacement_value = estimated_dynamics_first_moment[correction_model](*params_MLE)
        # TODO: put this back
        # assert replacement_value > 0, "INFO:\n\t value: {} \n\t sentence ID: {}".format(replacement_value, sentence)

        # Store for replacement operation in next loop
        dynamics_replacement_statistics[sentence] = (cut_off_value, replacement_value)

    # Search all delta timestamps and replace which exeed cut_off_value
    for sentence in sentences:
        # Loop over all subjects which have typed this sentence
        for subject in df.loc[(df.sentence_id == sentence)].participant_id.unique():

            # Make temporary conversion to numpy array
            x = corrected[sentence][subject]  # [1:]  # (remove the first entry as it is a NaN)
            corrected[sentence][subject] = np.concatenate(  # Add back NaN to maintain index order
                (
                    # [np.nan],
                    # Two conditions are used here
                    np.where(
                        np.logical_or(x > dynamics_replacement_statistics[sentence][0], x < 0),
                        dynamics_replacement_statistics[sentence][1],
                        x,
                    ),
                )
            )

    return corrected, dynamics_replacement_statistics


def iki_pause_correction(
    df: pd.DataFrame,
    char_count_response_threshold: int = 40,
    cut_off_percentile: int = 99,
    correction_model: str = "gengamma",
) -> Tuple[dict, dict]:
    """
    Function is used to correct the IKI, to attend to anomalies like subjects stopping mid-typing
    to attend to other matters, thus causing faulty temporal dynamics.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe which contains the raw typed sentences (MJFF or MRC dataset)
    char_count_response_threshold : int, optional
        The minimum number of characters required for an entry to be considered a valid attempt
    cut_off_percentile : float, optional
        IKI values above this threshold are replace by the first moment of correction_model, by default 99
    correction_model : str, optional
        Generative model used to do adjusments, by default "gengamma"

    Returns
    -------
    Tuple[dict, list]
        Dictionary of sentences indexed by subject
    """

    assert set(["participant_id", "key", "timestamp", "sentence_id"]).issubset(df.columns)
    # Filter out responses where the number of characters per typed
    # response, is below a threshold value (40 by default)
    df = df.groupby("sentence_id").filter(lambda x: x["sentence_id"].count() > char_count_response_threshold)
    assert not df.empty

    # Sentence identifiers
    sentences = sorted(set(df.sentence_id))  # NOTE: set() is weakly random

    # Store corrected sentences here
    corrected_ikis = defaultdict(dict)

    # Response time modelling
    pause_funcs = {"gengamma": gengamma.fit, "lognorm": lognorm.fit, "gamma": gamma.fit}
    pause_funcs_cut_off_quantile = {"gengamma": gengamma.ppf, "lognorm": lognorm.ppf, "gamma": gamma.ppf}
    pause_first_moment = {"gengamma": gengamma.mean, "lognorm": lognorm.mean, "gamma": gamma.mean}

    # Storage for critical values
    pause_replacement_statistics = {}

    # Loop over all sentences
    for sentence in sentences:
        sentence_iki = []
        # Loop over all subjects which have typed this sentence
        for subject in df.loc[(df.sentence_id == sentence)].participant_id.unique():

            # Get all delta timestamps for this sentence, across all subjects (this is IKI)
            tmp = df.loc[(df.sentence_id == sentence) & (df.participant_id == subject)].timestamp.diff().tolist()
            # Store for later
            corrected_ikis[sentence][subject] = np.array(tmp)
            # Append to get statistics over all participants
            sentence_iki.extend(tmp)

        # Move to numpy array for easier computation
        x = np.array(sentence_iki)
        # Remove all NANs and remove all NEGATIVE values (this removes the first NaN value too)
        x = x[~np.isnan(x)]
        # Have to do in two operations because of NaN presence
        x = x[x > 0.0]
        # Fit suitable density for modelling correct replacement value
        params_MLE = pause_funcs[correction_model](x)

        # Set cut off value
        cut_off_value = np.percentile(x, cut_off_percentile, interpolation="lower")

        assert cut_off_value > 0, "INFO:\n\t value: {} \n\t sentence ID: {}".format(cut_off_value, sentence)

        # Set replacement value
        replacement_value = pause_first_moment[correction_model](*params_MLE)
        assert replacement_value > 0, "INFO:\n\t value: {} \n\t sentence ID: {}".format(replacement_value, sentence)

        # Store for replacement operation in next loop
        pause_replacement_statistics[sentence] = (cut_off_value, replacement_value)

    # Search all delta timestamps and replace which exeed cut_off_value
    for sentence in sentences:
        # Loop over all subjects which have typed this sentence
        for subject in df.loc[(df.sentence_id == sentence)].participant_id.unique():

            # Make temporary conversion to numpy array
            x = corrected_ikis[sentence][subject][1:]  # (remove the first entry as it is a NaN)
            corrected_ikis[sentence][subject] = np.concatenate(  # Add back NaN to maintain index order
                (
                    [np.nan],
                    # Two conditions are used here
                    np.where(
                        np.logical_or(x > pause_replacement_statistics[sentence][0], x < 0),
                        pause_replacement_statistics[sentence][1],
                        x,
                    ),
                )
            )

    return corrected_ikis, pause_replacement_statistics


def create_char_iki_extended_data(
    df: pd.DataFrame,
    char_count_response_threshold=40,
    time_redux_fact=10,
    backspace_char="backspace",
    invokation_type=1,
) -> Tuple[dict, list]:
    """
    This function does the following given e.g. typed list ['a','b','c'] with corresponding
    timestamp values [3,7,10] then it creates the following "long-format" lists:
    L = [NaN, 'bbbb', 'ccc'] and then merges L[1:] as so ['bbbbccc']. This is what we call
    in the paper a 'long format' sentence.

    Parameters
    ----------
    df : pd.DataFrame
        [description]
    char_count_response_threshold : int, optional
        [description], by default 40
    time_redux_fact : int, optional
        [description], by default 10

    Returns
    -------
    Tuple[dict, list]
        [description]
    """

    assert set(["participant_id", "key", "timestamp", "sentence_id"]).issubset(df.columns)
    # Filter out responses where the number of characters per typed
    # response, is below a threshold value (40 by default)
    df = df[df.groupby(["participant_id", "sentence_id"]).key.transform("count") > char_count_response_threshold]
    assert not df.empty

    # Get the unique number of subjects
    subjects = sorted(set(df.participant_id))  # NOTE: set() is weakly random

    # All sentences will be stored here, indexed by their type
    long_format_sentences = defaultdict(dict)
    long_format_locations = defaultdict(dict)

    # Corrected inter-key-intervals (i.e. timestamp difference / delta)
    corrected_inter_key_intervals, iki_replacement_stats = iki_pause_correction(
        df, char_count_response_threshold=char_count_response_threshold
    )

    # Loop over subjects
    for subject in subjects:
        # Not all subjects have typed all sentences hence we have to do it this way
        for sentence in df.loc[(df.participant_id == subject)].sentence_id.unique():

            # print("subject: {} -- sentence: {}".format(subj_idx, sent_idx))

            # Locate df segment to extract
            coordinates = (df.participant_id == subject) & (df.sentence_id == sentence)

            # "correct" the sentence by operating on user backspaces
            corrected_sentence, character_indices_to_delete = universal_backspace_implementer(
                df.loc[coordinates, "key"].tolist(), removal_character=backspace_char, invokation_type=invokation_type
            )

            # Need to respect the lower limit on the sentence length
            if len(corrected_sentence) < char_count_response_threshold:
                continue

            # Check consistency
            L = len(corrected_inter_key_intervals[sentence][subject])
            assert set(character_indices_to_delete).issubset(
                range(L)
            ), "Indices to remove: {} -- total length of timestamp vector: {}".format(character_indices_to_delete, L)

            # Drop indices that were removed above
            idx = np.array(character_indices_to_delete) + 1
            inter_key_intervals = np.delete(corrected_inter_key_intervals[sentence][subject], idx)

            # Sanity check
            assert len(inter_key_intervals) > char_count_response_threshold, "Subject: {} and sentence: {}".format(
                subject, sentence
            )

            # Make long-format version of each typed, corrected, sentence
            # Note that we remove the first character to make the calculation correct.
            long_format_sentences[subject][sentence], ikis = make_long_format_sentence(
                inter_key_intervals, corrected_sentence, time_redux_fact
            )
            # Sanity check
            assert (
                len(long_format_sentences[subject][sentence]) > char_count_response_threshold
            ), "Subject: {}; sentence: {}; length: {}\n Sentence: {}\n Chars: {}\nIKIs: {}".format(
                subject,
                sentence,
                len(long_format_sentences[subject][sentence]),
                long_format_sentences[subject][sentence],
                corrected_sentence,
                inter_key_intervals,
            )

            if "location" in df.columns:
                # Get the key location [only for MRC dataset]
                keyboard_locs = df.loc[coordinates, "location"].values
                keyboard_locs = np.delete(keyboard_locs, character_indices_to_delete)
                out = list(chain.from_iterable([[i] * int(n) for i, n in zip(keyboard_locs[:-1], ikis[1:])]))
                assert len(out) == len(long_format_sentences[subject][sentence])
                long_format_locations[subject][sentence] = convert(out)

    return long_format_sentences, long_format_locations


def convert(a):
    # Converting integer list to string list
    # and joining the list using join()
    return "".join(map(str, a))


def create_char_data(
    df: pd.DataFrame, char_count_response_threshold: int = 40, backspace_char="backspace", invokation_type: int = 1
) -> dict:
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

    # Filter out responses where the number of characters per typed
    # response, is below a threshold value (40 by default)
    df = df[df.groupby(["participant_id", "sentence_id"]).key.transform("count") > char_count_response_threshold]
    assert not df.empty

    if set(["participant_id", "key", "timestamp", "sentence_id"]).issubset(df.columns):
        mode = "MJFF_MODE"
    elif set(["participant_id", "key", "keyup", "keydown", "sentence_id"]).issubset(df.columns):
        mode = "MRC_MODE"
        # MRC keydown is equivalent to MJFF keydown
        df["timestamp"] = df.keydown
        corrected_key_hold_times, _ = typing_dynamics_editor(df, "hold_time", char_count_response_threshold)
        # corrected_key_pause_times, _ = typing_dynamics_editor(df, "pause", char_count_response_threshold)

    print("Using mode: {}".format(mode))

    # Get the unique number of subjects
    subjects = sorted(set(df.participant_id))  # NOTE: set() is weakly random

    corrected_inter_key_intervals, _ = typing_dynamics_editor(df, "iki", char_count_response_threshold)

    # All sentences will be stored here, indexed by their type
    character_only_sentences = defaultdict(dict)
    character_only_sentences_list = defaultdict(dict)
    locations = defaultdict(dict)
    numeric_iki = defaultdict(dict)
    hold_down_time = defaultdict(dict)
    pause_time = defaultdict(dict)

    # Loop over subjects
    for subject in subjects:
        # Not all subjects have typed all sentences hence we have to do it this way
        for sentence in df.loc[(df.participant_id == subject)].sentence_id.unique():
            # print(subject, sentence)
            # Locate df segment to extract
            coordinates = (df.participant_id == subject) & (df.sentence_id == sentence)

            # "correct" the sentence by operating on user backspaces
            corrected_sentence, character_indices_to_delete = universal_backspace_implementer(
                df.loc[coordinates, "key"].tolist(), removal_character=backspace_char, invokation_type=invokation_type
            )

            # Note that sentences are lower-cased here
            character_only_sentences[subject][sentence] = "".join(corrected_sentence).lower()
            character_only_sentences_list[subject][sentence] = [c.lower() for c in corrected_sentence]
            # Drop indices that were removed above
            numeric_iki[subject][sentence] = list(
                np.delete(corrected_inter_key_intervals[sentence][subject], character_indices_to_delete)
            )
            assert len(numeric_iki[subject][sentence]) == len(character_only_sentences[subject][sentence]) - 1, (
                len(numeric_iki[subject][sentence]),
                len(character_only_sentences[subject][sentence]) - 1,
            )

            if "location" in df.columns:
                # Get the key location [only for MRC dataset]
                keyboard_locs = df.loc[coordinates, "location"].values
                keyboard_locs = np.delete(keyboard_locs, character_indices_to_delete)

                assert len(keyboard_locs) == len(character_only_sentences[subject][sentence])
                locations[subject][sentence] = convert(keyboard_locs)

            if mode == "MRC_MODE":

                keypress_timings = np.delete(corrected_key_hold_times[sentence][subject], character_indices_to_delete)
                assert len(keypress_timings) == len(character_only_sentences[subject][sentence]), (
                    len(keypress_timings),
                    len(character_only_sentences[subject][sentence]),
                )
                hold_down_time[subject][sentence] = list(keypress_timings)

                # Get pause between key-presses
                pause = [i - j for i, j in zip(numeric_iki[subject][sentence], hold_down_time[subject][sentence][:-1])]
                assert len(pause) == len(character_only_sentences[subject][sentence]) - 1, (
                    len(pause),
                    len(character_only_sentences[subject][sentence]) - 1,
                )
                pause_time[subject][sentence] = pause

    if mode == "MRC_MODE":
        return (
            character_only_sentences,
            locations,
            numeric_iki,
            character_only_sentences_list,
            hold_down_time,
            pause_time,
        )

    return character_only_sentences, locations, numeric_iki, character_only_sentences_list


def flatten(my_list):
    # Function to flatten a list of lists
    return [item for sublist in my_list for item in sublist]


def make_long_format_sentence(ikis, characters, time_redux_fact=10) -> str:
    """
    Function creates a long-format sentence, where each character is repeated for a discrete
    number of steps, commensurate with how long that character was compressed for, when
    the sentence was being typed by the participants.

    By design, we obviously cannot repeat the last character for a number of steps.

    Parameters
    ----------
    ikis : pd.Series
        inter-key intervals in milliseconds
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
    assert len(ikis) == len(characters), "Lengths are: {} and {}".format(len(ikis), len(characters))
    assert any(ikis[1:] > 0)

    # Get discretised inter-key intervals
    ikis = ikis // time_redux_fact

    # Check if any iki values are < 1
    if any(ikis[1:] < 1.0):
        ikis[np.where(ikis[1:] < 1.0)[0] + 1] = 1.0

    # Errorful indicators are only repeated once
    indicator_indices = np.where(np.array(characters) == "ω")[0]
    if any(indicator_indices == len(characters) - 1):
        # Catches any indicators that appear at the end of a sentence
        indicator_indices = np.delete(indicator_indices, np.argwhere(indicator_indices == len(characters) - 1)) + 1
    else:
        indicator_indices += 1
    try:
        ikis[indicator_indices] = 1
    except Exception as e:
        print(ikis)
        print(characters)

    # Notice that we make each sentence lower-case
    return "".join([c * int(n) for c, n in zip(characters[:-1], ikis[1:])]).lower(), ikis


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
                        universal_backspace_implementer(typed_keys[sub_id][sent_id])
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


def universal_backspace_implementer(
    sentence: list, removal_character="backspace", invokation_type=1, verbose: bool = False, indicator_character="ω"
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
    indicator_character: str, optional
        Indicates where in the sentence a correction has been mad (matches MRC data)

    Returns
    -------
    Tuple[list, list]
        Corrected sentece with correction indicator included
    """

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

    # Coordinates which will be replaced by an error-implementation indicator
    indicator_indices = []

    # Find the indices of all the remaining backspace occurences
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
                indicator_indices.extend([group[0] - 1, group[0]])

            else:
                indicator_indices.extend(range_extend(group))

    elif invokation_type == 1:
        # Replace all characters indicated by 'backspace' _except_ the last character which is kept
        for group in backspace_groups:
            if len(group) == 1:

                # Replace single backspace with indicator character inline
                sentence[group[0]] = indicator_character

            else:
                # This _may_ introduce negative indices at the start of a sentence
                # these are filtered out further down

                # This keeps the error and adds the indicator character right after
                errorful_sequence_and_backspaces = range_extend(group)
                # This invokes the n-1 backspaces
                indicator_indices.extend(errorful_sequence_and_backspaces[1:-1])
    else:
        raise ValueError

    # Filter out negative indices which are non-sensical for deletion (arises when more backspaces than characters in beginning of sentence)
    indicator_indices = list(filter(lambda x: x >= 0, indicator_indices))

    # Filter out deletion indices which appear at the end of the sentence as part of a contiguous group of backspaces
    indicator_indices = list(filter(lambda x: x < len(original_sentence), indicator_indices))

    # Delete at indicator coordinates
    invoked_sentence = np.delete(sentence, indicator_indices).tolist()

    # Replace remaining backspace with indicator
    invoked_sentence = [indicator_character if x == removal_character else x for x in invoked_sentence]

    if verbose:
        print("Original sentence: {}\n".format(original_sentence))
        print("Edited sentence: {} \n -----".format(invoked_sentence))

    return invoked_sentence, indicator_indices


def create_dataframe_from_processed_data(
    my_dict: dict, df_meta: pd.DataFrame, iki_dict: dict = None, sentence_list_dict: dict = None
) -> pd.DataFrame:
    """
    Function creates a pandas DataFrame which will be used by the NLP model
    downstream.

    Parameters
    ----------
    my_dict : dict
        Dictionary containing all the preprocessed typed sentences, indexed by subject
    df_meta : Pandas DataFrame
        Contains the mete information on each patient (for the MRC this is the main dataframe)

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
                        int(df_meta.loc[df_meta.participant_id == participant_id, "diagnosis"].unique()),
                        str(sent_id),
                        str(my_dict[participant_id][sent_id]),
                        iki_dict[participant_id][sent_id] if iki_dict is not None else np.nan,
                        sentence_list_dict[participant_id][sent_id] if sentence_list_dict is not None else np.nan,
                    ]
                    for sent_id in my_dict[participant_id].keys()
                ]
            )
        )
    df = pd.concat(final_out, axis=0)
    df.columns = [
        "Participant_ID",
        "Diagnosis",
        "Sentence_ID",
        "Preprocessed_typed_sentence",
        "IKI_timings",
        "PPTS_list",
    ]

    # Final check for empty values
    df["Preprocessed_typed_sentence"].replace("", np.nan, inplace=True)
    # Remove all such rows
    df.dropna(subset=["Preprocessed_typed_sentence"], inplace=True)
    # Final check to ensure that the last column is a string
    df = df.astype({"Preprocessed_typed_sentence": str})

    return df


def create_dataframe_from_processed_data_mrc(
    my_dict: dict,
    #location_dict: dict,
    iki_dict: dict,
    sentence_list_dict: dict,
    hold_down_dict: dict,
    pause_dict: dict,
    df_meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Function creates a pandas DataFrame which will be used by the NLP model
    downstream.

    Parameters
    ----------
    my_dict : dict
        Dictionary containing all the preprocessed typed sentences, indexed by subject
    df_meta : Pandas DataFrame
        Contains the mete information on each patient (for the MRC this is the main dataframe)

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
                        int(df_meta.loc[df_meta.participant_id == participant_id, "diagnosis"].unique()),
                        str(sent_id),
                        str(my_dict[participant_id][sent_id]),
                        #location_dict[participant_id][sent_id],
                        iki_dict[participant_id][sent_id] if iki_dict is not None else np.nan,
                        sentence_list_dict[participant_id][sent_id] if sentence_list_dict is not None else np.nan,
                        hold_down_dict[participant_id][sent_id] if hold_down_dict is not None else np.nan,
                        pause_dict[participant_id][sent_id] if hold_down_dict is not None else np.nan,
                    ]
                    for sent_id in my_dict[participant_id].keys()
                ]
            )
        )
    df = pd.concat(final_out, axis=0)
    df.columns = [
        "Participant_ID",
        "Diagnosis",
        "Sentence_ID",
        "Preprocessed_typed_sentence",
        #"locations",
        "IKI_timings",
        "PPTS_list",
        "hold_time",
        "pause_time",
    ]

    # Final check for empty values
    df["Preprocessed_typed_sentence"].replace("", np.nan, inplace=True)
    # Remove all such rows
    df.dropna(subset=["Preprocessed_typed_sentence"], inplace=True)
    # Final check to ensure that the last column is a string
    df = df.astype({"Preprocessed_typed_sentence": str})

    return df


def remap_English_MJFF_participant_ids(df):
    replacement_ids = {}
    for the_str in set(df.Participant_ID):
        base = str("".join(map(str, [int(s) for s in the_str.split()[0] if s.isdigit()])))
        replacement_ids[the_str] = base
    return df.replace({"Participant_ID": replacement_ids})


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
    data_root = "../data/MJFF"  # My local path
    data_root = Path(data_root)

    # Load raw text and meta data
    if language == "english":
        df_meta = pd.read_csv(
            data_root / "raw" / "EnglishParticipantKey.csv",
            header=0,
            names=["participant_id", "ID", "attempt", "diagnosis"],
        )
        df = pd.read_csv(data_root / "raw" / "EnglishData-duplicateeventsremoved.csv")
        # Select which attempt we are interested in, only English has attempts
        df = select_attempt(df, df_meta, attempt=attempt)
    elif language == "spanish":
        # Load raw text and meta data
        df = pd.read_csv(data_root / "raw" / "SpanishData-Nov_28_2019.csv")
        df_meta = pd.read_csv(data_root / "raw" / "SpanishDiagnosisAndMedicationInfo.csv", header=0)
        # 'correct' Spanish characters
        df = create_proper_spanish_letters(df)
    else:
        raise ValueError

    # Get the tuple (sentence ID, reference sentence) as a dataframe
    reference_sentences = df.loc[:, ["sentence_id", "sentence_text"]].drop_duplicates().reset_index(drop=True)

    if include_time:
        # Creates long sequences with characters repeated for IKI number of steps
        out, _ = create_char_iki_extended_data(df)
        iki_timings = None
        character_only_sentences_list = None

    else:
        # This option _only_ includes the characters.
        out, _, iki_timings, character_only_sentences_list = create_char_data(df, invokation_type=invokation_type)

    # Final formatting of typing data
    df = create_dataframe_from_processed_data(out, df_meta, iki_timings, character_only_sentences_list).reset_index(
        drop=True
    )

    if language == "english":
        # Remap participant identifiers so that that e.g. 10a -> 10 and 10b -> 10.
        return remap_English_MJFF_participant_ids(df), reference_sentences

    # Return the empirical data and the reference sentences for downstream tasks
    return df, reference_sentences


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

    for subject in subjects:
        # Not all subjects have typed all sentences hence we have to do it this way
        for sentence in df.loc[(df.participant_id == subject)].sentence_id.unique():

            # Get typed sentence
            typed_characters = df.loc[(df.participant_id == subject) & (df.sentence_id == sentence)].key.values
            typed_chars_index = df.loc[(df.participant_id == subject) & (df.sentence_id == sentence)].index
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

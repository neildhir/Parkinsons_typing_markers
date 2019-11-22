# Basline calculation stuff

from collections import Counter, defaultdict
from statistics import mean, median
from pathlib import Path
import os.path
import pickle

import numpy as np
import pandas as pd
from nltk import edit_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.svm import SVC  # Support vector classifier

from haberrspd.preprocess import (
    backspace_corrector,
    create_MJFF_dataset,
    remap_English_MJFF_participant_ids,
    select_attempt,
    sentence_level_pause_correction,
)


def calculate_iki_and_ed_baseline(
    df: pd.DataFrame, df_meta: pd.DataFrame = None, drop_shift=False, attempt=1, invokation_type: int = -1
):

    if df_meta is not None:
        # MJFF
        print("MJFF")
        which_dataset = "mjff"
        backspace_char = "backspace"
        ref = reference_sentences(which_dataset)
        # Select which attempt we are considering
        df = select_attempt(df, df_meta, attempt=attempt)
        assert len(df.attempt.unique()) == 1
        data_root = Path("../data/MJFF/")
        full_path = data_root / "sentence_level_pause_correct_mjff.pkl"

    else:
        # MRC
        print("MRC")
        which_dataset = "mrc"
        # Get all keyups and drop them in place
        idxs_keyup = df.index[(df.type == "keyup")]
        # In-place dropping of these rows
        df.drop(df.index[idxs_keyup], inplace=True)
        # Reset index so that we can sort it properly in the next step
        df.reset_index(drop=True, inplace=True)
        # Make sure that we've dropped the keyups in the MRC dataframe
        assert "keyup" not in df.type

        # Remove shift characters or not
        if drop_shift:
            # Shift indices
            idxs_shift = df.index[(df.key == "β")]
            # In-place dropping of these rows
            df.drop(df.index[idxs_shift], inplace=True)
            df.reset_index(drop=True, inplace=True)
            print("\n Number of shift-rows dropped: %i" % len(idxs_shift))

        backspace_char = "α"
        ref = reference_sentences(which_dataset)
        data_root = Path("../data/MRC/")
        full_path = data_root / "sentence_level_pause_correct_mrc.pkl"

    # TODO: special functions to apply to deal with shift and backspace for MRC

    if os.path.exists(full_path):
        pkl_file = open(full_path, "rb")
        corrected_inter_key_intervals = pickle.load(pkl_file)
    else:
        print("\n Pickled file wasn't found.")
        # Corrected inter-key-intervals (i.e. timestamp difference / delta)
        corrected_inter_key_intervals = sentence_level_pause_correction(df)

    data = []
    # Loop over sentence IDs
    for sentence in corrected_inter_key_intervals.keys():
        # Loop over participant IDs
        for participant in corrected_inter_key_intervals[sentence]:

            if invokation_type == 1:
                # Find the correction to the IKI

                # Locate df segment to extract
                coordinates = (df.participant_id == participant) & (df.sentence_id == sentence)

                # Not all participants typed all sentences, this conditions check that
                if len(df[coordinates]) != 0:

                    # "correct" the sentence by operating on user backspaces
                    corrected_character_sentence, removed_chars_indx = backspace_corrector(
                        df.loc[coordinates, "key"].tolist(),
                        removal_character=backspace_char,
                        invokation_type=invokation_type,
                    )

                    L = len(corrected_inter_key_intervals[sentence][participant])
                    assert set(removed_chars_indx).issubset(
                        range(L)
                    ), "Indices to remove: {} -- total length of timestamp vector: {}".format(removed_chars_indx, L)
                    # Adjust actual inter-key-intervals
                    iki = corrected_inter_key_intervals[sentence][participant].drop(index=removed_chars_indx)
                    # Calculate edit distance
                    reference_sentence = select_reference_sentence(which_dataset, sentence, participant, ref)
                    ed = edit_distance("".join(corrected_character_sentence), reference_sentence)

            elif invokation_type == -1:
                # Uncorrected IKI extracted here
                iki = corrected_inter_key_intervals[sentence][participant][1:]
                # Calculate edit distance
                reference_sentence = select_reference_sentence(which_dataset, sentence, participant, ref)
                ed = edit_distance(
                    "".join(df[(df.participant_id == participant) & (df.sentence_id == sentence)].key),
                    reference_sentence,
                )
            else:
                # TODO: clean this up for other options
                raise ValueError

            # Append to list which we'll pass to a dataframe in subsequent cells
            if df_meta is not None:
                # MJFF
                diagnosis = int(df_meta.loc[(df_meta.participant_id == participant), "diagnosis"])
            else:
                # MRC
                diagnosis = int(df[df.participant_id == participant].diagnosis.unique())

            # Data for dataframe
            data.append((participant, sentence, diagnosis, iki.mean(), iki.var(), ed))

    col_names = ["Patient_ID", "Sentence_ID", "Diagnosis", "Mean_IKI", "Var_IKI", "Edit_Distance"]
    if df_meta is not None:
        results = remap_English_MJFF_participant_ids(pd.DataFrame(data, columns=col_names))
    else:
        results = pd.DataFrame(data, columns=col_names)

    results.dropna(inplace=True)
    results.reset_index(drop=True, inplace=True)
    assert not results.isnull().values.any()

    return results


def select_reference_sentence(which_dataset, sentence, participant, reference_df):
    if which_dataset == "mjff":
        # An MJFF sentence
        return reference_df[reference_df["MJFF_IDS"] == sentence]["REFERENCE_TEXT"].values[0]
    else:
        # An MRC sentence
        if participant < 1000:
            # A patient
            column = "MRC_PATIENT_DATA_IDS"
        else:
            # A control
            column = "MRC_CONTROL_DATA_IDS"
        return reference_df[reference_df[column] == sentence]["REFERENCE_TEXT"].values[0]


def calculate_all_baseline_ROC_curves(df, test_size=0.25):
    measures = ["Edit_Distance", "Mean_IKI", "Diagnosis"]
    assert set(measures).issubset(df.columns)
    features = ["Mean_IKI", ["Edit_Distance", "Mean_IKI"]]
    # Store all results in a dict which will be passed to plotting
    results = {"I": None, "II": None}
    assert len(results) == len(features)
    for i, j in zip(features, results.keys()):
        # List of features
        if isinstance(i, list):
            # 100 here is the upper limit on the total number of participants
            if df[i].shape[0] < 100:
                X = np.hstack([np.vstack(df[j]) for j in i])
            else:
                X = df[i].to_numpy()
        # Singular feature
        else:
            # 100 here is the upper limit on the total number of participants
            if df[i].shape[0] < 100:
                X = np.vstack(df[i])
            else:
                X = df[i].to_numpy().reshape(-1, 1)

        # targets
        y = df.Diagnosis.to_numpy()

        # We use 25% of our data for test [sklearn default]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Classify
        # clf = SVC(class_weight="balanced", gamma="auto", probability=True)
        clf = RandomForestClassifier(n_estimators=500, class_weight="balanced")
        clf.fit(X_train, y_train)
        y_probas = clf.predict_proba(X_test)

        # Store
        results[j] = (y_test, y_probas[:, 1])

    return results


def convert_df_to_subject_level(df: pd.DataFrame) -> pd.DataFrame:

    subjects = sorted(df.Patient_ID.unique())
    sentences = set(df.Sentence_ID)

    missing_sentences = defaultdict(list)

    # Check to see what entries are missing and where
    for sub in subjects:
        # Check if all sentences have been attempted
        tmp_set = set(df[df.Patient_ID == sub].Sentence_ID)
        if not sentences.issubset(tmp_set):
            # Find intersection and insert np.nan att missing places
            missing_sentences[sub].extend(sentences - tmp_set)

    # Edit distance
    df_subject = df.groupby(["Patient_ID", "Diagnosis"]).Edit_Distance.apply(list).reset_index()
    # Mean IKI
    tmp = df.groupby(["Patient_ID", "Diagnosis"]).Mean_IKI.apply(list).reset_index()
    assert np.array_equal(df_subject.Patient_ID.values, tmp.Patient_ID.values)
    assert np.array_equal(df_subject.Diagnosis.values, tmp.Diagnosis.values)
    df_subject["Mean_IKI"] = tmp.Mean_IKI

    # Append N mean and median values at missig locations
    for sub in missing_sentences.keys():
        N = len(missing_sentences[sub])
        loc = df_subject.Patient_ID == sub
        ed = int(median(df_subject[loc].Edit_Distance.tolist()[0]))
        iki = mean(df_subject[loc].Mean_IKI.tolist()[0])
        # Extend
        df_subject[loc].Edit_Distance.tolist()[0].extend(N * [ed])
        df_subject[loc].Mean_IKI.tolist()[0].extend(N * [iki])

    df_subject.reset_index(inplace=True, drop=True)
    return df_subject


def test_different_splits_for_classification(Z):

    X, y = Z
    r = RandomForestClassifier(n_estimators=256, class_weight="balanced")
    rs = ShuffleSplit(n_splits=100, test_size=0.25)

    AUC = []

    # Crossvalidate the scores on a number of different random splits of the data
    for train_idx, test_idx in rs.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        r.fit(X_train, y_train)
        y_probas = r.predict_proba(X_test)
        y_pred = r.predict(X_test)
        # Confusion mat
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        # Main calculations here
        fpr, tpr, _ = roc_curve(y_test, y_probas[:, 1], pos_label=1)
        # Calculate area under the ROC curve here
        auc = np.trapz(tpr, fpr)
        AUC.append(auc)

    nauc = np.array(AUC)

    return nauc.mean().round(3), nauc.std().round(3)


def get_X_and_y_from_df(df):

    measures = ["Edit_Distance", "Mean_IKI", "Diagnosis"]
    # assert set(measures).issubset(df.columns)
    features = ["Mean_IKI", ["Edit_Distance", "Mean_IKI"]]

    # Store all results in a dict which will be passed to plotting
    sets = {"I": None, "II": None}
    # assert len(sets) == len(features)
    for i, j in zip(features, sets.keys()):

        # List of features
        if isinstance(i, list):
            assert len(i) == 2
            X = []
            for k in range(df.shape[0]):
                X.append(df.loc[k, i[0]] + df.loc[k, i[1]])
            X = np.vstack(X)

        # Singular feature
        else:
            X = np.vstack(df[i])

        # targets
        y = df.Diagnosis.to_numpy()

        sets[j] = (X, y)

    return sets


def reference_sentences(which_dataset):
    assert which_dataset in ["mrc", "mjff"]

    if which_dataset == "mjff":
        fields = ["MJFF_IDS", "REFERENCE_TEXT"]
    else:
        fields = ["MRC_PATIENT_DATA_IDS", "MRC_CONTROL_DATA_IDS", "REFERENCE_TEXT"]

    return pd.read_csv("../aux/sentence_IDs.csv", usecols=fields)


def remap_sentence_ids_for_control_subjects_mrc(df):
    # PD subjects have ID numbers which are all less than 1000
    control_subjects = [i for i in df.participant_id.unique() if i < 1000]
    correct_sentence_id_mapping = dict(zip(list(range(1, 16)), [3, 1, 5, 2, 4, 11, 8, 9, 7, 6, 14, 13, 15, 10, 12]))
    A = df.loc[(df.participant_id.isin(control_subjects))]["sentence_id"].map(correct_sentence_id_mapping).tolist()
    df.loc[(df.participant_id.isin(control_subjects)), "sentence_id"] = A
    return df


def remap_sentence_ids_for_pd_subjects_mrc(df):
    pd_subjects = [i for i in df.participant_id.unique() if i > 1000]
    correct_sentence_id_mapping = dict(zip(list(range(1, 16)), [3, 1, 5, 2, 4, 11, 8, 9, 7, 6, 14, 13, 15, 10, 12]))
    A = df.loc[(df.participant_id.isin(pd_subjects))]["sentence_id"].map(correct_sentence_id_mapping).tolist()
    df.loc[(df.participant_id.isin(pd_subjects)), "sentence_id"] = A
    return df

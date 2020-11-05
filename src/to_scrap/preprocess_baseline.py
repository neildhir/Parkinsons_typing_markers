# Basline calculation stuff

import os.path
import pickle
import pprint
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np
import pandas as pd
from nltk import edit_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, train_test_split
from sklearn.svm import SVC  # Support vector classifier

from src.preprocess import (
    create_MJFF_dataset,
    iki_pause_correction,
    remap_English_MJFF_participant_ids,
    select_attempt,
    universal_backspace_implementer,
)


def calculate_iki_and_ed_baseline(
    df: pd.DataFrame,
    df_meta: pd.DataFrame = None,
    which_dataset="mjff_english",
    drop_shift=False,
    attempt=1,
    invokation_type: int = -1,
    verbose=False,
):

    assert which_dataset in ["mrc", "mjff_spanish", "mjff_english"]

    if which_dataset == "mjff_english" or which_dataset == "mjff_spanish":
        print("MJFF")
        assert df_meta is not None
        backspace_char = "backspace"
        ref = reference_sentences(which_dataset)
        if which_dataset == "mjff_english":
            # Select which attempt we are considering
            df = select_attempt(df, df_meta, attempt=attempt)
            assert len(df.attempt.unique()) == 1

        data_root = Path("../data/MJFF/")
        full_path = data_root / "sentence_level_pause_correct_mjff.pkl"

    else:
        print("MRC")
        # In-place dropping of keyup rows
        df.drop(df.index[(df.type == "keyup")], inplace=True)
        # Reset index so that we can sort it properly in the next step
        df.reset_index(drop=True, inplace=True)
        # Make sure that we've dropped the keyups in the MRC dataframe
        assert "keyup" not in df.type.tolist()

        # Remove shift characters or not
        if drop_shift:
            # Drop shift indices
            idxs_shift = df.index[(df.key == "β")]
            # In-place dropping of these rows
            df.drop(df.index[(df.key == "β")], inplace=True)
            df.reset_index(drop=True, inplace=True)
            if verbose:
                print("\n Number of shift-rows dropped: %i" % len(idxs_shift))

        backspace_char = "α"
        ref = reference_sentences(which_dataset)
        data_root = Path("../data/MRC/")
        full_path = data_root / "sentence_level_pause_correct_mrc.pkl"

    if os.path.exists(full_path):
        corrected_inter_key_intervals = pickle.load(open(full_path, "rb"))
    else:
        # Corrected inter-key-intervals (i.e. timestamp difference / delta)
        corrected_inter_key_intervals, iki_replacement_stats = iki_pause_correction(df)
        if verbose:
            print("IKI replacement stats:\n")
            pprint.pprint(iki_replacement_stats, width=1)

    assert len(df.sentence_id.unique()) == len(corrected_inter_key_intervals.keys())

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
                    corrected_character_sentence, removed_chars_indx = universal_backspace_implementer(
                        df.loc[coordinates, "key"].tolist(),
                        removal_character=backspace_char,
                        invokation_type=invokation_type,
                    )

                    L = len(corrected_inter_key_intervals[sentence][participant])
                    assert set(removed_chars_indx).issubset(
                        range(L)
                    ), "Indices to remove: {} -- total length of timestamp vector: {}".format(removed_chars_indx, L)
                    # Adjust actual inter-key-intervals
                    iki = np.delete(corrected_inter_key_intervals[sentence][participant], removed_chars_indx)[1:]
                    assert ~np.isnan(np.sum(iki))
                    # Calculate edit distance
                    reference_sentence = select_reference_sentence(which_dataset, sentence, participant, ref)
                    ed = edit_distance("".join(corrected_character_sentence), reference_sentence)

            elif invokation_type == -1:
                # Uncorrected IKI extracted here
                iki = corrected_inter_key_intervals[sentence][participant][1:]
                assert ~np.isnan(np.sum(iki))
                # Calculate edit distance
                reference_sentence = select_reference_sentence(which_dataset, sentence, participant, ref)
                ed = edit_distance(
                    "".join(df[(df.participant_id == participant) & (df.sentence_id == sentence)].key),
                    reference_sentence,
                )
            else:
                raise ValueError

            # Append to list which we'll pass to a dataframe in subsequent cells
            if which_dataset == "mjff_english" or which_dataset == "mjff_spanish":
                # MJFF
                diagnosis = int(df_meta.loc[(df_meta.participant_id == participant), "diagnosis"])
                # Data for dataframe
                data.append((participant, sentence, diagnosis, iki.mean(), iki.var(), ed))
            else:
                # MRC
                diagnosis = int(df[df.participant_id == participant].diagnosis.unique())
                medication = int(df[df.participant_id == participant].medication.unique())
                # Data for dataframe (note the inclusion of medication)
                data.append((participant, sentence, diagnosis, medication, iki.mean(), iki.var(), ed))

    if which_dataset == "mjff_english":
        col_names = ["Participant_ID", "Sentence_ID", "Diagnosis", "Mean_IKI", "Var_IKI", "Edit_Distance"]
        results = remap_English_MJFF_participant_ids(pd.DataFrame(data, columns=col_names))
    elif which_dataset == "mjff_spanish":
        col_names = ["Participant_ID", "Sentence_ID", "Diagnosis", "Mean_IKI", "Var_IKI", "Edit_Distance"]
        results = pd.DataFrame(data, columns=col_names)
    else:
        col_names = ["Participant_ID", "Sentence_ID", "Diagnosis", "Medication", "Mean_IKI", "Var_IKI", "Edit_Distance"]
        results = pd.DataFrame(data, columns=col_names)

    results.dropna(inplace=True)
    results.reset_index(drop=True, inplace=True)
    assert not results.isnull().values.any()
    assert len(df.participant_id.unique()) == len(results.Participant_ID.unique()), (
        len(df.participant_id.unique()),
        len(results.Participant_ID.unique()),
    )

    return results


def reference_sentences(which_dataset):
    assert which_dataset in ["mrc", "mjff_spanish", "mjff_english"]

    if which_dataset == "mjff_english":
        fields = ["MJFF_IDS", "REFERENCE_TEXT"]
        return pd.read_csv("../misc/sentence_IDs.csv", usecols=fields)
    elif which_dataset == "mjff_spanish":
        return pd.read_csv("../misc/spanish_sentence_IDs.csv", header=0)
    elif which_dataset == "mrc":
        fields = ["MRC_PATIENT_DATA_IDS", "MRC_CONTROL_DATA_IDS", "REFERENCE_TEXT"]
        return pd.read_csv("../misc/sentence_IDs.csv", usecols=fields)
    else:
        raise ValueError


def select_reference_sentence(which_dataset, sentence_id, participant_id, reference_df):
    if which_dataset == "mjff_english":
        # An MJFF sentence
        return reference_df[reference_df["MJFF_IDS"] == sentence_id]["REFERENCE_TEXT"].values[0]
    elif which_dataset == "mjff_spanish":
        # An MJFF sentence
        return reference_df[reference_df["sentence_id"] == sentence_id]["sentence_text"].values[0]
    elif which_dataset == "mrc":
        # An MRC sentence
        if participant_id < 1000:
            # A patient
            column = "MRC_PATIENT_DATA_IDS"
        else:
            # A control
            column = "MRC_CONTROL_DATA_IDS"
        return reference_df[reference_df[column] == sentence_id]["REFERENCE_TEXT"].values[0]


def calculate_all_baseline_ROC_curves(df, test_size=0.25, n_reruns=None):
    measures = ["Edit_Distance", "Mean_IKI", "Diagnosis"]
    assert set(measures).issubset(df.columns)
    features = ["Mean_IKI", ["Edit_Distance", "Mean_IKI"]]
    # Store all results in a dict which will be passed to plotting
    results = {"I": None, "II": None}
    assert len(results) == len(features)
    for i, j in zip(features, results.keys()):

        # List of features
        if isinstance(i, list):
            X = df[i].to_numpy()
        # Singular feature
        else:
            X = df[i].to_numpy().reshape(-1, 1)

        # Get targets
        y = df.Diagnosis.to_numpy()

        if n_reruns:
            # Rerun the classification n times
            scores = []
            sss = StratifiedShuffleSplit(n_splits=n_reruns, test_size=test_size, random_state=42)
            # clf = SVC(class_weight="balanced", gamma="auto", probability=True)
            clf = RandomForestClassifier(n_estimators=100, class_weight="balanced")
            for train_index, test_index in sss.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Classify
                clf.fit(X_train, y_train)
                y_probas = clf.predict_proba(X_test)
                scores.append((y_test, y_probas[:, 1]))

            # Store
            results[j] = scores

        else:

            # We use 25% of our data for test [sklearn default]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Classify
            # clf = SVC(class_weight="balanced", gamma="auto", probability=True)
            clf = RandomForestClassifier(n_estimators=100, class_weight="balanced")
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
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced")
    rs = ShuffleSplit(n_splits=100, test_size=0.25)

    AUC = []

    # Crossvalidate the scores on a number of different random splits of the data
    for train_idx, test_idx in rs.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X_train, y_train)
        y_probas = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)
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

def get_same_hand_vector(ss):
    kb = [‘q’,‘w’,‘e’,‘r’,‘t’,‘y’,‘u’,‘i’,‘o’,‘p’,‘a’,‘s’,‘d’,‘f’,‘g’,‘h’,‘j’,‘k’,‘l’,’\;’,’z',‘x’,‘c’,‘v’,‘b’,‘n’,‘m’,‘\,‘,’.‘,’\/’]
    hands = [‘l’,‘l’,‘l’,‘l’,‘l’,‘r’,‘r’,‘r’,‘r’,‘r’,‘l’,‘l’,‘l’,‘l’,‘l’,‘r’,‘r’,‘r’,‘r’,‘r’,‘l’,‘l’,‘l’,‘l’,‘l’,‘r’,‘r’,‘r’,‘r’,‘r’]
    hand_vec = []
    hand_vec.insert(0,“different”)
    for i in range(1,len(ss)):
        if hands[kb.index(ss[i])] == hands[kb.index(ss[i-1])]:
           hand_vec.insert(i,“same”)
        else:
           hand_vec.insert(i,“different”)
    return(hand_vec)
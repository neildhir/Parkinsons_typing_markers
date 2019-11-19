# Basline calculation stuff

from collections import Counter, defaultdict
from statistics import mean, median

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


def create_mjff_baselines(df, df_meta, attempt=1, invokation_type=1):

    # Select which attempt we are considering
    df = select_attempt(df, df_meta, attempt=attempt)

    # First calculate the IKI for each sentence
    # TODO: does this need to be moved after the edit-distance?
    df_iki = calculate_iki_and_ed_baseline(df, df_meta, invokation_type=invokation_type)

    # Second calculate the edit-distance
    df, reference_sentences = create_MJFF_dataset("english", False, attempt, invokation_type)
    df_edit = get_edit_distance_df(df, reference_sentences)

    # Combine all measures
    df = combine_iki_and_edit_distance(df_edit, df_iki)
    assert not df.isnull().values.any()

    return df


def calculate_iki_and_ed_baseline(df: pd.DataFrame, df_meta: pd.DataFrame = None, invokation_type: int = -1):

    if df_meta:
        # MJFF

        assert len(df.attempt.unique()) == 1
        backspace_char = "backspace"
    else:
        # MRC

        # Get all keyups and drop them in place
        idxs_keyup = df.index[(df.type == "keyup")]
        # In-place dropping of these rows
        df.drop(df.index[idxs_keyup], inplace=True)
        # Reset index so that we can sort it properly in the next step
        df.reset_index(drop=True, inplace=True)
        # Make sure that we've dropped the keyups in the MRC dataframe
        assert "keyup" not in df.type

        backspace_char = "α"
        ref = reference_sentences("mrc")
        # TODO: special functions to apply to deal with shift and backspace for MRC

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
                    ed = edit_distance(
                        "".join(corrected_character_sentence), ref[ref.sentence_id == int(sentence)].sentence_text[0]
                    )

            elif invokation_type == -1:
                # Uncorrected IKI extracted here
                iki = corrected_inter_key_intervals[sentence][participant][1:]

            else:
                # TODO: clean this up for other options
                raise ValueError

            # Append to list which we'll pass to a dataframe in subsequent cells
            if df_meta:
                # MJFF
                diagnosis = int(df_meta.loc[(df_meta.participant_id == participant), "diagnosis"])
            else:
                # MRC
                diagnosis = int(df[df.participant_id == participant].diagnosis.unique())

            # Data for dataframe
            data.append((participant, sentence, diagnosis, iki.mean(), iki.var(), ed))

    col_names = ["Patient_ID", "Sentence_ID", "Diagnosis", "Mean_IKI", "Var_IKI", "Edit_Distance"]
    if df_meta:
        df_iki = remap_English_MJFF_participant_ids(pd.DataFrame(data, columns=col_names))
    else:
        df_iki = pd.DataFrame(data, columns=col_names)

    df_iki.dropna(inplace=True)
    df_iki.reset_index(drop=True, inplace=True)
    assert not df_iki.isnull().values.any()

    return df_iki


def calculate_all_baseline_ROC_curves(df, test_size=0.25):
    measures = ["edit_distance", "Mean_IKI", "Diagnosis"]
    assert set(measures).issubset(df.columns)
    features = ["Mean_IKI", ["edit_distance", "Mean_IKI"]]
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


def combine_iki_and_edit_distance(df_edit, df_iki):
    df_edit["Mean_IKI"] = ""
    for idx in df_edit.Patient_ID.unique():
        for sent_idx in df_edit.loc[df_edit.Patient_ID == idx].Sentence_ID:
            a = float(df_iki[(df_iki.Patient_ID == idx) & (df_iki.Sentence_ID == int(sent_idx))]["Mean_IKI"])
            df_edit.loc[(df_edit.Patient_ID == idx) & (df_edit.Sentence_ID == sent_idx), "Mean_IKI"] = a

    return df_edit


def get_edit_distance_df(df, ref):
    df["edit_distance"] = ""
    for idx in df.Patient_ID.unique():
        for sent_idx in df.loc[df.Patient_ID == idx].Sentence_ID:
            a = df[(df.Patient_ID == idx) & (df.Sentence_ID == sent_idx)]["Preprocessed_typed_sentence"].values[0]
            b = ref[ref.sentence_id == int(sent_idx)]["sentence_text"].values[0]
            df.loc[(df.Patient_ID == idx) & (df.Sentence_ID == sent_idx), "edit_distance"] = edit_distance(a, b)
    return df


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
    df_subject = df.groupby(["Patient_ID", "Diagnosis"]).edit_distance.apply(list).reset_index()
    # Mean IKI
    tmp = df.groupby(["Patient_ID", "Diagnosis"]).Mean_IKI.apply(list).reset_index()
    assert np.array_equal(df_subject.Patient_ID.values, tmp.Patient_ID.values)
    assert np.array_equal(df_subject.Diagnosis.values, tmp.Diagnosis.values)
    df_subject["Mean_IKI"] = tmp.Mean_IKI

    # Append N mean and median values at missig locations
    for sub in missing_sentences.keys():
        N = len(missing_sentences[sub])
        loc = df_subject.Patient_ID == sub
        ed = int(median(df_subject[loc].edit_distance.tolist()[0]))
        iki = mean(df_subject[loc].Mean_IKI.tolist()[0])
        # Extend
        df_subject[loc].edit_distance.tolist()[0].extend(N * [ed])
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

    measures = ["edit_distance", "Mean_IKI", "Diagnosis"]
    assert set(measures).issubset(df.columns)
    features = ["Mean_IKI", ["edit_distance", "Mean_IKI"]]

    # Store all results in a dict which will be passed to plotting
    sets = {"I": None, "II": None}
    assert len(sets) == len(features)
    for i, j in zip(features, sets.keys()):
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

        sets[j] = (X, y)

    return sets


def reference_sentences(which_dataset):
    assert which_dataset in ["mrc", "mjff"]
    sentences = np.array(
        [
            "However, religions other than Islam, use a different pronunciation for Allah, although the spelling is the same",
            "He is buried in Egypt, Aswan at the Mausoleum of Aga Khan",
            "Books include Penguin Island, a satire on the Dreyfus affair",
            "The w-shaped glyph above the second consonant that it geminates, is in fact the beginning of a small letter",
            "The Franks alliance was important exactly because of their renown hostility towards the Byzantine",
            "As of 2004, nearly 50% of Americans who were enrolled in employer health insurance plans were covered for acupuncture treatments",
            "Lincoln's coffin would be encased in concrete several feet thick, and surrounded by a cage, and buried beneath a rock slab",
            "Generally considered a part of Central Asia, it is sometimes ascribed to a regional bloc in either the Middle East or South Asia",
            "Although early behavioural or cognitive intervention can help children gain self-care, social, and communication skills, their is no known cure",
            "The Korean men have not fared so well in Olympic competition but still produce good results",
            "The novel explores the relationship between Patroclus and Achilles from boyhood to the fateful events of the Iliad",
            "Split-finger aiming requires the archer to place the index finger above the nocked arrow, while the middle and ring fingers are both placed below",
            "They fought a thirty years war on the side of the Lamtuna Arabized Berbers who claimed Himyarite ancestry from the early Islamic invasions ",
            "However, there is no evidence that those tattoos were used as acupuncture points or if they were just decorative in nature",
            "Over three million cattle are residents of the province at one time or another, and Alberta beef has a healthy worldwide market",
        ],
        dtype=object,
    )

    identifer = {"mrc": np.arange(1, 16).reshape(-1, 1), "mjff": np.arange(55, 70).reshape(-1, 1)}
    data = np.hstack((identifer[which_dataset], sentences.reshape(-1, 1)))
    return pd.DataFrame(data=data, columns=["sentence_id", "sentence_text"])


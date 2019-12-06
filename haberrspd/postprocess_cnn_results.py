import glob
from os.path import isdir, exists
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy import interp
from sklearn.metrics import auc as AUC
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from talos import Predict
from talos.utils.load_model import load_model
from tqdm import tqdm

from haberrspd.charCNN.data_utilities import create_training_data_keras


class PostprocessTalos:
    """
    Class to process models optimised by Talos.
    """

    def __init__(
        self, dataset: str = "mjff", which_information: str = "char", language: str = "english", attempt: int = 1
    ):
        assert attempt in [1, 2]
        assert language in ["english", "spanish"]
        assert which_information in ["char", "char_time", "char_time_space"]
        assert dataset in ["mjff", "mrc"]
        assert isdir("../results/MJFF")
        assert isdir("../results/MRC")
        if language == "english":
            csv_filename = language.capitalize() + "Data-preprocessed_attempt_{}".format(attempt) + ".csv"
        else:
            csv_filename = language.capitalize() + "Data-preprocessed.csv"

        self.which_information = which_information
        self.csv_filename = csv_filename
        self.data_root = Path("../data") / dataset.upper() / "preproc"
        self.results_root = Path("../results") / dataset.upper() / which_information
        # create zip paths
        if language == "english":
            path_to_zip = glob.glob(
                str(self.results_root) + "/" + "{}*".format(language) + "attempt_{}_*".format(attempt) + ".zip"
            )
            assert len(path_to_zip) == 1
            self.path_to_zip = path_to_zip[0]
            assert exists(self.path_to_zip)
        elif language == "spanish":
            path_to_zip = glob.glob(str(self.results_root) + "/" + "{}*".format(language) + ".zip")
            assert len(path_to_zip) == 1
            self.path_to_zip = path_to_zip[0]
            assert exists(self.path_to_zip)
        else:
            raise ValueError
        self.extract_to = self.path_to_zip.replace(".zip", "")
        self.package_name = self.extract_to.split("/")[-1]
        self.file_prefix = self.extract_to + "/" + self.package_name

        print("Load model from: {}".format(self.path_to_zip))
        print("Load test data from: {}".format(dataset + "/" + which_information + "/" + csv_filename))

    def load_trained_model(self):
        """
        This is a re-write of the native Talos function: https://github.com/autonomio/talos/blob/master/talos/commands/restore.py which does not work with 3D data (such as ours).
        """

        # extract the zip
        # unpack_archive(self.path_to_zip, self.extract_to)
        z = ZipFile(self.path_to_zip, mode="r")
        # Check if folder is already there, otherwise create
        if not isdir(self.extract_to):
            z.extractall(self.extract_to)

        # add params dictionary
        self.params = np.load(self.file_prefix + "_params.npy", allow_pickle=True).item()

        # add experiment details
        self.details = pd.read_csv(self.file_prefix + "_details.txt", header=None)

        # add model
        self.model = load_model(self.file_prefix + "_model")

        # add results
        self.results = pd.read_csv(self.file_prefix + "_results.csv")
        self.results.drop("Unnamed: 0", axis=1, inplace=True)

        # clean up
        del self.extract_to, self.file_prefix
        del self.package_name, self.path_to_zip

    def load_corresponding_dataset(self):
        # Get processed data, ready to insert into the trained model
        return create_training_data_keras(
            self.data_root, self.which_information, self.csv_filename, for_plotting_results=True
        )

    def create_crossvalidated_test_datasets(self, splits=10):
        # Note that these are received as proper arrays i.e. y is not a list.
        X, y = self.load_corresponding_dataset()
        # SSS
        sss = StratifiedShuffleSplit(n_splits=splits, test_size=0.25, random_state=0)
        targets = []
        X_tests = []
        for train_index, test_index in sss.split(X, y):
            X_tests.append(X[test_index])
            targets.append(y[test_index])
        assert len(X_tests) == len(targets) == splits
        return X_tests, targets

    def calculate_all_ROC_curves(self):
        self.load_trained_model()  # Get trained model
        X_tests, targets = self.create_crossvalidated_test_datasets()
        rocs = []  # Store all ROC 'curves' here
        for X_test, y_test in zip(X_tests, targets):
            labels_and_label_probs = np.zeros((len(X_test), 2))
            for i, (y, x) in tqdm(enumerate(zip(y_test, X_test))):
                # Note that keras takes a 3D array and not the standard 2D, hence extra axis
                labels_and_label_probs[i, :] = [y, float(self.model.predict(x[np.newaxis, :, :]))]
            rocs.append(labels_and_label_probs)

        return rocs

    def calculate_mean_and_variance_of_ROC_curves(self):

        # Get all samples
        data = self.calculate_all_ROC_curves()
        assert all(x.shape[0] == data[0].shape[0] for x in data)

        tprs = []
        aucs = []

        # False positive rate
        mean_fpr = np.linspace(0, 1, len(data[0]))
        # Get all results
        for out in data:
            # out[0] == y_true, out[1] == y_score
            fpr, tpr, _ = roc_curve(out[:, 0], out[:, 1], pos_label=1)
            # Calculate area under the ROC curve here
            roc_auc = AUC(tpr, fpr)
            aucs.append(roc_auc)
            tprs.append(interp(mean_fpr, fpr, tpr))
        # Mean ROC curve
        mean_tpr = np.vstack(tprs).mean(axis=0)  # np.mean(tprs, axis=1)
        assert len(mean_tpr) == len(mean_fpr)
        mean_tpr[-1] = 1.0
        mean_auc = AUC(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        print(mean_auc.round(3), std_auc.round(3))
        return mean_fpr, mean_tpr

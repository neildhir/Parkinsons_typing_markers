import glob
import pickle
from collections import defaultdict
from os.path import exists, isdir
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy import interp
from sklearn.metrics import auc as AUC
from sklearn.metrics import roc_curve
from talos.utils.load_model import load_model
from tqdm import tqdm

from haberrspd.plotting import plot_superimposed_roc_curves


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
        self.X_test_file = "X_test_" + self.file_prefix[-19:] + ".pkl"
        self.y_test_file = "y_test_" + self.file_prefix[-19:] + ".pkl"

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

    def load_test_dataset(self):
        X_test = pickle.load(open(self.results_root / self.X_test_file, "rb"))
        y_test = pickle.load(open(self.results_root / self.y_test_file, "rb"))
        return X_test, y_test

    def get_fpr_and_tpr(self):
        self.load_trained_model()  # Get trained model
        X_test, y_test = self.load_test_dataset()
        self.true_labels_and_label_probs = np.zeros((len(X_test), 2))
        for i, (y, x) in tqdm(enumerate(zip(y_test, X_test))):
            # Note that keras takes a 3D array and not the standard 2D, hence extra axis
            self.true_labels_and_label_probs[i, :] = [y, float(self.model.predict(x[np.newaxis, :, :]))]
        fpr, tpr, _ = roc_curve(y_test, self.true_labels_and_label_probs[:, 1], pos_label=1)
        return fpr, tpr


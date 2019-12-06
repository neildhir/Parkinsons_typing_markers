from talos.utils.load_model import load_model
from talos import Predict
import numpy as np
import pandas as pd
from pathlib import Path
from haberrspd.charCNN.data_utilities import create_training_data_keras
from sklearn.model_selection import StratifiedShuffleSplit
from zipfile import ZipFile
import glob
from tqdm import tqdm


class PostprocessTalos:
    """
    Class to process models optimised by Talos.
    """

    def __init__(
        self,
        dataset: str = "mjff",
        which_information: str = "char",
        language: str = "english",
        attempt: int = 1,
        csv_filename="EnglishData-preprocessed_attempt_1.csv",
    ):
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
        elif language == "spanish":
            path_to_zip = glob.glob(str(self.results_root) + "/" + "{}*".format(language) + ".zip")
            assert len(path_to_zip) == 1
            self.path_to_zip = path_to_zip[0]
        else:
            raise ValueError
        self.extract_to = self.path_to_zip.replace(".zip", "")
        self.package_name = self.extract_to.split("/")[-1]
        self.file_prefix = self.extract_to + "/" + self.package_name

    def load_trained_model(self):
        """
        This is a re-write of the native Talos function: https://github.com/autonomio/talos/blob/master/talos/commands/restore.py which does not work with 3D data (such as ours).


        """

        # extract the zip
        # unpack_archive(self.path_to_zip, self.extract_to)
        z = ZipFile(self.path_to_zip, mode="r")
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

import glob
import pickle
from collections import defaultdict
from os.path import exists, isdir
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy import interp
from keras.callbacks import EarlyStopping
from sklearn.metrics import auc as AUC
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from talos.utils.load_model import load_model

from tensorflow.keras import backend as K
from tqdm import tqdm

from src.charCNN.data_utilities import create_training_data_keras
from src.plotting import plot_superimposed_roc_curves


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
        assert isdir("../results/" + dataset.upper())
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

        # This is the best saved model by default
        self.model = load_model(self.file_prefix + "_model")

        # add results
        self.results = pd.read_csv(self.file_prefix + "_results.csv")
        self.results.drop("Unnamed: 0", axis=1, inplace=True)

    def load_test_dataset(self):
        self.X_test = pickle.load(open(self.results_root / self.X_test_file, "rb"))
        self.y_test = pickle.load(open(self.results_root / self.y_test_file, "rb")).reshape(-1, 1)

    def get_fpr_and_tpr(self):
        self.load_trained_model()  # Get trained model
        self.load_test_dataset()
        fpr, tpr, _ = roc_curve(self.y_test, self.model.predict(self.X_test), pos_label=1)
        return fpr, tpr

    def run_n_fold_cross_validation(self, folds=10):

        self.load_trained_model()

        X, y = create_training_data_keras(
            self.data_root, self.which_information, self.csv_filename, for_plotting_results=True
        )
        fold_rates = []
        for i in tqdm(range(folds)):
            # Stratefied split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
            # Normal split
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=(1 / 9), random_state=1)

            best_params = self.results.sort_values(by=["val_acc"], ascending=False).iloc[0, :].to_dict()
            if "Nadam" in best_params["optimizer"]:
                best_params["optimizer"] = "Nadam"
            else:
                best_params["optimizer"] = "Adam"
            # Compile
            self.model.compile(loss=best_params["loss.1"], optimizer=best_params["optimizer"], metrics=["accuracy"])
            # Run and reset weights each time.
            weights = self.model.get_weights()
            weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
            self.model.set_weights(weights)
            self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                verbose=0,
                class_weight={0: best_params["control_class_weight"], 1: best_params["pd_class_weight"]},
                callbacks=[EarlyStopping(patience=20, min_delta=0.0001)],
                batch_size=best_params["batch_size"],
                epochs=500,  # best_params["epochs"],
            )

            # Predict
            fpr, tpr, _ = roc_curve(y_test, self.model.predict(X_test), pos_label=1)

            # Store
            fold_rates.append((fpr, tpr))

        return fold_rates


def mean_std_auc(data):
    for i, item in enumerate(data.keys()):
        tprs = []
        aucs = []
        # False positive rate
        mean_fpr = np.linspace(0, 1, len(data[item][0][0]))
        # Get all results
        for out in data[item]:
            fpr, tpr = out
            # Calculate area under the ROC curve here
            roc_auc = AUC(fpr, tpr)
            aucs.append(roc_auc)
            tprs.append(interp(mean_fpr, fpr, tpr))
        # Mean ROC curve
        mean_tpr = np.mean(tprs, axis=0)
        assert len(mean_tpr) == len(mean_fpr)
        mean_tpr[-1] = 1.0
        mean_auc = AUC(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        print("%s: %0.2f $\pm$ %0.2f" % (item, mean_auc, std_auc))

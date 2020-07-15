import datetime
from operator import itemgetter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import seaborn as sns

# from flair.data import Sentence
# from flair.models import TextClassifier
from pandas import DataFrame, read_csv
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot
from scipy import interp
from sklearn.manifold import TSNE
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize


def split(word):
    return [char.lower() for char in word]


def get_sentence_stats(df, target_sentences, sentence_id, diagnosis=0, chars_to_consider=10):

    # load target sentences first
    # assert sentence_id in range(0,target_sentences.shape[0])
    ref_sent = split(
        list(target_sentences[sentence_id - 1, :])[0]
    )  # Note that the reference sentences are zero-indexed
    print("This is the target sentence: {}".format(target_sentences[sentence_id - 1, :]))
    print("Chars to consider: {}".format(ref_sent[:chars_to_consider]))
    ref_chars = ref_sent[:chars_to_consider]
    n = len(ref_chars)
    idxs = range(n)

    iki_stats = {k: [] for k in idxs}
    hold_time_stats = {k: [] for k in idxs}
    pause_time_stats = {k: [] for k in idxs}

    # Get the unique number of subjects
    subjects = sorted(set(df.Participant_ID))  # NOTE: set() is weakly rando
    # Loop over subjects
    for subject in subjects:
        # Not all subjects have typed all sentences hence we have to do it this way
        if (
            str(sentence_id)
            in df.loc[(df.Participant_ID == subject) & (df.Diagnosis == diagnosis)].Sentence_ID.unique()
        ):
            # Sentence has been typed so we collect the stats

            # Locate df segment to extract
            coordinates = (df.Participant_ID == subject) & (df.Sentence_ID == str(sentence_id))

            # To make more sense of the statistics we only collect it _if_ the subject has typed same character
            # the reference sentence.
            typed_sent = df.loc[coordinates, "Preprocessed_typed_sentence"].tolist()[0]
            iki = df.loc[coordinates, "IKI_timings"].tolist()[0]
            hold_time = df.loc[coordinates, "hold_time"].tolist()[0]
            pause_time = df.loc[coordinates, "pause_time"].tolist()[0]

            # Find char stats
            for i in idxs:
                # print(subject,sentence_id,typed_sent[c],ref_sent[c])
                if typed_sent[i] == ref_chars[i]:
                    iki_stats[i].append(iki[i])
                    hold_time_stats[i].append(hold_time[i])
                    pause_time_stats[i].append(pause_time[i])

    return iki_stats, hold_time_stats, pause_time_stats, ref_chars


def create_dataframe_for_plotting(dict_times, ref_chars, assigned_diagnosis):
    times = []
    chars = []
    diagnosis = []
    idxs = range(len(ref_chars))
    for i, c in enumerate(idxs):
        n = len(dict_times[i])
        chars.extend(n * [c])
        times.extend(dict_times[i])
        diagnosis.extend(n * [assigned_diagnosis])

    return DataFrame(data=list(zip(chars, times, diagnosis)), columns=["key", "time", "Class"])


def create_binary_dataframe(control_times, pd_times, ref_chars):
    A = create_dataframe_for_plotting(control_times, ref_chars, assigned_diagnosis="Controls")
    B = create_dataframe_for_plotting(pd_times, ref_chars, assigned_diagnosis="PwPD")
    return A.append(B)


def get_plotting_data(df, target_sentences, sent_ID, char_count):

    iki_0, hold_time_0, pause_time_0, ref_chars = get_sentence_stats(
        df, target_sentences=target_sentences, sentence_id=sent_ID, diagnosis=0, chars_to_consider=char_count
    )
    iki_1, hold_time_1, pause_time_1, ref_chars = get_sentence_stats(
        df, target_sentences=target_sentences, sentence_id=sent_ID, diagnosis=1, chars_to_consider=char_count
    )

    # Make dataframe
    A = create_binary_dataframe(hold_time_0, hold_time_1, ref_chars)
    B = create_binary_dataframe(iki_0, iki_1, ref_chars)
    C = create_binary_dataframe(pause_time_0, pause_time_1, ref_chars)
    A["type"] = "Hold-down"
    B["type"] = "Inter-key interval"
    C["type"] = "Pause"
    tmp = A.append(B)  # Replace with concat
    D = tmp.append(C)

    return D, ref_chars


def time_plot(df, ref_chars, sent_ID, y_min, y_max, save_me=False):
    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("ticks")
    g = sns.catplot(
        x="key",
        y="time",
        hue="Class",
        data=df,
        col="type",
        capsize=0.33,
        palette="colorblind",
        scale=0.9,
        height=5,
        legend_out=False,
        aspect=0.9,
        kind="point",
        lw=6,
        errwidth=2.0,
        ci="sd",
    )

    g.map(plt.axhline, y=0, lw=1.5, ls="--", c="0.75", zorder=0)
    g.set(ylim=(y_min, y_max))
    g.fig.get_axes()[0].legend(loc="upper center", handletextpad=0.2, ncol=1)
    g.set(xticklabels=ref_chars)
    g.set_titles("{col_name}")
    g.set_xlabels("")
    g.set_ylabels("Duration $[10^{-2} \ s]$")

    if save_me:
        save_to = "../figures/time_plots/time_plot_sentence_ID_" + str(sent_ID) + ".pdf"
        plt.savefig(save_to, bbox_inches="tight")
        plt.close()

    plt.show()


def plot_times(df, target_sentences, sentence_stats, save_me=False):
    """Function which plots the time-dynamics plot for all sentences.

    Parameters
    ----------
    df : pandas dataframe
        Preprocessed MRC data.
    target_sentences : np.array
        Array containing all the target sentences used for copy-typing.
    sentence_stats : dict
        Contains all the plotting options for each sentence.
    save_me : bool, optional
        Call to save the plots or not.

    Raises
    ------
    NotImplementedError
        Single plot option not yet implemented.
    """
    if isinstance(sentence_stats, dict):
        for sent_ID in sentence_stats.keys():
            print("\nSentence: {}\n".format(sent_ID))
            n_chars, y_min, y_max = sentence_stats[sent_ID]
            df_plot, ref_chars = get_plotting_data(df, target_sentences, sent_ID, n_chars)
            time_plot(df_plot, ref_chars, sent_ID, y_min, y_max, save_me=save_me)
    else:
        raise NotImplementedError


# Universal update for fonts: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
nice_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}


def plot_superimposed_roc_curves(data: dict, filename=None, with_confidence_bounds=False) -> None:
    """
    Note here that data comes as data[key] = (fpr,tpr).
    """

    mpl.rcParams.update(nice_fonts)
    if filename:
        assert isinstance(filename, str)

    # Set styles for paper
    sns.set_context("paper")
    mpl.rcParams.update(nice_fonts)

    lw = 2
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    # palette = sns.color_palette(palette="colorblind", n_colors=len(data))
    palette = sns.color_palette(n_colors=len(data))
    styles = ["-", "--", "-."]
    lws = [2, 3, 4]
    alphas = [1.0, 0.80, 0.6]
    # Plot ROC curves
    if with_confidence_bounds is False:
        for i, item in enumerate(data.keys()):
            fpr, tpr = data[item]
            # Calculate area under the ROC curve here
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr, tpr, color=palette[i], lw=lw, linestyle="-", alpha=alphas[i], label="%s: %0.2f" % (item, roc_auc)
            )
        ax.plot([0, 1], [0, 1], color="gray", lw=lw, linestyle="--", alpha=0.5, label="Chance: 0.50")  # Chance
    else:
        # ROC curves with confidence bounds
        for i, item in enumerate(data.keys()):
            # Each key is a information entity e.g. C or C+T
            tprs = []
            aucs = []
            # False positive rate
            mean_fpr = np.linspace(0, 1, len(data[item][0][0]))
            # Get all results
            for out in data[item]:
                fpr, tpr = out
                # Calculate area under the ROC curve here
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                tprs.append(interp(mean_fpr, fpr, tpr))
            # Mean ROC curve
            mean_tpr = np.mean(tprs, axis=0)
            assert len(mean_tpr) == len(mean_fpr)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            # Plot mean curve
            ax.plot(
                mean_fpr,
                mean_tpr,
                color=palette[i],
                lw=lw,
                alpha=0.7,
                label="%s: %0.2f $\pm$ %0.2f" % (item, mean_auc, std_auc),
            )
            # Fill inbetween
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + 2 * std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - 2 * std_tpr, 0)
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=palette[i], alpha=0.15)

        ax.plot([0, 1], [0, 1], color="gray", lw=lw, linestyle="--", alpha=0.5)  # Chance

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    # Legend
    ax.legend(loc="lower right", ncol=1, framealpha=1, fancybox=False, borderpad=0.5)
    # Grid
    ax.grid(True, alpha=0.2)
    # If given then save
    if filename:
        # Set reference time for save
        now = datetime.datetime.now()
        fig.savefig(
            "../figures/cnn_roc_curves-" + filename + "-" + now.strftime("%Y-%m-%d-%H:%M") + ".pdf", bbox_inches="tight"
        )
    else:
        plt.show()


def plot_superimposed_roc_curves_with_confidence_bounds(data: dict, filename=None) -> None:

    mpl.rcParams.update(nice_fonts)
    if filename:
        assert isinstance(filename, str)

    if ~isinstance(data, dict):
        # We've been passed a list of dicts
        assert all([isinstance(x, dict) for x in data])
        # Combine them
        if len(data) == 2:
            assert set(["I", "II"]).issubset(data[0].keys()), data[0].keys()
            assert set(["I", "II"]).issubset(data[1].keys()), data[1].keys()
            data[1]["III"] = data[1].pop("I")
            data[1]["IV"] = data[1].pop("II")
            data = {**data[0], **data[1]}
        else:
            raise ValueError

    # Set styles for paper
    sns.set_context("paper")
    mpl.rcParams.update(nice_fonts)

    lw = 2
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    palette = sns.color_palette(n_colors=len(data))

    if isinstance(data["I"], tuple):
        # Simple ROC curves
        for i, item in enumerate(data.keys()):
            y_true, y_scores = data[item]
            # Main calculations here
            fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
            # Calculate area under the ROC curve here
            roc_auc = auc(tpr, fpr)
            ax.plot(fpr, tpr, color=palette[i], lw=lw, alpha=0.8, label="%s: AUC = %0.2f" % (item, roc_auc))
    elif isinstance(data["I"], list):
        # ROC curves with confidence bounds
        for i, item in enumerate(data.keys()):
            tprs = []
            aucs = []
            # False positive rate
            mean_fpr = np.linspace(0, 1, len(data["I"][0][0]))
            # Get all results
            for out in data[item]:
                # out[1] == y_true, out[1] == y_score
                fpr, tpr, _ = roc_curve(out[0], out[1], pos_label=1)
                # Calculate area under the ROC curve here
                roc_auc = auc(tpr, fpr)
                aucs.append(roc_auc)
                tprs.append(interp(mean_fpr, fpr, tpr))
            # Mean ROC curve
            mean_tpr = np.mean(tprs, axis=0)
            assert len(mean_tpr) == len(mean_fpr)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            # Plot mean curve
            ax.plot(
                mean_fpr,
                mean_tpr,
                color=palette[i],
                lw=lw,
                alpha=0.7,
                label="%s: %0.2f $\pm$ %0.2f" % (item, mean_auc, std_auc),
            )
            # Fill inbetween
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + 2 * std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - 2 * std_tpr, 0)
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=palette[i], alpha=0.15)

    else:
        raise ValueError

    ax.plot([0, 1], [0, 1], color="gray", lw=lw, linestyle="--", alpha=0.5)  # Chance
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    # Legend
    ax.legend(loc="lower right", ncol=1, framealpha=1, fancybox=False, borderpad=0.5)
    # Grid
    ax.grid(True, alpha=0.2)

    if filename:
        # Set reference time for save
        now = datetime.datetime.now()
        fig.savefig(
            "../figures/baseline_roc_curves-" + filename + "-" + now.strftime("%Y-%m-%d-%H:%M") + ".pdf",
            bbox_inches="tight",
        )
    else:
        plt.show()


def plot_roc_curve_simple(y_true, y_scores, filename=None):
    """
    Plot receiver-operating curve.

    Parameters
    ----------
    y_true : array-like
        List or array containing the TRUE labels
    y_scores : array-like
        List or array containing the probabilities of the predicted labels
    filename : str
        Descriptive filename
    """

    # Set styles for paper
    sns.set_context("paper")
    sns.set_style("ticks")

    # Main calculations here
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    # Calculate area under the ROC curve here
    auc = np.trapz(tpr, fpr)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    lw = 2
    ax.plot(fpr, tpr, color="red", lw=lw, label="ROC curve (area = %0.2f)" % auc)
    ax.plot([0, 1], [0, 1], color="blue", lw=lw, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    # ax.set_title("Receiver Operating Curve")
    ax.legend(loc="lower right")

    if filename:
        # Set reference time for save
        now = datetime.datetime.now()
        fig.savefig(
            "../figures/roc_curve-" + filename + "-" + now.strftime("%Y-%m-%d-%H:%M") + ".pdf", bbox_inches="tight"
        )
    else:
        plt.show()


def plot_roc_curve(labels, label_probs, save_me=False):
    """
    Plot receiver-operating curve.

    Parameters
    ----------
    labels : array-like
        List or array containing the true labels
    label_probs : array-like
        List or array containing the probabilities of the predicted labels
    """

    assert len(set(labels)) == 2

    # Convert the labels to integer binary format
    a = label_binarize(labels, classes=list(set(labels)))
    b = np.logical_not(a).astype(int)
    y_test = np.hstack((a, b))
    n_classes = 2

    # Learn to predict each class against the other
    label_probs = np.reshape(label_probs, (-1, 1))
    y_score = np.hstack((label_probs, 1 - label_probs))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    lw = 2
    ax.plot(fpr[1], tpr[1], color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc[1])
    ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Curve")
    ax.legend(loc="lower right")

    if save_me:
        # Set reference time for save
        now = datetime.datetime.now()
        fig.savefig("../figures/roc_curve-" + now.strftime("%Y-%m-%d-%H:%M") + ".pdf", bbox_inches="tight")

    plt.show()


def sentence_embeddings_scatter_3d_plot(
    df: DataFrame, X: list, which_sentence: int = None, save_me: bool = False, filename: str = None
) -> None:
    """
    This is the main plotting function, for looking at embeddings. There are two ways
    in which this can be done; one) by looking at individual sentence embeddings in which
    case the label (PD/non-PD) will also be superimposed as a colour to see how useful the
    embedding actually is for separating typing behaviour. The second option is to merely plot
    all the embeddings, for all sentences, which will end up in a 3D point cloud, with one blob
    for each sentence ID.

    Parameters
    ----------
    df : DataFrame
        The preprocessed sentences (size: N x 4)
    X : list
        A list containing all the embeddings, per sentence.
    which_sentence : int, optional
        If you are interested in plotting just one type of sentence (by default None)
    save_me : bool, optional
        To save image, by default False
    filename : str, optional
        Descriptive filename, by default None
    """

    if which_sentence is not None:

        # If which_sentence is not None, we select a specific sentence to scatter plot, with labels
        assert which_sentence in set(df.Sentence_ID), "The sentence ID you've specified is not an available option."

        sentence_df = df.loc[df.Sentence_ID == which_sentence]

        # Get all rows which correspond to which_sentence; Get all corresponding embeddings
        X = itemgetter(*sentence_df.index.tolist())(X)
        # Dimensionality reduction
        y = TSNE(n_components=3).fit_transform(X)
        # Get all diagnoses which correspond to which_sentence
        targets = sentence_df.Diagnosis.tolist()
        # Set plotly data parameter
        data = [
            go.Scatter3d(
                x=[i[0] for i in y],
                y=[i[1] for i in y],
                z=[i[2] for i in y],
                name="Diagnosis",
                mode="markers",
                text=[
                    "SENTENCE: {}<br><br>DIAGNOSIS: {}".format(i, j)
                    for i, j in zip(sentence_df.Preprocessed_typed_sentence, targets)
                ],
                marker=dict(
                    size=10,
                    color=[i for i in targets],  # Set color equal to Diagnosis
                    opacity=0.2,
                    colorscale="RdBu",
                    showscale=False,
                ),
            )
        ]

    else:
        # Dimensionality reduction
        y = TSNE(n_components=3).fit_transform(X)
        # Set plotly data parameter in the absence of sentence-specific request
        data = [
            go.Scatter3d(
                x=[i[0] for i in y],
                y=[i[1] for i in y],
                z=[i[2] for i in y],
                name="Diagnosis",
                mode="markers",
                text=[i for i in df["Preprocessed_typed_sentence"]],
                marker=dict(
                    size=10,
                    color=[i for i in df["Sentence_ID"]],  # Set color equal to sentence ID
                    opacity=0.4,
                    colorscale="Viridis",
                    showscale=False,
                ),
            )
        ]

    # Instantiates plotly for notebooks, which is where this function is meant to be used.
    init_notebook_mode(connected=True)

    layout = go.Layout(
        showlegend=True,
        legend=dict(orientation="h"),
        autosize=False,
        width=800,
        height=800,
        margin=dict(l=0, r=0, b=0, t=0),
    )
    fig = go.Figure(data=data, layout=layout)

    if save_me:
        assert filename is not None, "You have to specify a distinctive filename."
        # Set reference time for save
        now = datetime.datetime.now()
        plot(
            fig,
            filename="../illustrations/sentence-embedding-3D-" + filename + now.strftime("-%Y-%m-%d-%H:%M") + ".html",
        )

    # Actual plotting happens here
    iplot(fig)

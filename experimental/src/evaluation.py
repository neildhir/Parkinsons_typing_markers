from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os
import argparse
import matplotlib as mpl


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
# Universal update for fonts: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
mpl.rcParams.update(nice_fonts)








def make_plot_folder(data_root: Path):
    plot_folder = data_root / 'figures'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)




def sentence_lvl_network_plot(data_root: Path):
    # Figure parameters
    lw = 3
    plt.figure(figsize=(5, 5))
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    AUC = []
    interp_tpr = []
    x = np.linspace(0, 1, 100)

    for pth in data_root.glob('fold*'):
        df = pd.read_csv(pth)
        # df = df[df.Attempt == 1]

        y = df['Diagnosis'].values
        pred = df['rough'].values

        fpr, tpr, _ = roc_curve(y, pred)
        roc_auc = auc(fpr, tpr)
        interp_tpr.append(np.interp(x, xp=fpr, fp=tpr))

        AUC.append(roc_auc)
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw - 1, alpha=0.3)

    mean = np.mean(interp_tpr, axis=0)
    std = np.std(interp_tpr, axis=0)

    plt.plot(x, mean, color='darkorange', lw=lw,
             label='AUC: {:.2f} ({:.2f}), best: {:.2}'.format(np.mean(AUC), np.std(AUC), np.max(AUC)))
    plt.fill_between(x, mean + std, mean - std, alpha=0.2, color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend(loc="lower right")

    save_as = data_root / 'figures' / 'network_ROC.pdf'
    plt.savefig(save_as)
    plt.close()

def sentence_lvl_baseline_plot(data_root: Path,df: pd.DataFrame):
    network_features = ['tuned']
    IKI_features = ['Mean_IKI', 'Var_IKI']
    ED_features = ['Edit_Distance']

    experiments = {'IKI_only' : IKI_features ,
                   'ED+IKI': ED_features + IKI_features,
                   'IKI+network': IKI_features + network_features,
                   'ED+IKI+network': ED_features + IKI_features + network_features}

    lw = 3
    for exp_name, exp_features in experiments.items():
        plt.figure(figsize=(5, 5))
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])

        interp_tpr = []
        AUC = []
        x = np.linspace(0, 1, 100)
        for fold in df.fold.unique():
            x_test = df[df.fold == fold][exp_features]
            y_test = df[df.fold == fold]['Diagnosis']
            x_train = df[df.fold != fold][exp_features]
            y_train = df[df.fold != fold]['Diagnosis']

            clf = RFC(n_estimators=300)
            clf.fit(x_train, y_train)
            p_test = clf.predict_proba(x_test)
            fpr, tpr, _ = roc_curve(y_test, p_test[:, 1])
            interp_tpr.append(np.interp(x, xp=fpr, fp=tpr))
            roc_auc = auc(fpr, tpr)
            # print(fold_idx , ': ',roc_auc)
            # print('acc:', np.sum(pred == y_test) / y_test.size)
            AUC.append(roc_auc)
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw - 1, alpha=0.3)

        mean = np.mean(interp_tpr, axis=0)
        std = np.std(interp_tpr, axis=0)
        plt.plot(x, mean, color='darkorange', lw=lw,
                 label='AUC: {:.2f} ({:.2f}), best: {:.2}'.format(np.mean(AUC), np.std(AUC), np.max(AUC)))
        plt.fill_between(x, mean + std, mean - std, alpha=0.2, color='darkorange')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc = 'lower right')
        save_to = data_root / 'figures' / (exp_name + '_ROC.pdf')
        plt.savefig(save_to)


    return 0


def sentence_lvl_plots(data_root: Path,df):

    sentence_lvl_network_plot(data_root)
    sentence_lvl_baseline_plot(data_root,df)

    return 0

def participant_lvl_plots():
    return 0



def main(data_root: str):
    data_root = Path(data_root)
    make_plot_folder(data_root)

    dfs = []
    for pth in data_root.glob('fold_*.csv'):
        dfs.append(pd.read_csv(pth))

    network_df = pd.concat(dfs)


    if 'MJFFENG' in data_root.stem:
        baseline_path = Path('../results/MJFFENG_baselines_raw.csv')
    elif 'MJFFSPAN' in data_root.stem:
        baseline_path = Path('../results/MJFFSPAN_baselines_raw.csv')

    baseline_df = pd.read_csv(baseline_path)

    df = pd.merge(network_df, baseline_df, on=['Participant_ID', 'Sentence_ID', 'Diagnosis'], how='inner', )


    sentence_lvl_plots(data_root,df)

    return 0



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path',
                        metavar='',
                        type=str,
                        required=True,
                        help='path to data')
    args = parser.parse_args()

    main(data_root = args.data_path)
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



def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))




def make_plot_folder(data_root: Path, attempt):
    for a in attempt:

        plot_folder = data_root / 'figures_attempt{}'.format(a)
        sentence_lvl_folder = plot_folder / 'sentence_lvl'
        participant_lvl_folder = plot_folder / 'participant_lvl'

        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
            os.makedirs(sentence_lvl_folder)
            os.makedirs(participant_lvl_folder)




def make_fold_data(fold, cols, data):
    y_train = data[data.fold != fold]['Diagnosis'].values
    x_train = data[data.fold != fold][cols].values

    y_test = data[data.fold == fold]['Diagnosis'].values
    x_test = data[data.fold == fold][cols].values
    return x_train, x_test, y_train, y_test

def transform2participant_lvl(network_df: pd.DataFrame):

    data_dict = {'Participant_ID': [],
                 'IKI_mean': [],
                 'IKI_std': [],
                 'ED_mean': [],
                 'ED_std': [],
                 'p_mean': []}
    for pID, data in network_df.groupby('Participant_ID'):
        timings_cat = np.concatenate(data.IKI_timings_original.values)
        data_dict['Participant_ID'].append(pID)
        data_dict['IKI_mean'].append(timings_cat.mean())
        data_dict['IKI_std'].append(timings_cat.std())
        data_dict['ED_mean'].append(data.Edit_Distance.mean())
        data_dict['ED_std'].append(data.Edit_Distance.std())
        data_dict['p_mean'].append(data.tuned.mean())

    bl_predictors = pd.DataFrame(data_dict)


    to_table = network_df[['Participant_ID', 'Sentence_ID', 'tuned']]
    diag_fold = network_df[['Participant_ID', 'Diagnosis', 'fold']].drop_duplicates()
    table = pd.pivot_table(to_table, index=['Participant_ID'], columns=['Sentence_ID'], values=['tuned']).reset_index()
    table.rename(columns={'': 'Participant_ID', 'Sentence_ID': ''}, inplace=True)
    table.columns = table.columns.droplevel()
    table = table.merge(diag_fold, on='Participant_ID')
    table = table.merge(bl_predictors, on='Participant_ID')
    table.fillna(table.mean(), inplace = True)

    return table






def sentence_lvl_network_plot(data_root: Path, folder_name: str, attempt: int = 0):
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
    num_samp = []
    for pth in data_root.glob('fold*'):
        df = pd.read_csv(pth)
        if 'Attempt' in df.columns:
            df = df[df.Attempt == attempt]

        y = df['Diagnosis'].values
        pred = df['rough'].values
        num_samp.append(len(pred))

        fpr, tpr, _ = roc_curve(y, pred)
        roc_auc = auc(fpr, tpr)
        interp_tpr.append(np.interp(x, xp=fpr, fp=tpr))

        AUC.append(roc_auc)
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw - 1, alpha=0.3)

    mean = np.mean(interp_tpr, axis=0)
    std = np.std(interp_tpr, axis=0)

    weights = np.array(num_samp)/np.sum(num_samp)
    weighted_avg_AUC, weighted_std_AUC = weighted_avg_and_std(AUC,weights)
    plt.plot(x, mean, color='darkorange', lw=lw,
             label='AUC: {:.2f} ({:.2f}), best: {:.2}'.format(weighted_avg_AUC, weighted_std_AUC, np.max(AUC)))
    plt.fill_between(x, mean + std, mean - std, alpha=0.2, color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label = 'Chance')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend(loc="lower right")

    save_as = data_root / folder_name / 'sentence_lvl' /'network_ROC.pdf'
    plt.savefig(save_as)

def sentence_lvl_baseline_plot(data_root: Path,df: pd.DataFrame, folder_name: str):
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
        num_samp = []
        for fold in df.fold.unique():
            x_train, x_test, y_train, y_test = make_fold_data(fold, exp_features, df)
            num_samp.append(len(y_test))


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

        weights = np.array(num_samp) / np.sum(num_samp)
        weighted_avg_AUC, weighted_std_AUC = weighted_avg_and_std(AUC, weights)


        plt.plot(x, mean, color='darkorange', lw=lw,
                 label='AUC: {:.2f} ({:.2f}), best: {:.2}'.format(weighted_avg_AUC, weighted_std_AUC, np.max(AUC)))
        plt.fill_between(x, mean + std, mean - std, alpha=0.2, color='darkorange')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label = 'Chance')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc = 'lower right')
        save_to = data_root / folder_name / 'sentence_lvl' /(exp_name + '_ROC.pdf')
        plt.savefig(save_to)


    return 0


def sentence_lvl_plots(data_root: Path,df: pd.DataFrame, attempt):
    folder_name = 'figures_attempt{}'.format(attempt)
    sentence_lvl_network_plot(data_root, folder_name, attempt=attempt)
    sentence_lvl_baseline_plot(data_root,df, folder_name)
    plt.close()
    return 0

def participant_lvl_plots(data_root: Path,participant_table: pd.DataFrame, network_features: list, folder_name: str):
    mean_p_features = ['p_mean']
    IKI_features = ['IKI_mean', 'IKI_std']
    ED_features = ['ED_mean', 'ED_std']

    experiments = {'mean_p': mean_p_features,
                   'IKI': IKI_features,
                   'IKI+ED': IKI_features + ED_features,
                   'mean_p+IKI': mean_p_features + IKI_features,
                   'mean_p+IKI+ED': mean_p_features + IKI_features + ED_features,
                   'network+IKI': network_features + IKI_features,
                   'network+IKI+ED': network_features + IKI_features + ED_features,
                   'network': network_features}

    for exp_name, exp_features in experiments.items():
        lw=3
        plt.figure(figsize=(5, 5))
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])

        interp_tpr = []
        AUC = []
        x = np.linspace(0, 1, 100)

        num_samp = []
        for fold in range(0,5):
            x_train, x_test, y_train, y_test = make_fold_data(fold,exp_features,participant_table)
            num_samp.append(len(y_test))
            if exp_name == 'mean_p':
                pred = x_test
            else:
                rfc = RFC(n_estimators=300)
                rfc.fit(x_train, y_train)
                pred = rfc.predict_proba(x_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test,pred)
            interp_tpr.append(np.interp(x, xp=fpr, fp=tpr))
            roc_auc = auc(fpr, tpr)
            AUC.append(roc_auc)
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw - 1, alpha=0.3)

        mean = np.mean(interp_tpr, axis=0)
        std = np.std(interp_tpr, axis=0)

        weights = np.array(num_samp) / np.sum(num_samp)
        weighted_avg_AUC, weighted_std_AUC = weighted_avg_and_std(AUC, weights)

        plt.plot(x, mean, color='darkorange', lw=lw,
                 label='AUC: {:.2f} ({:.2f}), best: {:.2}'.format(weighted_avg_AUC, weighted_std_AUC, np.max(AUC)))
        plt.fill_between(x, mean + std, mean - std, alpha=0.2, color='darkorange')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label = 'Chance')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        save_to = data_root / folder_name / 'participant_lvl' / (exp_name + '_ROC.pdf')
        plt.savefig(save_to)
        plt.close()



    return 0



def main(data_root: str):
    data_root = Path(data_root)


    dfs = []
    for pth in data_root.glob('fold_*.csv'):
        dfs.append(pd.read_csv(pth))

    network_df = pd.concat(dfs)
    network_df['IKI_timings_original'] = network_df.IKI_timings_original.apply(lambda x: eval(x))
    network_features = sorted(network_df.Sentence_ID.unique())

    if 'Attempt' in network_df.columns:
        attempt = network_df.Attempt.unique()
    else:
        attempt = [1]
    make_plot_folder(data_root, attempt)



    if 'MJFFENG' in data_root.stem:
        baseline_path = Path('../results/')
    elif 'MJFFSPAN' in data_root.stem:
        baseline_path = Path('../results/MJFFSPAN_baselines_raw.csv')
    elif 'MRC' in data_root.stem:
        baseline_path = Path('../../data/MRC/processed_MRC_all_mean_IKI_and_ED.csv')



    if 'Attempt' in network_df.columns:
        for attempt in network_df.Attempt.unique():
            sub_df = network_df[network_df.Attempt == attempt]

            baseline_df = pd.read_csv(baseline_path / 'MJFFENG_baselines_attempt_{}.csv'.format(attempt))

            sub_df = pd.merge(sub_df, baseline_df, on=['Participant_ID', 'Sentence_ID', 'Diagnosis'], how='inner', )
            print(sub_df.Attempt.unique())
            sentence_lvl_plots(data_root,sub_df, attempt)

            participant_table = transform2participant_lvl(sub_df)
            participant_lvl_plots(data_root, participant_table, network_features,'figures_attempt{}'.format(attempt))
    else:
        baseline_df = pd.read_csv(baseline_path)
        df = pd.merge(network_df, baseline_df, on=['Participant_ID', 'Sentence_ID', 'Diagnosis'], how='inner', )
        sentence_lvl_plots(data_root,df,1)
        participant_table = transform2participant_lvl(df)
        participant_lvl_plots(data_root, participant_table, network_features,'figures_attempt1')







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
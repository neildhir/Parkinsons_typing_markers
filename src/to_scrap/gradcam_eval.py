import sys

sys.path.append("..")
#sys.path.append('src/')
#from utils.utils import encode_sentence, process_sentences
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model
import pandas as pd
import json
import numpy as np
import argparse
from pathlib import Path
import os

from src.charCNN.data_utilities import create_training_data_keras
from src.gradcam.gradcam import compute_saliency
from src.gradcam.utils import reduce_input, vis_gradcam

def main(data_path, model_file, fold_idx, layer):
    data_path = Path(data_path)
    model_file = Path(model_file)
    csv_file = model_file.parents[0] / (model_file.stem + '.csv')
    save_dir = model_file.parents[0] / 'gradcam' / layer

    df = pd.read_csv(csv_file)
    df = df.set_index(keys=['Participant_ID', 'Sentence_ID'])

    df['reduced_typed_sentence'] = ""
    #df['reduced_typed_sentence'].astype(object)
    df[layer] = ""
    #df[layer] = df[layer].astype(object)






    #DATA_ROOT = Path("../data/") / args.which_dataset.upper() / "preproc"
    DATA_ROOT = data_path.parents[1]
    which_information = data_path.parents[0].stem
    char2idx_file = data_path.parents[0] / (data_path.stem + '_char2idx.json')

    with open(char2idx_file) as f:
        char2idx = json.load(f)
    idx2char = {value: key for key, value in char2idx.items()}


    X_train, X_val, X_test, y_train, y_val, y_test,test_subject_id,test_sentence_id, max_sentence_length, alphabet_size = create_training_data_keras(
        DATA_ROOT, which_information, args.data_path, test_fold_idx = fold_idx)





    model = load_model(model_file)
    print(model.summary())

    for x_input,y_,sentence_id,subject_id in zip(X_test,y_test,test_sentence_id,test_subject_id):
        sample_path = save_dir / 'sentence_{}'.format(sentence_id) / 'fold_{:03d}'.format(fold_idx)
        if not os.path.exists(sample_path):
            os.makedirs(sample_path)

        label = 'Participant: {}, Diag: {}'.format(subject_id, y_)
        plot_name = '{:4d}.png'.format(subject_id)
        save_as = sample_path / plot_name

        grads, prob = compute_saliency(model=model, x_input=x_input, layer_name=layer, cls = 1)
        grads = np.asarray(grads).reshape(-1)

        x_reduced, grad_segments = reduce_input(x_input,grads)
        string_reduced = [idx2char[x_] for x_ in x_reduced]
        vis_gradcam(string_reduced,grad_segments,label = label, save_as=save_as, show = False)

        df.loc[(subject_id,sentence_id),'reduced_typed_sentence'] = ''.join(string_reduced)
        df.loc[(subject_id, sentence_id),layer] = str(grad_segments)



    df.to_csv(csv_file)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path',
                        metavar='',
                        type=str,
                        required=True,
                        help='path to data')
    parser.add_argument('-r', '--results_path',
                        metavar='',
                        type=str,
                        required=True,
                        help='path to fold models')
    parser.add_argument('-l', '--layer',
                        metavar='',
                        type=str,
                        required=True,
                        help='name of layer to produce gradcam for')

    args = parser.parse_args()


    results_path = Path(args.results_path)
    for model_file in results_path.glob('*.h5'):
        fold_idx = int(model_file.stem.split('_')[1])

        main(data_path=args.data_path,
             fold_idx = fold_idx,
             model_file=model_file,
             layer = args.layer)
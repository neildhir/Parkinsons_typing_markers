import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.collections import LineCollection
from tensorflow.keras.models import load_model

import cv2


def grad_cam(input_model, image, cls, layer_name):
    """GradCAM method for visualizing input saliency."""

    grad_model = tf.keras.models.Model(
        [input_model.inputs], [input_model.get_layer(layer_name).output, input_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([image], dtype=np.float32))
        loss = predictions[:, cls]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    gate_f = tf.cast(output > 0, "float32")
    gate_r = tf.cast(grads > 0, "float32")
    guided_grads = tf.cast(output > 0, "float32") * tf.cast(grads > 0, "float32") * grads

    weights = tf.reduce_mean(guided_grads, axis=0)

    cam = np.ones(output.shape[0], dtype=np.float32)
    # return cam, weights, output
    for i, w in enumerate(weights):
        cam += w * output[:, i]

    cam = cv2.resize(cam.numpy(), (1, input_model.input.shape[1]))

    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())

    return heatmap


def plot_gradcam(g, sent, timings, hold_time, pause_time, saveto, metastring):
    fig, ax = plt.subplots(1, 1, figsize=(len(sent) / 2, 3))
    fig.tight_layout()
    # bup = ax.text(0.0,0.02,sent,fontsize = 62)

    xc = np.arange(0, len(sent) + 1)
    yc = np.zeros(len(sent) + 1)

    points = np.array([xc, yc]).T.reshape(-1, 1, 2)

    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(g.min(), g.max())
    lc = LineCollection(segments, cmap="jet", norm=norm)

    lc.set_array(g)
    lc.set_linewidth(15)

    line = ax.add_collection(lc)

    # fig.colorbar(line, ax=ax)

    points[:, :, 0] += 0.5
    for i, s in enumerate(sent):
        ax.text(
            points[i][0, 0], points[i][0, 1], s=s, fontsize=34, verticalalignment="bottom", horizontalalignment="center"
        )
        ax.text(
            points[i][0, 0],
            points[i][0, 1] - 0.003,
            s="{:.0f}".format(timings[i]),
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="center",
            rotation=0,
        )
        ax.text(
            points[i][0, 0],
            points[i][0, 1] - 0.006,
            s="{:.0f}".format(hold_time[i]),
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="center",
            rotation=0,
        )
        ax.text(
            points[i][0, 0],
            points[i][0, 1] - 0.009,
            s="{:.0f}".format(pause_time[i]),
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="center",
            rotation=0,
        )
    plt.autoscale(enable=True, axis="both", tight=None)

    ax.set_ylim([-0.02, 0.02])
    ax.set_xlim([0, len(sent)])
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xlabel(metastring,  fontsize=14)
    ax.text(0.1, -0.018, s=metastring, fontsize=16)

    plt.savefig(saveto)
    plt.clf()
    plt.close()


def main(root_dir):
    root = Path(root_dir)
    if not os.path.exists(root / "gradcam"):
        os.makedirs(root / "gradcam")

    gradcam_meta_dict = {}
    for fold_idx in range(5):
        df = pd.read_csv(root / "fold_{}.csv".format(fold_idx))
        df["PPTS_list"] = df.PPTS_list.apply(lambda x: eval(x))
        df["IKI_timings_original"] = df.IKI_timings_original.apply(lambda x: eval(x))
        df["hold_time_original"] = df.hold_time_original.apply(lambda x: eval(x))
        df["pause_time_original"] = df.pause_time_original.apply(lambda x: eval(x))

        layers = ["conv1d_1", "conv1d_2"]

        model = load_model(root / "tuned_{}.h5".format(fold_idx))
        X = np.load(root / "x_fold_{}.npy".format(fold_idx))

        gradcam_data = []
        df_meta = {"Participant_ID": [], "Sentence_ID": [], "Diagnosis": [], "Target": [], "Layer": [], "PPTS_list": []}
        for layer_name in layers:
            for i, x in enumerate(X):
                sent = df.PPTS_list[i]
                timings = np.insert(df.IKI_timings_original[i], 0, 0)
                hold_time = df.hold_time_original[i]
                pause_time = np.insert(df.pause_time_original[i], 0, 0)
                sID = df.Sentence_ID[i]
                pID = df.Participant_ID[i]
                diag = df.Diagnosis[i]
                prob = df.tuned[i]
                check_len = np.sum(0 != (x != 0).sum(axis=1))
                assert check_len == len(sent), "ERROR, Samples out of order!"
                for lbl in [0, 1]:
                    out = grad_cam(model, x, lbl, layer_name)

                    g = out.flatten()[-len(sent) :]
                    # gradcam_meta_dict['pID{}_sID{}_cls{}_{}'.format(pID, sID,lbl,layer_name)] = g.tolist()
                    df_meta["Participant_ID"].append(pID)
                    df_meta["Sentence_ID"].append(sID)
                    df_meta["Diagnosis"].append(diag)
                    df_meta["Target"].append(lbl)
                    df_meta["Layer"].append(layer_name)
                    df_meta["PPTS_list"].append(sent)
                    gradcam_data.append(g)

                    meta_string = "pID: {}, sID: {}, diag: {}, p: {:.2f}, target: {}".format(pID, sID, diag, prob, lbl)
                    saveto = root / "gradcam" / "pID{}_sID{}_cls{}_{}.png".format(pID, sID, lbl, layer_name)

                    plot_gradcam(g, sent, timings, hold_time, pause_time, saveto, meta_string)
                    print(i)
                    if i % 100 == 0:
                        print("Logging fold {}..".format(fold_idx))
                        df_out = pd.DataFrame(df_meta)
                        df_out["gradcam"] = gradcam_data
                        df_out.to_pickle(root / "gradcam" / "gradcam_meta_f{}.pkl".format(fold_idx))

        df.to_csv(root / "gradcam" / "fold_{}.csv".format(fold_idx), index=False)
        df_out = pd.DataFrame(df_meta)
        df_out["gradcam"] = gradcam_data
        df_out.to_pickle(root / "gradcam" / "gradcam_meta_f{}.pkl".format(fold_idx))

    # with open(root/ 'gradcam' / 'gradcam_meta.json','w') as fp:
    # json.dump(gradcam_meta_dict,fp,separators=(',', ':'), sort_keys=True, indent=4)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", metavar="", type=str, required=True, help="path to data")
    args = parser.parse_args()

    main(root_dir=args.data_path)

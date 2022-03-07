import argparse
import os
import sys

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from frog import INPUT_FPC, OUTPUT_FPC
from frog.label_codec import LabelCodec
from frog.preprocessing.preprocess_chords import import_chords
from frog.preprocessing.preprocess_scores import import_piano_roll
from frog.temperley_key_detection import (
    DEGREE2SCALE_MIN,
    DEGREE2SCALE_MAJ,
    TEMPERLEY_WEIGHTS,
    TEMPERLEY_STD,
)
from frog.temperley_key_detection.predict_key import predict_key


def _tonicise_key(key, tonicisation):
    """
    Provide the encoding for a local key after tonicisation, without pitch spelling
    :param key: encoded with no pitch spelling, 0-11 major, 12-23 minor
    :param tonicisation: encoded, 0-6 diatonic, 7-13 augmented, 14-20 diminished
    :return:
    """
    k = key % 12  # remove information about major / minor
    minor_key = bool(key // 12)  # false = major, true = minor

    # add the tonicisation to the key, knowing that it's built on the scale
    d = tonicisation % 7
    kt = k + (DEGREE2SCALE_MIN[d] if minor_key else DEGREE2SCALE_MAJ[d])
    d_mode = tonicisation // 7  # 0 = diatonic, 1 = augmented, 2 = diminished
    alteration = 1 if d_mode == 1 else -1 if d_mode == 2 else 0
    kt += alteration
    kt = kt % 12  # put everything back to the reference octave

    minor_tonicised = (not minor_key and d in [1, 2, 5, 6]) or (minor_key and d in [0, 1, 3, 6])
    kt = kt + (12 if minor_tonicised else 0)  # reinsert minor if needed
    return kt


def visualize_key_temperley(y_pred, y_true, name):
    plt.style.use("ggplot")
    cmap = sns.color_palette(["#d73027", "#f7f7f7", "#3027d7", "#000000"])

    # circle of fifths -> Ab, Eb, Bb, F, C, G, D, A, E, B, F#, C#
    circle_of_fifths = [8, 3, 10, 5, 0, 7, 2, 9, 4, 11, 6, 1]
    circle_of_fifths += [x + 12 for x in circle_of_fifths]  # adds minor keys

    # Create one-hot vectors (time, pitches) and order them according to the circle of fifths
    pred_one_hot = (np.eye(24)[y_pred])[:, circle_of_fifths]
    true_one_hot = (np.eye(24)[y_true])[:, circle_of_fifths]

    x = true_one_hot - pred_one_hot  # -1 for a FP, 0 for a TN or TP, 1 for a FN
    x[true_one_hot == 1] += 1  # -1 -> FP, 0 -> TN, 1 -> TP, 2 -> FN
    lc = LabelCodec(spelling=False, strict=False)
    ax = sns.heatmap(
        x.transpose(),
        cmap=cmap,
        vmin=-1,
        vmax=2,
        yticklabels=[lc.keys[i] for i in circle_of_fifths],
    )
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([-5 / 8, 1 / 8, 7 / 8, 13 / 8])
    colorbar.set_ticklabels(["False Pos", "True Neg", "True Pos", "False Neg"])
    ax.set(ylabel="key", xlabel="time", title=f"{name} - key")
    plt.show()
    return


def analyse_piece(sf, cf, ctx_size, hop_size):
    piano_roll = import_piano_roll(sf, "pitch", "class", INPUT_FPC)
    chords = import_chords(cf, "pitch", OUTPUT_FPC)
    lc = LabelCodec(spelling=False, strict=False)
    encoded_chords = lc.encode_chords(chords)
    assert len(piano_roll) / INPUT_FPC == len(encoded_chords) / OUTPUT_FPC
    keys_org = [c[0] for c in encoded_chords]
    keys_ton = [_tonicise_key(c[0], c[1]) for c in encoded_chords]

    assert ctx_size % 2 == 1, "ctx_size is an even number, make it odd to ensure symmetric padding"
    pad = int((ctx_size - 1) / 2)  # the amount to pad on each side of the input tensor
    piano_roll = np.pad(piano_roll, ((pad, pad), (0, 0)))  # gives enough context on borders
    keys_pred = [
        predict_key(piano_roll[i : i + ctx_size], TEMPERLEY_WEIGHTS, TEMPERLEY_STD)
        for i in range(0, len(piano_roll) - ctx_size, hop_size)
    ]
    return keys_pred, keys_org, keys_ton


def main(opts):
    parser = argparse.ArgumentParser(
        description="Analyse a dataset with Temperley's key detection algorithm"
    )
    parser.add_argument("chords_folder", action="store", help="The folder where chords are stored")
    parser.add_argument("scores_folder", action="store", help="The folder where scores are stored")
    parser.add_argument("-v", action="store_true", help="Visualise the score")
    args = parser.parse_args(opts)
    file_names = sorted([".".join(fn.split(".")[:-1]) for fn in os.listdir(args.chords_folder)])
    tp_org, tp_ton = 0, 0  # true positives for original and tonicised key
    N = 0  # total number of predictions already processed
    context = 65  # 4 quarter notes on each side + the current note
    hop_size = 4  # distance between successive predictions
    for fn in file_names:
        sf = os.path.join(args.scores_folder, fn + ".mxl")
        cf = os.path.join(args.chords_folder, fn + ".csv")
        y_pred, y_org, y_ton = analyse_piece(sf, cf, context, hop_size)
        if args.v:
            visualize_key_temperley(y_pred, y_org, sf)
        N_last = len(y_org)
        N += N_last
        tp_last = sum([yp == yt for yp, yt in zip(y_pred, y_org)])
        tp_ton_last = sum([yp == ya for yp, ya in zip(y_pred, y_ton)])
        print(
            f"{fn}, accuracy : {tp_last / N_last:.3f},"
            f" with tonicisation : {tp_ton_last / N_last:.3f}"
        )
        tp_org += tp_last
        tp_ton += tp_ton_last
    print(f"Total average accuracy : {tp_org / N:.3f}, with tonicisation : {tp_ton / N:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])

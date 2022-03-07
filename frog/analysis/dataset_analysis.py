"""
This is an entry point, no other file should import from this one.
Collect information about the dataset at hand.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from frog import INPUT_FPC
from frog.label_codec import LabelCodec
from frog.preprocessing.preprocess_chords import (
    _load_chord_labels,
)
from frog.preprocessing.preprocess_scores import (
    import_piano_roll,
)

columns = [
    "dataset",
    "file",
    "duration",
    "key",
    "degree",
    "quality",
    "inversion",
    "root",
]


def analyse_single_chords_file(cf, lc):
    chords = _load_chord_labels(cf, lc)
    rows = []
    for i, c in enumerate(chords):
        dur = c[1] - c[0]
        row = [dur, c[2].key, c[2].degree, c[2].quality, c[2].inversion, c[2].root]
        rows.append(row)
    info = pd.DataFrame(rows, columns=columns[2:])
    info["file"] = os.path.basename(cf)
    return info


def analyse_corpus(corpus_folder, spelling):
    corpus_info = pd.DataFrame(columns=columns[1:])
    lc = LabelCodec(spelling)
    for item in os.walk(corpus_folder):
        folder, sub_folders, files = item
        if "chords_B" in folder:  # For TAVERN, consider only the first set of annotations
            continue
        for f in files:
            if f.endswith(".csv"):
                analysis = analyse_single_chords_file(os.path.join(folder, f), lc)
                corpus_info = pd.concat([corpus_info, analysis])
    corpus_info["dataset"] = os.path.basename(corpus_folder)
    return corpus_info


def report_data_stats():
    root_folder = os.path.join("..", "..", "data", "datasets")
    datasets = [
        "Bach_WTC_1_Preludes",
        # "Beethoven_4tets",  # DO NOT USE THIS ONE!! This is protected by too strict a licence
        "BPS",
        "Early_Choral",
        "OpenScore-LiederCorpus",
        "Orchestral",
        "Quartets",
        "Variations_and_Grounds",
    ]
    return pd.concat(analyse_corpus(os.path.join(root_folder, ds), True) for ds in datasets)


def group_and_plot_column(info, col, output_folder):
    to_plot = info.groupby(col).sum()
    to_plot = to_plot.sort_values(by="duration", ascending=False).reset_index()
    plt.clf()  # Very important! Otherwise barplots leak from the previous figure (Why? Don't know!)
    fig1 = sns.barplot(data=to_plot, x=col, y="duration")
    _ = fig1.set_xticklabels(fig1.get_xticklabels(), rotation=90)
    fig1.set_yscale("log")
    fig1.figure.savefig(os.path.join(output_folder, col + ".png"))
    return


def create_subplot_column(info, col, axis):
    to_plot = info.groupby(col).sum() / info["duration"].sum()
    to_plot = to_plot.sort_values(by="duration", ascending=False).reset_index()
    sns.set_style()
    sns.barplot(data=to_plot, x=col, y="duration", ax=axis)
    _ = axis.set_ylabel("")
    _ = axis.set_xticklabels(axis.get_xticklabels(), rotation=90)
    return axis


def plot_data_distributions(info, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    # fmt:off
    q = [
        "M", "m", "d", "a", "m7", "M7", "D7", "d7",
        "+7", "h7", "+6", "Gr+6", "It+6", "Fr+6", "sus",
    ]
    # fmt:on
    info["quality"] = info["quality"].apply(lambda x: x if x in q else "other")
    info["tonicisation"] = info["degree"].apply(lambda x: "1" if "/" not in x else x.split("/")[1])
    info["degree"] = info["degree"].apply(lambda x: x if "/" not in x else x.split("/")[0])
    fig, axes = plt.subplots(2, 2, sharey="row")
    create_subplot_column(info, "quality", axes[0, 0])
    create_subplot_column(info, "key", axes[0, 1])
    create_subplot_column(info, "degree", axes[1, 0])
    create_subplot_column(info, "tonicisation", axes[1, 1])
    # axes = [[qlt, deg], [ton, key]]
    plt.savefig(os.path.join(output_folder, "grid.png"))
    return


def find_pitch_extremes(score_file):
    """
    Find the highest and lowest note in the piano_rolls,
     including transposition ranging from 6 down to 5 up.

    :return:
    """
    score = import_piano_roll(score_file, "pitch", "complete", INPUT_FPC)
    pitches, times = np.nonzero(score)
    return np.min(pitches), np.max(pitches)


if __name__ == "__main__":
    info = report_data_stats()
    output_folder = os.path.join("..", "..", "outputs", "plots")
    plot_data_distributions(info, output_folder)

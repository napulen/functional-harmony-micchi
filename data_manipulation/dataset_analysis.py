"""
This is an entry point, no other file should import from this one.
Collect information about the dataset at hand.
"""

import math
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config import FPQ, DATA_FOLDER, PITCH_FIFTHS
from utils_music import load_score_pitch_complete, calculate_number_transpositions_key, load_score_spelling_bass, \
    load_chord_labels


def find_pitch_extremes():
    """
    Find the highest and lowest note in the piano_rolls, including transposition ranging from 6 down to 5 up.

    :return:
    """
    min_note, max_note = 128, 0  # they are inverted on purpose! they are initial conditions that will certainly be updated
    for i in range(1, 33):
        piano_roll, t0 = load_score_pitch_complete(i, FPQ)
        min_i = np.where(np.max(piano_roll, axis=-1) == 1)[0][0] - 6  # -6 because we want room for transposing
        max_i = np.where(np.max(piano_roll, axis=-1) == 1)[0][-1] + 5  # +5 because we want room for transposing
        min_note = min(min_note, min_i)
        max_note = max(max_note, max_i)
        print(f"Sonata {i}, pitches from {min_i} to {max_i}")
    return min_note, max_note


def calculate_transpositions():
    """

    :return:
    """
    folders = [os.path.join(DATA_FOLDER, 'train'), os.path.join(DATA_FOLDER, 'valid')]

    flatwards, sharpwards = [], []
    for folder in folders:
        chords_folder = os.path.join(folder, 'chords')
        scores_folder = os.path.join(folder, 'scores')
        file_names = ['.'.join(fn.split('.')[:-1]) for fn in os.listdir(chords_folder)]
        for fn in file_names:
            # if fn not in ['ncs_Chausson_Ernest_-_7_Melodies_Op.2_No.7_-_Le_Colibri']:
            #     continue
            print(fn)
            cf = os.path.join(chords_folder, f"{fn}.csv")
            sf = os.path.join(scores_folder, f"{fn}.mxl")
            chord_labels = load_chord_labels(cf)
            piano_roll, nl_pitches, nr_pitches = load_score_spelling_bass(sf, 8)  # sharpwards in [1, 35]
            nl_keys, nr_keys = calculate_number_transpositions_key(chord_labels)  # sharpwards in [1, 35]
            flatwards.append(min(nl_keys, nl_pitches))
            sharpwards.append(min(nr_keys, nr_pitches))

    print(f"flatward transpositions  : {sorted(Counter(flatwards).items())}")
    print(f"sharpward transpositions : {sorted(Counter(sharpwards).items())}")
    return flatwards, sharpwards


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return average, math.sqrt(variance)


def plot_flattest_sharpest(ff, ss):
    sns.distplot(ff, bins=np.arange(36) - 0.5, label='flattest note')  # need 36 boundaries for 35 bins
    sns.distplot(ss, bins=np.arange(36) - 0.5, label='sharpest note')
    max_occurrence = max(Counter(ff).most_common()[0][1], Counter(ss).most_common()[0][1])
    max_ytick = max_occurrence - max_occurrence % 4
    n_yticks = max_ytick // 4 + 1
    plt.ylabel("occurrences")
    plt.xticks(np.arange(35), PITCH_FIFTHS, rotation=90)
    plt.yticks(np.linspace(0, max_ytick, n_yticks) / len(ff),
               [str(x) for x in np.linspace(0, max_ytick, n_yticks, dtype=int)])
    plt.legend()
    plt.savefig(os.path.join('../images', 'flattest_sharpest.png'), bbox_inches='tight')
    plt.show()
    return


def plot_range_histogram(r):
    sns.distplot(r, bins=np.arange(36) - 0.5, label='range of notes')
    max_occurrence = Counter(r).most_common()[0][1]
    max_ytick = max_occurrence - max_occurrence % 4
    n_yticks = max_ytick // 4 + 1
    plt.xticks(np.arange(35), np.arange(35), rotation=90)
    plt.yticks(np.linspace(0, max_ytick, n_yticks) / len(ff), [str(x) for x in np.linspace(0, max_ytick, n_yticks)])
    plt.legend()
    plt.savefig(os.path.join('../images', 'range_histogram.png'), bbox_inches='tight')
    plt.show()
    return


def plot_range_individual(ff, ss, r):
    """
    Experiments on how to plot individual ranges for all pieces.

    :param ff:
    :param ss:
    :param r:
    :return:
    """
    plt.style.use('ggplot')
    cm = [(s + f) / 2 for s, f in zip(ss, ff)]
    song_extremes = np.array([(x, y) for x, y in zip(ff, ss)], dtype=[('ff', '<i4'), ('ss', '<i4')])
    song_cmr = np.array([(x, y) for x, y in zip(cm, r)], dtype=[('cm', '<i4'), ('r', '<i4')])
    # order = np.argsort(song_extremes, order=('ff', 'ss'))
    # order = np.argsort(song_cmr, order=('r', 'cm'))
    order = np.argsort(song_cmr, order='cm')
    cmap = sns.color_palette("BrBG", 26)
    cm0 = min(cm)
    for n, o in enumerate(order):
        plt.hlines(y=n, linewidth=1.7, alpha=1., xmin=ff[o], xmax=ss[o])
        # plt.hlines(y=n, color=cmap[int((cm[o] - cm0) * 2)], linewidth=2, alpha=1., xmin=ff[o]-cm[o], xmax=ss[o]-cm[o])
    plt.scatter(ff[order], np.arange(len(order)))
    plt.scatter(ss[order], np.arange(len(order)))
    # plt.scatter([(s, y) for s, y in zip(ss[order], np.arange(len(order)))])
    plt.savefig(os.path.join('../images', 'range_individual.png'), bbox_inches='tight')
    plt.show()
    return


def calculate_distribution_of_repetitions(r):
    n = len(r)
    rep = 35 - r
    m_rep, s_rep = np.mean(rep), np.std(rep)
    mat = np.zeros((n, n))
    for i, x1 in enumerate(rep):
        for j, x2 in enumerate(rep):
            mat[i, j] = x1 / x2
    flat = mat[~np.eye(n, dtype=bool)]  # this removes the diagonal and flattens at the same time
    m_rel, s_rel = np.mean(flat), np.std(flat)

    rs = sorted(Counter(rep).items())
    res = np.zeros(len(rs))
    n = np.sum([x[1] for x in rs])
    for i in range(len(rs)):
        temp = 0
        for j in range(len(rs)):
            weight = rs[j][1] / n
            temp += (rs[i][0] / rs[j][0]) * weight
        res[i] = temp
    m_wgt, s_wgt = weighted_avg_and_std(res, [x[1] for x in rs])
    # TODO: understand why s_rel != s_wgt while m_rel == m_wgt
    return m_rel, s_rel, m_wgt, s_wgt


if __name__ == '__main__':
    # f = 'data/testvalid_bpsfh_spelling_bass_cut.tfrecords'
    # c = count_records(f)
    # print(f"{f} - {c} files")
    #
    # for m in INPUT_TYPES:
    #     tfrecords = setup_tfrecords_paths(DATA_FOLDER, m)
    #
    #     for f in tfrecords:
    #         c = count_records(f)
    #         print(f"{f} - {c} files")

    # pm, pM = find_pitch_extremes()
    # print(f'The pitch classes needed, including transpositions, are from {pm} to {pM}')
    # cProfile.run('visualize_transpositions()', sort='cumtime')
    fw, sw = calculate_transpositions()

    ff = np.array([x for x in fw])  # flattest note, between 0 and 34
    ss = np.array([35 - x for x in sw])  # sharpest, between 0 and 34 as well, because sharpwards is between 1 and 35
    r = ss - ff  # by constructions s >= f always, so r in [0, 34]
    plot_flattest_sharpest(ff, ss)
    plot_range_histogram(r)
    plot_range_individual(ff, ss, r)

    m1, s1, m2, s2 = calculate_distribution_of_repetitions(r)
    print(f"{m1} +- {s1},   {m2} +- {s2}")
    # c = count_records(TRAIN_TFRECORDS)
    # print(f'There is a total of {c} records in the train file')
    # c = count_records(VALID_TFRECORDS)
    # print(f'There is a total of {c} records in the validation file')

    repeats = range(3, 16)
    value = [3, 4, 4, 8, 13, 16, 16, 32, 25, 55, 23, 16, 10]  # coming from the analysis of the dataset as of 22/11/2019

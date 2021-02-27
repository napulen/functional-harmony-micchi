"""
This is an entry point, no other file should import from this one.
Analyse the results obtained from the model, with the possibility of generating predictions on new data, plus
obtaining the accuracy of different models on annotated data, and comparing them.
"""
import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.models import load_model

from config import FEATURES, NOTES, PITCH_FIFTHS, KEYS_PITCH, KEYS_SPELLING, QUALITY, DATA_FOLDER, MODEL_FOLDER
from load_data import load_tfrecords_dataset
from utils import setup_tfrecords_paths, find_input_type, write_tabular_annotations
from utils_music import Q2I, find_root_full_output

import tensorflow as tf
# These lines are specific to a problem I had with tf2
# https://github.com/tensorflow/tensorflow/issues/45044
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# PLOTS
def plot_chord_changes(y_true, y_pred, name, ts, inversions=True):
    """

    :param y_true: shape [outputs] (timesteps, output_features)
    :param y_pred:
    :param ts: the total number of timesteps in this prediction
    :param inversions:
    :return:
    """
    if inversions:
        yt = [np.argmax(y, axis=-1) for y in y_true]
        yp = [np.argmax(y, axis=-1) for y in y_pred]
    else:
        yt = [np.argmax(y, axis=-1) for y in y_true[:-1]]
        yp = [np.argmax(y, axis=-1) for y in y_pred[:-1]]

    change_true, change_pred = np.zeros(ts), np.zeros(ts)
    for m in range(ts - 1):
        if np.any([y[m + 1] != y[m] for y in yt]):
            change_true[m] = 1
        if np.any([y[m + 1] != y[m] for y in yp]):
            change_pred[m] = 1

    # Plotting the results
    cmap = sns.color_palette(['#d73027', '#f7f7f7', '#3027d7'])
    ax = sns.heatmap([change_true - change_pred], cmap=cmap, linewidths=.5)
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([-0.67, 0., 0.67])
    colorbar.set_ticklabels(['False Pos', 'True', 'False Neg'])
    ax.set(ylabel=['change_true', 'change_pred'], xlabel='time',
           title=f"Sonata {name} - chord consistency " +
                 ('with inversions' if inversions else 'without inversions'))
    plt.show()
    # zt = decode_results(yt)
    # zp = decode_results(yp)
    # wt = [' '.join([zt[0][i], zt[1][i], zt[2][i], zt[3][i]]) for i in range(ts)]
    # wp = [' '.join([zp[0][i], zp[1][i], zp[2][i], zp[3][i]]) for i in range(ts)]
    return


def plot_results(y_true, y_pred, name, start, mode='probabilities', pitch_spelling=True):
    """

    :param y_true: shape [outputs] (timesteps, features output)
    :param y_pred: same shape as above
    :param name: the title of the piece we are analysing
    :param start: the initial frame
    :param mode: probabilities or predictions
    :param pitch_spelling: this controls the shape and labels of the x axis in with keys
    :return:
    """
    plt.style.use("ggplot")
    if mode not in ['probabilities', 'predictions']:
        raise ValueError('mode should be either probabilities or predictions')
    cmap = sns.color_palette(['#d73027', '#f7f7f7', '#3027d7', '#000000']) if mode == 'predictions' else 'RdGy'

    tick_labels = [
        KEYS_SPELLING if pitch_spelling else KEYS_PITCH,
        [str(x + 1) for x in range(7)] + [str(x + 1) + 'b' for x in range(7)] + [str(x + 1) + '#' for x in range(7)],
        [str(x + 1) for x in range(7)] + [str(x + 1) + 'b' for x in range(7)] + [str(x + 1) + '#' for x in range(7)],
        QUALITY,
        [str(x) for x in range(4)],
        PITCH_FIFTHS if pitch_spelling else NOTES,
        PITCH_FIFTHS if pitch_spelling else NOTES,
    ]
    ylabels = FEATURES.copy()
    ylabels.append("root_der")
    for j in range(7):
        # if j > 0:  # temporary analysis tool, remove if not needed
        #     continue
        if j == 0:
            if pitch_spelling:
                ordering = [i + j for i in range(15) for j in [0, 15]]
                # ordering = [i + j for i in range(26) for j in [0, 29]]
                # [ordering.append(i) for i in [26, 27, 28]]
            else:
                ordering = [8, 3, 10, 5, 0, 7, 2, 9, 4, 11, 6, 1]
                ordering += [x + 12 for x in ordering]

            a = y_pred[j][:, ordering]
            b = y_true[j][:, ordering]
            yticklabels = [tick_labels[j][o] for o in ordering]
        else:
            if j == 6:
                a = find_root_full_output(y_pred, pitch_spelling=pitch_spelling)
                a = _indices_to_one_hot(a, 35 if pitch_spelling else 12)
                b = y_true[5]
            else:
                a = y_pred[j]
                b = y_true[j]

            yticklabels = tick_labels[j]

        if mode == 'predictions':
            a = (a == np.max(a, axis=-1, keepdims=True))
            x = b - a
            x[b == 1] += 1
            x = x.transpose()
            ax = sns.heatmap(x, cmap=cmap, vmin=-1, vmax=2, yticklabels=yticklabels, linewidths=.5)
            colorbar = ax.collections[0].colorbar
            colorbar.set_ticks([-5 / 8, 1 / 8, 7 / 8, 13 / 8])
            colorbar.set_ticklabels(['False Pos', 'True Neg', 'True Pos', 'False Neg'])
        else:
            x = b - a
            x = x.transpose()
            ax = sns.heatmap(x, cmap=cmap, center=0, vmin=-1, vmax=1, yticklabels=yticklabels, linewidths=.5)
            colorbar = ax.collections[0].colorbar
            colorbar.set_ticks([-1, 0, +1])
            colorbar.set_ticklabels(['False Pos', 'True', 'False Neg'])
        ax.set(ylabel=ylabels[j], xlabel='time',
               title=f"{name}, frames [{start}, {start + x.shape[1]}) - {ylabels[j]}")
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        plt.show()
    return


def plot_piano_roll(pr, name):
    """

    :param pr:
    :param name: the title of the piece we are analysing
    :return:
    """
    ax = sns.heatmap(pr.transpose(), vmin=0, vmax=1)
    ax.set(xlabel='time', ylabel='notes',
           title=f"{name} - piano roll data")
    plt.show()
    return


def plot_coherence(root_pred, root_der, n_classes, name):
    msk = (root_pred != root_der)
    c = np.zeros((n_classes, n_classes))
    for i, j in zip(root_pred[msk], root_der[msk]):
        c[i, j] += 1
    labels = PITCH_FIFTHS if n_classes == 35 else NOTES
    sns.heatmap(c, xticklabels=labels, yticklabels=labels, linewidths=.5)
    plt.title(f"{name} - root prediction")
    plt.xlabel("PREDICTED")
    plt.ylabel("DERIVED")
    plt.show()
    return


# ANALYSIS OF SINGLE MODEL
def _check_predictions(y_true, y_pred, index):
    """
    Check if the predictions are correct for each timestep.

    :param y_true:
    :param y_pred:
    :param index:
    :return: a boolean vector of dimension [samples]
    """
    return np.argmax(y_true[index], axis=-1) == np.argmax(y_pred[index], axis=-1)


def _indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def generate_results(data_folder, model_folder, model_name, dataset='valid', verbose=True):
    """
    The generated data is always in the shape:
    y -> [data points] [outputs] (timesteps, output_features)

    :param data_folder:
    :param model_folder:
    :param model_name:
    :param dataset:
    :param verbose:
    :return: ys_true, ys_pred, (file_names, start_frames, piano_rolls)
    """
    input_type = find_input_type(model_name)

    data_file, = setup_tfrecords_paths(data_folder, [dataset], input_type)
    test_data = load_tfrecords_dataset(data_file, batch_size=16, shuffle_buffer=1, input_type=input_type, repeat=False)

    clear_session()  # Very important to avoid memory problems
    model = load_model(os.path.join(model_folder, model_name + '.h5'))
    if verbose:
        model.summary()
        print(model_name)

    ys_true, timesteps, file_names = [], [], []  # ys_true structure = [n_data][n_labels](ts, classes)
    piano_rolls, start_frames = [], []
    n_data = 0
    for data_point in test_data.unbatch().as_numpy_iterator():
        n_data += 1
        (x, m, fn, tr, s), y = data_point
        timesteps.append(np.sum(y[0], dtype=int))  # every label has a single 1 per timestep
        file_names.append(fn[0].decode('utf-8'))
        piano_rolls.append(x[:4 * timesteps[-1]])
        start_frames.append(s[0])
        ys_true.append([label[:timesteps[-1], :] for label in y])

    # Predict new labels, same structure as ys_true
    temp = model.predict(test_data, verbose=True)
    ys_pred = [[d[e, :timesteps[e]] for d in temp] for e in range(n_data)]

    del model
    info = {
        "file_names": file_names,
        "start_frames": start_frames,
        "piano_rolls": piano_rolls,
        "timesteps": timesteps,
    }
    return ys_true, ys_pred, info


def analyse_results(ys_true, ys_pred, verbose=True):
    """
    Given the true and predicted labels, calculate several metrics on them.
    The features are key, deg1, deg2, qlt, inv, root

    :param ys_true: shape [n_data][n_labels](ts, classes)
    :param ys_pred:
    :return: a dictionary in which keys are the name of a feature we analyse and the value is its accuracy
    """
    # clear_session()  # Very important to avoid memory problems

    roman_tp = 0
    roman_inv_tp = 0
    root_tp = 0
    root_coherence = 0
    degree_tp, secondary_tp, secondary_total, d7_tp, d7_total = 0, 0, 0, 0, 0
    d7_corr = 0
    total_predictions = np.sum([_[0].shape[0] for _ in ys_true])
    n_data = len(ys_true)
    true_positives = np.zeros(6)  # true positives for each separate feature
    ps = (ys_true[0][-1].shape[1] == 35)  # TODO: Dodgy and risky. Maybe find a cleaner way
    for step in range(n_data):
        y_true, y_pred = ys_true[step], ys_pred[step]  # shape: [outputs], (timestep, output features)
        correct = np.array([_check_predictions(y_true, y_pred, j) for j in range(6)])  # shape: (output, timestep)
        true_positives += np.sum(correct, axis=-1)  # true positives per every output
        roman_tp += np.sum(np.prod(correct[:4], axis=0), axis=-1)
        roman_inv_tp += np.sum(np.prod(correct[:5], axis=0), axis=-1)
        degree_tp += np.sum(np.prod(correct[1:3], axis=0), axis=-1)
        secondary_msk = (np.argmax(y_true[1], axis=-1) != 0)  # only chords on secondary degrees
        secondary_total += sum(secondary_msk)
        secondary_tp += np.sum(np.prod(correct[1:3], axis=0)[secondary_msk], axis=-1)
        root_der = find_root_full_output(y_pred, pitch_spelling=ps)
        root_coherence += np.sum(
            root_der == np.argmax(y_pred[5], axis=-1))  # if predicted and derived root are the same
        root_tp += np.sum(root_der == np.argmax(y_true[5], axis=-1))
        d7_msk = (np.argmax(y_true[3], axis=-1) == Q2I['d7'])
        d7_total += sum(d7_msk)
        d7_tp += np.sum(np.prod(correct[:4], axis=0)[d7_msk], axis=-1)
        d7_corr += np.sum(correct[3][d7_msk], axis=-1)

    acc = 100 * true_positives / total_predictions
    derived_features = ['degree', 'secondary', 'derived root', 'roman', 'roman + inv', 'root coherence', 'd7 no inv']
    keys = FEATURES + derived_features
    values = [a for a in acc] + [
        100 * degree_tp / total_predictions,
        100 * secondary_tp / secondary_total,
        100 * root_tp / total_predictions,
        100 * roman_tp / total_predictions,
        100 * roman_inv_tp / total_predictions,
        100 * root_coherence / total_predictions,
        100 * d7_tp / d7_total,
    ]
    accuracies = dict(zip(keys, values))
    if verbose:
        print(f"accuracy for the different items:")
        for k, v in accuracies.items():
            print(f'{k:15}: {v:2.2f} %')

    return accuracies


# MODELS COMPARISON
def _write_comparison_file(model_outputs, fp_out):
    """
    Take the output from several models and writes them to a general comparison file.
    The outputs need to be stored in tuples. The first element is the model name, the second is a dictionary
    containing the name of a feature as key and the accuracy as value

    :param model_outputs:
    :param fp_out:
    :return:
    """
    features = list(model_outputs[0][1].keys())

    with open(fp_out, 'w+') as f:
        w = csv.writer(f)
        w.writerow(['model name'] + features)
        for model_name, accuracies in model_outputs:
            w.writerow([model_name] + [round(accuracies[feat], 2) for feat in features])
        bps_paper = {
            'key': 66.65,
            'quality': 60.59,
            'inversion': 59.1,
            'degree': 51.79,
            'secondary': 3.97,
            'roman + inv': 25.69,
        }

        ht_paper = {
            'key': 78.35,
            'quality': 74.60,
            'inversion': 62.13,
            'degree': 65.06,
            'secondary': 68.15,
        }

        temperley = {
            'key': 67.03,
        }

        for feat in features:
            if feat not in bps_paper.keys():
                bps_paper[feat] = 'NA'
            if feat not in ht_paper.keys():
                ht_paper[feat] = 'NA'
            if feat not in temperley.keys():
                temperley[feat] = 'NA'

        w.writerow(['bps-fh_paper'] + [bps_paper[feat] for feat in features])
        w.writerow(['ht_paper'] + [ht_paper[feat] for feat in features])
        w.writerow(['temperley'] + [temperley[feat] for feat in features])
    return


def _average_results(fp_in, fp_out):
    """
    Write to fp_out the results in fp_in marginalized over one feature at a time
    :param fp_in: The file path to the comparison file we want to average
    """
    data = pd.read_csv(fp_in, header=0, index_col=0)
    res = pd.DataFrame()
    res['c1_local'] = data.loc[data.index.str.contains('_local_')].mean()
    res['c1_global'] = data.loc[data.index.str.contains('conv_') & ~data.index.str.contains('_local_')].mean()
    res['c2_conv_dil'] = data.loc[data.index.str.contains('conv_dil')].mean()
    res['c2_conv_gru'] = data.loc[data.index.str.contains('conv_gru')].mean()
    res['c2_gru'] = data.loc[data.index.str.contains('gru_') & ~data.index.str.contains('conv_')].mean()
    res['c3_spelling'] = data.loc[data.index.str.contains('_spelling_')].mean()
    res['c3_pitch'] = data.loc[data.index.str.contains('_pitch_')].mean()
    res['c4_complete'] = data.loc[data.index.str.contains('_complete_')].mean()
    res['c4_bass'] = data.loc[data.index.str.contains('_bass_')].mean()
    res['c4_class'] = data.loc[data.index.str.contains('_class_')].mean()
    res = res.transpose()
    columns = ['key', 'degree', 'quality', 'inversion', 'roman + inv', 'secondary', 'd7 no inv']
    # (res - res.loc['c3_spelling']).loc[res.index.str.contains('c3'), columns]
    # return data, res
    res[columns].to_csv(fp_out)
    return


def _t_test_results(fp_in, columns=None):
    """
    Print to screen the results of the t-test on the importance of architecture choices.

    :param fp_in: The file path to the comparison file
    """
    data = pd.read_csv(fp_in, header=0, index_col=0)
    from scipy.stats import ttest_ind

    if columns is None:
        columns = data.columns
    for col in columns:
        c1_local = data.loc[data.index.str.contains('_local_'), col].to_numpy()
        c1_global = data.loc[data.index.str.contains('conv_') & ~data.index.str.contains('_local_'), col].to_numpy()
        c2_conv_dil = data.loc[data.index.str.contains('conv_dil'), col].to_numpy()
        c2_conv_gru = data.loc[data.index.str.contains('conv_gru'), col].to_numpy()
        c2_gru = data.loc[data.index.str.contains('gru_') & ~data.index.str.contains('conv_'), col].to_numpy()
        c3_spelling = data.loc[data.index.str.contains('_spelling_'), col].to_numpy()
        c3_pitch = data.loc[data.index.str.contains('_pitch_'), col].to_numpy()
        c4_complete = data.loc[data.index.str.contains('_complete_'), col].to_numpy()
        c4_bass = data.loc[data.index.str.contains('_bass_'), col].to_numpy()
        c4_class = data.loc[data.index.str.contains('_class_'), col].to_numpy()

        comparisons = [
            (c1_global, c1_local, 'global vs. local'),
            (c2_gru, c2_conv_dil, 'gru vs conv_dil'),
            (c2_gru, c2_conv_gru, 'gru vs conv_gru'),
            (c2_conv_dil, c2_conv_gru, 'conv_dil vs conv_gru'),
            (c3_pitch, c3_spelling, 'pitch vs. spelling'),
            (c4_complete, c4_class, 'complete vs. class'),
            (c4_complete, c4_bass, 'complete vs. bass'),
            (c4_class, c4_bass, 'class vs. bass'),
        ]

        print(col)
        for c in comparisons:
            a, b, t = c
            print(f'{t:<21}: p-value {ttest_ind(a, b).pvalue:.1e}')
        print("")
    return


def compare_results(data_folder, models_folder, dataset, export_annotations):
    """
    Check all the models in the log folder and calculate their accuracy scores, then write a comparison table to file

    :param data_folder:
    :param dataset: either beethoven (all 32 sonatas) or validation (7 sonatas not in training set)
    :param export_annotations: boolean, whether to write analyses to file
    :return:
    """
    models = os.listdir(models_folder)
    n = len(models)
    results = []
    for i, model_name in enumerate(models):
        # if model_name != 'conv_gru_pitch_bass_cut_1':
        #     continue
        print(f"model {i + 1} out of {n} - {model_name}")
        model_folder = os.path.join(models_folder, model_name)
        ys_true, ys_pred, info = generate_results(data_folder, model_folder, model_name, dataset, verbose=False)
        if export_annotations:
            write_tabular_annotations(ys_pred, info["timesteps"], info["file_names"], os.path.join(model_folder, 'analyses'))
        accuracies = analyse_results(ys_true, ys_pred, verbose=False)
        results.append((model_name, accuracies))

    return results


if __name__ == '__main__':
    # dataset = 'beethoven'
    # dataset = 'validation'
    dataset = 'valid'
    models_folder = MODEL_FOLDER

    # model_name = ''
    # mf = os.path.join(models_folder, model_name)
    # ys_true, ys_pred, info = generate_results(DATA_FOLDER, mf, model_name)
    # write_tabular_annotations(ys_pred, info["timesteps"], info["file_names"], os.path.join(mf, 'analyses'))
    # acc = analyse_results(ys_true, ys_pred)
    model_with_accuracies = compare_results(DATA_FOLDER, models_folder, dataset, export_annotations=True)
    print(model_with_accuracies)
    # comparison_fp = os.path.join(models_folder, '..', f'comparison_{datetime.now().strftime("%Y-%m-%d_%H-%M")}.csv')
    # _write_comparison_file(model_with_accuracies, comparison_fp)
    # _average_results(comparison_fp, comparison_fp.replace("comparison", "average"))
    # _t_test_results(comparison_fp)


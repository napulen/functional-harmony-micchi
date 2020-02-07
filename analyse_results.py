"""
Analyse the results obtained from the model
The data is always in the shape:
x -> [data points] (timesteps, pitches)
y -> [data points] [outputs] (timesteps, output_features)

"""
import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.models import load_model

from config import FEATURES, NOTES, N_VALID, PITCH_FIFTHS, \
    VALID_BATCH_SIZE, VALID_STEPS, TEST_BPS_BATCH_SIZE, TEST_BPS_STEPS, N_TEST_BPS, DATA_FOLDER, KEYS_PITCH, \
    KEYS_SPELLING, QUALITY
from load_data import load_tfrecords_dataset
from utils import create_dezrann_annotations, setup_tfrecords_paths, find_input_type, create_tabular_annotations
from utils_music import Q2I, find_root_full_output


def check_predictions(y_true, y_pred, index):
    """
    Check if the predictions are correct for each timestep.

    :param y_true:
    :param y_pred:
    :param index:
    :return: a boolean vector of dimension [samples]
    """
    return np.argmax(y_true[index], axis=-1) == np.argmax(y_pred[index], axis=-1)


def visualize_chord_changes(y_true, y_pred, name, ts, inversions=True):
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


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def visualize_results(y_true, y_pred, name, start, mode='probabilities', pitch_spelling=True):
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
                a = indices_to_one_hot(a, 35 if pitch_spelling else 12)
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


def visualize_piano_roll(pr, name):
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


def analyse_results(data_folder, models_folder, model_name, dataset='validation', comparison=False, visualize=True,
                    export_annotations=True):
    clear_session()
    model = load_model(os.path.join(models_folder, model_name, model_name + '.h5'))
    if comparison:  # force no visualization when comparing models
        visualize = False
    if not comparison:
        model.summary()
        print(model_name)

    input_type = find_input_type(model_name)
    datasets = ['valid', 'BPS', 'valid_bpsfh']
    valid, test_bps, valid_bpsfh = setup_tfrecords_paths(data_folder, datasets, input_type)
    ps = input_type.startswith('spelling')

    if dataset == 'beethoven':
        data_file = test_bps
        batch_size = TEST_BPS_BATCH_SIZE
        steps = TEST_BPS_STEPS
        n_chunks = N_TEST_BPS
    elif dataset == 'validation':
        data_file = valid
        batch_size = VALID_BATCH_SIZE
        steps = VALID_STEPS
        n_chunks = N_VALID
    elif dataset == 'validation_bpsfh':
        data_file = valid_bpsfh
        batch_size = 11
        steps = 9
        n_chunks = 99
    elif dataset == 'test':
        data_file = os.path.join(data_folder, f'test_{input_type}.tfrecords')
        batch_size = 1
        steps = 103
        n_chunks = 103
    else:
        raise ValueError("dataset should be either validation or beethoven")

    test_data = load_tfrecords_dataset(data_file, batch_size, shuffle_buffer=1, input_type=input_type)

    """ Retrieve the true labels """
    piano_rolls, test_true, timesteps, file_names, start_frames = [], [], [], [], []  # test_true structure = [n_chunks][LABELS](ts, classes)
    test_data_iter = test_data.make_one_shot_iterator()
    (x, m, fn, tr, s), y = test_data_iter.get_next()
    with tf.Session() as sess:
        for i in range(steps):
            file_name, piano_roll, labels, start = sess.run(
                [fn, x, y, s])  # shapes: (bs, b_ts, pitches), [output](bs, b_ts, output features)
            # all elements in the batch have different length, so we have to find the correct number of ts for each
            [timesteps.append(np.sum(d, dtype=int)) for d in labels[0]]  # every label has a single 1 per timestep
            [piano_rolls.append(d[:4 * ts]) for d, ts in zip(piano_roll, timesteps)]
            [file_names.append(fn[0].decode('utf-8')) for fn in file_name]
            [start_frames.append(sf[0]) for sf in start]
            for e in range(batch_size):  # e is the element in the batch, append shape (timesteps[j], output_features)
                test_true.append([d[e, :timesteps[e + i * batch_size], :] for d in labels])
    # test_true structure = [n_chunks][LABELS](ts, classes)

    """ Predict new labels """
    # test_pred = []  # It will have shape: [pieces][features](length of sonata, feature size)
    temp = model.predict(test_data, steps=steps, verbose=True)
    test_pred = [[d[e, :timesteps[e]] for d in temp] for e in range(n_chunks)]

    """ Visualize data """
    if visualize:
        classes_root = 35 if ps else 12  # the twelve notes without enharmonic duplicates
        for pr, y_true, y_pred, ts, fn, frame in zip(piano_rolls, test_true, test_pred, timesteps, file_names,
                                                     start_frames):
            if 'bps_06' not in fn:
                continue
            # if 'Einsamkeit' not in fn:
            #     continue
            # if 'wtc_i_prelude_01' not in fn:
            #     continue
            # visualize_piano_roll(pr, fn)
            visualize_results(y_true, y_pred, fn, frame, mode='predictions', pitch_spelling=ps)
            # visualize_results(y_true, y_pred, fn, frame, mode='probabilities', pitch_spelling=ps)
            # visualize_chord_changes(y_true, y_pred, fn, ts, True)
            # visualize_chord_changes(y_true, y_pred, fn, ts, False)
            # plot_coherence(np.argmax(y_pred[5], axis=-1), find_root_full_output(y_pred), n_classes=classes_root, name=fn)
            pass

    """ Create Dezrann annotations """
    if export_annotations:
        output_folder = os.path.join(models_folder, model_name, 'analyses')
        create_dezrann_annotations(test_pred, model_name, test_true, timesteps, file_names, output_folder=output_folder)
        create_tabular_annotations(test_pred, timesteps, file_names, output_folder=output_folder)

    """" Calculate accuracy etc. """
    roman_tp, roman_inv_tp, root_tp = 0, 0, 0
    root_coherence = 0
    degree_tp, secondary_tp, secondary_total, d7_tp, d7_total = 0, 0, 0, 0, 0
    d7_corr = 0
    total_predictions = np.sum(timesteps)  # one prediction per timestep
    true_positives = np.zeros(6)  # true positives for each separate feature
    for step in range(n_chunks):
        y_true, y_pred = test_true[step], test_pred[step]  # shape: [outputs], (timestep, output features)
        correct = np.array([check_predictions(y_true, y_pred, j) for j in range(6)])  # shape: (output, timestep)
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
    if not comparison:
        print(f"accuracy for the different items:")
        for k, v in accuracies.items():
            print(f'{k:15}: {v:2.2f} %')

    del model
    return accuracies


def compare_results(data_folder, models_folder, dataset, export_annotations):
    """
    Check all the models in the log folder and derive their accuracy scores, then write a comparison table to file

    :param data_folder:
    :param dataset: either beethoven (all 32 sonatas) or validation (7 sonatas not in training set)
    :param export_annotations: boolean, whether to write analyses to file
    :return:
    """
    models = sorted(os.listdir(models_folder))
    n = len(models)
    results = []
    for i, model_name in enumerate(models):
        # if model_name != 'conv_gru_pitch_bass_cut_1':
        #     continue
        print(f"model {i + 1} out of {n} - {model_name}")
        accuracies = analyse_results(data_folder, models_folder, model_name, dataset, comparison=True,
                                     export_annotations=export_annotations)
        results.append((model_name, accuracies))

    features = list(results[0][1].keys())
    file_path = os.path.join(models_folder, '..', f'comparison_{datetime.now().strftime("%Y-%m-%d_%H-%M")}.csv')
    with open(file_path, 'w+') as f:
        w = csv.writer(f)
        w.writerow(['model name'] + features)
        for model_name, accuracies in results:
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


def average_results(fp):
    """

    :param fp: The file path to the comparison file we want to average
    """
    data = pd.read_csv(fp, header=0, index_col=0)
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
    return data, res


def t_test_results(fp, column='roman + inv'):
    """

    :param fp: The file path to the comparison file we want to average
    """
    data = pd.read_csv(fp, header=0, index_col=0)
    from scipy.stats import ttest_ind

    c1_local = data.loc[data.index.str.contains('_local_'), column].to_numpy()
    c1_global = data.loc[data.index.str.contains('conv_') & ~data.index.str.contains('_local_'), column].to_numpy()
    c2_conv_dil = data.loc[data.index.str.contains('conv_dil'), column].to_numpy()
    c2_conv_gru = data.loc[data.index.str.contains('conv_gru'), column].to_numpy()
    c2_gru = data.loc[data.index.str.contains('gru_') & ~data.index.str.contains('conv_'), column].to_numpy()
    c3_spelling = data.loc[data.index.str.contains('_spelling_'), column].to_numpy()
    c3_pitch = data.loc[data.index.str.contains('_pitch_'), column].to_numpy()
    c4_complete = data.loc[data.index.str.contains('_complete_'), column].to_numpy()
    c4_bass = data.loc[data.index.str.contains('_bass_'), column].to_numpy()
    c4_class = data.loc[data.index.str.contains('_class_'), column].to_numpy()

    A = [c1_global, c2_gru, c2_gru, c3_pitch, c4_complete, c4_complete]
    B = [c1_local, c2_conv_dil, c2_conv_gru, c3_spelling, c4_class, c4_bass]
    title = ['global vs. local', 'gru vs. conv_dil', 'gru vs. conv_gru', 'pitch vs. spelling', 'complete vs. class', 'complete vs. bass']
    for a, b, t in zip(A, B, title):
        print(f'{t}: {ttest_ind(a, b)}')
    return


if __name__ == '__main__':
    # dataset = 'beethoven'
    # dataset = 'validation'
    dataset = 'test'
    # dataset = 'validation_bpsfh'
    # models_folder = os.path.join('runs', 'run_bps', 'models')
    models_folder = os.path.join('runs', 'run_06', 'models')
    data_folder = 'data_small'
    # data_folder = DATA_FOLDER
    # compare_results(data_folder, models_folder, dataset, export_annotations=True)
    analyse_results(data_folder, models_folder, 'conv_dil_spelling_bass_cut_2', dataset=dataset, visualize=False)

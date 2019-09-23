"""
Analyse the results obtained from the model
The data is always in the shape:
x -> [data points] (timesteps, pitches)
y -> [data points] [outputs] (timesteps, output_features)

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.python.keras.models import load_model

from config import FEATURES, TICK_LABELS, CIRCLE_OF_FIFTH, NOTES, \
    VALID_TFRECORDS, VALID_STEPS, VALID_INDICES, MODE2INPUT_SHAPE, MODE, N_VALID, CLASSES_ROOT, PITCH_LINE
from load_data import create_tfrecords_dataset
from preprocessing import Q2I
from utils import find_root_full_output, decode_results


def check_predictions(y_true, y_pred, index):
    """
    Check if the predictions are correct for each timestep.

    :param y_true:
    :param y_pred:
    :param index:
    :return: a boolean vector of dimension [samples]
    """
    return np.argmax(y_true[index], axis=-1) == np.argmax(y_pred[index], axis=-1)


def visualize_chord_changes(y_true, y_pred, ts, inversions=True):
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
    ax = sns.heatmap([change_true - change_pred], cmap=cmap)
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([-0.67, 0., 0.67])
    colorbar.set_ticklabels(['False Pos', 'True', 'False Neg'])
    ax.set(ylabel=['change_true', 'change_pred'], xlabel='time',
           title=f"Sonata {VALID_INDICES[step]} - chord consistency " +
                 ('with inversions' if inversions else 'without inversions'))
    plt.show()
    zt = decode_results(yt)
    zp = decode_results(yp)
    wt = [' '.join([zt[0][i], zt[1][i], zt[2][i], zt[3][i]]) for i in range(ts)]
    wp = [' '.join([zp[0][i], zp[1][i], zp[2][i], zp[3][i]]) for i in range(ts)]
    print("hi")
    return


def visualize_results(y_true, y_pred, name, mode='probabilities', pitch_spelling=True):
    """

    :param y_true: shape [outputs] (timesteps, features output)
    :param y_pred: same shape as above
    :param name: the title of the piece we are analysing
    :param mode: probabilities or predictions
    :param pitch_spelling: this controls the shape and labels of the x axis in with keys
    :return:
    """
    if mode not in ['probabilities', 'predictions']:
        raise ValueError('mode should be either probabilities or predictions')
    cmap = sns.color_palette(['#d73027', '#f7f7f7', '#3027d7', '#000000']) if mode == 'predictions' else 'RdGy'

    for j in range(6):
        a = y_pred[j] if (j > 0 or pitch_spelling) else y_pred[step][j][:, CIRCLE_OF_FIFTH]
        b = y_true[j] if (j > 0 or pitch_spelling) else y_true[step][j][:, CIRCLE_OF_FIFTH]
        if mode == 'predictions':
            a = (a == np.max(a, axis=-1, keepdims=True))
            x = b - a
            x[b == 1] += 1
            x = x.transpose()
            ax = sns.heatmap(x, cmap=cmap, vmin=-1, vmax=2, yticklabels=TICK_LABELS[j])
            colorbar = ax.collections[0].colorbar
            colorbar.set_ticks([-5 / 8, 1 / 8, 7 / 8, 13 / 8])
            colorbar.set_ticklabels(['False Pos', 'True Neg', 'True Pos', 'False Neg'])
        else:
            x = b - a
            x = x.transpose()
            ax = sns.heatmap(x, cmap=cmap, center=0, vmin=-1, vmax=1, yticklabels=TICK_LABELS[j])
            colorbar = ax.collections[0].colorbar
            colorbar.set_ticks([-1, 0, +1])
            colorbar.set_ticklabels(['False Pos', 'True', 'False Neg'])

        ax.set(ylabel=FEATURES[j], xlabel='time',
               title=f"{name} - {FEATURES[j]}")
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
    labels = PITCH_LINE if n_classes == 35 else NOTES
    sns.heatmap(c, xticklabels=labels, yticklabels=labels)
    plt.title(f"{name} - root prediction")
    plt.xlabel("PREDICTED")
    plt.ylabel("DERIVED")
    plt.show()
    return


i = 0
model_name = f'conv_dil_reduced_{MODE}_{i}'
model_folder = os.path.join('logs', model_name)
model = load_model(os.path.join(model_folder, model_name + '.h5'))
model.summary()
print(model_name)

bs = 7
n = N_VALID // bs
test_data = create_tfrecords_dataset(VALID_TFRECORDS, bs, shuffle_buffer=1, n=MODE2INPUT_SHAPE[MODE])

# Retrieve the true labels
piano_rolls, test_truth, timesteps = [], [], []
test_data_iter = test_data.make_one_shot_iterator()
x, y = test_data_iter.get_next()
with tf.Session() as sess:
    for i in range(n):
        data = sess.run([x, y])
        # all elements in the batch have different length
        [timesteps.append(np.sum(d, dtype=int)) for d in data[1][0]]  # there is a single 1 per timestep
        [piano_rolls.append(d[:4*ts]) for d, ts in zip(data[0], timesteps)]  # data[0] has shape (bs, timestep, pitches)
        for e in range(bs):  # data[1] has shape [output](bs, timestep, output features)
            test_truth.append([d[e, :timesteps[e]] for d in data[1]])

# Predict new labels
test_predict = []  # It will have shape: [pieces][features](length of sonata, feature size)
for step in range(n):
    print(f"step {step + 1} out of {n}")
    temp = model.predict(test_data.skip(step * bs), steps=1, verbose=False)
    for e in range(bs):
        test_predict.append([d[e, :timesteps[e]] for d in temp])

# Visualize some data
# for i in range(bs*n):
#     pr, y_true, y_pred, ts = piano_rolls[i], test_truth[i], test_predict[i], timesteps[i]
#     name = f'Sonata {VALID_INDICES[i]}'
#     visualize_piano_roll(pr, name)
#     visualize_results(y_true, y_pred, name, mode='predictions')
#     visualize_results(y_true, y_pred, name, mode='probabilities')
#     visualize_chord_changes(y_true, y_pred, ts, True)
#     visualize_chord_changes(y_true, y_pred, ts, False)
#     plot_coherence(np.argmax(y_pred[5], axis=-1), find_root_full_output(y_pred), n_classes=CLASSES_ROOT, name=name)

# for i in range(bs*n):
#     create_dezrann_annotations(test_truth[i], test_predict[i], n=VALID_INDICES[i], batch_size=BATCH_SIZE,
#                                model_folder=model_folder)

# Calculate accuracy etc.
roman_tp, root_tp = 0, 0
root_coherence = 0
degree_tp, secondary_tp, secondary_total, d7_tp, d7_total = 0, 0, 0, 0, 0
total_predictions = np.sum(timesteps)
true_positives = np.zeros(6)  # true positives for each separate feature
for step in range(bs * n):
    y_true, y_pred = test_truth[step], test_predict[step]  # shape: [outputs], (timestep, output features)
    correct = np.array([check_predictions(y_true, y_pred, j) for j in range(6)])  # shape: (output, timestep)
    true_positives += np.sum(correct, axis=-1)  # true positives per every output
    roman_tp += np.sum(np.prod(correct[:4], axis=0), axis=-1)
    degree_tp += np.sum(np.prod(correct[1:3], axis=0), axis=-1)
    secondary_msk = (np.argmax(y_true[1], axis=-1) != 0)  # only chords on secondary degrees
    secondary_total += sum(secondary_msk)
    secondary_tp += np.sum(np.prod(correct[1:3], axis=0)[secondary_msk], axis=-1)
    root_der = find_root_full_output(y_pred)
    root_coherence += np.sum(root_der == np.argmax(y_pred[5], axis=-1))  # if predicted and derived root are the same
    root_tp += np.sum(root_der == np.argmax(y_true[5], axis=-1))
    d7_msk = (np.argmax(y_true[3], axis=-1) == Q2I['d7'])
    d7_total += sum(d7_msk)
    d7_tp += np.sum(np.prod(correct[:4], axis=0)[d7_msk], axis=-1)

acc = true_positives / total_predictions
degree_acc = degree_tp / total_predictions
secondary_acc = secondary_tp / secondary_total
roman_acc = roman_tp / total_predictions
print(f"accuracy for the different items:")
for f, a in zip(FEATURES, acc):
    print(f"{f:10} : {a:.4f}")
print(f'')
print(f'degree        : {degree_tp / total_predictions:.4f}')
print(f'secondary     : {secondary_tp / secondary_total:.4f}')
print(f'derived root  : {root_tp / total_predictions:.4f}')
print(f"global roman  : {roman_acc:.4f}")
print(f'root_coherence: {root_coherence / total_predictions:.4f}')
print(f'd7 chords     : {d7_tp / d7_total:.4f}')
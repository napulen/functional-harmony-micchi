import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.python.keras.models import load_model

from config import BATCH_SIZE, TEST_STEPS, TEST_INDICES, FEATURES, TICK_LABELS, CIRCLE_OF_FIFTH, TEST_TFRECORDS, \
    QUALITY, NOTES
from load_data import create_tfrecords_dataset
from utils import create_dezrann_annotations, find_chord_symbol, Q2S, S2I


def check_predictions(y_true, y_pred, index):
    return np.argmax(y_true[index], axis=-1) == np.argmax(y_pred[index], axis=-1)


def visualize_results(mode='probabilities'):
    if mode not in ['probabilities', 'predictions']:
        raise ValueError('mode should be either probabilities or predictions')
    cmap = sns.color_palette(['#d73027', '#f7f7f7', '#3027d7', '#000000']) if mode == 'predictions' else 'RdGy'

    for j in range(7):
        a = test_predict[i][j][0] if j > 0 else test_predict[i][j][0][:, CIRCLE_OF_FIFTH]
        b = test_truth[i][j][0] if j > 0 else test_truth[i][j][0][:, CIRCLE_OF_FIFTH]
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
               title=f"Sonata {TEST_INDICES[i]} - {FEATURES[j]}")
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()


def visualize_piano_roll(pr, i):
    ax = sns.heatmap(pr[i].transpose(), vmin=0, vmax=1)
    ax.set(xlabel='time', ylabel='notes',
           title=f"Sonata {TEST_INDICES[i]} - piano roll data")
    plt.show()
    return


def plot_coherence(y_symb, y_func, n_classes, sonata):
    msk = (y_symb != y_func)
    y_symb = np.array(y_symb)[msk]
    y_func = np.array(y_func)[msk]
    c = np.zeros((n_classes, n_classes))
    for i, j in zip(y_symb, y_func):
        c[i, j] += 1
    sns.heatmap(c, xticklabels=NOTES, yticklabels=NOTES)
    plt.title(f"Sonata {sonata} - root prediction")
    plt.xlabel("SYMB")
    plt.ylabel("FUNC")
    plt.show()
    return


def find_root_from_output(y_pred):
    key, degree_den, degree_num = np.argmax(y_pred[0][0], axis=-1), np.argmax(y_pred[1][0], axis=-1), np.argmax(
        y_pred[2][0], axis=-1)
    deg2sem_maj = [0, 2, 4, 5, 7, 9, 11]
    deg2sem_min = [0, 2, 3, 5, 7, 8, 10]

    root_pred = []
    for i in range(len(key)):
        deg2sem = deg2sem_maj if key[i] // 2 == 0 else deg2sem_min
        n_den = deg2sem[degree_den[i] % 7]
        if degree_den[i] // 7 == 1:
            n_den -= 1
        elif degree_den[i] // 7 == 2:
            n_den += 1
        n_num = deg2sem[degree_num[i] % 7]
        if degree_num[i] // 7 == 1:
            n_num -= 1
        elif degree_num[i] // 7 == 2:
            n_num += 1
        root_pred.append((key[i] % 12 + n_num + n_den) % 12)
    return root_pred


test_data = create_tfrecords_dataset(TEST_TFRECORDS, BATCH_SIZE, shuffle_buffer=1)
test_data_iter = test_data.make_one_shot_iterator()
x, y = test_data_iter.get_next()

model_folder = os.path.join('logs', 'conv_gru_bass')
model = load_model(os.path.join(model_folder, 'conv_gru_bass.h5'))
model.summary()

# Retrieve the true labels
piano_rolls = []
test_truth = []
with tf.Session() as sess:
    for i in range(TEST_STEPS):
        data = sess.run([x, y])
        piano_rolls.append(data[0][0][0])  # meaning of zeros: x or y, piano roll or bass, first element of batch
        test_truth.append(data[1])

# visualize_piano_roll(piano_rolls, 0)

# Predict new labels and view the difference
test_predict = []  # It will have shape: [pieces, features, (batch size, length of sonata, feature size)]
for i in range(TEST_STEPS):
    print(f"step {i + 1} out of {TEST_STEPS}")
    temp = model.predict(test_data.skip(i), steps=1, verbose=False)
    test_predict.append(temp)

    visualize_results(mode='predictions')
    # visualize_results(mode='probabilities')

for i in range(TEST_STEPS):
    create_dezrann_annotations(test_truth[i], test_predict[i], n=TEST_INDICES[i], batch_size=BATCH_SIZE,
                               model_folder=model_folder)

# Calculate accuracy etc.
func_tp, name_tp, root_tp, symb_tp, total = 0, 0, 0, 0, 0
root_coherence, symb_coherence = 0, 0
degree_tp, secondary_tp, secondary_total = 0, 0, 0
cp = [[] for _ in range(7)]  # dimensions will be: features, sonatas, frames
tp = np.zeros(7)
for i in range(TEST_STEPS):
    y_true, y_pred = test_truth[i], test_predict[i]
    total += y_true[0].shape[1]
    for j in range(7):
        for a in check_predictions(y_true, y_pred, j):
            cp[j].append(a)  # cp has shape (7, batch_size*(i+1), l) where l varies
    # can't vectorialize the next three lines because the length of each sonata l is different
    tp += np.sum(np.array([a[-1] for a in cp]), axis=-1)
    func_tp += np.sum(np.prod(np.array([a[-1] for a in cp[:4]]), axis=0), axis=-1)
    name_tp += np.sum(np.prod(np.array([a[-1] for a in cp[5:]]), axis=0), axis=-1)
    degree_tp += np.sum(np.prod(np.array([a[-1] for a in cp[1:3]]), axis=0), axis=-1)
    secondary_msk = (np.argmax(y_true[1][0], axis=-1) != 0)
    secondary_total += sum(secondary_msk)
    secondary_tp += np.sum(np.prod(np.array([a[-1] for a in cp[1:3]]), axis=0)[secondary_msk], axis=-1)
    root_pred = find_root_from_output(y_pred)
    symb_pred = np.array([S2I[Q2S[QUALITY[np.argmax(q, axis=-1)]]] for q in y_pred[3][0]])
    # plot_coherence(np.argmax(y_pred[5], axis=-1)[0], root_pred, n_classes=12, sonata=TEST_INDICES[i])
    root_coherence += np.sum(root_pred == np.argmax(y_pred[5], axis=-1))
    symb_coherence += np.sum(symb_pred == np.argmax(y_pred[6], axis=-1))
    root_tp += np.sum(root_pred == np.argmax(y_true[5], axis=-1))
    symb_tp += np.sum(symb_pred == np.argmax(y_true[6], axis=-1))
acc = tp / total
degree_acc = degree_tp / total
secondary_acc = secondary_tp / secondary_total
func_acc = func_tp / total
name_acc = name_tp / total
print(f"accuracy for the different items:")
for f, a in zip(FEATURES, acc):
    print(f"{f:10} : {a:.4f}")
print(f'degree     : {degree_tp / total:.4f}')
print(f'secondary  : {secondary_tp / secondary_total:.4f}')
print(f'func root  : {root_tp / total:.4f}')
print(f'func symb  : {symb_tp / total:.4f}')
print(f"global functional : {func_acc:.4f}")
print(f"global symbolic   : {name_acc:.4f}")
print(f'root_coherence: {root_coherence / total:.4f}')
print(f'symb_coherence: {symb_coherence / total:.4f}')

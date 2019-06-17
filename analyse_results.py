import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.python.keras.models import load_model

from config import BATCH_SIZE, TEST_STEPS, TEST_INDICES, FEATURES, TICK_LABELS, CIRCLE_OF_FIFTH, TEST_TFRECORDS
from load_data import create_tfrecords_dataset
from utils import create_dezrann_annotations


def check_predictions(y_true, y_pred, index):
    return np.argmax(y_true[index], axis=-1) == np.argmax(y_pred[index], axis=-1)


def visualize_data(mode='probabilities'):
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
            ax = sns.heatmap(x, cmap=cmap, vmin=-1, vmax=2, xticklabels=TICK_LABELS[j])
            colorbar = ax.collections[0].colorbar
            colorbar.set_ticks([-5 / 8, 1 / 8, 7 / 8, 13 / 8])
            colorbar.set_ticklabels(['False Pos', 'True Neg', 'True Pos', 'False Neg'])
        else:
            x = b - a
            x = x.transpose()
            ax = sns.heatmap(x, cmap=cmap, center=0, vmin=-1, vmax=1, xticklabels=TICK_LABELS[j])
            colorbar = ax.collections[0].colorbar
            colorbar.set_ticks([-1, 0, +1])
            colorbar.set_ticklabels(['False Pos', 'True', 'False Neg'])

        ax.set(xlabel=FEATURES[j], ylabel='time',
               title=f"Sonata {TEST_INDICES[i]} - {FEATURES[j]}")
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()


test_data = create_tfrecords_dataset(TEST_TFRECORDS, BATCH_SIZE, shuffle_buffer=1)
test_data_iter = test_data.make_one_shot_iterator()
x, y = test_data_iter.get_next()

model_folder = os.path.join('logs', 'conv_gru_bass')
model = load_model(os.path.join(model_folder, 'conv_gru_bass.h5'))
model.summary()

# Retrieve the true labels
test_truth = []
with tf.Session() as sess:
    for i in range(TEST_STEPS):
        data = sess.run(y)
        test_truth.append(data)

# Predict new labels and view the difference
test_predict = []  # It will have shape: [pieces, features, (batch size, length of sonata, feature size)]
for i in range(TEST_STEPS):
    print(f"step {i + 1} out of {TEST_STEPS}")
    temp = model.predict(test_data.skip(i), steps=1, verbose=False)
    test_predict.append(temp)

    # visualize_data(mode='predictions')
    # visualize_data(mode='probabilities')

for i in range(TEST_STEPS):
    create_dezrann_annotations(test_truth[i], test_predict[i], n=TEST_INDICES[i], batch_size=BATCH_SIZE,
                               model_folder=model_folder)

# Calculate accuracy etc.
func_tp, symb_tp, total = 0, 0, 0
cp = [[] for _ in range(7)]  # dimensions will be: features, sonatas, frames
tp = np.zeros(7)
for i in range(TEST_STEPS):
    y_true, y_pred = test_truth[i], test_predict[i]
    total += y_true[0].shape[1]
    for j in range(7):
        for a in check_predictions(y_true, y_pred, j):  # this takes into account th
            cp[j].append(a)  # cp has shape (7, batch_size*(i+1), l) where l varies
    # can't vectorialize the next three lines because the length of each sonata l is different
    tp += np.sum(np.array([a[-1] for a in cp]), axis=-1)
    func_tp += np.sum(np.prod(np.array([a[-1] for a in cp[:5]]), axis=0), axis=-1)
    symb_tp += np.sum(np.prod(np.array([a[-1] for a in cp[5:]]), axis=0), axis=-1)
acc = tp / total
func_acc = func_tp / total
symb_acc = symb_tp / total
print(f"accuracy for the different items:")
for f, a in zip(FEATURES, acc):
    print(f"{f:10} : {a:.3f}")
print(f"global functional : {func_acc:.3f}")
print(f"global symbolic   : {symb_acc:.3f}")

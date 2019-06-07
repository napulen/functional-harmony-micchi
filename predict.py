import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt

from config import TEST_TFRECORDS, BATCH_SIZE, TEST_STEPS, TEST_INDICES, FEATURES, NOTES, QUALITY, SYMBOL
from load_data import create_tfrecords_dataset

TEST_STEPS = 1

test_data = create_tfrecords_dataset(TEST_TFRECORDS, BATCH_SIZE, shuffle_buffer=1)
test_data_iter = test_data.make_one_shot_iterator()
x, y = test_data_iter.get_next()


def check_predictions(y_true, y_pred, index):
    return np.argmax(y_true[index], axis=-1) == np.argmax(y_pred[index], axis=-1)


model = load_model('conv_lstm.h5')
model.summary()
test_predict = []  # It will have shape: [pieces, features, (batch size, length of sonata, feature size)]
for i in range(TEST_STEPS):
    print(f"step {i + 1} out of {TEST_STEPS}")
    temp = model.predict(test_data.skip(i), steps=1, verbose=False)
    test_predict.append(temp)

test_truth = []
with tf.Session() as sess:
    for i in range(TEST_STEPS):
        data = sess.run(y)
        test_truth.append(data)

circle_of_fifth_order = [8, 3, 10, 5, 0, 7, 2, 9, 4, 11, 6, 1]
circle_of_fifth_order += [x + 12 for x in circle_of_fifth_order]

notes_flat = NOTES.copy()
notes_flat[3] = 'E-'
notes_flat[8] = 'A-'
notes_flat[10] = 'B-'
tick_labels = [
    [(notes_flat + [n.lower() for n in notes_flat])[i] for i in circle_of_fifth_order],
    [str(x + 1) for x in range(7)] + [str(x + 1) + 'b' for x in range(7)] + [str(x + 1) + '#' for x in range(7)],
    [str(x + 1) for x in range(7)] + [str(x + 1) + 'b' for x in range(7)] + [str(x + 1) + '#' for x in range(7)],
    QUALITY,
    [str(x) for x in range(4)],
    NOTES,
    SYMBOL
]
for i in range(TEST_STEPS):
    for j in range(7):
        a = test_predict[i][j][0] if j > 0 else test_predict[i][j][0][:, circle_of_fifth_order]
        b = test_truth[i][j][0] if j > 0 else test_truth[i][j][0][:, circle_of_fifth_order]
        ax = sns.heatmap(b - a, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                         xticklabels=tick_labels[j])
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([-1, 0, +1])
        colorbar.set_ticklabels(['False Neg', 'True', 'False Pos'])

        ax.set(xlabel=FEATURES[j], ylabel='time',
               title=f"Sonata {TEST_INDICES[i // 12]}, transposed {i % 12 - 6} semitones - {FEATURES[j]}")
        plt.show()

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
print(f"accuracy for the different items: {acc}")
print(f"global accuracy with functional chord prediction: {func_acc}")
print(f"global accuracy with symbolic chord prediction: {symb_acc}")

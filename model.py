import time

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.backend import name_scope
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import Conv1D, Concatenate, MaxPooling1D, TimeDistributed, Dense, Lambda, \
    BatchNormalization, Masking, GRU, Bidirectional

from config import CLASSES_KEY, CLASSES_DEGREE, CLASSES_QUALITY, CLASSES_INVERSION, CLASSES_ROOT


class TimeOut(Callback):
    def __init__(self, t0, timeout):
        super().__init__()
        self.t0 = t0
        self.timeout = timeout  # time in minutes

    def on_train_batch_end(self, batch, logs=None):
        if time.time() - self.t0 > self.timeout * 60:  # 58 minutes
            print(f"\nReached {(time.time() - self.t0) / 60:.3f} minutes of training, stopping")
            self.model.stop_training = True


def DenseNetLayer(x, l, k, n=1):
    """
    Implementation of a DenseNetLayer
    :param x: input
    :param l: number of elementary blocks in the layer
    :param k: features generated at every block
    :param n: unique identifier of the DenseNetLayer
    :param training: passed to the batch normalization layers
    :return:
    """
    with name_scope(f"denseNet_{n}"):
        for _ in range(l):
            y = Conv1D(filters=4 * k, kernel_size=1, padding='same', data_format='channels_last', activation='relu')(x)
            y = BatchNormalization()(y)
            y = Conv1D(filters=k, kernel_size=32, padding='same', data_format='channels_last', activation='relu')(y)
            y = BatchNormalization()(y)
            x = Concatenate()([x, y])
    return x


def DilatedConvLayer(x, l, k):
    """
    Implementation of a DilatedConvolutionalLayer
    :param x: input
    :param l: number of levels in the layer
    :param k: number of filters
    :return:
    """
    with name_scope(f"dilatedConv"):
        for i in range(l):
            x = Conv1D(filters=k, kernel_size=3, padding='same', dilation_rate=3 ** i, data_format='channels_last',
                       activation='relu')(x)
    return x


def MultiTaskLayer(x, derive_root):
    o0 = TimeDistributed(Dense(CLASSES_KEY, activation='softmax'), name='key')(x)
    o1 = TimeDistributed(Dense(CLASSES_DEGREE, activation='softmax'), name='degree_1')(x)
    o2 = TimeDistributed(Dense(CLASSES_DEGREE, activation='softmax'), name='degree_2')(x)
    o3 = TimeDistributed(Dense(CLASSES_QUALITY, activation='softmax'), name='quality')(x)
    o4 = TimeDistributed(Dense(CLASSES_INVERSION, activation='softmax'), name='inversion')(x)
    if derive_root:
        o5 = Lambda(find_root_no_spelling, name='root')([o0, o1, o2])
    else:
        o5 = TimeDistributed(Dense(CLASSES_ROOT, activation='softmax'), name='root')(x)
    return [o0, o1, o2, o3, o4, o5]


def create_model(name, n, model_type, derive_root=False):
    """

    :param name:
    :param n: number of input features
    :param derive_root:
    :return:
    """
    if model_type not in ['conv_dil_reduced', 'conv_gru_reduced']:
        raise ValueError("model_type not supported, check its value")

    notes = Input(shape=(None, n), name="piano_roll_input")
    mask = Input(shape=(None, 1), name="mask_input")
    x = DenseNetLayer(notes, 4, 5, n=1)
    x = MaxPooling1D(2, 2, padding='same', data_format='channels_last')(x)
    x = DenseNetLayer(x, 4, 5, n=2)
    x = MaxPooling1D(2, 2, padding='same', data_format='channels_last')(x)

    if model_type == 'conv_dil_reduced':
        x = DilatedConvLayer(x, 4, 64)  # total context: 3**4 = 81 eight notes, typically 5 measures before and after

    # Super-ugly hack otherwise tensorflow can't save the model, see https://stackoverflow.com/a/55229794/5048010
    x = Lambda(lambda t: __import__('tensorflow').multiply(*t), name='apply_mask')((x, mask))
    x = Masking()(x)  # is this useless?

    if model_type == 'conv_gru_reduced':
        x = Bidirectional(GRU(64, return_sequences=True, dropout=0.3))(x)

    x = TimeDistributed(Dense(64, activation='tanh'))(x)
    y = MultiTaskLayer(x, derive_root)
    model = Model(inputs=[notes, mask], outputs=y, name=name)
    return model


def find_root_no_spelling(x):
    key, degree_den, degree_num = tf.argmax(x[0], axis=-1), tf.argmax(x[1], axis=-1), tf.argmax(
        x[2], axis=-1)

    deg2sem_maj = np.array([0, 2, 4, 5, 7, 9, 11], dtype=np.int64)
    deg2sem_min = np.array([0, 2, 3, 5, 7, 8, 10], dtype=np.int64)

    deg2sem = deg2sem_maj if key // 12 == 0 else deg2sem_min  # keys 0-11 are major, 12-23 minor
    n_den = tf.gather(deg2sem, degree_den % 7)  # (0-6 diatonic, 7-13 sharp, 14-20 flat)
    if degree_den // 7 == 1:  # raised root
        n_den += 1
    elif degree_den // 7 == 2:  # lowered root
        n_den -= 1
    n_num = tf.gather(deg2sem, degree_num % 7)
    if degree_num // 7 == 1:
        n_num += 1
    elif degree_num // 7 == 2:
        n_num -= 1
    # key % 12 finds the root regardless of major and minor, then both degrees are added, then sent back to 0-11
    # both degrees are added, yes: example: V/IV on C major.
    # primary degree = IV, secondary degree = V
    # in C, that corresponds to the dominant on the fourth degree: C -> F -> C again
    root_pred = (key % 12 + n_num + n_den) % 12
    return tf.one_hot(root_pred, depth=CLASSES_ROOT, axis=-1)

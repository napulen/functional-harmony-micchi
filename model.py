import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.backend import name_scope
from tensorflow.python.keras.layers import Conv1D, Concatenate, MaxPooling1D, TimeDistributed, Dense, Lambda

from config import CLASSES_KEY, CLASSES_DEGREE, CLASSES_QUALITY, CLASSES_INVERSION, CLASSES_ROOT


def DenseNetLayer(x, l, k, n=1):
    """
    Implementation of a DenseNetLayer
    :param x: input
    :param l: number of elementary blocks in the layer
    :param k: features generated at every block
    :param n: unique identifier of the DenseNetLayer
    :return:
    """
    with name_scope(f"denseNet_{n}"):
        for _ in range(l):
            y = Conv1D(filters=4 * k, kernel_size=1, padding='same', data_format='channels_last', activation='relu')(x)
            z = Conv1D(filters=k, kernel_size=32, padding='same', data_format='channels_last', activation='relu')(y)
            x = Concatenate()([x, z])
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


def MultiTaskLayer(x):
    o1 = TimeDistributed(Dense(CLASSES_KEY, activation='softmax'), name='key')(x)
    o2 = TimeDistributed(Dense(CLASSES_DEGREE, activation='softmax'), name='degree_1')(x)
    o3 = TimeDistributed(Dense(CLASSES_DEGREE, activation='softmax'), name='degree_2')(x)
    o4 = TimeDistributed(Dense(CLASSES_QUALITY, activation='softmax'), name='quality')(x)
    o5 = TimeDistributed(Dense(CLASSES_INVERSION, activation='softmax'), name='inversion')(x)
    # o6 = TimeDistributed(Dense(CLASSES_ROOT, activation='softmax'), name='root')(x)
    # return [o1, o2, o3, o4, o5, o6]
    return [o1, o2, o3, o4, o5]


def create_model(name, n):
    """

    :param name:
    :param n: number of input features
    :return:
    """
    notes = Input(shape=(None, n), name="piano_roll_input")
    x = DenseNetLayer(notes, 4, 5, n=1)
    x = MaxPooling1D(2, 2, padding='same', data_format='channels_last')(x)
    x = DenseNetLayer(x, 4, 5, n=2)
    x = MaxPooling1D(2, 2, padding='same', data_format='channels_last')(x)
    # x = Bidirectional(GRU(64, return_sequences=True, dropout=0.3))(x)
    x = DilatedConvLayer(x, 4, 64)  # total context: 3**4 = 81 eight notes, i.e., typically 5 measure before and after
    x = TimeDistributed(Dense(64, activation='tanh'))(x)
    y = MultiTaskLayer(x)
    y.append(Lambda(find_root, arguments=[y[0], y[1], y[2]]))  # Derive the root from all other information
    model = Model(inputs=notes, outputs=y, name=name)
    return model


# def find_root(x):
#     return tf.map_fn(_find_root, x)


def find_root(x):
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
    return tf.one_hot(root_pred, depth=CLASSES_ROOT)

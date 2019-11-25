import time

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.backend import name_scope
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import Conv1D, Concatenate, MaxPooling1D, TimeDistributed, Dense, Lambda, \
    BatchNormalization, Masking, GRU, Bidirectional, Activation

from config import INPUT_TYPE2INPUT_SHAPE, KEYS_SPELLING


class TimeOut(Callback):
    def __init__(self, t0, timeout):
        super().__init__()
        self.t0 = t0
        self.timeout = timeout  # time in minutes

    def on_train_batch_end(self, batch, logs=None):
        if time.time() - self.t0 > self.timeout * 60:  # 58 minutes
            print(f"\nReached {(time.time() - self.t0) / 60:.3f} minutes of training, stopping")
            self.model.stop_training = True


def DenseNetLayer(x, b, f, n=1):
    """
    Implementation of a DenseNetLayer
    :param x: input
    :param b: number of elementary blocks in the layer
    :param f: features generated at every block
    :param n: unique identifier of the DenseNetLayer
    :param training: passed to the batch normalization layers
    :return:
    """
    with name_scope(f"denseNet_{n}") as scope:
        for _ in range(b):
            y = BatchNormalization()(x)
            y = Conv1D(filters=4 * f, kernel_size=1, padding='same', data_format='channels_last')(y)
            y = Activation('relu')(y)
            y = BatchNormalization()(y)
            y = Conv1D(filters=f, kernel_size=8, padding='same', data_format='channels_last')(y)
            y = Activation('relu')(y)
            x = Concatenate()([x, y])
    return x


def PoolingLayer(x, k, s, n=1):
    """
    Implementation of a DenseNetLayer
    :param x: input
    :param k: feature maps before batch_norm
    :param s: stride for the Pooling Layer
    :param n: unique identifier of the Layer
    :return:
    """
    with name_scope(f"poolingLayer_{n}") as scope:
        y = BatchNormalization()(x)
        y = Conv1D(filters=k, kernel_size=1, padding='same', data_format='channels_last')(y)
        y = Activation('relu')(y)
        y = BatchNormalization()(y)
        y = MaxPooling1D(s, s, padding='same', data_format='channels_last')(y)
    return y


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
            x = Conv1D(filters=k, kernel_size=3, padding='same', dilation_rate=3 ** i, data_format='channels_last')(x)
            x = Activation('relu')(x)
    return x


def MultiTaskLayer(x, derive_root, input_type):
    classes_key = 30 if input_type.startswith('spelling') else 24  # Major keys: 0-11, Minor keys: 12-23
    classes_degree = 21  # 7 degrees * 3: regular, diminished, augmented
    classes_root = 35 if input_type.startswith('spelling') else 12  # the twelve notes without enharmonic duplicates
    classes_quality = 12  # ['M', 'm', 'd', 'a', 'M7', 'm7', 'D7', 'd7', 'h7', 'Gr+6', 'It+6', 'Fr+6']
    classes_inversion = 4  # root position, 1st, 2nd, and 3rd inversion (the last only for seventh chords)

    o0 = TimeDistributed(Dense(classes_key, activation='softmax'), name='key')(x)
    z = Concatenate()([x, o0])
    o1 = TimeDistributed(Dense(classes_degree, activation='softmax'), name='degree_1')(z)
    o2 = TimeDistributed(Dense(classes_degree, activation='softmax'), name='degree_2')(z)
    o3 = TimeDistributed(Dense(classes_quality, activation='softmax'), name='quality')(x)
    o4 = TimeDistributed(Dense(classes_inversion, activation='softmax'), name='inversion')(x)
    if derive_root and input_type.startswith('pitch'):
        o5 = Lambda(find_root_pitch, name='root')([o0, o1, o2])
    else:
        o5 = TimeDistributed(Dense(classes_root, activation='softmax'), name='root')(x)
    return [o0, o1, o2, o3, o4, o5]


def LocalMultiTaskLayer(x, input_type):
    classes_root = 35 if input_type.startswith('spelling') else 12  # the twelve notes without enharmonic duplicates
    classes_quality = 12  # ['M', 'm', 'd', 'a', 'M7', 'm7', 'D7', 'd7', 'h7', 'Gr+6', 'It+6', 'Fr+6']
    classes_inversion = 4  # root position, 1st, 2nd, and 3rd inversion (the last only for seventh chords)

    o_qlt = TimeDistributed(Dense(classes_quality, activation='softmax'), name='quality')(x)
    o_inv = TimeDistributed(Dense(classes_inversion, activation='softmax'), name='inversion')(x)
    o_roo = TimeDistributed(Dense(classes_root, activation='softmax'), name='root')(x)
    return [o_qlt, o_inv, o_roo]


def ProgressionMultiTaskLayer(x, input_type):
    classes_key = 30 if input_type.startswith('spelling') else 24  # Major keys: 0-11, Minor keys: 12-23
    classes_degree = 21  # 7 degrees * 3: regular, diminished, augmented

    o_key = TimeDistributed(Dense(classes_key, activation='softmax'), name='key')(x)
    z = Concatenate()([x, o_key])
    o_dg1 = TimeDistributed(Dense(classes_degree, activation='softmax'), name='degree_1')(z)
    o_dg2 = TimeDistributed(Dense(classes_degree, activation='softmax'), name='degree_2')(z)
    return [o_key, o_dg1, o_dg2]


def create_model(name, model_type, input_type, derive_root=False):
    """

    :param name:
    :param model_type:
    :param input_type:
    :param derive_root:
    :return:
    """
    allowed_models = ['conv_dil', 'conv_gru', 'gru']
    allowed_models += ['conv_dil_local', 'conv_gru_local']

    if model_type not in allowed_models:
        raise ValueError("model_type not supported, check its value")

    n = INPUT_TYPE2INPUT_SHAPE[input_type]
    notes = Input(shape=(None, n), name="piano_roll_input")
    mask = Input(shape=(None, 1), name="mask_input")

    if 'conv' in model_type:
        x = DenseNetLayer(notes, b=4, f=8, n=1)
        x = PoolingLayer(x, 32, 2, n=1)
        x = DenseNetLayer(x, 4, 5, n=2)
        x = PoolingLayer(x, 48, 2, n=1)
        if 'local' in model_type:
            y1 = LocalMultiTaskLayer(x, input_type)
    else:
        x = MaxPooling1D(4, 4, padding='same', data_format='channels_last')(notes)

    if 'dil' in model_type:
        x = DilatedConvLayer(x, 4, 64)  # total context: 3**4 = 81 eight notes, typically 5 measures before and after

    # Super-ugly hack otherwise tensorflow can't save the model, see https://stackoverflow.com/a/55229794/5048010
    x = Lambda(lambda t: __import__('tensorflow').multiply(*t), name='apply_mask')((x, mask))
    x = Masking()(x)  # is this useless?

    if 'gru' in model_type:
        x = Bidirectional(GRU(64, return_sequences=True, dropout=0.3))(x)

    x = TimeDistributed(Dense(64, activation='tanh'))(x)

    if 'local' in model_type:
        x = Concatenate()([x, y1])
        y2 = ProgressionMultiTaskLayer(x, input_type)
        y = y1 + y2
    else:
        y = MultiTaskLayer(x, derive_root, input_type)

    model = Model(inputs=[notes, mask], outputs=y, name=name)
    return model


def find_root_pitch(x):
    key, degree_den, degree_num = tf.argmax(x[0], axis=-1), tf.argmax(x[1], axis=-1), tf.argmax(x[2], axis=-1)

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
    return tf.one_hot(root_pred, depth=12, axis=-1)

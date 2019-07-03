from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.backend import name_scope
from tensorflow.python.keras.layers import Conv1D, Concatenate, MaxPooling1D, TimeDistributed, Dense

from config import N_PITCHES, CLASSES_BASS, CLASSES_KEY, CLASSES_DEGREE, CLASSES_QUALITY, CLASSES_INVERSION, \
    CLASSES_ROOT, CLASSES_SYMBOL


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
    o6 = TimeDistributed(Dense(CLASSES_ROOT, activation='softmax'), name='root')(x)
    o7 = TimeDistributed(Dense(CLASSES_SYMBOL, activation='softmax'), name='symbol')(x)
    return [o1, o2, o3, o4, o5, o6, o7]


def create_model():
    notes = Input(shape=(None, N_PITCHES), name="piano_roll_input")
    bass = Input(shape=(None, CLASSES_BASS), name="bass_input")
    x = DenseNetLayer(notes, 4, 12, n=1)
    x = MaxPooling1D(2, 2, padding='same', data_format='channels_last')(x)
    x = DenseNetLayer(x, 4, 12, n=2)
    x = MaxPooling1D(2, 2, padding='same', data_format='channels_last')(x)
    # x = Bidirectional(GRU(256, return_sequences=True, dropout=0.3))(x)
    x = DilatedConvLayer(x, 6, 256)
    # x = Concatenate(name=f"concatenate_bass")([x, bass])
    x = TimeDistributed(Dense(256, activation='tanh'))(x)
    y = MultiTaskLayer(x)
    model = Model(inputs=[notes, bass], outputs=y)
    return model

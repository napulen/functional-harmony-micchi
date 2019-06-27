import os

from tensorflow import enable_eager_execution
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.backend import name_scope
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.python.keras.layers import Bidirectional, Dense, Conv1D, Concatenate, TimeDistributed, GRU
from tensorflow.python.keras.layers.pooling import MaxPooling1D

from config import TRAIN_TFRECORDS, SHUFFLE_BUFFER, BATCH_SIZE, CLASSES_ROOT, \
    VALID_TFRECORDS, EPOCHS, STEPS_PER_EPOCH, N_PITCHES, CLASSES_KEY, CLASSES_DEGREE, CLASSES_INVERSION, \
    CLASSES_QUALITY, CLASSES_SYMBOL, VALIDATION_STEPS, CLASSES_BASS
from load_data import create_tfrecords_dataset
from utils import visualize_data

exploratory = False
if exploratory:
    enable_eager_execution()

train_data = create_tfrecords_dataset(TRAIN_TFRECORDS, BATCH_SIZE, SHUFFLE_BUFFER)
valid_data = create_tfrecords_dataset(VALID_TFRECORDS, BATCH_SIZE, SHUFFLE_BUFFER)

if exploratory:
    visualize_data(train_data)

model_folder = os.path.join('logs', 'conv_dil')


def DenseNetLayer(x, l, k, n=1):
    """
    Implementation of a DenseNetLayer
    :param x:
    :param l:
    :param k:
    :return:
    """
    with name_scope(f"denseNet_{n}"):
        for _ in range(l):
            y = Conv1D(filters=4 * k, kernel_size=1, padding='same', data_format='channels_last', activation='relu')(x)
            z = Conv1D(filters=k, kernel_size=32, padding='same', data_format='channels_last', activation='relu')(y)
            x = Concatenate()([x, z])
    return x


def DilatedConvLayer(x, l, k):
    with name_scope(f"dilatedConv"):
        for _ in range(l):
            x = Conv1D(filters=k, kernel_size=3, padding='same', dilation_rate=3 ** l, data_format='channels_last',
                       activation='relu')(x)
    return x


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
o1 = TimeDistributed(Dense(CLASSES_KEY, activation='softmax'), name='key')(x)
o2 = TimeDistributed(Dense(CLASSES_DEGREE, activation='softmax'), name='degree_1')(x)
o3 = TimeDistributed(Dense(CLASSES_DEGREE, activation='softmax'), name='degree_2')(x)
o4 = TimeDistributed(Dense(CLASSES_QUALITY, activation='softmax'), name='quality')(x)
o5 = TimeDistributed(Dense(CLASSES_INVERSION, activation='softmax'), name='inversion')(x)
o6 = TimeDistributed(Dense(CLASSES_ROOT, activation='softmax'), name='root')(x)
o7 = TimeDistributed(Dense(CLASSES_SYMBOL, activation='softmax'), name='symbol')(x)

model = Model(inputs=[notes, bass], outputs=[o1, o2, o3, o4, o5, o6, o7])
model.summary()

callbacks = [
    EarlyStopping(patience=3),
    TensorBoard(log_dir=model_folder)
]

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_data, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_data=valid_data,
          validation_steps=VALIDATION_STEPS, callbacks=callbacks)

model.save(os.path.join(model_folder, 'conv_dil.h5'))

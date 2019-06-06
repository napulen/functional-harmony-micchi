from tensorflow import enable_eager_execution
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.python.keras.layers import Bidirectional, LSTM, Dense, Conv1D, Concatenate, TimeDistributed
from tensorflow.python.keras.layers.pooling import Pooling1D, MaxPooling1D
from tensorflow.python.ops.nn_ops import max_pool

from config import TRAIN_TFRECORDS, SHUFFLE_BUFFER, BATCH_SIZE, TEST_TFRECORDS, CLASSES_ROOT, \
    VALID_TFRECORDS, EPOCHS, STEPS_PER_EPOCH, WSIZE, N_PITCHES, CLASSES_KEY, CLASSES_DEGREE, CLASSES_INVERSION, \
    CLASSES_QUALITY, CLASSES_SYMBOL, VALIDATION_STEPS
from load_data import create_tfrecords_iterator
from utils import visualize_data

exploratory = False
if exploratory:
    enable_eager_execution()

train_data = create_tfrecords_iterator(TRAIN_TFRECORDS, BATCH_SIZE, SHUFFLE_BUFFER)
valid_data = create_tfrecords_iterator(VALID_TFRECORDS, BATCH_SIZE, SHUFFLE_BUFFER)

if exploratory:
    visualize_data(train_data)


def DenseNetLayer(x, l, k):
    """
    Implementation of a DenseNetLayer
    :param x:
    :param l:
    :param k:
    :return:
    """
    for _ in range(l):
        y = Conv1D(filters=4 * k, kernel_size=1, padding='same', data_format='channels_last')(x)
        z = Conv1D(filters=k, kernel_size=32, padding='same', data_format='channels_last')(y)
        x = Concatenate()([x, z])
    return x


notes = Input(shape=(None, N_PITCHES))
x = DenseNetLayer(notes, 4, 12)
x = MaxPooling1D(2, 2, padding='same', data_format='channels_last')(x)
x = DenseNetLayer(x, 4, 12)
x = MaxPooling1D(2, 2, padding='same', data_format='channels_last')(x)
x = Bidirectional(LSTM(256, return_sequences=True))(x)
# x = TimeDistributed(Dense(256))(x)
o1 = TimeDistributed(Dense(CLASSES_KEY, activation='softmax', name='key'))(x)
o2 = TimeDistributed(Dense(CLASSES_DEGREE, activation='softmax', name='degree_1'))(x)
o3 = TimeDistributed(Dense(CLASSES_DEGREE, activation='softmax', name='degree_2'))(x)
o4 = TimeDistributed(Dense(CLASSES_QUALITY, activation='softmax', name='quality'))(x)
o5 = TimeDistributed(Dense(CLASSES_INVERSION, activation='softmax', name='inversion'))(x)
o6 = TimeDistributed(Dense(CLASSES_ROOT, activation='softmax', name='root'))(x)
o7 = TimeDistributed(Dense(CLASSES_SYMBOL, activation='softmax', name='symbol'))(x)

model = Model(inputs=notes, outputs=[o1, o2, o3, o4, o5, o6, o7])

callbacks = [
    EarlyStopping(patience=3),
    TensorBoard()
]

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_data, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_data=valid_data,
          validation_steps=VALIDATION_STEPS, callbacks=callbacks)

model.save('my_model.h5')

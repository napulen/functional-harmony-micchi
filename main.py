from tensorflow.python.keras import Sequential, Input
from tensorflow.python.keras.layers import Bidirectional, LSTM, Dense

import numpy as np
from config import TRAIN_TFRECORDS, SHUFFLE_BUFFER, BATCH_SIZE, TEST_TFRECORDS, CLASSES_TOTAL
from load_data import create_tfrecords_iterator
import matplotlib.pyplot as plt
import seaborn as sns

x_train, y_train, info_train = create_tfrecords_iterator(TRAIN_TFRECORDS, BATCH_SIZE, SHUFFLE_BUFFER)
x_test, y_test, info_test = create_tfrecords_iterator(TEST_TFRECORDS, BATCH_SIZE, SHUFFLE_BUFFER)

# a0 = Input(shape=(100,), dtype='int32', name='main_input')
# a = Bidirectional(LSTM(10, return_sequences=True), input_shape=(5, 10))(main_input)

# model = Sequential()
# model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(5, 10)))
# model.add(Dense(CLASSES_TOTAL))
# model.compile(loss='categorical_crossentropy', optimizer='adam')

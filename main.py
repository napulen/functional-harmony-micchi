from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.python.keras.layers import Bidirectional, LSTM, Dense

from config import TRAIN_TFRECORDS, SHUFFLE_BUFFER, BATCH_SIZE, TEST_TFRECORDS, CLASSES_ROOT, \
    VALID_TFRECORDS, EPOCHS, STEPS_PER_EPOCH, WSIZE, N_PITCHES, CLASSES_KEY, CLASSES_DEGREE, CLASSES_INVERSION, \
    CLASSES_QUALITY, CLASSES_SYMBOL, VALIDATION_STEPS
from load_data import create_tfrecords_iterator

train_data = create_tfrecords_iterator(TRAIN_TFRECORDS, BATCH_SIZE, SHUFFLE_BUFFER)
valid_data = create_tfrecords_iterator(VALID_TFRECORDS, BATCH_SIZE, SHUFFLE_BUFFER)

notes = Input(shape=(WSIZE, N_PITCHES))
x = Bidirectional(LSTM(256))(notes)
x = Dense(256)(x)
o1 = Dense(CLASSES_KEY, activation='softmax', name='key')(x)
o2 = Dense(CLASSES_DEGREE, activation='softmax', name='degree_1')(x)
o3 = Dense(CLASSES_DEGREE, activation='softmax', name='degree_2')(x)
o4 = Dense(CLASSES_QUALITY, activation='softmax', name='quality')(x)
o5 = Dense(CLASSES_INVERSION, activation='softmax', name='inversion')(x)
o6 = Dense(CLASSES_ROOT, activation='softmax', name='root')(x)
o7 = Dense(CLASSES_SYMBOL, activation='softmax', name='symbol')(x)

model = Model(inputs=notes, outputs=[o1, o2, o3, o4, o5, o6, o7])

callbacks = [
    EarlyStopping(patience=3),
    TensorBoard()
]

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_data, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_data=valid_data,
          validation_steps=VALIDATION_STEPS, callbacks=callbacks)

model.save('my_model.h5')

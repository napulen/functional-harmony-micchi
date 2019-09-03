import os

from tensorflow import enable_eager_execution
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard

from config import TRAIN_TFRECORDS, SHUFFLE_BUFFER, BATCH_SIZE, VALID_TFRECORDS, EPOCHS, STEPS_PER_EPOCH, \
    VALIDATION_STEPS
from load_data import create_tfrecords_dataset
from model import create_model
from utils import visualize_data

exploratory = False
if exploratory:
    enable_eager_execution()

train_data = create_tfrecords_dataset(TRAIN_TFRECORDS, BATCH_SIZE, SHUFFLE_BUFFER)
valid_data = create_tfrecords_dataset(VALID_TFRECORDS, BATCH_SIZE, SHUFFLE_BUFFER)

if exploratory:
    visualize_data(train_data)
model_name = 'conv_gru_pitch_class'
model_folder = os.path.join('logs', model_name)

model = create_model(name=model_name)
model.summary()

callbacks = [
    EarlyStopping(patience=3),
    TensorBoard(log_dir=model_folder)
]

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_data, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_data=valid_data,
          validation_steps=VALIDATION_STEPS, callbacks=callbacks)

model.save(os.path.join(model_folder, model_name + '.h5'))

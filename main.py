import os

from tensorflow import enable_eager_execution
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard

from config import TRAIN_TFRECORDS, SHUFFLE_BUFFER, BATCH_SIZE, VALID_TFRECORDS, EPOCHS, STEPS_PER_EPOCH, \
    VALID_STEPS, N_PITCHES
from load_data import create_tfrecords_dataset
from model import create_model
from utils import visualize_data

exploratory = False
if exploratory:
    enable_eager_execution()

# mode = 'pitch_class_beat_strength'
# mode = 'pitch_class'
mode = 'pitch_class_weighted_loss'
# mode = 'midi_number'
mode2input_shape = {
    'pitch_class': 24,
    'pitch_class_weighted_loss': 24,
    'pitch_class_beat_strength': 27,
    'midi_number': N_PITCHES
}
n = mode2input_shape[mode]

train_data = create_tfrecords_dataset(TRAIN_TFRECORDS, BATCH_SIZE, SHUFFLE_BUFFER, n)
valid_data = create_tfrecords_dataset(VALID_TFRECORDS, BATCH_SIZE, SHUFFLE_BUFFER, n)

if exploratory:
    visualize_data(train_data)

model_name = 'conv_dil_reduced_' + mode
model_folder = os.path.join('logs', model_name)

model = create_model(model_name, n)
model.summary()

callbacks = [
    EarlyStopping(patience=3),
    TensorBoard(log_dir=model_folder)
]
weights = [1., 0.5, 1., 1., 0.5, 2.]  # [y_key, y_dg1, y_dg2, y_qlt, y_inv, y_roo]
# weights = [1., 1. 1., 1., 1., 1.]  # [y_key, y_dg1, y_dg2, y_qlt, y_inv, y_roo]
model.compile(loss='categorical_crossentropy', loss_weights=weights, optimizer='adam', metrics=['accuracy'])

model.fit(train_data, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_data=valid_data,
          validation_steps=VALID_STEPS, callbacks=callbacks)

model.save(os.path.join(model_folder, model_name + '.h5'))

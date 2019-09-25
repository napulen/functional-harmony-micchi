import os
import time

from tensorflow import enable_eager_execution
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard

from config import TRAIN_TFRECORDS, SHUFFLE_BUFFER, BATCH_SIZE, VALID_TFRECORDS, EPOCHS, STEPS_PER_EPOCH, \
    VALID_STEPS, MODE, MODE2INPUT_SHAPE
from load_data import create_tfrecords_dataset
from model import create_model, TimeOut
from utils import visualize_data

exploratory = False
# exploratory = True
if exploratory:
    enable_eager_execution()
    model_name = 'temp'
else:
    i = 0
    model_name = f'conv_dil_reduced_{MODE}_{i}'
    while model_name in os.listdir('logs'):
        i += 1
        model_name = f'conv_dil_reduced_{MODE}_{i}'

model_folder = os.path.join('logs', model_name)
os.makedirs(model_folder)

n = MODE2INPUT_SHAPE[MODE]

train_data = create_tfrecords_dataset(TRAIN_TFRECORDS, BATCH_SIZE, SHUFFLE_BUFFER, n)
valid_data = create_tfrecords_dataset(VALID_TFRECORDS, 7, 1, n)

if exploratory:
    visualize_data(train_data)

model = create_model(model_name, n, derive_root=False)
model.summary()
print(model_name)

callbacks = [
    EarlyStopping(patience=3),
    TensorBoard(log_dir=model_folder),
    TimeOut(t0=time.time(), timeout=58),
]
# weights = [1., 0.5, 1., 1., 0.5, 2.]  # [y_key, y_dg1, y_dg2, y_qlt, y_inv, y_roo]
weights = [1., 1., 1., 1., 1., 1.]  # [y_key, y_dg1, y_dg2, y_qlt, y_inv, y_roo]
model.compile(loss='categorical_crossentropy', loss_weights=weights, optimizer='adam', metrics=['accuracy'])

model.fit(train_data, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_data=valid_data,
          validation_steps=1, callbacks=callbacks)

model.save(os.path.join(model_folder, model_name + '.h5'))

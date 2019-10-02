import os
import time

import seaborn as sns
from matplotlib import pyplot as plt
from tensorflow import enable_eager_execution
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

from config import TRAIN_TFRECORDS, SHUFFLE_BUFFER, BATCH_SIZE, VALID_TFRECORDS, EPOCHS, STEPS_PER_EPOCH, \
    MODE, MODE2INPUT_SHAPE
from load_data import create_tfrecords_dataset
from model import create_model, TimeOut


def visualize_data(data):
    for x, y in data:
        prs, masks = x
        for pr, mask in zip(prs, masks):
            sns.heatmap(pr)
            plt.show()
            plt.plot(mask.numpy())
            plt.show()
    return


def setup_paths(exploratory, model_type):
    if exploratory:
        enable_eager_execution()
        name = 'temp'
    else:
        i = 0
        name = '_'.join([model_type, MODE, str(i)])
        while name in os.listdir('logs'):
            i += 1
            name = '_'.join([model_type, MODE, str(i)])

    folder = os.path.join('logs', name)
    os.makedirs(folder)

    return folder, name


timeout = None
exploratory = False
model_type = 'conv_dil_reduced'
if __name__ == '__main__':
    model_folder, model_name = setup_paths(exploratory, model_type)
    model_path = os.path.join(model_folder, model_name + '.h5')
    n = MODE2INPUT_SHAPE[MODE]
    train_data = create_tfrecords_dataset(TRAIN_TFRECORDS, BATCH_SIZE, SHUFFLE_BUFFER, n)
    valid_data = create_tfrecords_dataset(VALID_TFRECORDS, 7, 1, n)
    # visualize_data(train_data)

    model = create_model(model_name, n, model_type=model_type, derive_root=False)
    model.summary()
    print(model_name)

    callbacks = [
        EarlyStopping(patience=3),
        TensorBoard(log_dir=model_folder),
        ModelCheckpoint(filepath=model_path, save_best_only=True)
    ]
    if timeout is not None:
        t0 = time.time()
        callbacks.append(TimeOut(t0=t0, timeout=timeout))

    # weights = [1., 0.5, 1., 1., 0.5, 2.]  # [y_key, y_dg1, y_dg2, y_qlt, y_inv, y_roo]
    weights = [1., 1., 1., 1., 1., 1.]  # [y_key, y_dg1, y_dg2, y_qlt, y_inv, y_roo]
    model.compile(loss='categorical_crossentropy', loss_weights=weights, optimizer='adam', metrics=['accuracy'])

    model.fit(train_data, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_data=valid_data,
              validation_steps=1, callbacks=callbacks)

    # model.save(model_path)

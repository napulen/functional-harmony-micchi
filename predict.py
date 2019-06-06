from tensorflow.python.keras.models import load_model

from config import TEST_TFRECORDS, BATCH_SIZE, SHUFFLE_BUFFER, TEST_STEPS
from load_data import create_tfrecords_iterator

test_data = create_tfrecords_iterator(TEST_TFRECORDS, BATCH_SIZE, SHUFFLE_BUFFER)

model = load_model('my_model.h5')
test_predict = model.predict(test_data, steps=TEST_STEPS, verbose=True)

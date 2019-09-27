import tensorflow as tf

from config import CLASSES_ROOT, CLASSES_INVERSION, CLASSES_QUALITY, CLASSES_DEGREE, \
    CLASSES_KEY


def _parse_function(proto, n):
    # Parse the input tf.Example proto using the dictionary defined in preprocessing_main.
    feature = {
        'name': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        'transposition': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'piano_roll': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'label_key': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'label_degree_primary': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'label_degree_secondary': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'label_quality': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'label_inversion': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'label_root': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    }

    parsed_features = tf.io.parse_single_example(proto, feature)
    filename = parsed_features['name']
    transposition = parsed_features['transposition']
    piano_roll = tf.transpose(tf.reshape(parsed_features['piano_roll'], (n, -1)))
    mask = tf.ones([tf.shape(parsed_features['label_key'])[0], 1], tf.bool)  # whatever label would work
    y_key = tf.one_hot(parsed_features['label_key'], depth=CLASSES_KEY)
    y_dg1 = tf.one_hot(parsed_features['label_degree_primary'], depth=CLASSES_DEGREE)
    y_dg2 = tf.one_hot(parsed_features['label_degree_secondary'], depth=CLASSES_DEGREE)
    y_qlt = tf.one_hot(parsed_features['label_quality'], depth=CLASSES_QUALITY)
    y_inv = tf.one_hot(parsed_features['label_inversion'], depth=CLASSES_INVERSION)
    y_roo = tf.one_hot(parsed_features['label_root'], depth=CLASSES_ROOT)
    return (piano_roll, mask, filename, transposition), tuple([y_key, y_dg1, y_dg2, y_qlt, y_inv, y_roo])


def create_tfrecords_dataset(input_path, batch_size, shuffle_buffer, n):
    """

    :param input_path:
    :param batch_size:
    :param shuffle_buffer:
    :param n: number of features
    :return:
    """

    def _parse(proto):
        return _parse_function(proto, n)

    ## They pad separately for each feature!
    padded_shapes = (
        (
            tf.TensorShape([None, n]),  # this None might be different from the next one
            tf.TensorShape([None, 1]),
            tf.TensorShape([None, ]),
            tf.TensorShape([None, ]),
        ),
        (
            tf.TensorShape([None, CLASSES_KEY]),
            tf.TensorShape([None, CLASSES_DEGREE]),
            tf.TensorShape([None, CLASSES_DEGREE]),
            tf.TensorShape([None, CLASSES_QUALITY]),
            tf.TensorShape([None, CLASSES_INVERSION]),
            tf.TensorShape([None, CLASSES_ROOT]),
        )
    )

    return tf.data.TFRecordDataset(input_path).map(_parse, num_parallel_calls=16).shuffle(
        shuffle_buffer).repeat().padded_batch(batch_size, padded_shapes).prefetch(2)

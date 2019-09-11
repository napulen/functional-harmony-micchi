import tensorflow as tf

from config import N_PITCHES, CLASSES_ROOT, CLASSES_INVERSION, CLASSES_QUALITY, CLASSES_DEGREE, \
    CLASSES_KEY


def _parse_function(proto, n):
    # Parse the input tf.Example proto using the dictionary above.
    feature = {
        'piano_roll': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'label_key': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'label_degree_primary': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'label_degree_secondary': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'label_quality': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'label_inversion': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'label_root': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'sonata': tf.io.FixedLenFeature([], tf.int64),
        'transposed': tf.io.FixedLenFeature([], tf.int64),
    }

    parsed_features = tf.io.parse_single_example(proto, feature)
    # piano_roll = tf.transpose(tf.reshape(parsed_features['piano_roll'], (N_PITCHES, -1)))
    # piano_roll = tf.transpose(tf.reshape(parsed_features['piano_roll'], (24, -1)))
    piano_roll = tf.transpose(tf.reshape(parsed_features['piano_roll'], (n, -1)))
    y_key = tf.one_hot(parsed_features['label_key'], depth=CLASSES_KEY)
    y_dg1 = tf.one_hot(parsed_features['label_degree_primary'], depth=CLASSES_DEGREE)
    y_dg2 = tf.one_hot(parsed_features['label_degree_secondary'], depth=CLASSES_DEGREE)
    y_qlt = tf.one_hot(parsed_features['label_quality'], depth=CLASSES_QUALITY)
    y_inv = tf.one_hot(parsed_features['label_inversion'], depth=CLASSES_INVERSION)
    y_roo = tf.one_hot(parsed_features['label_root'], depth=CLASSES_ROOT)
    sonata = parsed_features['sonata']
    transposed = parsed_features['transposed']
    return piano_roll, tuple([y_key, y_dg1, y_dg2, y_qlt, y_inv, y_roo])
    # return x, tuple([y_key, y_dg1, y_dg2, y_qlt, y_inv, y_roo, y_sym, sonata, transposed])


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

    dataset = tf.data.TFRecordDataset(input_path).map(_parse, num_parallel_calls=16).shuffle(
        shuffle_buffer).repeat().batch(batch_size).prefetch(2)
    return dataset

import tensorflow as tf

from config import CHUNK_SIZE, INPUT_TYPE2INPUT_SHAPE


def _parse_function(proto, n, classes_key, classes_degree, classes_quality, classes_inversion, classes_root):
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
    piano_roll = tf.transpose(tf.reshape(parsed_features['piano_roll'], (n, -1)))  # NB transpose!: final shape (-1, n)
    mask = tf.ones([tf.shape(parsed_features['label_key'])[0], 1], tf.bool)  # whatever label would work
    y_key = tf.one_hot(parsed_features['label_key'], depth=classes_key)
    y_dg1 = tf.one_hot(parsed_features['label_degree_primary'], depth=classes_degree)
    y_dg2 = tf.one_hot(parsed_features['label_degree_secondary'], depth=classes_degree)
    y_qlt = tf.one_hot(parsed_features['label_quality'], depth=classes_quality)
    y_inv = tf.one_hot(parsed_features['label_inversion'], depth=classes_inversion)
    y_roo = tf.one_hot(parsed_features['label_root'], depth=classes_root)
    return (piano_roll, mask, filename, transposition), tuple([y_key, y_dg1, y_dg2, y_qlt, y_inv, y_roo])


def create_tfrecords_dataset(input_path, batch_size, shuffle_buffer, input_type):
    """

    :param input_path:
    :param batch_size:
    :param shuffle_buffer:
    :param n: number of features
    :param input_type: like 'spelling_bass_cut', for example
    :return:
    """

    def _parse(proto):
        return _parse_function(proto, n, classes_key, classes_degree, classes_quality, classes_inversion, classes_root)

    n = INPUT_TYPE2INPUT_SHAPE[input_type]
    classes_key = 55 if input_type.startswith('spelling') else 24  # Major keys: 0-11, Minor keys: 12-23
    classes_degree = 21  # 7 degrees * 3: regular, diminished, augmented
    classes_root = 35 if input_type.startswith('spelling') else 12  # the twelve notes without enharmonic duplicates
    classes_quality = 12  # ['M', 'm', 'd', 'a', 'M7', 'm7', 'D7', 'd7', 'h7', 'Gr+6', 'It+6', 'Fr+6']
    classes_inversion = 4  # root position, 1st, 2nd, and 3rd inversion (the last only for seventh chords)

    ## They pad separately for each feature!
    if input_type.endswith('cut'):
        pad_notes = 4 * CHUNK_SIZE
        pad_chords = CHUNK_SIZE
    else:
        pad_notes = None  # get to the max of each batch, where this value is calculated separately for each feature
        pad_chords = None
    padded_shapes = (
        (
            tf.TensorShape([pad_notes, n]),
            tf.TensorShape([pad_chords, 1]),
            tf.TensorShape([None, ]),
            tf.TensorShape([None, ]),
        ),
        (
            tf.TensorShape([pad_chords, classes_key]),
            tf.TensorShape([pad_chords, classes_degree]),
            tf.TensorShape([pad_chords, classes_degree]),
            tf.TensorShape([pad_chords, classes_quality]),
            tf.TensorShape([pad_chords, classes_inversion]),
            tf.TensorShape([pad_chords, classes_root]),
        )
    )

    return tf.data.TFRecordDataset(input_path).map(_parse, num_parallel_calls=16).shuffle(
        shuffle_buffer).repeat().padded_batch(batch_size, padded_shapes).prefetch(2)



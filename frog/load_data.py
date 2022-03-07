"""
Utils for loading the tfrecords data.
"""

import tensorflow as tf

from frog import CHUNK_SIZE, INPUT_FPC, INPUT_TYPE2INPUT_SHAPE, OUTPUT_FPC
from frog.label_codec import LabelCodec


def _parse_function(proto, n, lc):
    """Parse the input tf.Example proto using the dictionary defined during preprocessing"""
    feature_description = {
        "name": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        "transposition": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "start": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "piano_roll": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        "structure": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    }
    for f in lc.output_features:
        feature_description[f] = tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)

    parsed_features = tf.io.parse_single_example(proto, feature_description)
    filename = parsed_features["name"]
    transposition = parsed_features["transposition"]
    start = parsed_features["start"]
    piano_roll = tf.reshape(parsed_features["piano_roll"], (-1, n))
    structure = tf.reshape(parsed_features["structure"], (-1, 2))
    # Get the mask by checking the length of the first output feature
    mask = tf.ones([tf.shape(parsed_features[lc.output_features[0]])[0]])
    x = (piano_roll, structure, mask)
    y = tuple([tf.one_hot(parsed_features[f], depth=lc.output_size[f]) for f in lc.output_features])
    meta = (filename, transposition, start)
    return x, y, meta
    # return x, y  # we discard meta for the time being


def load_tfrecords_dataset(
    input_path, compression, batch_size, shuffle_buffer, input_type, output_mode
):
    """
    :param input_path:
    :param compression:
    :param batch_size:
    :param shuffle_buffer:
    :param input_type: like 'spelling_bass', for example
    :param output_mode:
    :return:
    """
    spelling, octaves = input_type.split("_")
    assert spelling in ["spelling", "pitch"]
    assert octaves in ["complete", "bass", "class"]
    lc = LabelCodec(spelling=spelling == "spelling", mode=output_mode, strict=False)

    def _parse(proto):
        return _parse_function(proto, n, lc)

    n = INPUT_TYPE2INPUT_SHAPE[input_type]

    pad_notes = CHUNK_SIZE * INPUT_FPC
    pad_chords = CHUNK_SIZE * OUTPUT_FPC
    padded_shapes = (
        (
            tf.TensorShape([pad_notes, n]),
            tf.TensorShape([pad_notes, 2]),
            tf.TensorShape([pad_chords]),
        ),
        tuple([tf.TensorShape([pad_chords, v]) for _k, v in lc.output_size.items()]),
        (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None])),  # -> meta
    )
    return (
        tf.data.TFRecordDataset(input_path, compression_type=compression)
        .map(_parse, num_parallel_calls=16)
        .shuffle(shuffle_buffer)
        .padded_batch(batch_size, padded_shapes)
        .prefetch(2)
    )

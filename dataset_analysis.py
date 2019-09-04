import numpy as np
import tensorflow as tf

from config import FPQ, TRAIN_TFRECORDS, VALID_TFRECORDS
from preprocessing import load_score


def count_records(tfrecord):
    """ Count the number of lines in a tfrecord file. This is useful to establish 'steps_per_epoch' when training """
    c = 0
    if tf.__version__[0] == '2':
        for _ in tf.data.TFRecordDataset(tfrecord):
            c += 1
    else:
        for _ in tf.io.tf_record_iterator(tfrecord):
            c += 1
    return c


def find_pitch_extremes():
    """
    Find the highest and lowest note in the piano_rolls, including transposition ranging from 6 down to 5 up.

    :return:
    """
    min_note, max_note = 128, 0  # they are inverted on purpose! they are initial conditions that will certainly be updated
    for i in range(1, 33):
        piano_roll, t0 = load_score(i, FPQ)
        min_i = np.where(np.max(piano_roll, axis=-1) == 1)[0][0] - 6  # -6 because we want room for transposing
        max_i = np.where(np.max(piano_roll, axis=-1) == 1)[0][-1] + 5  # +5 because we want room for transposing
        min_note = min(min_note, min_i)
        max_note = max(max_note, max_i)
        print(f"Sonata {i}, pitches from {min_i} to {max_i}")
    return min_note, max_note


if __name__ == '__main__':
    pm, pM = find_pitch_extremes()
    print(f'The pitch classes needed, including transpositions, are from {pm} to {pM}')
    c = count_records(TRAIN_TFRECORDS)
    print(f'There is a total of {c} records in the train file')
    c = count_records(VALID_TFRECORDS)
    print(f'There is a total of {c} records in the validation file')

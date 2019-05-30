"""
create a tfrecord containing the following features:
x the piano roll input data, with shape [n_frames, pitches]
label_key the local key of the music
label_degree the chord degree with respect to the key, possibly fractional, e.g. V/V dominant of dominant
label_quality e.g. m, M, D7 for minor, major, dominant 7th etc.
label_inversion, from 0 to 3 depending on what note is at the basse
label_symbol, for example C7, d, etc.
"""

import numpy as np
import math
import xlrd
import os
import logging
import tensorflow as tf

from config import DATASET_FOLDER, TRAIN_INDICES, VALID_INDICES, TEST_INDICES, TRAIN_TFRECORDS, VALID_TFRECORDS, \
    TEST_TFRECORDS, HSIZE, WSIZE, FPQ, PITCH_LOW, PITCH_HIGH
from utils import NOTES, _find_chord_symbol, _encode_key, _encode_degree, _encode_quality, _encode_symbol

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HarmonicAnalysisError(Exception):
    """ Raised when the harmonic analysis associated to a certain time can't be found """
    pass


def load_notes(i, fpq, pitch_low=0, pitch_high=128):
    """
    Load notes in each piece, which is then represented as piano roll.
    :param pitch_low:
    :param pitch_high:
    :param i: which sonata to take (sonatas indexed from 1 to 32)
    :param fpq: frames per quarter note, default =  8 (that is, 32th note as 1 unit in piano roll)
    :return: pieces, tdeviation
    """

    dt = [('onset', 'float'), ('pitch', 'int'), ('mPitch', 'int'), ('duration', 'float'), ('staffNum', 'int'),
          ('measure', 'int')]  # datatype

    notes_file = os.path.join(DATASET_FOLDER, str(i).zfill(2), "notes.csv")
    notes = np.genfromtxt(notes_file, delimiter=',', dtype=dt)  # read notes from .csv file
    # length of the piece in piano roll frames, assuming the last note to stay was amongst the 20 last to be played
    length = math.ceil((max(notes[-20:]['onset'] + notes[-20:]['duration']) - notes[0]['onset']) * fpq)
    t0 = notes[0]['onset']
    piano_roll = np.zeros(shape=[128, length], dtype=np.int32)
    for note in notes:
        pitch = note['pitch']
        start = int(round((note['onset'] - t0) * fpq))
        end = start + max(int(round(note['duration'] * fpq)), 1)
        time = range(start, end)
        piano_roll[pitch, time] = 1  # add note to piano_roll
    return piano_roll[pitch_low:pitch_high], t0


def load_chord_labels(i):
    """
    Load chords of each piece and add chord symbols into the labels.
    :param i: which sonata to take (sonatas indexed from 0 to 31)
    :return: chord_labels
    """

    dt = [('onset', 'float'), ('end', 'float'), ('key', '<U10'), ('degree', '<U10'), ('quality', '<U10'),
          ('inversion', 'int'), ('chord_function', '<U10')]  # datatype
    chords_file = os.path.join(DATASET_FOLDER, str(i).zfill(2), "chords.xlsx")

    workbook = xlrd.open_workbook(chords_file)
    sheet = workbook.sheet_by_index(0)
    chords = []
    for rowx in range(sheet.nrows):
        cols = sheet.row_values(rowx)
        # xlrd.open_workbook automatically casts strings to float if they are compatible. Revert this.
        if isinstance(cols[3], float):  # if type(degree) == float
            cols[3] = str(int(cols[3]))
        chords.append(tuple(cols))
    return np.array(chords, dtype=dt)  # convert to structured array


def shift_chord_labels(chord_labels, s):
    """

    :param chord_labels:
    :param s:
    :return:
    """
    new_labels = chord_labels.copy()

    for i in range(len(new_labels)):

        key = chord_labels[i]['key']
        if '-' in key:
            new_key = NOTES[(NOTES.index(key[0].upper()) + s - 1) % 12]
        else:
            new_key = NOTES[(NOTES.index(key.upper()) + s) % 12]
        new_labels[i]['key'] = new_key if key.isupper() else new_key.lower()

    return new_labels


def segment_piano_roll(piano_roll, n_frames, wsize=32, hsize=4):
    """
    Segment each pianoroll.
    :param piano_roll:
    :param n_frames:
    :param wsize: window size,  default= 32 (4 beats)
    :param hsize: hop size, default = 4 (half a beat)
    :return: segments of piano roll
    """
    # Zero-padding
    npad = (piano_roll.shape[1] - wsize) % hsize
    piano_roll = np.pad(piano_roll, ((0, 0), (0, npad)), 'constant', constant_values=0)

    # Length of 3D output array along its axis=1
    segments = np.zeros((n_frames, piano_roll.shape[0] * wsize))
    for i in range(n_frames - 1):
        segments[i] = piano_roll[:, i * hsize:i * hsize + wsize].reshape([-1])
    return segments


def segment_chord_labels(chord_labels, n_frames, t0, wsize=32, hsize=4, fpq=8):
    # Get corresponding chord label (only chord symbol) for each segment
    labels = []
    for n in range(n_frames):
        seg_time = (n * hsize + 0.5 * wsize) / fpq + t0  # central time of the segment in quarter notes
        label = chord_labels[np.logical_and(chord_labels['onset'] <= seg_time, chord_labels['end'] > seg_time)]
        try:
            label = label[0]
        except IndexError:
            raise HarmonicAnalysisError(f"Cannot read label for Sonata N.{i + 1} at frame {n}")

        # TODO: clearly not optimal, since I'm calculating every
        labels.append((label['key'], label['degree'], label['quality'], label['inversion'], _find_chord_symbol(label)))
    return labels


def encode_chords(chords):
    """

    :param chord:
    :return:
    """
    chords_enc = []
    for chord in chords:
        key_enc = _encode_key(str(chord[0]))
        degree_p_enc, degree_s_enc = _encode_degree(str(chord[1]))
        quality_enc = _encode_quality(str(chord[2]))
        inversion_enc = int(chord[3])
        root_enc, symbol_enc = _encode_symbol(str(chord[4]))

        chords_enc.append((key_enc, degree_p_enc, degree_s_enc, quality_enc, inversion_enc, root_enc, symbol_enc))

    return chords_enc


def find_pitch_extremes():
    min_note, max_note = 128, 0  # they are inverted on purpose!
    for i in range(1, 33):
        piano_roll, t0 = load_notes(i, FPQ)
        min_i = np.where(np.max(piano_roll, axis=-1) == 1)[0][0] - 6  # -6 because we want room for transposing
        max_i = np.where(np.max(piano_roll, axis=-1) == 1)[0][-1] + 5  # +5 because we want room for transposing
        min_note = min(min_note, min_i)
        max_note = max(max_note, max_i)
    return min_note, max_note


if os.path.isfile(TRAIN_TFRECORDS):
    answer = input("tfrecords exist already. Are you sure that you want to erase and recalculate them? [yes/no]\n")
    while answer.lower().strip() not in ['yes', 'no']:
        answer = input("I didn't understand. Please confirm if you want to erase and recalculate tfrecords. [yes/no]\n")
    if answer.lower().strip() == 'no':
        print("You decided not to replace them. I guess, better safe than sorry. Goodbye!")
        quit()
for indices, output_file in zip([TRAIN_INDICES, VALID_INDICES, TEST_INDICES],
                                [TRAIN_TFRECORDS, VALID_TFRECORDS, TEST_TFRECORDS]):
    with tf.io.TFRecordWriter(output_file) as writer:
        logger.info(f'Working on {os.path.basename(output_file)}.')

        for i in indices:
            logger.info(f"Sonata N.{i}")
            piano_roll, t0 = load_notes(i, FPQ, PITCH_LOW, PITCH_HIGH)
            chord_labels = load_chord_labels(i)
            n_frames = int(math.ceil((piano_roll.shape[1] - WSIZE) / HSIZE)) + 1

            for s in range(-6, 6):
                pr_shifted = np.roll(piano_roll, shift=s, axis=0)
                pr_segments = segment_piano_roll(pr_shifted, n_frames, wsize=WSIZE, hsize=HSIZE)

                cl_shifted = shift_chord_labels(chord_labels, s)
                cl_segments = segment_chord_labels(cl_shifted, n_frames, t0, wsize=WSIZE, hsize=HSIZE, fpq=FPQ)
                cl_encoded = encode_chords(cl_segments)
                for frame in range(n_frames):
                    x = pr_segments[frame]
                    y = cl_encoded[frame]
                    feature = {
                        'x': tf.train.Feature(float_list=tf.train.FloatList(value=x)),
                        'label_key': tf.train.Feature(int64_list=tf.train.Int64List(value=[y[0]])),
                        'label_degree_primary': tf.train.Feature(int64_list=tf.train.Int64List(value=[y[1]])),
                        'label_degree_secondary': tf.train.Feature(int64_list=tf.train.Int64List(value=[y[2]])),
                        'label_quality': tf.train.Feature(int64_list=tf.train.Int64List(value=[y[3]])),
                        'label_inversion': tf.train.Feature(int64_list=tf.train.Int64List(value=[y[4]])),
                        'label_root': tf.train.Feature(int64_list=tf.train.Int64List(value=[y[5]])),
                        'label_symbol': tf.train.Feature(int64_list=tf.train.Int64List(value=[y[6]])),
                        'sonata': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                        'frame': tf.train.Feature(int64_list=tf.train.Int64List(value=[frame])),
                        'transposed': tf.train.Feature(int64_list=tf.train.Int64List(value=[s])),
                    }
                    writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())

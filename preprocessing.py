"""
create a tfrecord containing the following features:
x the piano roll input data, with shape [n_frames, pitches]
label_key the local key of the music
label_degree_primary the chord degree with respect to the key, possibly fractional, e.g. V/V dominant of dominant
label_degree_secondary the chord degree with respect to the key, possibly fractional, e.g. V/V dominant of dominant
label_quality e.g. m, M, D7 for minor, major, dominant 7th etc.
label_inversion, from 0 to 3 depending on what note is at the bass
label_root, the root of the chord in jazz notation
label_symbol, the quality of the chord in jazz notation
sonata, the index of the sonata that is analysed
transposed, the number of semitones of transposition (negative for down-transposition)

ATTENTION: despite the name, the secondary_degree is actually "more important" than the primary degree,
since the latter is almost always equal to 1.
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import xlrd
from music21 import converter
from music21.chord import Chord
from music21.repeat import ExpanderException

from config import DATASET_FOLDER, TRAIN_INDICES, VALID_INDICES, TEST_INDICES, TRAIN_TFRECORDS, VALID_TFRECORDS, \
    TEST_TFRECORDS, FPQ, PITCH_LOW, PITCH_HIGH, HSIZE, DATA_FOLDER, NOTES
from utils import find_chord_symbol, _encode_key, _encode_degree, _encode_quality, _encode_symbol

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HarmonicAnalysisError(Exception):
    """ Raised when the harmonic analysis associated to a certain time can't be found """
    pass


def load_score(i, fpq, pitch_low=0, pitch_high=128):
    """
    Load notes in each piece, which is then represented as piano roll.
    :param pitch_low:
    :param pitch_high:
    :param i: which sonata to take (sonatas indexed from 1 to 32)
    :param fpq: frames per quarter note, default =  8 (that is, 32th note as 1 unit in piano roll)
    :return: pieces, tdeviation
    """

    score_file = os.path.join(DATASET_FOLDER, str(i).zfill(2), "score.mxl")
    try:
        score = converter.parse(score_file).expandRepeats()
    except ExpanderException:
        score = converter.parse(score_file)
        logger.warning("Could not expand repeats. Maybe there are no repeats in the piece? Please check.")
    n_frames = int(score.duration.quarterLength * fpq)
    measure_offset = list(score.measureOffsetMap().keys())
    measure_length = np.diff(measure_offset)
    t0 = - measure_offset[1] if measure_length[0] != measure_length[1] else 0  # Pickup time
    piano_roll = np.zeros(shape=(128, n_frames), dtype=np.int32)
    for n in score.flat.notes:
        pitches = np.array([x.midi for x in n.pitches] if isinstance(n, Chord) else [n.pitch.midi])
        start = int(round(n.offset * fpq))
        end = start + max(int(round(n.duration.quarterLength * fpq)), 1)
        time = np.arange(start, end)
        # p = sns.heatmap(piano_roll[:, :end])
        # plt.show(p)
        for p in pitches:  # add notes to piano_roll
            piano_roll[p, time] = 1
    # sns.heatmap(piano_roll)
    # plt.show()
    return piano_roll[pitch_low:pitch_high], t0


def visualize_piano_roll(pr, sonata, fpq, start=None, end=None):
    p = sns.heatmap(pr[::-1, start:end])
    plt.title(f'Sonata {sonata}, quarters (start, end) = {start / fpq, end / fpq}')
    plt.show(p)
    return


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


def segment_chord_labels(chord_labels, n_frames, t0, hsize=4, fpq=8):
    """

    :param chord_labels:
    :param n_frames: total frames of the analysis
    :param t0:
    :param hsize: hop size between different frames
    :param fpq: frames per quarter note
    :return:
    """
    # Get corresponding chord label (only chord symbol) for each segment
    labels = []
    for n in range(n_frames):
        seg_time = (n * hsize / fpq) + t0  # central time of the segment in quarter notes
        label = chord_labels[np.logical_and(chord_labels['onset'] <= max(seg_time, 0), chord_labels['end'] >= seg_time)]
        try:
            label = label[0]
        except IndexError:
            raise HarmonicAnalysisError(f"Cannot read label for Sonata N.{i} at frame {n}, time {seg_time}")

        # TODO: clearly not optimal, since I'm calculating every chord separately
        labels.append((label['key'], label['degree'], label['quality'], label['inversion'], find_chord_symbol(label)))
    return labels


def encode_chords(chords):
    """
    Associate every chord element with an integer that represents its category.

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


def find_bass_notes(piano_roll):
    """
    Given a piano roll, return the pitch class of the lowest note every 4 time-step.
    The output encoding is C = 0, C# = 1, ... B = 11

    :param piano_roll:
    :return:
    """
    bass = np.argmax(piano_roll, axis=0)
    bass = np.array([np.min(bass[4 * i:4 * (i + 1)]) for i in range(len(bass) // 4)])
    return bass + PITCH_LOW - 60  # C4 is the midi pitch 60


if __name__ == '__main__':

    indices = [TRAIN_INDICES, VALID_INDICES, TEST_INDICES]
    tfrecords = [TRAIN_TFRECORDS, VALID_TFRECORDS, TEST_TFRECORDS]
    if os.path.isfile(TRAIN_TFRECORDS):
        answer = input(
            "tfrecords exist already. Do you want to erase them, backup them, write into a temporary file, or abort the calculation? "
            "[erase/backup/temp/abort]\n")
        while answer.lower().strip() not in ['erase', 'backup', 'temp', 'abort']:
            answer = input(
                "I didn't understand. Please choose an option (abort is safest). [erase/backup/temp/abort]\n")
        if answer.lower().strip() == 'abort':
            print("You decided not to replace them. I guess, better safe than sorry. Goodbye!")
            quit()

        elif answer.lower().strip() == 'temp':
            print("I'm going to write the files to train_temp.tfrecords and similar!")
            tfrecords_temp = [f.split(".") for f in tfrecords]
            for ft in tfrecords_temp:
                ft[0] += '_temp'
            tfrecords = ['.'.join(ft) for ft in tfrecords_temp]

        elif answer.lower().strip() == 'backup':
            i = 0
            tfrecords_backup = [f.split(".") for f in tfrecords]
            for fb in tfrecords_backup:
                fb[0] += f'_backup_{i}'
            tfrecords_backup = ['.'.join(fb) for fb in tfrecords_backup]
            while any([d in os.listdir(DATA_FOLDER) for d in tfrecords_backup]):
                i += 1
                tfrecords_backup = [f.split(".") for f in tfrecords]
                for fb in tfrecords_backup:
                    fb[0] += f'_backup_{i}'
                tfrecords_backup = ['.'.join(fb) for fb in tfrecords_backup]

            for src, dst in zip(tfrecords, tfrecords_backup):
                os.rename(src, dst)
            print(f"data backed up with backup index {i}")

    k = 0
    for indices, output_file in zip(indices, tfrecords):
        k += 1
        with tf.io.TFRecordWriter(output_file) as writer:
            logger.info(f'Working on {os.path.basename(output_file)}.')

            for i in indices:
                logger.info(f"Sonata N.{i}")
                piano_roll, t0 = load_score(i, FPQ, PITCH_LOW, PITCH_HIGH)
                # for j in range(len(piano_roll[0]) - 128, len(piano_roll[0]), 128):
                #     visualize_piano_roll(piano_roll, i, FPQ, j, j+FPQ*16)
                # Adjust the length of the piano roll to be an exact multiple of the HSIZE
                npad = (- piano_roll.shape[1]) % HSIZE
                piano_roll = np.pad(piano_roll, ((0, 0), (0, npad)), 'constant',
                                    constant_values=0)  # shape(PITCH_HIGH - PITCH_LOW, frames)
                n_frames_analysis = piano_roll.shape[1] // HSIZE

                chord_labels = load_chord_labels(i)
                # structure, score = load_structure(i)
                # structure_labels = segment_structure(structure, n_frames_analysis, hsize=HSIZE, fpq=FPQ)

                for s in range(-6, 6):
                    if k == 3 and s != 0:  # don't store all 12 transpositions for test data
                        continue
                    pr_shifted = np.roll(piano_roll, shift=s, axis=0)
                    bass = find_bass_notes(pr_shifted)

                    cl_shifted = shift_chord_labels(chord_labels, s)
                    cl_segments = segment_chord_labels(cl_shifted, n_frames_analysis, t0, hsize=HSIZE, fpq=FPQ)
                    cl_encoded = encode_chords(cl_segments)
                    feature = {
                        'piano_roll': tf.train.Feature(float_list=tf.train.FloatList(value=pr_shifted.reshape(-1))),
                        'bass': tf.train.Feature(int64_list=tf.train.Int64List(value=[b for b in bass])),
                        # 'label_structure': tf.train.Feature(
                        #     int64_list=tf.train.Int64List(value=[l for l in structure_labels])),
                        'label_key': tf.train.Feature(int64_list=tf.train.Int64List(value=[c[0] for c in cl_encoded])),
                        'label_degree_primary': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[c[1] for c in cl_encoded])),
                        'label_degree_secondary': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[c[2] for c in cl_encoded])),
                        'label_quality': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[c[3] for c in cl_encoded])),
                        'label_inversion': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[c[4] for c in cl_encoded])),
                        'label_root': tf.train.Feature(int64_list=tf.train.Int64List(value=[c[5] for c in cl_encoded])),
                        'label_symbol': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[c[6] for c in cl_encoded])),
                        'sonata': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                        'transposed': tf.train.Feature(int64_list=tf.train.Int64List(value=[s])),
                    }
                    writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())

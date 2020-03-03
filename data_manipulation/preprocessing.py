"""
This is an entry point, no other file should import from this one.
Augment and convert the data from .mxl plus .csv to .tfrecords for training the system.
This creates a tfrecord containing the following features:
'name': the name of the file
'transposition': the number of semitones of transposition (0 = original key)
'piano_roll': the input data, in format [n_frames, features]
'label_key': the local key of the music
'label_degree_primary': the denominator of chord degree with respect to the key, possibly fractional, e.g. V/V
'label_degree_secondary': the numerator of chord degree with respect to the key, possibly fractional, e.g. V/V
'label_quality': e.g. m, M, D7 for minor, major, dominant 7th etc.
'label_inversion': from 0 to 3 depending on what note is at the bass
'label_root': the root of the chord

ATTENTION: despite the name, the secondary_degree is actually "more important" than the primary degree,
since the latter is almost always equal to 1.
"""

import logging
import os

import numpy as np
import tensorflow as tf

from config import DATA_FOLDER, FPQ, HSIZE, CHUNK_SIZE, INPUT_TYPES
from utils import setup_tfrecords_paths
from utils_music import load_score_pitch_complete, load_chord_labels, transpose_chord_labels, segment_chord_labels, \
    encode_chords, load_score_pitch_bass, load_score_spelling_bass, calculate_number_transpositions_key, \
    attach_chord_root, load_score_pitch_class, load_score_spelling_complete, load_score_spelling_class

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def validate_tfrecords_paths(tfrecords, data_folder):
    existent = [f for f in tfrecords if os.path.isfile(f)]
    if len(existent) > 0:
        answer = input(
            f"{[os.path.basename(f) for f in existent]} exists already. "
            "Do you want to replace the tfrecords, backup them, write into a temporary file, or abort the calculation? "
            "[replace/backup/temp/abort]\n"
        )
        while answer.lower().strip() not in ['replace', 'backup', 'temp', 'abort']:
            answer = input(
                "I didn't understand. Please choose an option (abort is safest). [replace/backup/temp/abort]\n")

        if answer.lower().strip() == 'abort':
            print("You decided not to replace them. I guess, better safe than sorry. Goodbye!")
            quit()

        elif answer.lower().strip() == 'temp':
            tfrecords = [f.split("_")[0] for f in tfrecords]
            tfrecords = [f + "_temp.tfrecords" for f in tfrecords]
            logger.warning(f"I'm going to write the files to {tfrecords[0]} and similar!")

        elif answer.lower().strip() == 'backup':
            i = 0
            tfrecords_backup = [f.split(".") for f in tfrecords]
            for fb in tfrecords_backup:
                fb[0] += f'_backup_{i}'
            tfrecords_backup = ['.'.join(fb) for fb in tfrecords_backup]
            while any([d in os.listdir(data_folder) for d in tfrecords_backup]):
                i += 1
                tfrecords_backup = [f.split(".") for f in tfrecords]
                for fb in tfrecords_backup:
                    fb[0] += f'_backup_{i}'
                tfrecords_backup = ['.'.join(fb) for fb in tfrecords_backup]

            for src, dst in zip(tfrecords, tfrecords_backup):
                os.rename(src, dst)
            logger.warning(
                f"existing data backed up with backup index {i}, new data will be in {tfrecords[0]} and similar")

        elif answer.lower().strip() == 'replace':
            logger.warning(
                f"you chose to replace the old data with new one; the old data will be erased in the process")

    return tfrecords


def create_feature_dictionary(piano_roll, chords, name, s=None, start=None, end=None):
    feature = {
        'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[name.encode('utf-8')])),
        'transposition': tf.train.Feature(int64_list=tf.train.Int64List(value=[s])),
        'piano_roll': tf.train.Feature(float_list=tf.train.FloatList(value=piano_roll.reshape(-1))),
        'label_key': tf.train.Feature(int64_list=tf.train.Int64List(value=[c[0] for c in chords])),
        'label_degree_primary': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[c[1] for c in chords])),
        'label_degree_secondary': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[c[2] for c in chords])),
        'label_quality': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[c[3] for c in chords])),
        'label_inversion': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[c[4] for c in chords])),
        'label_root': tf.train.Feature(int64_list=tf.train.Int64List(value=[c[5] for c in chords])),
    }
    if start is not None:
        feature['start'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[start]))
    if end is not None:
        feature['end'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[end]))

    return feature


def create_tfrecords(input_type, data_folder):
    if input_type not in INPUT_TYPES:
        raise ValueError('Choose a valid value for input_type')

    print(f"Welcome to the preprocessing routine, whose goal is to create tfrecords for your model.\n"
          f"You are currently working in the {input_type} mode.\n"
          f"Thank you for choosing algomus productions and have a nice day!\n")

    datasets = [
        'train',
        'valid',
        'test',
    ]
    tfrecords = setup_tfrecords_paths(data_folder, datasets, input_type)
    tfrecords = validate_tfrecords_paths(tfrecords, data_folder)

    for ds, output_file in zip(datasets, tfrecords):
        folder = os.path.join(data_folder, ds)
        with tf.io.TFRecordWriter(output_file) as writer:
            logger.info(f'Working on {os.path.basename(output_file)}.')
            chords_folder = os.path.join(folder, 'chords')
            scores_folder = os.path.join(folder, 'scores')
            file_names = ['.'.join(fn.split('.')[:-1]) for fn in os.listdir(chords_folder) if not fn.startswith('.')]
            for fn in file_names:
                # if fn not in ['bsq_op127_no12_mov2']:
                #     continue
                sf = os.path.join(scores_folder, fn + ".mxl")
                cf = os.path.join(chords_folder, fn + ".csv")

                logger.info(f"Analysing {fn}")
                chord_labels = load_chord_labels(cf)
                cl_full = attach_chord_root(chord_labels, input_type.startswith('spelling'))

                # Load piano_roll and calculate transpositions to the left and right
                if input_type.startswith('pitch'):
                    nl, nr = 6, 6
                    if 'complete' in input_type:
                        piano_roll = load_score_pitch_complete(sf, FPQ)
                    elif 'bass' in input_type:
                        piano_roll = load_score_pitch_bass(sf, FPQ)
                    elif 'class' in input_type:
                        piano_roll = load_score_pitch_class(sf, FPQ)
                    else:
                        raise NotImplementedError("verify the input_type")
                elif input_type.startswith('spelling'):
                    if 'complete' in input_type:
                        piano_roll, nl_pitches, nr_pitches = load_score_spelling_complete(sf, FPQ)
                    elif 'bass' in input_type:
                        piano_roll, nl_pitches, nr_pitches = load_score_spelling_bass(sf, FPQ)
                    elif 'class' in input_type:
                        piano_roll, nl_pitches, nr_pitches = load_score_spelling_class(sf, FPQ)
                    else:
                        raise NotImplementedError("verify the input_type")
                    nl_keys, nr_keys = calculate_number_transpositions_key(chord_labels)
                    nl = min(nl_keys, nl_pitches)  # notice that they can be negative!
                    nr = min(nr_keys, nr_pitches)
                else:
                    raise NotImplementedError("verify the input_type")

                # Adjust the length of the piano roll to be an exact multiple of the HSIZE
                npad = (- piano_roll.shape[1]) % HSIZE
                piano_roll = np.pad(piano_roll, ((0, 0), (0, npad)), 'constant', constant_values=0)
                n_frames_analysis = piano_roll.shape[1] // HSIZE

                # Pre-process the chords
                cl_segmented = segment_chord_labels(cl_full, n_frames_analysis, hsize=HSIZE, fpq=FPQ)
                if ds == 'train':
                    logger.info(f"Transposing {nl} times to the left and {nr - 1} to the right")
                if nl < 0 or nr < 0:
                    logger.warning(
                        f"The original score doesn't satisfy the pitch and key constraints! nl={nl}, nr={nr}")
                for s in range(-nl, nr):
                    if ds != 'train' and s != 0:  # transpose only for training data
                        continue
                    if input_type.startswith('pitch'):
                        if 'complete' in input_type:
                            pr_transposed = np.roll(piano_roll, shift=s, axis=0)
                        else:
                            pr_transposed = np.zeros(piano_roll.shape, dtype=np.int32)
                            for i in range(12):  # transpose the main part
                                pr_transposed[i, :] = piano_roll[(i - s) % 12, :]  # the minus sign is correct!
                            for i in range(12, pr_transposed.shape[0]):  # transpose the bass, if present
                                pr_transposed[i, :] = piano_roll[((i - s) % 12) + 12, :]
                    elif input_type.startswith('spelling'):
                        # nL and nR are calculated s.t. transpositions never have three flats or sharps.
                        # this means that they will never get out of the allotted 35 slots, and
                        # general pitches and bass will never mix
                        # that's why we can safely use roll without any validation of the result
                        # we also don't transpose to different octaves
                        pr_transposed = np.roll(piano_roll, shift=s, axis=0)

                    # definition of proximity for pitches
                    pp = 'fifth' if input_type.startswith('spelling') else 'semitone'
                    cl_transposed = transpose_chord_labels(cl_segmented, s, pp)
                    chords = encode_chords(cl_transposed, pp)
                    if any([x is None for c in chords for x in c]):
                        logger.warning(f"skipping transposition {s}")
                        continue
                    if input_type.endswith('cut'):
                        start, end = 0, CHUNK_SIZE
                        while start < len(chords):
                            chord_partial = chords[start:end]
                            pr_partial = pr_transposed[:, 4 * start:4 * end]
                            feature = create_feature_dictionary(pr_partial, chord_partial, fn, s, start, end)
                            writer.write(
                                tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())
                            start += CHUNK_SIZE
                            end += CHUNK_SIZE
                    else:
                        feature = create_feature_dictionary(pr_transposed, chords, fn, s)
                        writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())
    return


if __name__ == '__main__':
    # input_type = INPUT_TYPES
    input_type = ['spelling_bass_cut']
    # data_folder = DATA_FOLDER
    data_folder = 'data_small'
    for it in input_type:
        create_tfrecords(it, data_folder)

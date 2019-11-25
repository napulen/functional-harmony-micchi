import logging
import os

import numpy as np
import tensorflow as tf

from config import DATA_FOLDER, FPQ, HSIZE, CHUNK_SIZE, INPUT_TYPES
from utils import setup_tfrecords_paths
from utils_music import load_score_pitch_complete, load_chord_labels, shift_chord_labels, segment_chord_labels, \
    encode_chords, load_score_pitch_bass, load_score_spelling_bass, calculate_number_transpositions_key, \
    attach_chord_root, load_score_pitch_class, load_score_spelling_complete, load_score_spelling_class

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def check_existence_tfrecords(tfrecords):
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
            while any([d in os.listdir(DATA_FOLDER) for d in tfrecords_backup]):
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


def create_tfrecords(input_type):
    if input_type not in INPUT_TYPES:
        raise ValueError('Choose a valid value for input_type')

    print(f"Welcome to the preprocessing routine, whose goal is to create tfrecords for your model.\n"
          f"You are currently working in the {input_type} mode.\n"
          f"Thank you for choosing algomus productions and have a nice day!\n")

    folders = [
        os.path.join(DATA_FOLDER, 'train'),
        os.path.join(DATA_FOLDER, 'valid'),
        os.path.join(DATA_FOLDER, 'BPS')
    ]
    tfrecords = setup_tfrecords_paths(DATA_FOLDER, input_type)

    # folders = [os.path.join(DATA_FOLDER, 'valid_bpsfh')]
    # tfrecords = [os.path.join(DATA_FOLDER, f'testvalid_bpsfh_{input_type}.tfrecords')]
    tfrecords = check_existence_tfrecords(tfrecords)

    for folder, output_file in zip(folders, tfrecords):
        with tf.io.TFRecordWriter(output_file) as writer:
            logger.info(f'Working on {os.path.basename(output_file)}.')
            chords_folder = os.path.join(folder, 'chords')
            scores_folder = os.path.join(folder, 'scores')
            file_names = ['.'.join(fn.split('.')[:-1]) for fn in os.listdir(chords_folder)]
            for fn in file_names:
                # if fn not in ['bsq_op127_no12_mov2']:
                #     continue
                sf = os.path.join(scores_folder, fn + ".mxl")
                cf = os.path.join(chords_folder, fn + ".csv")

                logger.info(f"Analysing {fn}")
                chord_labels = load_chord_labels(cf)
                cl_full = attach_chord_root(chord_labels, input_type.startswith('spelling'))

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
                    nl = min(nl_keys, nl_pitches)
                    nr = min(nr_keys, nr_pitches)
                else:
                    raise NotImplementedError("verify the input_type")

                # Adjust the length of the piano roll to be an exact multiple of the HSIZE
                npad = (- piano_roll.shape[1]) % HSIZE
                piano_roll = np.pad(piano_roll, ((0, 0), (0, npad)), 'constant', constant_values=0)
                n_frames_analysis = piano_roll.shape[1] // HSIZE

                # Pre-process the chords
                cl_segmented = segment_chord_labels(cl_full, n_frames_analysis, hsize=HSIZE, fpq=FPQ)

                logger.info(f"Transposing {nl} times to the left and {nr - 1} to the right")
                for s in range(-nl, nr):
                    if folder != folders[0] and s != 0:  # transpose only for training data
                        continue
                    if input_type.startswith('pitch'):
                        if 'complete' in input_type:
                            pr_shifted = np.roll(piano_roll, shift=s, axis=0)
                        else:
                            pr_shifted = np.zeros(piano_roll.shape, dtype=np.int32)
                            for i in range(12):  # transpose the main part
                                pr_shifted[i, :] = piano_roll[(i - s) % 12, :]  # the minus sign is correct!
                            for i in range(12, pr_shifted.shape[0]):  # transpose the bass, if present
                                pr_shifted[i, :] = piano_roll[((i - s) % 12) + 12, :]
                    elif input_type.startswith('spelling'):
                        # nL and nR are calculated s.t. transpositions never have three flats or sharps.
                        # this means that they will never get out of the allotted 35 slots, and
                        # general pitches and bass will never mix
                        # that's why we can safely use roll without any validation of the result
                        # we also don't transpose to different octaves
                        pr_shifted = np.roll(piano_roll, shift=s, axis=0)

                    pp = 'fifth' if input_type.startswith(
                        'spelling') else 'semitone'  # definition of proximity for pitches
                    cl_shifted = shift_chord_labels(cl_segmented, s, pp)
                    chords = encode_chords(cl_shifted, pp)
                    if any([x is None for c in chords for x in c]):
                        logger.warning(f"skipping transposition {s}")
                        continue
                    if input_type.endswith('cut'):
                        start, end = 0, CHUNK_SIZE
                        while start < len(chords):
                            chord_partial = chords[start:end]
                            pr_partial = pr_shifted[:, 4 * start:4 * end]
                            feature = create_feature_dictionary(pr_partial, chord_partial, fn, s, start, end)
                            writer.write(
                                tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())
                            start += CHUNK_SIZE
                            end += CHUNK_SIZE
                    else:
                        feature = create_feature_dictionary(pr_shifted, chords, fn, s)
                        writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())
    return


if __name__ == '__main__':
    input_type = INPUT_TYPES
    # input_type = ['spelling_bass_cut']
    for it in input_type:
        create_tfrecords(it)

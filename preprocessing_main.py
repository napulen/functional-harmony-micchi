import logging
import os

import numpy as np
import tensorflow as tf

from config import TRAIN_INDICES, VALID_INDICES, VALID_TFRECORDS, TRAIN_TFRECORDS, \
    DATA_FOLDER, FPQ, PITCH_LOW, PITCH_HIGH, HSIZE, MODE
from preprocessing import load_score_midi_number, load_chord_labels, shift_chord_labels, segment_chord_labels, \
    encode_chords, load_score_pitch_class, load_score_beat_strength, load_score_pitch_spelling, \
    calculate_number_transpositions_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def check_existence_tfrecords(tfrecords):
    if os.path.isfile(TRAIN_TFRECORDS):
        answer = input(
            f"{os.path.basename(TRAIN_TFRECORDS)} exists already. "
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
            tfrecords_temp = [f.split(".") for f in tfrecords]
            for ft in tfrecords_temp:
                ft[0] += '_temp'
            tfrecords = ['.'.join(ft) for ft in tfrecords_temp]
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
            logger.warning(f"existing data backed up with backup index {i}, new data will be in {tfrecords[0]} and similar")

        elif answer.lower().strip() == 'replace':
            logger.warning(f"you chose to replace the old data with new one; the old data will be erased in the process")

    return tfrecords


if __name__ == '__main__':
    print(f"Welcome to the preprocessing routine, whose goal is to create tfrecords for your model.\n"
          f"You are currently working in the {MODE} mode.\n"
          f"Thank you for choosing algomus productions and have a nice day!")

    indices = [TRAIN_INDICES, VALID_INDICES]
    tfrecords = [TRAIN_TFRECORDS, VALID_TFRECORDS]
    tfrecords = check_existence_tfrecords(tfrecords)

    k = 0
    for indices, output_file in zip(indices, tfrecords):
        k += 1
        with tf.io.TFRecordWriter(output_file) as writer:
            logger.info(f'Working on {os.path.basename(output_file)}.')

            for i in indices:
                logger.info(f"Sonata N.{i}")
                chord_labels = load_chord_labels(i)
                if MODE == 'pitch_class':
                    piano_roll, t0 = load_score_pitch_class(i, FPQ)
                    nl, nr = 6, 6
                elif MODE == 'pitch_class_beat_strength':
                    piano_roll, t0 = load_score_pitch_class(i, FPQ)
                    beat_strength = load_score_beat_strength(i, FPQ)
                    piano_roll = np.append(piano_roll, beat_strength, axis=0)
                    nl, nr = 6, 6
                elif MODE == 'midi_number':
                    piano_roll, t0 = load_score_midi_number(i, FPQ, PITCH_LOW, PITCH_HIGH)
                    nl, nr = 6, 6
                elif MODE == 'pitch_spelling':
                    piano_roll, t0, nl_pitches, nr_pitches = load_score_pitch_spelling(i, FPQ)
                    nl_keys, nr_keys = calculate_number_transpositions_key(chord_labels)
                    nl = min(nl_keys, nl_pitches)
                    nr = min(nr_keys, nr_pitches)
                    logger.info(f'Acceptable transpositions (pitches, keys): '
                                f'left {nl_pitches, nl_keys}; '
                                f'right {nr_pitches-1, nr_keys-1}.')
                else:
                    raise ReferenceError("I shouldn't be here. "
                                         "It looks like the name of some mode has been hard-coded in the wrong way.")
                # visualize piano rolls excerpts (indexed by j)
                # for j in range(len(piano_roll[0]) - 128, len(piano_roll[0]), 128):
                #     visualize_piano_roll(piano_roll, i, FPQ, j, j+FPQ*16)

                # Adjust the length of the piano roll to be an exact multiple of the HSIZE
                npad = (- piano_roll.shape[1]) % HSIZE
                piano_roll = np.pad(piano_roll, ((0, 0), (0, npad)), 'constant',
                                    constant_values=0)  # shape(PITCH_HIGH - PITCH_LOW, frames)
                n_frames_analysis = piano_roll.shape[1] // HSIZE

                logger.info(f"Transposing {nl} times to the left and {nr - 1} to the right")
                for s in range(-nl, nr):
                    if k == 2 and s != 0:  # don't store all 12 transpositions for validation data
                        continue
                    if MODE == 'pitch_class' or MODE == 'pitch_class_beat_strength':
                        pr_shifted = np.zeros(piano_roll.shape, dtype=np.int32)
                        for i in range(12):
                            pr_shifted[i, :] = piano_roll[(i - s) % 12, :]  # the minus sign is correct!
                            pr_shifted[i + 12, :] = piano_roll[((i - s) % 12) + 12, :]
                        for i in range(24, len(pr_shifted)):
                            pr_shifted[i, :] = piano_roll[i, :]
                    elif MODE == 'pitch_spelling':
                        # nL and nR are calculated s.t. transpositions never have three flats or sharps.
                        # this means that they will never get out of the allotted 35 slots, and
                        # general pitches and bass will never mix
                        # that's why we can safely use roll without any validation of the result
                        pr_shifted = np.roll(piano_roll, shift=s, axis=0)
                    elif MODE == 'midi_number':
                        pr_shifted = np.roll(piano_roll, shift=s, axis=0)

                    pp = 'fifth' if MODE == 'pitch_spelling' else 'semitone'  # definition of proximity for pitches
                    ps = True if MODE == 'pitch_spelling' else False
                    cl_shifted = shift_chord_labels(chord_labels, s, pp)
                    cl_segments = segment_chord_labels(i, cl_shifted, n_frames_analysis, t0, hsize=HSIZE, fpq=FPQ,
                                                       pitch_spelling=ps)
                    cl_encoded = encode_chords(cl_segments, pp)
                    feature = {
                        'piano_roll': tf.train.Feature(float_list=tf.train.FloatList(value=pr_shifted.reshape(-1))),
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
                        'sonata': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                        'transposed': tf.train.Feature(int64_list=tf.train.Int64List(value=[s])),
                    }
                    writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())

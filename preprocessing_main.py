import logging
import os
import numpy as np
import tensorflow as tf
from config import TRAIN_INDICES, VALID_INDICES, VALID_TFRECORDS, TRAIN_TFRECORDS, \
    DATA_FOLDER, FPQ, PITCH_LOW, PITCH_HIGH, HSIZE
from preprocessing import load_score, load_chord_labels, shift_chord_labels, segment_chord_labels, \
    encode_chords, load_score_pitch_class

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

mode = 'pitch_class'


def check_existence_tfrecords(tfrecords):
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
    return tfrecords


if __name__ == '__main__':
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
                if mode == 'pitch_class':
                    piano_roll, t0 = load_score_pitch_class(i, FPQ)
                else:
                    piano_roll, t0 = load_score(i, FPQ, PITCH_LOW, PITCH_HIGH)
                # visualize piano rolls excerpts (indexed by j)
                # for j in range(len(piano_roll[0]) - 128, len(piano_roll[0]), 128):
                #     visualize_piano_roll(piano_roll, i, FPQ, j, j+FPQ*16)

                # Adjust the length of the piano roll to be an exact multiple of the HSIZE
                npad = (- piano_roll.shape[1]) % HSIZE
                piano_roll = np.pad(piano_roll, ((0, 0), (0, npad)), 'constant',
                                    constant_values=0)  # shape(PITCH_HIGH - PITCH_LOW, frames)
                n_frames_analysis = piano_roll.shape[1] // HSIZE

                chord_labels = load_chord_labels(i)

                for s in range(-6, 6):
                    if k == 2 and s != 0:  # don't store all 12 transpositions for test data
                        continue
                    if mode == 'pitch_class':
                        pr_shifted = np.zeros(piano_roll.shape)
                        for i in range(12):
                            pr_shifted[i, :] = piano_roll[(i+s) % 12, :]
                            pr_shifted[i+12, :] = piano_roll[((i+s) % 12) + 12, :]
                    else:
                        pr_shifted = np.roll(piano_roll, shift=s, axis=0)

                    cl_shifted = shift_chord_labels(chord_labels, s)
                    cl_segments = segment_chord_labels(i, cl_shifted, n_frames_analysis, t0, hsize=HSIZE, fpq=FPQ)
                    cl_encoded = encode_chords(cl_segments)
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

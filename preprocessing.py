import numpy as np
import math
import xlrd
import numpy.lib.recfunctions as rfn
from scipy import stats
import itertools
import os
import logging
import tensorflow as tf
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOTS = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
NOTES = ['C', 'C+', 'D', 'D+', 'E', 'F', 'F+', 'G', 'G+', 'A', 'A+', 'B']
SCALES = {
    'C': ['C', 'D', 'E', 'F', 'G', 'A', 'B'], 'a': ['A', 'B', 'C', 'D', 'E', 'F', 'G+'],
    'G': ['G', 'A', 'B', 'C', 'D', 'E', 'F+'], 'e': ['E', 'F+', 'G', 'A', 'B', 'C', 'D+'],
    'D': ['D', 'E', 'F+', 'G', 'A', 'B', 'C+'], 'b': ['B', 'C+', 'D', 'E', 'F+', 'G', 'A+'],
    'A': ['A', 'B', 'C+', 'D', 'E', 'F+', 'G+'], 'f+': ['F+', 'G+', 'A', 'B', 'C+', 'D', 'E+'],
    'E': ['E', 'F+', 'G+', 'A', 'B', 'C+', 'D+'], 'c+': ['C+', 'D+', 'E', 'F+', 'G+', 'A', 'B+'],
    'B': ['B', 'C+', 'D+', 'E', 'F+', 'G+', 'A+'], 'g+': ['G+', 'A+', 'B', 'C+', 'D+', 'E', 'F++'],
    'F+': ['F+', 'G+', 'A+', 'B', 'C+', 'D+', 'E+'], 'd+': ['D+', 'E+', 'F+', 'G+', 'A+', 'B', 'C++'],
    'C+': ['C+', 'D+', 'E+', 'F+', 'G+', 'A+', 'B+'], 'a+': ['A+', 'B+', 'C+', 'D+', 'E+', 'F+', 'G++'],
    'G+': ['G+', 'A+', 'B+', 'C+', 'D+', 'E+', 'F++'], 'e+': ['E+', 'F++', 'G+', 'A+', 'B+', 'C+', 'D++'],
    'D+': ['D+', 'E+', 'F++', 'G+', 'A+', 'B+', 'C++'], 'b+': ['B+', 'C++', 'D+', 'E+', 'F++', 'G+', 'A++'],
    'A+': ['A+', 'B+', 'C++', 'D+', 'E+', 'F++', 'G++'], 'f++': ['F++', 'G++', 'A+', 'B+', 'C++', 'D+', 'E++'],
    'F': ['F', 'G', 'A', 'B-', 'C', 'D', 'E'], 'd': ['D', 'E', 'F', 'G', 'A', 'B-', 'C+'],
    'B-': ['B-', 'C', 'D', 'E-', 'F', 'G', 'A'], 'g': ['G', 'A', 'B-', 'C', 'D', 'E-', 'F+'],
    'E-': ['E-', 'F', 'G', 'A-', 'B-', 'C', 'D'], 'c': ['C', 'D', 'E-', 'F', 'G', 'A-', 'B'],
    'A-': ['A-', 'B-', 'C', 'D-', 'E-', 'F', 'G'], 'f': ['F', 'G', 'A-', 'B-', 'C', 'D-', 'E'],
    'D-': ['D-', 'E-', 'F', 'G-', 'A-', 'B-', 'C'], 'b-': ['B-', 'C', 'D-', 'E-', 'F', 'G-', 'A'],
    'G-': ['G-', 'A-', 'B-', 'C-', 'D-', 'E-', 'F'], 'e-': ['E-', 'F', 'G-', 'A-', 'B-', 'C-', 'D'],
    'C-': ['C-', 'D-', 'E-', 'F-', 'G-', 'A-', 'B-'], 'a-': ['A-', 'B-', 'C-', 'D-', 'E-', 'F-', 'G'],
    'F-': ['F-', 'G-', 'A-', 'B--', 'C-', 'D-', 'E-'], 'd-': ['D-', 'E-', 'F-', 'G-', 'A-', 'B--', 'C']}
R2I = dict([(e[1], e[0]) for e in enumerate(ROOTS)])
N2I = dict([(e[1], e[0]) for e in enumerate(NOTES)])
DATASET_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'BPS_FH_Dataset')
TRAIN_INDICES = [4, 11, 16, 20, 26, 31, 3, 8, 12, 17, 23, 21, 27, 29, 30, 10, 1, 2]
VALID_INDICES = [7, 18, 28, 15, 25, 5, 19]
TEST_INDICES = [0, 13, 22, 14, 19, 24, 6]
TRAIN_TFRECORDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'train.tfrecords')
VALID_TFRECORDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'valid.tfrecords')
TEST_TFRECORDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'test.tfrecords')


class HarmonicAnalysisError(Exception):
    """ Raised when the harmonic analysis associated to a certain time can't be found """
    pass


def _is_major(key):
    """ The input needs to be a string like "A-" for A flat major, "b" for b minor, etc. """
    return key[0].isupper()


def _flat_alteration(note):
    """ Ex: _flat_alteration(G) = G-,  _flat_alteration(G+) = G """
    return note[:-1] if '+' in note else note + '-'


def load_notes(i, fpq=8):
    """
    Load notes in each piece, which is then represented as piano roll.
    :param i: which sonata to take (sonatas indexed from 0 to 31)
    :param fpq: frames per quarter note, default =  8 (that is, 32th note as 1 unit in piano roll)
    :return: pieces, tdeviation
    """

    dt = [('onset', 'float'), ('pitch', 'int'), ('mPitch', 'int'), ('duration', 'float'), ('staffNum', 'int'),
          ('measure', 'int')]  # datatype

    notes_file = os.path.join(DATASET_FOLDER, str(i + 1).zfill(2), "notes.csv")
    notes = np.genfromtxt(notes_file, delimiter=',', dtype=dt)  # read notes from .csv file
    # length of the piece in piano roll frames, assuming the last note to stay was amongst the 20 last to be played
    length = math.ceil((max(notes[-20:]['onset'] + notes[-20:]['duration']) - notes[0]['onset']) * fpq)
    # TODO: Change tdev to be defined without the absolute value to make sure that it behaves correctly
    #   also when it is anyway positive (not sure if this can happen at all, but it's anyway simpler conceptually)
    tdev = abs(notes[0]['onset'])
    piano_roll = np.zeros(shape=[128, length], dtype=np.int32)
    for note in notes:
        pitch = note['pitch']
        start = int(round((note['onset'] + tdev) * fpq))
        end = start + min(int(round(note['duration'] * fpq)), 1)
        time = range(start, end)
        piano_roll[pitch, time] = 1  # add note to piano_roll
    return piano_roll, tdev


def load_chord_labels(i):
    """
    Load chords of each piece and add chord symbols into the labels.
    :param i: which sonata to take (sonatas indexed from 0 to 31)
    :return: chord_labels
    """

    dt = [('onset', 'float'), ('end', 'float'), ('key', '<U10'), ('degree', '<U10'), ('quality', '<U10'),
          ('inversion', 'int'), ('chord_function', '<U10')]  # datatype
    chords_file = os.path.join(DATASET_FOLDER, str(i + 1).zfill(2), "chords.xlsx")

    workbook = xlrd.open_workbook(chords_file)
    sheet = workbook.sheet_by_index(0)
    chords = []
    for rowx in range(sheet.nrows):
        cols = sheet.row_values(rowx)
        # xlrd.open_workbook automatically casts strings to float if they are compatible. Revert this.
        if isinstance(cols[3], float):  # if type(degree) == float
            cols[3] = str(int(cols[3]))
        chords.append(tuple(cols))
    chords = np.array(chords, dtype=dt)  # convert to structured array

    return _add_chord_symbol(chords)


def _add_chord_symbol(chords):
    """
    Translate roman numeral representations into chord symbols.
    :param chords:
    :return: chords_full
    """

    # Translate chords
    outputQ = {'M': 'M', 'm': 'm', 'M7': 'M7', 'm7': 'm7', 'D7': '7', 'a': 'aug', 'd': 'dim', 'd7': 'dim7',
               'h7': 'm7(b5)', 'a6': '7'}  # necessary only because the data is stored in non-standard notation
    chord_symbols = []
    for chord in chords:
        key = chord['key']
        str_degree = chord['degree']

        # FIND THE ROOT OF THE CHORD
        if len(str_degree) == 1 or str_degree == '1+':  # case: degree = x, 1+
            degree = int(str_degree[0])
            root = SCALES[key][degree - 1]

        elif str_degree == '+4':  # case: augmented 6th
            degree = 6
            root = SCALES[key][degree - 1]
            if _is_major(key):  # case: major key
                root = _flat_alteration(root)  # lower the sixth in a major key

        elif str_degree == '-2' or str_degree == '-6':  # case: neapolitan chord or 6b (?)
            degree = int(str_degree[1])
            root = SCALES[key][degree - 1]
            root = _flat_alteration(root)  # the fundamental of the chord is lowered

        elif '/' in str_degree:  # case: secondary chord
            degree = str_degree
            n = int(degree.split('/')[0]) if '+' not in degree.split('/')[0] else 6  # take care of augmented 6th chords
            d = int(degree.split('/')[1])  # denominator
            key2 = SCALES[key][abs(d) - 1]  # secondary key
            if d < 0:
                key2 = _flat_alteration(key2)

            root = SCALES[key2][n - 1]
            if '+' in degree.split('/')[0]:  # augmented 6th chords
                if _is_major(key2):  # case: major key
                    root = _flat_alteration(root)
        else:
            raise ValueError(f"Can't understand the following chord degree: {str_degree}")

        # Re-translate root for enharmonic equivalence
        if '++' in root:
            if 'B' in root or 'E' in root:
                root = ROOTS[(ROOTS.index(root[0]) + 1) % 7] + '+'
            else:
                root = ROOTS[ROOTS.index(root[0]) + 1]  # no problem when index == 6 because that's the case B++
        elif '--' in root:  # if root = x--
            if 'C' in root or 'F' in root:
                root = ROOTS[ROOTS.index(root[0]) - 1] + '-'
            else:
                root = ROOTS[ROOTS.index(root[0]) - 1]

        if root == 'F-' or root == 'C-':
            root = ROOTS[ROOTS.index(root[0]) - 1]
        elif root == 'E+' or root == 'B+':
            root = ROOTS[(ROOTS.index(root[0]) + 1) % 7]

        quality = outputQ[chord['quality']]
        chord_symbol = root + quality
        chord_symbols.append(chord_symbol)

    chord_symbols = np.array(chord_symbols, dtype=[('chord_symbol', '<U10')])
    chords_full = rfn.merge_arrays((chords, chord_symbols), flatten=True, usemask=False)
    return chords_full


# def shift_chord_labels(chord_labels, s):
#     """
#
#     :param chord_labels:
#     :param s:
#     :return:
#     """
#     for i in range(len(chord_labels)):
#         temp = chord_labels[i]['chord_symbol']
#         temp =
#         chord_labels[i]['chord_symbol'] = temp
#
#     return


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


def segment_chord_labels(chord_labels, n_frames, td, label_type, wsize=32, hsize=4, fpq=8):
    if label_type not in ['chord_symbol', 'chord_function']:
        raise ValueError(f"label_type should be either \'chord_symbol\' or \'chord_function\', not {label_type}.")

    # Get corresponding chord label (only chord symbol) for each segment
    labels = []
    for n in range(n_frames):
        seg_time = (n * hsize + 0.5 * wsize) / fpq - td  # central time of the segment in quarter notes
        # print(fonset, fend, fcenter)
        label = chord_labels[np.logical_and(chord_labels['onset'] <= seg_time, chord_labels['end'] > seg_time)]
        try:
            label = label[0]
        except IndexError:
            raise HarmonicAnalysisError(f"Cannot read label for Sonata N.{i + 1} at frame {n}")

        if label_type == 'chord_symbol':
            labels.append(label['chord_symbol'])
        else:
            labels.append((label['key'], label['degree'], label['quality'], label['inversion']))
    return labels


def tchord2onehot(labels):
    """
    Convert chord symbols into one-hot vectors
    :param labels:
    :return: onehots
    """

    root_template = ['C', 'C+', 'D', 'D+', 'E', 'F', 'F+', 'G', 'G+', 'A', 'A+', 'B']
    onehots = []
    for label in labels:
        chord_names = label['chord_names']
        onehot = [0 for _ in range(25)]
        if '+' not in chord_names and '-' not in chord_names:
            root = chord_names[0]
            quality = chord_names[1:]
        else:
            root = chord_names[:2]
            quality = chord_names[2:]

        chord_hot = root_template.index(root)
        if quality in ['M', 'm', 'M7', 'm7', '7']:
            if quality == 'm' or quality == 'm7':
                chord_hot += 12
        else:
            chord_hot = 24

        onehot[chord_hot] = 1
        onehots.append(onehot)

    return onehots


def rchord2onehot(chords):
    # Translate chords to onehot vectors
    tonic_template = ['C', 'C+', 'D', 'D+', 'E', 'F', 'F+', 'G', 'G+', 'A', 'A+', 'B']
    tonic_translation_dict = {'C-': 'B', 'D-': 'C+', 'E-': 'D+', 'E+': 'F', 'F-': 'E', 'G-': 'F+', 'A-': 'G+',
                              'B-': 'A+', 'B+': 'C'}
    quality_template = ['M', 'm', 'd', 'a', 'M7', 'm7', 'D7', 'd7', 'h7', 'a6']
    one_hot_vectors = []
    for chord in chords:

        # Get attributes in chord labels
        key = str(chord['key'])
        degree = str(chord['degree'])
        quality = str(chord['quality'])
        inversion = int(chord['inversion'])

        # Translate key to one-hot vector
        key_vector = [0 for _ in range(24)]  # 24 major and minor modes, 0-11 for major keys, 12-23 for minor keys
        tonic = key.capitalize()
        if tonic in tonic_translation_dict.keys():
            tonic = tonic_translation_dict[tonic]
        tonic_hot = tonic_template.index(tonic)
        # check mode
        if key[0].islower():
            tonic_hot += 12
        key_vector[tonic_hot] = 1

        # Translate degree to one-hot vector
        degree_numerator_vector = [0 for _ in range(
            21)]  # (7 diatonics *  3 chromatics  = 21: {0-6 diatonic, 7-13 sharp, 14-20 flat})
        degree_denominator_vector = [0 for _ in range(
            21)]  # (7 diatonics *  3 chromatics  = 21: {0-6 diatonic, 7-13 sharp, 14-20 flat})
        # check numerator and denominator of degree
        if '/' not in degree:
            denominator = 1
            numerator = translate_degree(degree)
        else:
            numarator_str = degree.split('/')[0]
            denominator_str = degree.split('/')[1]
            numerator = translate_degree(numarator_str)
            denominator = translate_degree(denominator_str)
        degree_numerator_vector[numerator - 1], degree_denominator_vector[denominator - 1] = 1, 1

        # Translate quality to one-hot vector
        quality_vector = [0 for _ in range(
            10)]  # {'M': 0, 'm': 1, 'd': 2, 'a': 3, 'M7': 4, 'm7': 5, 'D7': 6, 'd7': 7, 'h7': 8, 'a6': 9}
        quality_hot = quality_template.index(quality)
        quality_vector[quality_hot] = 1

        # Translate inversion to one-hot vector
        inversion_vector = [0 for _ in range(4)]  # {'ori.':0, '1st':1, '2nd', 2, '3rd': 3}
        inversion_hot = inversion
        inversion_vector[inversion_hot] = 1

        all_vectors = (key_vector,
                       degree_denominator_vector,
                       degree_numerator_vector,
                       quality_vector,
                       inversion_vector)

        one_hot_vectors.append(all_vectors)

    dt = [('key', object), ('pri_deg', object), ('sec_deg', object), ('quality', object), ('inversion', object)]
    return np.array(one_hot_vectors, dtype=dt)


def translate_degree(degree_str):
    if ('+' not in degree_str and '-' not in degree_str) or ('+' in degree_str and degree_str[1] == '+'):
        degree_hot = int(degree_str[0])
    elif degree_str[0] == '-':
        degree_hot = int(degree_str[1]) + 14
    elif degree_str[0] == '+':
        degree_hot = int(degree_str[1]) + 7

    return degree_hot


def augment_tchords(labels_onehot):
    """
    Augment chord labels (in one-hot representation)
    :param labels_onehot:
    :return: labels_aug
    """

    labels_aug = [None for _ in range(12)]
    for m in range(len(labels_aug)):
        temp = np.array(labels_onehot)
        for i in range(temp.shape[0]):
            if temp[i][24] != 1:
                key = list(temp[i][:12]) if any(temp[i][:12]) else list(temp[i][12:24])
                mode = 0 if any(temp[i][:12]) else 1
                if m < 7:
                    shift = m
                else:
                    shift = m - 12
                temp[i] = list(np.roll(key, shift=shift)) + [0 for _ in range(12)] + [0] if mode == 0 \
                    else [0 for _ in range(12)] + list(np.roll(key, shift=shift)) + [0]
        labels_aug[m] = temp

    return labels_aug


def augment_rchords(labels_onehot):
    labels_aug = [None for _ in range(12)]
    for m in range(len(labels_aug)):
        temp = np.array(labels_onehot)
        for i in range(temp.shape[0]):
            key = list(temp[i]['key'][:12]) if any(temp[i]['key'][:12]) else list(temp[i]['key'][12:])
            mode = 0 if any(temp[i]['key'][:12]) else 1  # major -> 0, minor -> 1
            if m < 7:
                shift = m
            else:
                shift = m - 12
            temp[i]['key'] = list(np.roll(key, shift=shift)) + [0 for _ in range(12)] if mode == 0 else [0 for _ in
                                                                                                         range(
                                                                                                             12)] + list(
                np.roll(key, shift=shift))
        labels_aug[m] = temp

    return labels_aug


def prepare_input_data(segments_pianoroll, segments_label, label_type, num_steps=64, feature_size=61 * 32, hop=32):
    """
    Rearrange segments_pianoroll and segments_label into the format [num_sequences, num_steps, feature_size] and [num_sequences, num_steps, num_classes] respectively
    :param segments_pianoroll:
    :param segments_label:
    :param hop: hop size of sequences, default = 32 (4 beats)
    :param num_steps: number of RNN time steps
    :param feature_size: input feature size
    :param label_type: string, 'chord_symbol' and 'chord_function'  are valid
    :return: input_segments, input_labels
    """

    if label_type not in ['chord_symbol', 'chord_function']:
        raise ValueError(f"label_type should be either \'chord_symbol\' or \'chord_function\', not {label_type}.")
    input_segments = [[None for _ in range(32)] for _ in range(12)]
    input_labels = [[None for _ in range(32)] for _ in range(12)]
    for m in range(12):  # the transpositions
        for p in range(32):  # the sonatas
            indices = list(range(len(segments_pianoroll[m][p])))  # indices of segments in the piece with m modulation
            seq_indices = [indices[i:i + num_steps] for i in itertools.takewhile(lambda x: x + num_steps < len(indices),
                                                                                 range(0, len(indices),
                                                                                       hop))]  # split indices into sequences of length n_steps with hop size = hop
            if (len(indices) - num_steps) / hop != 0:
                seq_indices.append(indices[-num_steps:])
            num_sequences = len(seq_indices)

            if label_type == 'chord_symbol':
                inputs = np.zeros(shape=(num_sequences, num_steps, feature_size), dtype=np.float32)
                labels = np.zeros(shape=(num_sequences, num_steps), dtype=np.int32)
                for n in range(num_sequences):
                    inputs[n, :, :] = [segments_pianoroll[m][p][index] for index in seq_indices[n]]
                    labels[n, :] = [np.argmax(vector) for vector in segments_label[m][p][seq_indices[n]]]
            elif label_type == 'chord_function':
                inputs = np.zeros(shape=(num_sequences, num_steps, feature_size), dtype=np.float32)
                dt = [('key', 'int'), ('pri_deg', 'int'), ('sec_deg', 'int'), ('quality', 'int'), ('inversion', 'int')]
                labels = np.zeros(shape=(num_sequences, num_steps), dtype=dt)
                for n in range(num_sequences):
                    inputs[n, :, :] = [segments_pianoroll[m][p][index] for index in seq_indices[n]]
                    labels[n, :]['key'] = [np.argmax(vector) for vector in segments_label[m][p]['key'][seq_indices[n]]]
                    labels[n, :]['pri_deg'] = [np.argmax(vector) for vector in
                                               segments_label[m][p]['pri_deg'][seq_indices[n]]]
                    labels[n, :]['sec_deg'] = [np.argmax(vector) for vector in
                                               segments_label[m][p]['sec_deg'][seq_indices[n]]]
                    labels[n, :]['quality'] = [np.argmax(vector) for vector in
                                               segments_label[m][p]['quality'][seq_indices[n]]]
                    labels[n, :]['inversion'] = [np.argmax(vector) for vector in
                                                 segments_label[m][p]['inversion'][seq_indices[n]]]
            else:
                print('LabelTypeError: %s,' % label_type,
                      'label_type should be \'chord_symbol\' or \'chord_function\'.')
                quit()
            input_segments[m][p] = inputs
            input_labels[m][p] = labels

    return input_segments, input_labels


def split_input_data(input_segments, input_labels):
    # split 32 pieces into three sets
    inputs_train = np.concatenate([input_segments[m][p] for m in range(12) for p in TRAIN_INDICES], axis=0)
    inputs_valid = np.concatenate([input_segments[0][p] for p in VALID_INDICES], axis=0)
    inputs_test = np.concatenate([input_segments[0][p] for p in TEST_INDICES], axis=0)

    labels_train = np.concatenate([input_labels[m][p] for m in range(12) for p in TRAIN_INDICES], axis=0)
    labels_valid = np.concatenate([input_labels[0][p] for p in VALID_INDICES], axis=0)
    labels_test = np.concatenate([input_labels[0][p] for p in TEST_INDICES], axis=0)

    return inputs_train, inputs_valid, inputs_test, labels_train, labels_valid, labels_test


"""
x is input data, y is label;
x has the shape [num_sequences, num_steps, feature_size];
if label_type == 'chord_symbol',
    y has the shape [num_sequences, num_steps];
if label_type == 'chord_function',
    y has the shape [num_sequences, num_steps],
    and chord functions can be access by y[num_sequences, num_steps][function_name],
    where 'key', 'pri_deg', 'sec_deg', 'quality', 'inversion' are valid function_name
"""
label_type = 'chord_function'
hsize = 4  # hopping size between frames in 32nd notes
wsize = 32  # window size for a frame in 32nd notes
fpq = 8  # number of frames per quarter note

for indices, output_file in zip([TRAIN_INDICES, VALID_INDICES, TEST_INDICES],
                                [TRAIN_TFRECORDS, VALID_TFRECORDS, TEST_TFRECORDS]):
    with tf.io.TFRecordWriter(output_file) as writer:
        logger.info(f'Working on {os.path.basename(output_file)}.')

        for i in indices:
            logger.info(f"Sonata N.{i + 1}")
            piano_roll, td = load_notes(i)
            chord_labels = load_chord_labels(i)
            n_frames = int(math.ceil((piano_roll.shape[1] - wsize) / hsize)) + 1

            for s in range(-6, 6):
                pr_shifted = np.roll(piano_roll, shift=s, axis=0)
                pr_segments = segment_piano_roll(pr_shifted, n_frames, wsize=wsize, hsize=hsize)

                # cl_shifted = shift_chord_labels(chord_labels, s)
                cl_segments = segment_chord_labels(chord_labels, n_frames, td, label_type, wsize=wsize, hsize=hsize,
                                                   fpq=fpq)

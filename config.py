import tensorflow as tf
import os
from math import ceil

MODES = [
    'pitch_spelling',
    'pitch_class',
    'pitch_class_beat_strength',
    'midi_number',
    'pitch_spelling_cut'
]

MODE = MODES[4]

TRAIN_INDICES = [5, 12, 17, 21, 27, 32, 4, 9, 13, 18, 24, 22, 28, 30, 31, 11, 2, 3, 1, 14, 23, 15, 10, 25, 7]
VALID_INDICES = [8, 19, 29, 16, 26, 6, 20]
DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
TRAIN_TFRECORDS = os.path.join(DATA_FOLDER, f'train_{MODE}.tfrecords')
VALID_TFRECORDS = os.path.join(DATA_FOLDER, f'valid_{MODE}.tfrecords')

HSIZE = 4  # hopping size between frames in 32nd notes, equivalent to 2 frames per quarter note
FPQ = 8  # number of frames per quarter note with 32nd note quantization (check: HSIZE * FPQ = 32)
PITCH_LOW = 18  # lowest midi pitch used, as returned by preprocessing.find_pitch_extremes()
PITCH_HIGH = 107  # lowest midi pitch not used, i.e., piano_roll = piano_roll[PITCH_LOW:PITCH_HIGH]
N_PITCHES = PITCH_HIGH - PITCH_LOW  # number of pitches kept out of total 128 midi pitches

FEATURES = ['key', 'degree 1', 'degree 2', 'quality', 'inversion', 'root']
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
PITCH_LINE = ['F--', 'C--', 'G--', 'D--', 'A--', 'E--', 'B--',
              'F-', 'C-', 'G-', 'D-', 'A-', 'E-', 'B-',
              'F', 'C', 'G', 'D', 'A', 'E', 'B',
              'F#', 'C#', 'G#', 'D#', 'A#', 'E#', 'B#',
              'F##', 'C##', 'G##', 'D##', 'A##', 'E##', 'B##']
SCALES = {
    'C--': ['C--', 'D--', 'E--', 'F--', 'G--', 'A--', 'B--'],
    # 'c--': ['C--', 'D--', 'E---', 'F--', 'G--', 'A---', 'B--'],
    'G--': ['G--', 'A--', 'B--', 'C--', 'D--', 'E--', 'F-'],
    # 'g--': ['G--', 'A--', 'B---', 'C--', 'D--', 'E---', 'F-'],
    'D--': ['D--', 'E--', 'F-', 'G--', 'A--', 'B--', 'C-'],  # 'd--': ['D--', 'E--', 'F--', 'G--', 'A--', 'B---', 'C-'],
    'A--': ['A--', 'B--', 'C-', 'D--', 'E--', 'F-', 'G-'], 'a--': ['A--', 'B--', 'C--', 'D--', 'E--', 'F--', 'G-'],
    'E--': ['E--', 'F-', 'G-', 'A--', 'B--', 'C-', 'D-'], 'e--': ['E--', 'F-', 'G--', 'A--', 'B--', 'C--', 'D-'],
    'B--': ['B--', 'C-', 'D-', 'E--', 'F-', 'G-', 'A-'], 'b--': ['B--', 'C-', 'D--', 'E--', 'F-', 'G--', 'A-'],
    'F-': ['F-', 'G-', 'A-', 'B--', 'C-', 'D-', 'E-'], 'f-': ['F-', 'G-', 'A--', 'B--', 'C-', 'D--', 'E-'],
    'C-': ['C-', 'D-', 'E-', 'F-', 'G-', 'A-', 'B-'], 'c-': ['C-', 'D-', 'E--', 'F-', 'G-', 'A--', 'B-'],
    'G-': ['G-', 'A-', 'B-', 'C-', 'D-', 'E-', 'F'], 'g-': ['G-', 'A-', 'B--', 'C-', 'D-', 'E--', 'F'],
    'D-': ['D-', 'E-', 'F', 'G-', 'A-', 'B-', 'C'], 'd-': ['D-', 'E-', 'F-', 'G-', 'A-', 'B--', 'C'],
    'A-': ['A-', 'B-', 'C', 'D-', 'E-', 'F', 'G'], 'a-': ['A-', 'B-', 'C-', 'D-', 'E-', 'F-', 'G'],
    'E-': ['E-', 'F', 'G', 'A-', 'B-', 'C', 'D'], 'e-': ['E-', 'F', 'G-', 'A-', 'B-', 'C-', 'D'],
    'B-': ['B-', 'C', 'D', 'E-', 'F', 'G', 'A'], 'b-': ['B-', 'C', 'D-', 'E-', 'F', 'G-', 'A'],
    'F': ['F', 'G', 'A', 'B-', 'C', 'D', 'E'], 'f': ['F', 'G', 'A-', 'B-', 'C', 'D-', 'E'],
    'C': ['C', 'D', 'E', 'F', 'G', 'A', 'B'], 'c': ['C', 'D', 'E-', 'F', 'G', 'A-', 'B'],
    'G': ['G', 'A', 'B', 'C', 'D', 'E', 'F#'], 'g': ['G', 'A', 'B-', 'C', 'D', 'E-', 'F#'],
    'D': ['D', 'E', 'F#', 'G', 'A', 'B', 'C#'], 'd': ['D', 'E', 'F', 'G', 'A', 'B-', 'C#'],
    'A': ['A', 'B', 'C#', 'D', 'E', 'F#', 'G#'], 'a': ['A', 'B', 'C', 'D', 'E', 'F', 'G#'],
    'E': ['E', 'F#', 'G#', 'A', 'B', 'C#', 'D#'], 'e': ['E', 'F#', 'G', 'A', 'B', 'C', 'D#'],
    'B': ['B', 'C#', 'D#', 'E', 'F#', 'G#', 'A#'], 'b': ['B', 'C#', 'D', 'E', 'F#', 'G', 'A#'],
    'F#': ['F#', 'G#', 'A#', 'B', 'C#', 'D#', 'E#'], 'f#': ['F#', 'G#', 'A', 'B', 'C#', 'D', 'E#'],
    'C#': ['C#', 'D#', 'E#', 'F#', 'G#', 'A#', 'B#'], 'c#': ['C#', 'D#', 'E', 'F#', 'G#', 'A', 'B#'],
    'G#': ['G#', 'A#', 'B#', 'C#', 'D#', 'E#', 'F##'], 'g#': ['G#', 'A#', 'B', 'C#', 'D#', 'E', 'F##'],
    'D#': ['D#', 'E#', 'F##', 'G#', 'A#', 'B#', 'C##'], 'd#': ['D#', 'E#', 'F#', 'G#', 'A#', 'B', 'C##'],
    'A#': ['A#', 'B#', 'C##', 'D#', 'E#', 'F##', 'G##'], 'a#': ['A#', 'B#', 'C#', 'D#', 'E#', 'F#', 'G##'],
    'E#': ['E#', 'F##', 'G##', 'A#', 'B#', 'C##', 'D##'], 'e#': ['E#', 'F##', 'G#', 'A#', 'B#', 'C#', 'D##'],
    'B#': ['B#', 'C##', 'D##', 'E#', 'F##', 'G##', 'A##'], 'b#': ['B#', 'C##', 'D#', 'E#', 'F##', 'G#', 'A##'],
    'F##': ['F##', 'G##', 'A##', 'B#', 'C##', 'D##', 'E##'], 'f##': ['F##', 'G##', 'A#', 'B#', 'C##', 'D#', 'E##'],
    'C##': ['C##', 'D##', 'E##', 'F##', 'G##', 'A##', 'B##'], 'c##': ['C##', 'D##', 'E#', 'F##', 'G##', 'A#', 'B##'],
}
QUALITY = ['M', 'm', 'd', 'a', 'M7', 'm7', 'D7', 'd7', 'h7', 'Gr+6', 'It+6', 'Fr+6']

CLASSES_BASS = 12  # the twelve notes without enharmonic duplicates
CLASSES_KEY = 55 if MODE == 'pitch_spelling' else 24  # Major keys: 0-11, Minor keys: 12-23
CLASSES_DEGREE = 21  # 7 degrees * 3: regular, diminished, augmented
CLASSES_ROOT = 35 if MODE == 'pitch_spelling' else 12  # the twelve notes without enharmonic duplicates
CLASSES_QUALITY = 12  # ['M', 'm', 'd', 'a', 'M7', 'm7', 'D7', 'd7', 'h7', 'Gr+6', 'It+6', 'Fr+6']
CLASSES_INVERSION = 4  # root position, 1st, 2nd, and 3rd inversion (the last only for seventh chords)
CLASSES_TOTAL = CLASSES_KEY + CLASSES_DEGREE * 2 + CLASSES_QUALITY + CLASSES_INVERSION + CLASSES_ROOT

KEYS_SPELLING = PITCH_LINE[1:30] + [p.lower() for p in PITCH_LINE[4:-5]]
NOTES_FLAT = ['C', 'C#', 'D', 'E-', 'E', 'F', 'F#', 'G', 'A-', 'A', 'B-', 'B']
KEYS_PITCH_CLASS = (NOTES_FLAT + [n.lower() for n in NOTES_FLAT])
TICK_LABELS = [
    KEYS_PITCH_CLASS if MODE != 'pitch_spelling' else KEYS_SPELLING,
    [str(x + 1) for x in range(7)] + [str(x + 1) + 'b' for x in range(7)] + [str(x + 1) + '#' for x in range(7)],
    [str(x + 1) for x in range(7)] + [str(x + 1) + 'b' for x in range(7)] + [str(x + 1) + '#' for x in range(7)],
    QUALITY,
    [str(x) for x in range(4)],
    NOTES if MODE != 'pitch_spelling' else PITCH_LINE,
]


def count_records(tfrecord):
    """ Count the number of lines in a tfrecord file. This is useful to establish 'steps_per_epoch' when training """
    c = 0
    for _ in tf.io.tf_record_iterator(tfrecord):
        c += 1
    return c


# number of records in the training dataset as coming from the utils.count_tfrecords function
N_TRAIN = count_records(TRAIN_TFRECORDS)  # if MODE == 'pitch_spelling' else 300
N_VALID = count_records(
    VALID_TFRECORDS)  # number of records in the validation dataset as coming from the utils.count_tfrecords function

BATCH_SIZE = 16  # 1


def find_validation_batch_size(n, bs):
    if not isinstance(n, int) or n < 1:
        raise ValueError("n should be a positive integer")

    found = False
    while not found and bs > 1:
        if n % bs == 0:
            found = True
        else:
            bs -= 1
    return bs


VALID_BATCH_SIZE = find_validation_batch_size(N_VALID, BATCH_SIZE)
SHUFFLE_BUFFER = 123  # 100_000
EPOCHS = 100
STEPS_PER_EPOCH = ceil(N_TRAIN / BATCH_SIZE)
VALID_STEPS = ceil(N_VALID / VALID_BATCH_SIZE)
MODE2INPUT_SHAPE = {
    'pitch_class': 24,
    'pitch_class_weighted_loss': 24,
    'pitch_class_beat_strength': 27,
    'pitch_spelling': 70,
    'pitch_spelling_cut': 70,
    'midi_number': N_PITCHES,
}

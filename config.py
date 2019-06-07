import os

DATASET_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'BPS_FH_Dataset')
TRAIN_INDICES = [5, 12, 17, 21, 27, 32, 4, 9, 13, 18, 24, 22, 28, 30, 31, 11, 2, 3]
VALID_INDICES = [8, 19, 29, 16, 26, 6, 20]
TEST_INDICES = [1, 14, 23, 15, 10, 25, 7]
DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
TRAIN_TFRECORDS = os.path.join(DATA_FOLDER, 'train.tfrecords')
VALID_TFRECORDS = os.path.join(DATA_FOLDER, 'valid.tfrecords')
TEST_TFRECORDS = os.path.join(DATA_FOLDER, 'test.tfrecords')

HSIZE = 4  # hopping size between frames in 32nd notes
FPQ = 8  # number of frames per quarter note with 32nd note quantization (check: HSIZE * FPQ = 32)
PITCH_LOW = 18  # lowest midi pitch used, as returned by preprocessing.find_pitch_extremes()
PITCH_HIGH = 107  # lowest midi pitch not used, i.e., piano_roll = piano_roll[PITCH_LOW:PITCH_HIGH]
N_PITCHES = PITCH_HIGH - PITCH_LOW  # number of pitches kept out of total 128 midi pitches

FEATURES = ['key', 'degree 1', 'degree 2', 'quality', 'inversion', 'root', 'symbol']
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
QUALITY = ['M', 'm', 'd', 'a', 'M7', 'm7', 'D7', 'd7', 'h7', 'a6']
SYMBOL = ['M', 'm', 'M7', 'm7', '7', 'aug', 'dim', 'dim7', 'm7(b5)']  # quality as encoded in chord symbols
CLASSES_KEY = 24  # Major keys: 0-11, Minor keys: 12-23
CLASSES_DEGREE = 21  # 7 degrees * 3: regular, diminished, augmented
CLASSES_QUALITY = 10  # ['M', 'm', 'd', 'a', 'M7', 'm7', 'D7', 'd7', 'h7', 'a6']
CLASSES_INVERSION = 4  # root position, 1st, 2nd, and 3rd inversion (the last only for seventh chords)
CLASSES_ROOT = 12  # the twelve notes without enharmonic duplicates
CLASSES_SYMBOL = 10  # ['M', 'm', 'M7', 'm7', '7', 'aug', 'dim', 'dim7', 'm7(b5)']
CLASSES_TOTAL = CLASSES_KEY + CLASSES_DEGREE * 2 + CLASSES_QUALITY + CLASSES_INVERSION + CLASSES_ROOT + CLASSES_SYMBOL

BATCH_SIZE = 1
SHUFFLE_BUFFER = 100_000
EPOCHS = 100
N_TRAIN = 216  # number of records in the training dataset as coming from the utils.count_tfrecords function
N_VALIDATION = 84  # number of records in the validation dataset as coming from the utils.count_tfrecords function
N_TEST = 84  # number of records in the validation dataset as coming from the utils.count_tfrecords function
STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE
VALIDATION_STEPS = N_VALIDATION // BATCH_SIZE
TEST_STEPS = N_TEST // BATCH_SIZE

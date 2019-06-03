import os

DATASET_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'BPS_FH_Dataset')
TRAIN_INDICES = [5, 12, 17, 21, 27, 32, 4, 9, 13, 18, 24, 22, 28, 30, 31, 11, 2, 3]
VALID_INDICES = [8, 19, 29, 16, 26, 6, 20]
TEST_INDICES = [1, 14, 23, 15, 10, 25, 7]
TRAIN_TFRECORDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'train.tfrecords')
VALID_TFRECORDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'valid.tfrecords')
TEST_TFRECORDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'test.tfrecords')

HSIZE = 4  # hopping size between frames in 32nd notes
WSIZE = 32  # window size for a frame in 32nd notes
FPQ = 8  # number of frames per quarter note
PITCH_LOW = 18  # lowest midi pitch used, as returned by preprocessing.find_pitch_extremes()
PITCH_HIGH = 107  # lowest midi pitch not used, i.e., piano_roll = piano_roll[PITCH_LOW:PITCH_HIGH]
N_PITCHES = PITCH_HIGH - PITCH_LOW  # number of pitches kept out of total 128 midi pitches

CLASSES_KEY = 24  # Major keys: 0-11, Minor keys: 12-23
CLASSES_DEGREE = 21  # 7 degrees * 3: regular, diminished, augmented
CLASSES_QUALITY = 10  # ['M', 'm', 'd', 'a', 'M7', 'm7', 'D7', 'd7', 'h7', 'a6']
CLASSES_INVERSION = 4  # root position, 1st, 2nd, and 3rd inversion (the last only for seventh chords)
CLASSES_ROOT = 12  # the twelve notes without enharmonic duplicates
CLASSES_SYMBOL = 10  # ['M', 'm', 'M7', 'm7', '7', 'aug', 'dim', 'dim7', 'm7(b5)']
CLASSES_TOTAL = CLASSES_KEY + CLASSES_DEGREE * 2 + CLASSES_QUALITY + CLASSES_INVERSION + CLASSES_ROOT + CLASSES_SYMBOL

BATCH_SIZE = 64
SHUFFLE_BUFFER = 100_000
EPOCHS = 10
N_VALIDATION = 184080  # number of records in the validation dataset as coming from the utils.count_tfrecords function
STEPS_PER_EPOCH = N_VALIDATION / BATCH_SIZE

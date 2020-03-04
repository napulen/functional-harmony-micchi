"""
Configuration file. The lowest level file in our project, it should not import any other file.
"""
import os
import logging

logging.basicConfig(level=logging.INFO)

INPUT_TYPES = [
    'pitch_complete_cut',
    'pitch_bass_cut',
    'pitch_class_cut',
    'spelling_complete_cut',
    'spelling_bass_cut',
    'spelling_class_cut',
]

DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

CHUNK_SIZE = 160  # dimension of each chunk when cutting the sonatas in chord time-steps
HSIZE = 4  # hopping size between frames in 32nd notes, equivalent to 2 frames per quarter note
FPQ = 8  # number of frames per quarter note with 32nd note quantization (check: HSIZE * FPQ = 32)
PITCH_LOW = 18  # lowest midi pitch used, as returned by preprocessing.find_pitch_extremes()
PITCH_HIGH = 107  # lowest midi pitch not used, i.e., piano_roll = piano_roll[PITCH_LOW:PITCH_HIGH]
N_PITCHES = PITCH_HIGH - PITCH_LOW  # number of pitches kept out of total 128 midi pitches

FEATURES = ['key', 'degree 1', 'degree 2', 'quality', 'inversion', 'root']
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
PITCH_FIFTHS = [
    'F--', 'C--', 'G--', 'D--', 'A--', 'E--', 'B--',
    'F-', 'C-', 'G-', 'D-', 'A-', 'E-', 'B-',
    'F', 'C', 'G', 'D', 'A', 'E', 'B',
    'F#', 'C#', 'G#', 'D#', 'A#', 'E#', 'B#',
    'F##', 'C##', 'G##', 'D##', 'A##', 'E##', 'B##'
]
PITCH_SEMITONES = [
    'C--', 'C-', 'C', 'D--', 'C#', 'D-', 'C##', 'D', 'E--', 'D#', 'E-', 'F--', 'D##', 'E', 'F-', 'E#', 'F', 'G--',
    'E##', 'F#', 'G-', 'F##', 'G', 'A--', 'G#', 'A-', 'G##', 'A', 'B--', 'A#', 'B-', 'A##', 'B', 'B#', 'B##'
]

SCALES = {
    'C--': ['C--', 'D--', 'E--', 'F--', 'G--', 'A--', 'B--'],
    'c--': ['C--', 'D--', 'E---', 'F--', 'G--', 'A---', 'B--'],
    'G--': ['G--', 'A--', 'B--', 'C--', 'D--', 'E--', 'F-'], 'g--': ['G--', 'A--', 'B---', 'C--', 'D--', 'E---', 'F-'],
    'D--': ['D--', 'E--', 'F-', 'G--', 'A--', 'B--', 'C-'], 'd--': ['D--', 'E--', 'F--', 'G--', 'A--', 'B---', 'C-'],
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

I2RN = {
    'triad0': '',
    'triad1': '6',
    'triad2': '64',
    'triad3': 'wi',
    'seventh0': '7',
    'seventh1': '65',
    'seventh2': '43',
    'seventh3': '42',
}
Q2RN = {  # (True=uppercase degree, 'triad' or 'seventh', quality)
    'M': (True, 'triad', ''),
    'm': (False, 'triad', ''),
    'd': (False, 'triad', 'o'),
    'a': (True, 'triad', '+'),
    'M7': (True, 'seventh', 'M'),
    'm7': (False, 'seventh', ''),
    'D7': (True, 'seventh', ''),
    'd7': (False, 'seventh', 'o'),
    'h7': (False, 'seventh', 'ø'),
    'Gr+6': (True, 'seventh', 'Gr'),
    'It+6': (True, 'seventh', 'It'),
    'Fr+6': (True, 'seventh', 'Fr'),
}

# including START, excluding END
KEY_START_MAJ, KEY_END_MAJ, KEY_START_MIN, KEY_END_MIN = [
    PITCH_FIFTHS.index(p) for p in ['C-', 'G#', 'a-'.upper(), 'e#'.upper()]
]
KEYS_SPELLING = PITCH_FIFTHS[KEY_START_MAJ:KEY_END_MAJ] + [p.lower() for p in PITCH_FIFTHS[KEY_START_MIN:KEY_END_MIN]]

KEYS_PITCH = (NOTES + [n.lower() for n in NOTES])

# number of records in datasets
N_VALID = 162  # count_records(VALID_TFRECORDS)
N_BPS = 401  # count_records(BPS_TFRECORDS)

INPUT_TYPE2INPUT_SHAPE = {
    'pitch_complete_cut': 12 * 7,
    'pitch_bass_cut': 12 * 2,
    'pitch_class_cut': 12,
    'spelling_complete_cut': 35 * 7,
    'spelling_bass_cut': 35 * 2,
    'spelling_class_cut': 35,
}

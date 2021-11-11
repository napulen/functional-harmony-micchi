"""
Configuration file. The lowest level file in our project, it should not import any other file.
"""
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_here = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER = os.path.join(_here, "..", "data")

# **** MODEL ****
MODEL_TYPES = [
    "Algomus",
    "ConvGru",
    "ConvDil",
    "Gru",
    "ConvGruLocal",
    "ConvDilLocal",
    "ConvGruBlocknade",
]
CHUNK_SIZE = 80  # dimension of each chunk in crotchets
HOP_SIZE = 40  # hop size between consecutive chunks in crotchets
INPUT_FPC = 8  # number of frames per crotchet in the input (piano roll)
OUTPUT_FPC = 2  # number of frames per crotchet in the output (chords)

# **** INPUT ****
INPUT_TYPES = [
    "pitch_complete",
    "pitch_bass",
    "pitch_class",
    "spelling_complete",
    "spelling_bass",
    "spelling_class",
]
INPUT_FEATURES = ["piano_roll", "structure", "mask"]
INPUT_TYPE2INPUT_SHAPE = {
    "pitch_complete": 12 * 7,
    "pitch_bass": 12 * 2,
    "pitch_class": 12,
    "spelling_complete": 35 * 7,
    "spelling_bass": 35 * 2,
    "spelling_class": 35,
}


FROG_MODEL_PATH = os.path.join(_here, "resources", "frog_model")

# **** MUSIC ****
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
N2I = {x: i for i, x in enumerate(NOTES)}
# PITCH_FIFTHS follow the pattern F--, C--, G--, D--, A--, E--, B--, F-, C-, ..., E##, B##
PITCH_FIFTHS = [name + alt for alt in ["--", "-", "", "#", "##"] for name in "FCGDAEB"]
# TODO: Is there a more compact way to represent PITCH_SEMITONES?
#  Also, in the unlikely case of enharmonicity, what should take precedence?
PITCH_SEMITONES = [
    "C--",
    "C-",
    "C",
    "D--",
    "C#",
    "D-",
    "C##",
    "D",
    "E--",
    "D#",
    "E-",
    "F--",
    "D##",
    "E",
    "F-",
    "E#",
    "F",
    "G--",
    "E##",
    "F#",
    "G-",
    "F##",
    "G",
    "A--",
    "G#",
    "A-",
    "G##",
    "A",
    "B--",
    "A#",
    "B-",
    "A##",
    "B",
    "B#",
    "B##",
]
SCALES = {
    "C--": ["C--", "D--", "E--", "F--", "G--", "A--", "B--"],
    "c--": ["C--", "D--", "E---", "F--", "G--", "A---", "B--"],
    "G--": ["G--", "A--", "B--", "C--", "D--", "E--", "F-"],
    "g--": ["G--", "A--", "B---", "C--", "D--", "E---", "F-"],
    "D--": ["D--", "E--", "F-", "G--", "A--", "B--", "C-"],
    "d--": ["D--", "E--", "F--", "G--", "A--", "B---", "C-"],
    "A--": ["A--", "B--", "C-", "D--", "E--", "F-", "G-"],
    "a--": ["A--", "B--", "C--", "D--", "E--", "F--", "G-"],
    "E--": ["E--", "F-", "G-", "A--", "B--", "C-", "D-"],
    "e--": ["E--", "F-", "G--", "A--", "B--", "C--", "D-"],
    "B--": ["B--", "C-", "D-", "E--", "F-", "G-", "A-"],
    "b--": ["B--", "C-", "D--", "E--", "F-", "G--", "A-"],
    "F-": ["F-", "G-", "A-", "B--", "C-", "D-", "E-"],
    "f-": ["F-", "G-", "A--", "B--", "C-", "D--", "E-"],
    "C-": ["C-", "D-", "E-", "F-", "G-", "A-", "B-"],
    "c-": ["C-", "D-", "E--", "F-", "G-", "A--", "B-"],
    "G-": ["G-", "A-", "B-", "C-", "D-", "E-", "F"],
    "g-": ["G-", "A-", "B--", "C-", "D-", "E--", "F"],
    "D-": ["D-", "E-", "F", "G-", "A-", "B-", "C"],
    "d-": ["D-", "E-", "F-", "G-", "A-", "B--", "C"],
    "A-": ["A-", "B-", "C", "D-", "E-", "F", "G"],
    "a-": ["A-", "B-", "C-", "D-", "E-", "F-", "G"],
    "E-": ["E-", "F", "G", "A-", "B-", "C", "D"],
    "e-": ["E-", "F", "G-", "A-", "B-", "C-", "D"],
    "B-": ["B-", "C", "D", "E-", "F", "G", "A"],
    "b-": ["B-", "C", "D-", "E-", "F", "G-", "A"],
    "F": ["F", "G", "A", "B-", "C", "D", "E"],
    "f": ["F", "G", "A-", "B-", "C", "D-", "E"],
    "C": ["C", "D", "E", "F", "G", "A", "B"],
    "c": ["C", "D", "E-", "F", "G", "A-", "B"],
    "G": ["G", "A", "B", "C", "D", "E", "F#"],
    "g": ["G", "A", "B-", "C", "D", "E-", "F#"],
    "D": ["D", "E", "F#", "G", "A", "B", "C#"],
    "d": ["D", "E", "F", "G", "A", "B-", "C#"],
    "A": ["A", "B", "C#", "D", "E", "F#", "G#"],
    "a": ["A", "B", "C", "D", "E", "F", "G#"],
    "E": ["E", "F#", "G#", "A", "B", "C#", "D#"],
    "e": ["E", "F#", "G", "A", "B", "C", "D#"],
    "B": ["B", "C#", "D#", "E", "F#", "G#", "A#"],
    "b": ["B", "C#", "D", "E", "F#", "G", "A#"],
    "F#": ["F#", "G#", "A#", "B", "C#", "D#", "E#"],
    "f#": ["F#", "G#", "A", "B", "C#", "D", "E#"],
    "C#": ["C#", "D#", "E#", "F#", "G#", "A#", "B#"],
    "c#": ["C#", "D#", "E", "F#", "G#", "A", "B#"],
    "G#": ["G#", "A#", "B#", "C#", "D#", "E#", "F##"],
    "g#": ["G#", "A#", "B", "C#", "D#", "E", "F##"],
    "D#": ["D#", "E#", "F##", "G#", "A#", "B#", "C##"],
    "d#": ["D#", "E#", "F#", "G#", "A#", "B", "C##"],
    "A#": ["A#", "B#", "C##", "D#", "E#", "F##", "G##"],
    "a#": ["A#", "B#", "C#", "D#", "E#", "F#", "G##"],
    "E#": ["E#", "F##", "G##", "A#", "B#", "C##", "D##"],
    "e#": ["E#", "F##", "G#", "A#", "B#", "C#", "D##"],
    "B#": ["B#", "C##", "D##", "E#", "F##", "G##", "A##"],
    "b#": ["B#", "C##", "D#", "E#", "F##", "G#", "A##"],
    "F##": ["F##", "G##", "A##", "B#", "C##", "D##", "E##"],
    "f##": ["F##", "G##", "A#", "B#", "C##", "D#", "E##"],
    "C##": ["C##", "D##", "E##", "F##", "G##", "A##", "B##"],
    "c##": ["C##", "D##", "E#", "F##", "G##", "A#", "B##"],
}
PF2I = {x: i for i, x in enumerate(PITCH_FIFTHS)}
PF2PS = {i: PITCH_SEMITONES.index(x) for i, x in enumerate(PITCH_FIFTHS)}
_PS2I = {x: i for i, x in enumerate(PITCH_SEMITONES)}
_PS2PF = {i: PITCH_FIFTHS.index(x) for i, x in enumerate(PITCH_SEMITONES)}


def find_enharmonic_equivalent(note):
    """
    Transform everything into a note with at most one sharp and no flats.
    Keeps the upper- or lower-case intact.

    :param note:
    :return:
    """
    if note is None:
        logger.warning("Trying to find the enharmonic equivalent of None. Returning None")
        return None
    note_up = note.upper()

    if note_up in NOTES:
        return note

    n, alt = N2I[note_up[0]], list(note_up[1:])
    while alt:
        x = alt.pop(-1)
        n = n + 1 if x == "#" else n - 1
    n = n % 12  # the notes are circular!
    return NOTES[n] if note.isupper() else NOTES[n].lower()


def _flat_alteration(note):
    """ Ex: _flat_alteration(G) = G-,  _flat_alteration(G#) = G """
    return note[:-1] if "#" in note else note + "-"


def _sharp_alteration(note):
    """ Ex: _sharp_alteration(G) = G#,  _sharp_alteration(G-) = G """
    return note[:-1] if "-" in note else note + "#"

"""
create a tfrecord containing the following features:
x the piano roll input data, with shape [n_frames, pitches]
label_key the local key of the music
label_degree_primary the chord degree with respect to the key, possibly fractional, e.g. V/V dominant of dominant
label_degree_secondary the chord degree with respect to the key, possibly fractional, e.g. V/V dominant of dominant
label_quality e.g. m, M, D7 for minor, major, dominant 7th etc.
label_inversion, from 0 to 3 depending on what note is at the bass
label_root, the root of the chord in jazz notation
label_symbol, the quality of the chord in jazz notation
sonata, the index of the sonata that is analysed
transposed, the number of semitones of transposition (negative for down-transposition)

ATTENTION: despite the name, the secondary_degree is actually "more important" than the primary degree,
since the latter is almost always equal to 1.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xlrd
from music21 import converter, note
from music21.chord import Chord
from music21.repeat import ExpanderException

from config import BPS_FH_FOLDER, PITCH_LOW, NOTES, PITCH_LINE, QUALITY, ROOTS, SCALES

F2S = dict()
N2I = dict([(e[1], e[0]) for e in enumerate(NOTES)])
Q2I = dict([(e[1], e[0]) for e in enumerate(QUALITY)])
P2I = dict([(e[1], e[0]) for e in enumerate(PITCH_LINE)])


class HarmonicAnalysisError(Exception):
    """ Raised when the harmonic analysis associated to a certain time can't be found """
    pass


def _load_score(i, fpq):
    score_file = os.path.join(BPS_FH_FOLDER, str(i).zfill(2), "score.mxl")
    try:
        score = converter.parse(score_file).expandRepeats()
    except ExpanderException:
        score = converter.parse(score_file)
        print("Could not expand repeats. Maybe there are no repeats in the piece? Please check.")
    n_frames = int(score.duration.quarterLength * fpq)
    measure_offset = list(score.measureOffsetMap().keys())
    measure_length = np.diff(measure_offset)
    t0 = - measure_offset[1] if measure_length[0] != measure_length[1] else 0  # Pickup time
    return score, t0, n_frames


def load_score_beat_strength(i, fpq):
    """

    :param i:
    :param fpq:
    :return:
    """
    score, _, n_frames = _load_score(i, fpq)
    # Throw away all notes in the score, we shouldn't use this score for anything except finding beat strength structure
    score = score.template()
    # Insert fake notes and use them to derive the beat strength through music21
    n = note.Note('C')
    n.duration.quarterLength = 1. / fpq
    offsets = np.arange(n_frames) / fpq
    score.repeatInsert(n, offsets)
    beat_strength = np.zeros(shape=(3, n_frames), dtype=np.int32)
    for n in score.flat.notes:
        bs = n.beatStrength
        if bs == 1.:
            i = 0
        elif bs == 0.5:
            i = 1
        else:
            i = 2
        time = int(round(n.offset * fpq))
        beat_strength[i, time] = 1
    # Show the result
    # x = beat_strength[:, -120:]
    # sns.heatmap(x)
    # plt.show()
    return beat_strength


def load_score_pitch_spelling(i, fpq):
    score, t0, n_frames = _load_score(i, fpq)
    score = score.chordify()
    piano_roll = np.zeros(shape=(35 * 2, n_frames), dtype=np.int32)
    flattest, sharpest = 35, 0
    numFlatwards, numSharpwards = 0, 0
    for chord in score.flat.notes:
        start = int(round(chord.offset * fpq))
        end = start + max(int(round(chord.duration.quarterLength * fpq)), 1)
        time = np.arange(start, end)
        for i, note in enumerate(chord):
            nn = note.pitch.name
            idx = P2I[nn]
            flattest = min(flattest, idx)
            sharpest = max(sharpest, idx)
            piano_roll[idx, time] = 1
            if i == 0:
                piano_roll[idx + 35, time] = 1

        numFlatwards = flattest  # these are transpositions to the LEFT, with our definition of PITCH_LINE
        numSharpwards = 35 - sharpest  # these are transpositions to the RIGHT, with our definition of PITCH_LINE
    # Show the result
    # sns.heatmap(piano_roll)
    # plt.show()
    return piano_roll, t0, numFlatwards, numSharpwards


def load_score_pitch_class(i, fpq):
    score, t0, n_frames = _load_score(i, fpq)
    score = score.chordify()
    piano_roll = np.zeros(shape=(24, n_frames), dtype=np.int32)
    for n in score.flat.notes:
        pitches = np.array(n.pitchClasses)
        start = int(round(n.offset * fpq))
        end = start + max(int(round(n.duration.quarterLength * fpq)), 1)
        time = np.arange(start, end)
        # p = sns.heatmap(piano_roll[:, :end])
        # plt.show(p)
        for p in pitches:  # add notes to piano_roll
            piano_roll[p, time] = 1
        piano_roll[pitches[0] + 12, time] = 1
    # Show the result
    # sns.heatmap(piano_roll)
    # plt.show()
    return piano_roll, t0


def load_score_midi_number(i, fpq, pitch_low=0, pitch_high=128):
    """
    Load notes in each piece, which is then represented as piano roll.
    :param pitch_low:
    :param pitch_high:
    :param i: which sonata to take (sonatas indexed from 1 to 32)
    :param fpq: frames per quarter note, default =  8 (that is, 32th note as 1 unit in piano roll)
    :return: pieces, tdeviation
    """
    score, t0, n_frames = _load_score(i, fpq)
    piano_roll = np.zeros(shape=(128, n_frames), dtype=np.int32)
    for n in score.flat.notes:
        pitches = np.array([x.midi for x in n.pitches] if isinstance(n, Chord) else [n.pitch.midi])
        start = int(round(n.offset * fpq))
        end = start + max(int(round(n.duration.quarterLength * fpq)), 1)
        time = np.arange(start, end)
        # p = sns.heatmap(piano_roll[:, :end])
        # plt.show(p)
        for p in pitches:  # add notes to piano_roll
            piano_roll[p, time] = 1
    # Show the result
    # sns.heatmap(piano_roll)
    # plt.show()
    return piano_roll[pitch_low:pitch_high], t0


def visualize_piano_roll(pr, sonata, fpq, start=None, end=None):
    p = sns.heatmap(pr[::-1, start:end])
    plt.title(f'Sonata {sonata}, quarters (start, end) = {start / fpq, end / fpq}')
    plt.show(p)
    return


def load_chord_labels(i):
    """
    Load chords of each piece and add chord symbols into the labels.
    :param i: which sonata to take (sonatas indexed from 0 to 31)
    :return: chord_labels
    """

    dt = [('onset', 'float'), ('end', 'float'), ('key', '<U10'), ('degree', '<U10'), ('quality', '<U10'),
          ('inversion', 'int'), ('chord_function', '<U10')]  # datatype
    chords_file = os.path.join(BPS_FH_FOLDER, str(i).zfill(2), "chords.xlsx")

    workbook = xlrd.open_workbook(chords_file)
    sheet = workbook.sheet_by_index(0)
    chords = []
    for rowx in range(sheet.nrows):
        cols = sheet.row_values(rowx)
        # xlrd.open_workbook automatically casts strings to float if they are compatible. Revert this.
        cols[2] = cols[2].replace('+', '#')  # BPS-FH people use + for sharps, while music21 uses #. We stick to #.
        if isinstance(cols[3], float):  # if type(degree) == float
            cols[3] = str(int(cols[3]))
        if cols[4] == 'a6':  # in the case of aug 6 chords, verify if they're italian, german, or french
            cols[4] = cols[6].split('/')[0]
        chords.append(tuple(cols))
    return np.array(chords, dtype=dt)  # convert to structured array


def shift_chord_labels(chord_labels, s, mode='semitone'):
    """

    :param chord_labels:
    :param s:
    :param mode: can be either 'semitone' or 'fifth' and describes how transpositions are done.
    :return:
    """
    new_labels = chord_labels.copy()

    for i in range(len(new_labels)):
        key = chord_labels[i]['key']
        if mode == 'semitone':
            # TODO: This never uses flats for keys!
            key = find_enharmonic_equivalent(key)
            idx = ((N2I[key[0].upper()] + s - 1) if ('-' in key) else (N2I[key.upper()] + s)) % 12
            new_key = NOTES[idx] if key.isupper() else NOTES[idx].lower()
        elif mode == 'fifth':
            idx = P2I[key.upper()] + s
            new_key = PITCH_LINE[idx] if key.isupper() else PITCH_LINE[idx].lower()
        else:
            raise ValueError('mode should be either "semitone" or "fifth"')
        new_labels[i]['key'] = new_key
    return new_labels


def segment_chord_labels(i, chord_labels, n_frames, t0, hsize=4, fpq=8, pitch_spelling=False):
    """
    Despite the name, this also finds the root for each chord

    :param i: sonata number
    :param chord_labels:
    :param n_frames: total frames of the analysis
    :param t0:
    :param hsize: hop size between different frames
    :param fpq: frames per quarter note
    :param pitch_spelling: if True, use the correct pitch spelling (e.g., F++ != G)
    :return:
    """
    # Get corresponding chord label (only chord symbol) for each segment
    labels = []
    for n in range(n_frames):
        seg_time = (n * hsize / fpq) + t0  # central time of the segment in quarter notes
        label = chord_labels[np.logical_and(chord_labels['onset'] <= max(seg_time, 0), chord_labels['end'] >= seg_time)]
        try:
            label = label[0]
        except IndexError:
            raise HarmonicAnalysisError(f"Cannot read label for Sonata N.{i} at frame {n}, time {seg_time}")

        labels.append((label['key'], label['degree'], label['quality'], label['inversion'],
                       find_chord_root(label, pitch_spelling)))
    return labels


def encode_chords(chords, mode='semitone'):
    """
    Associate every chord element with an integer that represents its category.

    :param chords:
    :param mode:
    :return:
    """
    chords_enc = []
    for chord in chords:
        key_enc = _encode_key(str(chord[0]), mode)
        degree_p_enc, degree_s_enc = _encode_degree(str(chord[1]))
        quality_enc = _encode_quality(str(chord[2]))
        inversion_enc = int(chord[3])
        root_enc = _encode_root(str(chord[4]), mode)

        chords_enc.append((key_enc, degree_p_enc, degree_s_enc, quality_enc, inversion_enc, root_enc))

    return chords_enc


def find_bass_notes(piano_roll):
    """
    Given a piano roll, return the pitch class of the lowest note every 4 time-step.
    The output encoding is C = 0, C# = 1, ... B = 11

    :param piano_roll:
    :return:
    """
    bass = np.argmax(piano_roll, axis=0)
    bass = np.array([np.min(bass[4 * i:4 * (i + 1)]) for i in range(len(bass) // 4)])
    return bass + PITCH_LOW - 60  # C4 is the midi pitch 60


def calculate_number_transpositions_key(chords):
    keys = set([c['key'] for c in chords])
    nl, nr = 35, 35  # number of transpositions to the left or to the right
    for k in keys:
        i = P2I[k.upper()]
        if k.isupper():
            l = i - 1  # we don't use the left-most major key (F--)
            r = 35 - i - 5  # we don't use the 5 right-most major keys AND it is the endpoint of a range
        else:
            l = i - 4  # we don't use the 4 left-most minor keys (yes! different from the major case!)
            r = 35 - i - 5  # we don't use the 5 right-most minor keys AND it is the endpoint of a range
        nl, nr = min(nl, l), min(nr, r)
    return nl, nr


def _encode_key(key, mode):
    """
    if mode == 'semitone', Major keys: 0-11, Minor keys: 12-23
    if mode == 'fifth',
    """
    # minor keys are always encoded after major keys
    if mode == 'semitone':
        # 12 because there are 12 pitch classes
        res = N2I[key.upper()] + (12 if key.islower() else 0)
    elif mode == 'fifth':
        # -1 because we don't use F-- as a key (it has triple flats) and that is the first element in the PITCH_LINE
        # + 35 because there are 35 total major keys and this is the theoretical distance between a major key
        # and its minor equivalent if all keys were used
        # -9 because we don't use the last 5 major keys (triple sharps) and the first 4 minor keys (triple flats)
        res = P2I[key.upper()] - 1 + (35 - 9 if key.islower() else 0)
    else:
        raise ValueError("_encode_key: Mode not recognized")
    return res


def _encode_root(root, mode):
    if mode == 'semitone':
        res = N2I[root]
    elif mode == 'fifth':
        res = P2I[root]
    else:
        raise ValueError("_encode_root: Mode not recognized")
    return res


def _encode_degree(degree):
    """
    7 diatonics *  3 chromatics  = 21; (0-6 diatonic, 7-13 sharp, 14-20 flat)
    :return: primary_degree, secondary_degree
    """

    if '/' in degree:
        num, den = degree.split('/')
        primary = _encode_degree_no_slash(den)  # 1-indexed as usual in musicology
        secondary = _encode_degree_no_slash(num)
    else:
        primary = 1  # 1-indexed as usual in musicology
        secondary = _encode_degree_no_slash(degree)
    return primary - 1, secondary - 1  # set 0-indexed


def _encode_degree_no_slash(degree_str):  # diatonic 1-7, raised 8-14, lowered 15-21
    if degree_str[0] == '-':
        offset = 14
    elif degree_str[0] == '+':
        offset = 7
    elif len(degree_str) == 2 and degree_str[1] == '+':  # the case of augmented chords (only 1+ ?)
        degree_str = degree_str[0]
        offset = 0
    else:
        offset = 0
    return abs(int(degree_str)) + offset  # 1-indexed as usual in musicology


def _encode_quality(quality):
    return Q2I[quality]


def find_enharmonic_equivalent(note):
    """ Transform everything into a note with at most one sharp and no flats """
    note_up = note.upper()

    if note_up in NOTES:
        return note_up

    if '##' in note_up:
        if 'B' in note_up or 'E' in note_up:
            note_up = ROOTS[(ROOTS.index(note_up[0]) + 1) % 7] + '#'
        else:
            note_up = ROOTS[ROOTS.index(note_up[0]) + 1]  # no problem when index == 6 because that's the case B++
    elif '--' in note_up:  # if root = x--
        if 'C' in note_up or 'F' in note_up:
            note_up = ROOTS[ROOTS.index(note_up[0]) - 1] + '-'
        else:
            note_up = ROOTS[ROOTS.index(note_up[0]) - 1]

    if note_up == 'F-' or note_up == 'C-':
        note_up = ROOTS[ROOTS.index(note_up[0]) - 1]
    elif note_up == 'E#' or note_up == 'B#':
        note_up = ROOTS[(ROOTS.index(note_up[0]) + 1) % 7]

    if note_up not in NOTES:  # there is a single flat left, and it's on a black key
        note_up = ROOTS[ROOTS.index(note_up[0]) - 1] + '#'

    return note_up if note.isupper() else note_up.lower()


def find_chord_root(chord, pitch_spelling):
    """
    Get the chord root from the roman numeral representation.
    :param chord:
    :param pitch_spelling: if True, use the correct pitch spelling (e.g., F++ != G)
    :return: chords_full
    """

    # Translate chords
    key = chord['key']
    degree_str = chord['degree']

    try:  # check if we have already seen the same chord (F2S = features to symbol)
        return F2S[','.join([key, degree_str])]
    except KeyError:
        pass

    d_enc, n_enc = _encode_degree(degree_str)

    d, d_alt = d_enc % 7, d_enc // 7
    key2 = SCALES[key][d]  # secondary key
    if (key.isupper() and d in [1, 2, 5, 6]) or (key.islower() and d in [0, 1, 3, 6]):
        key2 = key2.lower()
    if d_alt == 1:
        key2 = _sharp_alteration(key2).lower()  # when the root is raised, we go to minor scale
    elif d_alt == 2:
        key2 = _flat_alteration(key2).upper()  # when the root is lowered, we go to major scale

    n, n_alt = n_enc % 7, n_enc // 7
    root = SCALES[key2][n]
    if n_alt == 1:
        root = _sharp_alteration(root)
    elif n_alt == 2:
        root = _flat_alteration(root)

    if not pitch_spelling:
        root = find_enharmonic_equivalent(root)

    F2S[','.join([key, degree_str])] = root
    return root


def _flat_alteration(note):
    """ Ex: _flat_alteration(G) = G-,  _flat_alteration(G#) = G """
    return note[:-1] if '#' in note else note + '-'


def _sharp_alteration(note):
    """ Ex: _sharp_alteration(G) = G#,  _sharp_alteration(G-) = G """
    return note[:-1] if '-' in note else note + '#'

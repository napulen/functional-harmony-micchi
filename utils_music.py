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
import csv

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from deprecated import deprecated
from music21 import converter, note
from music21.repeat import ExpanderException
from numpy.lib.recfunctions import append_fields

from config import NOTES, PITCH_FIFTHS, QUALITY, SCALES, KEYS_SPELLING, PITCH_SEMITONES

F2S = dict()
N2I = dict([(e[1], e[0]) for e in enumerate(NOTES)])
Q2I = dict([(e[1], e[0]) for e in enumerate(QUALITY)])
PF2I = dict([(e[1], e[0]) for e in enumerate(PITCH_FIFTHS)])
PS2I = dict([(e[1], e[0]) for e in enumerate(PITCH_SEMITONES)])
PF2PS = dict([(n, PITCH_SEMITONES.index(p)) for n, p in enumerate(PITCH_FIFTHS)])
PS2PF = dict([(n, PITCH_FIFTHS.index(p)) for n, p in enumerate(PITCH_SEMITONES)])

DT_READ = [('onset', 'float'), ('end', 'float'), ('key', '<U10'), ('degree', '<U10'), ('quality', '<U10'),
           ('inversion', 'int')]  # datatype for reading data from BPS-FH
DT_FINAL = [('t', 'float'), ('key', '<U10'), ('degree', '<U10'), ('quality', '<U10'), ('inversion', 'int'),
            ('root', '<U10')]  # final datatype for the chords annotation we use before encoding


class HarmonicAnalysisError(Exception):
    """ Raised when the harmonic analysis associated to a certain time can't be found """
    pass


def _load_score(score_file, fpq):
    if 'bps' in score_file:
        try:
            score = converter.parse(score_file).expandRepeats()
        except ExpanderException:
            score = converter.parse(score_file)
            print(
                "Tried to expand repeats but didn't manage to. Maybe there are no repeats in the piece? Please check.")
    else:
        score = converter.parse(score_file)
    n_frames = int(score.duration.quarterLength * fpq)
    return score, n_frames


def load_score_beat_strength(score_file, fpq):
    """

    :param score_file:
    :param fpq:
    :return:
    """
    score, n_frames = _load_score(score_file, fpq)
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


####################
#
# Load the score for the pitch spelling representation
#
####################

def load_score_spelling_complete(score_file, fpq, mode='fifth'):
    if mode not in ['fifth', 'semitone']:
        raise NotImplementedError("Only modes fifth and semiton are accepted")
    score, n_frames = _load_score(score_file, fpq)
    piano_roll = np.zeros(shape=(35 * 7, n_frames), dtype=np.int32)
    flattest, sharpest = 35, 0
    for n in score.flat.notes:
        notes = np.array([x for x in n] if n.isChord else [n])
        start = int(round(n.offset * fpq))
        end = start + max(int(round(n.duration.quarterLength * fpq)), 1)
        time = np.arange(start, end)
        for i, note in enumerate(notes):
            pitch_name = note.pitch.name
            octave = note.pitch.octave - 1
            if octave < 0 or octave >= 7:  # we keep just 7 octaves in total
                print("skipped a note")
                continue
            idx = PF2I[pitch_name] if mode == 'fifth' else PS2I[pitch_name]
            flattest = min(flattest, idx)
            sharpest = max(sharpest, idx)
            piano_roll[idx + 35 * octave, time] = 1

    num_flatwards = flattest  # these are transpositions to the LEFT, with our definition of PITCH_LINE
    num_sharpwards = 35 - sharpest  # these are transpositions to the RIGHT, with our definition of PITCH_LINE
    return piano_roll, num_flatwards, num_sharpwards


@deprecated(reason="Use the version without chordify, this one is here just for testing")
def load_score_spelling_bass_chordify(score_file, fpq):
    score, n_frames = _load_score(score_file, fpq)
    score = score.chordify()  # this is very comfy but super slow (several seconds on long pieces)
    piano_roll = np.zeros(shape=(35 * 2, n_frames), dtype=np.int32)
    flattest, sharpest = 35, 0
    for chord in score.flat.notes:
        start = int(round(chord.offset * fpq))
        end = start + max(int(round(chord.duration.quarterLength * fpq)), 1)
        time = np.arange(start, end)
        for i, note in enumerate(chord):
            nn = note.pitch.name
            idx = PF2I[nn]
            flattest = min(flattest, idx)
            sharpest = max(sharpest, idx)
            piano_roll[idx, time] = 1
            if i == 0:  # chords always start from the bass
                piano_roll[idx + 35, time] = 1

    numFlatwards = flattest  # these are transpositions to the LEFT, with our definition of PITCH_LINE
    numSharpwards = 35 - sharpest  # these are transpositions to the RIGHT, with our definition of PITCH_LINE
    return piano_roll, numFlatwards, numSharpwards


def load_score_spelling_bass(score_file, fpq):
    """
     NOTE: DO NOT USE THIS!! IT DOESN"T WORK YET
     An attempt at getting rid of the very slow chordify, still not working because the bass is not properly done

    :param score_file:
    :param fpq:
    :return:
    """
    # Encode with an ordering that is comfortable for finding out the bass
    pr_complete, num_flatwards, num_sharpwards = load_score_spelling_complete(score_file, fpq, mode='semitone')
    n_frames = pr_complete.shape[1]
    piano_roll = np.zeros(shape=(35 * 2, n_frames), dtype=np.int32)

    # associate every note with its note name spelling
    p, t = pr_complete.nonzero()
    pf = np.vectorize(PS2PF.get)(p % 35)
    piano_roll[pf, t] = 1  # go to fifth ordering, that is comfortable for transpositions

    # store information on the bass
    b = np.argmax(pr_complete, axis=0)
    v = np.max(pr_complete, axis=0)
    t = np.arange(n_frames)
    bf = np.vectorize(PS2PF.get)(b % 35) + 35
    piano_roll[bf, t] = v

    # score, n_frames = _load_score(score_file, fpq)
    # piano_roll = np.zeros(shape=(35 * 2, n_frames), dtype=np.int32)
    # flattest, sharpest = 35, 0
    # for n in score.flat.notes:
    #     pitches = np.array([x for x in n] if n.isChord else [n])
    #     i_bass = np.argmin([p.pitch.midi for p in pitches])
    #     start = int(round(n.offset * fpq))
    #     end = start + max(int(round(n.duration.quarterLength * fpq)), 1)
    #     time = np.arange(start, end)
    #     for i, note in enumerate(pitches):
    #         nn = note.pitch.name
    #         idx = P2I[nn]
    #         flattest = min(flattest, idx)
    #         sharpest = max(sharpest, idx)
    #         piano_roll[idx, time] = 1
    #         if i == i_bass:
    #             piano_roll[idx + 35, time] = 1
    #
    # # v = np.max(piano_roll, axis=0)
    # # p = np.argmax(piano_roll, axis=0)
    # # for t in range(n_frames):
    # #     piano_roll[p[t] + 35, t] = v[t]
    # num_flatwards = flattest  # these are transpositions to the LEFT, with our definition of PITCH_LINE
    # num_sharpwards = 35 - sharpest  # these are transpositions to the RIGHT, with our definition of PITCH_LINE
    return piano_roll, num_flatwards, num_sharpwards


def load_score_spelling_class(score_file, fpq):
    score, n_frames = _load_score(score_file, fpq)
    piano_roll = np.zeros(shape=(35, n_frames), dtype=np.int32)
    flattest, sharpest = 35, 0
    for n in score.flat.notes:
        pitch_names = np.array([x.pitch.name for x in n] if n.isChord else [n.pitch.name])
        start = int(round(n.offset * fpq))
        end = start + max(int(round(n.duration.quarterLength * fpq)), 1)
        time = np.arange(start, end)
        for pn in pitch_names:
            idx = PF2I[pn]
            flattest = min(flattest, idx)
            sharpest = max(sharpest, idx)
            piano_roll[idx, time] = 1

    num_flatwards = flattest  # these are transpositions to the LEFT, with our definition of PITCH_LINE
    num_sharpwards = 35 - sharpest  # these are transpositions to the RIGHT, with our definition of PITCH_LINE
    return piano_roll, num_flatwards, num_sharpwards


####################
#
# Load the score for the midi pitch representation
#
####################


def load_score_pitch_complete(score_file, fpq):
    """
    Load notes in each piece, which is then represented as piano roll.
    :param score_file: the path to the file to analyse
    :param fpq: frames per quarter note, default =  8 (that is, 32th note as 1 unit in piano roll)
    :return: piano_roll, 128 different pitches
    """
    score, n_frames = _load_score(score_file, fpq)
    piano_roll = np.zeros(shape=(128, n_frames), dtype=np.int32)  # MIDI numbers are between 1 and 128
    for n in score.flat.notes:
        pitches = np.array([x.pitch.midi for x in n] if n.isChord else [n.pitch.midi])
        start = int(round(n.offset * fpq))
        end = start + max(int(round(n.duration.quarterLength * fpq)), 1)
        time = np.arange(start, end)
        for p in pitches:  # add notes to piano_roll
            piano_roll[p, time] = 1
    return piano_roll


@deprecated(reason="Use the version without chordify, this one is here just for testing")
def load_score_pitch_bass_chordify(score_file, fpq):
    """
    Load a score and create a piano roll with 12 pitch class + 12 bass
    :param score_file:
    :param fpq:
    :return:
    """
    score, n_frames = _load_score(score_file, fpq)
    score = score.chordify()  # super slow!
    piano_roll = np.zeros(shape=(24, n_frames), dtype=np.int32)
    for n in score.flat.notes:
        pitches = np.array(n.pitchClasses)
        start = int(round(n.offset * fpq))
        end = start + max(int(round(n.duration.quarterLength * fpq)), 1)
        time = np.arange(start, end)
        for p in pitches:  # add notes to piano_roll
            piano_roll[p, time] = 1
        piano_roll[pitches[0] + 12, time] = 1  # add the bass
    return piano_roll


def load_score_pitch_bass(score_file, fpq):
    """
    Load a score and create a piano roll with 12 pitch class + 12 bass
    :param score_file:
    :param fpq:
    :return:
    """
    pr_complete = load_score_pitch_complete(score_file, fpq)
    n_frames = pr_complete.shape[1]
    piano_roll = np.zeros(shape=(12 * 2, n_frames), dtype=np.int32)

    # associate every note with its class
    p, t = pr_complete.nonzero()
    piano_roll[p % 12, t] = 1

    # store information on the bass
    p = np.argmax(pr_complete, axis=0)
    v = np.max(pr_complete, axis=0)
    t = np.arange(n_frames)
    piano_roll[p % 12 + 12, t] = v

    return piano_roll


@deprecated(reason="Use the version without chordify, this one is here just for testing")
def load_score_pitch_class_chordify(score_file, fpq):
    """
    Load a score and create a piano roll with 12 pitch class
    :param score_file:
    :param fpq:
    :return:
    """
    score, n_frames = _load_score(score_file, fpq)
    score = score.chordify()
    piano_roll = np.zeros(shape=(12, n_frames), dtype=np.int32)
    for n in score.flat.notes:
        pitches = np.array(n.pitchClasses)
        start = int(round(n.offset * fpq))
        end = start + max(int(round(n.duration.quarterLength * fpq)), 1)
        time = np.arange(start, end)
        for p in pitches:  # add notes to piano_roll
            piano_roll[p, time] = 1
    return piano_roll


def load_score_pitch_class(score_file, fpq):
    """
    Load a score and create a piano roll with 12 pitch class
    :param score_file:
    :param fpq:
    :return:
    """
    score, n_frames = _load_score(score_file, fpq)
    piano_roll = np.zeros(shape=(12, n_frames), dtype=np.int32)
    for n in score.flat.notes:
        pitch_classes = np.array([x.pitch.pitchClass for x in n] if n.isChord else [n.pitch.pitchClass])
        start = int(round(n.offset * fpq))
        end = start + max(int(round(n.duration.quarterLength * fpq)), 1)
        time = np.arange(start, end)
        for pc in pitch_classes:  # add notes to piano_roll
            piano_roll[pc, time] = 1
    return piano_roll


####################
#
# Load the chord labels
#
####################


def load_chord_labels(chords_file):
    """
    Load chords of each piece and add chord symbols into the labels.
    :param chords_file: the path to the file with the harmonic analysis
    :return: chord_labels
    """

    chords = []
    with open(chords_file, mode='r') as f:
        data = csv.reader(f)
        for row in data:
            chords.append(tuple(row))
    return np.array(chords, dtype=DT_READ)


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
            idx = PF2I[key.upper()] + s
            new_key = PITCH_FIFTHS[idx] if key.isupper() else PITCH_FIFTHS[idx].lower()
        else:
            raise ValueError('mode should be either "semitone" or "fifth"')
        new_labels[i]['key'] = new_key
    return new_labels


def segment_chord_labels(chord_labels, n_frames, hsize=4, fpq=8):
    """
    Despite the name, this also finds the root for each chord

    :param chord_labels:
    :param n_frames: total frames of the analysis
    :param hsize: hop size between different frames
    :param fpq: frames per quarter note
    :return:
    """
    # Get corresponding chord label (only chord symbol) for each segment
    labels, label = [], []
    k = 1
    for n in range(n_frames):
        seg_time = (n * hsize / fpq)  # onset of the segment in quarter notes
        labels_found = chord_labels[np.logical_and(chord_labels['onset'] <= seg_time, seg_time < chord_labels['end'])]
        if len(labels_found) == 0:
            # raise HarmonicAnalysisError(f"Cannot read labels at frame {n}, time {seg_time}")
            if len(labels) > 0:
                labels_found = [label]
                print(f"Cannot read labels at frame {n}, time {seg_time}."
                      f" Assuming that the previous chord is still valid: {labels_found}")
            else:
                k += 1
                print(f"Cannot read labels at frame {n}, time {seg_time}."
                      f" I still haven't found any valid chord. I will read the next one and duplicate it.")
                continue

        if len(labels_found) > 1:
            # HarmonicAnalysisError(f"More than one chord at frame {n}, time {seg_time, seg_time + hsize / fpq}:\n"
            #                             f"{[l for l in labels_found]}")
            print(
                f"More than one chord at frame {n}, time {seg_time, seg_time + hsize / fpq}:\n{[l for l in labels_found]}")
        label = labels_found[0]
        label_array = np.array((seg_time, label['key'], label['degree'], label['quality'],
                                label['inversion'], label['root']), dtype=DT_FINAL)
        for _ in range(k):
            labels.append(label_array)
        k = 1
    return labels


def encode_chords(chords, mode='semitone'):
    """
    Associate every chord element with an integer that represents its category.

    :param chords: in the namedtuple format
    :param mode:
    :return:
    """
    chords_enc = []
    n = 0
    for chord in chords:
        key_enc = _encode_key(str(chord['key']), mode)
        degree_p_enc, degree_s_enc = _encode_degree(str(chord['degree']))
        quality_enc = _encode_quality(str(chord['quality']))
        inversion_enc = int(chord['inversion'])
        root_enc = _encode_root(str(chord['root']), mode, chord)

        chords_enc.append((key_enc, degree_p_enc, degree_s_enc, quality_enc, inversion_enc, root_enc))
        n += 1

    return chords_enc


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
        res = PF2I[key.upper()] - 1 + (35 - 9 if key.islower() else 0)
    else:
        raise ValueError("_encode_key: Mode not recognized")
    return res


def _encode_root(root, mode, chord):
    if mode == 'semitone':
        try:
            res = N2I[root]
        except KeyError:
            print(f'invalid root {root} for chord {chord}')
            res = None
            # raise KeyError(f'{root} for chord {chord}')
    elif mode == 'fifth':
        try:
            res = PF2I[root]
        except KeyError:
            print(f'invalid root {root} for chord {chord}')
            res = None
            # raise KeyError(f'{root} for chord {chord}')
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
    if len(degree_str) > 2:
        print(f'weird degree_str: {degree_str}, chucking off the first char to {degree_str[1:]}')
        degree_str = degree_str[1:]
    return abs(int(degree_str)) + offset  # 1-indexed as usual in musicology


def _encode_quality(quality):
    return Q2I[quality]


def find_enharmonic_equivalent(note):
    """
    Transform everything into a note with at most one sharp and no flats.
    Keeps the upper- or lower-case intact.

    :param note:
    :return:
    """
    note_up = note.upper()

    if note_up in NOTES:
        return note

    n, alt = N2I[note_up[0]], list(note_up[1:])
    while alt:
        x = alt.pop(-1)
        n = n + 1 if x == '#' else n - 1
    n = n % 12  # the notes are circular!
    return NOTES[n] if note.isupper() else NOTES[n].lower()


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

    if not pitch_spelling:
        key2 = find_enharmonic_equivalent(key2)

    n, n_alt = n_enc % 7, n_enc // 7
    try:
        root = SCALES[key2][n]
    except KeyError:
        print(f'secondary key {key2} for chord {chord}')
        return None
    if n_alt == 1:
        root = _sharp_alteration(root)
    elif n_alt == 2:
        root = _flat_alteration(root)

    if not pitch_spelling:
        root = find_enharmonic_equivalent(root)

    F2S[','.join([key, degree_str])] = root
    return root


def attach_chord_root(chord_labels, pitch_spelling=True):
    return append_fields(chord_labels, 'root', np.array([find_chord_root(c, pitch_spelling) for c in chord_labels]),
                         '<U10')


def _flat_alteration(note):
    """ Ex: _flat_alteration(G) = G-,  _flat_alteration(G#) = G """
    return note[:-1] if '#' in note else note + '-'


def _sharp_alteration(note):
    """ Ex: _sharp_alteration(G) = G#,  _sharp_alteration(G-) = G """
    return note[:-1] if '-' in note else note + '#'


def find_root_full_output(y_pred, pitch_spelling=True):
    """
    Calculate the root of the chord given the output prediction of the neural network.
    It uses key, primary degree and secondary degree.

    :param y_pred: the prediction, shape [output], (timestep, output_features)
    :return:
    """
    keys_encoded = np.argmax(y_pred[0], axis=-1)
    degree_den = np.argmax(y_pred[1], axis=-1)
    degree_num = np.argmax(y_pred[2], axis=-1)

    root_pred = []
    for key_enc, dd, dn in zip(keys_encoded, degree_den, degree_num):
        den, den_alt = dd % 7, dd // 7
        num, num_alt = dn % 7, dn // 7

        key = KEYS_SPELLING[key_enc]
        key2 = SCALES[key][den]  # secondary key
        if (key.isupper() and den in [1, 2, 5, 6]) or (key.islower() and den in [0, 1, 3, 6]):
            key2 = key2.lower()
        if den_alt == 1:
            key2 = _sharp_alteration(key2).lower()  # when the root is raised, we go to minor scale
        elif den_alt == 2:
            key2 = _flat_alteration(key2).upper()  # when the root is lowered, we go to major scale

        try:
            root = SCALES[key2][num]
        except KeyError:
            print(f'secondary key {key2} for chord {dd, dn} in key of {key}')
            root_pred.append(None)
            continue

        if num_alt == 1:
            root = _sharp_alteration(root)
        elif num_alt == 2:
            root = _flat_alteration(root)

        if not pitch_spelling:
            root = find_enharmonic_equivalent(PF2I[root])
        root_pred.append(PF2I[root])

    return np.array(root_pred)


def visualize_piano_roll(pr, sonata, fpq, start=None, end=None):
    p = sns.heatmap(pr[::-1, start:end])
    plt.title(f'Sonata {sonata}, quarters (start, end) = {start / fpq, end / fpq}')
    plt.show(p)
    return


def calculate_number_transpositions_key(chords):
    keys = set([c['key'] for c in chords])
    nl, nr = 35, 35  # number of transpositions to the left or to the right
    for k in keys:
        i = PF2I[k.upper()]
        if k.isupper():
            l = i - 1  # we don't use the left-most major key (F--)
            r = 35 - i - 5  # we don't use the 5 right-most major keys AND it is the endpoint of a range
        else:
            l = i - 4  # we don't use the 4 left-most minor keys (yes! different from the major case!)
            r = 35 - i - 5  # we don't use the 5 right-most minor keys AND it is the endpoint of a range
        nl, nr = min(nl, l), min(nr, r)
    return nl, nr


if __name__ == '__main__':
    # score_file = '/home/gianluca/PycharmProjects/functional-harmony/test_data/score.mxl'
    score_file = '/home/gianluca/PycharmProjects/functional-harmony/test_data/bps_01_01.mxl'
    # fpq = 8
    # pr, _, _ = load_score_spelling_bass(score_file, fpq)
    # pr2, _, _ = load_score_spelling_bass_chordify(score_file, fpq)
    # sns.heatmap(pr2 - pr)
    # plt.xticks(np.arange(8) * 32 + 8, np.arange(8) + 1)
    # plt.yticks(np.arange(70) + 0.5, PITCH_FIFTHS + ['b ' + p for p in PITCH_FIFTHS])
    # plt.show()
    # print('hi')

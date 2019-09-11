import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import ROOTS, NOTES, SCALES, QUALITY, SYMBOL, FEATURES, TICK_LABELS, PITCH_LINE

F2S = dict()
N2I = dict([(e[1], e[0]) for e in enumerate(NOTES)])
Q2I = dict([(e[1], e[0]) for e in enumerate(QUALITY)])
P2I = dict([(e[1], e[0]) for e in enumerate(PITCH_LINE)])
Q2S = {'M': 'M', 'm': 'm', 'M7': 'M7', 'm7': 'm7', 'D7': '7', 'a': 'aug', 'd': 'dim', 'd7': 'dim7',
       'h7': 'm7(b5)', 'Gr+6': 'Gr+6', 'It+6': 'It+6',
       'Fr+6': 'Fr+6'}  # necessary only because the data is stored in non-standard notation


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

    # FIND THE ROOT OF THE CHORD
    if '/' in degree_str:  # case: secondary chord
        n_str = degree_str.split('/')[0]  # numerator
        d_str = degree_str.split('/')[1]  # denominator
    else:
        n_str = degree_str
        d_str = '1'

    n_enc = _encode_degree_no_slash(n_str) - 1
    n = n_enc % 7
    n_alt = n_enc // 7

    d_enc = _encode_degree_no_slash(d_str) - 1
    d = d_enc % 7
    d_alt = d_enc // 7

    key2 = SCALES[key][d]  # secondary key
    if (key.isupper() and d in [1, 2, 5, 6]) or (key.islower() and d in [0, 1, 3, 6]):
        key2 = key2.lower()
    if d_alt == 1:
        key2 = _sharp_alteration(key2).lower()  # when the root is raised, we go to minor scale
    elif d_alt == 2:
        key2 = _flat_alteration(key2).upper()  # when the root is lowered, we go to major scale

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
    """ Ex: _flat_alteration(G) = G-,  _flat_alteration(G+) = G """
    return note[:-1] if '#' in note else note + '-'


def _sharp_alteration(note):
    """ Ex: _sharp_alteration(G) = G+,  _sharp_alteration(G-) = G """
    return note[:-1] if '-' in note else note + '#'


def _is_major(key):
    """ The input needs to be a string like "A-" for A flat major, "b" for b minor, etc. """
    return key[0].isupper()


def visualize_data(data):
    # data = tf.data.TFRecordDataset(input_path)
    temp = data.make_one_shot_iterator()
    x, y = temp.get_next()
    for pr in x:
        sns.heatmap(pr)
        plt.show()
    return


def create_dezrann_annotations(true, pred, n, batch_size, model_folder):
    """
    Create a JSON file for a single aspect of the analysis that is compatible with dezrann, www.dezrann.net
    This allows for a nice visualization of the analysis on top of the partition.
    :param true: The output of the machine learning model.
    :param n: the number of beethoven sonata
    :param batch_size: Just a check. It needs to be one
    :return:
    """

    if batch_size != 1:
        raise NotImplementedError("This script only works for a batch size of one!")

    for j in range(7):
        data_true = true[j]
        data_pred = pred[j]
        feature = FEATURES[j]
        x = {
            "meta": {
                'title': f"Beethoven sonata no.{n}",
                'name': f"{n} - {feature}",
                'date': str(datetime.now()),
                'producer': 'Algomus team'
            }
        }
        data_true = np.argmax(data_true[0], axis=-1)
        data_pred = np.argmax(data_pred[0], axis=-1)
        assert len(data_pred) == len(data_true)
        length = len(data_true)

        labels = []
        start_true, start_pred = 0, 0
        for t in range(length):
            if t > 0:
                if data_true[t] != data_true[t - 1] or t == length - 1:
                    duration_true = t / 2 - start_true
                    labels.append({
                        "type": feature,
                        "start": start_true,
                        "duration": duration_true,
                        'layers': ['true'],
                        "tag": TICK_LABELS[j][data_true[t - 1]]
                    })
                    start_true = t / 2
                if data_pred[t] != data_pred[t - 1] or t == length - 1:
                    duration_pred = t / 2 - start_pred
                    labels.append({
                        "type": feature,
                        "start": start_pred,
                        "duration": duration_pred,
                        "layers": ['pred'],
                        "tag": TICK_LABELS[j][data_true[t - 1]]
                    })
                    start_pred = t / 2
        x['labels'] = labels
        try:
            os.makedirs(os.path.join(model_folder, 'analyses'))
        except OSError:
            pass

        with open(os.path.join(model_folder, 'analyses', f'analysis_sonata{n}_{feature}.dez'), 'w') as fp:
            json.dump(x, fp)
    return


def _fill_level(l, i_p=None):
    """
    Fill a given level of the structural analysis with hold tokens.
    This is done to distinguish when a missing data should be interpreted as a continuation of the previous section or
    as a missing section. It uses the knowledge of section borders coming from the previous level because they enforce
    borders on the current level as well (a section in level n+1 can't span two section of level n, not even partially)

    :param l: the current level as coming from Mark Gotham's analysis
    :param i_p: the borders of the sections of the previous levels, leave None if it's the first level
    :return: a list filled with hold tokens when needed, and the list with all the borders from the current level
    """
    i_c = np.array([i for i, r in enumerate(l) if not pd.isna(r)])  # beginning of sections in this level
    if i_p is None:
        i_p = [len(l)]
    c, p = 0, 0  # indices over borders of current and previous level
    v = np.nan  # current value (the section)
    y, i_o = [], []
    for i, x in enumerate(l):
        hold = True  # becomes False when a new section starts
        if c < len(i_c) and i == i_c[c]:  # a new section in this level starts
            v = x
            c += 1
            i_o.append(i)
            hold = False
        if p < len(i_p) and i == i_p[p]:  # a new section in the previous level starts
            v = x  # this is nan if no new section in this level starts at this index
            p += 1
            if hold:  # if not hold, it means we have already added this border when checking i_c
                i_o.append(i)
            hold = False
        if pd.isna(v):
            y.append("Empty")
        elif not hold:
            y.append(v)
        else:
            y.append("Hold")
    return y, i_o


def find_root_full_output(y_pred_full):
    """
    Calculate the root of the chord given the output prediction of the neural network.
    It uses key, primary degree and secondary degree.

    :param y_pred_full: the prediction as a list over different timesteps
    :return:
    """
    key, degree_den, degree_num = np.argmax(y_pred_full[0][0], axis=-1), np.argmax(y_pred_full[1][0],
                                                                                   axis=-1), np.argmax(
        y_pred_full[2][0], axis=-1)
    deg2sem_maj = [0, 2, 4, 5, 7, 9, 11]
    deg2sem_min = [0, 2, 3, 5, 7, 8, 10]

    root_pred = []
    for i in range(len(key)):
        deg2sem = deg2sem_maj if key[i] // 12 == 0 else deg2sem_min  # keys 0-11 are major, 12-23 minor
        n_den = deg2sem[degree_den[i] % 7]  # (0-6 diatonic, 7-13 sharp, 14-20 flat)
        if degree_den[i] // 7 == 1:  # raised root
            n_den += 1
        elif degree_den[i] // 7 == 2:  # lowered root
            n_den -= 1
        n_num = deg2sem[degree_num[i] % 7]
        if degree_num[i] // 7 == 1:
            n_num += 1
        elif degree_num[i] // 7 == 2:
            n_num -= 1
        # key[i] % 12 finds the root regardless of major and minor, then both degrees are added, then sent back to 0-11
        # both degrees are added, yes: example: V/IV on C major.
        # primary degree = IV, secondary degree = V
        # in C, that corresponds to the dominant on the fourth degree: C -> F -> C again
        root_pred.append((key[i] % 12 + n_num + n_den) % 12)
    return root_pred


def _decode_key(i):
    lower = i // 12
    key = NOTES[i % 12]
    return key.lower() if lower else key


def _decode_degree(p, s):
    num_alt = s // 7
    num = _int_to_roman((s % 7) + 1)
    if num_alt == 1:
        num += '+'
    elif num_alt == 2:
        num += '-'
    den_alt = p // 7
    den = _int_to_roman((p % 7) + 1)
    if den_alt == 1:
        den += '+'
    elif den_alt == 2:
        den += '-'
    return num + '/' + den if den != _int_to_roman(1) else num


def _decode_quality(q):
    return QUALITY[q]


def _int_to_roman(input):
    """ Convert an integer to a Roman numeral. """

    if not 0 < input < 8:
        raise ValueError("Argument must be between 1 and 7")
    ints = (5, 4, 1)
    nums = ('V', 'IV', 'I')
    result = []
    for i in range(len(ints)):
        count = int(input / ints[i])
        result.append(nums[i] * count)
        input -= ints[i] * count
    return ''.join(result)


def _decode_inversion(i):
    return str(i)


def decode_results(y):
    """
    Transform a list of the class outputs into something readable by humans.

    :param y: it should have shape [features, timesteps], and every element should be an integer indicating the class
    :return:
    """
    key = [_decode_key(i) for i in y[0]]
    degree = [_decode_degree(i[0], i[1]) for i in zip(y[1], y[2])]
    quality = [_decode_quality(i) for i in y[3]]
    inversion = [_decode_inversion(i) for i in y[4]]
    return [key, degree, quality, inversion]

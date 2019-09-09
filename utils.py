import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import ROOTS, NOTES, SCALES, QUALITY, SYMBOL, FEATURES, TICK_LABELS

F2S = dict()
N2I = dict([(e[1], e[0]) for e in enumerate(NOTES)])
Q2I = dict([(e[1], e[0]) for e in enumerate(QUALITY)])
S2I = dict([(e[1], e[0]) for e in enumerate(SYMBOL)])
Q2S = {'M': 'M', 'm': 'm', 'M7': 'M7', 'm7': 'm7', 'D7': '7', 'a': 'aug', 'd': 'dim', 'd7': 'dim7',
       'h7': 'm7(b5)', 'Gr+6': 'Gr+6', 'It+6': 'It+6',
       'Fr+6': 'Fr+6'}  # necessary only because the data is stored in non-standard notation


def _encode_key(key):
    """ Major keys: 0-11, Minor keys: 12-23 """
    return N2I[key.upper()] + (12 if key.islower() else 0)


def _encode_degree(degree):
    """
    7 diatonics *  3 chromatics  = 21; (0-6 diatonic, 7-13 sharp, 14-20 flat)
    :return: primary_degree, secondary_degree
    """

    if '/' in degree:
        num, den = degree.split('/')
        primary = _translate_degree(den)  # 1-indexed as usual in musicology
        secondary = _translate_degree(num)
    else:
        primary = 1  # 1-indexed as usual in musicology
        secondary = _translate_degree(degree)
    return primary - 1, secondary - 1  # set 0-indexed


def _translate_degree(degree_str):
    if degree_str[0] == '-':
        offset = 14
    elif degree_str[0] == '+':
        offset = 7
    elif len(degree_str) == 2 and degree_str[1] == '+':  # the case of augmented chords (only 1+ ?)
        degree_str = degree_str[0]
        offset = 0
    else:
        offset = 0
    return int(degree_str) + offset


def _encode_quality(quality):
    return Q2I[quality]


def _encode_root(root):
    return N2I[root]


def _find_enharmonic_equivalent(note):
    """ Transform everything into one of the notes defined in NOTES """
    if note in NOTES:
        return note

    if '++' in note:
        if 'B' in note or 'E' in note:
            note = ROOTS[(ROOTS.index(note[0]) + 1) % 7] + '+'
        else:
            note = ROOTS[ROOTS.index(note[0]) + 1]  # no problem when index == 6 because that's the case B++
    elif '--' in note:  # if root = x--
        if 'C' in note or 'F' in note:
            note = ROOTS[ROOTS.index(note[0]) - 1] + '-'
        else:
            note = ROOTS[ROOTS.index(note[0]) - 1]

    if note == 'F-' or note == 'C-':
        note = ROOTS[ROOTS.index(note[0]) - 1]
    elif note == 'E+' or note == 'B+':
        note = ROOTS[(ROOTS.index(note[0]) + 1) % 7]

    if note not in NOTES:  # there is a single flat left, and it's on a black key
        note = ROOTS[ROOTS.index(note[0]) - 1] + '+'

    return note


def find_chord_root(chord):
    """
    Get the chord root from the roman numeral representation.
    :param chord:
    :return: chords_full
    """

    # Translate chords
    key = chord['key']
    degree_str = chord['degree']
    quality = chord['quality']

    try:  # check if we have already seen the same chord (F2S = features to symbol)
        return F2S[','.join([key, degree_str, quality])]
    except KeyError:
        pass

    # FIND THE ROOT OF THE CHORD
    if len(degree_str) == 1 or (len(degree_str) == 2 and degree_str[1] == '+'):  # case: degree = x, augmented chords
        degree = int(degree_str[0])
        root = SCALES[key][degree - 1]

    elif degree_str == '+4':  # case: augmented 6th
        degree = 4
        root = SCALES[key][degree - 1]
        root = _sharp_alteration(root)  # lower the sixth in a major key

    # TODO: Verify these cases!
    elif degree_str[0] == '-':  # case: chords on flattened degree
        degree = int(degree_str[1])
        root = SCALES[key][degree - 1]
        root = _flat_alteration(root)  # the fundamental of the chord is lowered

    elif '/' in degree_str:  # case: secondary chord
        degree = degree_str
        n = int(degree.split('/')[0]) if '+' not in degree.split('/')[0] else 6  # take care of augmented 6th chords
        d = int(degree.split('/')[1])  # denominator
        key2 = SCALES[key][abs(d) - 1]  # secondary key
        if d < 0:
            key2 = _flat_alteration(key2)
        key2 = _find_enharmonic_equivalent(key2)

        root = SCALES[key2][n - 1]
        if '+' in degree.split('/')[0]:  # augmented 6th chords
            if _is_major(key2):  # case: major key
                root = _flat_alteration(root)
    else:
        raise ValueError(f"Can't understand the following chord degree: {degree_str}")

    root = _find_enharmonic_equivalent(root)

    F2S[','.join([key, degree_str, quality])] = root
    return root


def _flat_alteration(note):
    """ Ex: _flat_alteration(G) = G-,  _flat_alteration(G+) = G """
    return note[:-1] if '+' in note else note + '-'


def _sharp_alteration(note):
    """ Ex: _sharp_alteration(G) = G+,  _sharp_alteration(G-) = G """
    return note[:-1] if '-' in note else note + '+'


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

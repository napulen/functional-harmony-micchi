import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from config import NOTES, QUALITY, KEYS_SPELLING


def create_dezrann_annotations(test_true, test_pred, timesteps, file_names, model_folder):
    """
    Create a JSON file for a single aspect of the analysis that is compatible with dezrann, www.dezrann.net
    This allows for a nice visualization of the analysis on top of the partition.
    :param test_true: The annotated labels coming from our data, shape [n_chunks][labels](ts, classes)
    :param test_pred: The output of the machine learning model, same shape as test_true
    :param timesteps: number of timesteps per each data point
    :param file_names: the name of the datafile where the chunk comes from
    :param model_folder:
    :return:
    """
    os.makedirs(os.path.join(model_folder, 'analyses'), exist_ok=True)

    n = len(test_true)  # same len as test_pred, timesteps, or filenames
    offsets = np.zeros(n)
    for i in range(1, n):
        if file_names[i] == file_names[i - 1]:
            offsets[i] = offsets[i - 1] + timesteps[i - 1]
    annotation, labels, current_file = dict(), [], None
    features = ['Tonality', 'Harmony']  # add a third element "Inversion" if needed
    lines = [('top.3', 'bot.2'), ('top.2', 'bot.1'), ('top.1', 'bot.3')]

    for y_true, y_pred, ts, name, t0 in zip(test_true, test_pred, timesteps, file_names, offsets):
        if name != current_file:  # a new sonata started
            if current_file is not None:  # save previous file, if it exists
                annotation['labels'] = labels
                with open(os.path.join(model_folder, 'analyses', f'{current_file}.dez'), 'w+') as fp:
                    json.dump(annotation, fp)
            annotation = {
                "meta": {
                    'title': name,
                    'name': name,
                    'date': str(datetime.now()),
                    'producer': 'Algomus team'
                }
            }
            current_file = name
            labels = []

        label_true_list = decode_results(y_true)
        label_pred_list = decode_results(y_pred)

        for feature, line, label_true, label_pred in zip(features, lines, label_true_list, label_pred_list):
            assert len(label_pred) == len(label_true)
            start_true, start_pred = t0 / 2, t0 / 2  # divided by two because we have one label every 8th note
            for t in range(ts):
                if t > 0:
                    if label_true[t] != label_true[t - 1] or t == ts - 1:
                        duration_true = (t + t0) / 2 - start_true
                        labels.append({
                            "type": feature,
                            "start": start_true,
                            "duration": duration_true,
                            "line": line[0],
                            "tag": label_true[t - 1],
                            "comment": "target"
                        })
                        start_true = (t + t0) / 2
                    if label_pred[t] != label_pred[t - 1] or t == ts - 1:
                        duration_pred = (t + t0) / 2 - start_pred
                        labels.append({
                            "type": feature,
                            "start": start_pred,
                            "duration": duration_pred,
                            "line": line[1],
                            "tag": label_pred[t - 1],
                            "comment": "prediction"
                        })
                        start_pred = (t + t0) / 2

    annotation['labels'] = labels
    with open(os.path.join(model_folder, 'analyses', f'{current_file}.dez'), 'w+') as fp:
        json.dump(annotation, fp)
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


def _decode_key(yk):
    n = len(yk)
    k = np.argmax(yk)
    if n == 24:
        lower = k // 12
        key = NOTES[k % 12]
        return key.lower() if lower else key
    elif n == 55:
        return KEYS_SPELLING[k]
    else:
        raise ValueError('weird number of classes in the key')


def _decode_roman(yp, ys, yq):
    s = np.argmax(ys)
    p = np.argmax(yp)
    q = np.argmax(yq)

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

    quality = QUALITY[q]
    if quality == 'M':
        num = num.upper()
        quality = ''
    elif quality == 'm':
        num = num.lower()
        quality = ''
    return num + quality + ('/' + den if den != 'I' else '')


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


def _roman_to_int(roman):
    r2i = {
        'I': 1,
        'II': 2,
        'III': 3,
        'IV': 4,
        'V': 5,
        'VI': 6,
        'VII': 7,
    }
    return r2i[roman.upper()]


def find_scale_and_alteration(degree_str, minor_key):
    nr = ''.join(filter(lambda x: x.upper() in ['V', 'I'], degree_str))
    ni = _roman_to_int(nr)

    a = ''.join(filter(lambda x: x.upper() not in ['V', 'I'], degree_str))
    a = a.replace('#', '+')
    a = a.replace('b', '-')
    if minor_key and ni == 7:
        if '+' in a:
            a = a[:-1]
        else:
            a = a + '-'
    return a, str(ni)


def degrees_dcml_to_bps(degree_num, degree_den='', key_minor=True):
    if not degree_den:
        a, n = find_scale_and_alteration(degree_num, key_minor)
        return a + n

    da, dn = find_scale_and_alteration(degree_den, key_minor)
    k2 = int(dn) - 1  # converts
    if '+' in da:  # all keys on augmented degrees are minor
        k2_minor = True
    elif '-' in da:  # all keys on flattened degrees are major
        k2_minor = False
    elif (not key_minor and k2 in [1, 2, 5, 6]) or (key_minor and k2 in [0, 1, 3, 6]):  # minor secondary key
        k2_minor = True
    else:  # major secondary keys
        k2_minor = False
    na, nn = find_scale_and_alteration(degree_num, k2_minor)
    return '/'.join([na + nn, da + dn])


def _decode_inversion(yi):
    i = np.argmax(yi)
    return str(i)


def decode_results(y):
    """
    Transform the outputs of the model into something readable by humans, example [G+, Vd7/V, '2']

    :param y: it should have shape [features, timesteps], and every element should be an integer indicating the class
    :return: keys, chords, inversions
    """
    key = [_decode_key(i) for i in y[0]]
    chord = [_decode_roman(i[0], i[1], i[2]) for i in zip(y[1], y[2], y[3])]
    inversion = [_decode_inversion(i) for i in y[4]]
    return key, chord, inversion

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from config import NOTES, QUALITY, FEATURES, TICK_LABELS


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

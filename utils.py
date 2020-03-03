"""
Regroups various functions used in the project.
"""
import csv
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from config import NOTES, QUALITY, KEYS_SPELLING, INPUT_TYPES, Q2RN, I2RN


def setup_tfrecords_paths(tfrecords_folder, tfrecords_basename, mode):
    return [os.path.join(tfrecords_folder, f'{bn}_{mode}.tfrecords') for bn in tfrecords_basename]


def create_dezrann_annotations(model_output, model_name, annotations, timesteps, file_names, output_folder):
    """
    Create a JSON file for a single aspect of the analysis that is compatible with dezrann, www.dezrann.net
    This allows for a nice visualization of the analysis on top of the partition.
    :param model_output: The output of the machine learning model, shape [n_chunks][labels](ts, classes)
    :param model_name: The name of the model
    :param annotations: The annotated labels coming from our data, same shape as the model_output, put None if no annotated data
    :param timesteps: number of timesteps per each data point
    :param file_names: the name of the datafile where the chunk comes from
    :param output_folder: could be typically the model folder or the score folder
    :return:
    """
    os.makedirs(output_folder, exist_ok=True)

    if annotations is None:
        annotations = model_output  # this allows to just consider one case
        save_reference = False
    else:
        save_reference = True

    offsets = _set_chunk_offset(file_names, timesteps)
    annotation, labels, current_file = dict(), [], None
    features = ['Tonality', 'Harmony']
    for y_true, y_pred, ts, name, t0 in zip(annotations, model_output, timesteps, file_names, offsets):
        if name != current_file:  # a new sonata started
            if current_file is not None:  # save previous file, if it exists
                annotation['labels'] = labels
                with open(os.path.join(output_folder, f'{current_file}.dez'), 'w+') as fp:
                    json.dump(annotation, fp)
            annotation = {
                "meta": {
                    'title': name,
                    'name': name,
                    'date': str(datetime.now()),
                    'producer': 'Algomus team',
                    'layout': [
                        {
                            "filter": {"type": "Harmony", "layers": ['reference']},
                            "style": {"line": "top.1"}
                        },
                        {
                            "filter": {"type": "Harmony", "layers": ['prediction', model_name]},
                            "style": {"line": "top.2"}
                        },
                        {
                            "filter": {"type": "Tonality", "layers": ['reference']},
                            "style": {"line": "bot.1"}
                        },
                        {
                            "filter": {"type": "Tonality", "layers": ['prediction', model_name]},
                            "style": {"line": "bot.2"}
                        },
                    ]
                }
            }
            current_file = name
            labels = []

        label_true_list = decode_results_dezrann(y_true)
        label_pred_list = decode_results_dezrann(y_pred)

        for feature, label_true, label_pred in zip(features, label_true_list, label_pred_list):
            assert len(label_pred) == len(label_true)
            start_true, start_pred = t0 / 2, t0 / 2  # divided by two because we have one label every 8th note
            for t in range(ts):
                if t > 0:
                    if save_reference and (label_true[t] != label_true[t - 1] or t == ts - 1):
                        duration_true = (t + t0) / 2 - start_true
                        labels.append({
                            "type": feature,
                            "layers": ['reference'],
                            "start": start_true,
                            "actual-duration": duration_true,
                            "tag": label_true[t - 1],
                        })
                        start_true = (t + t0) / 2
                    if label_pred[t] != label_pred[t - 1] or t == ts - 1:
                        duration_pred = (t + t0) / 2 - start_pred
                        labels.append({
                            "type": feature,
                            "tag": label_pred[t - 1],
                            "start": start_pred,
                            "actual-duration": duration_pred,
                            "layers": ['prediction', model_name],
                        })
                        start_pred = (t + t0) / 2

    annotation['labels'] = labels
    with open(os.path.join(output_folder, f'{current_file}.dez'), 'w+') as fp:
        json.dump(annotation, fp, indent=4)
    return


def _set_chunk_offset(file_names, timesteps):
    n = len(timesteps)
    offsets = np.zeros(n)
    for i in range(1, n):
        if file_names[i] == file_names[i - 1]:
            offsets[i] = offsets[i - 1] + timesteps[i - 1]
    return offsets


def create_tabular_annotations(model_output, timesteps, file_names, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    def _save_csv(current_file, data):
        with open(os.path.join(output_folder, f'{current_file}.csv'), 'w') as fp:
            w = csv.writer(fp)
            w.writerows(data)
        return

    offsets = _set_chunk_offset(file_names, timesteps)
    data, current_file, current_label, start, end = [], None, None, 0, None
    for y, ts, name, t0 in zip(model_output, timesteps, file_names, offsets):
        # Save previous analysis if a new one starts
        if name != current_file:  # a new sonata started
            if current_file is not None:  # save previous file, if it exists
                data.append([start, end, *current_label])
                _save_csv(current_file, data)

            data, current_file, current_label, start, end = [], name, None, 0, None

        labels = decode_results_tabular(y)
        for t in range(ts):
            new_label = labels[t]
            if current_label is None:
                current_label = new_label
            if np.any(new_label != current_label):
                end = (t + t0) / 2  # divided by two because we have one label every 8th note
                data.append([start, end, *current_label])
                start = end
                current_label = new_label
            if t == ts - 1:
                end = (t + t0 + 1) / 2  # used only at the beginning of the next chunk if a new piece starts

    # last file
    data.append([start, end, *current_label])
    _save_csv(current_file, data)
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


def int_to_roman(input):
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


def roman_to_int(roman):
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
    ni = roman_to_int(nr)

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


def _decode_key(yk):
    n = len(yk)
    k = np.argmax(yk)
    if n == 24:
        lower = k // 12
        key = NOTES[k % 12]
        return key.lower() if lower else key
    elif n == len(KEYS_SPELLING):
        return KEYS_SPELLING[k]
    else:
        raise ValueError('weird number of classes in the key')


def _decode_degree(yp, ys, roman=True):
    s = np.argmax(ys)
    p = np.argmax(yp)

    num_alt = s // 7
    num_temp = (s % 7) + 1
    num = int_to_roman(num_temp) if roman else str(num_temp)
    if num_alt == 1:
        num += '+'
    elif num_alt == 2:
        num += '-'

    den_alt = p // 7
    den_temp = (p % 7) + 1
    den = int_to_roman(den_temp) if roman else str(den_temp)
    if den_alt == 1:
        den += '+'
    elif den_alt == 2:
        den += '-'
    return num, den


def _decode_quality(yq):
    q = np.argmax(yq)
    quality = QUALITY[q]
    return quality


def decode_roman(num, den, quality, inversion):
    """
    Given degree (numerator and denominator), quality of the chord, and inversion, return a properly written RN.

    :param num: String with the numerator of the degree in arab numerals, e.g. '1', or '+4'
    :param den: Same as num, but for the denominator
    :param quality: Quality of the chord (major, minor, dominant seventh, etc.)
    :param inversion: Inversion as a string
    """
    upper, triad, qlt = Q2RN[quality]
    inv = I2RN[triad + inversion]
    if upper:
        num_prefix = ''
        while num[0] == 'b':
            num_prefix += num[0]
            num = num[1:]
        num = num_prefix + num.upper()
    else:
        num = num.lower()
    if num == 'IV' and qlt == 'M':  # the fourth degree is by default major seventh
        qlt = ''
    return num + qlt + inv + ('/' + den if den != 'I' else '')


def decode_results_dezrann(y):
    """
    Transform the outputs of the model into something readable by humans, example [G+, Vd7/V, '2']

    :param y: it should have shape [features, timesteps], and every element should be an integer indicating the class
    :return: keys, chords, inversions
    """

    key = [_decode_key(k) for k in y[0]]
    num, den = zip(*[_decode_degree(*i) for i in zip(y[1], y[2])])
    quality = [_decode_quality(q) for q in y[3]]
    inversion = [_decode_inversion(i) for i in y[4]]
    roman_numeral = [decode_roman(n, d, q, i) for n, d, q, i in zip(num, den, quality, inversion)]
    return key, roman_numeral


def decode_results_tabular(y):
    """
    Transform the outputs of the model into tabular format, example [G+, V/V, D7, '2']

    :param y: it should have shape [features, timesteps], and every element should be an integer indicating the class
    :return: keys, degree, quality, inversions
    """
    key = [_decode_key(i) for i in y[0]]
    degree_temp = [_decode_degree(i[0], i[1], roman=False) for i in zip(y[1], y[2])]
    degree = [num + ('/' + den if den != '1' else '') for num, den in degree_temp]
    quality = [_decode_quality(i) for i in y[3]]
    inversion = [_decode_inversion(i) for i in y[4]]
    return np.transpose([key, degree, quality, inversion])  # shape (timesteps, 4)


def find_input_type(model_name):
    input_type = None
    for ip in INPUT_TYPES:
        if ip in model_name:
            input_type = ip
            break
    if input_type is None:
        raise AttributeError("can't determine which data needs to be fed to the algorithm...")
    return input_type


def find_best_batch_size(n, bs):
    """

    :param n:
    :param bs: maximum batch size to start with
    :return:
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n should be a positive integer")

    while bs > 1:
        if n % bs == 0:
            break
        else:
            bs -= 1
    return bs

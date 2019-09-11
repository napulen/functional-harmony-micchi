import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import NOTES, QUALITY, FEATURES, TICK_LABELS


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

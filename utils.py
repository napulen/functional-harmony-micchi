from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import json
import numpy as np

from config import VALID_TFRECORDS, TRAIN_TFRECORDS, TEST_TFRECORDS, ROOTS, NOTES, SCALES, QUALITY, SYMBOL, FEATURES, \
    TICK_LABELS

F2S = dict()
R2I = dict([(e[1], e[0]) for e in enumerate(ROOTS)])
N2I = dict([(e[1], e[0]) for e in enumerate(NOTES)])
Q2I = dict([(e[1], e[0]) for e in enumerate(QUALITY)])
S2I = dict([(e[1], e[0]) for e in enumerate(SYMBOL)])
Q2S = {'M': 'M', 'm': 'm', 'M7': 'M7', 'm7': 'm7', 'D7': '7', 'a': 'aug', 'd': 'dim', 'd7': 'dim7',
       'h7': 'm7(b5)', 'a6': '7'}  # necessary only because the data is stored in non-standard notation


def _encode_key(key):
    """ Major keys: 0-11, Minor keys: 12-23 """
    return N2I[key.upper()] + (12 if key.islower() else 0)


def _encode_degree(degree):
    """
    7 diatonics *  3 chromatics  = 21: {0-6 diatonic, 7-13 sharp, 14-20 flat)
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
    elif len(degree_str) == 2 and degree_str[1] == '+':
        degree_str = degree_str[0]
        offset = 0
    else:
        offset = 0
    return int(degree_str) + offset


def _encode_quality(quality):
    return Q2I[quality]


def _encode_symbol(symbol):
    if '+' not in symbol and '-' not in symbol:
        chord_root = symbol[0]
        quality = symbol[1:]
    else:
        chord_root = symbol[:2]
        quality = symbol[2:]

    return N2I[chord_root], S2I[quality]


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


def _find_chord_symbol(chord):
    """
    Translate roman numeral representations into chord symbols.
    :param chord:
    :return: chords_full
    """

    # Translate chords
    key = chord['key']
    degree_str = chord['degree']
    quality = chord['quality']

    try:
        return F2S[','.join([key, degree_str, quality])]
    except KeyError:
        pass

    # FIND THE ROOT OF THE CHORD
    if len(degree_str) == 1 or (len(degree_str) == 2 and degree_str[1] == '+'):  # case: degree = x, augmented chords
        degree = int(degree_str[0])
        root = SCALES[key][degree - 1]

    elif degree_str == '+4':  # case: augmented 6th
        degree = 6
        root = SCALES[key][degree - 1]
        if _is_major(key):  # case: major key
            root = _flat_alteration(root)  # lower the sixth in a major key

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

    quality_out = Q2S[quality]
    chord_symbol = root + quality_out
    F2S[','.join([key, degree_str, quality])] = chord_symbol
    return chord_symbol


def _flat_alteration(note):
    """ Ex: _flat_alteration(G) = G-,  _flat_alteration(G+) = G """
    return note[:-1] if '+' in note else note + '-'


def _is_major(key):
    """ The input needs to be a string like "A-" for A flat major, "b" for b minor, etc. """
    return key[0].isupper()


def count_records(tfrecord):
    """ Count the number of lines in a tfrecord file. This is useful to establish 'steps_per_epoch' when training """
    c = 0
    if tf.__version__ == '1.12.0':
        for _ in tf.io.tf_record_iterator(tfrecord):
            c += 1
    else:
        for _ in tf.data.TFRecordDataset(tfrecord):
            c += 1
    return c


def visualize_data(data):
    # data = tf.data.TFRecordDataset(input_path)
    temp = data.make_one_shot_iterator()
    x, y = temp.get_next()
    for pr in x:
        sns.heatmap(pr)
        plt.show()
    return


def create_dezrann_annotations(output, n, batch_size, type):
    """
    Create a JSON file for a single aspect of the analysis that is compatible with dezrann, www.dezrann.net
    This allows for a nice visualization of the analysis on top of the partition.
    :param output: The output of the machine learning model.
    :param n: the number of beethoven sonata
    :param batch_size: Just a check. It needs to be one
    :param type: either "true" or "pred"
    :return:
    """

    if batch_size != 1:
        raise NotImplementedError("This script only works for a batch size of one!")
    if type not in ['true', 'pred']:
        raise ValueError(f"The type should be either true or pred, not {type}")

    for j in range(7):
        data = output[j]
        feature = FEATURES[j]
        x = {
            "meta": {
                'title': f"Beethoven sonata no.{n}",
                'name': f"{n} - {feature} {type}",
                'date': str(datetime.now()),
                'producer': 'Algomus team'
            }
        }
        data = data[0]
        data = np.argmax(data, axis=-1)
        labels = []
        start = 0
        for t in range(len(data)):
            if t > 0:
                if data[t] != data[t - 1] or t == len(data) - 1:
                    duration = t / 2 - start
                    labels.append({
                        "type": feature,
                        "start": start,
                        "duration": duration,
                        "staff": "top.1" if type == "true" else "top.2",
                        "tag": TICK_LABELS[j][data[-1]]
                    })
                    start = t / 2
        x['labels'] = labels
        with open(f'analysis_sonata{n}_{j}_{type}.json', 'w') as fp:
            json.dump(x, fp)
    return


if __name__ == '__main__':
    c = count_records(TRAIN_TFRECORDS)
    print(f'There is a total of {c} records in the train file')
    c = count_records(VALID_TFRECORDS)
    print(f'There is a total of {c} records in the validation file')
    c = count_records(TEST_TFRECORDS)
    print(f'There is a total of {c} records in the test file')

"""
Regroups various functions used in the project.
"""
import csv
import os

import numpy as np

from config import NOTES, QUALITY, KEYS_SPELLING, INPUT_TYPES


def setup_tfrecords_paths(tfrecords_folder, tfrecords_basename, mode):
    return [os.path.join(tfrecords_folder, f'{bn}_{mode}.tfrecords') for bn in tfrecords_basename]


def write_tabular_annotations(model_output, timesteps, file_names, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    def _save_csv(current_file, data):
        with open(os.path.join(output_folder, f'{current_file}.csv'), 'w') as fp:
            w = csv.writer(fp)
            w.writerows(data)
        return

    def _set_chunk_offset(file_names, timesteps):
        n = len(timesteps)
        offsets = np.zeros(n)
        for i in range(1, n):
            if file_names[i] == file_names[i - 1]:
                offsets[i] = offsets[i - 1] + timesteps[i - 1]
        return offsets

    offsets = _set_chunk_offset(file_names, timesteps)
    data, current_file, current_label, start, end = [], None, None, 0, None
    for y, ts, name, t0 in zip(model_output, timesteps, file_names, offsets):
        # Save previous analysis if a new one starts
        if name != current_file:  # a new piece has started
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


def int_to_roman(n):
    """ Convert an integer to a Roman numeral. """

    if not 0 < n < 8:
        raise ValueError("Argument must be between 1 and 7")
    ints = (5, 4, 1)
    nums = ('V', 'IV', 'I')
    result = []
    for i in range(len(ints)):
        count = int(n / ints[i])
        result.append(nums[i] * count)
        n -= ints[i] * count
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


def decode_results_tabular(y):
    """
    Transform the outputs of the model into tabular format, example [G+, V/V, D7, '2']

    :param y: it should have shape [features, timesteps], and every element should be an integer indicating the class
    :return: keys, degree, quality, inversions
    """

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

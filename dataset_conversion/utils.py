import csv
import os

import xlrd

from config import DATA_FOLDER, Q2RN, I2RN


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
    num = num.upper() if upper else num.lower()
    if num == 'IV' and qlt == 'M':  # the fourth degree is by default major 7th
        qlt = ''
    return num + qlt + inv + ('/' + den if den != 'I' else '')


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


def transform_bps_chords_file_to_csv(chords_file, output_file):
    workbook = xlrd.open_workbook(chords_file)
    sheet = workbook.sheet_by_index(0)
    chords = []
    t0 = None
    for rowx in range(sheet.nrows):
        cols = sheet.row_values(rowx)
        if t0 is None:
            t0 = cols[0]
        cols[0], cols[1] = cols[0] - t0, cols[1] - t0
        cols[2] = cols[2].replace('+', '#')  # BPS-FH people use + for sharps, while music21 uses #. We stick to #.

        # xlrd.open_workbook automatically casts strings to float if they are compatible. Revert this.
        if isinstance(cols[3], float):  # if type(degree) == float
            cols[3] = str(int(cols[3]))
        if cols[4] == 'a6':  # in the case of aug 6 chords, verify if they're italian, german, or french
            cols[4] = cols[6].split('/')[0]
        cols[5] = str(int(cols[5]))  # re-establish inversion as integers
        chords.append(tuple(cols[:-1]))

    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(chords)
    return


if __name__ == '__main__':
    source_files = [os.path.join(DATA_FOLDER, 'BPS_FH_original', f'{i:02d}', 'chords.xlsx') for i in range(1, 33)]
    output_files = [os.path.join(DATA_FOLDER, 'BPS', 'chords', f'chords_{i}.csv') for i in range(1, 33)]
    for sf, of in zip(source_files, output_files):
        transform_bps_chords_file_to_csv(sf, of)

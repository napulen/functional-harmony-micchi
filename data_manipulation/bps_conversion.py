"""
Transform the original bps dataset into our tabular format. We should never need this again.
"""
import csv
import os

import xlrd

from config import DATA_FOLDER


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

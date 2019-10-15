import csv
import os
from shutil import copyfile

import xlrd
from numpy.random.mtrand import choice, seed

from config import DATA_FOLDER, VALID_INDICES, TRAIN_INDICES


def transform_bps_chord_files_to_csv(chords_file, output_file):
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


def create_training_validation_set_bps(i_trn, i_vld):
    for i in i_trn:
        score_file = os.path.join(DATA_FOLDER, 'BPS', 'scores', f'bps_{i:02d}_01.mxl')
        output_score_file = os.path.join(DATA_FOLDER, 'train', 'scores', f'bps_{i:02d}_01.mxl')
        copyfile(score_file, output_score_file)

        chords_file = os.path.join(DATA_FOLDER, 'BPS', 'chords', f'bps_{i:02d}_01.csv')
        output_chords_file = os.path.join(DATA_FOLDER, 'train', 'chords', f'bps_{i:02d}_01.csv')
        copyfile(chords_file, output_chords_file)
    for i in i_vld:
        score_file = os.path.join(DATA_FOLDER, 'BPS', 'scores', f'bps_{i:02d}_01.mxl')
        output_score_file = os.path.join(DATA_FOLDER, 'valid', 'scores', f'bps_{i:02d}_01.mxl')
        copyfile(score_file, output_score_file)

        chords_file = os.path.join(DATA_FOLDER, 'BPS', 'chords', f'bps_{i:02d}_01.csv')
        output_chords_file = os.path.join(DATA_FOLDER, 'valid', 'chords', f'bps_{i:02d}_01.csv')
        copyfile(chords_file, output_chords_file)
    return


def create_training_validation_set_wtc(s=18):
    seed(s)
    i_trn = set(choice(range(1, 25), size=24, replace=False))
    i_vld = set(range(1, 25)).difference(i_trn)
    for i in i_trn:
        score_file = os.path.join(DATA_FOLDER, 'Bach_WTC_1_Preludes', 'scores', f"wtc_i_prelude_{i:02d}.mxl")
        output_score_file = os.path.join(DATA_FOLDER, 'train', 'scores', f'wtc_i_prelude_{i:02d}.mxl')
        copyfile(score_file, output_score_file)

        chords_file = os.path.join(DATA_FOLDER, 'Bach_WTC_1_Preludes', 'chords', f"wtc_i_prelude_{i:02d}.csv")
        output_chords_file = os.path.join(DATA_FOLDER, 'train', 'chords', f'wtc_i_prelude_{i:02d}.csv')
        copyfile(chords_file, output_chords_file)
    for i in i_vld:
        score_file = os.path.join(DATA_FOLDER, 'Bach_WTC_1_Preludes', 'scores', f"wtc_i_prelude_{i:02d}.mxl")
        output_score_file = os.path.join(DATA_FOLDER, 'valid', 'scores', f'wtc_i_prelude_{i:02d}.mxl')
        copyfile(score_file, output_score_file)

        chords_file = os.path.join(DATA_FOLDER, 'Bach_WTC_1_Preludes', 'chords', f"wtc_i_prelude_{i:02d}.csv")
        output_chords_file = os.path.join(DATA_FOLDER, 'valid', 'chords', f'wtc_i_prelude_{i:02d}.csv')
        copyfile(chords_file, output_chords_file)
    return


def create_training_validation_set_songs(s=18):
    file_names = [fn[:-4] for fn in os.listdir(os.path.join(DATA_FOLDER, '19th_Century_Songs', 'chords'))]
    n = len(file_names)
    seed(s)
    fn_trn = set(choice(file_names, size=n, replace=False))
    fn_vld = set(file_names).difference(fn_trn)
    for fn in fn_trn:
        score_file = os.path.join(DATA_FOLDER, '19th_Century_Songs', 'scores', f"{fn}.mxl")
        output_score_file = os.path.join(DATA_FOLDER, 'train', 'scores', f'ncs_{fn}.mxl')
        copyfile(score_file, output_score_file)

        chords_file = os.path.join(DATA_FOLDER, '19th_Century_Songs', 'chords', f"{fn}.csv")
        output_chords_file = os.path.join(DATA_FOLDER, 'train', 'chords', f'ncs_{fn}.csv')
        copyfile(chords_file, output_chords_file)
    for fn in fn_vld:
        score_file = os.path.join(DATA_FOLDER, '19th_Century_Songs', 'scores', f"{fn}.mxl")
        output_score_file = os.path.join(DATA_FOLDER, 'valid', 'scores', f'ncs_{fn}.mxl')
        copyfile(score_file, output_score_file)

        chords_file = os.path.join(DATA_FOLDER, '19th_Century_Songs', 'chords', f"{fn}.csv")
        output_chords_file = os.path.join(DATA_FOLDER, 'valid', 'chords', f'ncs_{fn}.csv')
        copyfile(chords_file, output_chords_file)
    return


def create_training_validation_set_bsq(s=18):
    file_names = [fn[:-4] for fn in os.listdir(os.path.join(DATA_FOLDER, 'Beethoven_4tets', 'chords'))]
    n = len(file_names)
    seed(s)
    fn_trn = set(choice(file_names, size=n, replace=False))
    fn_vld = set(file_names).difference(fn_trn)
    for fn in fn_trn:
        score_file = os.path.join(DATA_FOLDER, 'Beethoven_4tets', 'scores', f"{fn}.mxl")
        output_score_file = os.path.join(DATA_FOLDER, 'train', 'scores', f'bsq_{fn}.mxl')
        copyfile(score_file, output_score_file)

        chords_file = os.path.join(DATA_FOLDER, 'Beethoven_4tets', 'chords', f"{fn}.csv")
        output_chords_file = os.path.join(DATA_FOLDER, 'train', 'chords', f'bsq_{fn}.csv')
        copyfile(chords_file, output_chords_file)
    for fn in fn_vld:
        score_file = os.path.join(DATA_FOLDER, 'Beethoven_4tets', 'scores', f"{fn}.mxl")
        output_score_file = os.path.join(DATA_FOLDER, 'valid', 'scores', f'bsq_{fn}.mxl')
        copyfile(score_file, output_score_file)

        chords_file = os.path.join(DATA_FOLDER, 'Beethoven_4tets', 'chords', f"{fn}.csv")
        output_chords_file = os.path.join(DATA_FOLDER, 'valid', 'chords', f'bsq_{fn}.csv')
        copyfile(chords_file, output_chords_file)
    return


if __name__ == '__main__':
    os.makedirs(os.path.join(DATA_FOLDER, 'train', 'scores'), exist_ok=True)
    os.makedirs(os.path.join(DATA_FOLDER, 'train', 'chords'), exist_ok=True)
    os.makedirs(os.path.join(DATA_FOLDER, 'valid', 'scores'), exist_ok=True)
    os.makedirs(os.path.join(DATA_FOLDER, 'valid', 'chords'), exist_ok=True)

    create_training_validation_set_bps(TRAIN_INDICES, VALID_INDICES)
    create_training_validation_set_wtc()
    create_training_validation_set_songs()
    create_training_validation_set_bsq()

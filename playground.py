import csv
import logging
import os

import numpy as np
import xlrd

from config import DATA_FOLDER, TRAIN_INDICES, VALID_INDICES
from utils_music import load_chord_labels, shift_chord_labels, segment_chord_labels, encode_chords, attach_chord_root, \
    load_score_spelling_bass, _load_score, calculate_number_transpositions_key
from train_validation_split import create_training_validation_set_bps, create_training_validation_set_wtc, \
    create_training_validation_set_songs, create_training_validation_set_bsq, create_complete_set_bps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


if __name__ == '__main__':
    folders = [os.path.join(DATA_FOLDER, 'train'), os.path.join(DATA_FOLDER, 'valid')]

    os.makedirs(os.path.join(DATA_FOLDER, 'test-bps', 'scores'), exist_ok=True)
    os.makedirs(os.path.join(DATA_FOLDER, 'test-bps', 'chords'), exist_ok=True)
    os.makedirs(os.path.join(DATA_FOLDER, 'train', 'scores'), exist_ok=True)
    os.makedirs(os.path.join(DATA_FOLDER, 'train', 'chords'), exist_ok=True)
    os.makedirs(os.path.join(DATA_FOLDER, 'valid', 'scores'), exist_ok=True)
    os.makedirs(os.path.join(DATA_FOLDER, 'valid', 'chords'), exist_ok=True)

    create_training_validation_set_bps(TRAIN_INDICES, VALID_INDICES)
    create_complete_set_bps()
    create_training_validation_set_wtc()
    create_training_validation_set_songs()
    create_training_validation_set_bsq()

    for folder in folders:
        chords_folder = os.path.join(folder, 'chords')
        scores_folder = os.path.join(folder, 'scores')
        file_names = sorted([fn[:-4] for fn in os.listdir(chords_folder)])
        for fn in file_names:
            if fn not in ['ncs_Chausson_Ernest_-_7_Melodies_Op.2_No.7_-_Le_Colibri']:
                continue
            print(fn)
            cf = os.path.join(chords_folder, f"{fn}.csv")
            sf = os.path.join(scores_folder, f"{fn}.mxl")
            chord_labels = load_chord_labels(cf)
            # for c in chord_labels:
            #     if c['quality'] == 'D7':
            #         print(c)
            piano_roll, nl_pitches, nr_pitches = load_score_spelling_bass(sf, 8)
            nl_keys, nr_keys = calculate_number_transpositions_key(chord_labels)
            nl = min(nl_keys, nl_pitches)
            nr = min(nr_keys, nr_pitches)
            logger.info(f'Acceptable transpositions (pitches, keys): '
                        f'left {nl_pitches, nl_keys}; '
                        f'right {nr_pitches - 1, nr_keys - 1}.')

            score, n_frames = _load_score(sf, 8)
            # measure_offset = list(score.measureOffsetMap().keys())
            # measure_length = np.diff(measure_offset)

            # PROBLEMS IN TOTAL LENGTH IN THE FOLLOWING CASES
            npad = (- n_frames) % 4
            print(f'{fn} - padding length {npad}')
            n_frames_analysis = (n_frames + npad) // 4
            # Verify that the two lengths match
            if n_frames_analysis != chord_labels[-1]['end'] * 2:
                print(f"{fn} - score {n_frames_analysis}, chord labels {chord_labels[-1]['end'] * 2}")

            # for c in chord_labels:
            #     if round(c['end'] % 1., 3) not in [.0, .125, .133, .167, .25, .333, .375, .50, .625, .667, .75, .833,
            #                                        .875]:
            #         print(c['onset'], c['end'])
            # Verify that the two lengths match

            n_frames_chords = n_frames // 4
            for s in range(-nl, nr):
                cl_shifted = shift_chord_labels(chord_labels, s, 'fifth')
                # cl_shifted = chord_labels
                cl_full = attach_chord_root(cl_shifted, pitch_spelling=True)
                cl_segmented = segment_chord_labels(cl_full, n_frames_chords, hsize=4, fpq=8)
                cl_encoded = encode_chords(cl_segmented, 'fifth')

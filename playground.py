import logging
import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from config import DATA_FOLDER, HSIZE, FPQ, PITCH_FIFTHS
from utils_music import _load_score, load_chord_labels, attach_chord_root, segment_chord_labels, shift_chord_labels, \
    encode_chords, load_score_pitch_complete, load_score_spelling_bass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == '__main__':
    folders = [os.path.join(DATA_FOLDER, 'valid')]
    # folders = [os.path.join(DATA_FOLDER, 'train'), os.path.join(DATA_FOLDER, 'valid')]
    # folders = [os.path.join(DATA_FOLDER, 'BPS'), os.path.join(DATA_FOLDER, 'Beethoven_4tets')]

    # folders = [os.path.join(DATA_FOLDER, 'Tavern', 'Mozart'), os.path.join(DATA_FOLDER, 'Tavern', 'Beethoven'),
    #            os.path.join(DATA_FOLDER, '19th_Century_Songs'), os.path.join(DATA_FOLDER, 'Bach_WTC_1_Preludes')]

    for folder in folders:
        total_ql, total_measures, total_rn = 0, 0, 0
        chords_folder = os.path.join(folder, 'chords')
        scores_folder = os.path.join(folder, 'scores')
        file_names = sorted([fn[:-4] for fn in os.listdir(scores_folder)])
        for fn in file_names:
            # _, n, mov = fn.split("_")
            if fn not in ['wtc_i_prelude_01']:
                continue
            print(fn)
            input_type = 'pitch'
            pp = 'fifth' if input_type.startswith(
                'spelling') else 'semitone'  # definition of proximity for pitches

            sf = os.path.join(scores_folder, f"{fn}.mxl")
            cf = os.path.join(chords_folder, f"{fn}.csv")
            chord_labels = load_chord_labels(cf)
            n_frames_analysis = int(chord_labels[-1][1] * 2)
            cl_full = attach_chord_root(chord_labels, input_type.startswith('spelling'))
            cl_segmented = segment_chord_labels(cl_full, n_frames_analysis, hsize=HSIZE, fpq=FPQ)
            cl_shifted = shift_chord_labels(cl_segmented, 2, pp)
            chords = encode_chords(cl_shifted, pp)

            # if 'Tavern' in folder:
            #     cf1 = os.path.join(chords_folder, f"{fn}_A.csv")
            #     cf2 = os.path.join(chords_folder, f"{fn}_B.csv")
            #     cfs = [cf1, cf2]
            #     for cf in cfs:
            #         chord_labels = load_chord_labels(cf)
            #         total_rn += len(chord_labels)
            # else:
            #     cf = os.path.join(chords_folder, f"{fn}.csv")
            #     chord_labels = load_chord_labels(cf)
            #     total_rn += len(chord_labels)

            # for c in chord_labels:
            #     if c['quality'] == 'D7':
            #         print(c)
            # piano_roll, nl_pitches, nr_pitches = load_score_spelling_bass(sf, 8)
            # nl_keys, nr_keys = calculate_number_transpositions_key(chord_labels)
            # nl = min(nl_keys, nl_pitches)
            # nr = min(nr_keys, nr_pitches)
            # logger.info(f'Acceptable transpositions (pitches, keys): '
            #             f'left {nl_pitches, nl_keys}; '
            #             f'right {nr_pitches - 1, nr_keys - 1}.')

            # if int(n) < 32:
            #     continue

            # pr, _, _ = load_score_spelling_bass(sf, 8)
            # sns.set(rc={'figure.figsize': (10., 8.), 'axes.labelsize': 18, 'axes.titlesize': 26})
            # p = sns.heatmap(pr[:, :640])
            # yticks = [i + 7*j + 0.5 for j in range(10) for i in [1, 3, 5]]
            # yticklabels = [PITCH_FIFTHS[int(i) % 35] for i in yticks]
            # p.set(
            #     xticks=range(0, 641, 64), xticklabels=range(0, 641, 64), xlabel='frame',
            #     yticks=yticks, yticklabels=yticklabels, ylabel='bass pitch  global pitch',
            #     title='Bach WTC I Prelude #01 in C'
            # )
            # plt.show()

            pr = load_score_pitch_complete(sf, 8)
            sns.set(rc={'figure.figsize': (10., 8.), 'axes.labelsize': 18, 'axes.titlesize': 26})
            p = sns.heatmap(pr[:, :640])
            p.invert_yaxis()
            yticks = [i + 0.5 for i in range(0, 84, 4)]
            yticklabels = [int(i) + 24 for i in yticks]
            p.set(
                xticks=range(0, 641, 64), xticklabels=range(0, 641, 64), xlabel='frame',
                yticks=yticks, yticklabels=yticklabels, ylabel='midi numbers',
                title='Bach WTC I Prelude #01 in C'
            )
            plt.show()
            # score, n_frames = _load_score(sf, 8)
            # measure_offset = list(score.measureOffsetMap().keys()) + [score.duration.quarterLength]
            # measure_length = np.diff(measure_offset)
            #
            # total_ql += score.duration.quarterLength
            # total_measures += len(measure_length)
            # print(Counter(measure_length))

            # PROBLEMS IN TOTAL LENGTH IN THE FOLLOWING CASES
            # npad = (- n_frames) % 4
            # print(f'{fn} - padding length {npad}')
            # n_frames_analysis = (n_frames + npad) // 4
            # # Verify that the two lengths match
            # if n_frames_analysis != chord_labels[-1]['end'] * 2:
            #     print(f"{fn} - score {n_frames_analysis}, chord labels {chord_labels[-1]['end'] * 2}")
            #
            # # for c in chord_labels:
            # #     if round(c['end'] % 1., 3) not in [.0, .125, .133, .167, .25, .333, .375, .50, .625, .667, .75, .833,
            # #                                        .875]:
            # #         print(c['onset'], c['end'])
            # # Verify that the two lengths match
            #
            # n_frames_chords = n_frames // 4
            # for s in range(-nl, nr):
            #     cl_shifted = shift_chord_labels(chord_labels, s, 'fifth')
            #     # cl_shifted = chord_labels
            #     cl_full = attach_chord_root(cl_shifted, pitch_spelling=True)
            #     cl_segmented = segment_chord_labels(cl_full, n_frames_chords, hsize=4, fpq=8)
            #     cl_encoded = encode_chords(cl_segmented, 'fifth')
        print(f"{folder}: ql = {total_ql}, measures = {total_measures}, RN = {total_rn}")

"""
This is an entry point, no other file should import from this one.
Split the data into training and validation set for every corpora that we have separately.
"""

import os
from shutil import copyfile
from argparse import ArgumentParser

import numpy as np

from config import DATA_FOLDER


# TODO: Implement test set everywhere

def split_like_chen_su(data_folder, bpssynth=False):
    """
    Create the same dataset as Chen and Su for comparison.
    :return:
    """
    os.makedirs(os.path.join(data_folder, 'train', 'scores'))
    os.makedirs(os.path.join(data_folder, 'train', 'chords'))
    os.makedirs(os.path.join(data_folder, 'valid', 'scores'))
    os.makedirs(os.path.join(data_folder, 'valid', 'chords'))
    os.makedirs(os.path.join(data_folder, 'test', 'scores'))
    os.makedirs(os.path.join(data_folder, 'test', 'chords'))

    # The training, validation, and test sonatas used by Chen and Su in their paper, for comparison
    i_trn = [5, 12, 17, 21, 27, 32, 4, 9, 13, 18, 24, 22, 28, 30, 31, 11, 2, 3]
    i_vld = [8, 19, 29, 16, 26, 6, 20]
    i_tst = [1, 14, 23, 15, 10, 25, 7]

    bps_folder = "BPSSynth" if bpssynth else "BPS"

    for i in i_trn:
        score_file = os.path.join(data_folder, bps_folder, 'scores', f'bps_{i:02d}_01.mxl')
        output_score_file = os.path.join(data_folder, 'train', 'scores', f'bps_{i:02d}_01.mxl')
        copyfile(score_file, output_score_file)

        chords_file = os.path.join(data_folder, bps_folder, 'chords', f'bps_{i:02d}_01.csv')
        output_chords_file = os.path.join(data_folder, 'train', 'chords', f'bps_{i:02d}_01.csv')
        copyfile(chords_file, output_chords_file)
    for i in i_vld:
        score_file = os.path.join(data_folder, bps_folder, 'scores', f'bps_{i:02d}_01.mxl')
        output_score_file = os.path.join(data_folder, 'valid', 'scores', f'bps_{i:02d}_01.mxl')
        copyfile(score_file, output_score_file)

        chords_file = os.path.join(data_folder, bps_folder, 'chords', f'bps_{i:02d}_01.csv')
        output_chords_file = os.path.join(data_folder, 'valid', 'chords', f'bps_{i:02d}_01.csv')
        copyfile(chords_file, output_chords_file)
    for i in i_tst:
        score_file = os.path.join(data_folder, bps_folder, 'scores', f'bps_{i:02d}_01.mxl')
        output_score_file = os.path.join(data_folder, 'test', 'scores', f'bps_{i:02d}_01.mxl')
        copyfile(score_file, output_score_file)

        chords_file = os.path.join(data_folder, bps_folder, 'chords', f'bps_{i:02d}_01.csv')
        output_chords_file = os.path.join(data_folder, 'test', 'chords', f'bps_{i:02d}_01.csv')
        copyfile(chords_file, output_chords_file)

def split_like_micchi_et_al(data_folder):
    """
    Create the full dataset presented in Micchi et al., 2020.
    :return:
    """
    os.makedirs(os.path.join(data_folder, 'train', 'scores'))
    os.makedirs(os.path.join(data_folder, 'train', 'chords'))
    os.makedirs(os.path.join(data_folder, 'valid', 'scores'))
    os.makedirs(os.path.join(data_folder, 'valid', 'chords'))

    create_training_validation_set_bps(data_folder)
    create_training_validation_set_wtc(data_folder)
    create_training_validation_set_songs(data_folder)
    create_training_validation_set_bsq(data_folder)
    create_training_validation_set_tavern(data_folder)


def create_training_validation_set_bps(data_folder, i_trn=None, i_vld=None, i_tst=None, s=18):
    np.random.seed(s)
    i_trn = set(np.random.choice(range(1, 33), size=round(0.9 * 32), replace=False)) if i_trn is None else set(i_trn)
    if i_vld is None:
        i_vld = set(range(1, 33)).difference(i_trn)
        if 1 not in i_vld:  # add sonata 1 to the validation set
            j = np.random.choice(list(i_vld))
            i_vld.remove(j)
            i_trn.add(j)
            i_trn.remove(1)
            i_vld.add(1)
    else:
        i_vld = set(i_vld)

    for i in i_trn:
        score_file = os.path.join(data_folder, 'BPS', 'scores', f'bps_{i:02d}_01.mxl')
        output_score_file = os.path.join(data_folder, 'train', 'scores', f'bps_{i:02d}_01.mxl')
        copyfile(score_file, output_score_file)

        chords_file = os.path.join(data_folder, 'BPS', 'chords', f'bps_{i:02d}_01.csv')
        output_chords_file = os.path.join(data_folder, 'train', 'chords', f'bps_{i:02d}_01.csv')
        copyfile(chords_file, output_chords_file)
    for i in i_vld:
        score_file = os.path.join(data_folder, 'BPS', 'scores', f'bps_{i:02d}_01.mxl')
        output_score_file = os.path.join(data_folder, 'valid', 'scores', f'bps_{i:02d}_01.mxl')
        copyfile(score_file, output_score_file)

        chords_file = os.path.join(data_folder, 'BPS', 'chords', f'bps_{i:02d}_01.csv')
        output_chords_file = os.path.join(data_folder, 'valid', 'chords', f'bps_{i:02d}_01.csv')
        copyfile(chords_file, output_chords_file)
    if i_tst is not None:
        os.makedirs(os.path.join(data_folder, 'test', 'scores'))
        os.makedirs(os.path.join(data_folder, 'test', 'chords'))
        for i in i_tst:
            score_file = os.path.join(data_folder, 'BPS', 'scores', f'bps_{i:02d}_01.mxl')
            output_score_file = os.path.join(data_folder, 'test', 'scores', f'bps_{i:02d}_01.mxl')
            copyfile(score_file, output_score_file)

            chords_file = os.path.join(data_folder, 'BPS', 'chords', f'bps_{i:02d}_01.csv')
            output_chords_file = os.path.join(data_folder, 'test', 'chords', f'bps_{i:02d}_01.csv')
            copyfile(chords_file, output_chords_file)

    return


def create_training_validation_set_wtc(data_folder, s=18):
    np.random.seed(s)
    i_trn = set(np.random.choice(range(1, 25), size=round(0.9 * 24), replace=False))
    i_vld = set(range(1, 25)).difference(i_trn)
    if 1 not in i_vld:  # add prelude 1 to the validation set
        j = np.random.choice(list(i_vld))
        i_vld.remove(j)
        i_trn.add(j)
        i_trn.remove(1)
        i_vld.add(1)
    for i in i_trn:
        score_file = os.path.join(data_folder, 'Bach_WTC_1_Preludes', 'scores', f"wtc_i_prelude_{i:02d}.mxl")
        output_score_file = os.path.join(data_folder, 'train', 'scores', f'wtc_i_prelude_{i:02d}.mxl')
        copyfile(score_file, output_score_file)

        chords_file = os.path.join(data_folder, 'Bach_WTC_1_Preludes', 'chords', f"wtc_i_prelude_{i:02d}.csv")
        output_chords_file = os.path.join(data_folder, 'train', 'chords', f'wtc_i_prelude_{i:02d}.csv')
        copyfile(chords_file, output_chords_file)
    for i in i_vld:
        score_file = os.path.join(data_folder, 'Bach_WTC_1_Preludes', 'scores', f"wtc_i_prelude_{i:02d}.mxl")
        output_score_file = os.path.join(data_folder, 'valid', 'scores', f'wtc_i_prelude_{i:02d}.mxl')
        copyfile(score_file, output_score_file)

        chords_file = os.path.join(data_folder, 'Bach_WTC_1_Preludes', 'chords', f"wtc_i_prelude_{i:02d}.csv")
        output_chords_file = os.path.join(data_folder, 'valid', 'chords', f'wtc_i_prelude_{i:02d}.csv')
        copyfile(chords_file, output_chords_file)
    return


def create_training_validation_set_tavern(data_folder, s=18):
    for composer in ['Beethoven', 'Mozart']:
        file_names = [fn.split("_")[0]
                      for fn in os.listdir(os.path.join(data_folder, 'Tavern', composer, 'chords'))
                      if fn.split("_")[1].startswith('A')]
        n = len(file_names)
        np.random.seed(s)
        fn_trn = set(np.random.choice(file_names, size=round(0.9 * n), replace=False))
        fn_vld = set(file_names).difference(fn_trn)
        for fn in fn_trn:
            for ext in ["A", "B"]:
                score_file = os.path.join(data_folder, 'Tavern', composer, 'scores', f'{fn}.mxl')
                output_score_file = os.path.join(data_folder, 'train', 'scores', f'tvn_{"_".join([fn, ext])}.mxl')
                copyfile(score_file, output_score_file)

                chords_file = os.path.join(data_folder, 'Tavern', composer, 'chords', f'{"_".join([fn, ext])}.csv')
                output_chords_file = os.path.join(data_folder, 'train', 'chords', f'tvn_{"_".join([fn, ext])}.csv')
                copyfile(chords_file, output_chords_file)
        for fn in fn_vld:  # In the validation set, put only one analysis.
            score_file = os.path.join(data_folder, 'Tavern', composer, 'scores', f'{fn}.mxl')
            output_score_file = os.path.join(data_folder, 'valid', 'scores', f'tvn_{fn}_A.mxl')
            copyfile(score_file, output_score_file)

            chords_file = os.path.join(data_folder, 'Tavern', composer, 'chords', f'{fn}_A.csv')
            output_chords_file = os.path.join(data_folder, 'valid', 'chords', f'tvn_{fn}_A.csv')
            copyfile(chords_file, output_chords_file)
    return


def create_training_validation_set_songs(data_folder, s=18):
    file_names = [fn[:-4] for fn in os.listdir(os.path.join(data_folder, '19th_Century_Songs', 'chords'))]
    n = len(file_names)
    np.random.seed(s)
    fn_trn = set(np.random.choice(file_names, size=round(0.9 * n), replace=False))
    fn_vld = set(file_names).difference(fn_trn)
    example = 'Schubert_Franz_-_Winterreise_D.911_No.12_-_Einsamkeit'  # this we want to keep for validation
    if example not in fn_vld:  # add sonata 1 to the validation set
        j = np.random.choice(list(fn_vld))
        fn_vld.remove(j)
        fn_trn.add(j)
        fn_trn.remove(example)
        fn_vld.add(example)

    for fn in fn_trn:
        score_file = os.path.join(data_folder, '19th_Century_Songs', 'scores', f"{fn}.mxl")
        output_score_file = os.path.join(data_folder, 'train', 'scores', f'ncs_{fn}.mxl')
        copyfile(score_file, output_score_file)

        chords_file = os.path.join(data_folder, '19th_Century_Songs', 'chords', f"{fn}.csv")
        output_chords_file = os.path.join(data_folder, 'train', 'chords', f'ncs_{fn}.csv')
        copyfile(chords_file, output_chords_file)
    for fn in fn_vld:
        score_file = os.path.join(data_folder, '19th_Century_Songs', 'scores', f"{fn}.mxl")
        output_score_file = os.path.join(data_folder, 'valid', 'scores', f'ncs_{fn}.mxl')
        copyfile(score_file, output_score_file)

        chords_file = os.path.join(data_folder, '19th_Century_Songs', 'chords', f"{fn}.csv")
        output_chords_file = os.path.join(data_folder, 'valid', 'chords', f'ncs_{fn}.csv')
        copyfile(chords_file, output_chords_file)
    return


def create_training_validation_set_bsq(data_folder, s=18):
    file_names = [fn[:-4] for fn in os.listdir(os.path.join(data_folder, 'Beethoven_4tets', 'chords'))]
    n = len(file_names)
    np.random.seed(s)
    fn_trn = set(np.random.choice(file_names, size=round(0.9 * n), replace=False))
    fn_vld = set(file_names).difference(fn_trn)
    for fn in fn_trn:
        score_file = os.path.join(data_folder, 'Beethoven_4tets', 'scores', f"{fn}.mxl")
        output_score_file = os.path.join(data_folder, 'train', 'scores', f'bsq_{fn}.mxl')
        copyfile(score_file, output_score_file)

        chords_file = os.path.join(data_folder, 'Beethoven_4tets', 'chords', f"{fn}.csv")
        output_chords_file = os.path.join(data_folder, 'train', 'chords', f'bsq_{fn}.csv')
        copyfile(chords_file, output_chords_file)
    for fn in fn_vld:
        score_file = os.path.join(data_folder, 'Beethoven_4tets', 'scores', f"{fn}.mxl")
        output_score_file = os.path.join(data_folder, 'valid', 'scores', f'bsq_{fn}.mxl')
        copyfile(score_file, output_score_file)

        chords_file = os.path.join(data_folder, 'Beethoven_4tets', 'chords', f"{fn}.csv")
        output_chords_file = os.path.join(data_folder, 'valid', 'chords', f'bsq_{fn}.csv')
        copyfile(chords_file, output_chords_file)
    return


if __name__ == '__main__':
    parser = ArgumentParser(description='Split the training, validation, and test data')
    parser.add_argument('--chen-su', dest="split_function", action="store_const", const=split_like_chen_su,
                        help="split the data as in Chen and Su (2018) for direct comparison")
    parser.add_argument('--data_folder', action='store', type=str,
                        help=f'the folder where the data is located and where the train, validation, test splits will be created')
    parser.add_argument("--bps-synth", action="store_true",
                        help="work with the synthetic BPS data rather than the real scores")
    parser.set_defaults(data_folder=DATA_FOLDER, split_function=split_like_micchi_et_al, bps_synth=False)
    args = parser.parse_args()
    args.split_function(args.data_folder, bpssynth=args.bps_synth)
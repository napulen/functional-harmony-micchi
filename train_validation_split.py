import os
from shutil import copyfile
import numpy as np


from config import DATA_FOLDER, VALID_INDICES, TRAIN_INDICES


def create_training_validation_set_bps(i_trn=None):
    i_trn = set(np.random.choice(range(1, 33), size=round(0.9*32), replace=False)) if i_trn is None else set(i_trn)
    i_vld = set(range(1, 33)).difference(i_trn)
    if 1 not in i_vld:  # add sonata 1 to the validation set
        j = np.random.choice(list(i_vld))
        i_vld.remove(j)
        i_trn.add(j)
        i_trn.remove(1)
        i_vld.add(1)
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
    np.random.seed(s)
    i_trn = set(np.random.choice(range(1, 25), size=round(0.9*24), replace=False))
    i_vld = set(range(1, 25)).difference(i_trn)
    if 1 not in i_vld:  # add sonata 1 to the validation set
        j = np.random.choice(list(i_vld))
        i_vld.remove(j)
        i_trn.add(j)
        i_trn.remove(1)
        i_vld.add(1)
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


def create_training_validation_set_tavern(s=18):
    for composer in ['Beethoven', 'Mozart']:
        file_names = [fn.split("_")[0]
                      for fn in os.listdir(os.path.join(DATA_FOLDER, 'Tavern', composer, 'chords'))
                      if fn.split("_")[1].startswith('A')]
        n = len(file_names)
        np.random.seed(s)
        fn_trn = set(np.random.choice(file_names, size=round(0.9*n), replace=False))
        fn_vld = set(file_names).difference(fn_trn)
        for fn in fn_trn:
            for ext in ["A", "B"]:
                score_file = os.path.join(DATA_FOLDER, 'Tavern', composer, 'scores', f'{fn}.mxl')
                output_score_file = os.path.join(DATA_FOLDER, 'train', 'scores', f'tvn_{"_".join([fn, ext])}.mxl')
                copyfile(score_file, output_score_file)

                chords_file = os.path.join(DATA_FOLDER, 'Tavern', composer, 'chords', f'{"_".join([fn, ext])}.csv')
                output_chords_file = os.path.join(DATA_FOLDER, 'train', 'chords', f'tvn_{"_".join([fn, ext])}.csv')
                copyfile(chords_file, output_chords_file)
        for fn in fn_vld:  # In the validation set, put only one analysis.
            score_file = os.path.join(DATA_FOLDER, 'Tavern', composer, 'scores', f'{fn}.mxl')
            output_score_file = os.path.join(DATA_FOLDER, 'valid', 'scores', f'tvn_{fn}_A.mxl')
            copyfile(score_file, output_score_file)

            chords_file = os.path.join(DATA_FOLDER, 'Tavern', composer, 'chords', f'{fn}_A.csv')
            output_chords_file = os.path.join(DATA_FOLDER, 'valid', 'chords', f'tvn_{fn}_A.csv')
            copyfile(chords_file, output_chords_file)
    return


def create_training_validation_set_songs(s=18):
    file_names = [fn[:-4] for fn in os.listdir(os.path.join(DATA_FOLDER, '19th_Century_Songs', 'chords'))]
    n = len(file_names)
    np.random.seed(s)
    fn_trn = set(np.random.choice(file_names, size=round(0.9*n), replace=False))
    fn_vld = set(file_names).difference(fn_trn)
    example = 'Schubert_Franz_-_Winterreise_D.911_No.12_-_Einsamkeit'  # this we want to keep for validation
    if example not in fn_vld:  # add sonata 1 to the validation set
        j = np.random.choice(list(fn_vld))
        fn_vld.remove(j)
        fn_trn.add(j)
        fn_trn.remove(example)
        fn_vld.add(example)

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
    np.random.seed(s)
    fn_trn = set(np.random.choice(file_names, size=round(0.9*n), replace=False))
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
    os.makedirs(os.path.join(DATA_FOLDER, 'train', 'scores'))
    os.makedirs(os.path.join(DATA_FOLDER, 'train', 'chords'))
    os.makedirs(os.path.join(DATA_FOLDER, 'valid', 'scores'))
    os.makedirs(os.path.join(DATA_FOLDER, 'valid', 'chords'))

    create_training_validation_set_bps()
    create_training_validation_set_wtc()
    create_training_validation_set_songs()
    create_training_validation_set_bsq()
    create_training_validation_set_tavern()

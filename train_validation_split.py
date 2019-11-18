import os
from shutil import copyfile

from numpy.random.mtrand import choice, seed

from config import DATA_FOLDER, VALID_INDICES, TRAIN_INDICES


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


def create_training_validation_set_tavern(s=18):
    for composer in ['Beethoven', 'Mozart']:
        file_names = [fn[:-4] for fn in os.listdir(os.path.join(DATA_FOLDER, 'Tavern', composer, 'chords'))]
        n = len(file_names)
        seed(s)
        fn_trn = set(choice(file_names, size=n, replace=False))
        fn_vld = set(file_names).difference(fn_trn)
        for fn in fn_trn:
            score_file = os.path.join(DATA_FOLDER, 'Tavern', composer, 'scores', f'{fn.split("_")[0]}.mxl')
            output_score_file = os.path.join(DATA_FOLDER, 'train', 'scores', f'tvn_{fn}.mxl')
            copyfile(score_file, output_score_file)
    
            chords_file = os.path.join(DATA_FOLDER, 'Tavern', composer, 'chords', f"{fn}.csv")
            output_chords_file = os.path.join(DATA_FOLDER, 'train', 'chords', f'tvn_{fn}.csv')
            copyfile(chords_file, output_chords_file)
        for fn in fn_vld:
            score_file = os.path.join(DATA_FOLDER, 'Tavern', composer, 'scores', f'{fn.split("_")[0]}.mxl')
            output_score_file = os.path.join(DATA_FOLDER, 'valid', 'scores', f'tvn_{fn}.mxl')
            copyfile(score_file, output_score_file)
    
            chords_file = os.path.join(DATA_FOLDER, 'Tavern', composer, 'chords', f"{fn}.csv")
            output_chords_file = os.path.join(DATA_FOLDER, 'valid', 'chords', f'tvn_{fn}.csv')
            copyfile(chords_file, output_chords_file)
    return


def create_training_validation_set_songs(s=18):
    file_names = [fn[:-4] for fn in os.listdir(os.path.join(DATA_FOLDER, '19th_Century_Songs', 'chords'))]
    n = len(file_names)
    seed(s)
    fn_trn = set(choice(file_names, size=n, replace=False))
    fn_vld = set(file_names).difference(fn_trn)
    example = 'Schubert_Franz_-_Winterreise_D.911_No.12_-_Einsamkeit'  # this we want to keep for validation
    fn_trn.discard(example)
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
    create_training_validation_set_tavern()

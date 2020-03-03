import os
import sys
import logging
from argparse import ArgumentParser
from datetime import datetime
from math import ceil

import numpy as np
from tensorflow.python.keras.models import load_model

from config import DATA_FOLDER, FPQ, CHUNK_SIZE
from converters import ConverterTab2Rn
from utils import find_input_type, create_dezrann_annotations, create_tabular_annotations
from utils_music import load_score_pitch_complete, load_score_pitch_bass, load_score_pitch_class, \
    load_score_spelling_complete, load_score_spelling_bass, load_score_spelling_class

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def analyse_music(sf, model, model_name, input_type, analyses_folder, converter=None):
    score, mask = prepare_input_from_xml(sf, input_type)
    y_pred = model.predict((score, mask))

    ts = np.squeeze(np.sum(mask, axis=1), axis=1)
    n_chunks = len(ts)
    test_pred = [[d[e, :ts[e]] for d in y_pred] for e in range(n_chunks)]
    # song_name = os.path.basename(sf).split('.')[0]
    song_name = 'automatic'  # this effectively calls all analyses files 'automatic', use with care
    file_names = [song_name] * n_chunks
    create_dezrann_annotations(model_output=test_pred, model_name=model_name, annotations=None, timesteps=ts,
                               file_names=file_names, output_folder=analyses_folder)
    create_tabular_annotations(model_output=test_pred, timesteps=ts,
                               file_names=file_names, output_folder=analyses_folder)
    try:
        if converter is None:
            converter = ConverterTab2Rn()
        converter.convert_file(sf, os.path.join(analyses_folder, song_name + '.csv'),
                               os.path.join(analyses_folder, song_name + '.txt'))
    except:
        print(f"Couldn't create the rntxt version of {os.path.basename(sf).split('.')[0]}")

    return


def prepare_input_from_xml(sf, input_type):
    logger.info(f"Analysing {sf}")

    if input_type.startswith('pitch'):
        if 'complete' in input_type:
            piano_roll = load_score_pitch_complete(sf, FPQ)
        elif 'bass' in input_type:
            piano_roll = load_score_pitch_bass(sf, FPQ)
        elif 'class' in input_type:
            piano_roll = load_score_pitch_class(sf, FPQ)
        else:
            raise NotImplementedError("verify the input_type")
    elif input_type.startswith('spelling'):
        if 'complete' in input_type:
            piano_roll, _, _ = load_score_spelling_complete(sf, FPQ)
        elif 'bass' in input_type:
            piano_roll, _, _ = load_score_spelling_bass(sf, FPQ)
        elif 'class' in input_type:
            piano_roll, _, _ = load_score_spelling_class(sf, FPQ)
        else:
            raise NotImplementedError("verify the input_type")
    else:
        raise NotImplementedError("verify the input_type")

    if input_type.endswith('cut'):
        start, end = 0, CHUNK_SIZE
        score = []
        mask = []
        while 4 * start < piano_roll.shape[1]:
            pr = np.transpose(piano_roll[:, 4 * start:4 * end])
            ts = pr.shape[0]
            # 4*CHUNK_SIZE because it was adapted to chords, not piano rolls who have a higher resolution
            score.append(np.pad(pr, ((0, 4 * CHUNK_SIZE - ts), (0, 0)), mode='constant'))
            start += CHUNK_SIZE
            end += CHUNK_SIZE
            m = np.ones(CHUNK_SIZE, dtype=bool)  # correct size
            m[ceil(ts // 4):] = 0  # put zeroes where no data
            mask.append(m[:, np.newaxis])
    else:
        score = [np.transpose(piano_roll)]
        mask = np.ones(ceil(len(piano_roll) // 4))[:, np.newaxis]
    return np.array(score), np.array(mask)


def get_args():
    music_path = input("Please provide the path to the score score you want to analyse. "
                       "If the provided path is a folder, all compatible files inside will be analysed.\n")
    input_valid = False
    model = None
    while not input_valid:
        models = [m for m in os.listdir('models') if m + '.h5' in os.listdir(os.path.join('models', m))]
        model_choice = [str(n) + '. ' + m for n, m in enumerate(models)]
        print(f"Please choose the model you want to apply:")
        for c in model_choice:
            print(c)
        model_index = input()
        try:
            model_index = int(model_index)
        except ValueError:
            print("I didn't understand your choice.")
            continue
        if model_index < 0 or model_index > len(models):
            print(f"Your choice should be an integer between 0 and {len(models) - 1}")
            continue
        confirmation = input(f"You choose to use {models[model_index]}. Do you confirm? [y/n] ")
        while confirmation not in 'YyNn':
            print("I didn't understand. Please reply either y or n.")
            confirmation = input(f"You choose to use {models[model_index]}. Do you confirm? [y/n] ")
        if confirmation in 'Nn':
            print("Process aborted.")
        else:
            print("Choice validated.")
            model = models[model_index]
            input_valid = True
    return music_path, model


if __name__ == '__main__':
    model_name = 'conv_gru_spelling_bass_cut_3'
    input_type = find_input_type(model_name)

    music_path = os.path.join('data', 'OpenScore-LiederCorpus')
    # try:
    #     files = sorted([os.path.join(music_path, m) for m in os.listdir(music_path)])
    # except NotADirectoryError:
    #     files = [music_path]
    # analyses_folder = os.path.join('analyses', '_'.join([model_name, datetime.now().strftime("%Y-%m-%d_%H-%M")]))
    w = os.walk(music_path, topdown=False)

    # for i in w:
    #     analyses_folder = i[0]
    #     fn = [f for f in i[2] if f.startswith('automated')]
    #     for f in fn:
    #         print(f'removing {f}')
    #         os.remove(os.path.join(analyses_folder, f))

    # model_path = os.path.join('runs', 'run_06_(paper)', 'models', model_name, model_name + '.h5')
    model_path = 'run_model.h5'
    model = load_model(model_path)
    conv = ConverterTab2Rn()
    # TODO: The for loop currently returns an error when w is over, while it should just silently quit I think...
    for i in w:
        fn = [f for f in i[2] if f.endswith('mxl')]
        try:
            fn = fn[0]
        except IndexError:
            continue
        analyses_folder = i[0]
        sf = os.path.join(analyses_folder, fn)
        analyse_music(sf, model, model_name, input_type, analyses_folder, conv)

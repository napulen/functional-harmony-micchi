import os
from datetime import datetime
from math import ceil

import numpy as np

from argparse import ArgumentParser
from tensorflow.python.keras.models import load_model

from config import DATA_FOLDER, FPQ, CHUNK_SIZE
from preprocessing import logger
from utils import find_input_type, create_dezrann_annotations
from utils_music import load_score_pitch_complete, load_score_pitch_bass, load_score_pitch_class, \
    load_score_spelling_complete, load_score_spelling_bass, load_score_spelling_class


def analyse_music(sf, model, input_type, dezrann_output_folder):
    score, mask = prepare_input_from_xml(sf, input_type)
    y_pred = model.predict((score, mask))

    ts = np.squeeze(np.sum(mask, axis=1), axis=1)
    n_chunks = len(ts)
    test_pred = [[d[e, :ts[e]] for d in y_pred] for e in range(n_chunks)]
    file_names = [os.path.basename(sf).split('.')[0]] * n_chunks
    create_dezrann_annotations(model_output=test_pred, annotations=None, timesteps=ts,
                               file_names=file_names, output_folder=dezrann_output_folder)

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
    parser = ArgumentParser(description='Do a roman numeral analysis of the scores you provide.')
    parser.add_argument('-i', dest='interactive', action='store_true',
                        help='activate interactive mode for setting parameters')
    parser.add_argument('--score', dest='music_path', action='store', type=str,
                        help='score or folder containing the scores')
    parser.add_argument('--model', dest='model_name', action='store', type=str, help='name of the model')
    parser.set_defaults(interactive=False)
    parser.set_defaults(music_path=os.path.join(DATA_FOLDER, 'BPS_other-movements', 'scores'))
    parser.set_defaults(model_name='conv_gru_spelling_bass_cut_0')
    args = parser.parse_args()

    if args.interactive:
        args.music_path, args.model_name = get_args()
    else:
        print(f"Selected scores: {args.music_path}\n"
              f"Selected model: {args.model_name}\n"
              f"If that's not what you wanted, try to run the script with the option -i (interactive mode)")
    try:
        files = sorted([os.path.join(args.music_path, m) for m in os.listdir(args.music_path)])
    except NotADirectoryError:
        files = [args.music_path]

    dezrann_folder = os.path.join('analyses', '_'.join([args.model_name, datetime.now().strftime("%Y-%m-%d_%H-%M")]))
    model_folder = os.path.join('models', args.model_name)
    model = load_model(os.path.join(model_folder, args.model_name + '.h5'))
    input_type = find_input_type(args.model_name)
    for sf in files:
        analyse_music(sf, model, input_type, dezrann_folder)

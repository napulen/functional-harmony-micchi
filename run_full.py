"""
This is an entry point, no other file should import from this one.
Use the trained model to analyse the scores given. The paths must be set by hand in the code.
This is very similar to the rn_app we publish, but it can be adapted to different situations.
For example, a different structure of the data, when the scores are not in the same directory but are stored in
a tree-structured fashion such as 'composer/opus/number_in_opus/files'
"""
import logging
import os

import numpy as np
from tensorflow.python.keras.models import load_model

from converters import ConverterTab2Rn, ConverterTab2Dez
from utils import find_input_type, write_tabular_annotations
from utils_music import prepare_input_from_xml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def analyse_music(sf, model, input_type, analyses_folder, tab2rn=None, tab2dez=None):
    logger.info(f"Analysing {sf}")
    score, mask = prepare_input_from_xml(sf, input_type)
    y_pred = model.predict((score, mask))

    ts = np.squeeze(np.sum(mask, axis=1), axis=1)  # It is a vector with size equal to the number of chunks in which the song is split
    test_pred = [[d[n, :end] for d in y_pred] for n, end in enumerate(ts)]
    # song_name = os.path.basename(sf).split('.')[0]
    song_name = 'automatic'  # this effectively calls all analyses files 'automatic', use with care
    file_names = [song_name] * len(ts)
    write_tabular_annotations(model_output=test_pred, timesteps=ts, file_names=file_names, output_folder=analyses_folder)

    # Conversions
    out_fp_no_ext = os.path.join(analyses_folder, song_name)
    if tab2dez is None:
        tab2dez = ConverterTab2Dez()
    tab2dez.convert_file(sf, out_fp_no_ext + '.csv', out_fp_no_ext + '.dez')
    try:
        if tab2rn is None:
            tab2rn = ConverterTab2Rn()
        tab2rn.convert_file(sf, out_fp_no_ext + '.csv', out_fp_no_ext + '.txt')
    except:
        print(f"Couldn't create the rntxt version of {os.path.basename(sf).split('.')[0]}")

    return


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
        analyse_music(sf, model, input_type, analyses_folder, conv)

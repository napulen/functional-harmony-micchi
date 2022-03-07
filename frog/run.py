"""
This is an entry point, no other file should import from this one.
Use the trained model to analyse the scores given.
"""
import csv
import glob
import json
import logging
import os
from argparse import ArgumentParser
from datetime import datetime

import music21.converter
import numpy as np
import tensorflow as tf

from frog import CHUNK_SIZE, FROG_MODEL_PATH, OUTPUT_FPC
from frog.converters.annotation_converters import ConverterTab2Dez, ConverterTab2Rn
from frog.label_codec import LabelCodec
from frog.models.models import load_model_with_info
from frog.preprocessing.preprocess_scores import prepare_input_from_score_file

logger = logging.getLogger(__name__)


def create_annotated_musicxml(sf, rntxt_fp, annotated_fp, ground_truth_fp=None):
    score = music21.converter.parse(sf)
    analysis = music21.converter.parse(rntxt_fp, format="RomanText")
    analysis.parts[0].partName = "predicted"
    score.insert(0, analysis.parts[0])
    if ground_truth_fp is not None:
        ground_truth = music21.converter.parse(ground_truth_fp, format="RomanText")
        ground_truth.parts[0].partName = "ground truth"
        score.insert(0, ground_truth.parts[0])
    score.write("mxl", annotated_fp)
    return


def tabular_output_single_file(y, label_codec):
    labels = label_codec.decode_results_tabular(y)
    current_label = None
    start = 0
    data = []
    for t, label in enumerate(labels):
        if current_label is None:
            current_label = label
        if np.any(label != current_label):
            end = t / OUTPUT_FPC
            data.append([start, end, *current_label])
            start = end
            current_label = label
    data.append([start, len(labels) / OUTPUT_FPC, *current_label])
    return data


def analyse_file(
    sf,
    model,
    input_type,
    output_mode,
    analyses_folder=None,
    tab2rn=True,
    tab2dez=True,
    annotated=True,
    gt_fp=None,
    metrical=True,
):
    def join_outputs(y_pred, time_steps):
        features = []
        for f in y_pred:
            joined_chunks = tf.concat(tf.unstack(f), axis=0)
            removed_mask = tf.gather(joined_chunks, np.arange(time_steps), axis=0)
            features.append(removed_mask)
        return features

    logger.info(f"Analysing {sf}")
    x, meta = prepare_input_from_score_file(sf, input_type, CHUNK_SIZE, metrical)
    y_pred = model.sample(x)

    # time_steps is the vector of the number of valid time_steps in each chunk
    time_steps = np.sum(x[2], dtype=int)  # x[2] is the mask of the padding
    test_pred = join_outputs(y_pred, time_steps)
    # test_pred = [[d[n, :end] for d in y_pred] for n, end in enumerate(time_steps)]
    song_name = meta[0][0]
    spelling, _octave = input_type.split("_")
    lc = LabelCodec(spelling=spelling == "spelling", mode=output_mode, strict=False)
    data = tabular_output_single_file(test_pred, lc)

    if analyses_folder is not None:
        out_fp_no_ext = os.path.join(analyses_folder, song_name)
        with open(out_fp_no_ext + ".csv", "w") as fp:
            csv.writer(fp).writerows(data)
        # Conversions
        if tab2dez:
            dezrann_converter = ConverterTab2Dez()
            dezrann_converter.convert_file(sf, out_fp_no_ext + ".csv", out_fp_no_ext + ".dez")
        if tab2rn:
            try:
                rn_converter = ConverterTab2Rn()
                rn_converter.convert_file(sf, out_fp_no_ext + ".csv", out_fp_no_ext + ".rntxt")
                if annotated:
                    create_annotated_musicxml(
                        sf, out_fp_no_ext + ".rntxt", out_fp_no_ext + ".mxl", gt_fp
                    )
            except:
                logger.warning(f"Couldn't create the rntxt version of {song_name}")
    return data


def get_frog_data(fpath):
    model, info = load_model_with_info(FROG_MODEL_PATH)

    return analyse_file(fpath, model, info["input type"], info["output mode"], metrical=False)


def main(opts):
    parser = ArgumentParser(description="Do a roman numeral analysis of the scores you provide.")
    parser.add_argument("files", nargs="+", help="path to the scores (wildcard * accepted)")
    parser.add_argument(
        "-g",
        dest="ground_truth",
        help="path to the folder containing the ground truth analyses, if available",
    )
    parser.add_argument("-p", dest="model_path", help="path to the model folder")
    parser.add_argument("-o", dest="output_folder", help="The output root folder")
    parser.add_argument("-v", dest="verbose", action="store_true", help="Make all output formats")
    args = parser.parse_args(opts)
    files = [f for x in args.files for f in glob.glob(x)]
    logger.info(f"Selected scores:{files}")
    v = args.verbose
    model, model_info = load_model_with_info(args.model_path)

    output_folder = "_".join([args.output_folder, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")])
    os.makedirs(output_folder)
    run_info = {"model": os.path.basename(args.model_path)}
    with open(os.path.join(output_folder, "info.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    gt_fp = None
    for f in reversed(sorted(files)):
        if args.ground_truth is not None:
            fname = os.path.splitext(os.path.basename(f))[0]
            gt_fp = os.path.join(args.ground_truth, fname + ".txt")
            if not os.path.isfile(gt_fp):
                logger.warning(f"Couldn't find an analysis for the current file at {gt_fp}")
                gt_fp = None
        try:
            analyse_file(
                f,
                model,
                model_info["input type"],
                model_info["output mode"],
                output_folder,
                v,
                v,
                v,
                gt_fp,
            )
        except Exception as e:
            logger.warning(f"Couldn't analyse file {f}, skipping it.")

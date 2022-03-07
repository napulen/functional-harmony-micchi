import json
import logging
import os
from argparse import ArgumentParser

import numpy as np

from frog.analysis.analyse_results import (
    analyse_results,
    analyse_results_concatenated,
    compute_mir_eval_metrics,
    create_mir_eval_lab_files,
    make_plots,
)
from frog.label_codec import LabelCodec
from frog.preprocessing.preprocess_chords import import_chords

logger = logging.getLogger(__name__)


def get_results(fp, lc):
    chords_orig = lc.encode_chords(import_chords(fp, lc, 2))  # shape [T](NF,)
    chords = list(np.transpose(chords_orig))  # shape [NF](T)
    res = []
    for n, f in enumerate(lc.output_features):
        depth = lc.output_size[f]
        temp = np.eye(depth)[chords[n]]
        res.append(temp)
    return res


def model_analysis_single_file(ground_truth, prediction, spelling):
    lc = LabelCodec(spelling=spelling == "spelling", mode="legacy", strict=False)

    y_true = get_results(ground_truth, lc)
    y_pred = get_results(prediction, lc)
    nt, np = len(y_true[0]), len(y_pred[0])
    if nt != np:
        logger.warning(
            f"Different length between ground truth ({nt}) and prediction ({np})."
            f" Cutting to the shortest of the two"
        )
        n = min(nt, np)
        y_true = [y[:n] for y in y_true]
        y_pred = [y[:n] for y in y_pred]
    analysis_1 = analyse_results_concatenated(y_true, y_pred, spelling, "legacy")
    return analysis_1


def model_analysis_from_csv(folder_ground_truth, folder_predictions, spelling):
    file_names = sorted(os.listdir(folder_ground_truth))
    analysis = dict()
    for f in file_names:
        logger.info(f"Analysing file {f}")
        analysis[f] = model_analysis_single_file(
            os.path.join(folder_ground_truth, f), os.path.join(folder_predictions, f), spelling
        )
    return analysis


def model_analysis_global_output_for_HT(
        folder_ground_truth, folder_predictions, spelling, output_folder
):
    lc = LabelCodec(spelling=False, mode="legacy", strict=False)
    y_true = []
    y_pred = []

    for f in sorted(os.listdir(folder_ground_truth)):
        if ".DS_Store" not in f:
            ground_truth = os.path.join(folder_ground_truth, f)
            prediction = os.path.join(folder_predictions, f)
            y_true_file = get_results(ground_truth, lc)
            y_pred_file = get_results(prediction, lc)
            nt, np = len(y_true_file[0]), len(y_pred_file[0])
            if nt != np:
                logger.warning(
                    f"Different length between ground truth ({nt}) and prediction ({np})."
                    f" Cutting to the shortest of the two"
                )
                n = min(nt, np)
                y_true_file = [y[:n] for y in y_true_file]
                y_pred_file = [y[:n] for y in y_pred_file]

            y_true.append(y_true_file)
            y_pred.append(y_pred_file)

    ## Get visualisations
    make_plots(y_true, y_pred, spelling, False, output_folder)

    ## Create metrics from mir-eval library
    create_mir_eval_lab_files(y_true, y_pred, spelling, False, output_folder)

    ## Get mir-eval metrics
    compute_mir_eval_metrics(
        os.path.join(output_folder, "annotations_true"),
        os.path.join(output_folder, "annotations_predicted"),
        output_folder,
    )
    acc = analyse_results(y_true, y_pred, spelling, "legacy")

    with open(os.path.join(output_folder, "accuracy.json"), "w") as f:
        json.dump(acc, f, indent=2)
    return acc


def model_comparison_from_csv(folder_ground_truth, folder_1, folder_2, spelling):
    lc = LabelCodec(spelling=spelling == "spelling", mode="legacy", strict=False)
    file_names = sorted(os.listdir(folder_ground_truth))
    y_pred_1, y_pred_2, y_true = [], [], []
    processed, not_found, not_valid, inconsistent = [], [], [], []
    for f in file_names:
        logger.info(f"Analysing file {f}")
        if not (
            os.path.isfile(os.path.join(folder_1, f)) and os.path.isfile(os.path.join(folder_2, f))
        ):
            logger.warning(f"Couldn't find one of the outputs for {f}, skipping it.")
            not_found.append(f)
            continue
        try:
            temp_1 = get_results(os.path.join(folder_1, f), lc)
            temp_2 = get_results(os.path.join(folder_2, f), lc)
            temp_gt = get_results(os.path.join(folder_ground_truth, f), lc)
        except:
            logger.warning(f"At least one invalid analysis for {f}, skipping it.")
            not_valid.append(f)
            continue

        if len(temp_1[0]) == len(temp_2[0]) == len(temp_gt[0]):
            y_pred_1.append(temp_1)
            y_pred_2.append(temp_2)
            y_true.append(temp_gt)
            processed.append(f)
        else:
            logger.warning(
                f"{f} skipped because of different analyses lengths:"
                f" model 1: {len(temp_1[0])},"
                f" model 2: {len(temp_2[0])},"
                f" ground truth: {len(temp_gt[0])}."
            )
            inconsistent.append(f)
    logger.info(f"Giving accuracies for {len(processed)} files")
    analysis_1 = analyse_results(y_true, y_pred_1, spelling, "legacy")
    analysis_2 = analyse_results(y_true, y_pred_2, spelling, "legacy")
    return analysis_1, analysis_2


if __name__ == "__main__":
    parser = ArgumentParser(description="Compare two models from the csv they produce")
    parser.add_argument("-g", dest="ground_truth", help="The folder with ground truth csv files")
    parser.add_argument("-a", dest="res_1", help="The folder with results from model 1")
    parser.add_argument("-b", dest="res_2", help="The folder with results from model 2")
    parser.add_argument("-s", dest="spelling", choices=["pitch", "spelling"])
    parser.add_argument("-o", dest="output_folder")
    args = parser.parse_args()
    if args.res_2 is None:
        analysis = model_analysis_from_csv(args.ground_truth, args.res_1, args.spelling)
        with open(args.output_folder, "w") as f:
            json.dump(analysis, f, indent=2)
    else:
        a1, a2 = model_comparison_from_csv(args.ground_truth, args.res_1, args.res_2, args.spelling)
        m1 = f"model1: {os.path.basename(args.res_1)}"
        m2 = f"model2: {os.path.basename(args.res_2)}"
        analyses = {k: {m1: a1[k], m2: a2[k]} for k in a1}
        with open(args.output_folder, "w") as f:
            json.dump(analyses, f, indent=2)

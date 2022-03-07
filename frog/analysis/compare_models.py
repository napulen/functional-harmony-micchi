import argparse
import csv
import logging
import os
import sys

import pandas as pd

from frog import DATA_FOLDER, INPUT_TYPES, MODEL_TYPES
from frog.analysis.analyse_results import analyse_results, generate_results

logger = logging.getLogger(__name__)


# TODO: This module is broken! Remove?


def _find_input_type(model_name):
    for it in INPUT_TYPES:
        if it in model_name:
            return it
    raise AttributeError("can't determine which data needs to be fed to the algorithm...")


def _find_model_type(model_name):
    for mt in MODEL_TYPES:
        if mt in model_name:
            return mt
    raise AttributeError("can't determine which type of model to build...")


def _find_tfrecords_path(data_folder, input_type):
    candidates = os.listdir(data_folder)
    correct_input_type = [f for f in candidates if input_type in f]
    most_recent = sorted(correct_input_type)[-1]
    return os.path.join(data_folder, most_recent, f"test_{input_type}.tfrecords")


def compare_results(data_folder, logs_folder):
    """
    Check all the models in the log folder and calculate their accuracy scores, then write a comparison table to file

    :param data_folder:
    :param logs_folder:
    :param export_annotations: boolean, whether to write analyses to file
    :return:
    """
    results = []
    for i, model_name in enumerate(os.listdir(logs_folder)):
        print(f"model {i + 1} out of {len(logs_folder)} - {model_name}")
        model_folder = os.path.join(logs_folder, model_name)
        model_path = os.path.join(model_folder, model_name)
        if not os.path.isdir(model_path):
            logger.warning(f"Could not find a model in {model_path}. Skipping it.")
            continue
        input_type = _find_input_type(model_name)
        model_type = _find_model_type(model_name)
        data_path = _find_tfrecords_path(data_folder, input_type)
        ys_true, ys_pred, info = generate_results(
            data_path, model_path, model_type, input_type, verbose=False
        )
        accuracies = analyse_results(ys_true, ys_pred, verbose=False)
        results.append((model_name, accuracies))

    return results


def _write_comparison_file(model_outputs, fp_out):
    """
    Take the output from several models and writes them to a general comparison file.
    The outputs need to be stored in tuples. The first element is the model name, the second is a dictionary
    containing the name of a feature as key and the accuracy as value

    :param model_outputs:
    :param fp_out:
    :return:
    """
    features = list(model_outputs[0][1].keys())

    with open(fp_out, "w+") as f:
        w = csv.writer(f)
        w.writerow(["model name"] + features)
        for model_name, accuracies in model_outputs:
            w.writerow([model_name] + [round(accuracies[feat], 2) for feat in features])
        bps_paper = {
            "key": 66.65,
            "quality": 60.59,
            "inversion": 59.1,
            "degree": 51.79,
            "tonicised": 3.97,
            "roman + inv": 25.69,
        }

        ht_paper = {
            "key": 78.35,
            "quality": 74.60,
            "inversion": 62.13,
            "degree": 65.06,
            "tonicised": 68.15,
        }

        temperley = {
            "key": 67.03,
        }

        for feat in features:
            if feat not in bps_paper.keys():
                bps_paper[feat] = "NA"
            if feat not in ht_paper.keys():
                ht_paper[feat] = "NA"
            if feat not in temperley.keys():
                temperley[feat] = "NA"

        w.writerow(["bps-fh_paper"] + [bps_paper[feat] for feat in features])
        w.writerow(["ht_paper"] + [ht_paper[feat] for feat in features])
        w.writerow(["temperley"] + [temperley[feat] for feat in features])
    return


# FIXME: Fix this function! Those are unsafe ways to check for model and input type
def _average_results(fp_in, fp_out):
    """
    Write to fp_out the results in fp_in marginalized over one feature at a time
    :param fp_in: The file path to the comparison file we want to average
    """
    data = pd.read_csv(fp_in, header=0, index_col=0)
    res = pd.DataFrame()
    res["c1_local"] = data.loc[data.index.str.contains("_local_")].mean()
    res["c1_global"] = data.loc[
        data.index.str.contains("conv_") & ~data.index.str.contains("_local_")
    ].mean()
    res["c2_conv_dil"] = data.loc[data.index.str.contains("conv_dil")].mean()
    res["c2_conv_gru"] = data.loc[data.index.str.contains("conv_gru")].mean()
    res["c2_gru"] = data.loc[
        data.index.str.contains("gru_") & ~data.index.str.contains("conv_")
    ].mean()
    res["c3_spelling"] = data.loc[data.index.str.contains("_spelling_")].mean()
    res["c3_pitch"] = data.loc[data.index.str.contains("_pitch_")].mean()
    res["c4_complete"] = data.loc[data.index.str.contains("_complete_")].mean()
    res["c4_bass"] = data.loc[data.index.str.contains("_bass_")].mean()
    res["c4_class"] = data.loc[data.index.str.contains("_class_")].mean()
    res = res.transpose()
    # FIXME: This specification of the keys to call is a bit risky, refactor this
    columns = [
        "key",
        "degree",
        "quality",
        "inversion",
        "roman + inv",
        "tonicised degree",
        "d7 no inv",
    ]
    res[columns].to_csv(fp_out)
    return


# FIXME: Fix this function! Those are unsafe ways to check for model and input type
def _t_test_results(fp_in, columns=None):
    """
    Print to screen the results of the t-test on the importance of architecture choices.

    :param fp_in: The file path to the comparison file
    """
    data = pd.read_csv(fp_in, header=0, index_col=0)
    from scipy.stats import ttest_ind

    if columns is None:
        columns = data.columns
    for col in columns:
        c1_local = data.loc[data.index.str.contains("_local_"), col].to_numpy()
        c1_global = data.loc[
            data.index.str.contains("conv_") & ~data.index.str.contains("_local_"), col
        ].to_numpy()
        c2_conv_dil = data.loc[data.index.str.contains("conv_dil"), col].to_numpy()
        c2_conv_gru = data.loc[data.index.str.contains("conv_gru"), col].to_numpy()
        c2_gru = data.loc[
            data.index.str.contains("gru_") & ~data.index.str.contains("conv_"), col
        ].to_numpy()
        c3_spelling = data.loc[data.index.str.contains("_spelling_"), col].to_numpy()
        c3_pitch = data.loc[data.index.str.contains("_pitch_"), col].to_numpy()
        c4_complete = data.loc[data.index.str.contains("_complete_"), col].to_numpy()
        c4_bass = data.loc[data.index.str.contains("_bass_"), col].to_numpy()
        c4_class = data.loc[data.index.str.contains("_class_"), col].to_numpy()

        comparisons = [
            (c1_global, c1_local, "global vs. local"),
            (c2_gru, c2_conv_dil, "gru vs conv_dil"),
            (c2_gru, c2_conv_gru, "gru vs conv_gru"),
            (c2_conv_dil, c2_conv_gru, "conv_dil vs conv_gru"),
            (c3_pitch, c3_spelling, "pitch vs. spelling"),
            (c4_complete, c4_class, "complete vs. class"),
            (c4_complete, c4_bass, "complete vs. bass"),
            (c4_class, c4_bass, "class vs. bass"),
        ]

        print(col)
        for c in comparisons:
            a, b, t = c
            print(f"{t:<21}: p-value {ttest_ind(a, b).pvalue:.1e}")
        print("")
    return


def main(opts):
    parser = argparse.ArgumentParser(description="Analyse the results of a single model")
    parser.add_argument(
        "-m",
        dest="models",
        action="store",
        type=str,
        help="path to the logs folder where models are stored",
    )
    parser.add_argument(
        "-o",
        dest="output_path",
        action="store",
        type=str,
        help="path where to store comparison results, a csv file",
    )
    args = parser.parse_args(opts)

    model_with_accuracies = compare_results(DATA_FOLDER, args.models)
    comparison_fp = args.output_path
    _write_comparison_file(model_with_accuracies, comparison_fp)
    average_fp = "_average.".join(os.path.splitext(comparison_fp))
    _average_results(comparison_fp, average_fp)
    _t_test_results(comparison_fp)


if __name__ == "__main__":
    main(sys.argv[1:])

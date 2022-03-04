"""
This is an entry point, no other file should import from this one.
Analyse the results obtained from the model, with the possibility of generating predictions on new
data, plus obtaining the accuracy of different models on annotated data, and comparing them.
"""
import argparse
import glob
import json
import logging
import math
import os
import sys
from functools import reduce
from itertools import groupby

import matplotlib.pyplot as plt
import mir_eval
import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib.colors import LogNorm
from sklearn.metrics import confusion_matrix

from frog import INPUT_FPC, OUTPUT_FPC
from frog.label_codec import LabelCodec, QUALITIES_MIREX
from frog.load_data import load_tfrecords_dataset
from frog.models.models import load_model_with_info

logger = logging.getLogger(__name__)

# convert spelling to pitch-class:
spelling_to_pitch_class_conversion ={'C-': 'B', 
                                        'G-': 'A#', 
                                        'D-': 'C#',
                                        'A-': 'G#', 
                                        'E-': 'D#',
                                        'B-': 'A#', 
                                        'F': 'F', 
                                        'C': 'C',
                                        'G': 'G', 
                                        'D': 'D', 
                                        'A': 'A', 
                                        'E': 'E', 
                                        'B': 'B', 
                                        'F#': 'F#', 
                                        'C#': 'C#',
                                        'a-': 'g#', 
                                        'e-': 'd#',
                                        'b-': 'a#', 
                                        'f': 'f', 
                                        'c': 'c', 
                                        'g': 'g', 
                                        'd': 'd', 
                                        'a': 'a', 
                                        'e': 'e', 
                                        'b': 'b', 
                                        'f#': 'f#', 
                                        'c#': 'c#', 
                                        'g#': 'g#', 
                                        'd#': 'd#', 
                                        'a#': 'a#'}
## root:
# ['F--', 'C--', 'G--', 'D--', 'A--', 'E--', 'B--', 'F-', 'C-', 'G-', 'D-', 'A-', 'E-', 'B-', 'F', 'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'E#', 'B#', 'F##', 'C##', 'G##', 'D##', 'A##', 'E##', 'B##']

circle_of_fifth_for_pitch = ['F', 'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'f', 'c', 'g', 'd', 'a', 'e', 'b', 'f#', 'c#', 'g#', 'd#', 'a#']
mirex_keys = ['C major', 'C# major', 'D major', 'D# major', 'E major', 'F major', 'F# major', 'G major', 'G# major', 
              'A major', 'A# major', 'B major', 'C minor', 'C# minor', 'D minor', 'D# minor', 'E minor', 'F minor', 
              'F# minor', 'G minor', 'G# minor', 'A minor', 'A# minor', 'B minor']

bass_note_from_inversion = {"0": "", 
                            "1": "3", 
                            "2": "5",
                            "3": "7"}


def _check_predictions(y_true, y_pred, index):
    return np.argmax(y_true[index], axis=-1) == np.argmax(y_pred[index], axis=-1)


def _find_root_from_output(y_pred, lc):
    keys_enc, ton_enc, degree_enc, _quality_enc, _inversion_enc, _root_enc = lc.get_outputs(y_pred)
    root_pred = [lc.find_chord_root_enc(k, t, d) for k, t, d in zip(keys_enc, ton_enc, degree_enc)]
    return np.array(root_pred)


def _concat_numpy_outputs(x, y):
    return [np.concatenate([a, b], axis=0) for a, b in zip(x, y)]


def generate_results(data_folder, model, model_info):
    """
    The generated data is always in the shape:
    y -> [data points] [outputs] (timesteps, output_features)

    :param data_folder:
    :param model:
    :param model_info:
    :return: ys_true, ys_pred, (file_names, start_frames, piano_rolls)
    """
    data_path = os.path.join(data_folder, model_info["trained on"], "test.tfrecords")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Can't find dataset {data_path}")
    with open(os.path.join(data_folder, model_info["trained on"], "info.json"), "r") as f:
        data_info = json.load(f)
    data = load_tfrecords_dataset(
        data_path,
        data_info["compression"],
        batch_size=16,
        shuffle_buffer=1,
        input_type=model_info["input type"],
        output_mode=model_info["output mode"],
    )

    ys_true, info = read_y_true(data)
    ys_pred = predict_and_depad(data, model, info["timesteps"])
    return ys_true, ys_pred, info


def get_structure_info(data_folder, model, model_info):
    """
    The generated data is always in the shape:
    y -> [data points] [outputs] (timesteps, output_features)

    :param data_folder:
    :param model:
    :param model_info:
    :return: ys_true, ys_pred, (file_names, start_frames, piano_rolls)
    """
    data_path = os.path.join(data_folder, model_info["trained on"], "test.tfrecords")

    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Can't find dataset {data_path}")
    data = load_tfrecords_dataset(
        data_path, batch_size=16, shuffle_buffer=1, input_type=model_info["input type"]
    )

    # data_list = [d for d in data]
    # data_list[0][0][0] : size (16, 640, 70)  16: batch size, 640: time steps, 70: number of pitches 
    # structure info: data_list[0][0][1]
    
    ys_true, info = read_y_true(data)
    ys_pred = predict_and_depad(data, model, info["timesteps"])
    return ys_true, ys_pred, info


def read_y_true(data):
    """
    Collect ground truth data. The shape is [N][NF](T, D), where:
        N = total number of data points
        NF = number of features (e.g., key, degree, etc)
        T = timesteps
        D = depth AKA number of classes, it is different for each feature
    @param data: The tfrecords dataset
    @return: ys_true, with the shape described above; and a dictionary containing timesteps
     and piano_rolls.
    """
    ys_true, timesteps, piano_rolls = [], [], []
    for data_point in data.unbatch().as_numpy_iterator():
        x, y, _meta = data_point
        timesteps.append(np.sum(y[0], dtype=int))  # every label has a single 1 per timestep
        piano_rolls.append(x[0][: math.ceil(timesteps[-1] * INPUT_FPC / OUTPUT_FPC)])
        ys_true.append([label[: timesteps[-1], :] for label in y])
    return ys_true, {"timesteps": timesteps, "piano_rolls": piano_rolls}


def predict_and_depad(data, model, timesteps):
    """
    Collect predicted data. The shape is [N][NF](T, D), where:
        N = total number of data points
        NF = number of features (e.g., key, degree, etc)
        T = timesteps
        D = depth AKA number of classes, it is different for each feature
    @param data: The tfrecords dataset
    @param model: The model that we use to predict
    @param timesteps: This is a bit tricky: we should maybe read it from the data we provide
     directly, but there is
    @return: ys_true, with the shape described above; and a dictionary containing timesteps
     and piano_rolls.
    """
    raw_outputs = [[v.numpy() for v in model.sample(x + y, mode="argmax")] for x, y, _ in data]
    # reshape from raw_outputs [NB][NF] (Bj, T, D) to predictions [NF] (N, T, D)
    predictions = [
        np.concatenate([x[j] for x in raw_outputs], axis=0) for j in range(len(raw_outputs[0]))
    ]
    # Throw away the padding and reshape to [N][NF] (T, D)
    ys_pred = [[d[n, :t] for d in predictions] for n, t in enumerate(timesteps)]
    return ys_pred


def chen_su_symbols(y_true, y_pred, lc):
    equivalence_classes = [
        ("M", "M7", "D7", "+6", "Gr+6", "It+6", "Fr+6"),
        ("m", "m7"),
        ("a", "d", "d7", "h7"),
    ]
    quality_map = {qlt: n for n, ec in enumerate(equivalence_classes) for qlt in ec}

    def maj_min_other(qualities):
        return np.array([quality_map[lc.qualities[qe]] for qe in np.argmax(qualities, axis=-1)])

    q_true = maj_min_other(y_true[-3])
    q_pred = maj_min_other(y_pred[-3])
    root_true = np.argmax(y_true[-1], axis=-1)
    root_pred = np.argmax(y_pred[-1], axis=-1)
    # root_pred = _find_root_from_output(y_pred, pitch_spelling=ps)
    result = np.where(q_true == q_pred, np.logical_or(q_true == 2, root_true == root_pred), False)
    return result


def analyse_results_concatenated(yc_true, yc_pred, spelling, mode, verbose=True):
    """Analyse the concatenated outputs from the model, shape [n_labels], (N, classes)"""
    lc = LabelCodec(spelling=spelling == "spelling", mode=mode, strict=False)
    root_der = _find_root_from_output(yc_pred, lc)
    d7_msk = np.argmax(yc_true[-3], axis=-1) == lc.Q2I["d7"]

    success = np.array(
        [_check_predictions(yc_true, yc_pred, j) for j in range(len(lc.output_features))]
    )
    # the features are (key, tonicisation, degree, quality, inversion, root)
    accuracies = {
        "roman numeral w/ inversion": 100 * np.average(np.prod(success[:-1], axis=0), axis=-1),
        "roman numeral w/o key (chen and su)": 100 * np.average(np.prod(success[1:-1], axis=0), axis=-1),   ##  correct when: tonacisation, degree, quality and inversion are all correct. We exclude the key
        "roman numeral w/o inversion": 100 * np.average(np.prod(success[:-2], axis=0), axis=-1),
        "chord symbol chen and su": 100 * np.average(chen_su_symbols(yc_true, yc_pred, lc)),
        "chord symbol w/ inversion": 100 * np.average(np.prod(success[-3:], axis=0), axis=-1),
        "chord symbol w/o inversion": 100 * np.average(success[-3] * success[-1], axis=-1),
        "quality": 100 * np.average(success[-3], axis=-1),
        "inversion": 100 * np.average(success[-2], axis=-1),
        "root": 100 * np.average(success[-1], axis=-1),
        "root coherence": 100 * np.average(root_der == np.argmax(yc_pred[-1], axis=-1)),
        "diminished seventh": 100 * np.average(np.prod(success[:-3], axis=0)[d7_msk], axis=-1),
    }
    if lc.mode == "legacy":
        t_msk = np.argmax(yc_true[2], axis=-1) != 0  # True when the chord is tonicised
        accuracies_other = {
            "key": 100 * np.average(success[0], axis=-1),
            "tonicisation": 100 * np.average(success[1], axis=-1),
            "degree": 100 * np.average(success[2], axis=-1),
            "degree w/ tonicisation": 100 * np.average(np.prod(success[1:3], axis=0), axis=-1),
            "degree of tonicised": 100 * np.average(np.prod(success[1:3], axis=0)[t_msk], axis=-1),
        }
    else:
        t_msk = np.logical_or(
            np.argmax(yc_true[2], axis=-1) != 0,
            np.argmax(yc_true[3], axis=-1) != 0,
        )
        accuracies_other = {
            "key": 100 * np.average(np.prod(success[:2], axis=0), axis=-1),
            "tonicisation": 100 * np.average(np.prod(success[2:4], axis=0), axis=-1),
            "degree": 100 * np.average(np.prod(success[4:6], axis=0), axis=-1),
            "degree w/ tonicisation": 100 * np.average(np.prod(success[2:6], axis=0), axis=-1),
            "degree of tonicised": 100 * np.average(np.prod(success[2:6], axis=0)[t_msk], axis=-1),
        }
    accuracies = {**accuracies, **accuracies_other}

    if verbose:
        size = max(len(k) for k in accuracies) + 2
        print(f"Accuracy:")
        for k, v in accuracies.items():
            print(f"{k:{size}}: {v:2.2f} %")

    ### add mir_eval.key
    all_mirex_keys_metrics = []
    for k_gt, k_pred in list(zip(yc_true[0], yc_pred[0])):
        all_mirex_keys_metrics.append(mir_eval.key.weighted_score(mirex_keys[np.argmax(k_gt)], mirex_keys[np.argmax(k_pred)]))
    print ("MIREX_KEY:", np.mean(all_mirex_keys_metrics), np.std(all_mirex_keys_metrics))
    return accuracies


def analyse_results(ys_true, ys_pred, spelling, mode, verbose=True):
    """
    Given the true and predicted labels, calculate several metrics on them.
    The features are key, deg1, deg2, qlt, inv, root

    :param ys_true: shape [n_data][n_labels](ts, classes)
    :param ys_pred: shape [n_data][n_labels](ts, classes)
    :param spelling: either "pitch" or "spelling"
    :param mode: either "legacy" or "experimental"
    :param verbose: if True, print accuracies
    :return: a dictionary with structure {feature: accuracy}
    """
    # TODO: This reduce is a bit slow. Maybe find a faster way to do it?
    yc_true = reduce(_concat_numpy_outputs, ys_true)  # shape [NF](N, Fi)
    yc_pred = reduce(_concat_numpy_outputs, ys_pred)  # shape [NF](N, Fi)
    return analyse_results_concatenated(yc_true, yc_pred, spelling, mode, verbose)



def analyse_results_simple(ys_true, ys_pred, verbose=True):
    """
    Given the true and predicted labels, calculate several metrics on them.
    The features are key, deg1, deg2, qlt, inv, root  (["key", "tonicisation", "degree", "quality", "inversion", "root"])

    :param ys_true: shape [n_data][n_labels](ts, classes)
    :param ys_pred: shape [n_data][n_labels](ts, classes)
    :param verbose: if True, print accuracies
    :return: ys_true and ys_pred through reduce
    """
    # TODO: This reduce is a bit slow. Maybe find a faster way to do it?
    yc_true = reduce(_concat_numpy_outputs, ys_true)  # shape [NF](N, Fi)
    yc_pred = reduce(_concat_numpy_outputs, ys_pred)  # shape [NF](N, Fi)
    return yc_true, yc_pred


def do_confusion_matrix(true, pred, labels, plot_name, output_folder):
    
    xticks = labels
    yticks = labels

    FONTSIZE = 45

    ys_true_conc_argmax = [int(np.argmax(ys)) for ys in true]
    ys_pred_conc_argmax = [int(np.argmax(ys)) for ys in pred]


    # Convert the sequence of labels (and the coresponding lines in matrix) to circle of fifth. This is when spelling=pitch. 
    # circle_of_fifth_for_pitch = ['F', 'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'f', 'c', 'g', 'd', 'a', 'e', 'b', 'f#', 'c#', 'g#', 'd#', 'a#']  
    
    if labels == ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']:
        print ('Does the re-ordering')
        ys_true_conc_argmax = [circle_of_fifth_for_pitch.index(labels[old_l]) for old_l in ys_true_conc_argmax]    
        ys_pred_conc_argmax = [circle_of_fifth_for_pitch.index(labels[old_l]) for old_l in ys_pred_conc_argmax]    
                
        xticks = circle_of_fifth_for_pitch
        yticks = circle_of_fifth_for_pitch
        sns.set(font_scale=3.7)
    else:
        sns.set(font_scale=4)

    ## plot confusion matrix with no normalisation
    conf_matrix_row_no_norm = confusion_matrix(ys_true_conc_argmax, ys_pred_conc_argmax, normalize=None, labels=np.arange(len(labels)))    
    plt.figure(figsize=(26, 24)) 
    
    conf_matrix_plot = sns.heatmap(conf_matrix_row_no_norm, yticklabels=yticks, xticklabels=xticks, cmap=sns.color_palette("Blues"), cbar=False, norm=LogNorm(), linewidths=.5, annot=False)
    # conf_matrix_plot.axes.get_yaxis().set_visible(False)

    plt.xlabel('Prediction', fontsize=58, labelpad=10)
    plt.ylabel('Ground-truth', fontsize=58, labelpad=10)
    plt.tight_layout()    
    plt.savefig(os.path.join(output_folder, 'confusion_matrix_' + plot_name + os.path.basename(output_folder).split(".")[0] + '.png'))    
    plt.savefig(os.path.join(output_folder, 'confusion_matrix_' + plot_name + os.path.basename(output_folder).split(".")[0] + '.eps'))   
    plt.savefig(os.path.join(output_folder, 'confusion_matrix_' + plot_name + os.path.basename(output_folder).split(".")[0] +'.pdf'))     
    print ('File saved:', os.path.join(output_folder, 'confusion_matrix_' + plot_name + os.path.basename(output_folder).split(".")[0] + '.pdf'))

    return


def make_plots(ys_true, ys_pred, spelling, verbose, output_folder):
    ys_true_conc, ys_pred_conc = analyse_results_simple(ys_true, ys_pred, verbose=True)

    lc = LabelCodec(spelling=spelling == "spelling", strict=False, mode='legacy')
    print ('FEATURES: ', lc.output_features)

    codec_call = [lc.keys, lc.degrees, lc.degrees, lc.qualities, lc.inversions, lc.root]  ### TODO: get correct codec for tonicisation
    print (lc.keys)
    for idx, feat in enumerate(lc.output_features):
        do_confusion_matrix(ys_true_conc[idx], ys_pred_conc[idx], codec_call[idx], feat, output_folder)
    
    print ('Plots done!')
    return


def create_mir_eval_lab_files(ys_true, ys_pred, spelling, verbose, output_folder):
    
    def create_annotations_file_content(labels):
        start_times = []
        end_times = []
        final_labels_sequence = []

        unique_labels = list(set(labels))
        labels_map = [unique_labels.index(l) for l in labels]
        labels_grouped = [list(j) for i, j in groupby(labels_map)]  
        st = 1
        for group in labels_grouped:
            start_times.append(st)
            end_times.append(st + len(group))
            final_labels_sequence.append(unique_labels[group[0]])
            st += len(group)

        return start_times, end_times, final_labels_sequence

    def write_to_file(start_times, end_times, labels, output_file):
        data = np.array([np.array(start_times), np.array(end_times), np.array(labels)])
        data = data.T
        with open(output_file, 'w+') as datafile:
            np.savetxt(datafile, data, delimiter="\t", fmt='%s')

    def create_annotations_files(data, tag, output_folder):
        # the famous .lab files that are used in MIREX task

        # Create folder to store the .lab files 
        annotations_folder = os.path.join(output_folder, 'annotations_' + tag)
        if not os.path.exists(annotations_folder):
            os.mkdir(annotations_folder)

        lc = LabelCodec(spelling=spelling == "spelling", strict=False)
        # Gb to G- or D7 to 7

        ### make groundtruth .lab files
        # feature set: ["key", "tonicisation", "degree", "quality", "inversion", "root"]

        data_to_convert = []
        root_one_hot_lists = data[5]
        quality_one_hot_lists = data[3]
        inversion_one_hot_lists = data[4]

        for root, quality, inversion in zip(root_one_hot_lists, quality_one_hot_lists, inversion_one_hot_lists):
            if np.argmax(inversion) != 0:
                if bass_note_from_inversion[lc.decode_inversion(np.argmax(inversion))] == '3':
                    if QUALITIES_MIREX[np.argmax(quality)] in ['min', 'dim', 'min7', 'dim7', 'hdim7']:
                        bass_part = 'b' + bass_note_from_inversion[lc.decode_inversion(np.argmax(inversion))]
                    else:
                        bass_part = bass_note_from_inversion[lc.decode_inversion(np.argmax(inversion))]
                
                elif bass_note_from_inversion[lc.decode_inversion(np.argmax(inversion))] == '5':
                    if QUALITIES_MIREX[np.argmax(quality)] in ['dim', 'dim7', 'hdim7']:
                        bass_part = 'b' + bass_note_from_inversion[lc.decode_inversion(np.argmax(inversion))]
                    elif QUALITIES_MIREX[np.argmax(quality)] == 'aug':
                        bass_part = '#' + bass_note_from_inversion[lc.decode_inversion(np.argmax(inversion))]
                    else:
                        bass_part = bass_note_from_inversion[lc.decode_inversion(np.argmax(inversion))]
                

                elif bass_note_from_inversion[lc.decode_inversion(np.argmax(inversion))] == '7':
                    if QUALITIES_MIREX[np.argmax(quality)] in ['min7', '7', 'hdim7']:
                        bass_part = 'b' + bass_note_from_inversion[lc.decode_inversion(np.argmax(inversion))]
                    elif QUALITIES_MIREX[np.argmax(quality)] == 'dim7':
                        bass_part = 'bb' + bass_note_from_inversion[lc.decode_inversion(np.argmax(inversion))]
                    else:
                        bass_part = bass_note_from_inversion[lc.decode_inversion(np.argmax(inversion))]

                label = lc.root[np.argmax(root)].replace("-", "b") + ':' + QUALITIES_MIREX[np.argmax(quality)] + "/" + bass_part
            
            
            else:
                label = lc.root[np.argmax(root)].replace("-", "b") + ':' + QUALITIES_MIREX[np.argmax(quality)]
            data_to_convert.append(label)

        start_times, end_times, labels = create_annotations_file_content(data_to_convert)
        out_filename = os.path.join(annotations_folder, '_all.lab')
        write_to_file(start_times, end_times, labels, out_filename)

    # read concatenated data: shape: [feature, data point, feature_length]
    ys_true_conc, ys_pred_conc = analyse_results_simple(ys_true, ys_pred, verbose=True)

    # make prediction .lab files
    create_annotations_files(ys_true_conc, 'true', output_folder)
    print ('Ground-truth .lab files created!')
    create_annotations_files(ys_pred_conc, 'predicted', output_folder)
    print ('Predictions .lab files created!')
    return


def compute_mir_eval_metrics(folder_path_true, folder_path_predicted, output_folder):
    # true_lab_files = glob.glob(os.path.join(folder_path_true, '*.lab'))
    true_lab_files = glob.glob(os.path.join(folder_path_true, '_all.lab'))

    scores_root = []
    scores_majmin = []
    scores_majmin_inv = []
    scores_mirex = []
    scores_thirds = []
    scores_thirds_inv = []
    scores_triads = []
    scores_triads_inv = []
    scores_tetrads = []
    scores_tetrads_inv = []
    scores_sevenths = []
    scores_sevenths_inv = []
    scores_overseg = []
    scores_underseg = []
    scores_seg = []

    for true_lab_file in true_lab_files:
        print (true_lab_file)
        ground_truth_lab = true_lab_file
        prediction_lab = os.path.join(folder_path_predicted, os.path.basename(true_lab_file))
        print (prediction_lab)
        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(ground_truth_lab)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(prediction_lab)

        est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, 
                                                                    est_labels, 
                                                                    ref_intervals.min(), 
                                                                    ref_intervals.max(), 
                                                                    mir_eval.chord.NO_CHORD,
                                                                    mir_eval.chord.NO_CHORD)

        (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels, est_intervals, est_labels)

        durations = mir_eval.util.intervals_to_durations(intervals)

        comparisons_root = mir_eval.chord.root(ref_labels, est_labels)
        comparisons_majmin = mir_eval.chord.majmin(ref_labels, est_labels)
        comparisons_majmin_inv = mir_eval.chord.majmin_inv(ref_labels, est_labels)
        comparisons_mirex = mir_eval.chord.mirex(ref_labels, est_labels)
        comparisons_thirds = mir_eval.chord.thirds(ref_labels, est_labels)
        comparisons_thirds_inv = mir_eval.chord.thirds_inv(ref_labels, est_labels)
        comparisons_triads = mir_eval.chord.triads(ref_labels, est_labels)
        comparisons_triads_inv = mir_eval.chord.triads_inv(ref_labels, est_labels)
        comparisons_tetrads = mir_eval.chord.tetrads(ref_labels, est_labels)
        comparisons_tetrads_inv = mir_eval.chord.tetrads_inv(ref_labels, est_labels)
        comparisons_sevenths = mir_eval.chord.sevenths(ref_labels, est_labels)
        comparisons_sevenths_inv = mir_eval.chord.sevenths_inv(ref_labels, est_labels)
        comparisons_overseg = mir_eval.chord.overseg(ref_intervals, est_intervals)
        comparisons_underseg = mir_eval.chord.underseg(ref_intervals, est_intervals)
        comparisons_seg = mir_eval.chord.seg(ref_intervals, est_intervals)

        score_root = mir_eval.chord.weighted_accuracy(comparisons_root, durations)
        score_majmin = mir_eval.chord.weighted_accuracy(comparisons_majmin, durations)     
        score_majmin_inv = mir_eval.chord.weighted_accuracy(comparisons_majmin_inv, durations)    
        score_mirex = mir_eval.chord.weighted_accuracy(comparisons_mirex, durations)    
        score_thirds = mir_eval.chord.weighted_accuracy(comparisons_thirds, durations)    
        score_thirds_inv = mir_eval.chord.weighted_accuracy(comparisons_thirds_inv, durations)    
        score_triads = mir_eval.chord.weighted_accuracy(comparisons_triads, durations)    
        score_triads_inv = mir_eval.chord.weighted_accuracy(comparisons_triads_inv, durations)    
        score_tetrads = mir_eval.chord.weighted_accuracy(comparisons_tetrads, durations)    
        score_tetrads_inv = mir_eval.chord.weighted_accuracy(comparisons_tetrads_inv, durations)    
        score_sevenths = mir_eval.chord.weighted_accuracy(comparisons_sevenths, durations)    
        score_sevenths_inv = mir_eval.chord.weighted_accuracy(comparisons_sevenths_inv, durations)    
        score_overseg = mir_eval.chord.weighted_accuracy(comparisons_overseg, durations)    
        score_underseg = mir_eval.chord.weighted_accuracy(comparisons_underseg, durations)    
        score_seg = mir_eval.chord.weighted_accuracy(omparisons_seg, durations)    
        
        scores_root.append(score_root)
        scores_majmin.append(score_majmin)
        scores_majmin_inv.append(score_majmin_inv)
        scores_mirex.append(score_mirex)
        scores_thirds.append(score_thirds)
        scores_thirds_inv.append(score_thirds_inv)
        scores_triads.append(score_triads)
        scores_triads_inv.append(score_triads_inv)
        scores_tetrads.append(score_tetrads)
        scores_tetrads_inv.append(score_tetrads_inv)
        scores_sevenths.append(score_sevenths)
        scores_sevenths_inv.append(score_sevenths_inv)
        scores_overseg.append(comparisons_overseg)
        scores_underseg.append(comparisons_underseg)
        scores_seg.append(comparisons_seg)

    average_metrics = {"root": [np.mean(scores_root), np.std(scores_root)],
                       "majmin": [np.mean(scores_majmin), np.std(scores_majmin)],
                       "majmin_inv": [np.mean(scores_majmin_inv), np.std(scores_majmin_inv)],
                       "mirex": [np.mean(scores_mirex), np.std(scores_mirex)],
                       "thirds": [np.mean(scores_thirds), np.std(scores_thirds)],
                       "thirds_inv": [np.mean(scores_thirds_inv), np.std(scores_thirds_inv)],
                       "triads": [np.mean(scores_triads), np.std(scores_triads)],
                       "triads_inv": [np.mean(scores_triads_inv), np.std(scores_triads_inv)],
                       "tetrads": [np.mean(scores_tetrads), np.std(scores_tetrads)],
                       "tetrads_inv": [np.mean(scores_tetrads_inv), np.std(scores_tetrads_inv)],                 
                       "sevenths": [np.mean(scores_sevenths), np.std(scores_sevenths)],
                       "sevenths_inv": [np.mean(scores_sevenths_inv), np.std(scores_sevenths_inv)],
                       "overseg": [np.mean(scores_overseg), np.std(scores_overseg)],
                       "underseg": [np.mean(scores_underseg), np.std(scores_underseg)],
                       "seg": [np.mean(scores_seg), np.std(scores_seg)]}

    print ('MIREX metrics (mean, std)')
    print (average_metrics)

    with open(os.path.join(output_folder, "MIREX_metrics.json"), "w") as f:
        json.dump(average_metrics, f, indent=2)

    return

def main(opts):
    parser = argparse.ArgumentParser(description="Analyse the results of a single model")
    parser.add_argument("data_path", help="Data root. Its subfolders must contain test.tfrecords")
    parser.add_argument("-p", dest="model_path", help="path to the model folder")
    parser.add_argument("-o", dest="output_folder",help="Path to the output folder")
    parser.add_argument("-v", action="store_true", help="activate verbose mode")
    parser.add_argument("-e", dest="eager", action="store_true", help="Execute eagerly")
    args = parser.parse_args(opts)

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    tf.config.run_functions_eagerly(args.eager)
    model, model_info = load_model_with_info(args.model_path, verbose=args.v)
    ys_true, ys_pred, _ = generate_results(args.data_path, model, model_info)
    spelling, _octave = model_info["input type"].split("_")
    print(f"Analysing model {model_info['model name']}")
    ## Get accuracies
    acc = analyse_results(ys_true, ys_pred, spelling, model_info["output mode"])

    ## Get visualisations
    make_plots(ys_true, ys_pred, spelling, args.v, args.output_folder)

    ## Create metrics from mir-eval library
    create_mir_eval_lab_files(ys_true, ys_pred, spelling, args.v, args.output_folder)

    ## Get mir-eval metrics
    compute_mir_eval_metrics(os.path.join(args.output_folder, "annotations_true"), os.path.join(args.output_folder, "annotations_predicted"), args.output_folder)

    with open(os.path.join(args.output_folder, "accuracy.json"), "w") as f:
        json.dump(acc, f, indent=2)
    return acc


if __name__ == "__main__":
    main(sys.argv[1:])

"""
This is an entry point, no other file should import from this one.
Augment and convert the data from .mxl plus .csv to .tfrecords for training the system.
This creates a tfrecord containing the following features:
'name': the name of the file
'transposition': the transposition index (0 = original key)
'start': the number of crotchets from the beginning of the piece
'piano_roll': a representation of the score, flattened to 1D from 2D-shape (n_frames, features)
'key': the local key of the music
'tonicisation': the temporary key from which the chord borrows (0 if no borrowing)
'degree': the chord degree with respect to the tonicised key
'quality': e.g. m, M, D7 for minor, major, dominant 7th etc.
'inversion': from 0 to 3 depending on what note is at the bass
'root': the root of the chord
"""
import json
import logging
import os
from argparse import ArgumentParser
from datetime import datetime
from math import inf

import tensorflow as tf

from frog import CHUNK_SIZE, DATA_FOLDER, HOP_SIZE, INPUT_FPC, INPUT_TYPES, OUTPUT_FPC
from frog.label_codec import LabelCodec, OUTPUT_MODES
from frog.preprocessing.preprocess_chords import (
    calculate_lr_transpositions_key,
    generate_chord_chunks,
    import_chords,
    transpose_chord_labels,
)
from frog.preprocessing.preprocess_scores import (
    calculate_lr_transpositions_pitches,
    generate_input_chunks,
    get_metrical_information,
    import_piano_roll,
    transpose_piano_roll,
)
from frog.preprocessing.train_valid_test_split import train_valid_test_split

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _make_tfr_feature(piano_roll, structure, chords, label_codec, name, s, start,
                      beat_strength):
    feature = {
        "name": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[name.encode("utf-8")])),
        "transposition": tf.train.Feature(int64_list=tf.train.Int64List(value=[s])),
        "start": tf.train.Feature(int64_list=tf.train.Int64List(value=[start])),
        "piano_roll": tf.train.Feature(float_list=tf.train.FloatList(value=piano_roll.reshape(-1))),
    }
    for n, f in enumerate(label_codec.output_features):
        feature[f] = tf.train.Feature(int64_list=tf.train.Int64List(value=[c[n] for c in chords]))
    if beat_strength:
        feature["structure"] = tf.train.Feature(
            float_list=tf.train.FloatList(value=structure.reshape(-1))
        )

    return feature


def _preprocess_one_piece(
    score_file, chords_file, label_codec, spelling, octaves, beat_strength, train
):
    hop_size = HOP_SIZE if train else CHUNK_SIZE
    file_name = os.path.splitext(os.path.basename(score_file))[0]
    piano_roll = import_piano_roll(score_file, spelling, octaves, INPUT_FPC)
    pr_chunks = generate_input_chunks(piano_roll, CHUNK_SIZE, hop_size, INPUT_FPC)

    structure = get_metrical_information(score_file, INPUT_FPC)
    st_chunks = generate_input_chunks(structure, CHUNK_SIZE, hop_size, INPUT_FPC)

    # Pre-process the chords
    chords = import_chords(chords_file, label_codec, OUTPUT_FPC)
    chord_chunks = generate_chord_chunks(chords, CHUNK_SIZE, hop_size, OUTPUT_FPC)

    if len(piano_roll) != len(structure):
        logger.warning(
            f"The piano roll has {len(piano_roll)} frames but the structure {len(structure)}!"
        )
    if len(chords) / OUTPUT_FPC != len(piano_roll) / INPUT_FPC:
        logger.warning(
            f"The piano roll has {len(piano_roll)} frames but the chords {len(chords)},"
            f" which are equivalent to {len(chords) * INPUT_FPC / OUTPUT_FPC}!"
        )

    features = []
    skip = False
    for i, (pr, st, chords) in enumerate(zip(pr_chunks, st_chunks, chord_chunks)):
        for s in _find_available_transpositions(pr, chords, spelling, transpose=train):
            pr_transposed = transpose_piano_roll(pr, s, spelling, octaves)
            pitch_proximity = "fifth" if spelling == "spelling" else "semitone"
            chords_transposed = transpose_chord_labels(chords, s, pitch_proximity)
            enc_chords = label_codec.encode_chords(chords_transposed)
            for ec, tc in zip(enc_chords, chords_transposed):
                if any([x is None for x in ec]):
                    logger.warning(f"Couldn't encode properly chord {tc} -> {ec}")
                    skip = True
                    continue
            if skip:
                logger.warning(f"chunk skipped, transposition {s}")
                skip = False
            else:
                temp = _make_tfr_feature(
                    pr_transposed,
                    st,
                    enc_chords,
                    label_codec,
                    file_name,
                    s,
                    i * hop_size,
                    beat_strength,
                )
                features.append(temp)
    return features


def _find_available_transpositions(piano_roll, chords, spelling, transpose):
    nl_pitches, nr_pitches = calculate_lr_transpositions_pitches(piano_roll, spelling)
    nl_keys, nr_keys = calculate_lr_transpositions_key(chords, spelling)
    nl, nr = min(nl_keys, nl_pitches), min(nr_keys, nr_pitches)
    transpositions = list(range(-nl, nr + 1))
    if not transpose:
        # it would be tempting to choose simply 0, 0 but some pieces can not be encoded
        #  in their original key, so we choose the closest one to the original
        transpositions = [min(abs(x) for x in transpositions)]
    return transpositions


def _create_tfrecords(tfr_base_folder, input_type, output_mode, beat_strength, n_max, compression):
    # Preparation
    spelling, octaves = input_type.split("_")
    logger.info(f"You are generating tfrecords for the {input_type} mode.")
    in_folder = os.path.join(DATA_FOLDER, "datasets")
    tfr_folder = "_".join(["tfrecords", input_type, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")])
    out_folder = os.path.join(tfr_base_folder, tfr_folder)
    datasets = ["train", "valid", "test"]
    [
        os.makedirs(os.path.join(out_folder, ds, t))
        for ds in datasets
        for t in ["chords", "scores", "txt"]
    ]
    lc = LabelCodec(spelling=spelling == "spelling", mode=output_mode, strict=False)

    # Split the data into train/valid/test
    for sub_folder in os.listdir(in_folder):
        if sub_folder == "Beethoven_4tets":
            continue  # We are not legally allowed to use this dataset, for the moment
        if sub_folder.__contains__("Tavern"):
            continue  # Slightly different structure, not compatible yet
        if sub_folder == "BPS_other-movements":
            continue  # Data not uploaded
        # elif sub_folder == "BPS":
        #     split_bps_like_chen_su(os.path.join(in_folder, sub_folder), out_folder)
        # elif sub_folder == "Bach_WTC_1_Preludes":
        #     split_bach_like_chen_su(os.path.join(in_folder, sub_folder), out_folder)
        else:  # We want to have only Bach Preludes and BPS in our test set
            train_valid_test_split(
                os.path.join(in_folder, sub_folder), out_folder, seed=18, split=(0.80, 0.20, 0.0)
            )

    files_per_dataset = {}
    for ds in datasets:
        ds_folder = os.path.join(out_folder, ds)
        chords_folder = os.path.join(ds_folder, "chords")
        scores_folder = os.path.join(ds_folder, "scores")
        file_names = [
            ".".join(fn.split(".")[:-1])
            for fn in os.listdir(chords_folder)
            if not fn.startswith(".")
        ]

        output_file = os.path.join(out_folder, f"{ds}.tfrecords")
        logger.info(f"Working on {os.path.basename(output_file)}.")
        with tf.io.TFRecordWriter(output_file, options=compression) as writer:
            fpd = 0
            for n, fn in enumerate(file_names):
                if n >= n_max:
                    break
                sf = os.path.join(scores_folder, fn + ".mxl")
                cf = os.path.join(chords_folder, fn + ".csv")

                logger.info(f"Analysing {fn}")
                features = _preprocess_one_piece(
                    sf, cf, lc, spelling, octaves, beat_strength, ds == "train"
                )
                for feature in features:
                    writer.write(
                        tf.train.Example(
                            features=tf.train.Features(feature=feature)
                        ).SerializeToString()
                    )
                fpd += 1
            files_per_dataset[ds] = fpd
    info = {
        "input_type": input_type,
        "output_mode": output_mode,
        "output labels and size": lc.output_size,
        "beat strength": beat_strength,
        "files per dataset": files_per_dataset,
        "compression": compression,
    }
    with open(os.path.join(out_folder, "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(out_folder, "SUCCESS"), "w") as f:
        f.write("")
    return


def main(opts):
    parser = ArgumentParser(description="Preprocess data and transform it into tfrecords")
    parser.add_argument("input_types", nargs="+", choices=INPUT_TYPES, help="Input type(s) to use")
    parser.add_argument("-o", dest="output_mode", choices=OUTPUT_MODES, help="Output mode")
    parser.add_argument("-n", type=int, default=inf, help="Num of scores to take, defaults to all")
    parser.add_argument("-f", dest="folder", default="../data", help="Where to store the tfrecords")
    parser.add_argument("-no_bs", action="store_true", help="Deactivate the beat strength")
    parser.add_argument("-c", default="ZLIB", help="Type of compression to use")
    args = parser.parse_args(opts)
    for it in args.input_types:
        _create_tfrecords(args.folder, it, args.output_mode, not args.no_bs, args.n, args.c)

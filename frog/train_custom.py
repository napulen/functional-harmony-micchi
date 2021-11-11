"""
This is an entry point, no other file should import from this one.
Train a model to return a Roman Numeral analysis given a score in musicxml format.
"""
import json
import logging
import os
import random
import sys
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import tensorflow as tf
from tqdm import tqdm

tf.random.set_seed(18)
np.random.seed(18)
random.seed(18)

from frog import INPUT_FEATURES, MODEL_TYPES
from frog.analysis.analyse_results import read_y_true, predict_and_depad, analyse_results
from frog.label_codec import LabelCodec
from frog.load_data import load_tfrecords_dataset
from frog.models.models import create_model, hyper_params, hyper_algomus

patience = 3

logger = logging.getLogger(__name__)


def get_shapes(dataset, lc):
    input_specs, output_specs, _meta_specs = dataset.element_spec
    # The [1:] is to remove the batch dimension
    input_shapes = {k: s.shape[1:] for k, s in zip(INPUT_FEATURES, input_specs)}
    output_shapes = {k: s.shape[1:] for k, s in zip(lc.output_features, output_specs)}
    return input_shapes, output_shapes


def _accuracy_msg(lc, acc):
    return ", ".join(f"{f} accuracy: {ea.result():.1%}" for f, ea in zip(lc.output_features, acc))


def train(
    tfrecords_folder,
    model_base_folder,
    model_type,
    hyper_parameters,
    num_epochs,
    batch_size,
    structure,
    verbose=False,
):
    # remove trailing sep that would change the value of os.path.basename
    while tfrecords_folder.endswith(os.path.sep):
        tfrecords_folder = tfrecords_folder[:-1]
    with open(os.path.join(tfrecords_folder, "info.json"), "r") as f:
        data_info = json.load(f)
    input_type = data_info["input_type"]
    output_mode = data_info["output_mode"]
    train_path, valid_path, test_path = [
        os.path.join(tfrecords_folder, f"train.tfrecords"),
        os.path.join(tfrecords_folder, f"valid.tfrecords"),
        os.path.join(tfrecords_folder, f"test.tfrecords"),
    ]
    train_data = load_tfrecords_dataset(
        train_path, data_info["compression"], batch_size, 1_600, input_type, output_mode
    )
    valid_data = load_tfrecords_dataset(
        valid_path, data_info["compression"], batch_size, 1, input_type, output_mode
    )
    test_data = load_tfrecords_dataset(
        test_path, data_info["compression"], batch_size, 1, input_type, output_mode
    )
    spelling, octave = input_type.split("_")
    lc = LabelCodec(spelling=spelling == "spelling", mode=output_mode)
    input_shapes, output_shapes = get_shapes(train_data, lc)

    # Model i/o
    model_name = "_".join([model_type, input_type, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")])
    model_folder = os.path.join(model_base_folder, model_name)
    os.makedirs(model_folder)
    model = create_model(
        model_type, input_shapes, output_shapes, lc.output_features, hyper_parameters, structure
    )
    model.summary()
    logger.info(f"input type: {input_type}, output mode: {output_mode}, structure: {structure}")
    info = {
        "trained on": os.path.basename(tfrecords_folder),
        "input type": input_type,
        "output mode": output_mode,
        "model name": model_name,
        "model type": model_type,
        "use music structure info": structure,
        "input shapes": {k: [d for d in v] for k, v in input_shapes.items()},
        "output shapes": {k: [d for d in v] for k, v in output_shapes.items()},
        "hyper parameters": hyper_parameters,
    }
    with open(os.path.join(model_folder, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    train_summary_writer = tf.summary.create_file_writer(os.path.join(model_folder, "train"))
    valid_summary_writer = tf.summary.create_file_writer(os.path.join(model_folder, "valid"))

    train_loss = tf.keras.metrics.Mean()
    valid_loss = tf.keras.metrics.Mean()
    train_accuracies = [tf.keras.metrics.CategoricalAccuracy() for _ in lc.output_features]
    valid_accuracies = [tf.keras.metrics.CategoricalAccuracy() for _ in lc.output_features]

    best_valid_loss = np.inf
    waiting = 0
    for epoch in range(num_epochs):
        train_loss.reset_states()
        valid_loss.reset_states()
        [acc.reset_states() for acc in train_accuracies]
        [acc.reset_states() for acc in valid_accuracies]

        # training epoch
        for x, y, _meta in tqdm(train_data):
            train_loss_value = model.network_learn(x, y)
            train_loss.update_state(train_loss_value)  # Add current batch loss
            # tqdm.write(str(loss_value))
            if verbose:
                samples = model.sample(x + y, training=True)
                [ea.update_state(t, p) for ea, t, p in zip(train_accuracies, y, samples)]
        with train_summary_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), step=epoch)
            if verbose:
                for feature, acc in zip(lc.output_features, train_accuracies):
                    tf.summary.scalar(f"accuracy_{feature}", acc.result(), step=epoch)

        # validation epoch
        for x, y, _meta in valid_data:
            valid_loss_value = model.network_valid(x, y)
            valid_loss.update_state(valid_loss_value)  # Add current batch loss
            if verbose:
                samples = model.sample(x + y, training=True)
                [va.update_state(t, p) for va, t, p in zip(valid_accuracies, y, samples)]
        with valid_summary_writer.as_default():
            tf.summary.scalar("loss", valid_loss.result(), step=epoch)
            if verbose:
                for feature, acc in zip(lc.output_features, valid_accuracies):
                    tf.summary.scalar(f"accuracy_{feature}", acc.result(), step=epoch)

        tap = _accuracy_msg(lc, train_accuracies) if verbose else ""
        vap = _accuracy_msg(lc, valid_accuracies) if verbose else ""

        tqdm.write(
            f"Epoch {epoch+1:03d}:\n"
            f"Train: Loss: {train_loss.result():.3f}, {tap}\n"
            f"Valid: Loss: {valid_loss.result():.3f}, {vap}\n"
        )

        # Implement early stopping
        if valid_loss.result() > best_valid_loss:
            waiting += 1
        else:
            best_valid_loss = valid_loss.result()
            model.save_weights(os.path.join(model_folder, "model"))
            waiting = 0
        if waiting > patience:
            break

    with open(os.path.join(model_folder, "SUCCESS"), "w") as f:
        f.write("")

    # FIXME: It doesn't work here sometimes... :/ Why?
    ys_true, info = read_y_true(test_data)
    ys_pred = predict_and_depad(test_data, model, info["timesteps"])
    spelling, _octave = input_type.split("_")
    acc = analyse_results(ys_true, ys_pred, spelling, output_mode)
    with open(os.path.join(model_folder, "accuracy.json"), "w") as f:
        json.dump(acc, f, indent=2)
    return best_valid_loss, acc


def main(opts):
    parser = ArgumentParser(description="Train a neural network for Roman Numeral analysis")
    parser.add_argument("tfrecords_folder", help="The folder where the tfrecords are stored")
    parser.add_argument(
        "-f", dest="model_folder", default="../logs", help="Where to store the saved model"
    )
    parser.add_argument("-m", dest="model_type", choices=MODEL_TYPES, help="Model to use")
    parser.add_argument("-n", dest="num_epochs", default=200, type=int, help="Number of epochs")
    parser.add_argument("-b", dest="batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("-v", dest="verbose", action="store_true", help="Verbose execution (slow!)")
    parser.add_argument("-e", dest="eager", action="store_true", help="Execute eagerly")
    parser.add_argument("--no_struct", action="store_true", help="Remove music structure info")
    args = parser.parse_args(opts)
    tf.config.run_functions_eagerly(args.eager)
    train(
        args.tfrecords_folder,
        args.model_folder,
        args.model_type,
        hyper_algomus if args.model_type == "Algomus" else hyper_params,
        args.num_epochs,
        args.batch_size,
        not args.no_struct,
        args.verbose,
    )


if __name__ == "__main__":
    main(sys.argv[1:])

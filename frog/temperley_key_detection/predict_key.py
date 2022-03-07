"""
This is an entry point, no other file should import from this one.

Do a simple key detection using an algorithm inspired by Temperley's paper What's Key for Key.
"""
import argparse
import glob
import os
import sys

import numpy as np

import logging

from frog.preprocessing.preprocess_scores import import_piano_roll
from frog.temperley_key_detection import TEMPERLEY_STD, TEMPERLEY_WEIGHTS

logger = logging.getLogger(__name__)


def predict_key(piano_roll, weights, weights_std):
    """
    Predict the key using Temperley's algorithm
    :param piano_roll: shape (time, pitches); time should give a context of at least a few crotchets
    :param weights: shape (keys, pitches) == (24, 12), Temperley's pitch profiles for all keys
    :param weights_std: the standard deviation of the weights on the pitches axis,
     passed as a separate argument for performance improvement
    :return: the key that maximizes the internal product between the pitch profile of the
     piano_roll and the typical pitch profiles for each key
    """
    y = np.sum(piano_roll, axis=0, dtype=float)  # shape (12)
    y -= np.mean(y)
    return np.argmax(np.dot(weights, y) / (weights_std * np.std(y)))


def main(opts):
    parser = argparse.ArgumentParser(
        description="Key detection according to Temperley's paper 'What's Key for Key'?"
    )
    parser.add_argument(
        "files",
        action="store",
        nargs="+",
        type=str,
        help="path to the scores; you can use Unix's wildcard * to pass several scores",
    )
    args = parser.parse_args(opts)
    files = [f for x in args.files for f in glob.glob(x)]
    logger.info(f"Selected scores:{files}")
    keys = []
    for f in files:
        piano_roll = import_piano_roll(f, "pitch", "class", 2)
        key = predict_key(piano_roll, TEMPERLEY_WEIGHTS, TEMPERLEY_STD)
        print(os.path.basename(f), key)
        keys.append(key)
    return files, keys


if __name__ == "__main__":
    main(sys.argv[1:])

import os

import pytest

from frog import INPUT_FPC, OUTPUT_FPC
from frog.label_codec import LabelCodec
from frog.preprocessing.create_tfrecords import _find_available_transpositions
from frog.preprocessing.preprocess_chords import import_chords
from frog.preprocessing.preprocess_scores import import_piano_roll

data_folder = "resources"


# TODO: Add a test to verify if scores with a pickup have good alignment between chords and notes


@pytest.mark.parametrize("octaves", ["complete", "class", "bass"])
@pytest.mark.parametrize(
    "transpose, expected", [(True, [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]), (False, [0])]
)
def test_available_transpositions_pitch(octaves, transpose, expected):
    fp = os.path.join(data_folder, "wtc_i_prelude_01")
    score = import_piano_roll(fp + ".mxl", "pitch", octaves, INPUT_FPC)
    chords = import_chords(fp + ".csv", LabelCodec(False), OUTPUT_FPC)
    outcome = _find_available_transpositions(score, chords, "pitch", transpose)
    assert outcome == expected


@pytest.mark.parametrize("octaves", ["complete", "class", "bass"])
@pytest.mark.parametrize("transpose, expected", [(True, list(range(-7, 9))), (False, [0])])
def test_available_transpositions_spelling(octaves, transpose, expected):
    fp = os.path.join(data_folder, "wtc_i_prelude_01")
    score = import_piano_roll(fp + ".mxl", "spelling", octaves, INPUT_FPC)
    chords = import_chords(fp + ".csv", LabelCodec(True), OUTPUT_FPC)
    transpositions = _find_available_transpositions(score, chords, "spelling", transpose)
    assert transpositions == expected

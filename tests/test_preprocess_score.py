import os

import numpy as np
import pytest

from frog import INPUT_FPC, OUTPUT_FPC
from frog.preprocessing.preprocess_scores import (
    import_piano_roll,
    calculate_lr_transpositions_pitches,
    transpose_piano_roll,
    generate_input_chunks,
    generate_output_mask_chunks,
    get_metrical_information,
)

data_folder = "resources"


s = ["pitch", "pitch", "pitch", "spelling", "spelling", "spelling"]
o = ["complete", "bass", "class", "complete", "bass", "class"]
e = [
    [24, 28, 31, 48],
    [0, 4, 7, 12],
    [0, 4, 7],
    [85, 86, 89, 155],
    [15, 16, 19, 50],
    [15, 16, 19],
]


@pytest.mark.parametrize("spelling,octaves,expected", zip(s, o, e))
def test_decode_score(spelling, octaves, expected):
    fp = os.path.join(data_folder, "test_score.mxl")
    full_score = import_piano_roll(fp, spelling, octaves, INPUT_FPC)
    assert all([x == y for x, y in zip(np.nonzero(full_score[0, :])[0], expected)])


def test_spelling_bass():
    fp = os.path.join(data_folder, "test_score.mxl")
    full_outcome = import_piano_roll(fp, "spelling", "bass", INPUT_FPC)
    # The bass is a C sharp
    # 35 == offset for bass
    # 7 * 3 == go to sharps (7 for double flats, 7 for flats, 7 for diatonic)
    # 1 == C note (position on the circle of fifths FCGDAEB)
    assert np.nonzero(full_outcome[-1])[0][-1] == 35 + (7 * 3) + 1


@pytest.mark.parametrize("octaves", ["complete", "bass", "class"])
@pytest.mark.parametrize("spelling, expected", [("pitch", (6, 5)), ("spelling", (13, 13))])
def test_find_transposition(octaves, spelling, expected):
    fp = os.path.join(data_folder, "test_score.mxl")
    score = import_piano_roll(fp, spelling, octaves, INPUT_FPC)
    assert calculate_lr_transpositions_pitches(score, spelling) == expected


@pytest.mark.parametrize("spelling", ["pitch", "spelling"])
@pytest.mark.parametrize("octaves", ["complete", "bass", "class"])
@pytest.mark.parametrize("chunk_size, hop_size", [(4, 2)])
def test_score_chunking(spelling, octaves, chunk_size, hop_size):
    fp = os.path.join(data_folder, "test_score.mxl")
    score = import_piano_roll(fp, spelling, octaves, INPUT_FPC)
    chunks = generate_input_chunks(score, chunk_size, hop_size, INPUT_FPC)
    assert len(chunks) == 5
    for c in chunks:
        assert len(c) == 4 * INPUT_FPC
    assert np.sum(chunks[-1][-16:]) == 16 * (4 if octaves == "bass" else 3)


@pytest.mark.parametrize("spelling", ["pitch", "spelling"])
@pytest.mark.parametrize("octaves", ["complete", "bass", "class"])
def test_score_length(spelling, octaves):
    fp = os.path.join(data_folder, "test_score.mxl")
    score = import_piano_roll(fp, spelling, octaves, INPUT_FPC)
    assert len(score) == 12 * INPUT_FPC


@pytest.mark.parametrize("octaves", ["bass", "class"])
def test_transposition(octaves):
    fp = os.path.join(data_folder, "test_score.mxl")
    score = import_piano_roll(fp, "pitch", octaves, INPUT_FPC)
    transposed_score = transpose_piano_roll(score, 12, "pitch", octaves)
    assert np.alltrue(np.equal(score, transposed_score))


@pytest.mark.parametrize("spelling", ["pitch", "spelling"])
@pytest.mark.parametrize("octaves", ["complete", "bass", "class"])
@pytest.mark.parametrize("chunk_size, hop_size", [(4, 2)])
def test_generate_mask_chunks(spelling, octaves, chunk_size, hop_size):
    fp = os.path.join(data_folder, "test_score.mxl")
    score = import_piano_roll(fp, spelling, octaves, INPUT_FPC)
    mask_chunks = generate_output_mask_chunks(score, chunk_size, hop_size, INPUT_FPC, OUTPUT_FPC)
    assert len(mask_chunks) == 5
    for c in mask_chunks:
        assert len(c) == 4 * OUTPUT_FPC
    assert np.sum(mask_chunks[-1]) == 4 * OUTPUT_FPC


def test_metrical_information_mxl():
    fp = os.path.join(data_folder, "test_pickup.mxl")
    outcome = get_metrical_information(fp, 2)
    down_expected = [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    beat_expected = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    expected = np.array([[d, b] for d, b in zip(down_expected, beat_expected)])
    assert np.all(outcome == expected)


@pytest.mark.xfail(reason="Our metrical information code still does not work with MIDI files")
def test_metrical_information_midi():
    fp = os.path.join(data_folder, "test_pickup.mid")
    outcome = get_metrical_information(fp, 2)
    down_expected = [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    beat_expected = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    expected = np.array([[d, b] for d, b in zip(down_expected, beat_expected)])
    assert np.all(outcome == expected)

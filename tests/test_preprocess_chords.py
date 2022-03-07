import os

import pytest

from frog import OUTPUT_FPC
from frog.label_codec import LabelCodec
from frog.preprocessing.preprocess_chords import (
    import_chords,
    generate_chord_chunks,
    calculate_lr_transpositions_key,
    transpose_chord_labels,
    Chord,
)

data_folder = "resources"


@pytest.mark.parametrize(
    "spelling, expected",
    [("pitch", Chord("C", "1", "M", "0", "C")), ("spelling", Chord("C", "1", "M", "0", "C"))],
)
def test_import_chords(spelling, expected):
    cf = os.path.join(data_folder, "wtc_i_prelude_01.csv")
    chords = import_chords(cf, LabelCodec(spelling=spelling == spelling), OUTPUT_FPC)
    assert len(chords) == 140 * OUTPUT_FPC
    assert chords[0] == expected


@pytest.mark.parametrize("spelling", ["pitch", "spelling"])
def test_chunk_chords(spelling):
    cf = os.path.join(data_folder, "wtc_i_prelude_01.csv")
    chords = import_chords(cf, LabelCodec(spelling=spelling == spelling), OUTPUT_FPC)
    chord_chunks = generate_chord_chunks(chords, 10, 10, OUTPUT_FPC)
    assert len(chord_chunks) == 14


@pytest.mark.parametrize("spelling", ["pitch", "spelling"])
def test_transpose_chords(spelling):
    cf = os.path.join(data_folder, "wtc_i_prelude_01.csv")
    chords = import_chords(cf, LabelCodec(spelling=spelling == spelling), OUTPUT_FPC)
    nl, nr = calculate_lr_transpositions_key(chords, spelling)
    for s in range(-nl, nr + 1):
        pitch_proximity = "fifth" if spelling == "spelling" else "semitone"
        chords_transposed = transpose_chord_labels(chords, s, pitch_proximity)
    assert True


@pytest.mark.parametrize("spelling", ["pitch", "spelling"])
def test_encode_chords(spelling):
    cf = os.path.join(data_folder, "wtc_i_prelude_01.csv")
    chords = import_chords(cf, LabelCodec(spelling=spelling == spelling), OUTPUT_FPC)
    nl, nr = calculate_lr_transpositions_key(chords, spelling)
    lc = LabelCodec(spelling == "spelling", strict=False)
    for s in range(-nl, nr + 1):
        pitch_proximity = "fifth" if spelling == "spelling" else "semitone"
        chords_transposed = transpose_chord_labels(chords, s, pitch_proximity)
        encoded_chords = lc.encode_chords(chords_transposed)
    assert True


@pytest.mark.parametrize(
    "chords, spelling, outcome",
    [
        ([Chord("G", "1", "M", "0", "G")], "pitch", (6, 5)),
        ([Chord("C", "1", "M", "0", "C")], "spelling", (7, 10)),
        ([Chord("G", "1", "M", "0", "G")], "spelling", (8, 9)),
        ([Chord("A-", "1", "M", "0", "A-")], "spelling", (3, 14)),
        ([Chord("c##", "1", "m", "0", "c##")], "spelling", (21, -4)),
    ],
)
def test_calculate_lr_transposition_key(chords, spelling, outcome):
    assert calculate_lr_transpositions_key(chords, spelling) == outcome

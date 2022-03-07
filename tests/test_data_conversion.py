import os

import music21
import pytest

from frog.converters.annotation_converters import ConverterRn2Tab, remove_prima_volta

resources_folder = "resources"
dumps_folder = "test_dumps"


def test_conversion_Schubert():
    file_name = "Schubert_Franz_-_Winterreise_D911_-_14_Der_greise_Kopf"
    c = ConverterRn2Tab()
    analysis = c.load_input(os.path.join(resources_folder, file_name + ".txt"))
    score = music21.converter.parse(os.path.join(resources_folder, file_name + ".mxl"))
    data, _flag = c.run(analysis, score)
    c.write_output(data, os.path.join(dumps_folder, file_name + ".csv"))
    assert data[1][0] == 3.25


def test_conversion_pickup_measure():
    file_name = "Franz_Robert_-_6_Gesänge_Op14_-_5_Liebesfrühling"
    c = ConverterRn2Tab()
    analysis = c.load_input(os.path.join(resources_folder, file_name + ".txt"))
    score = music21.converter.parse(os.path.join(resources_folder, file_name + ".mxl"))
    data, _flag = c.run(analysis, score)
    assert data[0][1] > 0
    assert data[2][1] == 2.5


@pytest.mark.parametrize(
    "fn,long_expected,short_expected,last_measure_number",
    [
        ("test_repeat", 32, 16, 4),
        ("test_repeat_complex", 46, 30, 15),
        ("test_repeat_complexer", 52, 30, 15),
    ],
)
def test_remove_prima_volta(fn, long_expected, short_expected, last_measure_number):
    score = music21.converter.parse(os.path.join(resources_folder, fn + ".mxl"))
    assert score.expandRepeats().quarterLength == long_expected
    remove_prima_volta(score)
    assert score.quarterLength == short_expected
    assert (
        score.parts[0].getElementsByClass(music21.stream.Measure)[-1].measureNumber
        == last_measure_number
    )
    notes = [n for n in score.flat.notes]
    assert notes[-1].offset + notes[-1].duration.quarterLength == short_expected

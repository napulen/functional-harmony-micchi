import pytest

from frog.label_codec import LabelCodec


@pytest.mark.parametrize("spelling", [True, False])
@pytest.mark.parametrize(
    "degree, expected",
    [
        ("5/5", (4, 4)),
        ("3", (0, 2)),
        ("+4/-2", (15, 10)),
        ("+5/7", (6, 11)),
        ("1", (0, 0)),
        ("1+/+1", (7, 0)),
        ("-7/2", (1, 20)),
        ("++4", (0, None)),
    ],
)
def test_encode_degree(degree, expected, spelling):
    lc = LabelCodec(spelling, strict=True)
    assert lc.encode_degree(degree) == expected


@pytest.mark.parametrize(
    "spelling, key, expected",
    [
        (True, "A-", 3),
        (False, "A-", 8),
        (False, "a-", 20),
        (True, "a-", 21),
        (True, "C", 7),
        (True, "C#", 14),
        (True, "c#", 32),
    ],
)
@pytest.mark.parametrize("strict", [True, False])
def test_encode_key(spelling, key, expected, strict):
    lc = LabelCodec(spelling, strict=strict)
    assert lc.encode_key(key) == expected


@pytest.mark.parametrize(
    "spelling, root, expected",
    [
        (True, "A-", 11),
        (False, "A-", 8),
        (False, "a-", 8),
        (False, "G#", 8),
        (False, "C", 0),
        (True, "A", 18),
        (True, "C", 15),
        (True, "C#", 22),
        (True, "c#", 22),
    ],
)
@pytest.mark.parametrize("strict", [True, False])
def test_encode_root(spelling, root, expected, strict):
    lc = LabelCodec(spelling, strict=strict)
    assert lc.encode_root(root) == expected


@pytest.mark.parametrize("spelling", [True, False])
def test_decode_encode_key(spelling):
    lc = LabelCodec(spelling)
    for i in range(len(lc.keys)):
        assert lc.encode_key(lc.decode_key(i)) == i
    for k in lc.keys:
        assert lc.decode_key(lc.encode_key(k)) == k


@pytest.mark.parametrize("spelling", [True, False])
def test_decode_encode_degree(spelling):
    lc = LabelCodec(spelling)
    for d in range(len(lc.degrees)):
        for t in range(len(lc.degrees)):
            assert lc.encode_degree(lc.decode_degree(t, d, roman=False)) == (t, d)
    for d_str in lc.degrees:
        for t_str in lc.degrees:
            degree = "/".join([d_str, t_str]) if t_str != "1" else d_str
            assert lc.decode_degree(*lc.encode_degree(degree), roman=False) == degree


@pytest.mark.parametrize(
    "key, degree, root",
    [
        ("G", "+4", "C#"),
        ("C", "+4/5", "C#"),
        ("F-", "+1/+1", "F#"),
        ("B--", "7/5", "E-"),
        ("G#", "-3/+4", "E"),
        ("g#", "-3/+4", "E"),
        ("C", "3/3", "G"),
        ("c", "3/3", "G"),
        ("c", "5/3", "B-"),
        ("c", "3", "E-"),
        ("c", "6", "A-"),
    ],
)
def test_find_chord_root(key, degree, root):
    lc = LabelCodec(spelling=True)
    assert lc.find_chord_root_str(key, degree) == root

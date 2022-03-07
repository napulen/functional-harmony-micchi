import pytest

from frog import find_enharmonic_equivalent


@pytest.mark.parametrize(
    "original, equivalent",
    [
        ("C##", "D"),
        ("C-", "B"),
        ("D-", "C#"),
        ("C--", "A#"),
        ("B--", "A"),
        ("b--", "a"),
        ("c", "c"),
        ("B#", "C"),
    ],
)
def test_find_enharmonic_equivalent(original, equivalent):
    assert find_enharmonic_equivalent(original) == equivalent

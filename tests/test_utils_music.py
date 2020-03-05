from unittest import TestCase

from utils_music import _encode_degree, find_enharmonic_equivalent


class Test(TestCase):
    def test__encode_degree(self):
        self.assertEqual(_encode_degree("+5/7"), (6, 11))
        self.assertEqual(_encode_degree("1"), (0, 0))
        self.assertEqual(_encode_degree("3"), (0, 2))
        self.assertEqual(_encode_degree("-7/2"), (1, 20))

    def test_find_enharmonic_equivalent(self):
        self.assertEqual(find_enharmonic_equivalent('C##'), 'D')
        self.assertEqual(find_enharmonic_equivalent('C-'), 'B')
        self.assertEqual(find_enharmonic_equivalent('D-'), 'C#')
        self.assertEqual(find_enharmonic_equivalent('C--'), 'A#')
        self.assertEqual(find_enharmonic_equivalent('B--'), 'A')
        self.assertEqual(find_enharmonic_equivalent('b--'), 'a')
        self.assertEqual(find_enharmonic_equivalent('c'), 'c')
        self.assertEqual(find_enharmonic_equivalent('B#'), 'C')


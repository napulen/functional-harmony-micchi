from unittest import TestCase

from utils_music import find_enharmonic_equivalent


class TestFind_enharmonic_equivalent(TestCase):
    def test_1(self):
        self.assertEqual(find_enharmonic_equivalent('C##'), 'D')

    def test_2(self):
        self.assertEqual(find_enharmonic_equivalent('C-'), 'B')

    def test_3(self):
        self.assertEqual(find_enharmonic_equivalent('D-'), 'C#')

    def test_4(self):
        self.assertEqual(find_enharmonic_equivalent('C--'), 'A#')

    def test_5(self):
        self.assertEqual(find_enharmonic_equivalent('B--'), 'A')

    def test_6(self):
        self.assertEqual(find_enharmonic_equivalent('b--'), 'a')

    def test_7(self):
        self.assertEqual(find_enharmonic_equivalent('c'), 'c')

    def test_8(self):
        self.assertEqual(find_enharmonic_equivalent('B#'), 'C')

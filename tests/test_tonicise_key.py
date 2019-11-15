from unittest import TestCase

from config import KEYS_PITCH
from temperley_analysis import tonicise_key


class TestTonicise_key(TestCase):
    def test_1(self):
        for k in range(24):
            self.assertEqual(tonicise_key(k, 0), k)

    def test_2(self):
        for k in range(24):
            for pd in range(7):
                x = 3 + 4
                self.assertEqual((tonicise_key(k, pd) % 12), (tonicise_key(k, 7 + pd) - 1) % 12)
                self.assertEqual(tonicise_key(k, pd) % 12, (tonicise_key(k, 14 + pd) + 1) % 12)

    def test_3(self):
        key = ['G', 'A#', 'c', 'd#']
        deg = [5, 3, 2, 3]
        res = ['D', 'd', 'd', 'F#']
        [self.assertEqual(KEYS_PITCH[tonicise_key(KEYS_PITCH.index(k), d - 1)], r) for k, d, r in zip(key, deg, res)]

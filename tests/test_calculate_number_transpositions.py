from unittest import TestCase
import numpy as np

from preprocessing import calculate_number_transpositions


class TestCalculate_number_transpositions(TestCase):
    dt = [('key', '<U10')]  # datatype

    def test_1(self):
        chords = np.array([tuple(['G'])], dtype=self.dt)
        self.assertEqual(calculate_number_transpositions(chords), (15, 14))

    def test_2(self):
        chords = np.array([tuple(['c##'])], dtype=self.dt)
        self.assertEqual(calculate_number_transpositions(chords), (25, 1))

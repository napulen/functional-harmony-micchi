from unittest import TestCase

import numpy as np

from utils_music import find_chord_root


class TestFind_chord_root(TestCase):
    dt = [('key', '<U10'), ('degree', '<U10')]  # datatype

    def test_1(self):
        chord = np.array([tuple(['G', '+4'])], dtype=self.dt)[0]
        self.assertEqual(find_chord_root(chord, 'pitch_spelling'), 'C#')

    def test_2(self):
        chord = np.array([tuple(['C', '+4/5'])], dtype=self.dt)[0]
        self.assertEqual(find_chord_root(chord, 'pitch_spelling'), 'C#')

    def test_3(self):
        chord = np.array([tuple(['F-', '+1/+1'])], dtype=self.dt)[0]
        self.assertEqual(find_chord_root(chord, 'pitch_spelling'), 'F#')

    def test_4(self):
        chord = np.array([tuple(['B--', '7/5'])], dtype=self.dt)[0]
        self.assertEqual(find_chord_root(chord, 'pitch_spelling'), 'E-')

    def test_5(self):
        chord = np.array([tuple(['G#', '-3/+4'])], dtype=self.dt)[0]
        self.assertEqual(find_chord_root(chord, 'pitch_spelling'), 'E')

    def test_6(self):
        chord = np.array([tuple(['g#', '-3/+4'])], dtype=self.dt)[0]
        self.assertEqual(find_chord_root(chord, 'pitch_spelling'), 'E')

    def test_7(self):
        chord = np.array([tuple(['C', '3/3'])], dtype=self.dt)[0]
        self.assertEqual(find_chord_root(chord, 'pitch_spelling'), 'G')

    def test_8(self):
        chord = np.array([tuple(['c', '3/3'])], dtype=self.dt)[0]
        self.assertEqual(find_chord_root(chord, 'pitch_spelling'), 'G')

    def test_9(self):
        chord = np.array([tuple(['c', '5/3'])], dtype=self.dt)[0]
        self.assertEqual(find_chord_root(chord, 'pitch_spelling'), 'B-')

    def test_10(self):
        chord = np.array([tuple(['c', '3'])], dtype=self.dt)[0]
        self.assertEqual(find_chord_root(chord, 'pitch_spelling'), 'E-')

    def test_11(self):
        chord = np.array([tuple(['c', '6'])], dtype=self.dt)[0]
        self.assertEqual(find_chord_root(chord, 'pitch_spelling'), 'A-')

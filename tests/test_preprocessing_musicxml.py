"""
Analyse the scores to check if everything is imported correctly.
To use, run in python console followed by `s, ml, mo = f(i)` where i is the sonata number to check
s will contain the score, ml the measure lengths, and mo the measure offsets.
It is useful to run in python console because it can hold these quantities for manual analysis
and is much faster than debug mode in pycharm (orders of magnitude).

To check correctness, verify the measure lengths. If problems, find the troublesome measure and correct it in musescore.
Typical problems:
 - put each part on its own staff: select entire part then ctrl+shift+↑ (or ↓)
 - make sure that the last measure is correct both in the score and in the BPS-FH
 - remove or correct ornaments
 - check cadenzas
"""

import os
from collections import Counter

import numpy as np
import xlrd
from music21 import converter
from music21.repeat import ExpanderException

DATASET_FOLDER = os.path.join('.', 'BPS_FH_Dataset')

FPQ = 8


def f(i):
    score_file = os.path.join(DATASET_FOLDER, str(i).zfill(2), "score.mxl")
    try:
        score = converter.parse(score_file).expandRepeats()
    except ExpanderException:
        score = converter.parse(score_file)
        print("Could not expand repeats. Maybe there are no repeats in the piece? Please check.")
    n_frames = int(score.duration.quarterLength * FPQ)
    measure_offset = list(score.measureOffsetMap().keys())
    measure_length = np.diff(measure_offset)
    t0 = - measure_offset[1] if measure_length[0] != measure_length[1] else 0  # Pickup time
    print(f"The lengths of measures are {Counter(measure_length)}")
    chord_labels = load_chord_labels(i)
    print(f"t0 = {t0}, last chord label ends at {chord_labels[-1]['end']}, "
          f"last measure should end at {measure_length[-2] + measure_offset[-1] + t0} if full, "
          f"last frame ends at {(n_frames / FPQ) + t0}")
    return score, measure_length, measure_offset


def load_chord_labels(i):
    """
    Load chords of each piece and add chord symbols into the labels.
    :param i: which sonata to take (sonatas indexed from 0 to 31)
    :return: chord_labels
    """
    dt = [('onset', 'float'), ('end', 'float'), ('key', '<U10'), ('degree', '<U10'), ('quality', '<U10'),
          ('inversion', 'int'), ('chord_function', '<U10')]  # datatype
    chords_file = os.path.join(DATASET_FOLDER, str(i).zfill(2), "chords.xlsx")
    workbook = xlrd.open_workbook(chords_file)
    sheet = workbook.sheet_by_index(0)
    chords = []
    for rowx in range(sheet.nrows):
        cols = sheet.row_values(rowx)
        # xlrd.open_workbook automatically casts strings to float if they are compatible. Revert this.
        if isinstance(cols[3], float):  # if type(degree) == float
            cols[3] = str(int(cols[3]))
        chords.append(tuple(cols))
    return np.array(chords, dtype=dt)  # convert to structured array


import os

import numpy as np

from config import FPQ, HSIZE, DATA_FOLDER
from utils_music import load_score_pitch_class, load_chord_labels, attach_chord_root, encode_chords, \
    segment_chord_labels


def build_weight_matrix(pitch_profile_maj, pitch_profile_min):
    major_keys = [np.roll(pitch_profile_maj, s) for s in range(12)]
    minor_keys = [np.roll(pitch_profile_min, s) for s in range(12)]
    return np.array(major_keys + minor_keys)  # shape (keys, notes)


pitch_profile_maj = np.array([5., 2., 3.5, 2., 4.5, 4., 2., 4.5, 2., 3.5, 1.5, 4.])
pitch_profile_maj -= np.mean(pitch_profile_maj)
pitch_profile_min = [5., 2., 3.5, 4.5, 2., 4., 2., 4.5, 3.5, 2., 1.5, 4.]
pitch_profile_min -= np.mean(pitch_profile_min)

context = 65  # 4 quarter notes on each side + the current note
pad = 32  # the amount to pad on each side of the input tensor to give a context of the correct size
hop_size = 4  # distance between successive predictions
x = build_weight_matrix(pitch_profile_maj, pitch_profile_min)  # shape (keys, notes)
# x_var = np.sum(x ** 2, axis=1)  # shape (keys)
# x_var = np.var(x, axis=1)


def load_data(sf, cf):
    piano_roll = load_score_pitch_class(sf, FPQ)
    pad_additional = (- piano_roll.shape[1]) % HSIZE  # makes sure that the dimensions match
    piano_roll = np.pad(piano_roll, ((0, 0), (pad, pad + pad_additional)), 'constant', constant_values=0)
    n_frames_analysis = piano_roll.shape[1] // HSIZE
    chord_labels = load_chord_labels(cf)
    cl_full = attach_chord_root(chord_labels, pitch_spelling=False)
    cl_segmented = segment_chord_labels(cl_full, n_frames_analysis, hsize=HSIZE, fpq=FPQ)
    chords = encode_chords(cl_segmented, mode='semitone')
    keys = [c[0] for c in chords]
    return piano_roll, keys


def analyse_piece(sf, cf):
    notes, keys = load_data(sf, cf)  # shape (notes, timesteps), (timesteps)
    ts = notes.shape[1]
    res = []
    for i in range(0, ts - context, hop_size):
        y = np.sum(notes[:, i:i + context], axis=1, dtype=float)  # shape (12)
        y -= np.mean(y)
        pred = np.argmax(np.dot(x, y) / (np.std(x, axis=1) * np.std(y)))

        res.append(pred)

    return res, keys


if __name__ == '__main__':
    folder = os.path.join(DATA_FOLDER, 'BPS')
    chords_folder = os.path.join(folder, 'chords')
    scores_folder = os.path.join(folder, 'scores')
    file_names = sorted(['.'.join(fn.split('.')[:-1]) for fn in os.listdir(chords_folder)])
    tp = 0  # true positives
    N = 0  # total number of predictions
    for fn in file_names:
        sf = os.path.join(scores_folder, fn + ".mxl")
        cf = os.path.join(chords_folder, fn + ".csv")
        y_pred, y_true = analyse_piece(sf, cf)
        N_last = len(y_true)
        N += N_last
        tp_last = sum([yp == yt for yp, yt in zip(y_pred, y_true)])
        tp += tp_last
        print(f'{fn}, accuracy : {tp_last / N_last}')
    print(tp / N)

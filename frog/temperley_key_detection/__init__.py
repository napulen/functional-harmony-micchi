import numpy as np


def build_weight_matrix(pitch_profile_maj, pitch_profile_min):
    major_keys = [np.roll(pitch_profile_maj, s) for s in range(12)]
    minor_keys = [np.roll(pitch_profile_min, s) for s in range(12)]
    return np.array(major_keys + minor_keys)  # shape (keys, notes)


PITCH_PROFILE_MAJ = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
PITCH_PROFILE_MAJ -= np.mean(PITCH_PROFILE_MAJ)
PITCH_PROFILE_MIN = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])
PITCH_PROFILE_MIN -= np.mean(PITCH_PROFILE_MIN)
DEGREE2SCALE_MIN = [0, 2, 3, 5, 7, 8, 11]
DEGREE2SCALE_MAJ = [0, 2, 4, 5, 7, 9, 11]
TEMPERLEY_WEIGHTS = build_weight_matrix(PITCH_PROFILE_MAJ, PITCH_PROFILE_MIN)  # shape (keys, notes)
TEMPERLEY_STD = np.std(TEMPERLEY_WEIGHTS, axis=1)
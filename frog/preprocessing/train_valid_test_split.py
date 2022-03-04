"""
This is an entry point, no other file should import from this one.
Split the data into training and validation set for every corpora that we have separately.
"""
import logging
import os
import random
from shutil import copyfile

logger = logging.getLogger(__name__)


def split_bps_like_chen_su(in_folder, out_folder):
    """Create the same dataset as Chen and Su for comparison."""
    # The training, validation, and test sonatas used by Chen and Su in their first paper
    # i_trn = [5, 12, 17, 21, 27, 32, 4, 9, 13, 18, 24, 22, 28, 30, 31, 11, 2, 3]
    # i_vld = [8, 19, 29, 16, 26, 6, 20]
    # i_tst = [1, 14, 23, 15, 10, 25, 7]

    # These numbers refer to the index of the sonatas
    i_vld = list(range(1, 33, 4))  # They take one sonata every four for testing
    i_tst = i_vld  # They use only training and validation
    i_trn = [x for x in range(1, 33) if x not in i_vld]

    sub = ["train" for _ in i_trn] + ["valid" for _ in i_vld] + ["test" for _ in i_tst]
    file_names = [f"bps_{i:02d}_01" for i in i_trn + i_vld + i_tst]
    for s, f in zip(sub, file_names):
        for ext, t in zip([".mxl", ".csv", ".txt"], ["scores", "chords", "txt"]):
            copyfile(
                os.path.join(in_folder, t, f + ext),
                os.path.join(out_folder, s, t, f + ext),
            )
    return


def split_bach_like_chen_su(in_folder, out_folder):
    """Create the same dataset as Chen and Su for comparison."""
    # These numbers refer to the index of the Bach WTC I preludes
    # prelude 13 has some quantization issues, maybe remove it altogether?
    excluded = []
    i_vld = list(range(1, 25, 4))
    # i_vld = [2, 6, 10, 14, 18, 22]  # Take 1 out of 4 for testing
    i_tst = i_vld  # They use only training and validation
    i_trn = [x for x in range(1, 25) if x not in i_vld + excluded]

    sub = ["train" for _ in i_trn] + ["valid" for _ in i_vld] + ["test" for _ in i_tst]
    file_names = [f"wtc_i_prelude_{i:02d}" for i in i_trn + i_vld + i_tst]
    for s, f in zip(sub, file_names):
        for ext, t in zip([".mxl", ".csv", ".txt"], ["scores", "chords", "txt"]):
            copyfile(
                os.path.join(in_folder, t, f + ext),
                os.path.join(out_folder, s, t, f + ext),
            )
    return


def train_valid_test_split(in_folder, out_folder, seed=18, split=(0.8, 0.1, 0.1)):
    """
    Split a data folder into training, validation, and test set.
    @param in_folder:
    @param out_folder:
    @param seed: For replicability purposes
    @param split: Percentage of files in training, validation, and test set respectively
    """
    if not os.path.isdir(in_folder):
        logger.warning(
            f"{in_folder} is not a folder. Can't find the files to split. Skipping."
        )
        return
    if in_folder.startswith("."):
        logger.warning(f"{in_folder} is hidden. Skipping.")
        return

    assert sum(split) == 1, "Please give split sizes that sum to 1"
    assert all(
        x.replace(".csv", ".mxl") == y
        for x, y in zip(
            sorted(os.listdir(os.path.join(in_folder, "chords"))),
            sorted(os.listdir(os.path.join(in_folder, "scores"))),
        )
    ), f"The structure of the input folder {in_folder} is wrong"

    file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(os.path.join(in_folder, "chords"))
        if f.endswith(".csv")
    ]
    n_test = round(len(file_names) * split[2])
    n_valid = round(len(file_names) * split[1])
    n_train = len(file_names) - n_test - n_valid
    dataset = ["train"] * n_train + ["valid"] * n_valid + ["test"] * n_test

    random.seed(seed)
    random.shuffle(file_names)
    for ds, f in zip(dataset, file_names):
        for ext, t in zip([".mxl", ".csv", ".txt"], ["scores", "chords", "txt"]):
            copyfile(
                os.path.join(in_folder, t, f + ext),
                os.path.join(out_folder, ds, t, f + ext),
            )

# TODO: Unfinished
# def cross_valid_split(in_folder, out_folder, n_folds, seed=18, test_size=0.1):
#     """
#     Split a data folder into training, validation, and test set.
#     @param in_folder:
#     @param out_folder:
#     @param seed: For replicability purposes
#     @param split: Percentage of files in training, validation, and test set respectively
#     """
#     if not os.path.isdir(in_folder):
#         logger.warning(f"{in_folder} is not a folder. Can't find the files to split. Skipping.")
#         return
#     if in_folder.startswith("."):
#         logger.warning(f"{in_folder} is hidden. Skipping.")
#         return
#
#     file_names = [
#         os.path.splitext(f)[0]
#         for f in os.listdir(os.path.join(in_folder, "chords"))
#         if f.endswith(".csv")
#     ]
#     n_test = round(len(file_names) * test_size)
#     n_valid = round(len(file_names) * (1 - test_size) / n_folds)
#     n_train = len(file_names) - n_test - n_valid
#     dataset = ["train"] * n_train + ["valid"] * n_valid + ["test"] * n_test
#
#     random.seed(seed)
#     random.shuffle(file_names)
#     for fold in range(n_folds):
#         for i in range(n_test):
#             f = file_names[i]
#             for ext, t in zip([".mxl", ".csv", ".txt"], ["scores", "chords", "txt"]):
#                 copyfile(
#                     os.path.join(in_folder, t, f + ext),
#                     os.path.join(out_folder, "test", str(fold), t, f + ext),
#                 )
#         for i in range(n_valid):
#             f = file_names[i + n_test + fold * n_valid]
#             for ext, t in zip([".mxl", ".csv", ".txt"], ["scores", "chords", "txt"]):
#                 copyfile(
#                     os.path.join(in_folder, t, f + ext),
#                     os.path.join(out_folder, "test", str(fold), t, f + ext),
#                 )

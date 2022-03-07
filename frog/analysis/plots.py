import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from frog import NOTES, PITCH_FIFTHS
from frog.analysis.analyse_results import _find_root_from_output
from frog.label_codec import KEYS_PITCH, KEYS_SPELLING, OUTPUT_FEATURES, QUALITIES


# TODO: Fix this module, it is currently broken. Or just throw it away?


def plot_chord_changes(y_true, y_pred, name, ts, inversions=True):
    """
    Plot chord changes
    :param y_true: shape [outputs] (timesteps, output_features)
    :param y_pred:
    :param name:
    :param ts: the total number of timesteps in this prediction
    :param inversions:
    :return:
    """
    if inversions:
        yt = [np.argmax(y, axis=-1) for y in y_true]
        yp = [np.argmax(y, axis=-1) for y in y_pred]
    else:
        yt = [np.argmax(y, axis=-1) for y in y_true[:-1]]
        yp = [np.argmax(y, axis=-1) for y in y_pred[:-1]]

    change_true, change_pred = np.zeros(ts), np.zeros(ts)
    for m in range(ts - 1):
        if np.any([y[m + 1] != y[m] for y in yt]):
            change_true[m] = 1
        if np.any([y[m + 1] != y[m] for y in yp]):
            change_pred[m] = 1

    # Plotting the results
    cmap = sns.color_palette(["#d73027", "#f7f7f7", "#3027d7"])
    ax = sns.heatmap([change_true - change_pred], cmap=cmap, linewidths=0.5)
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([-0.67, 0.0, 0.67])
    colorbar.set_ticklabels(["False Pos", "True", "False Neg"])
    ax.set(
        ylabel=["change_true", "change_pred"],
        xlabel="time",
        title=f"Sonata {name} - chord consistency "
        + ("with inversions" if inversions else "without inversions"),
    )
    plt.show()
    # zt = decode_results(yt)
    # zp = decode_results(yp)
    # wt = [' '.join([zt[0][i], zt[1][i], zt[2][i], zt[3][i]]) for i in range(ts)]
    # wp = [' '.join([zp[0][i], zp[1][i], zp[2][i], zp[3][i]]) for i in range(ts)]
    return


def plot_results(y_true, y_pred, name, start, mode="probabilities", pitch_spelling=True):
    """

    :param y_true: shape [outputs] (timesteps, features output)
    :param y_pred: same shape as above
    :param name: the title of the piece we are analysing
    :param start: the initial frame
    :param mode: probabilities or predictions
    :param pitch_spelling: this controls the shape and labels of the x axis in with keys
    :return:
    """
    plt.style.use("ggplot")
    if mode not in ["probabilities", "predictions"]:
        raise ValueError("mode should be either probabilities or predictions")
    cmap = (
        sns.color_palette(["#d73027", "#f7f7f7", "#3027d7", "#000000"])
        if mode == "predictions"
        else "RdGy"
    )

    tick_labels = [
        KEYS_SPELLING if pitch_spelling else KEYS_PITCH,
        [str(x + 1) for x in range(7)]
        + [str(x + 1) + "b" for x in range(7)]
        + [str(x + 1) + "#" for x in range(7)],
        [str(x + 1) for x in range(7)]
        + [str(x + 1) + "b" for x in range(7)]
        + [str(x + 1) + "#" for x in range(7)],
        QUALITIES,
        [str(x) for x in range(4)],
        PITCH_FIFTHS if pitch_spelling else NOTES,
        PITCH_FIFTHS if pitch_spelling else NOTES,
    ]
    ylabels = OUTPUT_FEATURES.copy()
    ylabels.append("root_der")
    for j in range(7):
        # if j > 0:  # temporary analysis tool, remove if not needed
        #     continue
        if j == 0:
            if pitch_spelling:
                ordering = [i + j for i in range(15) for j in [0, 15]]
                # ordering = [i + j for i in range(26) for j in [0, 29]]
                # [ordering.append(i) for i in [26, 27, 28]]
            else:
                ordering = [8, 3, 10, 5, 0, 7, 2, 9, 4, 11, 6, 1]
                ordering += [x + 12 for x in ordering]

            a = y_pred[j][:, ordering]
            b = y_true[j][:, ordering]
            yticklabels = [tick_labels[j][o] for o in ordering]
        else:
            if j == 6:
                a = _find_root_from_output(y_pred, pitch_spelling=pitch_spelling)
                a = _indices_to_one_hot(a, 35 if pitch_spelling else 12)
                b = y_true[5]
            else:
                a = y_pred[j]
                b = y_true[j]

            yticklabels = tick_labels[j]

        if mode == "predictions":
            a = a == np.max(a, axis=-1, keepdims=True)
            x = b - a
            x[b == 1] += 1
            x = x.transpose()
            ax = sns.heatmap(x, cmap=cmap, vmin=-1, vmax=2, yticklabels=yticklabels, linewidths=0.5)
            colorbar = ax.collections[0].colorbar
            colorbar.set_ticks([-5 / 8, 1 / 8, 7 / 8, 13 / 8])
            colorbar.set_ticklabels(["False Pos", "True Neg", "True Pos", "False Neg"])
        else:
            x = b - a
            x = x.transpose()
            ax = sns.heatmap(
                x, cmap=cmap, center=0, vmin=-1, vmax=1, yticklabels=yticklabels, linewidths=0.5
            )
            colorbar = ax.collections[0].colorbar
            colorbar.set_ticks([-1, 0, +1])
            colorbar.set_ticklabels(["False Pos", "True", "False Neg"])
        ax.set(
            ylabel=ylabels[j],
            xlabel="time",
            title=f"{name}, frames [{start}, {start + x.shape[1]}) - {ylabels[j]}",
        )
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        plt.show()
    return


def plot_piano_roll(pr, name):
    """

    :param pr:
    :param name: the title of the piece we are analysing
    :return:
    """
    ax = sns.heatmap(pr.transpose(), vmin=0, vmax=1)
    ax.set(xlabel="time", ylabel="notes", title=f"{name} - piano roll data")
    plt.show()
    return


def plot_coherence(root_pred, root_der, n_classes, name):
    mask = root_pred != root_der
    c = np.zeros((n_classes, n_classes))
    for i, j in zip(root_pred[mask], root_der[mask]):
        c[i, j] += 1
    labels = PITCH_FIFTHS if n_classes == 35 else NOTES
    sns.heatmap(c, xticklabels=labels, yticklabels=labels, linewidths=0.5)
    plt.title(f"{name} - root prediction")
    plt.xlabel("PREDICTED")
    plt.ylabel("DERIVED")
    plt.show()
    return


def _indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

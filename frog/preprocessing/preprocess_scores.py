import logging
import math
import os

import music21
import numpy as np

from frog import INPUT_FPC, OUTPUT_FPC, PF2I, PF2PS
from frog.converters.annotation_converters import remove_prima_volta

logger = logging.getLogger(__name__)


def _load_score(score_file, input_fpc):
    score = music21.converter.parse(score_file)
    remove_prima_volta(score)
    n_frames = math.ceil(score.duration.quarterLength * input_fpc)
    return score, n_frames


def get_metrical_information(score_file, input_fpc):
    # TODO: Maybe this entire function could use part = score.parts[0] instead of score?
    score, n_frames = _load_score(score_file, input_fpc)
    score_mom = score.measureOffsetMap()
    # consider only measures that have not been marked as "excluded" in the musicxml
    # we assume all parts share the same measures (and take the part [0])
    measure_offsets = np.array(
        [k for k in score_mom.keys() if score_mom[k][0].numberSuffix is None]
    )
    offsets = np.arange(n_frames) / input_fpc

    # Get metrical info from time signatures
    ts_list = list(score.flat.getTimeSignatures())
    first_measure_number = 0 if any([ts.measureNumber == 0 for ts in ts_list]) else 1
    time_signatures = {max(ts.measureNumber - first_measure_number, 0): ts for ts in ts_list}
    ts_measures = np.array(sorted(time_signatures.keys()))
    starting_beat = ts_list[0].beat - 1  # Convert beats to zero index

    measures = np.searchsorted(measure_offsets, offsets, side="right") - 1
    offset_in_measure = offsets - measure_offsets[measures] + starting_beat * (measures == 0)
    beat_idx = ts_measures[np.searchsorted(ts_measures, measures, side="right") - 1]
    beat_duration = np.array([time_signatures[b].beatDuration.quarterLength for b in beat_idx])
    beats = offset_in_measure / beat_duration  # rntxt format has beats starting at 1
    metrical_info = np.array([[b == 0.0, b == int(b)] for b in beats]).astype(float)
    return metrical_info


def _load_score_complete(score_file, spelling, input_fpc):
    """
    Load notes in each piece, which is then represented as piano roll.
    :param score_file: the path to the file to analyse
    :param spelling: whether to use MIDI numbers ('pitch') or proper pitch spelling ('spelling')
    :param input_fpc: frames per crotchet
    :repeats: if True, expand the repeats in the score
    :return: piano_roll, shape (n_frames, n_pitches)
    """
    assert spelling in ["pitch", "spelling"], "Please select either pitch or spelling as mode"
    score, n_frames = _load_score(score_file, input_fpc)
    n_pitches = 12 if spelling == "pitch" else 35
    piano_roll = np.zeros(shape=(n_frames, n_pitches * 7), dtype=np.int32)

    for n in score.flat.notes:
        notes = np.array([x for x in n] if n.isChord else [n])
        start = int(round(n.offset * input_fpc))
        end = start + max(int(round(n.duration.quarterLength * input_fpc)), 1)
        time = np.arange(start, end)
        for note in notes:
            octave = note.pitch.octave - 1
            if octave < 0 or octave >= 7:  # we keep just 7 octaves in total
                logger.warning("Score outside the octave boundaries. Skipped a note.")
                continue
            idx = note.pitch.pitchClass if spelling == "pitch" else PF2I[note.pitch.name]
            piano_roll[time, idx + n_pitches * octave] = 1
    return piano_roll


def _complete_to_bass(pr_complete, mode):
    def find_bass_spelling_frame(frame):
        for i in range(7):
            temp = frame[n_classes * i : n_classes * (i + 1)]
            (active_pitches,) = np.nonzero(temp)
            if len(active_pitches) > 0:
                pitch_tuples = [(p, PF2PS[p]) for p in active_pitches]
                ordered_pitches = sorted(pitch_tuples, key=lambda x: x[1])
                return ordered_pitches[0][0]
        return 0

    def find_bass(pr_complete, mode):
        if mode == "pitch":
            bass_pitches = np.argmax(pr_complete, axis=1)  # argmax takes the first non-zero value
            bass_pitches %= n_classes
        else:
            bass_pitches = np.zeros(pr_complete.shape[0], dtype=int)
            for t, frame in enumerate(pr_complete):
                bass_pitches[t] = find_bass_spelling_frame(frame)
        values = np.max(pr_complete, axis=1)  # values == 1 if there are notes, 0 otherwise
        return bass_pitches, values

    # remove information on octaves
    assert mode in ["pitch", "spelling"], "Please specify either pitch or spelling as mode"
    pr_class = _complete_to_class(pr_complete)
    n_frames, n_classes = pr_class.shape
    assert n_classes == 12 if mode == "pitch" else 35
    pr_bass = np.concatenate([pr_class, np.zeros((n_frames, n_classes))], axis=1)

    # store information on the bass
    bass_pitches, values = find_bass(pr_complete, mode)
    pr_bass[np.arange(n_frames), n_classes + bass_pitches % n_classes] = values

    return pr_bass


def _complete_to_class(pr_complete):
    n_frames, n_pitches = pr_complete.shape
    n_classes = int(n_pitches / 7)
    assert n_classes * 7 == n_pitches, "the initial piano roll is not made of 7 octaves"
    piano_roll = np.zeros(shape=(n_frames, n_classes), dtype=np.int32)

    # remove information on octaves
    pitch_times, pitches = pr_complete.nonzero()
    piano_roll[pitch_times, pitches % n_classes] = 1
    return piano_roll


def import_piano_roll(score_file, spelling, octaves, input_fpc):
    """Return a piano_roll with shape (frames, pitches)"""
    assert octaves in ["complete", "bass", "class"]
    piano_roll = _load_score_complete(score_file, spelling, input_fpc)
    if octaves == "bass":
        piano_roll = _complete_to_bass(piano_roll, spelling)
    elif octaves == "class":
        piano_roll = _complete_to_class(piano_roll)
    return piano_roll


def generate_input_chunks(x, chunk_size, hop_size, input_fpc):
    """Chunk an input array x (e.g., piano_roll) in smaller, possibly overlapping, pieces"""
    res = []
    elapsed = 0
    while elapsed + chunk_size * input_fpc < len(x) + hop_size * input_fpc:
        chunk = x[elapsed : elapsed + chunk_size * input_fpc, :]
        # Pad until reaching a full crotchet
        padded_chunk = np.pad(chunk, [(0, (-len(chunk) % input_fpc)), (0, 0)])
        res.append(padded_chunk)
        elapsed += hop_size * input_fpc
    return res


def generate_output_mask_chunks(piano_roll, chunk_size, hop_size, input_fpc, output_fpc):
    # TODO: Maybe this should change. I'm sure we can come up with something simpler
    elapsed_crotchets = 0
    score_len_crotchets = len(piano_roll) / input_fpc
    chunk_masks = []
    while elapsed_crotchets + chunk_size < score_len_crotchets + hop_size:
        remaining_length_crotchets = math.ceil(score_len_crotchets - elapsed_crotchets)
        mask = np.ones((chunk_size * output_fpc))
        if remaining_length_crotchets < chunk_size:
            mask[remaining_length_crotchets * output_fpc :] = 0
        chunk_masks.append(mask)
        elapsed_crotchets += hop_size
    return chunk_masks


def calculate_lr_transpositions_pitches(piano_roll, spelling):
    assert spelling in ["pitch", "spelling"], "Please choose either pitch or spelling"
    if spelling == "pitch":
        return 6, 5

    n_pitches = 35
    nl, nr = 35, 35
    for i in range(7):
        temp = piano_roll[:, n_pitches * i : n_pitches * (i + 1)]
        times, active_pitches = np.nonzero(temp)
        if len(active_pitches) > 0:
            nl = min(nl, np.min(active_pitches))
            nr = min(nr, 35 - np.max(active_pitches))
    return nl, nr


def transpose_piano_roll(piano_roll, s, spelling, octaves):
    """Transpose a score of s units."""
    if spelling == "spelling" or octaves == "complete":
        return np.roll(piano_roll, shift=s, axis=1)

    # with MIDI pitch classes, pitch class 11 -> 0 (= 12) when transposed of +1
    pr_transposed = np.zeros(piano_roll.shape, dtype=np.int32)
    for i in range(12):  # transpose the main part
        pr_transposed[:, i] = piano_roll[:, (i - s) % 12]  # the minus sign is correct!
    for i in range(12, pr_transposed.shape[1]):  # transpose the bass, if present
        pr_transposed[:, i] = piano_roll[:, ((i - s) % 12) + 12]
    return pr_transposed


def prepare_input_from_score_file(sf, input_type, chunk_size, metrical=True):
    spelling, octaves = input_type.split("_")
    piano_roll = import_piano_roll(sf, spelling, octaves, INPUT_FPC)
    pr_chunks = generate_input_chunks(piano_roll, chunk_size, chunk_size, INPUT_FPC)
    pr_chunks = np.array([pad_to_length(c, chunk_size * INPUT_FPC) for c in pr_chunks])
    structure = np.zeros_like(piano_roll)
    if metrical:
        try:
            structure = get_metrical_information(sf, INPUT_FPC)
        except:
            logger.warning("Couldn't get metrical information, returning a vector of zeros")
    st_chunks = generate_input_chunks(structure, chunk_size, chunk_size, INPUT_FPC)
    st_chunks = np.array([pad_to_length(c, chunk_size * INPUT_FPC) for c in st_chunks])
    masks = generate_output_mask_chunks(piano_roll, chunk_size, chunk_size, INPUT_FPC, OUTPUT_FPC)
    n = len(pr_chunks)
    names = np.array([os.path.splitext(os.path.basename(sf))[0]] * n)
    transpositions = np.array([0] * n)
    starts = np.arange(n) * chunk_size * OUTPUT_FPC
    return (pr_chunks, st_chunks, np.array(masks)), (names, transpositions, starts)


def pad_to_length(piano_roll, length):
    pad_width = length - len(piano_roll)
    return np.pad(piano_roll, ((0, pad_width), (0, 0)))

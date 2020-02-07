"""

NB:
*** !!! NOT FOR PUBLIC RELEASE IN CURRENT FORM !!! ***


===============================
BPSFH to ROMANTEXT (converter_tabular2roman.py)
===============================

Mark Gotham, 2019


LICENCE:
===============================

Creative Commons Attribution-NonCommercial 4.0 International License.
https://creativecommons.org/licenses/by-nc/4.0/


ABOUT AND NOTES:
===============================

Tools for combining and converting the BPSFH dataset into other formats.

Can do:
- Imports the constituent parts (downbeats, chords, phrases);
- Deduces measure and beat info from the offset positions relative to downbeats (for chord and phrases separately);
- Combines chords and phrases;
- Deduces position of exposition repeats and assigns additional, adjusted 'real' (printed) measure numbers accordingly;
- Writes all that info into various formats.

Does not currently do (but could):
- repeats: expand to account for variable position of start repeat (e.g. sonata 32)
- repeats: second half (development-recapitulation) repeat as well as exposition

TODO (Cannot currently do):
- Time signatures. No representation in the source - can only deduce length, not structure.
- Beats. Again, no representation (due to MIDI). Assumed X/4. To correct:
-- for compound beats time signatures: Y = ((X-1)/1.5) + 1
-- for Y/2 time signatures. = ((X)/2) + 0.5
(^ Easy to imlpement this when starting from a score, of course).

MG notes:
- second half repeats = low priority -- just duplication of material, no change to offsets etc
- compress and tidy up RN writer (classes like csv writer).
- specific piece errors:
Errors and notes:
- measure/beat error in #2, #28. 'Works' if comment out intBeat, sortExpoRepeat, writeRoman
- 2 done kind of (missing last repeat which did not align)
- cadenza in 3: second chord beat = 'X'. Likewise in the late one (30?)
- missing first entry in several (e.g. #3 in all, #1 in chords)
- Further impossibility in cadenzae. Sonata 30 sorted very manually (Henle and Schenker eds).
"""

import os

import numpy as np
# ------------------------------------------------------------------------------
from music21 import converter
from music21.repeat import ExpanderException

from config import DATA_FOLDER
from utils import decode_roman, int_to_roman
from utils_music import load_chord_labels


def get_rn_row(datum, in_row=None):
    """
    Write the start of a line of RNTXT.
    To start a new line (one per measure with measure, beat, chord), set inString = None.
    To extend an existing line (measure already given), set inString to that existing list.

    :param datum: measure, beat, chord
    """

    if in_row is None:  # New line
        in_row = 'm' + str(datum[0])

    return ' '.join([in_row, f'b{datum[1]}', datum[2]] if datum[1] != 1 else [in_row, datum[2]])


def retrieve_measure_and_beat(offset, measure_offsets, beat_durations, ts_measures, beat_zero):
    measure = np.searchsorted(measure_offsets, offset, side='right') - 1

    offset_in_measure = offset - measure_offsets[measure]
    measure_with_pickup = measure + (0 if beat_zero > 1 else 1)
    beat_idx = ts_measures[np.searchsorted(ts_measures, measure_with_pickup, side='right') - 1]
    beat = (offset_in_measure / beat_durations[beat_idx]) + (beat_zero if measure_with_pickup == 0 else 1)
    beat = int(beat) if int(beat) == beat else round(float(beat), 2)

    return measure_with_pickup, beat


def interpret_degree(degree):
    if '/' in degree:
        num, den = degree.split('/')
    else:
        num, den = degree, '1'

    num_prefix = ''
    while num[0] in ['+', '-']:
        num_prefix += 'b' if num[0] == '-' else '#'
        num = num[1:]
    if num == '1+':
        print("Degree 1+, ignoring the +")
    num = num_prefix + int_to_roman(int(num[0]))

    den_prefix = ''
    while den[0] in ['+', '-']:
        den_prefix += 'b' if num[0] == '-' else '#'
        den = den[1:]
    den = den_prefix + int_to_roman(int(den[0]))

    return num, den


def tabular2roman(tabular, score):
    rn = []
    starting_beat_measure_zero = score.flat.getTimeSignatures()[0].beat

    score_mom = score.measureOffsetMap()
    # consider only measures that have not been marked as "excluded" in the musicxml (for example using Musescore)
    score_measure_offset = [k for k in score_mom.keys() if
                            score_mom[k][0].numberSuffix is None]  # the [0] because there are two parts
    score_measure_offset.append(score.duration.quarterLength)
    beat_durations = dict([(ts.measureNumber, ts.beatDuration.quarterLength) for ts in score.flat.getTimeSignatures()])
    time_signatures = dict([(ts.measureNumber, ts.ratioString) for ts in score.flat.getTimeSignatures()])
    ts_measures = sorted(time_signatures.keys())
    ts_offsets = [score_measure_offset[m if (starting_beat_measure_zero > 1) else m - 1] for m in ts_measures]
    previous_measure, previous_end, previous_key, j = -1, 0, None, 0

    for row in tabular:
        start, end, key, degree, quality, inversion = row
        num, den = interpret_degree(degree)
        chord = decode_roman(num, den, str(quality), str(inversion))
        key = key.replace('-', 'b')
        annotation = f'{key}: {chord}' if key != previous_key else chord

        # Time signature change
        while j < len(ts_offsets) and start >= ts_offsets[j]:
            rn.append('')
            rn.append(f"Time Signature : {time_signatures[ts_measures[j]]}")
            rn.append('')
            j += 1

        # No chords passage
        if start != previous_end:
            m, b = retrieve_measure_and_beat(end, score_measure_offset, beat_durations, ts_measures,
                                             starting_beat_measure_zero)
            if m == previous_measure:
                rn[-1] = get_rn_row([m, b, ''], in_row=rn[-1])
            else:
                rn.append(get_rn_row([m, b, '']))
            previous_measure = m

        m, b = retrieve_measure_and_beat(start, score_measure_offset, beat_durations, ts_measures,
                                         starting_beat_measure_zero)
        if m == previous_measure:
            rn[-1] = get_rn_row([m, b, annotation], in_row=rn[-1])
        else:
            rn.append(get_rn_row([m, b, annotation]))
        previous_measure, previous_end, previous_key = m, end, key

    return rn


def convert_file(score_path, csv_path, txt_path):
    tabular = load_chord_labels(csv_path)
    # score = converter.parse(score_path)
    if 'bps' in score_path:
        try:
            score = converter.parse(score_path).expandRepeats()
        except ExpanderException:
            score = converter.parse(score_path)
    else:
        score = converter.parse(score_path)
    rn = tabular2roman(tabular, score)
    with open(txt_path, 'w') as f:
        for row in rn:
            f.write(row + os.linesep)
    return


def convert_corpus(base_folder, corpus):
    txt_folder = os.path.join(base_folder, corpus, 'txt_generated')
    score_folder = os.path.join(base_folder, corpus, 'scores')
    csv_folder = os.path.join(base_folder, corpus, 'chords')
    os.makedirs(txt_folder, exist_ok=True)

    file_list = []
    for csv_file in os.listdir(csv_folder):
        if csv_file.endswith('.csv'):
            file_list.append(csv_file)
    file_list = sorted(file_list)

    for csv_file in file_list:
        # number = int(csv_file.split('.')[0][-2:])  # bach
        number = int(csv_file.split('_')[1])  # bps
        # if number != 1:
        #     continue
        # if '066' not in csv_file:
        #     continue
        print(csv_file)
        score_file = f'{csv_file.split("_")[0]}.mxl' if 'Tavern' in corpus else f'{csv_file[:-4]}.mxl'
        txt_file = f'{csv_file[:-4]}.txt'
        convert_file(os.path.join(score_folder, score_file),
                     os.path.join(csv_folder, csv_file),
                     os.path.join(txt_folder, txt_file))


if __name__ == '__main__':
    base_folder = DATA_FOLDER

    corpora = [
        os.path.join('Tavern', 'Beethoven'),
        os.path.join('Tavern', 'Mozart'),
        'Bach_WTC_1_Preludes',
        '19th_Century_Songs',
        'Beethoven_4tets',
        'BPS',
    ]

    for c in corpora[5:]:
        convert_corpus(base_folder, c)

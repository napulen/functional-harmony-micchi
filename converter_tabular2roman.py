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


def _get_rn_row(datum, in_row=None):
    """
    Write the start of a line of RNTXT.
    To start a new line (one per measure with measure, beat, chord), set in_row = None.
    To extend an existing line (measure already given), set in_row to that existing list.

    :param datum: measure, beat, annotation
    :param in_row: if None, this is a new measure
    """

    measure, beat, annotation = datum

    if in_row is None:  # New line
        in_row = 'm' + str(measure)

    beat = int(beat) if int(beat) == beat else round(float(beat), 2)  # just reformat it prettier

    return ' '.join([in_row, f'b{beat}', annotation] if beat != 1 else [in_row, annotation])


def _retrieve_measure_and_beat(offset, measure_offsets, time_signatures, ts_measures, beat_zero):
    # find what measure we are by looking at all offsets
    measure = np.searchsorted(measure_offsets, offset, side='right') - 1
    rntxt_measure_number = measure + (0 if beat_zero else 1)  # the measure number we will write in the output

    offset_in_measure = offset - measure_offsets[measure]
    beat_idx = ts_measures[np.searchsorted(ts_measures, measure, side='right') - 1]
    beat_duration = time_signatures[beat_idx].beatDuration.quarterLength

    beat = (offset_in_measure / beat_duration) + 1  # rntxt format has beats starting at 1
    if rntxt_measure_number == 0:  # add back the anacrusis to measure 0
        beat += beat_zero

    return rntxt_measure_number, beat


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


def _get_measure_offsets(score):
    """
    The measure_offsets are zero-indexed: the first measure in the score will be at index zero, regardless of anacrusis.

    :param score:
    :return: a list where at index m there is the offset in quarter length of measure m
    """
    score_mom = score.measureOffsetMap()
    # consider only measures that have not been marked as "excluded" in the musicxml (for example using Musescore)
    measure_offsets = [k for k in score_mom.keys() if
                       score_mom[k][0].numberSuffix is None]  # the [0] because there are more than one parts
    measure_offsets.append(score.duration.quarterLength)
    return measure_offsets


def tabular2roman(tabular, score):
    """
    Convert from tabular format to rntxt format.
    Pay attention to the measure numbers because there are three conventions at play:
      - for python, every list or array is 0-indexed
      - for music21, measures in a score are always 1-indexed
      - for rntxt, measures are 0-indexed if there is anacrusis and 1-indexed if there is not
    We solve by moving everything to 0-indexed and adjusting the rntxt output in the retrieve_measure_and_beat function

    Similarly we do for the beat, which is 1-indexed in music21 and in rntxt but which is mathematically
    more comfortable if 0-indexed. We convert to 0-indexed and then back at the end.

    :param tabular:
    :param score:
    :return:
    """
    rn = []  # store the final result

    # (starting beat == 0) <=> no anacrusis
    starting_beat = score.flat.getTimeSignatures()[0].beat - 1  # Convert beats to zero index
    measure_offsets = _get_measure_offsets(score)

    # Convert measure numbers given by music21 from 1-indexed to 0-indexed
    ts_list = list(score.flat.getTimeSignatures())
    time_signatures = dict([(ts.measureNumber - 1, ts) for ts in ts_list])
    ts_measures = sorted(time_signatures.keys())

    # ts_offsets = _get_ts_offsets(measure_offsets, ts_measures, starting_beat > 1)
    ts_offsets = [measure_offsets[m] for m in ts_measures]

    previous_measure, previous_end, previous_key = -1, 0, None
    j = 0

    for row in tabular:
        start, end, key, degree, quality, inversion = row
        num, den = interpret_degree(degree)
        chord = decode_roman(num, den, str(quality), str(inversion))
        key = key.replace('-', 'b')
        annotation = f'{key}: {chord}' if key != previous_key else chord

        # Time signature change
        while j < len(ts_offsets) and start >= ts_offsets[j]:
            rn.append('')
            rn.append(f"Time Signature : {time_signatures[ts_measures[j]].ratioString}")
            rn.append('')
            j += 1

        # No chords passage
        if start != previous_end:
            m, b = _retrieve_measure_and_beat(end, measure_offsets, time_signatures, ts_measures, starting_beat)
            if m == previous_measure:
                rn[-1] = _get_rn_row([m, b, ''], in_row=rn[-1])
            else:
                rn.append(_get_rn_row([m, b, '']))
            previous_measure = m

        m, b = _retrieve_measure_and_beat(start, measure_offsets, time_signatures, ts_measures, starting_beat)
        if m == previous_measure:
            rn[-1] = _get_rn_row([m, b, annotation], in_row=rn[-1])
        else:
            rn.append(_get_rn_row([m, b, annotation]))
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
    folder = "/home/gianluca/PycharmProjects/functional-harmony/data/OpenScore-LiederCorpus/scores/Chaminade,_Cécile/_/Amour_d'automne/"
    convert_file(folder + 'lc4999304.mxl', folder + 'automatic.csv', folder + 'test.txt')
    #
    # folder = "/home/gianluca/PycharmProjects/functional-harmony/data/OpenScore-LiederCorpus/scores/Chaminade,_Cécile/_/Amoroso/"
    # convert_file(folder + 'lc4999292.mxl', folder+'automatic.csv', folder+'test.txt')
    #

    folder = '/home/gianluca/PycharmProjects/functional-harmony/data/BPS/'
    convert_file(folder + 'scores/bps_08_01.mxl', folder + 'chords/bps_08_01.csv',
                 folder + 'txt_generated/bps_08_01_test.txt')

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
        # convert_corpus(base_folder, c)
        pass

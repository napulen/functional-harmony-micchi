"""
Converts music21 Roman text files into the BPS-FH format.
Work in progress.

ISSUES / TODO:
- usual problems with 6 & 7 in minor. See '# TODO: better solution to this'
- range of chord 'quality' types. Check if any are still returning '?????'?
- Extremely dodgy solution to unrecognised rns. Seems to work for now though.

Should now be sorted:
- Format of beats (not fractions);
- Discrepancy between start/end of score/analysis;
"""

import os
import csv

import numpy as np
from music21 import converter, roman, pitch


# ------------------------------------------------------------------------------

def roman2bps(analysis, score, test):
    in_data = analysis.recurse().getElementsByClass('RomanNumeral')

    out_data = []

    initial_beat_length = score.recurse().stream().getTimeSignatures()[0].beatDuration.quarterLength

    score_mom = score.measureOffsetMap()
    # consider only measures that have not been marked as "excluded" in the musicxml (for example using Musescore)
    score_measure_offset = [k for k in score_mom.keys() if
                            score_mom[k][0].numberSuffix is None]  # the [0] because there are two parts
    score_measure_offset.append(score.duration.quarterLength)

    if test:
        test_time_signatures(analysis, score, score_measure_offset)

    measure_zero = (in_data[0].measureNumber == 0)
    current_label = None
    start_offset = 0
    N = len(in_data)
    for n, x in enumerate(in_data):
        key = x.key.tonicPitchNameWithCase
        degree = get_degree(x)
        quality = get_quality(x)
        inversion = x.inversion()
        new_label = [key, degree, quality, inversion]

        if current_label is None:
            current_label = new_label
        if np.any(new_label != current_label):
            _, end_offset = find_offset(in_data[n - 1], score_measure_offset, initial_beat_length, measure_zero)
            out_data.append([round(start_offset, 3), round(end_offset, 3), *current_label])
            start_offset = end_offset
            current_label = new_label
        if n == N - 1:
            _, end_offset = find_offset(in_data[n], score_measure_offset, initial_beat_length, measure_zero)
            out_data.append([round(start_offset, 3), round(end_offset, 3), *current_label])

    # Check end of piece. NB AFTER adjusting start
    end_of_analysis = out_data[-1][1]
    end_of_piece = score.duration.quarterLength

    if end_of_analysis != end_of_piece:
        print(f'Reamining gap: {end_of_piece - end_of_analysis}\n'
              f'if > 0, the score is longer than the analysis, which could be due to the final chord lasting several measures')
        out_data[-1][1] = end_of_piece

    return out_data


def test_time_signatures(analysis, score, score_measure_offset):
    """
    Test script that checks if the measure length in the score aligns with the time signature changes in the rntxt file.
    """
    score_measure_length = np.diff(score_measure_offset)
    n_score_measures = len(score_measure_length)
    rn_measure_offset = list(analysis.measureOffsetMap().keys())
    rn_measure_length = np.diff(rn_measure_offset)
    rn_measure_length = np.append(rn_measure_length, score.duration.quarterLength - rn_measure_offset[-1])
    n_measures = len(rn_measure_length)
    score_time_change = []
    rn_time_change = []
    for i in range(len(rn_measure_length) - 2):  # remove the last measure, which often has weird length.
        if rn_measure_length[i + 1] != rn_measure_length[i]:
            rn_time_change.append(i + 1)  # +1 for indexing
    for i in range(len(score_measure_length) - 2):
        if score_measure_length[i + 1] != score_measure_length[i]:
            score_time_change.append(i + 1)  # +1 for indexing
    for ctc in rn_time_change:
        if ctc not in score_time_change:
            print(f'time signature in chords changes after measure {ctc} (1-indexed), but not in the score')
    for stc in score_time_change:
        if stc not in rn_time_change:
            print(f'time signature in score changes after measure {stc} (1-indexed), but not in the chords')
    return


def find_offset(rn, score_measure_offset, initial_beat_length, measure_zero):
    """
    Given a roman numeral element from an analysis parsed by music21, find its offset in quarter notes length.
    It automatically adapts to the presence of pickup measures thanks to the Boolean measure_zero.

    :param rn: a Roman Numeral chord, one element of an analysis rntxt file parsed by music21
    :param score_measure_offset: a list where element n gives the offset of measure n in quarter length
    :param initial_beat_length: Beat length in the first measure; e.g. if the piece starts in 4/4, ibl=1; if it starts in 12/8, ibl=1.5
    :param measure_zero: Boolean, True if there's a measure counted as zero (pickup measure)
    """
    measure = rn.measureNumber if measure_zero else rn.measureNumber - 1  # 0-indexed
    offset_in_measure = rn.offset
    if measure == 0:
        offset_in_measure -= - score_measure_offset[1] % initial_beat_length
    start_offset = float(score_measure_offset[measure] + offset_in_measure)
    duration = float(rn.quarterLength)
    end_offset = min(start_offset + duration, float(score_measure_offset[measure + 1]))
    return start_offset, end_offset


# ------------------------------------------------------------------------------

accidentalDict = {
    'double-sharp': '++',
    'sharp': '+',
    'natural': '',
    'flat': '-',
    'double-flat': '--'
}


def get_degree(x):
    # Whether there's secondary or otherwise
    degree = str(x.scaleDegreeWithAlteration[0])
    accidental = x.scaleDegreeWithAlteration[1]

    # TODO: This is a temporary hack because of a bug in music21 that assigns no accidental to the degree of aug6 chords
    augmented_sixths = ['German augmented sixth chord', 'French augmented sixth chord', 'Italian augmented sixth chord']
    if x.commonName in augmented_sixths:
        accidental = pitch.Accidental('sharp')
    # end of hack

    if accidental is not None:
        degree = accidentalDict[accidental.fullName] + degree

    # TODO: The hack for degree 7 is correct only if we assume that the leading tone is always there (safe assumption?)
    # Use harmonic scale for minor keys - case no secondary key
    if x.secondaryRomanNumeral is None:
        if x.key.mode == 'minor' and '7' in degree:
            degree = _lower_degree(degree)  # music21 uses natural scale, so that the leading tone is +7 instead of 7
    else:
        secondary_degree = str(x.secondaryRomanNumeral.scaleDegreeWithAlteration[0])
        secondary_accidental = x.secondaryRomanNumeral.scaleDegreeWithAlteration[1]
        if secondary_accidental is not None:
            secondary_degree = accidentalDict[secondary_accidental.fullName] + str(secondary_degree)

        if x.key.mode == 'minor' and '7' in secondary_degree:
            secondary_degree = _lower_degree(secondary_degree)

        if x.secondaryRomanNumeral.figure.islower() and '7' in degree:  # use the harmonic scale of the tonicised key
            degree = _lower_degree(degree)
        # # TODO: better solution to this
        # if secondary_degree == '+7':
        #     secondary_degree = '7'

        # if '6' in degree:
        #     print("hi")
        # if '++6' in secondary_degree:
        #     print('+6 in denominator')

        # Notice that music21 uses the more logical denomination of degree1 / degree2.
        degree = degree + '/' + secondary_degree

    return degree


def _raise_degree(deg):
    """ deg needs to be in a format where alterations precede the actual degree, and they are stored as - and + """
    return deg[1:] if deg[0] == '-' else '+' + deg


def _lower_degree(deg):
    """ deg needs to be in a format where alterations precede the actual degree, and they are stored as - and + """
    return deg[1:] if deg[0] == '+' else '-' + deg


# ------------------------------------------------------------------------------

qualityDict = {'major triad': 'M',
               'minor triad': 'm',
               'diminished triad': 'd',
               'augmented triad': 'a',

               'minor seventh chord': 'm7',
               'major seventh chord': 'M7',
               'dominant seventh chord': 'D7',
               'incomplete dominant-seventh chord': 'D7',  # For '75' and '73'
               'diminished seventh chord': 'd7',
               'half-diminished seventh chord': 'h7',

               'augmented sixth': 'a6',  # TODO: This should never happen!!
               'German augmented sixth chord': 'Gr+6',
               'French augmented sixth chord': 'Fr+6',
               'Italian augmented sixth chord': 'It+6',
               'minor-augmented tetrachord': 'm',  # I know, but we have to stay consistent with BPS-FH ...
               # 'Neapolitan chord': 'N6'  # N/A: major triad  TODO: Add support to Neapolitan chords?
               }


def get_quality(x):
    if x.commonName in [x for x in qualityDict.keys()]:
        quality = qualityDict[x.commonName]

    # TODO compress
    elif '[' in x.figure:  # Retrieve quality from figure sans addition
        fig = str(x.figure)
        fig = fig.split('[')[0]
        rn = roman.RomanNumeral(fig, x.key)
        quality = get_quality(rn)
        # quality = qualityDict[rn.commonName]

    elif 'Fr' in x.figure:
        quality = 'Fr+6'
    elif 'Ger' in x.figure:
        quality = 'Gr+6'
    elif 'It' in x.figure:
        quality = 'It+6'
    elif '9' in x.figure:
        quality = 'D7'  # Setting all 9ths as dominants. Not including 9ths in this dataset

    elif len(str(x.figure)) > 0:  # TODO this is especially dodgy and risky ****
        fig = str(x.figure)[:-1]
        # print(x.figure, fig, x.measureNumber)
        rn = roman.RomanNumeral(fig, x.key)
        quality = get_quality(rn)
        # quality = qualityDict[rn.commonName]

    else:
        quality = '?????'
        print(f'Issue with chord quality for chord {x.figure} at measure {x.measureNumber}')

    return quality


def write_csv(data, out_path):
    with open(out_path, 'w') as fp:
        w = csv.writer(fp)
        w.writerows(data)
    return


# ------------------------------------------------------------------------------

def convert_file(score_path, txt_path, csv_path, test=False):
    score = converter.parse(score_path)
    analysis = converter.parse(txt_path, format='romanText')
    data = roman2bps(analysis, score, test)
    write_csv(data, csv_path)
    return


def convert_corpus(base_folder, corpus):
    txt_folder = os.path.join(base_folder, corpus, 'txt')
    score_folder = os.path.join(base_folder, corpus, 'scores')
    csv_folder = os.path.join(base_folder, corpus, 'chords')
    os.makedirs(csv_folder, exist_ok=True)

    file_list = []
    for txt_file in os.listdir(txt_folder):
        if txt_file.endswith('.txt'):
            file_list.append(txt_file)

    file_list = sorted(file_list)
    test = True
    for txt_file in file_list:
        if 'op18_no6_mov4' not in txt_file:
            continue
        print(txt_file)
        score_file = f'{txt_file.split("_")[0]}.mxl' if 'Tavern' in corpus else f'{txt_file[:-4]}.mxl'
        csv_file = f'{txt_file[:-4]}.csv'
        convert_file(os.path.join(score_folder, score_file),
                     os.path.join(txt_folder, txt_file),
                     os.path.join(csv_folder, csv_file),
                     test)


# ------------------------------------------------------------------------------

# One file

# file = 'op18_no3_mov3.txt'
# analysis = converter.parse(txtPath + file, format='romanText')
# # thisScore = converter.parse(f'{scorePath}{file[:-4]}.mxl')
# data = roman2bps(analysis)#, scoreForComparison=thisScore)
# writeCSV(data, csvPath, file[:-4])

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    base_folder = os.path.join('..', 'data')

    corpora = [
        os.path.join('Tavern', 'Beethoven'),
        os.path.join('Tavern', 'Mozart'),
        'Bach_WTC_1_Preludes',
        '19th_Century_Songs',
        'Beethoven_4tets',
    ]

    for c in corpora:
        convert_corpus(base_folder, corpora[4])

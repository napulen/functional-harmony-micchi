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
import unittest

import numpy as np
from music21 import converter, roman


# ------------------------------------------------------------------------------

def roman2bps(analysis, score, test):
    inData = analysis.recurse().getElementsByClass('RomanNumeral')

    outData = []

    initialBeatLength = score.recurse().stream().getTimeSignatures()[0].beatDuration.quarterLength  # Is this always 1?
    # scoreMeasureOffset = list(score.measureOffsetMap().keys())
    scoreMOM = score.measureOffsetMap()
    scoreMeasureOffset = [k for k in scoreMOM.keys() if
                          scoreMOM[k][0].numberSuffix is None]  # the [0] because there are two parts
    scoreMeasureOffset.append(score.duration.quarterLength)

    if test:
        scoreMeasureLength = np.diff(scoreMeasureOffset)
        nScoreMeasures = len(scoreMeasureLength)

        measureOffset = list(analysis.measureOffsetMap().keys())
        measureLength = np.diff(measureOffset)
        measureLength = np.append(measureLength, score.duration.quarterLength - measureOffset[-1])
        nMeasures = len(measureLength)

        scoreTimeChange = []
        timeChange = []
        for i in range(len(measureLength) - 2):  # remove the last measure, which often has weird length.
            if measureLength[i + 1] != measureLength[i]:
                timeChange.append(i + 1)  # +1 for indexing
        for i in range(len(scoreMeasureLength) - 2):
            if scoreMeasureLength[i + 1] != scoreMeasureLength[i]:
                scoreTimeChange.append(i + 1)  # +1 for indexing
        for ctc in timeChange:
            if ctc not in scoreTimeChange:
                print(f'time signature in chords changes after measure {ctc} (1-indexed), but not in the score')
        for stc in scoreTimeChange:
            if stc not in timeChange:
                print(f'time signature in score changes after measure {stc} (1-indexed), but not in the chords')

    flag = False
    measureZero = False
    for x in inData:
        if x.measureNumber == 0:
            measureZero = True
        measure = x.measureNumber if measureZero else x.measureNumber - 1  # 0-indexed
        offsetInMeasure = x.offset
        if measure == 0:
            offsetInMeasure -= - scoreMeasureOffset[1] % initialBeatLength
        startOffset = float(scoreMeasureOffset[measure] + offsetInMeasure)
        duration = float(x.quarterLength)

        endOffset = min(startOffset + duration, float(scoreMeasureOffset[measure + 1]))

        key = x.key.tonicPitchNameWithCase

        degree = getDegree(x)

        quality = getQuality(x)

        inversion = x.inversion()

        outData.append([round(startOffset, 3), round(endOffset, 3), key, degree, quality, inversion])

        if flag:
            if outData[-2][1] > outData[-1][0]:
                print(f'Hey, we got a problem here: {outData[-2], outData[-1]}')
        flag = True

    # Check end of piece. NB AFTER adjusting start
    endOfAnalysis = outData[-1][1]
    endOfPiece = score.duration.quarterLength

    if endOfAnalysis != endOfPiece:
        print(f'Reamining gap: {endOfPiece - endOfAnalysis}\n'
              f'if > 0, the score is longer than the analysis, which could be due to the final chord lasting several measures')
        outData[-1][1] = endOfPiece

    return outData


# ------------------------------------------------------------------------------

accidentalDict = {
    'double-sharp': '++',
    'sharp': '+',
    'natural': '',
    'flat': '-',
    'double-flat': '--'
}


def getDegree(x):
    # Whether there's secondary or otherwise
    degree = str(x.scaleDegreeWithAlteration[0])
    accidental = x.scaleDegreeWithAlteration[1]
    if accidental:
        degree = accidentalDict[accidental.fullName] + degree

    # TODO: better solution to this
    if degree == '+7':
        degree = '7'
    # if '6' in degree:
    #     print("hi")
    if '++6' in degree:
        print("+6 in numerator")

    # With secondary
    if x.secondaryRomanNumeral:  # NB do not use if '/' in x.figure:
        secondaryDegree = str(x.secondaryRomanNumeral.scaleDegreeWithAlteration[0])
        secondaryAccidental = x.secondaryRomanNumeral.scaleDegreeWithAlteration[1]
        if secondaryAccidental:
            secondaryDegree = accidentalDict[secondaryAccidental.fullName] + str(secondaryDegree)

        # TODO: better solution to this
        if secondaryDegree == '+7':
            secondaryDegree = '7'

        # if '6' in degree:
        #     print("hi")
        if '++6' in secondaryDegree:
            print('+6 in denominator')

        degree = degree + '/' + secondaryDegree

    return degree


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

               'augmented sixth': 'a6',
               'German augmented sixth chord': 'Gr+6',
               'French augmented sixth chord': 'Fr+6',
               'Italian augmented sixth chord': 'It+6',
               'minor-augmented tetrachord': 'm',  # I know, but we have to stay consisten with BPS-FH ...
               # 'Neapolitan chord': 'N6'  # N/A: major triad
               }


def getQuality(x):
    if x.commonName in [x for x in qualityDict.keys()]:
        quality = qualityDict[x.commonName]

    # TODO compress
    elif '[' in x.figure:  # Retrieve quality from figure sans addition
        fig = str(x.figure)
        fig = fig.split('[')[0]
        rn = roman.RomanNumeral(fig, x.key)
        quality = getQuality(rn)
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
        quality = getQuality(rn)
        # quality = qualityDict[rn.commonName]

    else:
        quality = '?????'
        print(f'Issue with chord quality for chord {x.figure} at measure {x.measureNumber}')

    return quality


def writeCSV(data, outPath, file):
    # Make the csv
    csv = open(os.path.join(outPath, f'{file}.csv'), "w")
    for entry in data:
        line = [str(x) for x in entry]
        line = ','.join(line) + '\n'
        csv.write(line)
    csv.close()


# ------------------------------------------------------------------------------

class Test(unittest.TestCase):
    """
    Test two cases: full and partial analyses.
    """

    def testCorpus(self, score=True):

        base = os.path.join('..', 'data')

        # corpus = os.path.join('Tavern', 'Beethoven')
        # corpus = os.path.join('Tavern', 'Mozart')
        corpus = 'Bach_WTC_1_Preludes'
        # corpus = '19th_Century_Songs'
        # corpus = 'Beethoven_4tets/'

        # txtPath = f'{base}{corpus}/temp/'
        txtPath = os.path.join(base, corpus, 'txt')
        scorePath = os.path.join(base, corpus, 'scores')
        csvPath = os.path.join(base, corpus, 'chords')
        os.makedirs(csvPath, exist_ok=True)

        fileList = []
        for file in os.listdir(txtPath):
            if file.endswith('.txt'):
                fileList.append(file)

        fileList = sorted(fileList)
        test = True
        for file in fileList:  # [0:2]:
            # op = file.split('_')[0][2:]
            # no = file.split('_')[1][2:]
            # mv = file.split('_')[2][3]
            # if op != '18' or no != '6' or mv != '4':
            #     test = True
            #     continue
            # if 'K398' not in file:
            #     continue
            # print(f'====== Op. {op} No. {no} mov {mv} ======')
            # sf = f'op. {op} No. {no}'
            # score = converter.parse(os.path.join(scorePath, sf, f'{file[:-4]}.mxl'))
            print(file)
            sf = f'{file.split("_")[0]}.mxl' if 'Tavern' in corpus else f'{file[:-4]}.mxl'
            score = converter.parse(os.path.join(scorePath, sf))
            analysis = converter.parse(os.path.join(txtPath, file), format='romanText')
            data = roman2bps(analysis, score, test)
            writeCSV(data, csvPath, file[:-4])


# ------------------------------------------------------------------------------

# One file

# file = 'op18_no3_mov3.txt'
# analysis = converter.parse(txtPath + file, format='romanText')
# # thisScore = converter.parse(f'{scorePath}{file[:-4]}.mxl')
# data = roman2bps(analysis)#, scoreForComparison=thisScore)
# writeCSV(data, csvPath, file[:-4])

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()

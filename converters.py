"""
Ideally, all converters should be subclasses of a general Converter class. Maybe implement it someday?
"""
import csv
import logging
import os
from abc import ABC, abstractmethod

import music21
import numpy as np

from utils import int_to_roman, decode_roman


class AnnotationConverter(ABC):

    def __init__(self, in_ext=None, out_ext=None):
        self.in_ext = in_ext
        self.out_ext = out_ext
        self.logger = logging.getLogger("converter")
        self.logger.setLevel('INFO')
        return

    @abstractmethod
    def _load_input(self, in_path):
        pass

    @abstractmethod
    def run(self, in_data, score):
        pass

    @abstractmethod
    def _write_output(self, out_data, out_path):
        pass

    @staticmethod
    def _get_measure_offsets(score):
        """
        The measure_offsets are zero-indexed: the first measure in the score will be at index zero, regardless of anacrusis.

        :param score:
        :return: a list where at index m there is the offset in quarter length of measure m
        """
        score_mom = score.measureOffsetMap()
        # consider only measures that have not been marked as "excluded" in the musicxml (for example using Musescore)
        # the [0] because there might be more than one parts (e.g. piano right and left hand, other instruments, etc.)
        measure_offsets = [k for k in score_mom.keys() if score_mom[k][0].numberSuffix is None]
        measure_offsets.append(score.duration.quarterLength)
        return measure_offsets

    def convert_file(self, score_path, in_path, out_path):
        """
        Convert a file in in_path and store it in out_path, using information contained in the score

        :param score_path:
        :param in_path:
        :param out_path:
        :return:
        """
        self.logger.info(f"Converting file {in_path} to {out_path}")
        score = music21.converter.parse(score_path)
        in_data = self._load_input(in_path)
        out_data = self.run(in_data, score)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self._write_output(out_data, out_path)
        return

    # TODO Put an "experimental" decorator here
    def convert_corpus(self, corpus_id, score_folder, in_folder, out_folder):
        """

        :param corpus_id:
        :param score_folder:
        :param in_folder:
        :param out_folder:
        :return:
        """
        os.makedirs(out_folder, exist_ok=True)

        dir_entries = [x for x in os.scandir(in_folder) if x.isfile and x.name.endswith(self.in_ext)]

        for in_file in dir_entries:
            print(in_file.name)
            score_file = f'{in_file.name[:-6]}.mxl' if 'Tavern' in corpus_id else f'{in_file.name[:-4]}.mxl'
            out_file = '.'.join([in_file.name[:-4], self.out_ext])
            self.convert_file(os.path.join(score_folder, score_file), in_file.path, os.path.join(out_folder, out_file))
        return


class ConverterRn2Tab(AnnotationConverter):
    def __init__(self):
        super().__init__(in_ext='csv', out_ext='txt')
        self.qualityDict = {
            'major triad': 'M',
            'minor triad': 'm',
            'diminished triad': 'd',
            'augmented triad': 'a',

            'minor seventh chord': 'm7',
            'major seventh chord': 'M7',
            'dominant seventh chord': 'D7',
            'incomplete dominant-seventh chord': 'D7',  # For '75' and '73'
            'diminished seventh chord': 'd7',
            'half-diminished seventh chord': 'h7',

            'augmented sixth': 'a6',  # This should never happen!!
            'German augmented sixth chord': 'Gr+6',
            'French augmented sixth chord': 'Fr+6',
            'Italian augmented sixth chord': 'It+6',
            'minor-augmented tetrachord': 'm',  # I know, but we have to stay consistent with BPS-FH ...

            # 'Neapolitan chord': 'N6'  # N/A: major triad  TODO: Add support to Neapolitan chords?
        }
        self.accidentalDict = {
            'double-sharp': '++',
            'sharp': '+',
            'natural': '',
            'flat': '-',
            'double-flat': '--'
        }
        self.augmented_sixths = [
            'German augmented sixth chord',
            'French augmented sixth chord',
            'Italian augmented sixth chord'
        ]

    def _get_full_degree(self, rn):
        def _lower_degree(deg):
            return deg[1:] if deg[0] == '+' else '-' + deg

        def _degree_str_from_rn(rn):
            deg = str(rn.scaleDegreeWithAlteration[0])
            acc = rn.scaleDegreeWithAlteration[1]

            # TODO: This is a temporary hack because a bug in music21 assigns no accidental to the degree of aug6 chords
            if rn.commonName in self.augmented_sixths:
                deg = '4'
                acc = music21.pitch.Accidental('sharp')
            # end of hack

            if acc is not None:
                deg = self.accidentalDict[acc.fullName] + deg
            return deg

        degree = _degree_str_from_rn(rn)

        # music21 uses natural scales while we want to use harmonic scale for minor keys, we deal with it here
        if rn.secondaryRomanNumeral is not None:
            secondary_degree = _degree_str_from_rn(rn.secondaryRomanNumeral)

            if rn.key.mode == 'minor' and '7' in secondary_degree:  # use the harmonic scale on the base key
                secondary_degree = _lower_degree(secondary_degree)

            if rn.secondaryRomanNumeralKey.mode == 'minor' and '7' in degree:  # use the harmonic scale of the tonicised key
                degree = _lower_degree(degree)

            # Notice that music21 uses the (more logical?) naming convention degree1 / degree2.
            degree = degree + '/' + secondary_degree
        else:
            if rn.key.mode == 'minor' and '7' in degree:
                degree = _lower_degree(degree)

        return degree

    def _get_quality(self, rn):
        if rn.commonName in self.qualityDict.keys():
            quality = self.qualityDict[rn.commonName]

        # TODO compress
        elif '[' in rn.figure:  # Retrieve quality from figure sans addition
            fig = str(rn.figure)
            fig = fig.split('[')[0]
            rn = music21.roman.RomanNumeral(fig, rn.key)
            quality = self._get_quality(rn)
            # quality = qualityDict[rn.commonName]
        elif '9' in rn.figure:
            quality = 'D7'  # Setting all 9ths as dominants. Not including 9ths in this dataset
        elif 'Fr' in rn.figure:
            quality = 'Fr+6'
        elif 'Ger' in rn.figure:
            quality = 'Gr+6'
        elif 'It' in rn.figure:
            quality = 'It+6'
        # TODO Document well the behavior on ninths

        # elif len(str(x.figure)) > 0:  # TODO this is especially dodgy and risky ****
        #     fig = str(x.figure)[:-1]
        #     # print(x.figure, fig, x.measureNumber)
        #     rn = music21.roman.RomanNumeral(fig, x.key)
        #     quality = self.get_quality(rn)
        #     # quality = qualityDict[rn.commonName]
        #
        else:
            quality = '?????'
            self.logger.warning(f'Issue with chord quality for chord {rn.figure} at measure {rn.measureNumber}')

        return quality

    def _find_offset(self, rn, score_measure_offset, initial_beat_length, measure_zero):
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
        # TODO: check if we really need to stop the offset at the beginning of the next measure
        # Normally, this should be already implied in the parsing of the rntxt files
        end_offset = min(start_offset + duration, float(score_measure_offset[measure + 1]))
        return round(start_offset, 3), round(end_offset, 3)

    def _write_output(self, out_data, out_path):
        with open(out_path, 'w') as fp:
            w = csv.writer(fp)
            w.writerows(out_data)
        return

    def _load_input(self, txt_path):
        """
        Load the rntxt analysis file into a music21 object
        :param txt_path: the path to the rntxt file with the harmonic analysis
        :return:
        """
        analysis = music21.converter.parse(txt_path, format='romanText')
        # We need to keep the recurse so that we have the offset inside the measure without further calculations
        return analysis.recurse().getElementsByClass('RomanNumeral')

    def run(self, rntxt, score):
        """
        Convert from rntxt format to tabular format.
        Pay attention to the measure numbers because there are three conventions at play:
          - for python, every list or array is 0-indexed
          - for music21, measures in a score are always 1-indexed
          - for rntxt, measures are 0-indexed if there is anacrusis and 1-indexed if there is not
        We solve by moving everything to 0-indexed and adjusting the rntxt output in the retrieve_measure_and_beat function

        Similarly we do for the beat, which is 1-indexed in music21 and in rntxt but which is mathematically
        more comfortable if 0-indexed. We convert to 0-indexed and then back at the end.

        :param rntxt:
        :param score:
        :return:
        """

        def _get_rn_attributes(rn):
            key = rn.key.tonicPitchNameWithCase
            degree = self._get_full_degree(rn)
            quality = self._get_quality(rn)
            inversion = min(rn.inversion(), 3)  # fourth inversions on ninths are sent to 3 arbitrarily
            # TODO: Document our behavior with ninths
            return [key, degree, quality, inversion]

        def _correct_final_offset_inplace(out_data, score):
            """
            rntxt files don't repeat the last chord indefinitely, but only give it once.
            This can cause problems because the total length of the score and of the analysis differ.
            For tabular representation we need to refer to the total length of the score, so we modify the
            last end_offset to reflect that.
            :param out_data:
            :param score:
            :return:
            """
            end_of_analysis = out_data[-1][1]
            end_of_piece = score.duration.quarterLength

            if end_of_analysis != end_of_piece:
                self.logger.warning(
                    f'A remaining gap of {end_of_piece - end_of_analysis} quarter lengths has been closed.\n'
                    f'If > 0, it means that the score is longer than the analysis, which could be due to the final chord lasting several measures.'
                )
            out_data[-1][1] = end_of_piece

        out_data = []
        measure_offsets = self._get_measure_offsets(score)

        initial_beat_length = score.flat.getTimeSignatures()[0].beatDuration.quarterLength
        measure_zero = (rntxt[0].measureNumber == 0)  # notice that rntxt can have measureNumber == 0, unlike scores

        current_rn, current_label = None, None
        start_offset = 0.
        for new_rn in rntxt:
            new_label = _get_rn_attributes(new_rn)
            if current_rn is None:  # initialize the system
                current_rn = new_rn
                current_label = _get_rn_attributes(current_rn)

            if any([n != c for n, c in zip(new_label, current_label)]):
                _, end_offset = self._find_offset(current_rn, measure_offsets, initial_beat_length, measure_zero)
                out_data.append([start_offset, end_offset, *current_label])
                start_offset, current_rn, current_label = end_offset, new_rn, new_label

            else:  # update the offsets
                current_rn = new_rn

        # write the last chord
        _, end_offset = self._find_offset(current_rn, measure_offsets, initial_beat_length, measure_zero)
        out_data.append([start_offset, end_offset, *current_label])

        _correct_final_offset_inplace(out_data, score)

        return out_data


class ConverterTab2Rn(AnnotationConverter):
    def __init__(self):
        super().__init__(in_ext='csv', out_ext='txt')
        self.datatype_chord = [
            ('onset', 'float'),
            ('end', 'float'),
            ('key', '<U10'),
            ('degree', '<U10'),
            ('quality', '<U10'),
            ('inversion', 'int')
        ]

    def _get_rn_row(self, datum, in_row=None):
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

    def _retrieve_measure_and_beat(self, offset, measure_offsets, time_signatures, ts_measures, beat_zero):
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

    def interpret_degree(self, degree):
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

    def _write_output(self, out_data, out_path):
        with open(out_path, 'w') as f:
            for row in out_data:
                f.write(row + os.linesep)
        return

    def _load_input(self, csv_path):
        """
        Load chords of each piece and add chord symbols into the labels.
        :param csv_path: the path to the file with the harmonic analysis
        :return: chord_labels, an array of tuples (start, end, key, degree, quality, inversion)
        """

        chords = []
        with open(csv_path, mode='r') as f:
            data = csv.reader(f)
            for row in data:
                chords.append(tuple(row))
        return np.array(chords, dtype=self.datatype_chord)

    def run(self, tabular, score):
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
        out_data = []
        measure_offsets = self._get_measure_offsets(score)

        # Convert measure numbers given by music21 from 1-indexed to 0-indexed
        ts_list = list(score.flat.getTimeSignatures())  # we need to call flat to create measure numbers
        time_signatures = dict([(ts.measureNumber - 1, ts) for ts in ts_list])
        ts_measures = sorted(time_signatures.keys())
        ts_offsets = [measure_offsets[m] for m in ts_measures]

        # (starting beat == 0) <=> no anacrusis
        starting_beat = ts_list[0].beat - 1  # Convert beats to zero index

        previous_measure, previous_end, previous_key = -1, 0, None
        ts_index = 0  # this runs over the time signatures to find where to put them in the output

        for row in tabular:
            start, end, key, degree, quality, inversion = row
            num, den = self.interpret_degree(degree)
            chord = decode_roman(num, den, str(quality), str(inversion))
            key = key.replace('-', 'b')
            annotation = f'{key}: {chord}' if key != previous_key else chord

            # Was there a time signature change?
            while ts_index < len(ts_offsets) and start >= ts_offsets[ts_index]:
                out_data.append('')
                out_data.append(f"Time Signature : {time_signatures[ts_measures[ts_index]].ratioString}")
                out_data.append('')
                ts_index += 1

            # Was there a no-chord passage in between?
            if start != previous_end:
                m, b = self._retrieve_measure_and_beat(end, measure_offsets, time_signatures, ts_measures,
                                                       starting_beat)
                if m == previous_measure:
                    out_data[-1] = self._get_rn_row([m, b, ''], in_row=out_data[-1])
                else:
                    out_data.append(self._get_rn_row([m, b, '']))
                previous_measure = m

            # Standard annotation conversion
            m, b = self._retrieve_measure_and_beat(start, measure_offsets, time_signatures, ts_measures, starting_beat)
            if m == previous_measure:
                out_data[-1] = self._get_rn_row([m, b, annotation], in_row=out_data[-1])
            else:
                out_data.append(self._get_rn_row([m, b, annotation]))
            previous_measure, previous_end, previous_key = m, end, key

        return out_data


if __name__ == '__main__':
    t2r = ConverterTab2Rn()
    sp = '/home/gianluca/PycharmProjects/functional-harmony/data/Bach_WTC_1_Preludes/scores/wtc_i_prelude_01.mxl'
    ip = '/home/gianluca/PycharmProjects/functional-harmony/data/Bach_WTC_1_Preludes/chords/wtc_i_prelude_01.csv'
    op = '/home/gianluca/PycharmProjects/functional-harmony/data/Bach_WTC_1_Preludes/txt_generated/wtc_i_prelude_01.txt2'
    t2r.convert_file(sp, ip, op)

    r2t = ConverterRn2Tab()
    sp = '/home/gianluca/PycharmProjects/functional-harmony/data/Bach_WTC_1_Preludes/scores/wtc_i_prelude_01.mxl'
    ip = '/home/gianluca/PycharmProjects/functional-harmony/data/Bach_WTC_1_Preludes/txt/wtc_i_prelude_01.txt'
    op = '/home/gianluca/PycharmProjects/functional-harmony/data/Bach_WTC_1_Preludes/chords_generated/wtc_i_prelude_01b.csv'
    r2t.convert_file(sp, ip, op)

    sp = '/home/gianluca/PycharmProjects/functional-harmony/data/Beethoven_4tets/scores/op18_no2_mov2.mxl'
    ip = '/home/gianluca/PycharmProjects/functional-harmony/data/Beethoven_4tets/txt/op18_no2_mov2.txt'
    op = '/home/gianluca/PycharmProjects/functional-harmony/data/Beethoven_4tets/chords_generated/op18_no2_mov2.csv'
    r2t.convert_file(sp, ip, op)
"""
Ideally, all converters should be subclasses of a general Converter class. Maybe implement it someday?
"""
import csv
import json
import logging
import os
from abc import ABC, abstractmethod

import music21
import numpy as np


def roman_to_int(roman):
    r2i = {
        'I': 1,
        'II': 2,
        'III': 3,
        'IV': 4,
        'V': 5,
        'VI': 6,
        'VII': 7,
    }
    return r2i[roman.upper()]


def int_to_roman(n):
    """ Convert an integer to a Roman numeral. """

    if not 0 < n < 8:
        raise ValueError("Argument must be between 1 and 7")
    ints = (5, 4, 1)
    nums = ('V', 'IV', 'I')
    result = []
    for i in range(len(ints)):
        count = int(n / ints[i])
        result.append(nums[i] * count)
        n -= ints[i] * count
    return ''.join(result)


class AnnotationConverter(ABC):

    def __init__(self, in_ext=None, out_ext=None):
        self.in_ext = in_ext
        self.out_ext = out_ext
        self.logger = logging.getLogger("converter")
        self.logger.setLevel(logging.INFO)
        self.quality_music21_to_tab = {
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
        self.accidental_music21_to_tab = {
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
        self.inversion_tab2rn = {
            'triad0': '',
            'triad1': '6',
            'triad2': '64',
            'triad3': 'wi',
            'seventh0': '7',
            'seventh1': '65',
            'seventh2': '43',
            'seventh3': '42',
        }
        self.quality_tab2rn = {  # (True=uppercase degree, 'triad' or 'seventh', quality)
            'M': (True, 'triad', ''),
            'm': (False, 'triad', ''),
            'd': (False, 'triad', 'o'),
            'a': (True, 'triad', '+'),
            'M7': (True, 'seventh', 'M'),
            'm7': (False, 'seventh', ''),
            'D7': (True, 'seventh', ''),
            'd7': (False, 'seventh', 'o'),
            'h7': (False, 'seventh', 'ø'),
            'Gr+6': (True, 'seventh', 'Gr'),
            'It+6': (True, 'seventh', 'It'),
            'Fr+6': (True, 'seventh', 'Fr'),
        }

        return

    @staticmethod
    def _get_measure_offsets(score):
        """
        The measure_offsets are zero-indexed: the first measure in the score will be at index zero, regardless of anacrusis.
        This is implemented by keeping the order of the measure without looking at the keys.

        :param score:
        :return: a list where at index m there is the offset in quarter length of measure m
        """
        score_mom = score.measureOffsetMap()
        # consider only measures that have not been marked as "excluded" in the musicxml (for example using Musescore)
        # the [0] because there might be more than one parts (e.g. piano right and left hand, other instruments, etc.)
        measure_offsets = [k for k in score_mom.keys() if score_mom[k][0].numberSuffix is None]
        measure_offsets.append(score.duration.quarterLength)
        return measure_offsets

    def chord_tab_to_rn(self, degree, quality, inversion):
        """
        Given degree (numerator and denominator), quality of the chord, and inversion, return a properly written RN.

        :param num: String with the numerator of the degree in arab numerals, e.g. '1', or '+4'
        :param den: Same as num, but for the denominator
        :param quality: Quality of the chord (major, minor, dominant seventh, etc.)
        :param inversion: Inversion as a string
        """

        def interpret_degree(degree):
            """
            Given a degree written in our tabular format, split it into num + den in music21 format. Ex.:
              - interpret_degree('1') = ('I', 'I')
              - interpret_degree('----7/-3') = ('bbbbVII', 'bIII')
              - interpret_degree('+5/5') = ('#V', 'V')
            :param degree:
            :return:
            """
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
                den_prefix += 'b' if den[0] == '-' else '#'
                den = den[1:]
            den = den_prefix + int_to_roman(int(den[0]))

            return num, den

        num, den = interpret_degree(degree)
        upper, triad, qlt = self.quality_tab2rn[quality]
        inv = self.inversion_tab2rn[triad + inversion]
        if upper:
            num_prefix = ''
            while num[0] == 'b':
                num_prefix += num[0]
                num = num[1:]
            num = num_prefix + num.upper()
        else:
            num = num.lower()
        if num == 'IV' and qlt == 'M':  # the fourth degree is by default major seventh
            qlt = ''
        return num + qlt + inv + ('/' + den if den != 'I' else '')

    def chord_rn_to_tab(self, chord):
        rn = music21.roman.RomanNumeral(chord, "C")  # arbitrary key, we throw it away later anyway
        _, degree, quality, inversion = self.chord_music21_to_tab(rn)
        return degree, quality, inversion

    def chord_music21_to_tab(self, rn):
        def _get_full_degree(rn):
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
                    deg = self.accidental_music21_to_tab[acc.fullName] + deg
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

        def _get_quality(rn):
            if rn.commonName in self.quality_music21_to_tab.keys():
                quality = self.quality_music21_to_tab[rn.commonName]

            # TODO compress
            elif '[' in rn.figure:  # Retrieve quality from figure sans addition
                fig = str(rn.figure)
                fig = fig.split('[')[0]
                rn = music21.roman.RomanNumeral(fig, rn.key)
                quality = _get_quality(rn)
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

        key = rn.key.tonicPitchNameWithCase
        degree = _get_full_degree(rn)
        quality = _get_quality(rn)
        inversion = min(rn.inversion(), 3)  # fourth inversions on ninths are sent to 3 arbitrarily
        # TODO: Document our behavior with ninths
        return [key, degree, quality, inversion]

    @abstractmethod
    def load_input(self, in_path):
        pass

    @abstractmethod
    def run(self, in_data, score):
        pass

    @abstractmethod
    def write_output(self, out_data, out_path):
        pass

    def convert_file(self, score_path, in_path, out_path, **kwargs):
        """
        Convert a file in in_path and store it in out_path, using information contained in the score

        :param score_path:
        :param in_path:
        :param out_path:
        :return:
        """
        self.logger.info(f"Converting file {in_path} to {out_path}")
        score = music21.converter.parse(score_path)
        in_data = self.load_input(in_path)
        out_data, flag = self.run(in_data, score)
        if flag:
            print(f"{score_path}")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self.write_output(out_data, out_path)
        return

    # TODO Put an "experimental" decorator here
    def convert_corpus(self, corpus_id, score_folder, in_folder, out_folder, **kwargs):
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
            self.convert_file(os.path.join(score_folder, score_file), in_file.path, os.path.join(out_folder, out_file),
                              **kwargs)
        return


class ConverterRn2Tab(AnnotationConverter):
    def __init__(self):
        super().__init__(in_ext='csv', out_ext='txt')

    def write_output(self, out_data, out_path):
        with open(out_path, 'w') as fp:
            w = csv.writer(fp)
            w.writerows(out_data)
        return

    def load_input(self, txt_path):
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

        def _find_offset(rn, score_measure_offset, initial_beat_length, measure_zero):
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
            start = float(score_measure_offset[measure] + offset_in_measure)
            duration = float(rn.quarterLength)
            # TODO: check if we really need to stop the offset at the beginning of the next measure
            # Normally, this should be already implied in the parsing of the rntxt files
            end = min(start + duration, float(score_measure_offset[measure + 1]))
            return round(start, 3), round(end, 3)

        def _correct_final_offset_inplace(out_data, score):
            """
            rntxt files don't repeat the last chord indefinitely, but only give it once.
            This can cause problems because the total length of the score and the analysis differ.
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
                    f'If > 0, it means that the score is longer than the analysis, '
                    f'which could be due to the final chord lasting several measures.'
                )
            out_data[-1][1] = end_of_piece
            return

        out_data = []
        flag = False
        measure_offsets = self._get_measure_offsets(score)

        initial_beat_length = score.flat.getTimeSignatures()[0].beatDuration.quarterLength
        measure_zero = (rntxt[0].measureNumber == 0)  # notice that rntxt can have measureNumber == 0, unlike scores

        current_rn, current_label = None, None
        start_offset = 0.
        for new_rn in rntxt:
            new_label = self.chord_music21_to_tab(new_rn)
            if current_rn is None:  # initialize the system
                current_rn = new_rn
                current_label = self.chord_music21_to_tab(current_rn)

            if any([n != c for n, c in zip(new_label, current_label)]):
                _, end_offset = _find_offset(current_rn, measure_offsets, initial_beat_length, measure_zero)
                out_data.append([start_offset, end_offset, *current_label])
                start_offset, current_rn, current_label = end_offset, new_rn, new_label

            else:  # update the offsets
                current_rn = new_rn

        # write the last chord
        _, end_offset = _find_offset(current_rn, measure_offsets, initial_beat_length, measure_zero)
        out_data.append([start_offset, end_offset, *current_label])

        _correct_final_offset_inplace(out_data, score)

        return out_data, flag


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

    def write_output(self, out_data, out_path):
        with open(out_path, 'w') as f:
            for row in out_data:
                f.write(row + os.linesep)
        return

    def load_input(self, csv_path):
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

        out_data = []
        flag = False  # the flag is used for debugging
        measure_offsets = self._get_measure_offsets(score)

        # Get time signature positions while converting measure numbers given by music21 to 0-indexed
        ts_list = list(score.flat.getTimeSignatures())  # we need to call flat to create measure numbers
        first_measure_number = 0 if any([ts.measureNumber == 0 for ts in ts_list]) else 1
        time_signatures = dict([(max(ts.measureNumber - first_measure_number, 0), ts) for ts in ts_list])
        ts_measures = sorted(time_signatures.keys())
        ts_offsets = [measure_offsets[m] for m in ts_measures]

        if any([ts.measureNumber == 0 for ts in ts_list]) and len(ts_list) > 3:  # debugging tool
            flag = True

        # (starting beat == 0) <=> no anacrusis
        starting_beat = ts_list[0].beat - 1  # Convert beats to zero index

        previous_measure, previous_end, previous_key = -1, 0, None
        ts_index = 0  # this runs over the time signatures to find where to put them in the output

        for row in tabular:
            start, end, key, degree, quality, inversion = row
            chord = self.chord_tab_to_rn(degree, str(quality), str(inversion))
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
                m, b = _retrieve_measure_and_beat(end, measure_offsets, time_signatures, ts_measures,
                                                       starting_beat)
                if m == previous_measure:
                    out_data[-1] = _get_rn_row([m, b, ''], in_row=out_data[-1])
                else:
                    out_data.append(_get_rn_row([m, b, '']))
                previous_measure = m

            # Standard annotation conversion
            m, b = _retrieve_measure_and_beat(start, measure_offsets, time_signatures, ts_measures, starting_beat)
            if m == previous_measure:
                out_data[-1] = _get_rn_row([m, b, annotation], in_row=out_data[-1])
            else:
                out_data.append(_get_rn_row([m, b, annotation]))
            previous_measure, previous_end, previous_key = m, end, key

        return out_data, flag


class ConverterTab2Dez(AnnotationConverter):
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

    def write_output(self, out_data, out_path):
        """

        :param out_data:
        :param out_path:
        :return:
        """
        annotation = {
            "labels": out_data
        }

        with open(out_path, 'w') as fp:
            json.dump(annotation, fp)

        return

    def load_input(self, csv_path):
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

    def run(self, tabular, score, layer="automated"):
        """
        Convert from tabular format to dezrann format.

        :param tabular:
        :param score: This is actually not used in this particular case.
        :param layer: This should be set either to "reference" or to "automated", according to the origin of the annotation
        :return:
        """

        def _get_keys(tabular):
            out_keys = []
            start_previous, end_previous, key_previous, end = None, None, None, None
            for row in tabular:
                start, end, key, degree, quality, inversion = row
                key = key.replace('-', 'b')
                if key != key_previous or start != end_previous:  # the key changes or no-chord passage
                    # TODO: no-chord does not necessarily mean no-key. How to do better?
                    if key_previous is not None:
                        out_keys.append([start_previous, end_previous, key_previous])
                    start_previous, end_previous, key_previous = start, end, key
                else:  # same key
                    end_previous = end

            if key_previous is not None:  # last element
                out_keys.append([start_previous, end, key_previous])

            return out_keys

        def _get_rn(tabular):
            out_rn = []
            start_previous, end_previous, key_previous, rn_previous, end = None, None, None, None, None
            for row in tabular:
                start, end, key, degree, quality, inversion = row
                rn = self.chord_tab_to_rn(degree, str(quality), str(inversion))
                if key != key_previous or rn != rn_previous or start != end_previous:  # the rn changes or no-chord passage
                    if rn_previous is not None:
                        out_rn.append([start_previous, end_previous, rn_previous])
                    start_previous, end_previous, key_previous, rn_previous = start, end, key, rn
                else:  # same rn
                    end_previous = end

            if rn_previous is not None:  # last element
                out_rn.append([start_previous, end, rn_previous])

            return out_rn

        out_data = []
        flag = False

        # The same key can be associated with multiple chords, and therefore multiple lines.
        # We don't want a label that is all broken in Dezrann, so we somehow have to compact the information.
        # We do it by doing two for loops, one to compact the information and one to finally write it out.
        # The same is done also for the RN chord symbols, even if we want a different line for every change in key or rn
        # In fact, this allows us to catch errors in the csv if the same chord is mistakenly broken in several lines
        # The following code is conceptually simple but not efficient at all, as it does 4 loops instead of 1:
        # Two for-loops are apparent below, and one each hides in _get_keys() and _get_rn()
        # However, it is so fast that I prefer to keep it like this, for the moment.
        for start, end, key in _get_keys(tabular):
            out_data.append({
                "type": 'Tonality',
                "layers": [layer],
                "start": start,
                "actual-duration": end - start,
                "tag": key,
            })
        for start, end, rn in _get_rn(tabular):
            out_data.append({
                "type": 'Harmony',
                "layers": [layer],
                "start": start,
                "actual-duration": end - start,
                "tag": rn,
            })

        return out_data, flag


class ConverterDez2Tab(AnnotationConverter):
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

    def write_output(self, out_data, out_path):
        with open(out_path, 'w') as fp:
            w = csv.writer(fp)
            w.writerows(out_data)
        return

    def load_input(self, dez_path):
        """
        Load the dezrann file containing the annotations
        """
        with open(dez_path, 'r') as f:
            data = json.load(f)
        return data

    def run(self, dezrann, score):
        """
        Convert from dezrann format to tabular format.

        :param dezrann:
        :param score: This is actually not used in this particular case.
        :return:
        """
        out_data = []
        flag = False

        for x in dezrann['labels']:
            if x['type'] != 'Harmony':
                continue
            start, duration, chord = x["start"], x["actual-duration"], x["tag"]
            end = start + duration
            degree, quality, inversion = self.chord_rn_to_tab(chord)
            out_data.append([start, end, degree, quality, inversion])

        i = 0
        for x in dezrann['labels']:
            if x['type'] != 'Tonality':
                continue
            start, duration, key = x["start"], x["actual-duration"], x["tag"]
            end = start + duration
            while i < len(out_data) and out_data[i][0] < end:  # out_data[i] is the start of the annotation
                out_data[i].insert(2, key)
                i += 1
        return out_data, flag


if __name__ == '__main__':
    c = ConverterTab2Dez()
    sp = '/home/gianluca/PycharmProjects/functional-harmony/data/Bach_WTC_1_Preludes/scores/wtc_i_prelude_01.mxl'
    ip = '/home/gianluca/PycharmProjects/functional-harmony/data/Bach_WTC_1_Preludes/chords/wtc_i_prelude_01.csv'
    op = '/home/gianluca/PycharmProjects/functional-harmony/data/Bach_WTC_1_Preludes/dezrann_generated/wtc_i_prelude_01.dez'
    c.convert_file(sp, ip, op)

    c = ConverterDez2Tab()
    sp = '/home/gianluca/PycharmProjects/functional-harmony/data/Bach_WTC_1_Preludes/scores/wtc_i_prelude_01.mxl'
    ip = '/home/gianluca/PycharmProjects/functional-harmony/data/Bach_WTC_1_Preludes/dezrann_generated/wtc_i_prelude_01.dez'
    op = '/home/gianluca/PycharmProjects/functional-harmony/data/Bach_WTC_1_Preludes/chords_generated/wtc_i_prelude_01b.csv'
    c.convert_file(sp, ip, op)

    t2r = ConverterTab2Rn()
    # sp = '/home/gianluca/PycharmProjects/functional-harmony/data/Bach_WTC_1_Preludes/scores/wtc_i_prelude_01.mxl'
    # ip = '/home/gianluca/PycharmProjects/functional-harmony/data/Bach_WTC_1_Preludes/chords/wtc_i_prelude_01.csv'
    # op = '/home/gianluca/PycharmProjects/functional-harmony/data/Bach_WTC_1_Preludes/txt_generated/wtc_i_prelude_01b.txt'
    # t2r.convert_file(sp, ip, op)

    sp = "/home/gianluca/PycharmProjects/functional-harmony/data/OpenScore-LiederCorpus/scores/Mahler,_Gustav/Lieder_eines_fahrenden_Gesellen/4_-_Die_zwei_blauen_Augen_von_meinem_Schatz/lc5026316.mxl"
    ip = "/home/gianluca/PycharmProjects/functional-harmony/data/OpenScore-LiederCorpus/scores/Mahler,_Gustav/Lieder_eines_fahrenden_Gesellen/4_-_Die_zwei_blauen_Augen_von_meinem_Schatz/automatic.csv"
    op = "/home/gianluca/PycharmProjects/functional-harmony/data/OpenScore-LiederCorpus/scores/Mahler,_Gustav/Lieder_eines_fahrenden_Gesellen/4_-_Die_zwei_blauen_Augen_von_meinem_Schatz/automatic2.txt"
    t2r.convert_file(sp, ip, op)

    sp = "/home/gianluca/PycharmProjects/functional-harmony/data/OpenScore-LiederCorpus/scores/Schumann,_Robert/Dichterliebe,_Op.48/01_-_Im_wunderschönen_Monat_Mai/lc4976777.mxl"
    ip = "/home/gianluca/PycharmProjects/functional-harmony/data/OpenScore-LiederCorpus/scores/Schumann,_Robert/Dichterliebe,_Op.48/01_-_Im_wunderschönen_Monat_Mai/automatic.csv"
    op = "/home/gianluca/PycharmProjects/functional-harmony/data/OpenScore-LiederCorpus/scores/Schumann,_Robert/Dichterliebe,_Op.48/01_-_Im_wunderschönen_Monat_Mai/automatic2.txt"
    t2r.convert_file(sp, ip, op)

    # sp = "/home/gianluca/PycharmProjects/functional-harmony/data/OpenScore-LiederCorpus/scores/Chaminade,_Cécile/_/Amour_d'automne/lc4999304.mxl"
    # ip = "/home/gianluca/PycharmProjects/functional-harmony/data/OpenScore-LiederCorpus/scores/Chaminade,_Cécile/_/Amour_d'automne/automatic.csv"
    # op = "/home/gianluca/PycharmProjects/functional-harmony/data/OpenScore-LiederCorpus/scores/Chaminade,_Cécile/_/Amour_d'automne/automatic2.txt"
    # t2r.convert_file(sp, ip, op)

    # sp = "/home/gianluca/PycharmProjects/functional-harmony/data/OpenScore-LiederCorpus/scores/Brahms,_Johannes/8_Lieder_und_Gesänge,_Op.58/1_-_Blinde_Kuh/lc5123355.mxl"
    # ip = "/home/gianluca/PycharmProjects/functional-harmony/data/OpenScore-LiederCorpus/scores/Brahms,_Johannes/8_Lieder_und_Gesänge,_Op.58/1_-_Blinde_Kuh/automatic.csv"
    # op = "/home/gianluca/PycharmProjects/functional-harmony/data/OpenScore-LiederCorpus/scores/Brahms,_Johannes/8_Lieder_und_Gesänge,_Op.58/1_-_Blinde_Kuh/automatic2.txt"
    # t2r.convert_file(sp, ip, op)

    # sp = '/home/gianluca/PycharmProjects/functional-harmony/data/OpenScore-LiederCorpus/scores/Reichardt,_Louise/Zwölf_Deutsche_und_Italiänische_Romantische_Gesänge/01_-_Frühlingslied/lc5067312.mxl'
    # ip = '/home/gianluca/PycharmProjects/functional-harmony/data/OpenScore-LiederCorpus/scores/Reichardt,_Louise/Zwölf_Deutsche_und_Italiänische_Romantische_Gesänge/01_-_Frühlingslied/automatic.csv'
    # op = '/home/gianluca/PycharmProjects/functional-harmony/data/OpenScore-LiederCorpus/scores/Reichardt,_Louise/Zwölf_Deutsche_und_Italiänische_Romantische_Gesänge/01_-_Frühlingslied/automatic2.txt'
    # t2r.convert_file(sp, ip, op)

    r2t = ConverterRn2Tab()
    # sp = '/home/gianluca/PycharmProjects/functional-harmony/data/Bach_WTC_1_Preludes/scores/wtc_i_prelude_01.mxl'
    # ip = '/home/gianluca/PycharmProjects/functional-harmony/data/Bach_WTC_1_Preludes/txt/wtc_i_prelude_01.txt'
    # op = '/home/gianluca/PycharmProjects/functional-harmony/data/Bach_WTC_1_Preludes/chords_generated/wtc_i_prelude_01b.csv'
    # r2t.convert_file(sp, ip, op)

    # sp = '/home/gianluca/PycharmProjects/functional-harmony/data/Beethoven_4tets/scores/op18_no2_mov2.mxl'
    # ip = '/home/gianluca/PycharmProjects/functional-harmony/data/Beethoven_4tets/txt/op18_no2_mov2.txt'
    # op = '/home/gianluca/PycharmProjects/functional-harmony/data/Beethoven_4tets/chords_generated/op18_no2_mov2.csv'
    # r2t.convert_file(sp, ip, op)

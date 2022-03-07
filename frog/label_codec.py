import logging
import random

import numpy as np

from frog import (
    NOTES,
    PITCH_FIFTHS,
    SCALES,
    _flat_alteration,
    _sharp_alteration,
    find_enharmonic_equivalent,
)

OUTPUT_MODES = ["legacy", "experimental"]
KEY_START, KEY_END = [PITCH_FIFTHS.index(p) for p in ["C-", "A#"]]
KEYS_FIFTH = PITCH_FIFTHS[KEY_START : KEY_END + 1]

QUALITIES = ["M", "m", "d", "a", "M7", "m7", "D7", "d7", "h7", "+6"]
# QUALITIES = ["M", "m", "d", "a", "M7", "m7", "D7", "d7", "h7", "Gr+6", "It+6", "Fr+6"]
QUALITIES_MIREX = ["maj", "min", "dim", "aug", "maj7", "min7", "7", "dim7", "hdim7",
                   "7"]
Q2I = {x: i for i, x in enumerate(QUALITIES)}

DEGREES_DIATONIC = ["1", "2", "3", "4", "5", "6", "7"]
DEGREES_ALTERATIONS = ["", "+", "-"]

logger = logging.getLogger(__name__)


class LabelCodec:
    def __init__(self, spelling, mode="legacy", strict=True):
        # degrees and tonicisations have the same output shape
        assert mode in OUTPUT_MODES
        assert isinstance(spelling, bool)
        key_temp = KEYS_FIFTH if spelling else NOTES
        self.mode = mode
        self.qualities = QUALITIES
        self.inversions = ["0", "1", "2", "3"]
        self.root = PITCH_FIFTHS if spelling else NOTES

        self.Chord2Root = dict()
        self.spelling = spelling  # True if there is full pitch spelling
        self.strict = strict
        if self.mode == "legacy":
            self.keys = key_temp + [p.lower() for p in key_temp]
            self.degrees = [alt + deg for alt in DEGREES_ALTERATIONS for deg in DEGREES_DIATONIC]
            self.K2I = {x: i for i, x in enumerate(self.keys)}
            self.D2I = {x: i for i, x in enumerate(self.degrees)}

            self.output_features = ["key", "tonicisation", "degree", "quality", "inversion", "root"]
            temp = [
                self.keys,
                self.degrees,
                self.degrees,
                self.qualities,
                self.inversions,
                self.root,
            ]
            self.output_size = dict(zip(self.output_features, [len(x) for x in temp]))

        else:
            self.keys = (key_temp, ["major", "minor"])
            self.K2I = {
                **{x: (i, 0) for i, x in enumerate(self.keys[0])},
                **{x.lower(): (i, 1) for i, x in enumerate(self.keys[0])},
            }
            self.degrees = (DEGREES_DIATONIC, DEGREES_ALTERATIONS)
            self.D2I = {
                a + d: (i, j)
                for i, d in enumerate(self.degrees[0])
                for j, a in enumerate(self.degrees[1])
            }

            self.output_features = [
                "key_root",
                "key_mode",
                "tonicisation_root",
                "tonicisation_alteration",
                "degree_root",
                "degree_alteration",
                "quality",
                "inversion",
                "root",
            ]
            temp = [
                *self.keys,
                *self.degrees,
                *self.degrees,
                self.qualities,
                self.inversions,
                self.root,
            ]
            self.output_size = dict(zip(self.output_features, [len(x) for x in temp]))

        self.Q2I = {x: i for i, x in enumerate(self.qualities)}
        self.I2I = {x: i for i, x in enumerate(self.inversions)}
        self.R2I = {x: i for i, x in enumerate(self.root)}

        return

    def find_chord_root_enc(self, key, ton, deg):
        """Get the chord root from key, tonicisation, and degree. All encoded as integers"""
        key_str = self.decode_key(key)
        deg_str = self.decode_degree(ton, deg)
        root_str = self.find_chord_root_str(key_str, deg_str)
        return self.encode_root(root_str)

    def find_chord_root_str(self, key_str, degree_str):
        """
        Get the chord root from key and tonicised degree. Strings in input and output
        :param key_str: as a string, e.g. c#
        :param degree_str: as a string, e.g. 7/4
        :return: root of the chord, as a string
        """
        try:
            return self.Chord2Root[f"{key_str},{degree_str}"]
        except KeyError:
            pass

        ton_enc, deg_enc = self.encode_degree(degree_str)
        if ton_enc is None or deg_enc is None:
            logger.warning(f"Found a weird degree, {degree_str}. Returning None as root.")
            return None

        if self.mode == "legacy":
            deg, deg_alt = deg_enc % 7, deg_enc // 7
            ton, ton_alt = ton_enc % 7, ton_enc // 7
        else:
            deg, deg_alt = deg_enc
            ton, ton_alt = ton_enc
        key2 = SCALES[key_str][ton]  # secondary key
        if (key_str.isupper() and ton in [1, 2, 5, 6]) or (
            key_str.islower() and ton in [0, 1, 3, 6]
        ):
            key2 = key2.lower()
        if ton_alt == 1:
            key2 = _sharp_alteration(key2).lower()  # when the root is raised, we go to minor scale
        elif ton_alt == 2:
            key2 = _flat_alteration(key2).upper()  # when the root is lowered, we go to major scale

        if not self.spelling:
            key2 = find_enharmonic_equivalent(key2)

        try:
            root = SCALES[key2][deg]
        except KeyError:
            logger.warning(f"Found secondary key {key2} not in allowed keys")
            return None
        if deg_alt == 1:
            root = _sharp_alteration(root)
        elif deg_alt == 2:
            root = _flat_alteration(root)

        if not self.spelling:
            root = find_enharmonic_equivalent(root)

        self.Chord2Root[f"{key_str},{degree_str}"] = root
        return root

    def encode_key(self, key):
        if not self.spelling:
            key = find_enharmonic_equivalent(key)
        if key not in self.K2I:
            logger.warning(f"Can't find an encoding for key {key}, returning None.")
        return self.K2I.get(key, None)

    def encode_degree(self, degree):
        """
        If the input is 7/5, the output is (4, 6):
          - the two numbers are inverted because we output tonicisations first
          - everything is reduced by one because of 1-index vs 0-index conventions
        7 diatonics *  3 chromatics  = 21; (0-6 diatonic, 7-13 sharp, 14-20 flat)
        :param degree: It must be a string following the conventions of tabular representation
        :return: tonicisation, degree
        """
        if "/" in degree:  # the case of tonicised chords
            num, den = degree.split("/")
            _, tonicisation = self.encode_degree(den)
            _, degree = self.encode_degree(num)
            return tonicisation, degree

        while degree.endswith("+"):  # the data format uses 1+ for the degree of augmented tonic
            degree = degree[:-1]
        # There is a very small number of degrees that are non-valid. In order not to throw
        #  away entire chunks of valuable data, we introduce a little noise.
        if not self.strict and degree not in self.D2I:
            old_degree = degree
            degree = random.choice([d for d in self.D2I])
            logger.warning(
                f"Can't find the appropriate class for degree {old_degree}."
                f" Selecting a random new one: {degree}."
            )
        if degree not in self.D2I:
            logger.warning(f"Can't find degree {degree}, returning None.")
        return self.D2I.get("1"), self.D2I.get(degree, None)

    def encode_root(self, root):
        if root is None:
            if not self.strict:
                root = random.choice(self.root)
                logger.warning(f"I have received a root None. Selecting a random new one: {root}")
                return self.R2I.get(root, None)
            return None
        root = root.upper()
        if not self.spelling:
            root = find_enharmonic_equivalent(root)
        if not self.strict and root not in self.R2I:
            old_root = root
            root = random.choice(self.root)
            logger.warning(
                f"Can't find the appropriate class for root {old_root}."
                f" Selecting a random new one: {root}."
            )
        return self.R2I.get(root, None)

    def encode_quality(self, quality):
        if quality in ["Gr+6", "Fr+6", "It+6"]:  # regroup all augmented sixth chords
            quality = "+6"
        if quality == "+7":  # Map augmented sevenths to augmented triads
            quality = "a"
        if not self.strict and quality not in self.Q2I:
            old_quality = quality
            quality = np.random.choice(self.qualities)
            logger.warning(
                f"Can't find the appropriate class for quality {old_quality}."
                f" Selecting a random new one: {quality}."
            )
        return self.Q2I.get(quality, None)

    def encode_inversion(self, inversion):
        return self.I2I[inversion]

    def encode_chords(self, chords):
        """
        Associate every chord element with an integer that represents its category.

        :param chords: A list of named tuples: [key, degree, quality, inversion, root]
        :return:
        """
        if self.mode == "legacy":
            encoded_chords = [
                (
                    self.encode_key(chord.key),
                    *self.encode_degree(chord.degree),
                    self.encode_quality(chord.quality),
                    self.encode_inversion(chord.inversion),
                    self.encode_root(chord.root),
                )
                for chord in chords
            ]
        else:
            encoded_chords = []
            for chord in chords:
                key_root, key_mode = self.encode_key(chord.key)
                (ton, ton_alt), (degree, degree_alt) = self.encode_degree(chord.degree)
                encoded_chords.append(
                    (
                        key_root,
                        key_mode,
                        ton,
                        ton_alt,
                        degree,
                        degree_alt,
                        self.encode_quality(chord.quality),
                        self.encode_inversion(chord.inversion),
                        self.encode_root(chord.root),
                    )
                )
        return encoded_chords

    def decode_inversion(self, inv_enc):
        return self.inversions[inv_enc]

    def decode_key(self, key_enc):
        if self.mode == "legacy":
            return self.keys[key_enc]
        else:
            key = self.keys[0][key_enc[0]]
            return key if key_enc[1] == 0 else key.lower()

    def _decode_degree_base(self, deg_enc, roman=True):
        if self.mode == "legacy":
            deg_number = (deg_enc % 7) + 1
            deg_alt = deg_enc // 7
        else:
            deg_number = deg_enc[0] + 1
            deg_alt = deg_enc[1]
        deg_num_str = int_to_roman(deg_number) if roman else str(deg_number)
        return ("-" if deg_alt == 2 else "+" if deg_alt == 1 else "") + deg_num_str

    def decode_degree(self, ton_enc, deg_enc, roman=False):
        """Take encoded tonicisation and degree and convert them to a string"""
        deg = self._decode_degree_base(deg_enc, roman)
        ton = self._decode_degree_base(ton_enc, roman)
        return deg if ton == "1" else "/".join([deg, ton])

    def decode_quality(self, qlt_enc):
        qlt = self.qualities[qlt_enc]
        # FIXME: This choice of augmented sixth giving always Italian sixth is quite arbitrary...
        if qlt == "+6":
            qlt = "It+6"
        return qlt

    def decode_results_tabular(self, y):
        """
        Transform the outputs of the model into tabular format, example [G+, V/V, D7, '2']

        :param y: the predictions coming from the model, it can be either one-hot or probabilities,
         in which case this is sampled with an argmax. Shape [features](timesteps, depth)
        :return: keys, degree, quality, inversions
        """
        if self.mode == "legacy":
            key = [self.decode_key(np.argmax(k)) for k in y[0]]
            degree = [
                self.decode_degree(np.argmax(t), np.argmax(d), roman=False)
                for t, d in zip(y[1], y[2])
            ]
            quality = [self.decode_quality(np.argmax(q)) for q in y[3]]
            inversion = [self.decode_inversion(np.argmax(i)) for i in y[4]]
        else:
            key = [self.decode_key((np.argmax(k), np.argmax(m))) for k, m in zip(y[0], y[1])]
            degree = [
                self.decode_degree(
                    (np.argmax(t), np.argmax(ta)), (np.argmax(d), np.argmax(da)), roman=False
                )
                for t, ta, d, da in zip(y[2], y[3], y[4], y[5])
            ]
            quality = [self.decode_quality(np.argmax(q)) for q in y[6]]
            inversion = [self.decode_inversion(np.argmax(i)) for i in y[7]]

        return np.transpose([key, degree, quality, inversion])  # shape (timesteps, 4)

    def get_outputs(self, y):
        if self.mode == "legacy":
            keys_enc = np.argmax(y[0], axis=-1)
            ton_enc = np.argmax(y[1], axis=-1)
            degree_enc = np.argmax(y[2], axis=-1)
        else:
            keys_enc = np.transpose([np.argmax(y[0], axis=-1), np.argmax(y[1], axis=-1)])
            ton_enc = np.transpose([np.argmax(y[2], axis=-1), np.argmax(y[3], axis=-1)])
            degree_enc = np.transpose([np.argmax(y[4], axis=-1), np.argmax(y[5], axis=-1)])
        qlt_enc = np.argmax(y[-3], axis=-1)
        inv_enc = np.argmax(y[-2], axis=-1)
        roo_enc = np.argmax(y[-1], axis=-1)

        return keys_enc, ton_enc, degree_enc, qlt_enc, inv_enc, roo_enc


def int_to_roman(n):
    """Convert an integer to a Roman numeral"""
    if not 0 < n < 8:
        raise ValueError("Argument must be between 1 and 7")
    ints = (5, 4, 1)
    nums = ("V", "IV", "I")
    result = []
    for i in range(len(ints)):
        count = int(n / ints[i])
        result.append(nums[i] * count)
        n -= ints[i] * count
    return "".join(result)


def roman_to_int(roman):
    r2i = {
        "I": 1,
        "II": 2,
        "III": 3,
        "IV": 4,
        "V": 5,
        "VI": 6,
        "VII": 7,
    }
    return r2i[roman.upper()]

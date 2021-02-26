import os
import music21
from config import DATA_FOLDER
from pathlib import Path
from converters import ConverterRn2Tab

# Taken from Micchi's converter.py
quality_music21_to_tab = {
    "major triad": "M",
    "minor triad": "m",
    "diminished triad": "d",
    "augmented triad": "a",
    "minor seventh chord": "m7",
    "major seventh chord": "M7",
    "dominant seventh chord": "D7",
    "incomplete dominant-seventh chord": "D7",  # For '75' and '73'
    "diminished seventh chord": "d7",
    "half-diminished seventh chord": "h7",
    "augmented sixth": "a6",  # This should never happen!!
    "German augmented sixth chord": "Gr+6",
    "French augmented sixth chord": "Fr+6",
    "Italian augmented sixth chord": "It+6",
    "minor-augmented tetrachord": "m",  # I know, but we have to stay consistent with BPS-FH ...
    # 'Neapolitan chord': 'N6'  # N/A: major triad  TODO: Add support to Neapolitan chords?
    "incomplete half-diminished seventh chord": "h7"
}


def rn2tab(s, outpath):
    """Translates RomanText annotations into CSV chord files.

    Technically, there is a ConverterRn2Tab class that does this in Micchi's code.
    It crashes with the examples due to the crazy things done to normalize measures.
    However, measures in an annotation RomanText and its generated score with .show()
    align perfectly. Always that I've tried. So, it's easier to just write a simple
    converter here.

    s is a parsed RomanText file
    s = music21.converter.parse(input_file, format="romantext")
    """
    fout = open(outpath, "w")
    for rn in s.flat.getElementsByClass("RomanNumeral"):
        cols = []
        offset = round(float(rn.offset), 4)
        end = round(float(rn.quarterLength) + offset, 4)
        key = rn.key.tonicPitchNameWithCase
        scaleDegree, alteration = rn.scaleDegreeWithAlteration
        if alteration:
            scaleDegree = f"{alteration.modifier}{scaleDegree}"
        else:
            scaleDegree = f"{scaleDegree}"
        secondaryDegree = rn.secondaryRomanNumeral
        if secondaryDegree:
            scaleDegree, alteration = secondaryDegree.scaleDegreeWithAlteration
            if alteration:
                scaleDegree = f"{alteration.modifier}{scaleDegree}"
            else:
                scaleDegree = f"{scaleDegree}"
            secondaryDegree = scaleDegree
        else:
            secondaryDegree = None
        if secondaryDegree:
            scaleDegree = f"{scaleDegree}/{secondaryDegree}"
        quality = quality_music21_to_tab[rn.commonName]
        inversion = rn.inversion()
        row = f"{offset},{end},{key},{scaleDegree},{quality},{inversion}\n"
        fout.write(row)
    fout.close()


if __name__ == '__main__':
    BPS_ROMANTEXT_FOLDER = os.path.join(DATA_FOLDER, "BPS/txt")
    BPSSYNTH_FOLDER = os.path.join(DATA_FOLDER, "BPSSynth")
    Path(os.path.join(BPSSYNTH_FOLDER, "chords")).mkdir(
        parents=True, exist_ok=True
    )
    Path(os.path.join(BPSSYNTH_FOLDER, "scores")).mkdir(
        parents=True, exist_ok=True
    )

    for rntxt in sorted(os.listdir(BPS_ROMANTEXT_FOLDER)):
        print(rntxt)
        rntxtpath = os.path.join(BPS_ROMANTEXT_FOLDER, rntxt)
        scorepath = os.path.join(
            BPSSYNTH_FOLDER, "scores", rntxt.replace(".txt", ".mxl")
        )
        chordspath = os.path.join(
            BPSSYNTH_FOLDER, "chords", rntxt.replace(".txt", ".csv")
        )
        s = music21.converter.parse(rntxtpath, format="romantext")
        s.write(fp=scorepath, fmt="mxl")
        rn2tab(s, chordspath)
        # print(rn2tab)

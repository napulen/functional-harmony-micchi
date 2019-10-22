import os

from music21 import converter
from music21.humdrum.spineParser import HumdrumException

folder = os.path.join('..', 'data', 'Tavern', 'scores')
krn_names = [f for f in os.listdir(folder) if f.endswith('krn')]
xml_names = [".".join([kn.split(".")[0], "xml"]) for kn in krn_names]
krn_files = [os.path.join(folder, kn) for kn in krn_names]
xml_files = [os.path.join(folder, xn) for xn in xml_names]

for kf, xf in zip(krn_files, xml_files):
    print(kf)
    try:
        score = converter.parse(xf)
        score.write('musicxml', fp=xf)
    except HumdrumException:
        print("could not parse this one")
        pass

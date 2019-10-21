import os

from music21 import converter
from music21.humdrum.spineParser import HumdrumException

folder = os.path.join('..', 'data', 'Tavern', 'scores')
file_names = os.listdir(folder)
krn_files = [os.path.join(folder, fn) for fn in file_names if fn.endswith('krn')]
mxl_files = [os.path.join(folder, fn) for fn in file_names if fn.endswith('mxl')]

for kf, mf in zip(krn_files, mxl_files):
    print(kf)
    try:
        score = converter.parse(kf)
        score.write('mxl', fp=mf)
    except HumdrumException:
        print("could not parse this one")
        pass

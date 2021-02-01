"""
This is an entry point, no other file should import from this one.
Augment and convert the data from .krn to .mxl. I don't even know why we have this one here, probably can be removed.
"""
import json
import os

from music21 import converter
from music21.humdrum.spineParser import HumdrumException

folder = os.path.join('', 'data', 'Tavern', 'scores')
krn_names = [f for f in os.listdir(folder) if f.endswith('krn')]
xml_names = [".".join([kn.split(".")[0], "xml"]) for kn in krn_names]
mxl_names = [".".join([kn.split(".")[0], "mxl"]) for kn in krn_names]
krn_files = [os.path.join(folder, kn) for kn in krn_names]
xml_files = [os.path.join(folder, xn) for xn in xml_names]

instructions = []
for kf, xf, xn, mn in zip(krn_files, xml_files, xml_names, mxl_names):
    print(kf)
    try:
        score = converter.parse(kf)
        score.write('musicxml', fp=xf)
        d = dict()
        d['in'] = xn
        d['out'] = mn
        instructions.append(d)
    except HumdrumException:
        print("could not parse this one")
        pass

json_path = os.path.join(folder, 'job.json')
with open(json_path, 'w') as fp:
    json.dump(instructions, fp)

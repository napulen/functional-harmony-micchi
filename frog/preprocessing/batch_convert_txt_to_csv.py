import logging
import os

import music21

from frog.converters.annotation_converters import ConverterRn2Tab, remove_prima_volta
from frog.run import create_annotated_musicxml

logger = logging.getLogger(__name__)

datasets = [
    "Bach_WTC_1_Preludes",
    # "Beethoven_4tets",  # No!! This should stay commented due to copyright licencing issues!
    "BPS",
    "Early_Choral",
    "OpenScore-LiederCorpus",
    "Orchestral",
    "Quartets",
    "Variations_and_Grounds",
]
for data_id in datasets:
    folder = os.path.join("..", "..", "data", "datasets", data_id)
    temp_folder = os.path.join("..", "..", "temp", "datasets", data_id)
    file_names = sorted(
        [
            os.path.splitext(f)[0]
            for f in os.listdir(os.path.join(folder, "scores"))
            if f != ".DS_Store"
        ]
    )
    c = ConverterRn2Tab()
    for f in file_names:
        logger.info(f)
        s_fp = os.path.join(folder, "scores", f + ".mxl")
        txt_fp = os.path.join(folder, "txt", f + ".txt")
        csv_fp = os.path.join(folder, "chords", f + ".csv")
        score = music21.converter.parse(s_fp)
        remove_prima_volta(score)  # This works inplace, due to music21
        analysis = c.load_input(txt_fp)
        data, _flag = c.run(analysis, score)
        c.write_output(data, csv_fp)
        annotated = False  # careful, annotation is SLOW!
        if annotated:  # Creates mxl files with chords as a separate part
            as_fp = os.path.join(temp_folder, "scores", f + "_annotated.mxl")  # temp_folder!
            create_annotated_musicxml(s_fp, txt_fp, as_fp)

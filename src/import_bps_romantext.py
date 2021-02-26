"""This script was used to import the RomanText version of each BPS file.

The files have been curated by Mark Gotham and taken from https://github.com/MarkGotham/When-in-Rome

Commit: 5f6bce4dfecce876cad4c0870c3c7d0e08e517f5

Assuming that the When-in-Rome repo has been cloned (on specific commit) as a sibbling of this repo
should suffice to be able to run the script.
"""

import os
import shutil

nameMapping = {
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op002_No1/1/analysis.txt": "bps_01_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op002_No2/1/analysis.txt": "bps_02_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op002_No3/1/analysis.txt": "bps_03_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op007/1/analysis.txt": "bps_04_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op010_No1/1/analysis_BPS.txt": "bps_05_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op010_No2/1/analysis.txt": "bps_06_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op010_No3/1/analysis.txt": "bps_07_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op013(Pathetique)/1/analysis_BPS.txt": "bps_08_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op014_No1/1/analysis.txt": "bps_09_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op014_No2/1/analysis.txt": "bps_10_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op022/1/analysis.txt": "bps_11_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op026/1/analysis_BPS.txt": "bps_12_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op027_No1/1/analysis_BPS.txt": "bps_13_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op027_No2(Moonlight)/1/analysis.txt": "bps_14_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op028(Pastorale)/1/analysis.txt": "bps_15_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op031_No1/1/analysis.txt": "bps_16_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op031_No2/1/analysis_BPS.txt": "bps_17_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op031_No3/1/analysis.txt": "bps_18_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op049_No1/1/analysis.txt": "bps_19_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op049_No2/1/analysis.txt": "bps_20_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op053/1/analysis.txt": "bps_21_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op054/1/analysis.txt": "bps_22_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op057(Appassionata)/1/analysis.txt": "bps_23_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op078/1/analysis_BPS.txt": "bps_24_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op079(Sonatina)/1/analysis.txt": "bps_25_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op081a(Les_Adieux)/1/analysis.txt": "bps_26_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op090/1/analysis.txt": "bps_27_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op101/1/analysis.txt": "bps_28_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op106(Hammerklavier)/1/analysis.txt": "bps_29_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op109/1/analysis.txt": "bps_30_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op110/1/analysis.txt": "bps_31_01.txt",
    "../../When-in-Rome/Corpus/Piano_Sonatas/Beethoven,_Ludwig_van/Op111/1/analysis.txt": "bps_32_01.txt",
}

BPS_ROMANTEXT_PATH = "../data/BPS/txt"
if not os.path.exists(BPS_ROMANTEXT_PATH):
    os.makedirs(BPS_ROMANTEXT_PATH)

for whenInRome, bps in nameMapping.items():
    dest = os.path.join(BPS_ROMANTEXT_PATH, bps)
    shutil.copy(whenInRome, dest)
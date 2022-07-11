import subprocess
import os

for f in sorted(os.listdir("phd_testset")):
    print(f)
    subprocess.call(
        f"python -m scripts.cra_run -p logs/ConvGruBlocknade_spelling_bass_2022-06-30_20-30-35/ phd_testset/{f} -o outputs/{f} -v".split()
    )

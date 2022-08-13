import subprocess
import os
import time

if __name__ == "__main__":
    log = open("execution_times.log", "w")
    for f in sorted(os.listdir("phd_testset")):
        print(f)
        start = time.time()
        subprocess.call(
            f"python -m scripts.cra_run -p logs/ConvGruBlocknade_spelling_bass_2022-06-30_20-30-35/ phd_testset/{f} -o outputs/{f} -v".split()
        )
        end = time.time()
        log.write(f"{f}: {end - start:.2f}\n")
        log.flush()
    log.close()

from pathlib import Path
import subprocess
import os

def _run_at_path(cmd, path) -> int:
    p = subprocess.Popen(cmd, cwd=str(path.resolve()))
    return p.wait()

def init():
    path = Path(os.path.abspath(__file__)).parents[1]

    CORENLP = "stanford-corenlp-full-2015-12-09"
    SPICELIB = "eval/pycocoevalcap/spice/lib"
    JAR = "stanford-corenlp-3.6.0"

    if (path / SPICELIB / (JAR + ".jar")).exists():
        print("Found Stanford CoreNLP.")
    else:
        print("Downloading...")
        _run_at_path(["wget", f"http://nlp.stanford.edu/software/{CORENLP}.zip"], path)
        print("Unzipping...")
        _run_at_path(["unzip", f"{CORENLP}.zip", "-d", f"{SPICELIB}/"], path)
        _run_at_path(["mv", f"{SPICELIB}/{CORENLP}/{JAR}.jar", f"{SPICELIB}/"], path)
        _run_at_path(["mv", f"{SPICELIB}/{CORENLP}/{JAR}-models.jar", f"{SPICELIB}/"], path)
        _run_at_path(["rm", "-f", f"{CORENLP}.zip"], path)
        _run_at_path(["rm", "-rf", f"{SPICELIB}/{CORENLP}/"], path)
        print("Done.")

if __name__ == '__main__':
    init()
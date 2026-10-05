"""Re-run every experiment reported in the README, one run at a time.

Runs already present in experiments/*.jsonl are skipped, so the script can be stopped and
resumed. Runs sequentially: on CPU, running several trainings in parallel was measured to be
no faster overall (memory contention).

    cd meshcnn-repro
    python -m gmapnet.run_experiments
Then copy experiments/*.jsonl into results/ and run `python results/summarize.py`.
"""
import json
import os
import subprocess
import sys

DEFAULT = [64, 128, 256]

ORBIT_RUNS = (
    [(code, s, DEFAULT) for code in ("-", "E", "F", "V", "E,F,V") for s in (0, 1)]       # ablation
    + [("-", s, [84, 168, 336]) for s in (0, 1)]                                        # capacity control
    + [(code, s, DEFAULT) for code in ("E,F,V+n", "E,F,V+d+n") for s in (0, 1, 2, 3)]   # main comparison
    + [("E,F,V+n", s, [76, 152, 304]) for s in (0, 1)]                                  # capacity control
)
MESHCNN_SEEDS = (0, 1, 2, 3)


def done(path, key):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {key(json.loads(line)) for line in f if line.strip()}


def run(args):
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-u", "-m", *args], check=False)


if __name__ == "__main__":
    finished = done("experiments/orbit_ablations.jsonl",
                    lambda r: (r["O"], r["seed"], r.get("widths", DEFAULT)))
    for code, seed, widths in ORBIT_RUNS:
        if (code, seed, widths) not in finished:
            run(["gmapnet.train_orbit", code, str(seed), ",".join(map(str, widths))])
    finished = done("experiments/meshcnn_protocol.jsonl", lambda r: (r.get("gemm"), r["seed"]))
    for seed in MESHCNN_SEEDS:
        if ("oriented", seed) not in finished:
            run(["gmapnet.train_meshcnn_protocol", str(seed)])

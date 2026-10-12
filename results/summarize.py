"""Rebuild the README tables from the raw run logs in results/*.jsonl
(orbit_cross_runs.jsonl: runs on mixed triangle/quad meshes, with cross-representation tests).

    cd meshcnn-repro
    python results/summarize.py

Metric: test accuracy at the epoch with the best VALIDATION accuracy (never selected on test),
mean ± standard deviation over seeds.
"""
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WIDTHS = [64, 128, 256]


def load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def row(label, runs):
    tests = [100 * r["test_at_best_val"] for r in runs]
    vals = [100 * r["val"] for r in runs]
    spread = f" ± {st.stdev(tests):.1f}" if len(tests) > 1 else ""
    per_seed = " / ".join(f"{t:.1f}" for t in tests)
    return (f"| {label} | {runs[0]['params'] / 1000:.0f}k | {len(runs)} | {per_seed} | "
            f"**{st.mean(tests):.1f}**{spread} | {st.mean(vals):.1f} |")


def table(title, entries):
    print(f"\n### {title}\n")
    print("| model | params | seeds | test per seed | test (mean ± sd) | val (mean) |")
    print("|---|---|---|---|---|---|")
    for label, runs in entries:
        runs = sorted(runs, key=lambda r: r["seed"])
        if runs:
            print(row(label, runs))


def main():
    orbit = load("orbitcnn_runs.jsonl")
    mesh = load("meshcnn_runs.jsonl")

    def o(code, widths=DEFAULT_WIDTHS, seeds=None):
        return [r for r in orbit if r["O"] == code and r["widths"] == widths
                and (seeds is None or r["seed"] in seeds)]

    def m(gemm, seeds=None):
        return [r for r in mesh if r["gemm"] == gemm and (seeds is None or r["seed"] in seeds)]

    s01 = {0, 1}
    table("Orbit-type ablation (no normalisation, seeds 0-1)", [
        ("no orbit pooling (plain DartConv)", o("-", seeds=s01)),
        ("edge ⟨α₀,α₂⟩", o("E", seeds=s01)),
        ("face ⟨α₀,α₁⟩", o("F", seeds=s01)),
        ("vertex ⟨α₁,α₂⟩", o("V", seeds=s01)),
        ("edge + face + vertex", o("E,F,V", seeds=s01)),
        ("control: no orbit pooling, widened", o("-", widths=[84, 168, 336])),
    ])
    table("Main comparison (seeds 0-3)", [
        ("OrbitCNN E,F,V + LayerNorm", o("E,F,V+n")),
        ("OrbitCNN E,F,V + LayerNorm + dispersion", o("E,F,V+d+n")),
        ("MeshCNN reproduction (no MeshPool)", m("oriented")),
        ("control: E,F,V + LayerNorm, widened", o("E,F,V+n", widths=[76, 152, 304])),
    ])
    table("MeshCNN reproduction: gemm convention (seeds 0-1)", [
        ("index-based convention (buggy)", m("index-based (buggy)", seeds=s01)),
        ("oriented convention (fixed)", m("oriented", seeds=s01)),
    ])

    # Mixed triangle/quad meshes (gmapnet.quadify), E,F,V + LayerNorm + dispersion, seeds 0-3.
    cross = load("orbit_cross_runs.jsonl")

    def c(mesh):
        return sorted([r for r in cross if r["mesh"] == mesh], key=lambda r: r["seed"])

    table("Mixed triangle/quad meshes (OrbitCNN E,F,V + LayerNorm + dispersion, seeds 0-3)", [
        ("triangles (re-run)", c("tri")),
        ("mixed, min_dihedral = 160° (66 % of triangles merged)", c("quad160")),
        ("mixed, min_dihedral = 140° (79 % of triangles merged)", c("quad140")),
    ])
    names = {"tri": "triangles", "quad160": "mixed 160°", "quad140": "mixed 140°"}
    print("\n### Transfer without retraining (test accuracy, seeds 0-3)\n")
    print("| trained on | tested on | test per seed | test (mean ± sd) |")
    print("|---|---|---|---|")
    for train in ("tri", "quad160", "quad140"):
        runs = c(train)
        for other in runs[0]["cross"]:
            accs = [100 * r["cross"][other] for r in runs]
            per_seed = " / ".join(f"{a:.1f}" for a in accs)
            print(f"| {names[train]} | {names[other]} | {per_seed} | "
                  f"**{st.mean(accs):.1f}** ± {st.stdev(accs):.1f} |")


if __name__ == "__main__":
    main()

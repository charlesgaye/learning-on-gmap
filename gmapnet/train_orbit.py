"""Train / evaluate an OrbitCNN on SHREC-16 for one orbit configuration and one seed.

Protocol (shared with gmapnet/train_meshcnn_protocol.py):
  - a VALIDATION split is held out from the training set: 4 meshes per class
    (12 train / 4 validation / 4 test per class), drawn with the seed;
  - the reported epoch is chosen on VALIDATION, and TEST accuracy is reported at that epoch;
  - the test set is never used to make any decision.

Usage:
    cd meshcnn-repro
    python -m gmapnet.train_orbit E,F,V+d+n 0      # 𝒪 = {edge, face, vertex}, dispersion, LayerNorm, seed 0
    python -m gmapnet.train_orbit - 1              # 𝒪 = ∅ (plain DartConv), seed 1
    python -m gmapnet.train_orbit E,F,V+n 0 76,152,304   # custom widths (parameter-matched control)
Orbit codes: E = edge ⟨α₀,α₂⟩, F = face ⟨α₀,α₁⟩, V = vertex ⟨α₁,α₂⟩.
Suffixes: "+d" adds orbit DISPERSION; "+n" adds a per-dart LayerNorm before each ReLU.
Output: one JSON line appended to experiments/orbit_ablations.jsonl.
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from gmapnet.orbitconv import OrbitCNNClassifier, EDGE, FACE, VERTEX
from gmapnet.data import list_classes, load_split, with_labels, split_val, VAL_PER_CLASS
from src.train import standardize

CODES = {"E": EDGE, "F": FACE, "V": VERTEX}
EPOCHS = 60
LR = 1e-3
WIDTHS = (64, 128, 256)


def parse_code(code: str):
    """'E,F,V+d+n' -> (orbit types, dispersion, norm)."""
    base, *flags = code.split("+")      # "E,F,V+d+n" -> base = "E,F,V", flags = ["d", "n"]
    types = tuple(CODES[c] for c in base.split(",")) if base != "-" else ()
    return types, "d" in flags, "n" in flags


@torch.no_grad()
def accuracy(model, data) -> float:
    model.eval()
    ok = 0
    for x, alpha, labs, y in data:
        ok += int(model(x[None], alpha[None], [l[None] for l in labs]).argmax(1).item() == y)
    return ok / len(data)


def main() -> None:
    code = sys.argv[1] if len(sys.argv) > 1 else "E,F,V"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    widths = tuple(int(w) for w in sys.argv[3].split(",")) if len(sys.argv) > 3 else WIDTHS
    types, dispersion, norm = parse_code(code)

    classes = list_classes()
    tr_raw, te_raw = load_split("train", classes), load_split("test", classes)
    all_d = torch.cat([x for x, _, _ in tr_raw], dim=1)
    mean, std = all_d.mean(1, keepdim=True), all_d.std(1, keepdim=True) + 1e-6
    tr_all = with_labels(standardize(tr_raw, mean, std), types)
    test = with_labels(standardize(te_raw, mean, std), types)
    train, val = split_val(tr_all, len(classes), VAL_PER_CLASS, seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = OrbitCNNClassifier(tr_raw[0][0].shape[0], len(classes), widths, orbit_types=types,
                               dispersion=dispersion, norm=norm)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    print(f"O={code} seed={seed} | train={len(train)} val={len(val)} test={len(test)} | "
          f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    best = {"val": -1.0}
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for i in np.random.permutation(len(train)):
            x, alpha, labs, y = train[i]
            loss = crit(model(x[None], alpha[None], [l[None] for l in labs]), torch.tensor([y]))
            opt.zero_grad(); loss.backward(); opt.step()
        va = accuracy(model, val)
        if va > best["val"]:
            best = {"val": va, "test": accuracy(model, test), "epoch": epoch}
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"  epoch {epoch:2d}  val {va:.2f}  (best val {best['val']:.2f} @ {best['epoch']})", flush=True)
    final_test = accuracy(model, test)
    res = {"O": code, "seed": seed, "widths": list(widths),
           "params": sum(p.numel() for p in model.parameters()), "val": best["val"], "test_at_best_val": best["test"],
           "best_epoch": best["epoch"], "test_final": final_test, "minutes": round((time.time() - t0) / 60, 1)}
    print(json.dumps(res), flush=True)
    os.makedirs("experiments", exist_ok=True)
    with open("experiments/orbit_ablations.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()

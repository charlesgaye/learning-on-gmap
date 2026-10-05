"""The MeshCNN reproduction (src/) evaluated with the SAME protocol as the OrbitCNN runs.

Same validation split (4 meshes per class, drawn with the same seed, hence the same meshes as
gmapnet/train_orbit.py), epoch chosen on validation, test accuracy reported at that epoch.
Only the representation differs: edges + gemm with the 5 MeshCNN features.

    cd meshcnn-repro
    python -m gmapnet.train_meshcnn_protocol 0      # seed 0
Output: one JSON line appended to experiments/meshcnn_protocol.jsonl.
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from src.dataset import ShrecDataset
from src.model import MeshCNNClassifier
from src.train import standardize
from gmapnet.data import split_val, VAL_PER_CLASS
from gmapnet.train_orbit import EPOCHS, LR

ROOT = "data/shrec_16"


@torch.no_grad()
def accuracy(model, data) -> float:
    model.eval()
    return sum(int(model(x[None], g[None]).argmax(1).item() == y) for x, g, y in data) / len(data)


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    tr_ds = ShrecDataset(ROOT, "train")
    te_ds = ShrecDataset(ROOT, "test", tr_ds.classes)
    all_e = torch.cat([x for x, _, _ in tr_ds.items], dim=1)
    mean, std = all_e.mean(1, keepdim=True), all_e.std(1, keepdim=True) + 1e-6
    tr_all, test = standardize(tr_ds.items, mean, std), standardize(te_ds.items, mean, std)
    train, val = split_val(tr_all, len(tr_ds.classes), VAL_PER_CLASS, seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MeshCNNClassifier(in_channels=5, num_classes=len(tr_ds.classes))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    params = sum(p.numel() for p in model.parameters())
    print(f"MeshCNN seed={seed} | train={len(train)} val={len(val)} test={len(test)} | params={params:,}", flush=True)

    best = {"val": -1.0}
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for i in np.random.permutation(len(train)):
            x, g, y = train[i]
            loss = crit(model(x[None], g[None]), torch.tensor([y]))
            opt.zero_grad(); loss.backward(); opt.step()
        va = accuracy(model, val)
        if va > best["val"]:
            best = {"val": va, "test": accuracy(model, test), "epoch": epoch}
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"  epoch {epoch:2d}  val {va:.2f}  (best val {best['val']:.2f} @ {best['epoch']})", flush=True)
    res = {"model": "MeshCNN", "gemm": "oriented", "seed": seed, "params": params, "val": best["val"],
           "test_at_best_val": best["test"], "best_epoch": best["epoch"],
           "test_final": accuracy(model, test), "minutes": round((time.time() - t0) / 60, 1)}
    print(json.dumps(res), flush=True)
    os.makedirs("experiments", exist_ok=True)
    with open("experiments/meshcnn_protocol.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()

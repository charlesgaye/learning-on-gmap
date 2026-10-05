"""Quick MeshCNN training on the full SHREC-16 train split, evaluated on the test split.

Recipe: standardised input features (training statistics), MeshConv + ReLU, global MAX
pooling, linear head, one mesh per step (batch = 1).

This script reports the FINAL test accuracy and selects nothing on the test set. The numbers in
the README use the stricter protocol of `gmapnet/train_meshcnn_protocol.py` (held-out
validation split, epoch chosen on validation).

    cd meshcnn-repro
    python -m src.train
"""
from __future__ import annotations
import time

import numpy as np
import torch
import torch.nn as nn

from .dataset import ShrecDataset
from .model import MeshCNNClassifier

ROOT = "data/shrec_16"
# None = all 30 classes. For a quick first run, e.g.:
#   CLASSES = ["alien", "ants", "bird1", "camel", "cat", "dog2"]
CLASSES: list[str] | None = None
EPOCHS = 60
LR = 1e-3


def standardize(items, mean, std):
    """Apply (x - mean) / std to every item. mean, std: (C, 1)."""
    return [((x - mean) / std, gemm, y) for x, gemm, y in items]


@torch.no_grad()
def evaluate(model, data) -> float:
    """Accuracy on a list of (x, gemm, y) items."""
    model.eval()
    correct = 0
    for x, gemm, y in data:
        logits = model(x[None], gemm[None])          # add the batch dimension -> (1, 5, E)
        correct += int(logits.argmax(1).item() == y)
    return correct / len(data)


def main() -> None:
    t0 = time.time()
    train_ds = ShrecDataset(ROOT, "train", CLASSES)
    test_ds = ShrecDataset(ROOT, "test", train_ds.classes)   # same classes -> same labels
    num_classes = len(train_ds.classes)
    print(f"{num_classes} classes | train={len(train_ds)} test={len(test_ds)} "
          f"| data ready in {time.time()-t0:.0f}s")

    # standardisation: per-feature statistics over all TRAINING edges
    all_edges = torch.cat([x for x, _, _ in train_ds.items], dim=1)   # (5, total_E)
    mean = all_edges.mean(1, keepdim=True)
    std = all_edges.std(1, keepdim=True) + 1e-6
    train = standardize(train_ds.items, mean, std)
    test = standardize(test_ds.items, mean, std)

    torch.manual_seed(0)
    model = MeshCNNClassifier(in_channels=5, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        for i in np.random.permutation(len(train)):
            x, gemm, y = train[i]
            logits = model(x[None], gemm[None])
            loss = criterion(logits, torch.tensor([y]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            print(f"epoch {epoch:2d}  train {evaluate(model, train):.2f}  test {evaluate(model, test):.2f}")
    print(f"\nfinal test accuracy: {evaluate(model, test)*100:.1f}%  (chance = {100/num_classes:.1f}%)")


if __name__ == "__main__":
    main()

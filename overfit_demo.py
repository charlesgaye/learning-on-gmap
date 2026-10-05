"""Sanity check: the MeshCNN stack learns a toy task (spheres vs cubes) in a few seconds on CPU.

Validates the whole pipeline (mesh -> features -> MeshConv -> model -> loss -> backward ->
step) on a tiny synthetic task before running SHREC.

    python overfit_demo.py
"""
import numpy as np
import torch
import torch.nn as nn
import trimesh

from src.mesh import Mesh, edge_features
from src.model import MeshCNNClassifier


def mesh_to_tensors(tm: trimesh.Trimesh):
    """trimesh mesh -> (x, gemm) ready for the model."""
    m = Mesh(vertices=np.asarray(tm.vertices, dtype=float), faces=np.asarray(tm.faces))
    m.build_edges()
    feats = edge_features(m)                                  # (E, 5)
    x = torch.tensor(feats.T, dtype=torch.float)[None]        # (1, 5, E)
    gemm = torch.tensor(np.array(m.gemm), dtype=torch.long)[None]  # (1, E, 4)
    return x, gemm


def make_dataset(n_per_class: int = 6, seed: int = 0):
    """n_per_class noisy spheres (class 0) and cubes (class 1).

    The features are rotation/scale invariant, so variety comes from vertex NOISE
    (a rotation would give identical features).
    """
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n_per_class):
        for kind in (0, 1):
            base = trimesh.creation.icosphere(subdivisions=1) if kind == 0 \
                else trimesh.creation.box().subdivide()
            v = np.asarray(base.vertices, dtype=float) + rng.normal(0, 0.02, base.vertices.shape)
            tm = trimesh.Trimesh(vertices=v, faces=np.asarray(base.faces), process=False)
            x, gemm = mesh_to_tensors(tm)
            data.append((x, gemm, kind))
    return data


def main():
    data = make_dataset()
    print(f"{len(data)} meshes (spheres vs cubes)")

    torch.manual_seed(0)
    model = MeshCNNClassifier(in_channels=5, num_classes=2, widths=(32, 64))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(30):
        model.train()
        correct = 0
        for x, gemm, y in data:
            target = torch.tensor([y])
            logits = model(x, gemm)
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += int(logits.argmax(1).item() == y)

        if epoch % 5 == 0:
            print(f"epoch {epoch:2d}  train acc {correct/len(data):.2f}")
    print(f"final train acc {correct/len(data):.2f}")


if __name__ == "__main__":
    main()

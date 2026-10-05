"""MeshCNNClassifier: output shape and end-to-end differentiability.

A forward pass must return (N, num_classes) and a backward pass must propagate a gradient down
to the first convolution.

    cd meshcnn-repro
    python -m tests.test_model
"""
import numpy as np
import torch

from src.mesh import Mesh, edge_features
from src.model import MeshCNNClassifier

V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
F = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])


def run():
    m = Mesh(vertices=V, faces=F)
    m.build_edges()
    feats = edge_features(m)

    x = torch.tensor(feats.T, dtype=torch.float)[None]              # (1, 5, E)
    gemm = torch.tensor(np.array(m.gemm), dtype=torch.long)[None]   # (1, E, 4)

    torch.manual_seed(0)
    model = MeshCNNClassifier(in_channels=5, num_classes=3, widths=(16, 32))

    out = model(x, gemm)
    assert tuple(out.shape) == (1, 3), f"output {tuple(out.shape)}, expected (1, 3)"

    model.zero_grad()
    out.sum().backward()
    g = model.convs[0].conv.weight.grad
    assert g is not None and g.abs().sum().item() > 0, "no gradient on the first convolution"

    print("OK - MeshCNNClassifier: output (1, 3), gradient reaches the first convolution.")
    print(f"     {sum(p.numel() for p in model.parameters())} parameters.")


if __name__ == "__main__":
    run()

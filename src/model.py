"""MeshCNN classifier (simplified: no MeshPool)."""

from __future__ import annotations
import torch.nn as nn

from .layers import MeshConv


class MeshCNNClassifier(nn.Module):
    """Mesh classification.

        [MeshConv -> ReLU] x len(widths) -> global MAX pooling over edges -> linear head.

    Input features are standardised upstream (see train.py).

    No BatchNorm: training uses one mesh per step (batch = 1), and BatchNorm's batch
    statistics then create a train/eval mismatch (measured: 0.54 vs 0.92 test accuracy on a
    6-class subset). Global MAX pooling rather than mean: the mean washed out the signal and
    the model did not learn.

    Args:
        in_channels : 5 (edge features).
        num_classes : number of SHREC classes.
        widths      : widths of the conv blocks, e.g. (64, 128, 256).

    Forward:
        x    : (N, in_channels, E)   edge features.
        gemm : (N, E, 4)             edge neighbourhoods.
        -> (N, num_classes)          logits.
    """

    def __init__(self, in_channels: int = 5, num_classes: int = 30,
                 widths: tuple[int, ...] = (64, 128, 256)):
        super().__init__()
        # MeshConv needs gemm in forward -> ModuleList rather than nn.Sequential.
        self.convs = nn.ModuleList()
        prev = in_channels
        for w in widths:
            self.convs.append(MeshConv(prev, w))
            prev = w
        self.head = nn.Linear(prev, num_classes)

    def forward(self, x, gemm):
        for conv in self.convs:
            x = conv(x, gemm).relu()
        pooled = x.max(dim=2).values        # global max over edges -> (N, C_last)
        return self.head(pooled)

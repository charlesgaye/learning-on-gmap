"""MeshCNN edge convolution (MeshConv).

Reference: Hanocka et al., "MeshCNN: A Network with an Edge", SIGGRAPH 2019 (§3.2).
MeshPool (learned edge collapse, §3.3) is not implemented in this reproduction.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class MeshConv(nn.Module):
    """Convolution on edges.

    Each edge e (feature x_e) has 4 neighbours (a, b, c, d) ordered by gemm so that (a, c) and
    (b, d) are the symmetric pairs (see Mesh.build_edges). The neighbourhood is symmetrised to
    remove the ordering ambiguity,
        [ x_e , a + c , |a - c| , b + d , |b - d| ],
    and a learned 1x5 kernel, shared by all edges, mixes these 5 positions.

    Forward:
        x    : (N, in_channels, E)   edge features (batch N).
        gemm : (N, E, 4)             indices of the 4 neighbouring edges.
        -> (N, out_channels, E)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5))

    def _gather(self, x, gemm):
        """Features of the 4 neighbours of each edge: (a, b, c, d), each of shape (N, C, E)."""
        N, C, E = x.shape
        idx = gemm.reshape(N, 1, E * 4).expand(N, C, E * 4)   # (N, C, E*4)
        neigh = torch.gather(x, 2, idx).reshape(N, C, E, 4)   # (N, C, E, 4)
        return neigh[..., 0], neigh[..., 1], neigh[..., 2], neigh[..., 3]

    def forward(self, x, gemm):  # noqa: D401
        a, b, c, d = self._gather(x, gemm)          # each (N, C, E)
        positions = [x, a+c, (a-c).abs(), b+d, (b-d).abs()]
        G = torch.stack(positions, dim=3)
        return self.conv(G).squeeze(3)

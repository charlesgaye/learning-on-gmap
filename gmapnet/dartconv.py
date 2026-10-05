"""DartConv: convolution over the darts of a G-map; the neighbourhood is given by the involutions.

Compared with MeshConv:
  - the unit is the dart, not the edge;
  - the neighbourhood [d, α₀(d), α₁(d), …, αₙ(d)] has CANONICAL labels (one per dimension), so no
    symmetrisation is needed: one learned weight per involution;
  - the layer is dimension-agnostic: alpha of shape (D, n+1) gives an n+2 position kernel.
The layer is equivariant to any renumbering of the darts (G-map isomorphism) by construction.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class DartConv(nn.Module):
    """Convolution over darts.

    Kernel positions:
        position 0     : the dart itself, x_d
        position 1 + i : its neighbour through αᵢ, x_{αᵢ(d)}      (i = 0 … n)
    A fixed point αᵢ(d) = d (boundary) naturally gives x_{αᵢ(d)} = x_d.

    Forward:
        x     : (N, C_in, D)   features per dart (batch N).
        alpha : (N, D, K)      involutions, K = n + 1 (3 for a 2-G-map).
        -> (N, C_out, D)
    """

    def __init__(self, in_channels: int, out_channels: int, n_involutions: int = 3):
        super().__init__()
        self.n_involutions = n_involutions
        # 1 × (1 + K) kernel mixing [self, α₀, α₁, …], shared by all darts
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1 + n_involutions))

    def forward(self, x, alpha):  # noqa: D401
        N, C, D = x.shape
        K = alpha.shape[2]
        idx = alpha.reshape(N, 1, D * K).expand(N, C, D * K)        # neighbour indices, copied per channel
        neigh = torch.gather(x, 2, idx).reshape(N, C, D, K)          # neigh[..., i] = x ∘ αᵢ
        positions = [x] + [neigh[..., i] for i in range(K)]
        g = torch.stack(positions, dim=3)
        return self.conv(g).squeeze(3)


class DartCNNClassifier(nn.Module):
    """[DartConv -> ReLU] × len(widths) -> global MAX over darts -> linear head.

    Kept as the « no orbit pooling » baseline; `OrbitCNNClassifier` with an empty orbit set
    is equivalent.
    """

    def __init__(self, in_channels: int, num_classes: int = 30,
                 widths: tuple[int, ...] = (64, 64, 128, 128, 256, 256),
                 n_involutions: int = 3):
        super().__init__()
        self.convs = nn.ModuleList()
        prev = in_channels
        for w in widths:
            self.convs.append(DartConv(prev, w, n_involutions))
            prev = w
        self.head = nn.Linear(prev, num_classes)

    def forward(self, x, alpha):
        for conv in self.convs:
            x = conv(x, alpha).relu()
        return self.head(x.max(dim=2).values)

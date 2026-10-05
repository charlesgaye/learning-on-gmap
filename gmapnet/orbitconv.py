"""OrbitDartConv: dart convolution + pooling over chosen orbit types.

For a dart d:
    x'_d = b + W_self x_d + Σᵢ Wᵢ x_{αᵢ(d)} + Σ_{O ∈ 𝒪} ( U_O · mean_O(x)(d) [+ V_O · disp_O(x)(d)] )
where mean_O(x)(d) is the mean of x over THE orbit of type O containing d (orbits of one type
partition the darts, so this is well defined) and disp_O its dispersion. All weights are shared
by all darts; neighbouring orbits are reached through depth. 𝒪 = ∅ gives exactly DartConv.

Implementation: orbit terms are EXTRA POSITIONS of the same kernel,
    [self, α₀, …, α_{K-1}, mean_{O₁}, (disp_{O₁}), …]   -> 1 × (1 + K + m) or 1 × (1 + K + 2m).

mean_O is a pushforward followed by a pullback:
    pushforward: per-orbit sums  (scatter_add_, the mirror of gather: it WRITES by accumulating)
                 divided by the orbit sizes;
    pullback   : gather, which gives each dart the value of its orbit.
"""

from __future__ import annotations
import torch
import torch.nn as nn

EDGE, FACE, VERTEX = (0, 2), (0, 1), (1, 2)       # usual orbit types of a 2-G-map


def orbit_mean(x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean of x over the orbit of each dart, copied back onto the darts.

    x      : (N, C, D) features per dart.
    labels : (N, D)    orbit label of each dart (integers 0..M_n-1; M_n may vary with n).
    -> (N, C, D)       out[n, c, d] = mean of x[n, c, ·] over the darts labelled like d.
    """
    N, C, D = x.shape
    M = int(labels.max()) + 1
    idx = labels.reshape(N, 1, D).expand(N, C, D)
    orbit_sums = torch.zeros(N, C, M)
    orbit_sums.scatter_add_(2, idx, x)
    orbit_size = torch.zeros(N, 1, M)
    orbit_size.scatter_add_(2, labels.reshape(N, 1, D), torch.ones(N, 1, D))
    out = orbit_sums/orbit_size
    return torch.gather(out, 2, idx)


def orbit_dispersion(x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Dispersion of x over the orbit of each dart: orbit mean of |x − orbit mean|.

    Generalises MeshCNN's symmetrisation to orbits of any size: for a pair {a, c},
    (mean, dispersion) = ((a+c)/2, |a−c|/2), a complete invariant of unordered pairs.
    Same shapes as orbit_mean.
    """
    return orbit_mean((x - orbit_mean(x, labels)).abs(), labels)

class OrbitDartConv(nn.Module):
    """Forward:
        x            : (N, C_in, D)
        alpha        : (N, D, K)
        orbit_labels : list of m tensors (N, D), one per entry of `orbit_types`, IN THAT ORDER.
        -> (N, C_out, D)

    dispersion=False : positions [self, α₀…, mean_{O₁}, …, mean_{O_m}]                     (1 + K + m)
    dispersion=True  : positions [self, α₀…, mean_{O₁}, disp_{O₁}, …, mean_{O_m}, disp_{O_m}] (1 + K + 2m)
    """

    def __init__(self, in_channels: int, out_channels: int, n_involutions: int = 3,
                 orbit_types: tuple = (EDGE, FACE, VERTEX), dispersion: bool = False):
        super().__init__()
        self.n_involutions = n_involutions
        self.orbit_types = tuple(orbit_types)
        self.dispersion = dispersion
        width = 1 + n_involutions + len(self.orbit_types) * (2 if dispersion else 1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, width))

    def forward(self, x, alpha, orbit_labels):  # noqa: D401
        N, C, D = x.shape
        K = alpha.shape[2]
        idx = alpha.reshape(N, 1, D * K).expand(N, C, D * K)        # neighbour indices, copied per channel
        neigh = torch.gather(x, 2, idx).reshape(N, C, D, K)          # neigh[..., i] = x ∘ αᵢ

        position = [x] + [neigh[..., i] for i in range(K)]
        for labels in orbit_labels:
            position.append(orbit_mean(x, labels))
            if self.dispersion:
                position.append(orbit_dispersion(x, labels))
        g = torch.stack(position, dim=3)
        return self.conv(g).squeeze(3)


class OrbitCNNClassifier(nn.Module):
    """[OrbitDartConv -> (LayerNorm) -> ReLU] × len(widths) -> global MAX over darts -> linear head.

    norm=True: LayerNorm over the CHANNELS of each dart, between the convolution and the ReLU.
      Why: from layer 2 on, the inputs are ReLU outputs (≥ 0), and so are orbit means and
      dispersions; pre-activations are then not centred and one gradient step can push them ALL
      below 0, killing a whole layer of ReLUs (observed: 0 % active units in layer 2 within ~150
      steps). The per-dart LayerNorm recentres them; it depends neither on the batch nor on the
      other darts, so equivariance is preserved.
    """

    def __init__(self, in_channels: int, num_classes: int = 30,
                 widths: tuple[int, ...] = (64, 128, 256), n_involutions: int = 3,
                 orbit_types: tuple = (EDGE, FACE, VERTEX), dispersion: bool = False,
                 norm: bool = False):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if norm else None
        prev = in_channels
        for w in widths:
            self.convs.append(OrbitDartConv(prev, w, n_involutions, orbit_types, dispersion))
            if norm:
                self.norms.append(nn.LayerNorm(w))   # normalises the LAST dimension, of size w
            prev = w
        self.head = nn.Linear(prev, num_classes)

    def forward(self, x, alpha, orbit_labels, return_acts: bool = False):
        acts = []                                     # ReLU outputs (diagnostics, see gmapnet/diag_relu.py)
        for k, conv in enumerate(self.convs):
            x = conv(x, alpha, orbit_labels)          # (N, C, D)
            if self.norms is not None:
                # LayerNorm acts on the last dimension: move channels last, normalise, move back
                x = self.norms[k](x.permute(0,2,1)).permute(0,2,1)
            x = x.relu()
            acts.append(x)
        out = self.head(x.max(dim=2).values)
        return (out, acts) if return_acts else out

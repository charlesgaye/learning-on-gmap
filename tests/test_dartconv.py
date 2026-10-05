"""DartConv (gmapnet.dartconv).

1. output shape;
2. ORACLE: equality with a naive per-dart computation  out_d = b + W₀ x_d + Σᵢ W_{1+i} x_{αᵢ(d)};
3. EQUIVARIANCE: renumbering the darts (G-map isomorphism) permutes the output identically;
4. batch: each element of a batch is gathered with its OWN alpha;
5. classifier: INVARIANT to dart renumbering (hence to mesh vertex renumbering).

    cd meshcnn-repro
    python -m tests.test_dartconv
"""
import numpy as np
import torch

from gmapnet.gmap import build_gmap
from gmapnet.dartconv import DartConv, DartCNNClassifier
from tests.test_gmap import TETRA, SQUARE, torus

ATOL = 1e-5


def as_torch(g):
    return torch.tensor(np.asarray(g.alpha), dtype=torch.long)[None]      # (1, D, 3)


def naive(conv, x, alpha):
    """Readable reference: one dart at a time."""
    W = conv.conv.weight[:, :, 0, :]            # (C_out, C_in, 1+K)
    b = conv.conv.bias
    N, C, D = x.shape
    K = alpha.shape[2]
    out = torch.empty(N, W.shape[0], D)
    for n in range(N):
        for d in range(D):
            acc = b + W[:, :, 0] @ x[n, :, d]
            for i in range(K):
                acc = acc + W[:, :, 1 + i] @ x[n, :, alpha[n, d, i]]
            out[n, :, d] = acc
    return out


def relabel(alpha, perm):
    """Renumber the darts: dart d becomes perm[d], so alpha'[perm[d], i] = perm[alpha[d, i]]."""
    a = alpha[0].numpy()
    new = np.empty_like(a)
    new[perm] = perm[a]
    return torch.tensor(new)[None]


def run():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    print("DartConv:")

    for name, F in (("tetrahedron", TETRA), ("square", SQUARE), ("torus", torus())):
        alpha = as_torch(build_gmap(F))
        D = alpha.shape[1]
        x = torch.randn(1, 5, D)
        conv = DartConv(5, 7)

        out = conv(x, alpha)
        assert tuple(out.shape) == (1, 7, D), f"[{name}] output {tuple(out.shape)}, expected (1, 7, {D})"

        ref = naive(conv, x, alpha)
        assert torch.allclose(out, ref, atol=ATOL), \
            f"[{name}] does not match the naive computation (position order [self, a0, a1, a2]?)"

        perm = rng.permutation(D)
        x_p = torch.empty_like(x); x_p[:, :, perm] = x          # dart d becomes perm[d]
        out_p = conv(x_p, relabel(alpha, perm))
        assert torch.allclose(out_p[:, :, perm], out, atol=ATOL), \
            f"[{name}] not equivariant to dart renumbering"
        print(f"  OK {name:<12} shape, naive oracle, equivariance (D={D})")

    # batch of 2: each element uses its own alpha
    alpha = as_torch(build_gmap(TETRA))
    D = alpha.shape[1]
    perm = rng.permutation(D)
    alphas = torch.cat([alpha, relabel(alpha, perm)])                        # (2, D, 3)
    x = torch.randn(2, 5, D)
    conv = DartConv(5, 7)
    assert torch.allclose(conv(x, alphas), naive(conv, x, alphas), atol=ATOL), \
        "batch N=2: each element must be gathered with its OWN alpha"
    print("  OK batch N=2 (one alpha per element)")

    # classifier invariant to renumbering
    alpha = as_torch(build_gmap(torus()))
    D = alpha.shape[1]
    model = DartCNNClassifier(in_channels=5, num_classes=4, widths=(8, 8))
    x = torch.randn(1, 5, D)
    perm = rng.permutation(D)
    x_p = torch.empty_like(x); x_p[:, :, perm] = x
    assert torch.allclose(model(x, alpha), model(x_p, relabel(alpha, perm)), atol=ATOL), \
        "classifier not invariant to dart renumbering"
    print("  OK classifier invariant to renumbering (by construction, no symmetrisation)")
    print("OK - DartConv verified.")


if __name__ == "__main__":
    run()

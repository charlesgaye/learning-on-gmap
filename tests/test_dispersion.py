"""Orbit dispersion (gmapnet.orbitconv.orbit_dispersion and dispersion=True).

1. on a pair (orbit ⟨α₂⟩): dispersion = |a - c| / 2, i.e. MeshCNN's symmetrisation;
2. loop oracle on edge / face / vertex orbits; non-negative;
3. independent of the names of the labels; batched;
4. layer with dispersion=True: naive formula [self, α₀.., mean_O1, disp_O1, ...] + equivariance.

    cd meshcnn-repro
    python -m tests.test_dispersion
"""
import numpy as np
import torch

from gmapnet.gmap import build_gmap, orbits
from gmapnet.orbitconv import orbit_dispersion, OrbitDartConv, EDGE, FACE, VERTEX
from tests.test_gmap import TETRA, SQUARE, torus
from tests.test_orbitconv import mean_ref, relabel

ATOL = 1e-5
TYPES = (EDGE, FACE, VERTEX)


def disp_ref(x, lab):
    """Oracle: loop over the orbits."""
    out = torch.empty_like(x)
    for n in range(x.shape[0]):
        for k in torch.unique(lab[n]):
            m = lab[n] == k
            v = x[n][:, m]
            out[n][:, m] = (v - v.mean(1, keepdim=True)).abs().mean(1, keepdim=True)
    return out


def tensors(F, types):
    g = build_gmap(F)
    alpha = torch.tensor(np.asarray(g.alpha), dtype=torch.long)[None]
    return alpha, [torch.tensor(orbits(g.alpha, t), dtype=torch.long)[None] for t in types]


def naive(conv, x, alpha, labs):
    W, b = conv.conv.weight[:, :, 0, :], conv.conv.bias
    K = alpha.shape[2]
    extra = []
    for l in labs:
        extra += [mean_ref(x, l), disp_ref(x, l)]
    out = torch.empty(x.shape[0], W.shape[0], x.shape[2])
    for n in range(x.shape[0]):
        for d in range(x.shape[2]):
            acc = b + W[:, :, 0] @ x[n, :, d]
            for i in range(K):
                acc = acc + W[:, :, 1 + i] @ x[n, :, alpha[n, d, i]]
            for j, e in enumerate(extra):
                acc = acc + W[:, :, 1 + K + j] @ e[n, :, d]
            out[n, :, d] = acc
    return out


def run():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    print("Orbit dispersion:")

    # 1. pairs <alpha2>: |a - c| / 2
    g = build_gmap(torus())
    a2 = torch.tensor(np.asarray(g.alpha)[:, 2])
    lab2 = torch.tensor(orbits(g.alpha, (2,)), dtype=torch.long)[None]
    x = torch.randn(1, 4, g.n_darts)
    expected = (x - x[:, :, a2]).abs() / 2
    assert torch.allclose(orbit_dispersion(x, lab2), expected, atol=ATOL), \
        "on a pair {a, c}, the dispersion must equal |a - c| / 2"
    print("  OK pair: disp = |a - c| / 2  (MeshCNN symmetrisation recovered)")

    # 2-3. oracle, non-negativity, label names, batch
    alpha, labs = tensors(torus(), TYPES)
    x = torch.randn(1, 4, alpha.shape[1])
    for t, l in zip(("edge", "face", "vertex"), labs):
        got = orbit_dispersion(x, l)
        assert tuple(got.shape) == tuple(x.shape), f"[{t}] shape {tuple(got.shape)}"
        assert torch.allclose(got, disp_ref(x, l), atol=ATOL), f"[{t}] wrong dispersion"
        assert (got >= -ATOL).all(), f"[{t}] negative dispersion"
        ren = torch.tensor(rng.permutation(int(l.max()) + 1))[l]
        assert torch.allclose(orbit_dispersion(x, ren), got, atol=ATOL), f"[{t}] depends on label NAMES"
    _, lt = tensors(TETRA, (VERTEX,))
    D = lt[0].shape[1]
    lb = torch.cat([lt[0], lt[0][:, torch.tensor(rng.permutation(D))]])
    xb = torch.randn(2, 3, D)
    assert torch.allclose(orbit_dispersion(xb, lb), disp_ref(xb, lb), atol=ATOL), "batch N=2 wrong"
    print("  OK oracle on edge / face / vertex, >= 0, label names, batch")

    # 4. layer
    for name, F in (("tetrahedron", TETRA), ("square", SQUARE), ("torus", torus())):
        alpha, labs = tensors(F, TYPES)
        D = alpha.shape[1]
        x = torch.randn(1, 5, D)
        conv = OrbitDartConv(5, 7, dispersion=True)
        out = conv(x, alpha, labs)
        assert tuple(out.shape) == (1, 7, D), f"[{name}] output {tuple(out.shape)}"
        assert torch.allclose(out, naive(conv, x, alpha, labs), atol=ATOL), \
            f"[{name}] != naive formula (order: [self, a0, a1, a2, mean_O1, disp_O1, mean_O2, disp_O2, ...])"
        perm = rng.permutation(D)
        x_p = torch.empty_like(x); x_p[:, :, perm] = x
        a_p, l_p = relabel(alpha, labs, perm)
        assert torch.allclose(conv(x_p, a_p, l_p)[:, :, perm], out, atol=ATOL), f"[{name}] not equivariant"
        print(f"  OK {name:<12} layer with dispersion: naive formula, equivariance")
    print("OK - orbit dispersion verified.")


if __name__ == "__main__":
    run()

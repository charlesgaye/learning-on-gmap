"""OrbitDartConv (gmapnet.orbitconv).

1. orbit_mean: exact (loop oracle), independent of the NAMES of the labels, batched;
2. layer: shape + naive oracle of  b + W_self x_d + Σ Wᵢ x_{αᵢd} + Σ U_O mean_O(x)(d);
3. 𝒪 = ∅ gives exactly DartConv (same weights);
4. equivariance to dart renumbering;
5. classifier invariant to dart renumbering.

    cd meshcnn-repro
    python -m tests.test_orbitconv
"""
import numpy as np
import torch

from gmapnet.gmap import build_gmap, orbits
from gmapnet.dartconv import DartConv
from gmapnet.orbitconv import orbit_mean, OrbitDartConv, OrbitCNNClassifier, EDGE, FACE, VERTEX
from tests.test_gmap import TETRA, SQUARE, torus

ATOL = 1e-5
TYPES = (EDGE, FACE, VERTEX)


def tensors(F, types=TYPES):
    g = build_gmap(F)
    alpha = torch.tensor(np.asarray(g.alpha), dtype=torch.long)[None]
    labs = [torch.tensor(orbits(g.alpha, t), dtype=torch.long)[None] for t in types]
    return alpha, labs


def mean_ref(x, lab):
    """Oracle: loop over the orbits."""
    out = torch.empty_like(x)
    for n in range(x.shape[0]):
        for k in torch.unique(lab[n]):
            m = lab[n] == k
            out[n][:, m] = x[n][:, m].mean(1, keepdim=True)
    return out


def naive(conv, x, alpha, labs):
    W, b = conv.conv.weight[:, :, 0, :], conv.conv.bias
    K = alpha.shape[2]
    means = [mean_ref(x, l) for l in labs]
    out = torch.empty(x.shape[0], W.shape[0], x.shape[2])
    for n in range(x.shape[0]):
        for d in range(x.shape[2]):
            acc = b + W[:, :, 0] @ x[n, :, d]
            for i in range(K):
                acc = acc + W[:, :, 1 + i] @ x[n, :, alpha[n, d, i]]
            for j, mj in enumerate(means):
                acc = acc + W[:, :, 1 + K + j] @ mj[n, :, d]
            out[n, :, d] = acc
    return out


def relabel(alpha, labs, perm):
    """Renumber the darts (dart d becomes perm[d]) in alpha and in the orbit labels."""
    a = alpha[0].numpy()
    new = np.empty_like(a); new[perm] = perm[a]
    new_labs = []
    for l in labs:
        nl = np.empty_like(l[0].numpy()); nl[perm] = l[0].numpy()
        new_labs.append(torch.tensor(nl)[None])
    return torch.tensor(new)[None], new_labs


def run():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    print("OrbitDartConv:")

    # 1. orbit_mean
    alpha, labs = tensors(torus())
    D = alpha.shape[1]
    x = torch.randn(1, 4, D)
    for t, l in zip(("edge", "face", "vertex"), labs):
        got = orbit_mean(x, l)
        assert tuple(got.shape) == (1, 4, D), f"orbit_mean [{t}]: shape {tuple(got.shape)}"
        assert torch.allclose(got, mean_ref(x, l), atol=ATOL), f"orbit_mean [{t}]: wrong mean"
        ren = torch.tensor(rng.permutation(int(l.max()) + 1))[l]          # rename the labels
        assert torch.allclose(orbit_mean(x, ren), got, atol=ATOL), f"orbit_mean [{t}] depends on label NAMES"
    a1, l1 = tensors(TETRA)
    xb = torch.randn(2, 3, a1.shape[1])                                    # batch: 2 meshes with the same D
    lb = torch.cat([l1[2], l1[2][:, torch.tensor(rng.permutation(a1.shape[1]))]])
    assert torch.allclose(orbit_mean(xb, lb), mean_ref(xb, lb), atol=ATOL), "orbit_mean: batch N=2 wrong"
    print("  OK orbit_mean (exact, independent of label names, batched)")

    # 2-4. layer
    for name, F in (("tetrahedron", TETRA), ("square", SQUARE), ("torus", torus())):
        alpha, labs = tensors(F)
        D = alpha.shape[1]
        x = torch.randn(1, 5, D)
        conv = OrbitDartConv(5, 7)
        out = conv(x, alpha, labs)
        assert tuple(out.shape) == (1, 7, D), f"[{name}] output {tuple(out.shape)}"
        assert torch.allclose(out, naive(conv, x, alpha, labs), atol=ATOL), \
            f"[{name}] != naive formula (positions: [self, alpha0, alpha1, alpha2, means in orbit_types order])"
        perm = rng.permutation(D)
        x_p = torch.empty_like(x); x_p[:, :, perm] = x
        a_p, l_p = relabel(alpha, labs, perm)
        assert torch.allclose(conv(x_p, a_p, l_p)[:, :, perm], out, atol=ATOL), \
            f"[{name}] not equivariant to dart renumbering"
        print(f"  OK {name:<12} shape, naive formula, equivariance")

    alpha, _ = tensors(torus())
    x = torch.randn(1, 5, alpha.shape[1])
    oc = OrbitDartConv(5, 7, orbit_types=())
    dc = DartConv(5, 7)
    dc.conv.load_state_dict(oc.conv.state_dict())
    assert torch.allclose(oc(x, alpha, []), dc(x, alpha), atol=ATOL), "empty orbit set must give DartConv"
    print("  OK empty orbit set = DartConv")

    # 5. classifier
    alpha, labs = tensors(torus())
    D = alpha.shape[1]
    model = OrbitCNNClassifier(5, num_classes=4, widths=(8, 8))
    x = torch.randn(1, 5, D)
    perm = rng.permutation(D)
    x_p = torch.empty_like(x); x_p[:, :, perm] = x
    a_p, l_p = relabel(alpha, labs, perm)
    assert torch.allclose(model(x, alpha, labs), model(x_p, a_p, l_p), atol=ATOL), "classifier not invariant"
    print("  OK classifier invariant to renumbering")
    print("OK - OrbitDartConv verified.")


if __name__ == "__main__":
    run()

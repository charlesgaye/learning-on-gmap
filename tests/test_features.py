"""Edge features (edge_features), tested by INVARIANCE.

No external oracle: the 5 features must be invariant to translation, rotation and uniform
scaling. We transform the mesh and check that the features do not move, plus a few bounds.

    cd meshcnn-repro
    python -m tests.test_features
"""
import numpy as np

from src.mesh import Mesh, edge_features

# right-angled tetrahedron (edges with varied geometry -> non-trivial features)
V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
F = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])


def _rand_rotation(seed):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1          # keep a proper rotation (det = +1)
    return Q


def _features(vertices):
    m = Mesh(vertices=vertices, faces=F)
    m.build_edges()
    return edge_features(m)


def run():
    f0 = _features(V)
    assert tuple(f0.shape) == (6, 5), f"expected (6,5), got {tuple(f0.shape)}"

    ft = _features(V + np.array([5.0, -3.0, 2.0]))       # translation
    fr = _features(V @ _rand_rotation(1).T)              # rotation
    fs = _features(V * 7.3)                              # scaling

    assert np.allclose(f0, ft), "features NOT translation invariant"
    assert np.allclose(f0, fr), "features NOT rotation invariant"
    assert np.allclose(f0, fs), "features NOT scale invariant"

    angles = f0[:, 1:3]
    ratios = f0[:, 3:5]
    assert ((angles >= 0) & (angles <= np.pi)).all(), "angles outside [0, pi]"
    assert (ratios > 0).all(), "ratios not strictly positive"
    # symmetric pairs sorted (min, max)
    assert (f0[:, 1] <= f0[:, 2] + 1e-9).all(), "angles not sorted (min,max)"
    assert (f0[:, 3] <= f0[:, 4] + 1e-9).all(), "ratios not sorted (min,max)"

    print("OK - features invariant to translation, rotation and scaling,")
    print("     correct bounds, symmetric pairs sorted.")
    print("     example feats[0] =", np.round(f0[0], 4))


if __name__ == "__main__":
    run()

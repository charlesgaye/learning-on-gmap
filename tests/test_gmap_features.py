"""Orbit features (gmapnet.features).

Checks: shape, constancy on the RIGHT orbit, geometric identities (π per face, discrete
Gauss–Bonnet), independence from orientation (mirror symmetry), and the MeshCNN oracle:
the 5 MeshCNN edge features (src.mesh.edge_features) are recovered exactly from orbit features.

    cd meshcnn-repro
    python -m tests.test_gmap_features
"""
from pathlib import Path
import numpy as np

from gmapnet.gmap import build_gmap, orbits
from gmapnet import features as FT
from src.mesh import Mesh, edge_features
from tests.test_gmap import TETRA, SQUARE, torus

TAU = 2 * np.pi
ATOL = 1e-6


# ---------------------------------------------------------------- meshes WITH geometry
def tetra():
    return np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float), TETRA


def square():
    return np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float), SQUARE


def torus_geo(n=6, m=4, R=3.0, r=1.0):
    u = TAU * np.arange(n)[:, None] / n
    v = TAU * np.arange(m)[None, :] / m
    P = np.stack([(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u),
                  r * np.sin(v) * np.ones_like(u)], -1).reshape(-1, 3)
    return P, torus(n, m)


def shrec():
    files = sorted(Path("data/shrec_16").glob("*/train/*.obj"))
    if not files:
        return None
    import trimesh
    tm = trimesh.load(files[0], process=False)
    return np.asarray(tm.vertices, float), np.asarray(tm.faces)


# ---------------------------------------------------------------- helpers
def constant_on(x, labels):
    """Max over orbits of (max − min) of x: 0 <=> x is constant on each orbit."""
    k = labels.max() + 1
    hi = np.full(k, -np.inf); lo = np.full(k, np.inf)
    np.maximum.at(hi, labels, x); np.minimum.at(lo, labels, x)
    return float(np.max(hi - lo))


def mirrored(P):
    """Mirror symmetry: flips the geometric orientation; must change no feature."""
    return P * np.array([-1.0, 1.0, 1.0])


# ---------------------------------------------------------------- checks
def check_mesh(name, P, F, closed):
    g = build_gmap(F)
    D = g.n_darts
    lab = {k: orbits(g.alpha, k) for k in [(1,), (0,), (0, 2), (1, 2), (0, 1)]}

    # 1. corner angle: constant on <alpha1>, angles of a face sum to pi
    x = FT.corner_angle(g, P)
    assert x.shape == (D,), f"[{name}] corner_angle: shape {x.shape} instead of ({D},)"
    c = constant_on(x, lab[(1,)])
    assert c < ATOL, f"[{name}] corner_angle not constant on corners <alpha1> (gap {c:.2e})"
    per_face = np.bincount(lab[(0, 1)], weights=x) / 2        # each corner counted twice
    err = np.max(np.abs(per_face - np.pi))
    assert err < ATOL, f"[{name}] corner_angle: the angles of a face do not sum to pi (gap {err:.2e})"

    # 2. incidence ratio: constant on <alpha0>, positive
    x = FT.incidence_ratio(g, P)
    assert x.shape == (D,), f"[{name}] incidence_ratio: shape {x.shape}"
    c = constant_on(x, lab[(0,)])
    assert c < ATOL, f"[{name}] incidence_ratio not constant on <alpha0> (gap {c:.2e})"
    assert np.all(x > 0) and np.all(np.isfinite(x)), f"[{name}] incidence_ratio must be > 0 and finite"

    # 3. dihedral: constant on <alpha0,alpha2>, finite, mirror invariant, pi on a plane
    x = FT.dihedral(g, P)
    assert x.shape == (D,), f"[{name}] dihedral: shape {x.shape}"
    assert np.all(np.isfinite(x)), f"[{name}] dihedral: non-finite values (boundary?)"
    c = constant_on(x, lab[(0, 2)])
    assert c < ATOL, f"[{name}] dihedral not constant on edges <alpha0,alpha2> (gap {c:.2e})"
    xm = FT.dihedral(g, mirrored(P))
    assert np.allclose(x, xm, atol=ATOL), f"[{name}] dihedral changes under a mirror symmetry (orientation dependent)"
    if name == "square":
        inner = g.alpha[:, 2] != np.arange(D)
        assert np.allclose(x[inner], np.pi, atol=ATOL), "[square] interior dihedral of a plane must be pi"

    # 4. angle defect: constant on <alpha1,alpha2>, Gauss-Bonnet on closed surfaces
    x = FT.angle_defect(g, P)
    assert x.shape == (D,), f"[{name}] angle_defect: shape {x.shape}"
    c = constant_on(x, lab[(1, 2)])
    assert c < ATOL, f"[{name}] angle_defect not constant on vertices <alpha1,alpha2> (gap {c:.2e})"
    if closed:
        vlab = lab[(1, 2)]
        first = np.unique(vlab, return_index=True)[1]         # one dart per vertex
        total = x[first].sum()
        chi = (vlab.max() + 1) - (lab[(0, 2)].max() + 1) + (lab[(0, 1)].max() + 1)
        assert abs(total - TAU * chi) < 1e-5, \
            f"[{name}] Gauss-Bonnet: sum of defects = {total:.6f}, expected 2*pi*chi = {TAU * chi:.6f}"

    # 5. MeshCNN oracle (closed surfaces)
    if closed:
        x = FT.meshcnn_features(g, P)
        elab = lab[(0, 2)]
        E = elab.max() + 1
        assert x.shape == (E, 5), f"[{name}] meshcnn_features: shape {x.shape} instead of ({E},5)"
        m = Mesh(vertices=P, faces=np.asarray(F)); m.build_edges()
        ref = edge_features(m)
        ref_of = {tuple(sorted(map(int, e))): ref[k] for k, e in enumerate(m.edges)}
        first = np.unique(elab, return_index=True)[1]
        for k, d in enumerate(first):
            key = tuple(sorted((int(g.vertex[d]), int(g.vertex[g.alpha[d, 0]]))))
            assert np.allclose(x[k], ref_of[key], atol=1e-5), \
                f"[{name}] edge {key}: {np.round(x[k], 4)} != MeshCNN {np.round(ref_of[key], 4)}"

    checks = "corner, ratio, dihedral, defect" + (", MeshCNN oracle" if closed else "")
    print(f"  OK {name:<12} {checks}")


def run():
    print("Orbit features:")
    cases = [("tetrahedron", *tetra(), True), ("square", *square(), False), ("torus", *torus_geo(), True)]
    s = shrec()
    if s is not None:
        cases.append(("shrec", *s, True))
    for name, P, F, closed in cases:
        check_mesh(name, P, F, closed)
    print("OK - orbit features verified; MeshCNN features recovered as a special case.")


if __name__ == "__main__":
    run()

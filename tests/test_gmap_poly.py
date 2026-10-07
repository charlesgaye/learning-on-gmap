"""Polygon mesh -> 2-G-map connector (gmapnet.gmap.build_gmap on faces of any size).

Same philosophy as tests/test_gmap.py: the tests ARE the G-map axioms, the meaning of each
involution and the cell counts (V, E, F, χ). No dart-numbering convention is assumed.

Faces are given as a list of vertex sequences (ragged); an (F, 3) array must still work.

Meshes: single quad and pentagon (boundary, χ=1), quad + triangle (mixed, boundary, χ=1),
cube of quads (closed, χ=2), square pyramid (mixed, closed, χ=2), quad torus (χ=0),
quad Möbius strip (boundary, non-orientable), quad Klein bottle (CLOSED, non-orientable:
non-orientability is detected without help from a boundary), a non-manifold quad case
that must raise ValueError, and the triangle meshes of tests/test_gmap.py as regression.

    cd meshcnn-repro
    python -m tests.test_gmap_poly
"""
from pathlib import Path
import numpy as np

from gmapnet import gmap as G
from gmapnet.gmap import orbits
from tests.test_gmap import TETRA, SQUARE, torus, mobius


# ---------------------------------------------------------------- meshes
QUAD = [[0, 1, 2, 3]]
PENTAGON = [[0, 1, 2, 3, 4]]
HOUSE = [[0, 1, 2, 3], [3, 2, 4]]                       # quad + triangle sharing edge (2,3)
CUBE = [[0, 1, 2, 3], [4, 5, 6, 7],
        [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
PYRAMID = [[0, 1, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
NONMANIFOLD_QUAD = [[0, 1, 2, 3], [0, 1, 4, 5], [0, 1, 6, 7]]   # edge (0,1) has 3 faces


def quad_torus(n=4, m=3):
    """n x m grid glued in both directions, 1 quad per cell."""
    idx = lambda i, j: (i % n) * m + (j % m)
    return [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
            for i in range(n) for j in range(m)]


def quad_mobius(n=5):
    """Strip of n quads, width 1; the last cell glues column 0 back FLIPPED."""
    top = lambda i: i
    bot = lambda i: n + i
    F = []
    for i in range(n):
        t0, b0 = top(i), bot(i)
        t1, b1 = (top(i + 1), bot(i + 1)) if i < n - 1 else (bot(0), top(0))
        F.append([t0, b0, b1, t1])
    return F


def quad_klein(n=4, m=3):
    """n x m grid; glued normally along i, glued with the flip i -> -i when j wraps around."""
    def idx(i, j):
        if j % m == 0 and j != 0:          # j = m: crossing the twisted seam
            i = -i
        return (i % n) * m + (j % m)
    return [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
            for i in range(n) for j in range(m)]


# ---------------------------------------------------------------- mesh-side counts
def sides(f):
    """Undirected edges of a polygon face (consecutive vertices, cyclically)."""
    f = [int(v) for v in f]
    return [tuple(sorted((a, b))) for a, b in zip(f, f[1:] + f[:1])]


def edge_set(F):
    return {e for f in F for e in sides(f)}


def boundary_edges(F):
    cnt = {}
    for f in F:
        for e in sides(f):
            cnt[e] = cnt.get(e, 0) + 1
    return sum(1 for c in cnt.values() if c == 1)


# ---------------------------------------------------------------- checks
def check(name, F):
    F = [list(map(int, f)) for f in F]
    V_ = len({v for f in F for v in f})
    E_ = len(edge_set(F))
    F_ = len(F)
    nb = boundary_edges(F)
    D = sum(2 * len(f) for f in F)                     # 2 darts per corner

    g = G.build_gmap(F)
    a, vx, fc = np.asarray(g.alpha), np.asarray(g.vertex), np.asarray(g.face)
    ar = np.arange(D)

    # 1. shapes
    assert a.shape == (D, 3), f"[{name}] alpha must be ({D},3), got {a.shape}"
    assert vx.shape == (D,) and fc.shape == (D,), f"[{name}] vertex/face must be ({D},), got {vx.shape}; {fc.shape}"
    assert np.issubdtype(a.dtype, np.integer), f"[{name}] alpha must be integer"
    assert a.min() >= 0 and a.max() < D, f"[{name}] dart indices outside [0,{D}), got {a.min()}; {a.max()}"

    # 2. involutions
    for i in range(3):
        assert np.array_equal(a[a[:, i], i], ar), f"[{name}] alpha{i} is not an involution"

    # 3. fixed points: never for alpha0, alpha1; for alpha2 exactly on boundary darts
    assert not np.any(a[:, 0] == ar), f"[{name}] alpha0 has a fixed point"
    assert not np.any(a[:, 1] == ar), f"[{name}] alpha1 has a fixed point"
    fix2 = int(np.sum(a[:, 2] == ar))
    assert fix2 == 2 * nb, f"[{name}] alpha2: {fix2} fixed points, expected {2 * nb} (2 per boundary edge)"

    # 4. G-map axiom: alpha0 alpha2 is an involution (<=> alpha0 and alpha2 commute)
    c = a[a[:, 2], 0]
    assert np.array_equal(c[c], ar), f"[{name}] alpha0 alpha2 is not an involution"

    # 5. meaning of the involutions through the embedding (vertex, edge, face)
    edge = lambda d: frozenset((int(vx[d]), int(vx[a[d, 0]])))
    for d in range(D):
        e0, e1, e2 = a[d]
        assert fc[e0] == fc[d] and vx[e0] != vx[d] and edge(e0) == edge(d), \
            f"[{name}] alpha0({d}) must keep face+edge and change vertex"
        assert fc[e1] == fc[d] and vx[e1] == vx[d] and edge(e1) != edge(d), \
            f"[{name}] alpha1({d}) must keep face+vertex and change edge"
        if e2 != d:
            assert vx[e2] == vx[d] and edge(e2) == edge(d) and fc[e2] != fc[d], \
                f"[{name}] alpha2({d}) must keep vertex+edge and change face"
        # the edge of a dart is a SIDE of its face (catches α0 pairing non-consecutive corners)
        assert tuple(sorted(edge(d))) in sides(F[fc[d]]), \
            f"[{name}] dart {d}: edge {tuple(sorted(edge(d)))} is not a side of face {F[fc[d]]}"

    # 6. each face of size k: 2k darts, each of its k vertices carried exactly twice
    for f in range(F_):
        ds = np.where(fc == f)[0]
        k = len(F[f])
        assert len(ds) == 2 * k, f"[{name}] face {f}: {len(ds)} darts instead of {2 * k}"
        vs, cnt = np.unique(vx[ds], return_counts=True)
        assert set(vs) == set(F[f]) and np.all(cnt == 2), \
            f"[{name}] face {f}: dart vertices {vx[ds]} inconsistent with {F[f]}"
    assert set(edge(d) for d in range(D)) == {frozenset(e) for e in edge_set(F)}, \
        f"[{name}] the darts' edges do not match the mesh edges"

    # 7. orbits = cells; a face orbit has 2k darts
    nV = orbits(a, (1, 2)).max() + 1
    nE = orbits(a, (0, 2)).max() + 1
    nF = orbits(a, (0, 1)).max() + 1
    assert (nV, nE, nF) == (V_, E_, F_), \
        f"[{name}] orbits (V,E,F) = {(nV, nE, nF)}, expected {(V_, E_, F_)}"
    lab = orbits(a, (1, 2))
    for k in range(nV):
        assert len(np.unique(vx[lab == k])) == 1, f"[{name}] vertex orbit {k} mixes several vertices"
    lab = orbits(a, (0, 1))
    for k in range(nF):
        f = int(fc[lab == k][0])
        assert np.sum(lab == k) == 2 * len(F[f]), f"[{name}] face orbit {k} does not have {2 * len(F[f])} darts"

    chi = nV - nE + nF
    print(f"  OK {name:<14} D={D:<5} (V,E,F)=({nV},{nE},{nF})  chi={chi:>2}  boundary edges={nb}")
    return g, chi


def run():
    print("Polygon mesh -> 2-G-map connector:")
    for name, F, exp_chi in (("quad", QUAD, 1), ("pentagon", PENTAGON, 1), ("house", HOUSE, 1),
                             ("cube", CUBE, 2), ("pyramid", PYRAMID, 2),
                             ("quad torus", quad_torus(), 0), ("quad mobius", quad_mobius(), 0),
                             ("quad klein", quad_klein(), 0),
                             ("tri tetra", TETRA, 2), ("tri square", SQUARE, 1),
                             ("tri torus", torus(), 0), ("tri mobius", mobius(), 0)):
        _, chi = check(name, F)
        assert chi == exp_chi, f"[{name}] chi = {chi}, expected {exp_chi}"

    # an (F, 3) numpy array must still be accepted (data.py passes trimesh faces)
    check("tri tetra (array)", np.asarray(TETRA))

    try:
        G.build_gmap(NONMANIFOLD_QUAD)
    except ValueError:
        print("  OK non-manifold quad -> ValueError")
    else:
        raise AssertionError("[non-manifold quad] an edge with 3 faces must raise ValueError")

    shrec = sorted(Path("data/shrec_16").glob("*/train/*.obj"))
    if shrec:
        import trimesh
        tm = trimesh.load(shrec[0], process=False)
        check(f"shrec:{shrec[0].stem}", np.asarray(tm.faces))
    else:
        print("  (SHREC not found: real-mesh check skipped)")

    # Orientability. The Klein bottle is CLOSED: a "no" there cannot come from the boundary.
    for name, F, exp in (("cube", CUBE, True), ("pyramid", PYRAMID, True), ("house", HOUSE, True),
                         ("quad torus", quad_torus(), True), ("quad mobius", quad_mobius(), False),
                         ("quad klein", quad_klein(), False)):
        got = G.is_orientable(G.build_gmap(F))
        assert got == exp, f"[{name}] is_orientable = {got}, expected {exp}"
    print("  OK orientability (cube, pyramid, house, torus: yes; Mobius, Klein: no)")

    print("OK - all G-map axioms and cell counts verified on polygon meshes.")


if __name__ == "__main__":
    run()

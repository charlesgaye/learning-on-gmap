"""Triangle mesh -> 2-G-map connector (gmapnet.gmap.build_gmap).

The tests ARE the G-map axioms, plus the meaning of each involution and the cell counts
(V, E, F, χ). No dart-numbering convention is assumed.

Meshes: tetrahedron (closed, χ=2), square (boundary, χ=1), torus (χ=0), Möbius strip
(boundary + NON-orientable, χ=0), a SHREC mesh if available, and a non-manifold case that must
raise ValueError. Orientability is checked on the same meshes.

    cd meshcnn-repro
    python -m tests.test_gmap
"""
from pathlib import Path
import numpy as np

from gmapnet import gmap as G
from gmapnet.gmap import orbits


# ---------------------------------------------------------------- meshes
TETRA = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
SQUARE = np.array([[0, 1, 2], [0, 2, 3]])
NONMANIFOLD = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])   # edge (0,1) has 3 faces


def torus(n=4, m=3):
    """n x m grid glued in both directions, 2 triangles per cell."""
    idx = lambda i, j: (i % n) * m + (j % m)
    F = []
    for i in range(n):
        for j in range(m):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            F += [[a, b, c], [a, c, d]]
    return np.array(F)


def mobius(n=5):
    """Strip of n cells, width 1; the last cell glues column 0 back FLIPPED."""
    top = lambda i: i
    bot = lambda i: n + i
    F = []
    for i in range(n):
        t0, b0 = top(i), bot(i)
        t1, b1 = (top(i + 1), bot(i + 1)) if i < n - 1 else (bot(0), top(0))
        F += [[t0, b0, t1], [b0, b1, t1]]
    return np.array(F)


def edge_set(F):
    return {tuple(sorted((int(a), int(b)))) for f in F for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))}


def boundary_edges(F):
    cnt = {}
    for f in F:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            k = tuple(sorted((int(a), int(b))))
            cnt[k] = cnt.get(k, 0) + 1
    return sum(1 for c in cnt.values() if c == 1)


# ---------------------------------------------------------------- checks
def check(name, F):
    F = np.asarray(F)
    V_ = len(np.unique(F))
    E_ = len(edge_set(F))
    F_ = len(F)
    nb = boundary_edges(F)

    g = G.build_gmap(F)
    a, vx, fc = np.asarray(g.alpha), np.asarray(g.vertex), np.asarray(g.face)
    D = 6 * F_
    ar = np.arange(D)

    # 1. shapes
    assert a.shape == (D, 3), f"[{name}] alpha must be ({D},3), got {a.shape}"
    assert vx.shape == (D,) and fc.shape == (D,), f"[{name}] vertex/face must be ({D},)"
    assert np.issubdtype(a.dtype, np.integer), f"[{name}] alpha must be integer"
    assert a.min() >= 0 and a.max() < D, f"[{name}] dart indices outside [0,{D})"

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

    # 6. each face: 6 darts, each of its 3 vertices carried exactly twice
    for f in range(F_):
        ds = np.where(fc == f)[0]
        assert len(ds) == 6, f"[{name}] face {f}: {len(ds)} darts instead of 6"
        vs, cnt = np.unique(vx[ds], return_counts=True)
        assert set(vs) == set(int(x) for x in F[f]) and np.all(cnt == 2), \
            f"[{name}] face {f}: dart vertices {vx[ds]} inconsistent with {F[f]}"
    assert set(edge(d) for d in range(D)) == {frozenset(e) for e in edge_set(F)}, \
        f"[{name}] the darts' edges do not match the mesh edges"

    # 7. orbits = cells
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
        assert np.sum(lab == k) == 6, f"[{name}] face orbit {k} does not have 6 darts"

    chi = nV - nE + nF
    print(f"  OK {name:<12} D={D:<5} (V,E,F)=({nV},{nE},{nF})  chi={chi:>2}  boundary edges={nb}")
    return g, chi


def run():
    print("Mesh -> 2-G-map connector:")
    _, chi = check("tetrahedron", TETRA);  assert chi == 2
    _, chi = check("square", SQUARE);      assert chi == 1
    _, chi = check("torus", torus());      assert chi == 0
    _, chi = check("mobius", mobius());    assert chi == 0

    try:
        G.build_gmap(NONMANIFOLD)
    except ValueError:
        print("  OK non-manifold -> ValueError")
    else:
        raise AssertionError("[non-manifold] an edge with 3 faces must raise ValueError")

    shrec = sorted(Path("data/shrec_16").glob("*/train/*.obj"))
    if shrec:
        import trimesh
        tm = trimesh.load(shrec[0], process=False)
        check(f"shrec:{shrec[0].stem}", np.asarray(tm.faces))
    else:
        print("  (SHREC not found: real-mesh check skipped)")

    # Orientability: tetrahedron, square (with boundary) and torus yes; Mobius strip no.
    for name, F, exp in (("tetrahedron", TETRA, True), ("square", SQUARE, True),
                         ("torus", torus(), True), ("mobius", mobius(), False)):
        got = G.is_orientable(G.build_gmap(F))
        assert got == exp, f"[{name}] is_orientable = {got}, expected {exp}"
    print("  OK orientability (tetrahedron, square, torus: yes; Mobius: no)")

    print("OK - all G-map axioms and cell counts verified.")


if __name__ == "__main__":
    run()

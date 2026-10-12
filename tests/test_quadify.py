"""Triangle mesh -> mixed triangle/quad mesh (gmapnet.quadify.quadify).

Expected API:
    quadify(P, F, min_dihedral=np.deg2rad(160)) -> list of faces (3 or 4 vertex indices)
Pairs of adjacent triangles are merged into a quad when
    (1) the dihedral angle of their shared edge, measured on the TRIANGLE mesh, is >= min_dihedral
        (the edge disappears with its dihedral: only nearly flat edges may be removed);
    (2) the resulting quad is convex: every corner angle < π, in the sense of
        features.corner_angle on the quad actually produced (exact even for non-planar pairs).
Candidates are taken greedily, flattest shared edge first, with a deterministic tie-break.

Checks (no vertex-order convention imposed on the quads):
    1. the output is a mesh of triangles and quads that PARTITIONS the input triangles: every
       quad is the union of two input triangles split along one of its diagonals, every
       output triangle is an input triangle, every input triangle is used exactly once;
    2. every quad side is an input edge, its diagonal an interior input edge;
    3. counts: V unchanged, E and F decrease by the number of quads, χ unchanged,
       D = (darts of the triangle mesh) − 4 per quad; build_gmap accepts the output;
    4. the two criteria hold for every quad, and the result is MAXIMAL: no two adjacent
       output triangles could still be merged;
    5. greedy order (flattest first), determinism, and the threshold behaves as a threshold;
    6. exact cases: a grid of planar squares folded along the grid lines merges exactly into
       its squares; a 90° fold and a planar but non-convex pair are not merged.

    cd meshcnn-repro
    python -m tests.test_quadify
"""
from pathlib import Path
import numpy as np

from gmapnet.gmap import build_gmap, orbits
from gmapnet import features as FT
from gmapnet.quadify import quadify
from tests.test_gmap_features_poly import tetra, tri_torus, shrec

MIN = np.deg2rad(160)
ATOL = 1e-9


# ---------------------------------------------------------------- meshes
def flat_tri_grid(n=4, m=3):
    """Planar n x m grid of unit squares, each split by its diagonal into 2 triangles."""
    idx = lambda i, j: i * (m + 1) + j
    P = np.array([[i, j, 0] for i in range(n + 1) for j in range(m + 1)], float)
    F = []
    for i in range(n):
        for j in range(m):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            F += [[a, b, c], [a, c, d]]
    return P, np.array(F)


def zigzag_tri_grid(n=4, m=3, h=0.8):
    """Same grid with z = h·(i%2 + j%2): every square stays PLANAR (its diagonal is flat, 180°)
    but neighbouring squares fold by ~116° along the grid lines. Only the diagonals pass the
    threshold, so the expected result (all unit squares) does not depend on any tie-break."""
    P, F = flat_tri_grid(n, m)
    P[:, 2] = h * (P[:, 0].astype(int) % 2 + P[:, 1].astype(int) % 2)
    return P, F


def fold_90():
    """Two triangles sharing edge (0,1), folded at 90 degrees."""
    P = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0, 1]], float)
    return P, np.array([[0, 1, 2], [1, 0, 3]])


def planar_nonconvex():
    """Planar pair whose union a-c-b-d has a reflex corner at a (45° + 168.7° > 180°)."""
    P = np.array([[0, 0, 0], [2, 0, 0], [1, 1, 0], [-1, -0.2, 0]], float)
    return P, np.array([[0, 1, 2], [1, 0, 3]])


def strip3():
    """Three triangles in a row t0 - t1 - t2: edge (t0,t1) bent by 15°, edge (t1,t2) by 5°.
    Both pass a 160° threshold, but t1 can join only one of them: the flatter, t2."""
    P = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, -0.6, 0]], float)
    F = np.array([[0, 1, 4], [0, 1, 2], [1, 3, 2]])          # t0 below (0,1), t1, t2 across (1,2)
    t = np.deg2rad(15)                                     # lift the tip of t0: fold of 15° on (0,1)
    P[4] = [0.5, -0.6 * np.cos(t), 0.6 * np.sin(t)]
    axis = (P[2] - P[1]) / np.linalg.norm(P[2] - P[1])    # fold t2 by 5° around (1,2)
    v, s = P[3] - P[1], np.deg2rad(5)
    P[3] = P[1] + v * np.cos(s) + np.cross(axis, v) * np.sin(s) + axis * (axis @ v) * (1 - np.cos(s))
    return P, F


# ---------------------------------------------------------------- helpers
def edges_of(faces):
    return {tuple(sorted((int(f[i]), int(f[(i + 1) % len(f)])))) for f in faces for i in range(len(f))}


def tri_dihedral_by_edge(P, F):
    """Dihedral of each edge of the triangle mesh, keyed by the sorted vertex pair."""
    g = build_gmap(F)
    x = FT.dihedral(g, P)
    return {tuple(sorted((int(g.vertex[d]), int(g.vertex[g.alpha[d, 0]])))): float(x[d]) for d in range(g.n_darts)}


def split(q):
    """The two diagonal splits of a quad, as pairs of triangles (frozensets) + diagonal."""
    a, b, c, d = map(int, q)
    return [({frozenset((a, b, c)), frozenset((a, c, d))}, (a, c)),
            ({frozenset((b, c, d)), frozenset((b, d, a))}, (b, d))]


def chi(faces):
    V = len({int(v) for f in faces for v in f})
    return V - len(edges_of(faces)) + len(faces)


# ---------------------------------------------------------------- checks
def check(name, P, F, min_dihedral=MIN):
    F = np.asarray(F)
    out = quadify(P, F, min_dihedral=min_dihedral)
    tris_in = [frozenset(map(int, t)) for t in F]
    tri_set = set(tris_in)
    dih = tri_dihedral_by_edge(P, F)
    interior = {e for e, c in _edge_counts(F).items() if c == 2}

    assert all(len(f) in (3, 4) for f in out), f"[{name}] output faces must have 3 or 4 vertices"
    quads = [list(map(int, f)) for f in out if len(f) == 4]
    tris = [frozenset(map(int, f)) for f in out if len(f) == 3]

    # 1-2. partition of the input triangles; quad sides are input edges, diagonal interior
    used = list(tris)
    for q in quads:
        ok = [(pair, diag) for pair, diag in split(q) if pair <= tri_set]
        assert len(ok) >= 1, f"[{name}] quad {q} is not two input triangles split along a diagonal"
        pair, diag = ok[0]
        used += list(pair)
        for e in edges_of([q]):
            assert e in edges_of(F), f"[{name}] quad {q}: side {e} is not an input edge (bad vertex order?)"
        assert tuple(sorted(diag)) in interior, f"[{name}] quad {q}: diagonal {diag} is not an interior input edge"
    assert sorted(map(sorted, used)) == sorted(map(sorted, tris_in)), \
        f"[{name}] the output does not use every input triangle exactly once"

    # 3. counts, χ, G-map
    nq = len(quads)
    V_in, E_in = len({int(v) for v in F.ravel()}), len(edges_of(F))
    assert len({v for f in out for v in map(int, f)}) == V_in, f"[{name}] V must not change"
    assert len(edges_of(out)) == E_in - nq, f"[{name}] E must decrease by the number of quads"
    assert len(out) == len(F) - nq, f"[{name}] F must decrease by the number of quads"
    assert chi(out) == chi(F), f"[{name}] chi changed: {chi(F)} -> {chi(out)}"
    g = build_gmap(out)
    assert g.n_darts == 6 * len(F) - 4 * nq, f"[{name}] D = {g.n_darts}, expected {6 * len(F) - 4 * nq}"

    # 4. criteria and maximality
    for q in quads:
        diag = [d for pair, d in split(q) if pair <= tri_set][0]
        assert dih[tuple(sorted(diag))] >= min_dihedral - ATOL, \
            f"[{name}] quad {q} removed an edge of dihedral {np.rad2deg(dih[tuple(sorted(diag))]):.1f}° < threshold"
    ca = FT.corner_angle(g, P)
    is_quad = np.bincount(g.face)[g.face] == 8
    assert np.all(ca[is_quad] < np.pi + ATOL), f"[{name}] a quad has a reflex corner"
    _assert_maximal(name, P, tris, dih, min_dihedral)

    # 5. determinism
    out2 = quadify(P, F, min_dihedral=min_dihedral)
    assert [list(map(int, f)) for f in out] == [list(map(int, f)) for f in out2], f"[{name}] quadify is not deterministic"

    frac = 2 * nq / len(F)
    print(f"  OK {name:<16} {nq} quads, {len(tris)} triangles left, {100 * frac:.0f}% of triangles merged")
    return out, frac


def _edge_counts(F):
    cnt = {}
    for f in F:
        for i in range(3):
            e = tuple(sorted((int(f[i]), int(f[(i + 1) % 3]))))
            cnt[e] = cnt.get(e, 0) + 1
    return cnt


def _quad_is_convex(P, t1, t2, shared):
    """Corners of the would-be quad, via a G-map of the single quad (reuses corner_angle)."""
    a, b = shared
    c = next(iter(t1 - {a, b}))
    d = next(iter(t2 - {a, b}))
    idx = [a, c, b, d]
    g = build_gmap([[0, 1, 2, 3]])
    return bool(np.all(FT.corner_angle(g, P[idx]) < np.pi))


def _assert_maximal(name, P, tris, dih, min_dihedral):
    by_edge = {}
    for t in tris:
        s = sorted(t)
        for e in ((s[0], s[1]), (s[1], s[2]), (s[0], s[2])):
            by_edge.setdefault(e, []).append(t)
    for e, ts in by_edge.items():
        if len(ts) == 2 and dih[e] >= min_dihedral - ATOL and _quad_is_convex(P, ts[0], ts[1], e):
            raise AssertionError(f"[{name}] not maximal: triangles {sorted(ts[0])} and {sorted(ts[1])} "
                                 f"could still be merged (edge {e}, {np.rad2deg(dih[e]):.1f}°)")


def run():
    print("Triangle mesh -> mixed triangle/quad mesh:")

    check("flat grid", *flat_tri_grid())            # all edges flat: ties everywhere, invariants only
    P, F = zigzag_tri_grid()
    out, frac = check("zigzag grid", P, F)
    assert frac == 1.0 and all(len(f) == 4 for f in out), "[zigzag grid] every pair must merge into its square"
    squares = {frozenset(map(int, f)) for f in out}
    cells = {frozenset((int(a), int(b), int(c))) | {int(d)} for (a, b, c), (_, _, d) in zip(F[0::2], F[1::2])}
    assert squares == cells, "[zigzag grid] the quads must be the grid squares"

    out, _ = check("fold 90", *fold_90())
    assert all(len(f) == 3 for f in out), "[fold 90] a 90° fold must not be merged"
    out, _ = check("planar nonconvex", *planar_nonconvex())
    assert all(len(f) == 3 for f in out), "[planar nonconvex] a reflex quad must not be produced"

    P, F = strip3()
    out, _ = check("strip3", P, F)
    q = [set(map(int, f)) for f in out if len(f) == 4]
    assert q == [{0, 1, 2, 3}], f"[strip3] the flatter edge (1,2) must win: got quads {q}"

    out, _ = check("tetra", *tetra())
    assert all(len(f) == 3 for f in out), "[tetra] no edge of a tetrahedron is flat"
    check("tri torus", *tri_torus())

    # the threshold behaves as a threshold
    P, F = flat_tri_grid()
    out, _ = check("grid, no merge", P, F, min_dihedral=np.pi + 1e-3)
    assert all(len(f) == 3 for f in out), "[grid, no merge] a threshold above pi must merge nothing"

    s = shrec()
    if s is not None:
        check("shrec", *s)
    else:
        print("  (SHREC not found: real-mesh check skipped)")

    # Regression: nearly flat but NON-planar pairs where "sum of the two triangle angles < π"
    # accepted a quad whose corner_angle is reflex (196.5° in T547 at 160°, 213.5° in T54 at 140°).
    # The convexity test must agree with corner_angle on the quad actually produced.
    import trimesh
    for path, deg in (("data/shrec_16/alien/test/T547.obj", 160), ("data/shrec_16/alien/test/T547.obj", 140),
                      ("data/shrec_16/armadillo/train/T54.obj", 140)):
        if Path(path).exists():
            tm = trimesh.load(path, process=False)
            check(f"{Path(path).stem} @{deg}°", np.asarray(tm.vertices, float), np.asarray(tm.faces),
                  min_dihedral=np.deg2rad(deg))

    print("OK - quadify verified: partition, counts, criteria, maximality, greedy order, determinism.")


if __name__ == "__main__":
    run()

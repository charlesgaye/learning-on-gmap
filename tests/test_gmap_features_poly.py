"""Orbit features on POLYGON meshes (gmapnet.features).

Expected API (all (g, P) -> per-dart arrays):
    face_normal(g, P)     (D, 3) vector area ½ Σ p_i × p_{i+1} of the dart's face, the face being
                          traversed FROM the dart (vertex(d) -> vertex(α₀(d)) -> …). |n| = area
                          for a planar face; n(α₀(d)) = n(α₁(d)) = −n(d).
    corner_angle(g, P)    (D,)  inner angle in ]0, 2π[: > π on a reflex corner.
    incidence_ratio(g, P) (D,)  |AB|² / mean(|AB × AC|, |BA × BC'|), C and C' the other
                          neighbours of A and B in the face.
    dihedral(g, P)        (D,)  angle between n(d) and n(α₂(d)); π on a plane, 0 on a boundary.
    angle_defect(g, P)    (D,)  2π − Σ corner angles around the vertex.

Checks:
    1. triangle REGRESSION: on triangle meshes the new features equal the original ones
       (frozen copies below) — so the triangle results stay valid without new runs;
    2. constancy on the right orbit, on polygon meshes, planar AND non-planar;
    3. exact values: trapezoid ratio (the case where the old ratio was not constant), reflex
       corner, interior angles of a planar k-gon sum to (k−2)π, dihedral π on a plane and
       π/2 on a cube, 0 on a boundary;
    4. discrete Gauss–Bonnet on closed meshes with planar faces, including NON-CONVEX faces;
    5. independence from the input orientation (faces given in reversed order) and from a
       mirror symmetry; a Möbius strip of quads runs without NaN.

    cd meshcnn-repro
    python -m tests.test_gmap_features_poly
"""
from pathlib import Path
import numpy as np

from gmapnet.gmap import build_gmap, orbits
from gmapnet import features as FT
from tests.test_gmap import TETRA, SQUARE, torus
from tests.test_gmap_poly import CUBE, PYRAMID, quad_torus

TAU = 2 * np.pi
ATOL = 1e-6


# ---------------------------------------------------------------- frozen original features (triangles only)
def _old_corner_angle(g, P):
    d = np.arange(g.n_darts)
    s = P[g.vertex[d]]
    A, B, C = s[g.alpha[d, 0]], s[FT.walk(g, d, 1, 0)], s[d]
    return np.arccos(np.clip(np.sum((C - A) * (C - B), 1)
                             / (np.linalg.norm(C - A, axis=1) * np.linalg.norm(C - B, axis=1)), -1, 1))


def _old_incidence_ratio(g, P):
    d = np.arange(g.n_darts)
    s = P[g.vertex[d]]
    A, B, C = s[d], s[g.alpha[d, 0]], s[FT.walk(g, d, 1, 0)]
    return np.linalg.norm(A - B, axis=1) ** 2 / np.linalg.norm(np.cross(B - A, C - A), axis=1)


def _old_dihedral(g, P):
    d = np.arange(g.n_darts)
    s = P[g.vertex[d]]
    A, B, C, D = s[d], s[g.alpha[d, 0]], s[FT.walk(g, d, 1, 0)], s[FT.walk(g, d, 2, 1, 0)]
    n1, n2 = np.cross(A - B, A - C), np.cross(A - B, B - D)
    return np.arccos(np.clip(np.sum(n1 * n2, 1) / (np.linalg.norm(n1, axis=1) * np.linalg.norm(n2, axis=1)), -1, 1))


# ---------------------------------------------------------------- meshes with geometry
def tetra():
    return np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float), TETRA


def tri_square():
    return np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float), SQUARE


def tri_torus(n=6, m=4, R=3.0, r=1.0):
    return torus_points(n, m, R, r), torus(n, m)


def torus_points(n, m, R=3.0, r=1.0):
    u = TAU * np.arange(n)[:, None] / n
    v = TAU * np.arange(m)[None, :] / m
    return np.stack([(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u),
                     r * np.sin(v) * np.ones_like(u)], -1).reshape(-1, 3)


def shrec():
    files = sorted(Path("data/shrec_16").glob("*/train/*.obj"))
    if not files:
        return None
    import trimesh
    tm = trimesh.load(files[0], process=False)
    return np.asarray(tm.vertices, float), np.asarray(tm.faces)


def trapezoid():
    """A(0,0) B(2,0) C(2,1) D(0,3): CD not parallel to AB, heights of C and D differ (1 vs 3)."""
    return np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 3, 0]], float), [[0, 1, 2, 3]]


def arrow():
    """Planar non-convex quad, reflex corner at C (vertex 2)."""
    return np.array([[0, 0, 0], [4, 0, 0], [1, 1, 0], [0, 4, 0]], float), [[0, 1, 2, 3]]


def pentagon():
    t = TAU * np.arange(5) / 5
    return np.stack([np.cos(t), np.sin(t), np.zeros(5)], 1), [[0, 1, 2, 3, 4]]


def cube():
    P = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    return P, CUBE


def cube_bent():
    """Cube with one vertex pushed off: three of its quads become NON-planar."""
    P, F = cube()
    P = P.copy(); P[6] += [0.3, -0.2, 0.4]
    return P, F


def pyramid():
    P = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]], float)
    return P, PYRAMID


def l_prism():
    """Extruded L-shape: closed, planar faces, two NON-convex hexagons (reflex corner at (1,1))."""
    L = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    P = np.array([[x, y, z] for z in (0, 1) for x, y in L], float)
    F = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]] + [[i, (i + 1) % 6, (i + 1) % 6 + 6, i + 6] for i in range(6)]
    return P, F


def quad_torus_geo(n=8, m=6):
    return torus_points(n, m), quad_torus(n, m)


def flat_grid(n=3, m=2):
    """Planar n x m grid of quads (boundary)."""
    idx = lambda i, j: i * (m + 1) + j
    P = np.array([[i, j, 0] for i in range(n + 1) for j in range(m + 1)], float)
    F = [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)] for i in range(n) for j in range(m)]
    return P, F


def mobius_geo(n=12):
    """Möbius strip of n quads embedded in R³ (non-orientable, boundary)."""
    t = TAU * np.arange(n) / n
    def pt(theta, s):
        return [(2 + s * np.cos(theta / 2)) * np.cos(theta), (2 + s * np.cos(theta / 2)) * np.sin(theta),
                s * np.sin(theta / 2)]
    P = np.array([pt(a, 0.5) for a in t] + [pt(a, -0.5) for a in t], float)
    F = []
    for i in range(n):
        t0, b0 = i, n + i
        t1, b1 = (i + 1, n + i + 1) if i < n - 1 else (n, 0)     # flipped gluing
        F.append([t0, b0, b1, t1])
    return P, F


# ---------------------------------------------------------------- helpers
def constant_on(x, labels):
    """Max over orbits of (max − min) of x: 0 <=> x is constant on each orbit."""
    k = labels.max() + 1
    hi = np.full(k, -np.inf); lo = np.full(k, np.inf)
    np.maximum.at(hi, labels, x); np.minimum.at(lo, labels, x)
    return float(np.max(hi - lo))


def dart_key(g):
    """(face, vertex, other vertex of the edge) identifies a dart independently of numbering."""
    return [(int(g.face[d]), int(g.vertex[d]), int(g.vertex[g.alpha[d, 0]])) for d in range(g.n_darts)]


def all_features(g, P):
    return {"corner": FT.corner_angle(g, P), "ratio": FT.incidence_ratio(g, P),
            "dihedral": FT.dihedral(g, P), "defect": FT.angle_defect(g, P)}


def chi_of(g):
    return (orbits(g.alpha, (1, 2)).max() + 1) - (orbits(g.alpha, (0, 2)).max() + 1) + (orbits(g.alpha, (0, 1)).max() + 1)


# ---------------------------------------------------------------- checks
def check_regression(name, P, F):
    g = build_gmap(F)
    for fn, old in ((FT.corner_angle, _old_corner_angle), (FT.incidence_ratio, _old_incidence_ratio),
                    (FT.dihedral, _old_dihedral)):
        err = np.max(np.abs(fn(g, P) - old(g, P)))
        assert err < ATOL, f"[{name}] {fn.__name__} differs from the original triangle feature (gap {err:.2e})"
    print(f"  OK {name:<14} triangle regression (corner, ratio, dihedral = original)")


def check_mesh(name, P, F, closed=False, planar=True):
    g = build_gmap(F)
    D = g.n_darts
    ar = np.arange(D)
    lab = {k: orbits(g.alpha, k) for k in [(1,), (0,), (0, 2), (1, 2), (0, 1)]}

    # face normal: shape, sign rules, |n| = area for planar faces
    n = FT.face_normal(g, P)
    assert n.shape == (D, 3), f"[{name}] face_normal: shape {n.shape} instead of ({D},3)"
    assert np.allclose(n[g.alpha[:, 0]], -n, atol=ATOL), f"[{name}] face_normal: n(alpha0 d) must be -n(d)"
    assert np.allclose(n[g.alpha[:, 1]], -n, atol=ATOL), f"[{name}] face_normal: n(alpha1 d) must be -n(d)"

    f = all_features(g, P)
    for k, x in f.items():
        assert x.shape == (D,), f"[{name}] {k}: shape {x.shape} instead of ({D},)"
        assert np.all(np.isfinite(x)), f"[{name}] {k}: non-finite values"

    # constancy on the right orbit
    for k, orb in (("corner", (1,)), ("ratio", (0,)), ("dihedral", (0, 2)), ("defect", (1, 2))):
        c = constant_on(f[k], lab[orb])
        assert c < ATOL, f"[{name}] {k} not constant on <{','.join('alpha' + str(i) for i in orb)}> (gap {c:.2e})"

    assert np.all(f["corner"] > 0) and np.all(f["corner"] < TAU), f"[{name}] corner angles must lie in ]0, 2pi["
    assert np.all(f["ratio"] > 0), f"[{name}] incidence_ratio must be > 0"
    assert np.allclose(f["dihedral"][g.alpha[:, 2] == ar], 0, atol=ATOL), f"[{name}] dihedral must be 0 on a boundary"

    # planar faces: interior angles of a k-gon sum to (k−2)π
    if planar:
        k_of = np.bincount(lab[(0, 1)]) / 2                      # 2 darts per corner
        per_face = np.bincount(lab[(0, 1)], weights=f["corner"]) / 2
        err = np.max(np.abs(per_face - (k_of - 2) * np.pi))
        assert err < ATOL, f"[{name}] corner angles of a planar k-gon must sum to (k-2)pi (gap {err:.2e})"

    # Gauss–Bonnet on closed meshes with planar faces
    if closed and planar:
        first = np.unique(lab[(1, 2)], return_index=True)[1]
        total = f["defect"][first].sum()
        assert abs(total - TAU * chi_of(g)) < 1e-5, \
            f"[{name}] Gauss-Bonnet: sum of defects = {total:.6f}, expected 2*pi*chi = {TAU * chi_of(g):.6f}"

    # mirror symmetry and reversed input orientation change nothing
    fm = all_features(g, P * np.array([-1.0, 1.0, 1.0]))
    for k in f:
        assert np.allclose(f[k], fm[k], atol=ATOL), f"[{name}] {k} changes under a mirror symmetry"
    Fr = [list(face)[::-1] for face in F]
    gr = build_gmap(Fr)
    fr = all_features(gr, P)
    pos = {key: d for d, key in enumerate(dart_key(gr))}
    perm = np.array([pos[key] for key in dart_key(g)])
    for k in f:
        assert np.allclose(f[k], fr[k][perm], atol=ATOL), f"[{name}] {k} depends on the input orientation of the faces"

    tags = "normal, constancy, mirror, reversed faces" + (", (k-2)pi" if planar else "") + \
           (", Gauss-Bonnet" if closed and planar else "")
    print(f"  OK {name:<14} {tags}")
    return g, f


def run():
    print("Polygon orbit features:")

    # 1. triangle regression
    cases = [("tetrahedron", *tetra()), ("tri square", *tri_square()), ("tri torus", *tri_torus())]
    s = shrec()
    if s is not None:
        cases.append(("shrec", *s))
    for name, P, F in cases:
        check_regression(name, P, F)

    # 2-5. polygon meshes
    g, f = check_mesh("trapezoid", *trapezoid())
    ab = [d for d in range(g.n_darts) if {int(g.vertex[d]), int(g.vertex[g.alpha[d, 0]])} == {0, 1}]
    assert np.allclose(f["ratio"][ab], 1.0, atol=ATOL), \
        f"[trapezoid] ratio of AB = |AB|^2 / mean(2*3, 2*1) = 1, got {f['ratio'][ab]}"
    n = FT.face_normal(g, trapezoid()[0])
    assert np.allclose(np.linalg.norm(n, axis=1), 4.0, atol=ATOL), "[trapezoid] |face_normal| must be the area (4)"

    g, f = check_mesh("arrow", *arrow())
    c = f["corner"][g.vertex == 2]
    assert np.allclose(c, TAU - np.arccos(-0.6), atol=ATOL), \
        f"[arrow] reflex corner must be 2pi - arccos(-0.6) = {TAU - np.arccos(-0.6):.4f}, got {c}"

    check_mesh("pentagon", *pentagon())
    g, f = check_mesh("flat grid", *flat_grid())
    inner = g.alpha[:, 2] != np.arange(g.n_darts)
    assert np.allclose(f["dihedral"][inner], np.pi, atol=ATOL), "[flat grid] interior dihedral of a plane must be pi"

    g, f = check_mesh("cube", *cube(), closed=True)
    assert np.allclose(f["dihedral"], np.pi / 2, atol=ATOL), "[cube] every dihedral of a cube must be pi/2"
    check_mesh("pyramid", *pyramid(), closed=True)
    check_mesh("L-prism", *l_prism(), closed=True)
    check_mesh("quad torus", *quad_torus_geo(), closed=True)
    check_mesh("bent cube", *cube_bent(), closed=True, planar=False)
    check_mesh("mobius", *mobius_geo(), planar=False)

    print("OK - polygon features verified; triangle features unchanged.")


if __name__ == "__main__":
    run()

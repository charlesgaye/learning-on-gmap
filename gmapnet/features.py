"""Geometric features carried by ORBITS and copied onto their darts.

Darts compute, orbits carry meaning: each quantity lives on the orbit where it is defined and
is copied onto every dart of that orbit:
    inner angle             -> corner            ⟨α₁⟩
    length / height ratio   -> edge-face pair    ⟨α₀⟩
    dihedral angle          -> edge              ⟨α₀, α₂⟩
    angle defect            -> vertex            ⟨α₁, α₂⟩   (discrete Gaussian curvature)
Faces may be any polygons. Orientation-sensitive quantities rely on `face_normal`, which is
defined FROM the dart (no global orientation is needed, so non-orientable surfaces work).
On triangle meshes every feature coincides with its original triangle-only definition, and
the 5 MeshCNN edge features are recovered exactly as words in the involutions
(see `meshcnn_features`).
"""

from __future__ import annotations
import numpy as np

from gmapnet.gmap import GMap2, orbits


def walk(g: GMap2, darts, *path) -> np.ndarray:
    """Apply the involutions of `path` in APPLICATION order (left to right).

    walk(g, d, 1, 0) = α₀(α₁(d)): α₁ first, then α₀ — the word α₀α₁ applied to d.
    `darts` may be a single dart or an array of darts.
    """
    d = np.asarray(darts)
    for i in path:
        d = g.alpha[d, i]
    return d


def pos(g: GMap2, P: np.ndarray, darts) -> np.ndarray:
    """Position (…, 3) of the vertex carried by each dart."""
    return P[g.vertex[np.asarray(darts)]]

def face_normal(g: GMap2, P: np.ndarray) -> np.ndarray:
    """(D, 3) vector area of the face of each dart, the face being traversed FROM the dart.

    The traversal starts at vertex[d] and goes to vertex[α₀(d)], one step α₁∘α₀ per vertex:
    p₀ = vertex[d], p₁, …, p_{k−1}. Then n(d) = ½ Σ pᵢ × pᵢ₊₁, computed as the fan of
    triangles (p₀, pᵢ, pᵢ₊₁): all darts advance together, one recursion level per step, and a
    mask stops the darts whose face is already complete.
    Properties: independent of the origin and of the starting vertex; |n| = area for a planar
    face (a reflex corner contributes negatively instead of flipping n); for a non-planar face,
    n is the normal of the plane onto which the face has the largest projected area; n(α₀(d)) = n(α₁(d)) = −n(d) (reversed traversal); on a triangle,
    n = ½ (B − A) × (C − A).
    """
    D = g.n_darts
    d = np.arange(D)
    s_pos = pos(g, P, d)
    face_degree = np.bincount(g.face)/2
    max_degree = np.max(face_degree)

    def normal_aux(p0, pi, pj, count):

        if count == max_degree:
            return 0
        A, B, C = s_pos[p0], s_pos[pi], s_pos[pj]
        return (count < face_degree[g.face[p0]])[:,None] * np.cross(A - B, A - C, axis=1) + normal_aux(p0, walk(g, pi, 0, 1), walk(g, pj, 0, 1), count + 1)

    return normal_aux(d, walk(g, d, 0, 1), walk(g, d, 0, 1, 0, 1), 2)/2

def corner_angle(g: GMap2, P: np.ndarray) -> np.ndarray:
    """(D,) inner angle of the corner of each dart (at vertex[d], in the face of d), in ]0, 2π[.

    With u, v the edges from vertex[d] to its two neighbours in the face, the magnitude is
    θ = atan2(|u × v|, u · v) (exact 3D angle, stable near 0 and π). The side is read from
    face_normal: (u × v) · n(d) < 0 means a reflex corner, and the angle is then 2π − θ.
    Constant on ⟨α₁⟩ (α₁ flips n and swaps u, v). The corners of a planar k-gon sum to (k−2)π.
    On triangles (always convex) it coincides with the original arccos definition.
    """
    D = g.n_darts
    d = np.arange(D)
    s_pos = pos(g, P, d)
    def aux(s):
        A, B, C = s_pos[s], s_pos[g.alpha[s,0]], s_pos[walk(g, d, 1, 0)]
        theta = np.atan2(np.linalg.norm(np.cross(A - B, A - C, axis=1), axis=1), np.sum((A - B) * (A - C), axis=1))
        return np.where(np.sum(face_normal(g, P)[s] * np.cross(A - B, A - C, axis=1), axis=1) > 0, theta, 2 * np.pi - theta)

    return aux(d)

def incidence_ratio(g: GMap2, P: np.ndarray) -> np.ndarray:
    """(D,) ratio length(edge) / height for the edge AB of the dart, inside the face of the dart.

    ratio = |AB|² / mean(|AB × AC|, |BA × BC'|), where C = α₀α₁(d) and C' = α₀α₁α₀(d) are the
    other neighbours of A and of B in the face: the height is the mean distance of C and C' to
    the line AB. Symmetric in A and B, hence constant on ⟨α₀⟩; dimensionless.
    On a triangle C = C' is the third vertex and this is the original |AB|² / |AB × AC|.
    """
    D = g.n_darts
    d = np.arange(D)
    s_pos = pos(g, P, d)
    def aux(s):
        A, B, C, C2 = s_pos[s], s_pos[g.alpha[s,0]], s_pos[walk(g, d, 1, 0)], s_pos[walk(g, d, 0, 1, 0)]
        norm_AB = np.linalg.norm(A - B, axis=1)
        cross_1, cross_2 = np.linalg.norm(np.cross(B - A, C - A, axis=1), axis=1), np.linalg.norm(np.cross(B - A, C2 - B, axis=1), axis=1)
        return 2 * np.pow(norm_AB, 2) / (cross_1 + cross_2)

    return aux(d)


def dihedral(g: GMap2, P: np.ndarray) -> np.ndarray:
    """(D,) dihedral angle of the edge of the dart: angle between n(d) and n(α₂(d)).

    d and α₂(d) traverse their faces along the shared edge in the SAME direction, so on a flat
    surface their normals are opposite and the angle is π (same convention as
    src.mesh.edge_features, with which it coincides on triangles). α₀ flips both normals, so the
    value is constant on ⟨α₀, α₂⟩ and does not depend on the input orientation of the faces.
    Uses the whole face through face_normal, so it stays well defined on non-planar faces.
    On a boundary edge (α₂(d) = d) both normals coincide and the value is 0.
    """
    D = g.n_darts
    d = np.arange(D)
    s_pos = pos(g, P, d)
    def aux(s):
        s_a2 = g.alpha[s, 2]
        n = face_normal(g, P)
        n1, n2 = n[s] , n[s_a2]
        return np.arccos(np.clip(np.sum(n1 * n2, axis=1) / (np.linalg.norm(n1, axis=1) * np.linalg.norm(n2, axis=1)), -1, 1))

    return aux(d)


def angle_defect(g: GMap2, P: np.ndarray) -> np.ndarray:
    """(D,) angle defect of the vertex of the dart: 2π − Σ of the corner angles around it.

    Aggregation over the vertex orbit ⟨α₁, α₂⟩ (each corner is carried by 2 darts, hence /2).
    On a closed surface the defects sum to 2πχ (discrete Gauss–Bonnet).
    """
    angles = corner_angle(g, P) / 2
    vertex = orbits(g.alpha, (1, 2))
    orbit_angle_defect = 2 * np.pi - np.bincount(vertex, angles)
    return orbit_angle_defect[vertex]


def meshcnn_features(g: GMap2, P: np.ndarray) -> np.ndarray:
    """(E, 5) the 5 MeshCNN edge features, read from the orbit features above.

    Row k = edge with label k in `orbits(g.alpha, (0, 2))`; same format as
    src.mesh.edge_features: [dihedral, angle_min, angle_max, ratio_min, ratio_max].
    From one dart d of the edge: the opposite angles are the corners of α₀α₁(d) and α₀α₁α₂(d);
    the ratios are those of d and α₂(d). Closed surfaces only.
    """
    angles = corner_angle(g, P)
    ratios = incidence_ratio(g, P)
    dihedrals = dihedral(g, P)
    edges = orbits(g.alpha, (0, 2))
    first = np.unique(edges, return_index=True)[1]


    return np.stack([dihedrals[first],
            np.minimum(angles[walk(g, first, 1, 0)], angles[walk(g, first, 2, 1, 0)]), np.maximum(angles[walk(g, first, 1, 0)], angles[walk(g, first, 2, 1, 0)]),
            np.minimum(ratios[first], ratios[g.alpha[first, 2]]), np.maximum(ratios[first], ratios[g.alpha[first, 2]])],
            axis=1)

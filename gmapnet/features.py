"""Geometric features carried by ORBITS and copied onto their darts.

Darts compute, orbits carry meaning: each quantity lives on the orbit where it is defined and
is copied onto every dart of that orbit:
    inner angle             -> corner            ⟨α₁⟩
    length / height ratio   -> edge-face pair    ⟨α₀⟩
    dihedral angle          -> edge              ⟨α₀, α₂⟩
    angle defect            -> vertex            ⟨α₁, α₂⟩   (discrete Gaussian curvature)
The 5 MeshCNN edge features are recovered exactly as words in the involutions
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


def corner_angle(g: GMap2, P: np.ndarray) -> np.ndarray:
    """(D,) inner angle of the corner of each dart (at vertex[d], in the face of d).

    Constant on ⟨α₁⟩; the three corners of a triangle sum to π.
    """
    D = g.n_darts
    d = np.arange(D)
    s_pos = pos(g, P, d)
    def aux(s):
        A, B, C = s_pos[g.alpha[s,0]], s_pos[walk(g, d, 1, 0)], s_pos[s]
        return np.arccos(np.clip(np.sum((C - A) * (C - B), axis=1) / (np.linalg.norm(C - A, axis=1) * np.linalg.norm(C - B, axis=1)), -1, 1))

    return aux(d)

def incidence_ratio(g: GMap2, P: np.ndarray) -> np.ndarray:
    """(D,) ratio length(edge) / height for the edge of the dart, inside the face of the dart.

    height = distance from the third vertex of the face to the line carrying the edge.
    Constant on ⟨α₀⟩.
    """
    D = g.n_darts
    d = np.arange(D)
    s_pos = pos(g, P, d)
    def aux(s):
        A, B, C = s_pos[s], s_pos[g.alpha[s,0]], s_pos[walk(g, d, 1, 0)]
        norm_AB = np.linalg.norm(A - B, axis=1)
        return np.pow(norm_AB, 2) / np.linalg.norm(np.cross(B - A, C - A, axis=1), axis=1)

    return aux(d)


def dihedral(g: GMap2, P: np.ndarray) -> np.ndarray:
    """(D,) dihedral angle of the edge of the dart, same definition as src.mesh.edge_features
    (π on a flat surface). Constant on ⟨α₀, α₂⟩ and independent of any orientation.
    On a boundary edge (α₂(d) = d) both opposite vertices coincide and the value is 0.
    """
    D = g.n_darts
    d = np.arange(D)
    s_pos = pos(g, P, d)
    def aux(s):
        A, B, C, D = s_pos[s], s_pos[g.alpha[s,0]], s_pos[walk(g, d, 1, 0)], s_pos[walk(g, d, 2, 1, 0)]
        n1, n2 = np.cross(A - B, A - C, axis=1), np.cross(A - B, B - D, axis=1)
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
    vertex = orbits(g.alpha, (0, 2))
    first = np.unique(vertex, return_index=True)[1]


    return np.stack([dihedrals[first],
            np.minimum(angles[walk(g, first, 1, 0)], angles[walk(g, first, 2, 1, 0)]), np.maximum(angles[walk(g, first, 1, 0)], angles[walk(g, first, 2, 1, 0)]),
            np.minimum(ratios[first], ratios[g.alpha[first, 2]]), np.maximum(ratios[first], ratios[g.alpha[first, 2]])],
            axis=1)

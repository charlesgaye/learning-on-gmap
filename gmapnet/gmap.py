"""Triangle mesh -> 2-G-map (darts and involutions).

A dart is a (vertex, edge, face) incidence, i.e. a triangle of the barycentric subdivision.
Three involutions connect darts:
    α₀(d): same face, same edge,   other vertex;
    α₁(d): same face, same vertex, other edge;
    α₂(d): same vertex, same edge, other face — or α₂(d) = d on a boundary edge.
Cells are orbits: vertex = ⟨α₁,α₂⟩, edge = ⟨α₀,α₂⟩, face = ⟨α₀,α₁⟩.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class GMap2:
    """2-G-map of a triangle mesh.

    alpha  : (D, 3) int — alpha[d, i] = αᵢ(d). A fixed point αᵢ(d) = d encodes a boundary.
    vertex : (D,)   int — index of the mesh vertex carried by dart d.
    face   : (D,)   int — index (in `faces`) of the face of dart d.

    `vertex` and `face` are the minimal embedding needed to read geometry from the darts.
    """
    alpha: np.ndarray
    vertex: np.ndarray
    face: np.ndarray

    @property
    def n_darts(self) -> int:
        return len(self.alpha)


def build_gmap(faces: np.ndarray) -> GMap2:
    """Build the 2-G-map of a triangle mesh (F, 3) -> GMap2 with D = 6F darts, in O(F).

    Dart layout inside face f = (f0, f1, f2): darts 6f..6f+5 carry vertices
    [f0, f0, f1, f1, f2, f2]; α₁ swaps (0 1)(2 3)(4 5) and α₀ swaps (0 5)(1 2)(3 4).
    α₂ is sewn edge by edge: the partner dart is chosen by its VERTEX (not by traversal
    direction), so non-orientable surfaces need no special case.
    Raises ValueError on a non-manifold edge (more than 2 incident faces).
    """
    alpha_0 = [5,2,1,4,3,0]
    alpha_1 = [1,0,3,2,5,4]
    alpha_list = np.array([[6 * idf + alpha_0[i], 6 * idf + alpha_1[i], 6 * idf + i] for idf in range(len(faces)) for i in range(6)])
    vertex_list = np.repeat([face[i] for face in faces for i in range(3)], 2)
    face_list = np.repeat(list(range(len(faces))), 6)

    edge_to_brin = {}
    for idf in range(len(faces)):
        [a, b, c] = faces[idf]
        edges = [tuple(sorted((a,b))), tuple(sorted((b,c))), tuple(sorted((a,c)))]
        for i in range(3):
            edge = edges[i]
            if edge not in edge_to_brin:
                edge_to_brin[edge] = [6*idf + 2*i + 1]
            elif len(edge_to_brin[edge]) > 1:
                raise ValueError (f"non-manifold edge {tuple(edge)}")
            else:
                edge_to_brin[edge].append(6*idf + 2*i + 1)

    for brins in edge_to_brin.values():
        if len(brins) == 2:
            b0, b1 = brins[0], brins[1]
            idf0, idf1 = brins[0] // 6, brins[1] // 6
            if vertex_list[b0] == vertex_list[b1]:
                alpha_list[b0, 2] = b1
                alpha_list[b1, 2] = b0
                alpha_list[6 * idf0 + alpha_0[b0 % 6], 2] = 6 * idf1 + alpha_0[b1 % 6]
                alpha_list[6 * idf1 + alpha_0[b1 % 6], 2] = 6 * idf0 + alpha_0[b0 % 6]
            else:
                alpha_list[b0, 2] = 6 * idf1 + alpha_0[b1 % 6]
                alpha_list[b1, 2] = 6 * idf0 + alpha_0[b0 % 6]
                alpha_list[6 * idf0 + alpha_0[b0 % 6], 2] = b1
                alpha_list[6 * idf1 + alpha_0[b1 % 6], 2] = b0


    return GMap2(alpha_list, vertex_list, face_list)

def is_orientable(G: GMap2) -> bool:
    """True iff the surface is orientable.

    Orientable <=> the darts can be 2-coloured so that every non-fixed αᵢ links darts of
    different colours (each colour class is then the set of half-edges of an oriented map).
    Fixed points (boundary) impose no constraint.
    """
    D = G.n_darts
    label = np.full(D, -1, dtype=int)
    for s in range(D):
        if label[s] >= 0:
            continue
        label[s] = 0
        stack = [s]
        while stack:
            d = stack.pop()
            for i in range(3):
                e = int(G.alpha[d, i])
                if label[e] < 0:
                    label[e] = 1 - label[d]
                    stack.append(e)
                elif label[e] == label[d] and e != d:
                    return False

    return True


def orbits(alpha: np.ndarray, gens) -> np.ndarray:
    """Label the orbits ⟨α_i : i ∈ gens⟩. -> (D,) labels 0..k-1.

    In a 2-G-map: (1, 2) = vertices, (0, 2) = edges, (0, 1) = faces, (1,) = corners.
    """
    D = len(alpha)
    label = np.full(D, -1, dtype=int)
    k = 0
    for s in range(D):
        if label[s] >= 0:
            continue
        label[s] = k
        stack = [s]
        while stack:
            d = stack.pop()
            for i in gens:
                e = int(alpha[d, i])
                if label[e] < 0:
                    label[e] = k
                    stack.append(e)
        k += 1
    return label

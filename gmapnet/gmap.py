"""Polygon mesh -> 2-G-map (darts and involutions).

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
    """2-G-map of a polygon mesh.

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
    """Build the 2-G-map of a polygon mesh (F, S(f)) with S(f) the size of face f
        -> GMap2 with D = Sum_F 2 * S(f) darts, in O(D).

    Dart layout inside face f = (f0, ..., f_{s-1}): darts FI(f)..FI(f)+2s-1 carry vertices
    [f0, f0, f1, f1, ..., f_{s-1}, f_{s-1}]; α₁ swaps 2i and 2i+1 and α₀ swaps 2i+1 and 2i+2 mod 2s.
    α₂ is sewn edge by edge: the partner dart is chosen by its VERTEX (not by traversal
    direction), so non-orientable surfaces need no special case.
    Raises ValueError on a non-manifold edge (more than 2 incident faces), and on
    degenerated faces (less than 3 vertex, vertex used twice)
    """
    def alpha_0(i, k): return (i + 1 - 2 * (1 - i%2))%k
    def alpha_1(i): return (i + 1 - 2 * (i%2))
    face_nb_darts = [2 * len(face) for face in faces]
    first_index = np.cumsum([0] + face_nb_darts[:-1])
    alpha_list = np.array([[first_index[i] + alpha_0(j, face_nb_darts[i]), first_index[i] + alpha_1(j), first_index[i] + j]
                           for i in range(len(faces)) for j in range(face_nb_darts[i])])
    vertex_list = np.repeat([face[i] for face in faces for i in range(len(face))], 2)
    face_list = np.repeat(list(range(len(faces))), face_nb_darts)

    edge_to_brin = {}
    for idf in range(len(faces)):
        face = faces[idf]
        s = len(face)
        if s < 3:
            raise ValueError (f"degenerated face only {s} vertices")
        if len(set(face)) < s:
            raise ValueError (f"degerated face self looping")
        edges = [tuple(sorted((face[i], face[(i+1)%s]))) for i in range(s)]
        for i in range(s):
            edge = edges[i]
            if edge not in edge_to_brin:
                edge_to_brin[edge] = [first_index[idf] + 2*i + 1]
            elif len(edge_to_brin[edge]) > 1:
                raise ValueError (f"non-manifold edge {tuple(edge)}")
            else:
                edge_to_brin[edge].append(first_index[idf] + 2*i + 1)

    for brins in edge_to_brin.values():
        if len(brins) == 2:
            b0, b1 = brins[0], brins[1]
            a_00, a_01 = alpha_list[b0, 0], alpha_list[b1, 0]
            if vertex_list[b0] == vertex_list[b1]:
                alpha_list[b0, 2] = b1
                alpha_list[b1, 2] = b0
                alpha_list[a_00, 2] = a_01
                alpha_list[a_01, 2] = a_00
            else:
                alpha_list[b0, 2] = a_01
                alpha_list[b1, 2] = a_00
                alpha_list[a_00, 2] = b1
                alpha_list[a_01, 2] = b0


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

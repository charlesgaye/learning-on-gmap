"""Transformation of triangle meshes into mixed meshes of quads and triangles.

The only function is quadify, which takes a triangle mesh and outputs a mixed
mesh obtained with a greedy approach. Triangles are merged by pairs; the value
of the dihedral angle they form gives the order in which the pairs are considered.
Pairs whose dihedral angle is too small are rejected: the shared edge disappears
with its dihedral angle, one of the geometric features given to the network, so
only nearly flat edges, whose dihedral angle carries little shape information,
are removed.
"""


from __future__ import annotations
import numpy as np

from gmapnet.gmap import orbits, build_gmap
from gmapnet.features import dihedral, walk

def quadify(P, F, min_dihedral=np.deg2rad(160)) -> list[list[int]]:
    """Build the list of faces list[list[int]] of the output mesh.

    P : (V, 3) array of the 3D positions of the vertices
    F : (F, 3) array (or list) of the initial triangular faces
    min_dihedral : filtering criterion, in radians (π = flat edge)

    From the faces of the object, the edges are listed. Then two filters are
    applied: one removes the boundary edges, the other removes the edges that are
    important for the shape of the object, identified by the dihedral angle
    between their two adjacent faces (dihedral < min_dihedral).
    The remaining edges are sorted by decreasing dihedral angle (flattest first);
    ties are broken by dart number (stable sort), so the output is deterministic.
    The candidate quads go through one last filter, which removes the quads that
    would not be convex. At the two new corners a and b, the cross product of the
    corner's edges must have a positive dot product with the normal of the quad
    (the convention of features.corner_angle). The corners c and d' keep their
    triangle corner and stay convex as long as min_dihedral >= π/2.

    Then a greedy loop creates the merged faces, each represented by a list of size
    4, [a, c, b, d'], where (a, b) is the removed edge, which becomes a diagonal.
    One pass is enough: the criteria do not depend on earlier merges, and a merged
    triangle is never available again, so a rejected candidate stays rejected.
    Every face that was not merged is finally added to the list. This process
    preserves the vertices, the Euler characteristic of the mesh, and every input
    triangle is used exactly once (the output is a partition of the input).
    """

    init_gmap = build_gmap(F)

    dihedrals = dihedral(init_gmap, P)
    edges = orbits(init_gmap.alpha, (0, 2))
    first_idx = np.unique(edges, return_index=True)[1]
    interior_idx = first_idx[first_idx != init_gmap.alpha[first_idx, 2]]

    interior_dihedrals = dihedrals[interior_idx]
    interior_flat_idx = interior_idx[interior_dihedrals >= min_dihedral]

    interior_flat_dihedrals = dihedrals[interior_flat_idx]
    sigma = np.argsort(-interior_flat_dihedrals, kind="stable")

    idx = interior_flat_idx[sigma]

    def convex_face_filter(d):
        brin_on_face = [d, walk(init_gmap, d, 1, 0), walk(init_gmap, d, 0), walk(init_gmap, d, 2, 1, 0)]
        A, C, B, D = P[init_gmap.vertex[brin_on_face]]
        n1, n2 = np.cross(C - A, D - A, axis=1), np.cross(D - B, C - B, axis=1)
        return (np.sum(n1 * np.cross(A - B, C - D, axis=1), axis=1) > 0) & (np.sum(n2 * np.cross(A - B, C - D, axis=1), axis=1) > 0)

    idx = idx[convex_face_filter(idx)]

    out_faces = []
    fusioned_faces = set()
    for d in idx:
        face1, face2 = init_gmap.face[d], init_gmap.face[init_gmap.alpha[d,2]]
        if face1 not in fusioned_faces and face2 not in fusioned_faces:
            fusioned_faces.add(face1)
            fusioned_faces.add(face2)
            brin_on_face = [d, walk(init_gmap, d, 1, 0), walk(init_gmap, d, 0), walk(init_gmap, d, 2, 1, 0)]
            out_faces.append(init_gmap.vertex[brin_on_face].tolist())

    for i in range(len(F)):
        if i not in fusioned_faces:
            out_faces.append([int(v) for v in F[i]])

    return out_faces

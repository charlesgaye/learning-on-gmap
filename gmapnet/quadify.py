"""Transformation of triangle meshes into mixed meshes of quads and triangles.

The only function is quadify, which takes a triangle mesh and outputs a mixed
mesh obtained with a greedy approach. Triangles are merged by pairs; the value
of the dihedral angle they form gives the order in which the pairs are considered.
Pairs whose dihedral angle is too small are rejected, in order to preserve the
shape of the mesh.
"""


from __future__ import annotations
import numpy as np

from gmapnet.gmap import orbits, build_gmap
from gmapnet.features import dihedral, walk

def quadify(P, F, min_dihedral=np.deg2rad(160)) -> list[list[int]]:
    """Build the list of faces list[list[int]] of the output mesh.

    From the faces of the object, the edges are listed. Then two filters are
    applied: one removes the boundary edges, the other removes the edges that are
    important for the shape of the object, identified by the dihedral angle
    between their two adjacent faces.
    The remaining edges are sorted by dihedral angle, and a greedy loop creates
    the merged faces. Every face that was not merged is finally added to the list.
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

"""Edge-centred mesh representation and invariant input features (MeshCNN).

The unit of computation in MeshCNN is the edge, not the vertex or the face. On a closed
manifold triangle mesh every edge has exactly 4 neighbouring edges: the two other edges of
each of its two incident triangles.

Reference: Hanocka et al., "MeshCNN: A Network with an Edge", SIGGRAPH 2019 (§3.1, §3.2).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import trimesh


@dataclass
class Mesh:
    """Closed manifold triangle mesh.

    vertices : (V, 3) positions
    faces    : (F, 3) vertex indices
    edges    : (E, 2) vertex indices, each edge stored once
    gemm     : (E, 4) indices of the 4 neighbouring edges of each edge (order matters for
               MeshConv). "gemm" = graph of edges, as in the original implementation.
    """
    vertices: np.ndarray
    faces: np.ndarray
    edges: np.ndarray | None = None
    gemm: np.ndarray | None = None

    @classmethod
    def load(cls, path: str) -> "Mesh":
        """Load an .obj/.off file with trimesh and build the edge topology.

        `process=False` keeps the vertices exactly as stored (no merging or reordering).
        Requires a closed manifold mesh (build_edges assumes 2 faces per edge).
        """
        tm = trimesh.load(path, process=False)
        mesh = cls(vertices=np.asarray(tm.vertices, float), faces=np.asarray(tm.faces))
        mesh.build_edges()
        return mesh

    def build_edges(self) -> None:
        """Build self.edges (E, 2) and self.gemm (E, 4) from self.faces.

        gemm convention (oriented, as in the paper). Let e be an edge bordered by triangles
        f1 and f2 (consistently oriented closed manifold mesh), with opposite vertices w1 (in f1)
        and w2 (in f2), and let (u -> v) be the direction in which f1 traverses e. Then

            gemm[e] = [ edge(v,w1), edge(u,w1), edge(u,w2), edge(v,w2) ]
                        =    a    ,     b     ,     c    ,     d

        MeshConv symmetrises the pairs (a, c) and (b, d). Choosing f2 as the first face reverses
        the direction (v -> u) and yields [c, d, a, b], i.e. the 180° rotation of the edge, which
        the symmetrisation a+c, |a-c|, b+d, |b-d| absorbs. The convention depends on the face
        orientation only, never on vertex indices, so MeshConv is invariant to vertex renumbering
        (checked in tests/test_meshconv.py).
        """
        id = 0
        edge_to_idx = {}
        for face in self.faces:
            sorted_face_index = sorted(face)
            a = (sorted_face_index[0], sorted_face_index[1])
            b = (sorted_face_index[0], sorted_face_index[2])
            c = (sorted_face_index[1], sorted_face_index[2])

            for edge in [a,b,c]:
                if edge not in edge_to_idx:
                    edge_to_idx[edge] = id
                    id += 1

        self.edges = np.array([list(edge) for edge in edge_to_idx.keys()])

        gemm_array = [None] * id

        for edge in self.edges:
            key = (edge[0], edge[1])
            f1, f2 = [face for face in self.faces if set(edge).issubset(set(face))]
            # (u -> v) = direction of the edge in f1; independent of vertex indices.
            u, v = edge
            i = list(f1).index(u)
            if f1[(i + 1) % 3] != v:
                u, v = v, u
            w1 = (set(f1) - {u, v}).pop()
            w2 = (set(f2) - {u, v}).pop()
            ordered_edges = [(v, w1), (u, w1), (u, w2), (v, w2)]
            gemm_array[edge_to_idx[key]] = np.array([edge_to_idx[tuple(sorted(elt))] for elt in ordered_edges])

        self.gemm = np.array(gemm_array)

def _opposite_vertices(mesh: "Mesh") -> np.ndarray:
    """For each edge (in the order of mesh.edges), its 2 opposite vertices. -> (E, 2)

    Replays the face -> edge incidence to find, for each edge, the third vertex of each of
    its two incident triangles. Assumes a closed manifold mesh (exactly 2 faces per edge).
    """
    idx = {(min(int(a), int(b)), max(int(a), int(b))): e
           for e, (a, b) in enumerate(mesh.edges)}
    opp: list[list[int]] = [[] for _ in range(len(mesh.edges))]
    for i, j, k in mesh.faces:
        for a, b, w in ((i, j, k), (j, k, i), (k, i, j)):
            key = (min(int(a), int(b)), max(int(a), int(b)))
            opp[idx[key]].append(int(w))
    return np.array(opp)


def edge_features(mesh: Mesh) -> np.ndarray:
    """The 5 invariant features of each edge -> (E, 5).

    For an edge with endpoints A, B shared by two triangles with third vertices C and D:
      f[0]        dihedral angle between triangles ABC and ABD (angle between their normals) ;
      f[1], f[2]  the two inner angles OPPOSITE the edge (at C in ABC, at D in ABD) ;
      f[3], f[4]  the two ratios length(AB) / height, height = distance from the opposite
                  vertex to the line AB.
    All are invariant to rotation, translation and uniform scaling. Each symmetric pair is
    sorted (min, max) so that the features do not depend on which face comes first:
      feats[e] = [dihedral, angle_min, angle_max, ratio_min, ratio_max]
    """
    V = mesh.vertices
    opp = _opposite_vertices(mesh)              # (E, 2)

    feats = np.zeros((len(mesh.edges), 5))
    for e, (u, v) in enumerate(mesh.edges):
        A, B = V[u], V[v]                       # edge endpoints
        C, D = V[opp[e, 0]], V[opp[e, 1]]       # third vertex of each triangle

        n1, n2 = np.cross(A - B, A - C), np.cross(A - B, B - D)
        feats[e,0] = np.arccos(np.clip(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2)), -1, 1))

        theta1 = np.arccos(np.clip(np.dot(C - A, C - B) / (np.linalg.norm(C - A) * np.linalg.norm(C - B)), -1, 1))
        theta2 = np.arccos(np.clip(np.dot(D - A, D - B) / (np.linalg.norm(D - A) * np.linalg.norm(D - B)), -1, 1))

        if theta1 < theta2: feats[e,1], feats[e,2] = theta1, theta2
        else: feats[e,1], feats[e,2] = theta2, theta1

        norm_AB = np.linalg.norm(A - B)
        h1 = np.linalg.norm(np.cross(A - B, A - C) / norm_AB)
        h2 = np.linalg.norm(np.cross(A - B, A - D)/ norm_AB)

        if h1 > h2: feats[e,3], feats[e,4] = norm_AB/h1, norm_AB/h2
        else: feats[e,3], feats[e,4] = norm_AB/h2, norm_AB/h1
    return feats

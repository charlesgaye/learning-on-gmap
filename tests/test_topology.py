"""Edge topology (Mesh.build_edges) on a tetrahedron.

The tetrahedron is the smallest closed manifold mesh: 4 vertices, 4 faces, 6 edges, each edge
bordered by exactly 2 faces, hence 4 neighbours each.

    cd meshcnn-repro
    python -m tests.test_topology
"""
import numpy as np
from collections import defaultdict

from src.mesh import Mesh

V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
F = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])   # consistently oriented


def key(a, b):
    return (min(int(a), int(b)), max(int(a), int(b)))


def _reference(F):
    faces_of = defaultdict(list)
    for fi, (i, j, k) in enumerate(F):
        for e in (key(i, j), key(j, k), key(k, i)):
            faces_of[e].append(fi)
    nbrs = {}
    for e, fs in faces_of.items():
        s = set()
        for fi in fs:
            i, j, k = F[fi]
            for ee in (key(i, j), key(j, k), key(k, i)):
                if ee != e:
                    s.add(ee)
        nbrs[e] = s
    return faces_of, nbrs


def run():
    m = Mesh(vertices=V, faces=F)
    m.build_edges()
    E, G = m.edges, m.gemm
    faces_of, ref_nbrs = _reference(F)

    assert E is not None and G is not None, "edges / gemm not built"
    assert len(E) == 6, f"expected 6 edges, got {len(E)}"
    assert tuple(G.shape) == (6, 4), f"gemm must be (6,4), got {tuple(G.shape)}"
    assert set(key(*e) for e in E) == set(faces_of), "wrong edge set"

    for i, e in enumerate(E):
        row = [int(x) for x in G[i]]
        u, v = key(*e)
        assert len(set(row)) == 4, f"edge {i}: neighbours not distinct {row}"
        assert i not in row, f"edge {i}: contains itself"
        got = set(key(*E[j]) for j in row)
        assert got == ref_nbrs[(u, v)], f"edge {i}: neighbours {got} != {ref_nbrs[(u, v)]}"
        # oriented convention: (u -> v) = direction of e in its first face; a,d contain v; b,c contain u
        A, B, C, D = (set(int(x) for x in E[j]) for j in row)
        shared_ab = (A & B).pop()          # opposite vertex w1 (shared by a and b)
        uu, vv = (B - {shared_ab}).pop(), (A - {shared_ab}).pop()
        assert {uu, vv} == {u, v}, f"edge {i}: a,b must carry the two endpoints of the edge"
        assert vv in D and uu in C, f"edge {i}: c must contain u={uu} and d must contain v={vv}"
        assert (A & B) != (C & D), f"edge {i}: (a,b) and (c,d) must come from the two different faces"

    for i in range(6):
        for j in G[i]:
            assert i in [int(x) for x in G[int(j)]], f"asymmetry: {i}~{int(j)} not reciprocal"

    print("OK - tetrahedron topology:")
    print("  6 edges, gemm (6,4), correct neighbours, oriented convention (a,d contain v; b,c contain u), symmetric.")


if __name__ == "__main__":
    run()

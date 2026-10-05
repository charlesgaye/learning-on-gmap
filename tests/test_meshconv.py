"""MeshConv: output shape, invariance to the face order, invariance to vertex renumbering.

Swapping the two faces of an edge permutes gemm as (a,b,c,d) -> (c,d,a,b); the symmetrisation
(a+c, |a-c|, ...) must make the output identical. Renumbering the vertices of the mesh must not
change the output of any edge either (this fails with an index-based gemm convention).

    cd meshcnn-repro
    python -m tests.test_meshconv
"""
import numpy as np
import torch

from src.mesh import Mesh, edge_features
from src.layers import MeshConv

V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
F = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])   # consistently oriented


def run():
    m = Mesh(vertices=V, faces=F)
    m.build_edges()
    feats = edge_features(m)                                   # (E, 5)
    E = len(m.edges)

    x = torch.tensor(feats.T, dtype=torch.float)[None]         # (1, 5, E)
    gemm = torch.tensor(np.array(m.gemm), dtype=torch.long)[None]   # (1, E, 4)

    torch.manual_seed(0)
    conv = MeshConv(5, 8)

    out = conv(x, gemm)
    assert tuple(out.shape) == (1, 8, E), f"output {tuple(out.shape)}, expected (1, 8, {E})"

    gemm_swapped = gemm[:, :, [2, 3, 0, 1]]                     # swap the two faces
    out_swapped = conv(x, gemm_swapped)
    assert torch.allclose(out, out_swapped, atol=1e-6), (
        "MeshConv NOT invariant to the face order -> wrong symmetrisation"
    )

    # Vertex renumbering sigma: same mesh and oriented faces, different indices.
    # The output of each edge (identified by its endpoints) must be unchanged.
    for sigma in ([3, 0, 2, 1], [1, 3, 0, 2], [2, 1, 3, 0]):
        sigma = np.array(sigma)
        V2 = np.empty_like(V); V2[sigma] = V                   # vertex i becomes sigma[i]
        m2 = Mesh(vertices=V2, faces=sigma[F]); m2.build_edges()
        x2 = torch.tensor(edge_features(m2).T, dtype=torch.float)[None]
        out2 = conv(x2, torch.tensor(np.array(m2.gemm), dtype=torch.long)[None])
        idx2 = {tuple(sorted(map(int, e))): k for k, e in enumerate(m2.edges)}
        for k, (a, b) in enumerate(m.edges):
            k2 = idx2[tuple(sorted((int(sigma[a]), int(sigma[b]))))]
            assert torch.allclose(out[0, :, k], out2[0, :, k2], atol=1e-5), (
                f"MeshConv changes when vertices are renumbered (sigma={sigma.tolist()}) "
                "-> the gemm convention depends on indices"
            )

    print(f"OK - MeshConv: output (1, 8, {E}), invariant to the face order")
    print("     and to vertex renumbering (oriented gemm convention).")


if __name__ == "__main__":
    run()

"""Mesh.load: read an .obj file and build the edge topology.

An icosphere (closed manifold mesh) is exported to a temporary .obj, reloaded with Mesh.load,
and the resulting topology is checked.

    cd meshcnn-repro
    python -m tests.test_load
"""
import os
import tempfile

import numpy as np
import trimesh

from src.mesh import Mesh


def run():
    tm = trimesh.creation.icosphere(subdivisions=1)      # 80 faces, closed
    d = tempfile.mkdtemp()
    path = os.path.join(d, "sphere.obj")
    tm.export(path)

    m = Mesh.load(path)

    assert m.edges is not None and m.gemm is not None, "topology not built by load()"
    E_expected = 3 * len(tm.faces) // 2                  # closed manifold: E = 3F/2
    assert len(m.edges) == E_expected, f"E={len(m.edges)}, expected {E_expected}"
    assert tuple(np.array(m.gemm).shape) == (E_expected, 4), "malformed gemm"
    assert m.vertices.shape == (len(tm.vertices), 3), "vertices not loaded"

    print(f"OK - Mesh.load: {len(tm.faces)} faces -> {len(m.edges)} edges, gemm ({E_expected}, 4).")


if __name__ == "__main__":
    run()

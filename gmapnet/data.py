"""SHREC-16 as G-maps: per-dart features, orbit labels, validation split.

Features per dart (gmapnet.features), chosen with FEATURES:
    "dihedral" : dihedral angle of the dart's edge      ⟨α₀,α₂⟩
    "corner"   : inner angle of the dart's corner        ⟨α₁⟩
    "ratio"    : length / height of the edge-face pair   ⟨α₀⟩
    "defect"   : angle defect of the dart's vertex       ⟨α₁,α₂⟩   (not available to MeshCNN)
The default (first three) is the fair comparison with MeshCNN: the 4 darts of an edge then
carry the same information as the 5 MeshCNN features (opposite angle = π − the edge's two
corners in that face).

Mesh variant: by default the SHREC triangle meshes are used as they are. With
quad_deg = θ (degrees), each mesh is first turned into a mixed triangle/quad mesh by
gmapnet.quadify.quadify(min_dihedral = θ): same shapes, same files, same splits; only the
representation changes. The variant, θ and QUADIFY_VERSION are part of the cache key, so the
triangle and the quad caches can never be confused, and a change to quadify cannot silently
reuse a stale cache.
"""
from __future__ import annotations
import glob
import hashlib
import os

import numpy as np
import torch
import trimesh

from gmapnet.gmap import build_gmap, orbits
from gmapnet import features as FT
from gmapnet.quadify import quadify

ROOT = "data/shrec_16"
FEATURES = ("dihedral", "corner", "ratio")
VAL_PER_CLASS = 4
QUADIFY_VERSION = 2     # part of the quad cache key: bump it whenever quadify's output changes

_FN = {"dihedral": FT.dihedral, "corner": FT.corner_angle,
       "ratio": FT.incidence_ratio, "defect": FT.angle_defect}


def list_classes(root: str = ROOT) -> list[str]:
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def load_split(split: str, classes: list[str], quad_deg: float | None = None):
    """-> list of (x (C, D) float, alpha (D, 3) long, label). Cached on disk (.pt).

    quad_deg: None for the triangle meshes, or the quadify threshold in degrees.
    """
    # empty tag keeps the triangle key unchanged; the version invalidates stale quad caches
    tag = "" if quad_deg is None else f"|quad{quad_deg:g}|v{QUADIFY_VERSION}"
    key = hashlib.md5(("|".join(classes) + "|" + split + "|" + ",".join(FEATURES) + tag).encode()).hexdigest()[:8]
    cache = os.path.join(ROOT, f".cache_dart_{split}_{key}.pt")
    if os.path.exists(cache):
        return torch.load(cache, weights_only=False)
    items = []
    for label, cls in enumerate(classes):
        for path in sorted(glob.glob(os.path.join(ROOT, cls, split, "*.obj"))):
            tm = trimesh.load(path, process=False)
            P, F = np.asarray(tm.vertices, float), np.asarray(tm.faces)
            if quad_deg is not None:
                F = quadify(P, F, min_dihedral=np.deg2rad(quad_deg))
            g = build_gmap(F)
            x = np.stack([_FN[f](g, P) for f in FEATURES])          # (C, D)
            items.append((torch.tensor(x, dtype=torch.float),
                          torch.tensor(np.asarray(g.alpha), dtype=torch.long), label))
    torch.save(items, cache)
    return items


def with_labels(items, types):
    """(x, alpha, y) -> (x, alpha, [orbit labels per type], y). Labels computed once."""
    out = []
    for x, alpha, y in items:
        a = alpha.numpy()
        labs = [torch.tensor(orbits(a, t), dtype=torch.long) for t in types]
        out.append((x, alpha, labs, y))
    return out


def split_val(items, n_classes, per_class, seed):
    """Hold out `per_class` items of each class for validation (the seed also fixes the split)."""
    rng = np.random.default_rng(seed)
    by_class = {c: [i for i, it in enumerate(items) if it[-1] == c] for c in range(n_classes)}
    val = {i for c in by_class for i in rng.choice(by_class[c], per_class, replace=False)}
    return [it for i, it in enumerate(items) if i not in val], [items[i] for i in sorted(val)]

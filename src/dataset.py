"""SHREC-16 loader (mesh classification).

Expected layout:  root/<class>/<split>/*.obj   (30 classes, split = train | test).

Each item is (x, gemm, label):
    x     : (5, E)  RAW edge features (standardisation is done later with training statistics).
    gemm  : (E, 4)  edge neighbourhoods (indices).
    label : int     class index.

Computing the features takes ~0.3 s per mesh (~3-4 min for SHREC), so results are cached
on disk (.pt) and later runs load instantly.
"""
from __future__ import annotations
import os
import glob
import hashlib

import numpy as np
import torch
from torch.utils.data import Dataset

from .mesh import Mesh, edge_features


class ShrecDataset(Dataset):
    def __init__(self, root: str, split: str = "train",
                 classes: list[str] | None = None, use_cache: bool = True):
        # classes=None -> every class sub-directory, sorted for a stable label mapping
        self.classes = classes if classes is not None else sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))

        # "gemm-v2": oriented gemm convention; invalidates caches built with the earlier one
        key = hashlib.md5(("|".join(self.classes) + "|" + split + "|gemm-v2").encode()).hexdigest()[:8]
        cache_path = os.path.join(root, f".cache_{split}_{key}.pt")
        if use_cache and os.path.exists(cache_path):
            self.items = torch.load(cache_path, weights_only=False)
            return

        self.items = []
        for label, cls in enumerate(self.classes):
            for path in sorted(glob.glob(os.path.join(root, cls, split, "*.obj"))):
                m = Mesh.load(path)
                feats = edge_features(m)                                   # (E, 5)
                x = torch.tensor(feats.T, dtype=torch.float)               # (5, E)
                gemm = torch.tensor(np.asarray(m.gemm), dtype=torch.long)  # (E, 4)
                self.items.append((x, gemm, label))

        if use_cache:
            torch.save(self.items, cache_path)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]

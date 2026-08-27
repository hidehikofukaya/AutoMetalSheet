"""OCC-free core of the wtok package: quantization constants, the sparse-W
container, and the two small utilities the training path needs.

Why this module exists: `convert.py` imports pythonocc at module level (it does
B-Rep work), but training / sampling / evaluation need only these constants.
Keeping them here lets the whole learning path run on machines without
pythonocc (Kaggle). `convert.py` re-exports from here, so existing local
pipelines are unaffected.
"""
from __future__ import annotations

import os
import pathlib
import time
import zlib
from dataclasses import dataclass, field

import numpy as np

BITS = 14                  # quantization bits per axis (theory §3, b=14)
HI_BITS = 7                # coarse digit width; fine digit is BITS - HI_BITS
VTYPES = ("FIX", "END", "MID")           # CTRL unused until SPLINE is enabled
ETYPES = ("LINE", "ARC", "CIRCLE", "CIRCLE_C")
E_ARITY = {"LINE": 2, "ARC": 3, "CIRCLE": 3, "CIRCLE_C": 2}


@dataclass
class SparseW:
    vertices: list = field(default_factory=list)   # dicts: {xyz, T, nf?}
    edges: list = field(default_factory=list)      # dicts: {tau, refs, cls}

    def add_vertex(self, xyz, vtype, nf=None) -> int:
        self.vertices.append({"xyz": np.asarray(xyz, dtype=np.float64),
                              "T": vtype, "nf": nf})
        return len(self.vertices) - 1


def stable_seed(base: int, key: str) -> int:
    """Deterministic per-part seed (must match wireflow.dataset.stable_seed)."""
    return int((int(base) + zlib.crc32(key.encode("utf-8"))) % (2**32 - 1))


def safe_save(obj, path: pathlib.Path, retries: int = 5) -> None:
    """Atomic save with retry: Windows AV/indexer can transiently lock the
    destination, which killed a 100-epoch run once. Write to tmp + replace."""
    import torch

    tmp = pathlib.Path(path).with_suffix(".tmp")
    for attempt in range(retries):
        try:
            torch.save(obj, tmp)
            os.replace(tmp, path)
            return
        except (RuntimeError, OSError) as exc:
            if attempt == retries - 1:
                print(f"[warn] giving up saving {pathlib.Path(path).name}: {exc}",
                      flush=True)
                return
            time.sleep(2.0 * (attempt + 1))

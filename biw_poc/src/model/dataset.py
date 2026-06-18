"""
dataset.py -- PyTorch Dataset for the Stage A *_dataset.h5 files
(see CLAUDE.md section 13 / src/preprocess/midsurface_sampler.py).

Each h5 file holds one part:
    points         [N,3]   float32   noisy training input point cloud
    normals        [N,3]   float32   per-point normals (input features)
    query_xyz      [Nq,3]  float32   UDF query points (near/far/boundary-shadow)
    query_udf      [Nq]    float32   GT unsigned distance at each query
    query_grad     [Nq,3]  float32   GT unit gradient direction at each query (R7-01)
    query_category [Nq]    int64     0=near 1=far 2=boundary-shadow (R7-09).
                                      Required -- regenerate via
                                      midsurface_sampler.py if a dataset
                                      predates R7-09 and lacks this field.

One dataset item = one full part (the encoder consumes the whole point
cloud at once; there is no notion of "batching within a part"). When
n_query_sample is set, a random subset of the Nq queries is drawn per
__getitem__ call so a fixed-size batch can be stacked across parts whose
raw Nq may differ, and so a fresh decoder query is seen every epoch.

Coordinate normalization (R7-02, CLAUDE.md section 13)
--------------------------------------------------------
Raw STEP coordinates are absolute world-space mm values, hundreds of mm
from the origin (a BIW assembly's local coordinate system, not part-local).
Feeding that directly into the encoder/decoder MLPs starves training --
small-weight-init layers cannot represent large constant offsets, and an
initial sanity run confirmed the loss simply does not move. Every item is
therefore re-centered on its own point-cloud centroid and isotropically
rescaled so coordinates fall roughly in [-1, 1]. `center`/`scale` are
returned alongside the normalized tensors so a trained model can be run on
a part at any absolute position (Stage C denormalizes back to world mm).
query_udf is left in mm (the training loop rescales the model's normalized
UDF prediction by `scale` before computing the loss against it); query_grad
is a unit direction, which isotropic scaling does not change.
"""

import pathlib

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class MidSurfaceUDFDataset(Dataset):
    def __init__(self, h5_paths, n_query_sample: int | None = 4096,
                 n_points_sample: int | None = None, seed: int = 0):
        """
        h5_paths: list of *_dataset.h5 paths, OR a directory to glob recursively.
        n_query_sample:  if set, randomly subsample this many query points per item.
        n_points_sample: if set, randomly subsample this many input points per item
                          (None = use all points as stored, e.g. 8192).
        """
        if isinstance(h5_paths, (str, pathlib.Path)):
            root = pathlib.Path(h5_paths)
            h5_paths = sorted(root.rglob("*_dataset.h5")) if root.is_dir() else [root]
        self.h5_paths = [pathlib.Path(p) for p in h5_paths]
        if not self.h5_paths:
            raise ValueError("No *_dataset.h5 files found")
        self.n_query_sample = n_query_sample
        self.n_points_sample = n_points_sample
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.h5_paths)

    def __getitem__(self, idx):
        path = self.h5_paths[idx]
        with h5py.File(path, "r") as f:
            points = f["points"][:]
            normals = f["normals"][:]
            query_xyz = f["query_xyz"][:]
            query_udf = f["query_udf"][:]
            query_grad = f["query_grad"][:]
            if "query_category" not in f:
                raise KeyError(
                    f"{path} has no 'query_category' dataset (R7-09). "
                    "Regenerate it via midsurface_sampler.py."
                )
            query_category = f["query_category"][:]
            thickness_mm = float(f.attrs.get("thickness_mm", 0.0))

        if self.n_points_sample is not None and self.n_points_sample < len(points):
            sel = self._rng.choice(len(points), size=self.n_points_sample, replace=False)
            points, normals = points[sel], normals[sel]

        if self.n_query_sample is not None and self.n_query_sample < len(query_xyz):
            sel = self._rng.choice(len(query_xyz), size=self.n_query_sample, replace=False)
            query_xyz = query_xyz[sel]
            query_udf = query_udf[sel]
            query_grad = query_grad[sel]
            query_category = query_category[sel]

        # R7-02: center + isotropic rescale to ~[-1,1] for stable optimization.
        center = points.mean(axis=0).astype(np.float32)
        extent = float(np.abs(points - center).max())
        scale = extent * 1.05 + 1e-6
        points = (points - center) / scale
        query_xyz = (query_xyz - center) / scale

        return {
            "points": torch.from_numpy(points),
            "normals": torch.from_numpy(normals),
            "query_xyz": torch.from_numpy(query_xyz),
            "query_udf": torch.from_numpy(query_udf),          # mm, NOT normalized
            "query_grad": torch.from_numpy(query_grad),        # unit direction, scale-invariant
            "query_category": torch.from_numpy(query_category.astype(np.int64)),  # R7-09
            "center": torch.from_numpy(center),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "thickness_mm": torch.tensor(thickness_mm, dtype=torch.float32),
            "source": str(path),
        }


def collate_fixed_size(batch):
    """
    Default-collate works as long as every item in the batch has the same
    points/N and query/Nq counts -- guaranteed when n_points_sample and
    n_query_sample are both set to fixed values (the normal training case).
    """
    out = {}
    for key in ("points", "normals", "query_xyz", "query_udf", "query_grad",
                "query_category", "center", "scale", "thickness_mm"):
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    out["source"] = [b["source"] for b in batch]
    return out

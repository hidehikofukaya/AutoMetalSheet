"""Dataset contract for the independent softmin-guidance model.

The Stage A v1 guidance schema is required and validated eagerly.  Legacy
``query_udf``/``query_grad`` aliases are intentionally insufficient: this
reader requires the independent potential, step, direction, ambiguity, and
direction-strength fields written by the Round 18 Stage A pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import pathlib
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


STAGE_A_SCHEMA_VERSION = "stage_a.softmin_guidance.v1"
BASE_REQUIRED_DATASETS = ("points", "normals", "query_xyz", "query_category")
GUIDANCE_REQUIRED_DATASETS = (
    "query_soft_potential",
    "query_step_distance",
    "query_soft_direction",
    "query_branch_ambiguity",
    "query_direction_strength",
)


@dataclass(frozen=True)
class _ResolvedFields:
    signed_potential: str
    step_distance: str
    toward_surface: str
    ambiguity: str
    direction_strength: str


def _resolve_guidance_fields(
    h5_file: h5py.File, source: pathlib.Path
) -> _ResolvedFields:
    required = BASE_REQUIRED_DATASETS + GUIDANCE_REQUIRED_DATASETS
    missing = [name for name in required if name not in h5_file]
    if missing:
        raise KeyError(
            f"{source} is missing required Stage A guidance datasets: {missing}"
        )
    schema_version = h5_file.attrs.get("schema_version")
    if schema_version != STAGE_A_SCHEMA_VERSION:
        raise ValueError(
            f"{source}: schema_version must be {STAGE_A_SCHEMA_VERSION!r}, "
            f"got {schema_version!r}"
        )
    if h5_file.attrs.get("gradient_convention") != "toward_surface":
        raise ValueError(
            f"{source}: gradient_convention must be 'toward_surface'"
        )
    if h5_file.attrs.get("coordinate_unit") != "mm":
        raise ValueError(f"{source}: coordinate_unit must be 'mm'")

    return _ResolvedFields(
        signed_potential="query_soft_potential",
        step_distance="query_step_distance",
        toward_surface="query_soft_direction",
        ambiguity="query_branch_ambiguity",
        direction_strength="query_direction_strength",
    )


def _validate_shapes(
    h5_file: h5py.File, source: pathlib.Path, fields: _ResolvedFields
) -> None:
    points_shape = h5_file["points"].shape
    normals_shape = h5_file["normals"].shape
    query_shape = h5_file["query_xyz"].shape
    if len(points_shape) != 2 or points_shape[1:] != (3,) or points_shape[0] == 0:
        raise ValueError(f"{source}: points must have non-empty shape [N,3]")
    if normals_shape != points_shape:
        raise ValueError(f"{source}: normals shape must match points shape")
    if len(query_shape) != 2 or query_shape[1:] != (3,) or query_shape[0] == 0:
        raise ValueError(f"{source}: query_xyz must have non-empty shape [Nq,3]")

    n_query = query_shape[0]
    scalar_fields = (
        fields.signed_potential,
        fields.step_distance,
        fields.ambiguity,
        fields.direction_strength,
    )
    for name in scalar_fields:
        if h5_file[name].shape != (n_query,):
            raise ValueError(f"{source}: {name} must have shape [{n_query}]")
    if h5_file[fields.toward_surface].shape != (n_query, 3):
        raise ValueError(
            f"{source}: {fields.toward_surface} must have shape [{n_query},3]"
        )
    if h5_file["query_category"].shape != (n_query,):
        raise ValueError(f"{source}: query_category must have shape [{n_query}]")
    for name in BASE_REQUIRED_DATASETS + GUIDANCE_REQUIRED_DATASETS:
        if not np.isfinite(h5_file[name][:]).all():
            raise ValueError(f"{source}: {name} contains non-finite values")
    if np.any(h5_file[fields.step_distance][:] < 0):
        raise ValueError(f"{source}: query_step_distance must be non-negative")
    ambiguity = h5_file[fields.ambiguity][:]
    if np.any((ambiguity < 0) | (ambiguity > 1)):
        raise ValueError(f"{source}: query_branch_ambiguity must lie in [0,1]")
    strength = h5_file[fields.direction_strength][:]
    if np.any((strength < 0) | (strength > 1)):
        raise ValueError(f"{source}: query_direction_strength must lie in [0,1]")
    category = h5_file["query_category"][:]
    if np.any((category < 0) | (category > 2)):
        raise ValueError(f"{source}: query_category must use 0=near, 1=far, 2=boundary")


class SoftminGuidanceDataset(Dataset):
    """One HDF5 part per item with normalized coordinates and world-mm targets."""

    def __init__(
        self,
        h5_paths,
        n_query_sample: int | None = 4096,
        n_points_sample: int | None = None,
        seed: int = 0,
        query_indices: Sequence[Sequence[int] | np.ndarray | None] | None = None,
        fixed_sampling: bool = False,
    ):
        if isinstance(h5_paths, (str, pathlib.Path)):
            root = pathlib.Path(h5_paths)
            h5_paths = (
                sorted(root.rglob("*_dataset.h5")) if root.is_dir() else [root]
            )
        self.h5_paths = [pathlib.Path(path) for path in h5_paths]
        if not self.h5_paths:
            raise ValueError("No *_dataset.h5 files found")
        if n_query_sample is not None and n_query_sample <= 0:
            raise ValueError("n_query_sample must be positive or None")
        if n_points_sample is not None and n_points_sample <= 0:
            raise ValueError("n_points_sample must be positive or None")

        self.n_query_sample = n_query_sample
        self.n_points_sample = n_points_sample
        self._rng = np.random.default_rng(seed)
        self.fixed_sampling = fixed_sampling
        self._resolved_fields: list[_ResolvedFields] = []
        self.query_counts: list[int] = []
        self.query_indices: list[np.ndarray] = []
        self._fixed_query_samples: list[np.ndarray | None] = []
        self._fixed_point_samples: list[np.ndarray | None] = []
        if query_indices is not None and len(query_indices) != len(self.h5_paths):
            raise ValueError("query_indices must have one entry per HDF5 part")

        # Eager metadata validation is deliberate: incompatible Stage A files
        # fail at construction, before a training worker starts an epoch.
        fixed_rng = np.random.default_rng(seed)
        for index, path in enumerate(self.h5_paths):
            with h5py.File(path, "r") as h5_file:
                fields = _resolve_guidance_fields(h5_file, path)
                _validate_shapes(h5_file, path, fields)
                self._resolved_fields.append(fields)
                n_query = int(h5_file["query_xyz"].shape[0])
                n_points = int(h5_file["points"].shape[0])
                self.query_counts.append(n_query)

            requested = None if query_indices is None else query_indices[index]
            if requested is None:
                allowed = np.arange(n_query, dtype=np.int64)
            else:
                allowed = np.asarray(requested, dtype=np.int64)
                if allowed.ndim != 1 or len(allowed) == 0:
                    raise ValueError(
                        f"{path}: query_indices must be a non-empty 1D sequence"
                    )
                if np.any((allowed < 0) | (allowed >= n_query)):
                    raise IndexError(f"{path}: query_indices are out of range")
                if len(np.unique(allowed)) != len(allowed):
                    raise ValueError(f"{path}: query_indices must be unique")
                allowed = allowed.copy()
            self.query_indices.append(allowed)

            fixed_query_sample = None
            if (
                fixed_sampling
                and n_query_sample is not None
                and n_query_sample < len(allowed)
            ):
                positions = fixed_rng.choice(
                    len(allowed), size=n_query_sample, replace=False
                )
                fixed_query_sample = allowed[positions]
            self._fixed_query_samples.append(fixed_query_sample)

            fixed_point_sample = None
            if (
                fixed_sampling
                and n_points_sample is not None
                and n_points_sample < n_points
            ):
                fixed_point_sample = fixed_rng.choice(
                    n_points, size=n_points_sample, replace=False
                )
            self._fixed_point_samples.append(fixed_point_sample)

    def __len__(self) -> int:
        return len(self.h5_paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.h5_paths[index]
        fields = self._resolved_fields[index]
        with h5py.File(path, "r") as h5_file:
            points = np.asarray(h5_file["points"][:], dtype=np.float32)
            normals = np.asarray(h5_file["normals"][:], dtype=np.float32)
            query_xyz = np.asarray(h5_file["query_xyz"][:], dtype=np.float32)
            signed_potential = np.asarray(
                h5_file[fields.signed_potential][:], dtype=np.float32
            )
            step_distance = np.asarray(
                h5_file[fields.step_distance][:], dtype=np.float32
            )
            toward_surface = np.asarray(
                h5_file[fields.toward_surface][:], dtype=np.float32
            )
            ambiguity = np.asarray(
                h5_file[fields.ambiguity][:], dtype=np.float32
            )
            direction_strength = np.asarray(
                h5_file[fields.direction_strength][:], dtype=np.float32
            )
            query_category = (
                np.asarray(h5_file["query_category"][:], dtype=np.int64)
                if "query_category" in h5_file
                else None
            )
            thickness_mm = float(h5_file.attrs.get("thickness_mm", 0.0))

        fixed_point_sample = self._fixed_point_samples[index]
        if fixed_point_sample is not None:
            point_index = fixed_point_sample
            points = points[point_index]
            normals = normals[point_index]
        elif self.n_points_sample is not None and self.n_points_sample < len(points):
            point_index = self._rng.choice(
                len(points), size=self.n_points_sample, replace=False
            )
            points = points[point_index]
            normals = normals[point_index]

        query_index = self.query_indices[index]
        fixed_query_sample = self._fixed_query_samples[index]
        if fixed_query_sample is not None:
            query_index = fixed_query_sample
        elif self.n_query_sample is not None and self.n_query_sample < len(query_index):
            positions = self._rng.choice(
                len(query_index), size=self.n_query_sample, replace=False
            )
            query_index = query_index[positions]

        # Every query-aligned field is indexed exactly once with this same
        # array. Never independently sample individual supervision heads.
        query_xyz = query_xyz[query_index]
        signed_potential = signed_potential[query_index]
        step_distance = step_distance[query_index]
        toward_surface = toward_surface[query_index]
        ambiguity = ambiguity[query_index]
        direction_strength = direction_strength[query_index]
        if query_category is not None:
            query_category = query_category[query_index]

        center = points.mean(axis=0, dtype=np.float64).astype(np.float32)
        extent = float(np.abs(points - center).max())
        scale = extent * 1.05 + 1.0e-6
        points_normalized = (points - center) / scale
        query_normalized = (query_xyz - center) / scale

        item: dict[str, object] = {
            "points": torch.from_numpy(points_normalized),
            "normals": torch.from_numpy(normals),
            "query_xyz": torch.from_numpy(query_normalized),
            # Both scalar targets stay in world millimetres.
            "query_soft_potential": torch.from_numpy(signed_potential),
            "query_step_distance": torch.from_numpy(step_distance),
            "query_soft_direction": torch.from_numpy(toward_surface),
            "query_branch_ambiguity": torch.from_numpy(ambiguity),
            "query_direction_strength": torch.from_numpy(direction_strength),
            "center": torch.from_numpy(center),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "thickness_mm": torch.tensor(thickness_mm, dtype=torch.float32),
            "query_index": torch.from_numpy(query_index.copy()),
            "source": str(path),
        }
        if query_category is not None:
            item["query_category"] = torch.from_numpy(query_category)
        return item


def collate_softmin_guidance(batch: list[dict[str, object]]) -> dict[str, object]:
    """Stack fixed-size items while preserving source paths as strings."""

    tensor_keys = (
        "points",
        "normals",
        "query_xyz",
        "query_soft_potential",
        "query_step_distance",
        "query_soft_direction",
        "query_branch_ambiguity",
        "query_direction_strength",
        "center",
        "scale",
        "thickness_mm",
    )
    output: dict[str, object] = {
        key: torch.stack(
            [item[key] for item in batch], dim=0  # type: ignore[list-item]
        )
        for key in tensor_keys
    }
    if all("query_category" in item for item in batch):
        output["query_category"] = torch.stack(
            [item["query_category"] for item in batch], dim=0  # type: ignore[list-item]
        )
    output["query_index"] = torch.stack(
        [item["query_index"] for item in batch], dim=0  # type: ignore[list-item]
    )
    output["source"] = [item["source"] for item in batch]
    return output

"""Datasets built from fill_volume filled midsurface STEP files."""

from __future__ import annotations

import json
import os
import zlib
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import Dataset

from .step_tessellate import (
    TessellatedMesh,
    TessellationConfig,
    load_or_tessellate,
    sample_boundary_points,
    sample_corner_points,
    sample_crease_points,
    sample_surface_points,
)


MIRROR_AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class PartRecord:
    assembly_id: str
    part_id_raw: str
    canonical_part_id: str
    stp_path: Path
    joints_path: Path | None


def discover_fill_parts(
    fill_volume_root: str | Path,
    assemblies: list[str] | None = None,
    max_file_mb: float | None = None,
) -> list[PartRecord]:
    root = Path(fill_volume_root)
    base = root / "fill_mid_surf"
    if not base.exists():
        raise FileNotFoundError(f"fill_mid_surf not found under {root}")

    records: list[PartRecord] = []
    for assembly_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if assemblies and assembly_dir.name not in assemblies:
            continue
        fill_dir = assembly_dir / "fill"
        if not fill_dir.exists():
            continue
        joints_path = assembly_dir / "annotations" / "joints.json"
        for stp_path in sorted(fill_dir.glob("*.stp")):
            if max_file_mb is not None and stp_path.stat().st_size > max_file_mb * 1024 * 1024:
                continue
            part_id = stp_path.stem.split("_", 1)[0]
            records.append(
                PartRecord(
                    assembly_id=assembly_dir.name,
                    part_id_raw=part_id,
                    canonical_part_id=f"{assembly_dir.name}:{part_id}",
                    stp_path=stp_path,
                    joints_path=joints_path if joints_path.exists() else None,
                )
            )
    return records


def _joint_points_for_part(record: PartRecord) -> np.ndarray:
    if record.joints_path is None:
        return np.zeros((0, 3), dtype=np.float32)
    try:
        data = json.loads(record.joints_path.read_text(encoding="utf-8"))
    except Exception:
        return np.zeros((0, 3), dtype=np.float32)
    pts: list[list[float]] = []
    for joint in data.get("joints", []):
        for per_part in joint.get("per_part", []):
            if str(per_part.get("part_id")) != str(record.part_id_raw):
                continue
            if "hole_center_xyz" in per_part:
                pts.append(per_part["hole_center_xyz"])
            elif "contact_xyz" in per_part:
                pts.append(per_part["contact_xyz"])
    return np.asarray(pts, dtype=np.float32) if pts else np.zeros((0, 3), dtype=np.float32)


class MidsurfacePointCloudDataset(Dataset):
    """Fixed-size area-sampled point clouds from filled midsurface STEP files."""

    def __init__(
        self,
        records: list[PartRecord],
        cache_dir: str | Path,
        n_points: int = 1024,
        seed: int = 0,
        tessellation: TessellationConfig | None = None,
        use_joint_distance: bool = False,
        boundary_sample_fraction: float = 0.0,
        use_boundary_feature: bool = False,
        scaffold_target_points: int = 0,
        scaffold_target_boundary_fraction: float = 0.35,
        scaffold_target_crease_fraction: float = 0.35,
        scaffold_target_corner_fraction: float = 0.10,
        mirror_axes: tuple[str, ...] | list[str] | None = None,
        resample_each_epoch: bool = False,
    ) -> None:
        if not records:
            raise ValueError("records must not be empty")
        if not (0.0 <= boundary_sample_fraction < 1.0):
            raise ValueError("boundary_sample_fraction must satisfy 0.0 <= fraction < 1.0")
        if scaffold_target_points < 0:
            raise ValueError("scaffold_target_points must be non-negative")
        fractions = [
            scaffold_target_boundary_fraction,
            scaffold_target_crease_fraction,
            scaffold_target_corner_fraction,
        ]
        if any(f < 0.0 for f in fractions) or sum(fractions) > 1.0 + 1.0e-9:
            raise ValueError("scaffold target fractions must be non-negative and sum to at most 1.0")
        self.mirror_axes = normalize_mirror_axes(mirror_axes)
        self.records = records
        self.cache_dir = Path(cache_dir)
        self.n_points = n_points
        self.seed = seed
        self.tessellation = tessellation or TessellationConfig()
        self.use_joint_distance = use_joint_distance
        self.boundary_sample_fraction = boundary_sample_fraction
        self.use_boundary_feature = use_boundary_feature
        self.scaffold_target_points = scaffold_target_points
        self.scaffold_target_boundary_fraction = scaffold_target_boundary_fraction
        self.scaffold_target_crease_fraction = scaffold_target_crease_fraction
        self.scaffold_target_corner_fraction = scaffold_target_corner_fraction
        self.augmentation_variants = ("original", *[f"mirror_{axis}" for axis in self.mirror_axes])
        self.resample_each_epoch = bool(resample_each_epoch)
        self.resample_step = 0

    def __len__(self) -> int:
        return len(self.records) * len(self.augmentation_variants)

    @property
    def augmentation_factor(self) -> int:
        return len(self.augmentation_variants)

    def set_resample_step(self, step: int) -> None:
        self.resample_step = int(step)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record_index = index // self.augmentation_factor
        variant_index = index % self.augmentation_factor
        variant = self.augmentation_variants[variant_index]
        mirror_axis = variant.removeprefix("mirror_") if variant.startswith("mirror_") else None
        record = self.records[record_index]
        mesh = load_or_tessellate(record.stp_path, self.cache_dir, self.tessellation)
        sample_seed = sample_seed_for_variant(
            self.seed,
            record.canonical_part_id,
            variant,
            self.resample_step if self.resample_each_epoch else None,
        )
        center = (mesh.bounds_min + mesh.bounds_max) * 0.5
        scale = float(np.max(mesh.bounds_max - mesh.bounds_min))
        if scale <= 0:
            scale = 1.0
        n_boundary = int(round(self.n_points * self.boundary_sample_fraction))
        n_boundary = min(n_boundary, max(self.n_points - 1, 0))
        if mesh.n_boundary_edges == 0:
            n_boundary = 0
        n_surface = self.n_points - n_boundary
        pts_world, normals = sample_surface_points(mesh, n_surface, seed=sample_seed)
        boundary_world, boundary_normals = sample_boundary_points(mesh, n_boundary, seed=sample_seed + 104729)
        if len(boundary_world):
            pts_world = np.concatenate([pts_world, boundary_world], axis=0)
            normals = np.concatenate([normals, boundary_normals], axis=0)
            boundary_mask = np.concatenate(
                [
                    np.zeros((n_surface,), dtype=bool),
                    np.ones((len(boundary_world),), dtype=bool),
                ]
            )
            order_rng = np.random.default_rng(sample_seed + 1009)
            order = order_rng.permutation(len(pts_world))
            pts_world = pts_world[order]
            normals = normals[order]
            boundary_mask = boundary_mask[order]
        else:
            boundary_mask = np.zeros((len(pts_world),), dtype=bool)

        pts = (pts_world - center[None, :]) / scale
        if mirror_axis is not None:
            pts = mirror_points(pts, mirror_axis)
            normals = mirror_points(normals, mirror_axis)
        typed_scaffold_targets = sample_typed_scaffold_target_points(
            mesh=mesh,
            n_points=self.scaffold_target_points,
            boundary_fraction=self.scaffold_target_boundary_fraction,
            crease_fraction=self.scaffold_target_crease_fraction,
            corner_fraction=self.scaffold_target_corner_fraction,
            seed=sample_seed + 211,
        )
        scaffold_targets_world = typed_scaffold_targets["all_points"]
        scaffold_targets = (scaffold_targets_world - center[None, :]) / scale if len(scaffold_targets_world) else np.zeros(
            (0, 3),
            dtype=np.float32,
        )
        typed_scaffold_targets_norm = normalize_typed_scaffold_targets(typed_scaffold_targets, center, scale)
        if mirror_axis is not None:
            scaffold_targets = mirror_points(scaffold_targets, mirror_axis)
            typed_scaffold_targets_norm = mirror_typed_scaffold_targets(typed_scaffold_targets_norm, mirror_axis)

        if self.use_joint_distance:
            joints_world = _joint_points_for_part(record)
            if len(joints_world):
                joints = (joints_world - center[None, :]) / scale
            else:
                joints = np.zeros((0, 3), dtype=np.float32)
            joint_d = _nearest_distance(pts, joints)
        else:
            joint_d = np.ones((len(pts),), dtype=np.float32)

        feature_parts = [
            pts.astype(np.float32),
            normals.astype(np.float32),
            joint_d[:, None].astype(np.float32),
        ]
        if self.use_boundary_feature:
            feature_parts.append(boundary_mask.astype(np.float32)[:, None])
        features = np.concatenate(feature_parts, axis=1)
        return {
            "points": torch.from_numpy(pts.astype(np.float32)),
            "normals": torch.from_numpy(normals.astype(np.float32)),
            "features": torch.from_numpy(features),
            "boundary_mask": torch.from_numpy(boundary_mask),
            "scaffold_targets": torch.from_numpy(scaffold_targets.astype(np.float32)),
            "scaffold_target_mask": torch.from_numpy(typed_scaffold_targets["all_mask"]),
            "scaffold_boundary_targets": torch.from_numpy(typed_scaffold_targets_norm["boundary_points"]),
            "scaffold_boundary_target_mask": torch.from_numpy(typed_scaffold_targets["boundary_mask"]),
            "scaffold_crease_targets": torch.from_numpy(typed_scaffold_targets_norm["crease_points"]),
            "scaffold_crease_target_mask": torch.from_numpy(typed_scaffold_targets["crease_mask"]),
            "scaffold_corner_targets": torch.from_numpy(typed_scaffold_targets_norm["corner_points"]),
            "scaffold_corner_target_mask": torch.from_numpy(typed_scaffold_targets["corner_mask"]),
            "center": torch.from_numpy(center.astype(np.float32)),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "record_index": torch.tensor(record_index, dtype=torch.long),
            "sample_seed": torch.tensor(sample_seed, dtype=torch.long),
            "name": record.canonical_part_id if variant == "original" else f"{record.canonical_part_id}|{variant}",
            "augmentation": variant,
        }


def normalize_mirror_axes(axes: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if not axes:
        return ()
    result: list[str] = []
    for axis in axes:
        normalized = str(axis).lower()
        if normalized not in MIRROR_AXIS_TO_INDEX:
            raise ValueError("mirror axes must be one or more of: x, y, z")
        if normalized not in result:
            result.append(normalized)
    return tuple(result)


def mirror_points(points: np.ndarray, axis: str) -> np.ndarray:
    mirrored = np.asarray(points, dtype=np.float32).copy()
    if mirrored.size == 0:
        return mirrored
    mirrored[..., MIRROR_AXIS_TO_INDEX[axis]] *= -1.0
    return mirrored


def mirror_typed_scaffold_targets(targets: dict[str, np.ndarray], axis: str) -> dict[str, np.ndarray]:
    return {key: mirror_points(value, axis) for key, value in targets.items()}


def normalize_typed_scaffold_targets(
    targets: dict[str, np.ndarray],
    center: np.ndarray,
    scale: float,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key in ("boundary_points", "crease_points", "corner_points"):
        pts = targets[key]
        result[key] = ((pts - center[None, :]) / scale).astype(np.float32) if len(pts) else pts.astype(np.float32)
    return result


def scaffold_target_counts(
    n_points: int,
    boundary_fraction: float,
    crease_fraction: float,
    corner_fraction: float,
) -> tuple[int, int, int, int]:
    n_corner = int(round(n_points * corner_fraction))
    n_crease = int(round(n_points * crease_fraction))
    n_boundary = int(round(n_points * boundary_fraction))
    requested = n_corner + n_crease + n_boundary
    if requested > n_points:
        scale = n_points / max(requested, 1)
        n_corner = int(round(n_corner * scale))
        n_crease = int(round(n_crease * scale))
        n_boundary = min(n_points - n_corner - n_crease, n_boundary)
    n_area = max(0, n_points - n_corner - n_crease - n_boundary)
    return n_boundary, n_crease, n_corner, n_area


def fixed_feature_sample(
    points: np.ndarray,
    normals: np.ndarray,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_points <= 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=bool),
        )
    if len(points) == 0:
        return (
            np.zeros((n_points, 3), dtype=np.float32),
            np.zeros((n_points, 3), dtype=np.float32),
            np.zeros((n_points,), dtype=bool),
        )
    return points.astype(np.float32), normals.astype(np.float32), np.ones((len(points),), dtype=bool)


def sample_typed_scaffold_target_points(
    mesh: TessellatedMesh,
    n_points: int,
    boundary_fraction: float,
    crease_fraction: float,
    corner_fraction: float,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    n_boundary, n_crease, n_corner, _n_area = scaffold_target_counts(
        n_points,
        boundary_fraction,
        crease_fraction,
        corner_fraction,
    )
    boundary_points, boundary_normals, boundary_mask = fixed_feature_sample(
        *sample_boundary_points(mesh, n_boundary, seed=seed + 3),
        n_points=n_boundary,
    )
    crease_points, crease_normals, crease_mask = fixed_feature_sample(
        *sample_crease_points(mesh, n_crease, seed=seed + 2),
        n_points=n_crease,
    )
    corner_points, corner_normals, corner_mask = fixed_feature_sample(
        *sample_corner_points(mesh, n_corner, seed=seed + 1),
        n_points=n_corner,
    )
    valid_feature_points = [
        boundary_points[boundary_mask],
        crease_points[crease_mask],
        corner_points[corner_mask],
    ]
    valid_feature_normals = [
        boundary_normals[boundary_mask],
        crease_normals[crease_mask],
        corner_normals[corner_mask],
    ]
    n_valid_features = sum(len(points) for points in valid_feature_points)
    n_area_fill = max(0, n_points - n_valid_features)
    area_points, area_normals = sample_surface_points(mesh, n_area_fill, seed=seed + 4) if n_area_fill else (
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((0, 3), dtype=np.float32),
    )
    all_parts = [*valid_feature_points, area_points]
    all_normals = [*valid_feature_normals, area_normals]
    all_points = np.concatenate(all_parts, axis=0) if all_parts else np.zeros((0, 3), dtype=np.float32)
    all_normal_values = np.concatenate(all_normals, axis=0) if all_normals else np.zeros((0, 3), dtype=np.float32)
    if len(all_points) != n_points:
        raise RuntimeError(f"scaffold target sampler returned {len(all_points)} points, expected {n_points}")
    all_mask = np.ones((len(all_points),), dtype=bool)
    return {
        "all_points": all_points.astype(np.float32),
        "all_normals": all_normal_values.astype(np.float32),
        "all_mask": all_mask,
        "boundary_points": boundary_points,
        "boundary_normals": boundary_normals,
        "boundary_mask": boundary_mask,
        "crease_points": crease_points,
        "crease_normals": crease_normals,
        "crease_mask": crease_mask,
        "corner_points": corner_points,
        "corner_normals": corner_normals,
        "corner_mask": corner_mask,
    }


def sample_scaffold_target_points(
    mesh: TessellatedMesh,
    n_points: int,
    boundary_fraction: float,
    crease_fraction: float,
    corner_fraction: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if n_points <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    sampled = sample_typed_scaffold_target_points(
        mesh,
        n_points=n_points,
        boundary_fraction=boundary_fraction,
        crease_fraction=crease_fraction,
        corner_fraction=corner_fraction,
        seed=seed,
    )
    return sampled["all_points"].astype(np.float32), sampled["all_normals"].astype(np.float32)


def _nearest_distance(points: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    if len(anchors) == 0:
        return np.ones((len(points),), dtype=np.float32)
    diff = points[:, None, :] - anchors[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    return np.min(d, axis=1).astype(np.float32)


def stable_record_seed(base_seed: int, canonical_part_id: str) -> int:
    part_hash = zlib.crc32(canonical_part_id.encode("utf-8")) & 0xFFFFFFFF
    return int((int(base_seed) + part_hash) % (2**32 - 1))


def sample_seed_for_variant(
    base_seed: int,
    canonical_part_id: str,
    variant: str,
    resample_step: int | None = None,
) -> int:
    key = f"{canonical_part_id}|{variant}"
    if resample_step is not None:
        key = f"{key}|resample:{int(resample_step)}"
    return stable_record_seed(base_seed, key)

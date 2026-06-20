"""Deterministic mesh metrics used by the R0 audit pipeline."""

from __future__ import annotations

import hashlib
import math
import pathlib
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import trimesh


METRIC_DEFINITION_ID = "r0_mesh_metrics_v1"
ASPECT_DEFINITION_ID = "triangle_altitude_aspect_ratio_v1"
DISTANCE_DEFINITION_ID = "area_uniform_point_to_triangle_surface_v1"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_triangle_mesh(path: pathlib.Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]
        if not meshes:
            raise ValueError(f"No triangle mesh found in {path}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type in {path}: {type(loaded)!r}")
    if loaded.faces.ndim != 2 or loaded.faces.shape[1] != 3:
        raise ValueError(f"R0 supports triangle meshes only: {path}")
    return loaded


def nearest_rank(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return math.nan
    ordered = np.sort(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile / 100.0 * len(ordered)) - 1))
    return float(ordered[index])


def _triangle_geometry(mesh: trimesh.Trimesh, area_floor: float) -> dict[str, np.ndarray]:
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    edge_lengths = np.stack(
        [
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ],
        axis=1,
    )
    area = np.asarray(mesh.area_faces, dtype=np.float64)
    longest = edge_lengths.max(axis=1)
    aspect = np.full(len(area), np.inf, dtype=np.float64)
    valid = area > area_floor
    aspect[valid] = longest[valid] ** 2 / (2.0 * area[valid])

    a = edge_lengths[:, 0]
    b = edge_lengths[:, 1]
    c = edge_lengths[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        angles = np.stack(
            [
                np.degrees(np.arccos(np.clip((a * a + c * c - b * b) / (2 * a * c), -1, 1))),
                np.degrees(np.arccos(np.clip((a * a + b * b - c * c) / (2 * a * b), -1, 1))),
                np.degrees(np.arccos(np.clip((b * b + c * c - a * a) / (2 * b * c), -1, 1))),
            ],
            axis=1,
        )
    angles[~np.isfinite(angles)] = 0.0
    return {
        "area": area,
        "edge_lengths": edge_lengths,
        "aspect": aspect,
        "angles": angles,
    }


def _edge_counts(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.sort(
        np.concatenate(
            [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
        ),
        axis=1,
    )
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique, counts


def _duplicate_face_count(faces: np.ndarray) -> int:
    canonical = np.sort(faces, axis=1)
    _, counts = np.unique(canonical, axis=0, return_counts=True)
    return int(np.maximum(counts - 1, 0).sum())


def _vertex_component_count(vertex_count: int, unique_edges: np.ndarray) -> int:
    if vertex_count == 0:
        return 0
    if len(unique_edges) == 0:
        return vertex_count
    rows = np.concatenate([unique_edges[:, 0], unique_edges[:, 1]])
    cols = np.concatenate([unique_edges[:, 1], unique_edges[:, 0]])
    graph = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(vertex_count, vertex_count),
    ).tocsr()
    count, _ = connected_components(graph, directed=False)
    return int(count)


def _face_component_count(mesh: trimesh.Trimesh) -> int:
    face_count = len(mesh.faces)
    if face_count == 0:
        return 0
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if adjacency.size == 0:
        return face_count
    rows = np.concatenate([adjacency[:, 0], adjacency[:, 1]])
    cols = np.concatenate([adjacency[:, 1], adjacency[:, 0]])
    graph = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(face_count, face_count),
    ).tocsr()
    count, _ = connected_components(graph, directed=False)
    return int(count)


def mesh_metrics(mesh: trimesh.Trimesh, area_floor: float) -> dict[str, Any]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    geometry = _triangle_geometry(mesh, area_floor)
    unique_edges, edge_use = _edge_counts(faces)
    aspect = geometry["aspect"]
    angles = geometry["angles"].reshape(-1)
    finite_aspect = aspect[np.isfinite(aspect)]
    return {
        "metric_definition_id": METRIC_DEFINITION_ID,
        "aspect_definition_id": ASPECT_DEFINITION_ID,
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(faces)),
        "surface_area_mm2": float(geometry["area"].sum()),
        "boundary_edge_count": int(np.count_nonzero(edge_use == 1)),
        "non_manifold_edge_count": int(np.count_nonzero(edge_use > 2)),
        "duplicate_face_count": _duplicate_face_count(faces),
        "zero_length_edge_count": int(
            np.count_nonzero(geometry["edge_lengths"].reshape(-1) <= 0.0)
        ),
        "degenerate_face_count": int(np.count_nonzero(geometry["area"] <= area_floor)),
        "vertex_connected_component_count": _vertex_component_count(
            len(mesh.vertices), unique_edges
        ),
        "face_connected_component_count": _face_component_count(mesh),
        "minimum_angle_deg": float(angles.min()) if len(angles) else math.nan,
        "angle_p01_deg": nearest_rank(angles, 1),
        "maximum_angle_deg": float(angles.max()) if len(angles) else math.nan,
        "altitude_aspect_ratio_p95": (
            math.inf if len(finite_aspect) != len(aspect) else nearest_rank(aspect, 95)
        ),
        "altitude_aspect_ratio_max": (
            math.inf if len(finite_aspect) != len(aspect)
            else float(finite_aspect.max()) if len(finite_aspect) else math.nan
        ),
        "edge_length_min_mm": float(geometry["edge_lengths"].min()),
        "edge_length_p01_mm": nearest_rank(geometry["edge_lengths"].reshape(-1), 1),
        "edge_length_p05_mm": nearest_rank(geometry["edge_lengths"].reshape(-1), 5),
        "edge_length_median_mm": nearest_rank(geometry["edge_lengths"].reshape(-1), 50),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
    }


def sample_surface_points(
    mesh: trimesh.Trimesh, sample_count: int, seed: int
) -> np.ndarray:
    area = np.asarray(mesh.area_faces, dtype=np.float64)
    total = float(area.sum())
    if total <= 0:
        raise ValueError("Cannot sample a zero-area mesh")
    rng = np.random.default_rng(seed)
    face_index = rng.choice(len(area), size=sample_count, p=area / total)
    triangles = np.asarray(mesh.triangles)[face_index]
    u = rng.random(sample_count)
    v = rng.random(sample_count)
    reflected = u + v > 1.0
    u[reflected] = 1.0 - u[reflected]
    v[reflected] = 1.0 - v[reflected]
    return (
        triangles[:, 0]
        + u[:, None] * (triangles[:, 1] - triangles[:, 0])
        + v[:, None] * (triangles[:, 2] - triangles[:, 0])
    )


def point_to_surface_stats(
    source: trimesh.Trimesh,
    target: trimesh.Trimesh,
    sample_count: int,
    seed: int,
    sample_artifact_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    samples = sample_surface_points(source, sample_count, seed)
    _, distances, _ = trimesh.proximity.closest_point(target, samples)
    distances = np.asarray(distances, dtype=np.float64)
    result = {
        "metric_definition_id": DISTANCE_DEFINITION_ID,
        "sample_count": int(sample_count),
        "seed": int(seed),
        "mean_mm": float(distances.mean()),
        "median_mm": nearest_rank(distances, 50),
        "p95_mm": nearest_rank(distances, 95),
        "p99_mm": nearest_rank(distances, 99),
        "sampled_max_mm": float(distances.max()),
    }
    if sample_artifact_path is not None:
        np.savez_compressed(
            sample_artifact_path,
            sample_xyz_mm=samples.astype(np.float32),
            distance_mm=distances.astype(np.float32),
        )
        result["sample_artifact"] = {
            "path": sample_artifact_path.name,
            "sha256": sha256_file(sample_artifact_path),
        }
    return result

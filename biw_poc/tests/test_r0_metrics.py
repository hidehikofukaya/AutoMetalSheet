from __future__ import annotations

import pathlib
import sys

import numpy as np
import trimesh

R0_DIR = pathlib.Path(__file__).parents[1] / "src" / "r0"
sys.path.insert(0, str(R0_DIR))

from metrics import mesh_metrics, nearest_rank, point_to_surface_stats


def test_nearest_rank_is_deterministic():
    assert nearest_rank(np.array([5.0, 1.0, 3.0, 2.0, 4.0]), 50) == 3.0
    assert nearest_rank(np.arange(1, 101), 95) == 95.0


def test_altitude_aspect_ratio_for_equilateral_triangle():
    height = np.sqrt(3.0) / 2.0
    mesh = trimesh.Trimesh(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, height, 0.0]]),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )
    result = mesh_metrics(mesh, area_floor=1.0e-12)
    assert np.isclose(result["altitude_aspect_ratio_p95"], 2.0 / np.sqrt(3.0))
    assert result["boundary_edge_count"] == 3
    assert result["non_manifold_edge_count"] == 0


def test_point_to_surface_uses_triangle_surface_not_vertices():
    target = trimesh.Trimesh(
        vertices=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )
    source = target.copy()
    source.vertices[:, 2] = 2.0
    stats = point_to_surface_stats(source, target, sample_count=1000, seed=7)
    assert np.isclose(stats["mean_mm"], 2.0)
    assert np.isclose(stats["sampled_max_mm"], 2.0)

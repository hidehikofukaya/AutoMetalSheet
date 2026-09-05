from __future__ import annotations

import pathlib
import sys

import numpy as np
import trimesh

R0_DIR = pathlib.Path(__file__).parents[1] / "src" / "r0"
sys.path.insert(0, str(R0_DIR))

from adaptive_remesh import (
    RefinementConfig,
    canonical_mesh_hash,
    refine,
    select_non_conflicting_edges,
    split_edges_conformingly,
    short_edge_count,
    unique_edges_with_faces,
)


def square_mesh() -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]]
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        process=False,
    )


def test_shared_edge_split_is_conforming():
    mesh = square_mesh()
    vertices, faces = split_edges_conformingly(
        np.asarray(mesh.vertices), np.asarray(mesh.faces), np.array([[0, 2]])
    )
    assert len(vertices) == 5
    assert len(faces) == 4
    edges, incident = unique_edges_with_faces(faces)
    assert not any(tuple(edge) == (0, 2) for edge in edges)
    midpoint_edges = {
        tuple(edge): len(face_ids) for edge, face_ids in zip(edges, incident) if 4 in edge
    }
    assert midpoint_edges[(0, 4)] == 2
    assert midpoint_edges[(2, 4)] == 2


def test_selection_obeys_floor_fraction_and_growth_budget():
    mesh = square_mesh()
    config = RefinementConfig(
        target_edge_mm=0.5,
        h_floor_mm=0.1,
        max_split_edge_fraction_per_sweep=0.25,
        max_vertex_growth_ratio_per_sweep=1.25,
        hard_max_vertices=100,
    )
    selected, report = select_non_conflicting_edges(
        np.asarray(mesh.vertices), np.asarray(mesh.faces), config
    )
    assert len(selected) <= 1
    assert report["selection_limit"] == 1


def test_refine_is_deterministic_and_bounded():
    mesh = square_mesh()
    config = RefinementConfig(
        target_edge_mm=0.7,
        h_floor_mm=0.2,
        max_sweeps=3,
        max_split_edge_fraction_per_sweep=0.5,
        max_vertex_growth_ratio_per_sweep=1.5,
        hard_max_vertices=20,
    )
    stages_a, reports_a = refine(mesh, config)
    stages_b, reports_b = refine(mesh, config)
    assert reports_a == reports_b
    assert len(stages_a[-1].vertices) <= 20
    assert np.array_equal(stages_a[-1].faces, stages_b[-1].faces)
    assert np.array_equal(stages_a[-1].vertices, stages_b[-1].vertices)


def test_selection_rejects_short_opposite_child_edge():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 0.1, 0.0]]
    )
    faces = np.array([[0, 1, 2]])
    config = RefinementConfig(
        target_edge_mm=4.0,
        h_floor_mm=0.5,
        max_split_edge_fraction_per_sweep=1.0,
        max_vertex_growth_ratio_per_sweep=2.0,
        hard_max_vertices=10,
    )
    selected, report = select_non_conflicting_edges(vertices, faces, config)
    assert len(selected) == 0
    assert report["rejected_child_floor_count"] >= 1


def test_canonical_mesh_hash_is_deterministic():
    mesh = square_mesh()
    first = canonical_mesh_hash(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    second = canonical_mesh_hash(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    assert first == second


def test_short_edge_count_tracks_existing_floor_violations():
    mesh = square_mesh()
    assert short_edge_count(
        np.asarray(mesh.vertices), np.asarray(mesh.faces), threshold_mm=0.5
    ) == 0

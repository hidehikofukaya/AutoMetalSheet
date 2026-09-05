from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
import torch

from cae_mesh_generator.model.hierarchical_ae import (
    HierarchicalMidsurfaceAutoencoder,
    StructuredScaffoldAutoencoder,
    TypedCrossAttentionLattice,
    chamfer_loss,
    normal_chamfer_loss,
    prediction_surface_fraction,
    prediction_surface_loss,
    scaffold_chamfer_loss,
    target_coverage_fraction,
    target_coverage_loss,
)
from cae_mesh_generator.data.fill_volume_dataset import sample_seed_for_variant, stable_record_seed
from cae_mesh_generator.data.fill_volume_dataset import MidsurfacePointCloudDataset, PartRecord, mirror_points
from cae_mesh_generator.data.fill_volume_dataset import sample_scaffold_target_points
from cae_mesh_generator.data.fill_volume_dataset import sample_typed_scaffold_target_points
from cae_mesh_generator.data.source_fingerprint import compare_fingerprint_rows
from cae_mesh_generator.data.step_tessellate import (
    TessellatedMesh,
    TessellationConfig,
    boundary_edges_from_faces,
    corner_points_from_edges,
    crease_edges_from_faces,
    face_normals,
)
from cae_mesh_generator.evaluate_autoencoder import (
    fraction_within,
    load_model_from_checkpoint,
    make_metrics,
    nearest_distances,
    select_indexes,
    summarize_distances,
)
from cae_mesh_generator.diagnose_scaffold_placement import ScaffoldPlacementMetrics, aggregate_group
from cae_mesh_generator.train_autoencoder import (
    PartSizeProfile,
    accumulate_metric_totals,
    apply_resume_model_args,
    boundary_distance_metrics,
    best_metric_mode_for,
    build_extra_best_trackers,
    checkpoint_name_for_best_metric,
    composite_cae_score,
    filter_size_profile_by_quantile,
    finalize_metric_totals,
    feature_dim_from_args,
    feature_scaffold_target_metrics,
    initial_best_metric_value,
    is_better_metric,
    model_config_from_model,
    nearest_distance_metrics_mm,
    refinement_occupancy_metrics,
    resolve_best_metric,
    should_run_validation,
    split_indices,
    take_evenly_spaced_by_size,
    validate_missing_best_metric,
    zero_metric_totals,
)


def test_autoencoder_forward_shapes() -> None:
    model = HierarchicalMidsurfaceAutoencoder(
        feature_dim=7,
        token_dim=32,
        n_points_out=64,
        n_coarse=16,
        n_patches=8,
        k_neighbors=8,
        n_latents=8,
    )
    points = torch.rand(2, 64, 3) - 0.5
    normals = torch.nn.functional.normalize(torch.rand(2, 64, 3) - 0.5, dim=-1)
    joint_d = torch.rand(2, 64, 1)
    features = torch.cat([points, normals, joint_d], dim=-1)
    out = model(points, features)
    assert out["points"].shape == (2, 64, 3)
    assert out["normals"].shape == (2, 64, 3)
    assert out["latent"].shape == (2, 8, 32)
    assert torch.isfinite(chamfer_loss(out["points"], points))
    assert torch.isfinite(normal_chamfer_loss(out["points"], out["normals"], points, normals))


def test_structured_autoencoder_forward_shapes() -> None:
    model = StructuredScaffoldAutoencoder(
        feature_dim=7,
        token_dim=32,
        n_points_out=64,
        n_coarse=16,
        n_patches=12,
        k_neighbors=8,
        n_latents=8,
        n_scaffold=16,
        points_per_scaffold=4,
        n_local_tokens=4,
    )
    points = torch.rand(2, 64, 3) - 0.5
    normals = torch.nn.functional.normalize(torch.rand(2, 64, 3) - 0.5, dim=-1)
    joint_d = torch.ones(2, 64, 1)
    features = torch.cat([points, normals, joint_d], dim=-1)
    out = model(points, features)
    assert out["points"].shape == (2, 64, 3)
    assert out["normals"].shape == (2, 64, 3)
    assert out["scaffold_points"].shape == (2, 16, 3)
    assert out["refinement_logits"].shape == (2, 64)
    assert out["refinement_logits_grid"].shape == (2, 16, 4)
    assert out["active_scaffold_mask"].shape == (2, 16)
    assert torch.all(out["active_scaffold_mask"])
    assert torch.all(out["scaffold_point_counts"] == 4)
    assert torch.isfinite(chamfer_loss(out["points"], points))


def test_structured_autoencoder_anchor_mode_uses_bounded_residuals() -> None:
    model = StructuredScaffoldAutoencoder(
        feature_dim=7,
        token_dim=32,
        n_points_out=64,
        n_coarse=16,
        n_patches=12,
        k_neighbors=8,
        n_latents=8,
        n_scaffold=16,
        points_per_scaffold=4,
        n_local_tokens=4,
        scaffold_mode="anchored",
        scaffold_anchor_source="coarse_fine",
        scaffold_anchor_residual_scale=0.05,
    )
    points = torch.rand(2, 64, 3) - 0.5
    normals = torch.nn.functional.normalize(torch.rand(2, 64, 3) - 0.5, dim=-1)
    joint_d = torch.ones(2, 64, 1)
    features = torch.cat([points, normals, joint_d], dim=-1)
    out = model(points, features)
    assert out["points"].shape == (2, 64, 3)
    assert out["scaffold_points"].shape == (2, 16, 3)
    assert out["scaffold_anchor_points"].shape == (2, 16, 3)
    assert out["scaffold_residuals"].shape == (2, 16, 3)
    assert out["scaffold_tokens"].shape == (2, 16, 32)
    assert float(torch.max(torch.abs(out["scaffold_residuals"]))) <= 0.050001
    assert torch.allclose(
        out["scaffold_points"],
        out["scaffold_anchor_points"] + out["scaffold_residuals"],
        atol=1.0e-6,
    )
    config = model_config_from_model(model)
    assert config["scaffold_mode"] == "anchored"
    assert config["scaffold_anchor_source"] == "coarse_fine"
    assert config["scaffold_anchor_residual_scale"] == pytest.approx(0.05)


def test_structured_autoencoder_anchor_mode_repeats_when_candidates_are_few() -> None:
    model = StructuredScaffoldAutoencoder(
        feature_dim=7,
        token_dim=16,
        n_points_out=24,
        n_coarse=4,
        n_patches=4,
        k_neighbors=4,
        n_latents=4,
        n_scaffold=12,
        points_per_scaffold=2,
        n_local_tokens=2,
        scaffold_mode="anchored",
        scaffold_anchor_source="fine",
        scaffold_anchor_residual_scale=0.02,
    )
    points = torch.rand(1, 24, 3) - 0.5
    normals = torch.nn.functional.normalize(torch.rand(1, 24, 3) - 0.5, dim=-1)
    joint_d = torch.ones(1, 24, 1)
    features = torch.cat([points, normals, joint_d], dim=-1)
    out = model(points, features)
    assert out["scaffold_points"].shape == (1, 12, 3)
    assert out["scaffold_anchor_points"].shape == (1, 12, 3)
    assert torch.allclose(out["scaffold_anchor_points"][:, :4, :], out["scaffold_anchor_points"][:, 4:8, :])
    assert torch.allclose(out["scaffold_anchor_points"][:, :4, :], out["scaffold_anchor_points"][:, 8:12, :])


def test_evaluate_loader_restores_anchored_scaffold_checkpoint() -> None:
    model = StructuredScaffoldAutoencoder(
        feature_dim=7,
        token_dim=16,
        n_points_out=40,
        n_coarse=8,
        n_patches=6,
        k_neighbors=4,
        n_latents=4,
        n_scaffold=10,
        points_per_scaffold=4,
        n_local_tokens=2,
        scaffold_mode="anchored",
        scaffold_anchor_source="coarse_fine",
        scaffold_anchor_residual_scale=0.03,
    )
    payload = {
        "model_state": model.state_dict(),
        "args": {
            "model_kind": "structured",
            "token_dim": 16,
            "n_points": 40,
            "n_coarse": 8,
            "n_patches": 6,
            "k_neighbors": 4,
            "n_latents": 4,
            "n_local_tokens": 2,
            "use_boundary_feature": False,
        },
        "model_config": model_config_from_model(model),
    }
    restored = load_model_from_checkpoint(payload, torch.device("cpu"))
    assert isinstance(restored, StructuredScaffoldAutoencoder)
    assert restored.scaffold_mode == "anchored"
    assert restored.scaffold_anchor_residual_scale == pytest.approx(0.03)
    points = torch.rand(1, 40, 3) - 0.5
    normals = torch.nn.functional.normalize(torch.rand(1, 40, 3) - 0.5, dim=-1)
    joint_d = torch.ones(1, 40, 1)
    features = torch.cat([points, normals, joint_d], dim=-1)
    out = restored(points, features)
    assert out["points"].shape == (1, 40, 3)
    assert out["scaffold_anchor_points"].shape == (1, 10, 3)


def test_structured_autoencoder_accepts_boundary_feature_token() -> None:
    model = StructuredScaffoldAutoencoder(
        feature_dim=8,
        token_dim=32,
        n_points_out=64,
        n_coarse=16,
        n_patches=12,
        k_neighbors=8,
        n_latents=8,
        n_scaffold=16,
        points_per_scaffold=4,
        n_local_tokens=4,
        boundary_feature_index=7,
    )
    points = torch.rand(2, 64, 3) - 0.5
    normals = torch.nn.functional.normalize(torch.rand(2, 64, 3) - 0.5, dim=-1)
    joint_d = torch.ones(2, 64, 1)
    boundary = torch.zeros(2, 64, 1)
    boundary[:, :8] = 1.0
    features = torch.cat([points, normals, joint_d, boundary], dim=-1)
    out = model(points, features)
    assert out["points"].shape == (2, 64, 3)
    assert out["latent"].shape == (2, 8, 32)
    assert model.feature_dim == 8
    assert model.boundary_feature_index == 7
    no_boundary_features = features.clone()
    no_boundary_features[..., 7] = 0.0
    assert torch.allclose(model.boundary(no_boundary_features), torch.zeros(2, 1, 32), atol=1.0e-6)


def test_typed_cross_attention_lattice_updates_streams() -> None:
    lattice = TypedCrossAttentionLattice(token_dim=16, n_layers=2, n_heads=4)
    global_tokens = torch.rand(2, 5, 16)
    local_tokens = torch.rand(2, 7, 16)
    boundary_tokens = torch.rand(2, 3, 16)
    out = lattice(global_tokens, local_tokens, boundary_tokens)
    assert out["global"].shape == global_tokens.shape
    assert out["local"].shape == local_tokens.shape
    assert out["boundary"].shape == boundary_tokens.shape
    out_no_boundary = lattice(global_tokens, local_tokens, None)
    assert out_no_boundary["global"].shape == global_tokens.shape
    assert out_no_boundary["local"].shape == local_tokens.shape
    assert out_no_boundary["boundary"] is None


def test_structured_autoencoder_accepts_lattice_processor() -> None:
    model = StructuredScaffoldAutoencoder(
        feature_dim=8,
        token_dim=32,
        n_points_out=64,
        n_coarse=16,
        n_patches=12,
        k_neighbors=8,
        n_latents=8,
        n_scaffold=16,
        points_per_scaffold=4,
        n_local_tokens=4,
        boundary_feature_index=7,
        boundary_token_count=4,
        lattice_layers=1,
        lattice_heads=4,
    )
    points = torch.rand(2, 64, 3) - 0.5
    normals = torch.nn.functional.normalize(torch.rand(2, 64, 3) - 0.5, dim=-1)
    joint_d = torch.ones(2, 64, 1)
    boundary = torch.zeros(2, 64, 1)
    boundary[:, :8] = 1.0
    features = torch.cat([points, normals, joint_d, boundary], dim=-1)
    out = model(points, features)
    assert out["points"].shape == (2, 64, 3)
    assert out["scaffold_points"].shape == (2, 16, 3)
    assert out["latent"].shape == (2, 8, 32)
    assert model.n_lattice_layers == 1
    assert model.boundary_token_count == 4
    assert model.boundary(features).shape == (2, 4, 32)
    no_boundary_features = features.clone()
    no_boundary_features[..., 7] = 0.0
    assert torch.allclose(model.boundary(no_boundary_features), torch.zeros(2, 4, 32), atol=1.0e-6)


def test_structured_autoencoder_tangent_refinement_outputs_patch_frame() -> None:
    model = StructuredScaffoldAutoencoder(
        feature_dim=7,
        token_dim=32,
        n_points_out=64,
        n_coarse=16,
        n_patches=12,
        k_neighbors=8,
        n_latents=8,
        n_scaffold=16,
        points_per_scaffold=4,
        n_local_tokens=4,
        refinement_mode="tangent",
        tangent_offset_scale=0.15,
        normal_offset_scale=0.01,
        patch_type_count=4,
    )
    points = torch.rand(2, 64, 3) - 0.5
    normals = torch.nn.functional.normalize(torch.rand(2, 64, 3) - 0.5, dim=-1)
    joint_d = torch.ones(2, 64, 1)
    features = torch.cat([points, normals, joint_d], dim=-1)
    out = model(points, features)
    assert out["points"].shape == (2, 64, 3)
    assert out["refinement_frame_normals"].shape == (2, 16, 3)
    assert out["refinement_frame_tangent_u"].shape == (2, 16, 3)
    assert out["refinement_frame_tangent_v"].shape == (2, 16, 3)
    assert out["patch_scales"].shape == (2, 16, 3)
    assert out["patch_type_logits"].shape == (2, 16, 4)
    assert torch.all(out["patch_scales"][..., :2] <= 0.15)
    assert torch.all(out["patch_scales"][..., 2] <= 0.01)
    assert torch.allclose(out["refinement_frame_normals"].norm(dim=-1), torch.ones(2, 16), atol=1.0e-5)


def test_structured_autoencoder_keeps_requested_point_count() -> None:
    model = StructuredScaffoldAutoencoder(
        feature_dim=7,
        token_dim=24,
        n_points_out=70,
        n_coarse=16,
        n_patches=12,
        k_neighbors=8,
        n_latents=8,
        n_scaffold=16,
        points_per_scaffold=None,
        n_local_tokens=4,
    )
    points = torch.rand(2, 70, 3) - 0.5
    normals = torch.nn.functional.normalize(torch.rand(2, 70, 3) - 0.5, dim=-1)
    joint_d = torch.ones(2, 70, 1)
    features = torch.cat([points, normals, joint_d], dim=-1)
    out = model(points, features)
    assert out["points"].shape == (2, 70, 3)
    assert out["normals"].shape == (2, 70, 3)
    assert out["scaffold_points"].shape == (2, 16, 3)
    assert out["refinement_logits"].shape == (2, 70)
    assert out["refinement_logits_grid"].shape == (2, 16, 5)
    assert out["active_scaffold_mask"][0].tolist() == [True] * 14 + [False] * 2
    assert out["scaffold_point_counts"][0].tolist() == [5] * 14 + [0] * 2


def test_target_coverage_loss_penalizes_missed_target_points() -> None:
    target = torch.tensor([[[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]])
    near_pred = torch.tensor([[[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]])
    far_pred = torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]])
    scale = torch.tensor([100.0])
    near_loss = target_coverage_loss(near_pred, target, scale, threshold_mm=5.0)
    far_loss = target_coverage_loss(far_pred, target, scale, threshold_mm=5.0)
    assert near_loss.item() == 0
    assert far_loss.item() > near_loss.item()
    assert target_coverage_fraction(near_pred, target, scale, threshold_mm=5.0).item() == 1.0
    assert target_coverage_fraction(far_pred, target, scale, threshold_mm=5.0).item() == 0.5


def test_prediction_surface_loss_penalizes_false_positive_points() -> None:
    target = torch.tensor([[[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]])
    near_pred = torch.tensor([[[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]])
    far_pred = torch.tensor([[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]])
    scale = torch.tensor([100.0])
    near_loss = prediction_surface_loss(near_pred, target, scale, threshold_mm=5.0)
    far_loss = prediction_surface_loss(far_pred, target, scale, threshold_mm=5.0)
    assert near_loss.item() == 0
    assert far_loss.item() > near_loss.item()
    assert prediction_surface_fraction(near_pred, target, scale, threshold_mm=5.0).item() == 1.0
    assert prediction_surface_fraction(far_pred, target, scale, threshold_mm=5.0).item() == 0.5


def test_refinement_occupancy_metrics_labels_near_and_far_predictions() -> None:
    pred = torch.tensor([[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.2, 0.0, 0.0]]])
    target = torch.tensor([[[0.0, 0.0, 0.0]]])
    logits = torch.tensor([[4.0, -4.0, 0.0]])
    metrics = refinement_occupancy_metrics(
        pred,
        target,
        logits,
        torch.tensor([100.0]),
        positive_threshold_mm=5.0,
        negative_threshold_mm=50.0,
    )
    assert torch.isfinite(metrics["loss"])
    assert metrics["labeled_fraction"].item() == pytest.approx(2.0 / 3.0)
    assert metrics["positive_fraction"].item() == pytest.approx(0.5)
    assert metrics["active_fraction"].item() == pytest.approx(2.0 / 3.0)
    assert metrics["accuracy"].item() == pytest.approx(1.0)


def test_scaffold_chamfer_loss_is_finite_with_active_mask() -> None:
    target = torch.rand(2, 32, 3) - 0.5
    scaffold = torch.rand(2, 8, 3) - 0.5
    mask = torch.tensor([[True, True, True, True, False, False, False, False]] * 2)
    loss = scaffold_chamfer_loss(scaffold, target, active_mask=mask)
    assert torch.isfinite(loss)


def test_boundary_edges_are_extracted_after_coordinate_welding() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    normals = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    edges, edge_normals = boundary_edges_from_faces(vertices, faces, normals, weld_tolerance=1.0e-6)
    assert edges.shape == (4, 2, 3)
    assert edge_normals.shape == (4, 3)
    lengths = np.sort(np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1))
    assert np.allclose(lengths, np.ones(4))


def test_crease_edges_are_extracted_from_normal_discontinuity() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [0, 3, 1]], dtype=np.int64)
    normals = face_normals(vertices, faces)
    edges, edge_normals = crease_edges_from_faces(vertices, faces, normals, angle_degrees=30.0)
    assert edges.shape == (1, 2, 3)
    assert edge_normals.shape == (1, 3)
    assert np.linalg.norm(edges[0, 1] - edges[0, 0]) == pytest.approx(1.0)


def test_corner_points_are_extracted_from_sharp_edge_turns() -> None:
    edges = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    points, normals = corner_points_from_edges(edges)
    assert points.shape[1] == 3
    assert normals.shape == points.shape
    assert np.any(np.all(np.isclose(points, [1.0, 0.0, 0.0]), axis=1))


def test_scaffold_target_sampler_returns_fixed_count() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [0, 3, 1]], dtype=np.int64)
    normals = face_normals(vertices, faces)
    boundary_edges, boundary_normals = boundary_edges_from_faces(vertices, faces, normals)
    crease_edges, crease_normals = crease_edges_from_faces(vertices, faces, normals)
    mesh = TessellatedMesh(
        vertices=vertices,
        faces=faces,
        face_normals=normals,
        boundary_edges=boundary_edges,
        boundary_edge_normals=boundary_normals,
        crease_edges=crease_edges,
        crease_edge_normals=crease_normals,
        bounds_min=vertices.min(axis=0),
        bounds_max=vertices.max(axis=0),
        source_path="synthetic",
        config=TessellationConfig(),
    )
    points, target_normals = sample_scaffold_target_points(
        mesh,
        n_points=12,
        boundary_fraction=0.3,
        crease_fraction=0.3,
        corner_fraction=0.2,
        seed=1,
    )
    assert points.shape == (12, 3)
    assert target_normals.shape == (12, 3)


def test_typed_scaffold_target_sampler_returns_masks() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [0, 3, 1]], dtype=np.int64)
    normals = face_normals(vertices, faces)
    boundary_edges, boundary_normals = boundary_edges_from_faces(vertices, faces, normals)
    crease_edges, crease_normals = crease_edges_from_faces(vertices, faces, normals)
    mesh = TessellatedMesh(
        vertices=vertices,
        faces=faces,
        face_normals=normals,
        boundary_edges=boundary_edges,
        boundary_edge_normals=boundary_normals,
        crease_edges=crease_edges,
        crease_edge_normals=crease_normals,
        bounds_min=vertices.min(axis=0),
        bounds_max=vertices.max(axis=0),
        source_path="synthetic",
        config=TessellationConfig(),
    )
    sampled = sample_typed_scaffold_target_points(
        mesh,
        n_points=12,
        boundary_fraction=0.3,
        crease_fraction=0.3,
        corner_fraction=0.2,
        seed=1,
    )
    assert sampled["all_points"].shape == (12, 3)
    assert sampled["all_mask"].shape == (12,)
    assert sampled["boundary_points"].shape[1] == 3
    assert sampled["crease_points"].shape[1] == 3
    assert sampled["corner_points"].shape[1] == 3
    assert sampled["all_mask"].sum() == 12


def test_typed_scaffold_target_sampler_backfills_missing_feature_slots() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    normals = face_normals(vertices, faces)
    mesh = TessellatedMesh(
        vertices=vertices,
        faces=faces,
        face_normals=normals,
        boundary_edges=np.zeros((0, 2, 3), dtype=np.float32),
        boundary_edge_normals=np.zeros((0, 3), dtype=np.float32),
        crease_edges=np.zeros((0, 2, 3), dtype=np.float32),
        crease_edge_normals=np.zeros((0, 3), dtype=np.float32),
        bounds_min=vertices.min(axis=0),
        bounds_max=vertices.max(axis=0),
        source_path="synthetic",
        config=TessellationConfig(),
    )
    sampled = sample_typed_scaffold_target_points(
        mesh,
        n_points=10,
        boundary_fraction=0.4,
        crease_fraction=0.4,
        corner_fraction=0.1,
        seed=2,
    )
    assert sampled["all_points"].shape == (10, 3)
    assert sampled["all_mask"].all()
    assert not sampled["boundary_mask"].any()
    assert not sampled["crease_mask"].any()
    assert not sampled["corner_mask"].any()
    assert np.any(np.linalg.norm(sampled["all_points"], axis=1) > 0.0)


def test_mirror_points_reflects_requested_axis() -> None:
    points = np.asarray([[1.0, -2.0, 3.0]], dtype=np.float32)
    mirrored = mirror_points(points, "y")
    assert mirrored.tolist() == [[1.0, 2.0, 3.0]]
    assert points.tolist() == [[1.0, -2.0, 3.0]]


def test_dataset_mirror_augmentation_expands_train_records_only_shape() -> None:
    record = PartRecord(
        assembly_id="asm",
        part_id_raw="001",
        canonical_part_id="asm:001",
        stp_path=Path("dummy.stp"),
        joints_path=None,
    )
    dataset = MidsurfacePointCloudDataset(
        records=[record],
        cache_dir=Path("."),
        n_points=8,
        mirror_axes=["y"],
    )
    assert len(dataset) == 2
    assert dataset.augmentation_factor == 2
    assert dataset.augmentation_variants == ("original", "mirror_y")


def test_boundary_distance_metrics_use_masked_targets() -> None:
    target = torch.tensor([[[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.8, 0.0, 0.0]]])
    pred = torch.tensor([[[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]])
    mask = torch.tensor([[True, False, True]])
    metrics = boundary_distance_metrics(pred, target, mask, torch.tensor([100.0]), threshold_mm=5.0)
    assert metrics["point_count"].item() == 2.0
    assert metrics["within_threshold"].item() == 0.5
    assert metrics["coverage_loss"].item() > 0.0
    assert metrics["p95_mm"].item() > 0.0


def test_boundary_distance_metrics_are_point_weighted() -> None:
    target = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.2, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
        ]
    )
    pred = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    mask = torch.tensor([[True, True, False], [True, False, False]])
    metrics = boundary_distance_metrics(pred, target, mask, torch.tensor([100.0, 100.0]), threshold_mm=5.0)
    assert metrics["point_count"].item() == 3.0
    assert metrics["within_threshold"].item() == pytest.approx(2.0 / 3.0)


def test_feature_scaffold_target_metrics_are_one_way() -> None:
    scaffold = torch.tensor([[[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]])
    targets = torch.tensor([[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]])
    metrics = feature_scaffold_target_metrics(scaffold, targets, None, torch.tensor([100.0]))
    assert metrics["target_count"].item() == 2.0
    assert metrics["loss"].item() > 0.0
    assert metrics["p95_mm"].item() > 0.0


def test_feature_scaffold_target_metrics_ignore_masked_targets() -> None:
    scaffold = torch.tensor([[[0.0, 0.0, 0.0]]])
    targets = torch.tensor([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    mask = torch.tensor([[True, False]])
    metrics = feature_scaffold_target_metrics(scaffold, targets, None, torch.tensor([100.0]), target_mask=mask)
    assert metrics["target_count"].item() == 1.0
    assert metrics["loss"].item() == pytest.approx(0.0)
    assert metrics["p95_mm"].item() == pytest.approx(0.0)


def test_scaffold_target_metrics_are_weighted_by_valid_target_count() -> None:
    totals = zero_metric_totals()
    first = {key: 0.0 for key in totals}
    second = {key: 0.0 for key in totals}
    first.update({"boundary_scaffold_p95_mm": 0.0, "boundary_scaffold_target_count": 0.0})
    second.update({"boundary_scaffold_p95_mm": 10.0, "boundary_scaffold_target_count": 4.0})
    accumulate_metric_totals(totals, first, batch_size=1)
    accumulate_metric_totals(totals, second, batch_size=1)
    averaged = finalize_metric_totals(totals, sample_count=2, boundary_point_count=0.0)
    assert averaged["boundary_scaffold_p95_mm"] == pytest.approx(10.0)
    assert averaged["boundary_scaffold_target_count"] == pytest.approx(2.0)


def test_size_quantile_filter_selects_middle_lower_band() -> None:
    profile = [
        PartSizeProfile(f"part:{i}", f"{i}.stp", bbox_diagonal=float(i), bbox_max_extent=float(i))
        for i in range(10)
    ]
    selected = filter_size_profile_by_quantile(profile, "bbox_diagonal", 0.2, 0.4)
    assert [p.canonical_part_id for p in selected] == ["part:2", "part:3"]


def test_evenly_spaced_size_selection_covers_band() -> None:
    profile = [
        PartSizeProfile(f"part:{i}", f"{i}.stp", bbox_diagonal=float(i), bbox_max_extent=float(i))
        for i in range(6)
    ]
    selected = take_evenly_spaced_by_size(profile, max_parts=3, metric="bbox_diagonal")
    assert [p.canonical_part_id for p in selected] == ["part:0", "part:2", "part:5"]


def test_even_split_holds_out_across_selected_band() -> None:
    train, val = split_indices(n_items=10, val_count=2, strategy="even")
    assert val == [0, 9]
    assert train == [1, 2, 3, 4, 5, 6, 7, 8]


def test_validation_schedule_and_best_metric() -> None:
    assert resolve_best_metric("auto", has_validation=True) == "val_loss"
    assert resolve_best_metric("auto", has_validation=False) == "train_loss"
    assert resolve_best_metric("train_loss", has_validation=True) == "train_loss"
    assert resolve_best_metric("val_target_within_threshold", has_validation=True) == "val_target_within_threshold"
    assert resolve_best_metric("val_pred_within_threshold", has_validation=True) == "val_pred_within_threshold"
    assert resolve_best_metric("val_cae_score", has_validation=True) == "val_cae_score"
    assert best_metric_mode_for("val_loss") == "min"
    assert best_metric_mode_for("val_target_p95_mm") == "min"
    assert best_metric_mode_for("val_target_within_threshold") == "max"
    assert best_metric_mode_for("val_pred_within_threshold") == "max"
    assert best_metric_mode_for("val_cae_score") == "min"
    assert initial_best_metric_value("min") == float("inf")
    assert initial_best_metric_value("max") == -float("inf")
    assert is_better_metric(0.9, 0.8, "max")
    assert not is_better_metric(0.7, 0.8, "max")
    assert is_better_metric(10.0, 12.0, "min")
    assert not is_better_metric(14.0, 12.0, "min")
    validate_missing_best_metric("val_target_p95_mm", {"train_loss": 1.0}, epoch=2)
    assert should_run_validation(1, max_epochs=10, eval_every=5, has_validation=True)
    assert should_run_validation(5, max_epochs=10, eval_every=5, has_validation=True)
    assert should_run_validation(10, max_epochs=10, eval_every=5, has_validation=True)
    assert not should_run_validation(2, max_epochs=10, eval_every=5, has_validation=True)
    assert not should_run_validation(5, max_epochs=10, eval_every=5, has_validation=False)


def test_extra_best_trackers_skip_primary_and_encode_mode() -> None:
    trackers = build_extra_best_trackers(
        ["val_loss", "val_target_within_threshold", "val_target_p95_mm", "val_cae_score"],
        primary_metric="val_loss",
        has_validation=True,
    )
    assert list(trackers) == ["val_target_within_threshold", "val_target_p95_mm", "val_cae_score"]
    assert trackers["val_target_within_threshold"]["mode"] == "max"
    assert trackers["val_target_within_threshold"]["value"] == -float("inf")
    assert trackers["val_target_p95_mm"]["mode"] == "min"
    assert trackers["val_target_p95_mm"]["value"] == float("inf")
    assert trackers["val_cae_score"]["mode"] == "min"
    assert checkpoint_name_for_best_metric("val_target_p95_mm") == "best_by_val_target_p95_mm.pt"


def test_resume_model_args_restore_lattice_shape_args() -> None:
    args = SimpleNamespace(
        model_kind="point",
        use_boundary_feature=False,
        n_points=32,
        token_dim=16,
        n_coarse=8,
        n_patches=8,
        k_neighbors=4,
        n_latents=4,
        n_scaffold=8,
        points_per_scaffold=None,
        n_local_tokens=2,
        boundary_token_count=1,
        lattice_layers=0,
        lattice_heads=4,
        refinement_mode="free",
        tangent_offset_scale=0.20,
        normal_offset_scale=0.02,
        patch_type_count=4,
        scaffold_mode="learned",
        scaffold_anchor_source="coarse_fine",
        scaffold_anchor_residual_scale=0.10,
    )
    payload = {
        "args": {
            "model_kind": "structured",
            "use_boundary_feature": True,
            "n_points": 64,
            "token_dim": 32,
            "n_coarse": 16,
            "n_patches": 12,
            "k_neighbors": 8,
            "n_latents": 8,
            "n_scaffold": 16,
            "points_per_scaffold": 4,
            "n_local_tokens": 4,
            "boundary_token_count": 4,
            "lattice_layers": 2,
            "lattice_heads": 4,
            "refinement_mode": "tangent",
            "tangent_offset_scale": 0.15,
            "normal_offset_scale": 0.01,
            "patch_type_count": 5,
            "scaffold_mode": "anchored",
            "scaffold_anchor_source": "coarse",
            "scaffold_anchor_residual_scale": 0.06,
        },
        "model_config": {
            "feature_dim": 8,
            "n_points_out": 64,
            "n_scaffold": 16,
            "points_per_scaffold": 4,
            "boundary_token_count": 4,
            "lattice_layers": 2,
            "lattice_heads": 4,
            "refinement_mode": "tangent",
            "tangent_offset_scale": 0.15,
            "normal_offset_scale": 0.01,
            "patch_type_count": 5,
            "scaffold_mode": "anchored",
            "scaffold_anchor_source": "fine",
            "scaffold_anchor_residual_scale": 0.04,
        },
    }
    apply_resume_model_args(args, payload)
    assert args.model_kind == "structured"
    assert args.use_boundary_feature
    assert args.n_points == 64
    assert args.boundary_token_count == 4
    assert args.lattice_layers == 2
    assert args.refinement_mode == "tangent"
    assert args.tangent_offset_scale == pytest.approx(0.15)
    assert args.normal_offset_scale == pytest.approx(0.01)
    assert args.patch_type_count == 5
    assert args.scaffold_mode == "anchored"
    assert args.scaffold_anchor_source == "fine"
    assert args.scaffold_anchor_residual_scale == pytest.approx(0.04)


def test_composite_cae_score_rewards_boundary_and_target_coverage() -> None:
    args = SimpleNamespace(
        cae_score_target_p95_weight=1.0,
        cae_score_recon_p95_weight=0.1,
        cae_score_boundary_p95_weight=1.0,
        cae_score_chamfer_weight=0.25,
        cae_score_target_within_weight=10.0,
        cae_score_pred_within_weight=5.0,
        cae_score_boundary_within_weight=10.0,
        use_boundary_feature=True,
    )
    weak = {
        "recon_mean_mm": 12.0,
        "recon_p95_mm": 35.0,
        "target_mean_mm": 10.0,
        "target_p95_mm": 30.0,
        "target_within_threshold": 0.1,
        "pred_within_threshold": 0.1,
        "boundary_point_count": 16.0,
        "boundary_p95_mm": 40.0,
        "boundary_within_threshold": 0.1,
    }
    better = dict(weak)
    better.update(
        {
            "target_p95_mm": 20.0,
            "recon_p95_mm": 20.0,
            "target_within_threshold": 0.25,
            "pred_within_threshold": 0.25,
            "boundary_p95_mm": 25.0,
            "boundary_within_threshold": 0.25,
        }
    )
    assert composite_cae_score(better, args) < composite_cae_score(weak, args)
    assert feature_dim_from_args(args) == 8


def test_nearest_distance_training_metrics_are_in_world_mm() -> None:
    pred = torch.tensor([[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]])
    target = torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]])
    metrics = nearest_distance_metrics_mm(pred, target, torch.tensor([100.0]))
    assert metrics["recon_mean_mm"] == 5.0
    assert metrics["target_mean_mm"] == 5.0
    assert metrics["recon_p95_mm"] > metrics["recon_mean_mm"]
    assert metrics["target_p95_mm"] > metrics["target_mean_mm"]


def test_evaluate_select_indexes_filters_by_split() -> None:
    labels = {0: "train", 1: "train", 2: "val", 3: "val"}
    assert select_indexes(4, requested=None, max_parts=0, split_filter="val", split_by_index=labels) == [2, 3]
    assert select_indexes(4, requested=None, max_parts=1, split_filter="train", split_by_index=labels) == [0]


def test_nearest_distance_metrics() -> None:
    query = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]).numpy()
    reference = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).numpy()
    distances, indexes = nearest_distances(query, reference)
    assert distances.tolist() == [0.0, 1.0]
    assert indexes.tolist() == [0, 1]
    summary = summarize_distances(distances)
    assert summary.mean == 0.5
    assert fraction_within(distances, 0.5) == 0.5


def test_fingerprint_compare_reports_missing_checkpoint_fingerprints() -> None:
    current = [{"canonical_part_id": "asm:001", "exists": True, "size_bytes": 10, "sha256": "abc"}]
    report = compare_fingerprint_rows(None, current)
    assert report["status"] == "not_in_checkpoint"


def test_fingerprint_compare_detects_mismatch() -> None:
    expected = [{"canonical_part_id": "asm:001", "exists": True, "size_bytes": 10, "sha256": "abc"}]
    current = [{"canonical_part_id": "asm:001", "exists": True, "size_bytes": 11, "sha256": "def"}]
    report = compare_fingerprint_rows(expected, current)
    assert report["status"] == "mismatch"
    assert len(report["mismatches"]) == 2


def test_stable_record_seed_does_not_depend_on_dataset_index() -> None:
    assert stable_record_seed(7, "asm:001") == stable_record_seed(7, "asm:001")
    assert stable_record_seed(7, "asm:001") != stable_record_seed(7, "asm:002")


def test_sample_seed_for_variant_changes_only_when_resampling_step_is_used() -> None:
    fixed = sample_seed_for_variant(7, "asm:001", "original")
    assert sample_seed_for_variant(7, "asm:001", "original") == fixed
    assert sample_seed_for_variant(7, "asm:001", "mirror_y") != fixed
    assert sample_seed_for_variant(7, "asm:001", "original", resample_step=1) != fixed
    assert sample_seed_for_variant(7, "asm:001", "original", resample_step=1) == sample_seed_for_variant(
        7,
        "asm:001",
        "original",
        resample_step=1,
    )


def test_evaluate_metrics_include_sampling_floor_and_bbox_ratio() -> None:
    recon_to_target = np.asarray([1.0, 3.0], dtype=np.float64)
    target_to_recon = np.asarray([2.0, 4.0], dtype=np.float64)
    metrics = make_metrics(
        part_name="asm:001",
        record_index=0,
        split="val",
        scale_mm=100.0,
        bbox_diagonal_mm=200.0,
        sampling_floor_chamfer_mm=5.0,
        recon_to_target=recon_to_target,
        target_to_recon=target_to_recon,
        boundary_to_recon=np.asarray([], dtype=np.float64),
        n_target=2,
        n_recon=2,
        n_boundary=0,
    )
    assert metrics.chamfer_mm == pytest.approx(5.0)
    assert metrics.chamfer_bbox_diag_pct == pytest.approx(2.5)
    assert metrics.sampling_floor_bbox_diag_pct == pytest.approx(2.5)
    assert metrics.chamfer_to_sampling_floor == pytest.approx(1.0)


def test_scaffold_diagnostic_aggregate_tracks_worst_parts() -> None:
    common = {
        "split": "val",
        "scale_mm": 100.0,
        "bbox_diagonal_mm": 200.0,
        "n_scaffold": 4,
        "n_target": 8,
        "n_boundary": 2,
        "n_crease": 2,
        "n_corner": 1,
        "scaffold_within_5mm": 0.5,
        "scaffold_within_10mm": 0.75,
        "target_within_5mm": 0.25,
        "target_within_10mm": 0.5,
        "boundary_within_5mm": 0.25,
        "boundary_within_10mm": 0.5,
        "scaffold_p95_bbox_diag_pct": 2.0,
        "target_p95_bbox_diag_pct": 3.0,
    }
    first = ScaffoldPlacementMetrics(
        part_name="asm:001",
        record_index=0,
        scaffold_to_target_mm=summarize_distances(np.asarray([1.0, 2.0])),
        target_to_scaffold_mm=summarize_distances(np.asarray([3.0, 4.0])),
        boundary_to_scaffold_mm=summarize_distances(np.asarray([5.0, 6.0])),
        crease_to_scaffold_mm=summarize_distances(np.asarray([2.0, 3.0])),
        corner_to_scaffold_mm=summarize_distances(np.asarray([1.0, 1.0])),
        **common,
    )
    second = ScaffoldPlacementMetrics(
        part_name="asm:002",
        record_index=1,
        scaffold_to_target_mm=summarize_distances(np.asarray([10.0, 20.0])),
        target_to_scaffold_mm=summarize_distances(np.asarray([30.0, 40.0])),
        boundary_to_scaffold_mm=summarize_distances(np.asarray([50.0, 60.0])),
        crease_to_scaffold_mm=summarize_distances(np.asarray([20.0, 30.0])),
        corner_to_scaffold_mm=summarize_distances(np.asarray([10.0, 10.0])),
        **common,
    )
    row = aggregate_group("val", [first, second])
    assert row["count"] == 2
    assert row["worst_scaffold_p95_part"] == "asm:002"
    assert row["worst_target_p95_part"] == "asm:002"
    assert row["worst_boundary_p95_part"] == "asm:002"

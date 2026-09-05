from __future__ import annotations

import pathlib
import sys

import h5py
import numpy as np
import pytest
import torch


MODEL_DIR = pathlib.Path(__file__).parents[1] / "src" / "model"
sys.path.insert(0, str(MODEL_DIR))

from reconstruct_softmin_guidance import (
    InsufficientEvidenceError,
    bounded_project,
    candidate_ranking_score,
    evaluate_grid,
    load_input_h5,
    load_model,
    select_ranked_candidates,
)
from softmin_guidance import SoftminGuidanceModel, SoftminGuidancePrediction


def _small_model() -> SoftminGuidanceModel:
    return SoftminGuidanceModel(
        token_dim=24,
        n_tokens=4,
        k_neighbors=3,
        n_latents=3,
        enc_layers=1,
        dec_layers=1,
        n_heads=4,
        n_freqs=2,
    )


def test_bounded_projection_never_uses_negative_potential_as_reverse_step():
    points = np.zeros((2, 3), dtype=np.float32)
    projected = bounded_project(
        points,
        soft_potential=np.array([-100.0, -1.0], dtype=np.float32),
        step_distance=np.array([3.0, -4.0], dtype=np.float32),
        toward_direction=np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        ),
        scale_mm=2.0,
        max_step_mm=4.0,
        branch_ambiguity=None,
        ambiguity_damping=0.0,
    )

    np.testing.assert_allclose(projected[0], [2.0, 0.0, 0.0])
    np.testing.assert_allclose(projected[1], [0.0, 0.0, 0.0])


def test_bounded_projection_caps_world_displacement_and_can_damp_ambiguity():
    projected = bounded_project(
        np.zeros((2, 3), dtype=np.float32),
        soft_potential=np.array([2.0, 2.0], dtype=np.float32),
        step_distance=np.array([10.0, 1.0], dtype=np.float32),
        toward_direction=np.array(
            [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=np.float32
        ),
        scale_mm=2.0,
        max_step_mm=3.0,
        branch_ambiguity=np.array([0.0, 0.75], dtype=np.float32),
        ambiguity_damping=1.0,
    )

    # First point is capped to 3 mm = 1.5 normalized units.
    np.testing.assert_allclose(projected[0], [1.5, 0.0, 0.0], atol=1.0e-7)
    # Second point uses a unit toward direction and 25% of its predicted step.
    np.testing.assert_allclose(projected[1], [0.0, 0.25, 0.0], atol=1.0e-7)


def test_projection_and_rank_selection_are_deterministic_with_ties():
    kwargs = dict(
        soft_potential=np.array([-2.0, -2.0, 0.5], dtype=np.float32),
        step_distance=np.array([0.2, 0.2, 0.2], dtype=np.float32),
        toward_direction=np.tile(
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (3, 1)
        ),
        scale_mm=1.0,
        max_step_mm=1.0,
        branch_ambiguity=np.zeros(3, dtype=np.float32),
        ambiguity_damping=1.0,
    )
    points = np.zeros((3, 3), dtype=np.float32)
    first = bounded_project(points, **kwargs)
    second = bounded_project(points, **kwargs)
    np.testing.assert_array_equal(first, second)

    grid = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
        dtype=np.float32,
    )
    selected = select_ranked_candidates(
        grid,
        kwargs["soft_potential"],
        input_points_norm=grid,
        scale_mm=1.0,
        max_input_distance_mm=0.01,
        candidate_count=2,
        minimum_required=2,
        branch_ambiguity=np.zeros(3, dtype=np.float32),
        ranking_mode="raw_signed",
    )
    np.testing.assert_array_equal(selected, np.array([0, 1]))


def test_input_cloud_gate_fails_closed_instead_of_using_ungated_points():
    grid = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    with pytest.raises(InsufficientEvidenceError, match="pass input-cloud gate"):
        select_ranked_candidates(
            grid,
            np.array([3.0, -100.0, -200.0], dtype=np.float32),
            input_points_norm=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            scale_mm=1.0,
            max_input_distance_mm=0.1,
            candidate_count=2,
            minimum_required=2,
            branch_ambiguity=np.zeros(3, dtype=np.float32),
        )


def test_candidate_scores_do_not_prefer_large_negative_potential():
    potential = np.array([-5.0, -0.1, 0.2], dtype=np.float32)
    ambiguity = np.array([1.0, 0.1, 0.0], dtype=np.float32)

    raw = candidate_ranking_score(
        potential, scale_mm=1.0, ranking_mode="raw_signed"
    )
    safe = candidate_ranking_score(
        potential,
        scale_mm=1.0,
        ranking_mode="abs_potential_ambiguity",
        branch_ambiguity=ambiguity,
        ambiguity_penalty_mm=1.0,
    )

    assert raw.argmin() == 0
    assert safe.argmin() == 1


def test_spatial_balanced_ranking_covers_separate_cells():
    grid = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    selected = select_ranked_candidates(
        grid,
        np.array([0.0, 0.01, 0.02, 0.5], dtype=np.float32),
        input_points_norm=grid,
        scale_mm=1.0,
        max_input_distance_mm=None,
        candidate_count=2,
        minimum_required=2,
        branch_ambiguity=np.zeros(4, dtype=np.float32),
        ranking_mode="spatial_balanced",
    )

    assert 0 in selected
    assert 3 in selected


def test_grid_evaluation_is_bbox_bounded_and_preserves_guidance_heads():
    class AnalyticModel:
        def decode(self, query_xyz, latent):
            batch_shape = query_xyz.shape[:-1]
            direction = torch.zeros_like(query_xyz)
            direction[..., 0] = 1.0
            return SoftminGuidancePrediction(
                soft_potential=query_xyz.sum(dim=-1),
                step_distance=torch.ones(batch_shape, device=query_xyz.device),
                direction_toward_surface=direction,
                branch_ambiguity=torch.full(
                    batch_shape, 0.25, device=query_xyz.device
                ),
            )

    points = np.array(
        [[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], dtype=np.float32
    )
    grid = evaluate_grid(
        AnalyticModel(),
        torch.zeros(1, 1, 1),
        points,
        grid_res=3,
        device="cpu",
        bbox_pad_fraction=0.0,
        chunk_size=5,
    )

    assert grid.xyz.shape == (27, 3)
    np.testing.assert_array_equal(grid.xyz.min(axis=0), points.min(axis=0))
    np.testing.assert_array_equal(grid.xyz.max(axis=0), points.max(axis=0))
    np.testing.assert_allclose(grid.soft_potential, grid.xyz.sum(axis=1))
    np.testing.assert_array_equal(grid.step_distance, np.ones(27))
    np.testing.assert_array_equal(
        grid.toward_direction, np.tile([1.0, 0.0, 0.0], (27, 1))
    )
    np.testing.assert_array_equal(
        grid.branch_ambiguity, np.full(27, 0.25, dtype=np.float32)
    )


def test_h5_points_and_normals_are_normalized(tmp_path: pathlib.Path):
    path = tmp_path / "input.h5"
    points = np.array(
        [[-2.0, 0.0, 1.0], [2.0, 2.0, 1.0], [0.0, 4.0, 3.0]],
        dtype=np.float32,
    )
    normals = np.array(
        [[0.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("points", data=points)
        h5_file.create_dataset("normals", data=normals)
        h5_file.attrs["thickness_mm"] = 1.2

    cloud = load_input_h5(path)
    expected_center = points.mean(axis=0)
    expected_scale = np.abs(points - expected_center).max() * 1.05 + 1.0e-6
    np.testing.assert_allclose(cloud.center, expected_center)
    assert cloud.scale_mm == pytest.approx(expected_scale)
    np.testing.assert_allclose(
        cloud.points_norm,
        (points - expected_center) / expected_scale,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        np.linalg.norm(cloud.normals, axis=1), np.ones(3), atol=1.0e-7
    )
    assert cloud.thickness_mm == pytest.approx(1.2)


def test_checkpoint_load_restores_architecture_weights_and_eval_mode(
    tmp_path: pathlib.Path,
):
    torch.manual_seed(19)
    expected = _small_model()
    checkpoint = tmp_path / "softmin.pt"
    torch.save(
        {
            "model": expected.state_dict(),
            "metadata": expected.checkpoint_metadata(),
            "epoch": 7,
        },
        checkpoint,
    )

    actual = load_model(checkpoint, "cpu")

    assert isinstance(actual, SoftminGuidanceModel)
    assert actual.training is False
    assert actual._architecture == expected._architecture
    for expected_parameter, actual_parameter in zip(
        expected.parameters(), actual.parameters(), strict=True
    ):
        assert torch.equal(expected_parameter, actual_parameter)


def test_checkpoint_load_rejects_unversioned_metadata(tmp_path: pathlib.Path):
    model = _small_model()
    checkpoint = tmp_path / "unversioned.pt"
    metadata = model.checkpoint_metadata()
    metadata.pop("schema_version")
    torch.save({"model": model.state_dict(), "metadata": metadata}, checkpoint)

    with pytest.raises(ValueError, match="unsupported schema_version"):
        load_model(checkpoint, "cpu")

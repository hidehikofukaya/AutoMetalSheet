from __future__ import annotations

import json
import pathlib
import sys

import h5py
import numpy as np
import pytest
import torch
import trimesh


MODEL_DIR = pathlib.Path(__file__).parents[1] / "src" / "model"
sys.path.insert(0, str(MODEL_DIR))

from analyze_softmin_guidance_field import (
    ANALYSIS_SCHEMA_VERSION,
    AnalysisConfig,
    ambiguity_metrics,
    process,
    projection_transition_metrics,
    rank_candidate_indices,
    regression_metrics,
    reliability_bins,
)
from softmin_guidance import SoftminGuidancePrediction


class AnalyticPlaneModel:
    """Exact normalized-coordinate guidance for the plane z=0."""

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def encode(self, points, normals):
        return torch.zeros(
            (points.shape[0], 1, 1), dtype=points.dtype, device=points.device
        )

    def decode(self, query_xyz, latent):
        z = query_xyz[..., 2]
        direction = torch.zeros_like(query_xyz)
        direction[..., 2] = torch.where(
            z >= 0.0, -torch.ones_like(z), torch.ones_like(z)
        )
        return SoftminGuidancePrediction(
            soft_potential=torch.abs(z),
            step_distance=torch.abs(z),
            direction_toward_surface=direction,
            branch_ambiguity=torch.zeros_like(z),
        )


def test_regression_and_ambiguity_metrics_are_numerically_explicit():
    prediction = np.array([0.0, 2.0, 5.0])
    target = np.array([1.0, 2.0, 3.0])
    metrics = regression_metrics(prediction, target)

    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(5.0 / 3.0))
    assert metrics["bias"] == pytest.approx(1.0 / 3.0)
    assert metrics["absolute_error"]["quantiles"]["q50"] == pytest.approx(1.0)

    ambiguity = ambiguity_metrics(
        np.array([0.1, 0.8, 0.5]), np.array([0.0, 1.0, 0.5]), n_bins=2
    )
    assert ambiguity["mae"] == pytest.approx(0.1)
    assert ambiguity["brier"] == pytest.approx((0.01 + 0.04) / 3.0)
    assert ambiguity["correlation"] is not None
    assert ambiguity["spearman_correlation"] == pytest.approx(1.0)
    assert ambiguity["high_ambiguity_detection"]["auroc"] == pytest.approx(1.0)
    assert (
        ambiguity["high_ambiguity_detection"][
            "false_safe_fraction_of_positive"
        ]
        == 0.0
    )
    assert sum(row["count"] for row in ambiguity["reliability_bins"]) == 3


def test_reliability_bins_include_zero_and_one_without_overlap():
    rows = reliability_bins(
        np.array([0.0, 0.49, 0.5, 1.0]),
        np.array([0.0, 0.4, 0.6, 1.0]),
        n_bins=2,
    )

    assert [row["count"] for row in rows] == [2, 2]
    assert rows[0]["mean_prediction"] == pytest.approx(0.245)
    assert rows[1]["mean_prediction"] == pytest.approx(0.75)


def test_candidate_rankings_are_deterministic_and_semantically_distinct():
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    potential = np.array([-4.0, -1.0, 0.2, 0.5], dtype=np.float32)
    step = np.array([4.0, 1.0, 0.7, 0.1], dtype=np.float32)
    ambiguity = np.array([0.0, 0.9, 0.0, 0.8], dtype=np.float32)
    pool = np.arange(4)

    raw = rank_candidate_indices(
        "raw_signed", xyz, potential, step, ambiguity, pool, 2
    )
    absolute = rank_candidate_indices(
        "abs_potential", xyz, potential, step, ambiguity, pool, 2
    )
    by_step = rank_candidate_indices(
        "step_distance", xyz, potential, step, ambiguity, pool, 2
    )
    penalized = rank_candidate_indices(
        "abs_potential_plus_ambiguity",
        xyz,
        potential,
        step,
        ambiguity,
        pool,
        2,
        ambiguity_penalty_mm=2.0,
    )

    np.testing.assert_array_equal(raw, [0, 1])
    np.testing.assert_array_equal(absolute, [2, 3])
    np.testing.assert_array_equal(by_step, [3, 2])
    np.testing.assert_array_equal(penalized, [2, 3])
    np.testing.assert_array_equal(
        rank_candidate_indices(
            "spatial_balanced",
            xyz,
            potential,
            step,
            ambiguity,
            pool,
            2,
            ambiguity_penalty_mm=2.0,
        ),
        rank_candidate_indices(
            "spatial_balanced",
            xyz,
            potential,
            step,
            ambiguity,
            pool,
            2,
            ambiguity_penalty_mm=2.0,
        ),
    )


def test_projection_transition_reports_improvement_and_worsening_rates():
    before = np.array(
        [[0.0, 0.0, 2.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    after = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=np.float32
    )
    metrics = projection_transition_metrics(
        before,
        after,
        before_gt_distance_mm=np.array([2.0, 1.0]),
        after_gt_distance_mm=np.array([1.0, 2.0]),
    )

    assert metrics["movement_mm"]["mean"] == pytest.approx(1.0)
    assert metrics["improved_fraction"] == pytest.approx(0.5)
    assert metrics["worsened_fraction"] == pytest.approx(0.5)
    assert metrics["unchanged_fraction"] == pytest.approx(0.0)


def _write_synthetic_h5(path: pathlib.Path) -> None:
    xy = np.array(
        [
            [-1.0, -1.0],
            [-1.0, 0.0],
            [-1.0, 1.0],
            [0.0, -1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    points = np.column_stack(
        [xy, np.tile(np.array([-0.1, 0.0, 0.1]), 3)]
    ).astype(np.float32)
    normals = np.tile(
        np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (len(points), 1)
    )
    query_xyz = np.array(
        [
            [-0.5, -0.5, 0.2],
            [0.0, 0.0, -0.3],
            [0.5, 0.5, 0.05],
            [0.25, -0.25, -0.1],
            [-0.25, 0.25, 0.4],
            [0.75, 0.0, -0.2],
        ],
        dtype=np.float32,
    )
    step = np.abs(query_xyz[:, 2])
    direction = np.zeros_like(query_xyz)
    direction[:, 2] = np.where(query_xyz[:, 2] >= 0.0, -1.0, 1.0)
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("points", data=points)
        h5_file.create_dataset("normals", data=normals)
        h5_file.create_dataset("query_xyz", data=query_xyz)
        h5_file.create_dataset("query_soft_potential", data=step)
        h5_file.create_dataset("query_step_distance", data=step)
        h5_file.create_dataset("query_soft_direction", data=direction)
        h5_file.create_dataset(
            "query_branch_ambiguity", data=np.zeros(len(query_xyz), dtype=np.float32)
        )
        h5_file.create_dataset(
            "query_direction_strength", data=np.ones(len(query_xyz), dtype=np.float32)
        )
        h5_file.create_dataset(
            "query_category", data=np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        )
        h5_file.attrs["schema_version"] = "stage_a.softmin_guidance.v1"
        h5_file.attrs["gradient_convention"] = "toward_surface"
        h5_file.attrs["coordinate_unit"] = "mm"
        h5_file.attrs["thickness_mm"] = 1.0


def _write_square_gt_ply(path: pathlib.Path) -> None:
    mesh = trimesh.Trimesh(
        vertices=np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        process=False,
    )
    mesh.export(path)


def test_small_synthetic_analysis_writes_json_markdown_and_npz(
    tmp_path: pathlib.Path,
):
    h5_path = tmp_path / "synthetic_dataset.h5"
    gt_path = tmp_path / "synthetic_gt.ply"
    _write_synthetic_h5(h5_path)
    _write_square_gt_ply(gt_path)
    prefix = tmp_path / "analysis"
    config = AnalysisConfig(
        checkpoint=tmp_path / "unused.pt",
        data_h5=h5_path,
        gt_ply=gt_path,
        output_prefix=prefix,
        seed=17,
        chunk_size=7,
        grid_res=4,
        candidate_count=8,
        projection_iterations=2,
        device="cpu",
        bbox_pad_fraction=0.0,
        max_input_distance_mm=0.8,
        max_step_mm=1.0,
        coverage_sample_count=32,
        reliability_bins=4,
    )

    paths = process(config, model=AnalyticPlaneModel())

    assert all(path.exists() for path in paths.values())
    report = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert report["schema_version"] == ANALYSIS_SCHEMA_VERSION
    assert report["saved_queries"]["query_count"] == 6
    assert report["saved_queries"]["by_category"]["all"]["potential"]["mae"] < 1e-5
    assert report["dense_grid"]["grid_count"] == 64
    assert set(report["candidate_rankings"]["without_ood_gate"]) == {
        "raw_signed",
        "abs_potential",
        "step_distance",
        "abs_potential_plus_ambiguity",
        "spatial_balanced",
    }
    raw_projection = report["candidate_rankings"]["without_ood_gate"][
        "raw_signed"
    ]["projection"]
    assert len(raw_projection) == 3
    assert (
        raw_projection[-1]["point_to_gt_mm"]["mean"]
        <= raw_projection[0]["point_to_gt_mm"]["mean"] + 1e-6
    )
    assert "Potential autograd consistency" in paths["markdown"].read_text(
        encoding="utf-8"
    )
    with np.load(paths["npz"]) as arrays:
        assert arrays["query_xyz_world"].shape == (6, 3)
        assert arrays["grid_xyz_world"].shape == (64, 3)
        assert "without_ood_gate_raw_signed_projection_2_world" in arrays

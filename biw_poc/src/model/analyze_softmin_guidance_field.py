"""Comprehensive, mesh-free analysis of a trained softmin guidance field.

The script consumes a model checkpoint, its Stage-A H5 supervision, and a
ground-truth PLY surface.  It evaluates saved queries, a dense Cartesian grid,
five candidate-ranking policies, bounded projection trajectories, the input
support (OOD) gate, and the autograd gradient of the potential head.

No surface is reconstructed from the selected candidates.  The GT PLY is used
only as a distance oracle and as the source of deterministic coverage samples.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
import pathlib
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata
import torch
import trimesh

try:
    from . import reconstruct_softmin_guidance as RSG
    from .softmin_guidance_dataset import SoftminGuidanceDataset
except ImportError:  # Support direct execution from src/model.
    import reconstruct_softmin_guidance as RSG  # type: ignore[no-redef]
    from softmin_guidance_dataset import (  # type: ignore[no-redef]
        SoftminGuidanceDataset,
    )


ANALYSIS_SCHEMA_VERSION = "analysis.softmin_guidance_field.v1"
CATEGORY_NAMES = {0: "near", 1: "far", 2: "boundary"}
RANKING_METHODS = (
    "raw_signed",
    "abs_potential",
    "step_distance",
    "abs_potential_plus_ambiguity",
    "spatial_balanced",
)
DEFAULT_DISTANCE_BINS_MM = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, math.inf)
DEFAULT_SUPPORT_BINS_MM = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, math.inf)
SUMMARY_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
PROVISIONAL_QUALITY_GATES = {
    "near_potential_mae_mm": 0.25,
    "near_potential_p95_mm": 0.75,
    "near_step_mae_mm": 0.15,
    "near_step_p95_mm": 0.60,
    "near_clear_direction_p95_deg": 45.0,
    "near_clear_direction_over_90_fraction": 0.01,
    "near_ambiguity_mae": 0.10,
    "near_ambiguity_spearman": 0.70,
    "near_high_ambiguity_auroc": 0.90,
    "near_high_ambiguity_false_safe_fraction": 0.05,
    "near_head_consistency_violation_fraction": 0.05,
    "input_cloud_coverage_p95_mm": 5.0,
    "input_cloud_coverage_within_5mm_fraction": 0.95,
    "dense_grid_ghost_fraction": 0.01,
    "projected_point_p95_mm": 0.75,
    "coverage_p95_mm": 5.0,
    "projection_worsened_fraction": 0.02,
}


@dataclass(frozen=True)
class AnalysisConfig:
    checkpoint: pathlib.Path
    data_h5: pathlib.Path
    gt_ply: pathlib.Path
    output_prefix: pathlib.Path
    seed: int = 0
    chunk_size: int = 100_000
    grid_res: int = 48
    candidate_count: int = 4_096
    projection_iterations: int = 3
    device: str = "auto"
    bbox_pad_fraction: float = 0.15
    max_input_distance_mm: float = 10.0
    max_step_mm: float = 5.0
    ambiguity_damping: float = 1.0
    ambiguity_penalty_mm: float = 2.0
    coverage_sample_count: int = 10_000
    reliability_bins: int = 10
    near_threshold_mm: float = 2.0
    ghost_distance_mm: float = 10.0
    missed_prediction_mm: float = 5.0

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        for name in ("chunk_size", "candidate_count", "coverage_sample_count"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.grid_res < 2:
            raise ValueError("grid_res must be at least 2")
        if self.projection_iterations < 0:
            raise ValueError("projection_iterations must be non-negative")
        if self.reliability_bins < 2:
            raise ValueError("reliability_bins must be at least 2")
        for name in (
            "bbox_pad_fraction",
            "max_input_distance_mm",
            "max_step_mm",
            "ambiguity_penalty_mm",
            "near_threshold_mm",
            "ghost_distance_mm",
            "missed_prediction_mm",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 <= self.ambiguity_damping <= 1.0:
            raise ValueError("ambiguity_damping must lie in [0,1]")


@dataclass(frozen=True)
class SavedQueries:
    xyz_norm: np.ndarray
    xyz_world: np.ndarray
    category: np.ndarray
    potential_mm: np.ndarray
    step_mm: np.ndarray
    direction: np.ndarray
    ambiguity: np.ndarray
    direction_strength: np.ndarray


@dataclass
class GroundTruthSurface:
    """Exact triangle-distance or nearest-vertex fallback for a GT PLY."""

    vertices: np.ndarray
    mesh: trimesh.Trimesh | None
    vertex_tree: cKDTree
    distance_mode: str

    def closest(
        self, points_world: np.ndarray, chunk_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        points = _validated_xyz(points_world, "points_world").astype(
            np.float64, copy=False
        )
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        closest = np.empty_like(points)
        distance = np.empty(len(points), dtype=np.float64)
        for start in range(0, len(points), chunk_size):
            stop = min(start + chunk_size, len(points))
            query = points[start:stop]
            if self.mesh is not None:
                closest_chunk, distance_chunk, _ = trimesh.proximity.closest_point(
                    self.mesh, query
                )
            else:
                distance_chunk, index = self.vertex_tree.query(query, k=1)
                closest_chunk = self.vertices[index]
            closest[start:stop] = closest_chunk
            distance[start:stop] = distance_chunk
        if not np.all(np.isfinite(closest)) or not np.all(np.isfinite(distance)):
            raise ValueError("GT distance query returned non-finite values")
        return closest.astype(np.float32), distance.astype(np.float32)

    def sample(self, count: int, seed: int) -> np.ndarray:
        if count <= 0:
            raise ValueError("count must be positive")
        rng = np.random.default_rng(seed)
        if self.mesh is None:
            replace = count > len(self.vertices)
            index = rng.choice(len(self.vertices), size=count, replace=replace)
            return self.vertices[index].astype(np.float32, copy=False)

        triangles = np.asarray(self.mesh.triangles, dtype=np.float64)
        area = np.asarray(self.mesh.area_faces, dtype=np.float64)
        valid = np.isfinite(area) & (area > 1.0e-12)
        if not np.any(valid):
            raise ValueError("GT mesh has no positive-area triangle")
        triangles = triangles[valid]
        probabilities = area[valid] / area[valid].sum()
        face_index = rng.choice(
            len(triangles), size=count, replace=True, p=probabilities
        )
        selected = triangles[face_index]
        u = rng.random(count)
        v = rng.random(count)
        sqrt_u = np.sqrt(u)
        barycentric = np.column_stack(
            [1.0 - sqrt_u, sqrt_u * (1.0 - v), sqrt_u * v]
        )
        samples = np.einsum("ni,nij->nj", barycentric, selected)
        return samples.astype(np.float32)


def _validated_xyz(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
        raise ValueError(f"{name} must have non-empty shape [N,3]")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validated_vector(value: np.ndarray, length: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape [{length}]")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _quantile_key(quantile: float) -> str:
    return f"q{int(round(quantile * 100)):02d}"


def distribution_summary(
    values: np.ndarray, quantiles: Sequence[float] = SUMMARY_QUANTILES
) -> dict[str, Any]:
    """Return a strict-JSON-friendly distribution summary."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "quantiles": {},
        }
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "quantiles": {
            _quantile_key(q): float(np.quantile(array, q)) for q in quantiles
        },
    }


def regression_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    quantiles: Sequence[float] = SUMMARY_QUANTILES,
) -> dict[str, Any]:
    """MAE/RMSE/bias and signed/absolute-error quantiles."""

    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    if pred.shape != truth.shape:
        raise ValueError("prediction and target must have identical shapes")
    valid = np.isfinite(pred) & np.isfinite(truth)
    error = pred[valid] - truth[valid]
    if len(error) == 0:
        return {
            "count": 0,
            "mae": None,
            "rmse": None,
            "bias": None,
            "error": distribution_summary(error, quantiles),
            "absolute_error": distribution_summary(error, quantiles),
        }
    return {
        "count": int(len(error)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
        "error": distribution_summary(error, quantiles),
        "absolute_error": distribution_summary(np.abs(error), quantiles),
    }


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("correlation inputs must have identical shapes")
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 2 or np.std(a) <= 1.0e-12 or np.std(b) <= 1.0e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def direction_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Cosine and angular error for direction vectors."""

    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if (
        pred.ndim != 2
        or pred.shape[1:] != (3,)
        or truth.ndim != 2
        or truth.shape[1:] != (3,)
    ):
        raise ValueError("direction arrays must have shape [N,3]")
    if pred.shape != truth.shape:
        raise ValueError("direction arrays must have identical shapes")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(truth)):
        raise ValueError("direction arrays must be finite")
    pred_norm = np.linalg.norm(pred, axis=1)
    truth_norm = np.linalg.norm(truth, axis=1)
    valid = (pred_norm > 1.0e-8) & (truth_norm > 1.0e-8)
    cosine = np.full(len(pred), np.nan, dtype=np.float64)
    cosine[valid] = np.einsum("ij,ij->i", pred[valid], truth[valid]) / (
        pred_norm[valid] * truth_norm[valid]
    )
    cosine[valid] = np.clip(cosine[valid], -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine[valid]))
    result: dict[str, Any] = {
        "count": int(valid.sum()),
        "cosine": distribution_summary(cosine[valid]),
        "angle_deg": distribution_summary(angle),
        "over_45_deg_fraction": (
            float(np.mean(angle > 45.0)) if len(angle) else None
        ),
        "over_90_deg_fraction": (
            float(np.mean(angle > 90.0)) if len(angle) else None
        ),
    }
    if weights is not None:
        weight = _validated_vector(weights, len(pred), "weights").astype(np.float64)
        weighted_valid = valid & (weight > 0.0)
        denominator = weight[weighted_valid].sum()
        result["weighted_count"] = int(weighted_valid.sum())
        result["weighted_mean_cosine"] = (
            float(np.sum(cosine[weighted_valid] * weight[weighted_valid]) / denominator)
            if denominator > 0.0
            else None
        )
        weighted_angles = np.degrees(np.arccos(cosine[weighted_valid]))
        result["weighted_mean_angle_deg"] = (
            float(np.sum(weighted_angles * weight[weighted_valid]) / denominator)
            if denominator > 0.0
            else None
        )
    return result


def reliability_bins(
    prediction: np.ndarray, target: np.ndarray, n_bins: int = 10
) -> list[dict[str, Any]]:
    """Reliability rows for continuous ambiguity targets in [0,1]."""

    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    if pred.shape != truth.shape:
        raise ValueError("prediction and target must have identical shapes")
    valid = np.isfinite(pred) & np.isfinite(truth)
    pred = np.clip(pred[valid], 0.0, 1.0)
    truth = np.clip(truth[valid], 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.minimum(np.digitize(pred, edges[1:-1], right=False), n_bins - 1)
    rows: list[dict[str, Any]] = []
    for bin_index in range(n_bins):
        mask = index == bin_index
        rows.append(
            {
                "bin": bin_index,
                "lower": float(edges[bin_index]),
                "upper": float(edges[bin_index + 1]),
                "count": int(mask.sum()),
                "mean_prediction": float(pred[mask].mean()) if np.any(mask) else None,
                "mean_target": float(truth[mask].mean()) if np.any(mask) else None,
                "mae": (
                    float(np.mean(np.abs(pred[mask] - truth[mask])))
                    if np.any(mask)
                    else None
                ),
                "brier": (
                    float(np.mean(np.square(pred[mask] - truth[mask])))
                    if np.any(mask)
                    else None
                ),
            }
        )
    return rows


def ambiguity_metrics(
    prediction: np.ndarray, target: np.ndarray, n_bins: int = 10
) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    metrics = regression_metrics(pred, truth)
    bins = reliability_bins(pred, truth, n_bins)
    total = sum(row["count"] for row in bins)
    ece = (
        sum(
            row["count"] * abs(row["mean_prediction"] - row["mean_target"])
            for row in bins
            if row["count"] and row["mean_prediction"] is not None
        )
        / total
        if total
        else None
    )
    positive = truth >= 0.5
    false_safe = positive & (pred < 0.2)
    return {
        "count": metrics["count"],
        "mae": metrics["mae"],
        "brier": (
            float(np.mean(np.square(pred - truth)))
            if len(pred) and np.all(np.isfinite(pred)) and np.all(np.isfinite(truth))
            else None
        ),
        "correlation": safe_correlation(pred, truth),
        "spearman_correlation": safe_correlation(rankdata(pred), rankdata(truth)),
        "ece": ece,
        "high_ambiguity_detection": {
            "target_threshold": 0.5,
            "false_safe_prediction_threshold": 0.2,
            "positive_count": int(positive.sum()),
            "positive_fraction": float(np.mean(positive)) if len(positive) else None,
            "auroc": binary_auroc(pred, positive),
            "false_safe_count": int(false_safe.sum()),
            "false_safe_fraction_of_positive": (
                float(false_safe.sum() / positive.sum())
                if np.any(positive)
                else None
            ),
        },
        "reliability_bins": bins,
    }


def binary_auroc(score: np.ndarray, positive: np.ndarray) -> float | None:
    """Compute AUROC from average ranks, including tied predictions."""

    score = np.asarray(score, dtype=np.float64).reshape(-1)
    positive = np.asarray(positive, dtype=bool).reshape(-1)
    if score.shape != positive.shape or not np.all(np.isfinite(score)):
        raise ValueError("AUROC inputs must be finite and have identical shape")
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    if n_positive == 0 or n_negative == 0:
        return None
    ranks = rankdata(score, method="average")
    positive_rank_sum = float(ranks[positive].sum())
    return (
        positive_rank_sum - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)


def query_metric_group(
    predicted_potential_mm: np.ndarray,
    predicted_step_mm: np.ndarray,
    predicted_direction: np.ndarray,
    predicted_ambiguity: np.ndarray,
    target_potential_mm: np.ndarray,
    target_step_mm: np.ndarray,
    target_direction: np.ndarray,
    target_ambiguity: np.ndarray,
    direction_strength: np.ndarray,
    reliability_bin_count: int,
) -> dict[str, Any]:
    return {
        "count": int(len(predicted_potential_mm)),
        "potential": regression_metrics(
            predicted_potential_mm, target_potential_mm
        ),
        "step": regression_metrics(predicted_step_mm, target_step_mm),
        "direction": direction_metrics(
            predicted_direction, target_direction, direction_strength
        ),
        "ambiguity": ambiguity_metrics(
            predicted_ambiguity, target_ambiguity, reliability_bin_count
        ),
        "head_consistency": {
            "potential_above_step_plus_0_1mm_fraction": (
                float(np.mean(predicted_potential_mm > predicted_step_mm + 0.1))
                if len(predicted_potential_mm)
                else None
            )
        },
    }


def grouped_query_metrics(
    category: np.ndarray,
    predicted_potential_mm: np.ndarray,
    predicted_step_mm: np.ndarray,
    predicted_direction: np.ndarray,
    predicted_ambiguity: np.ndarray,
    target_potential_mm: np.ndarray,
    target_step_mm: np.ndarray,
    target_direction: np.ndarray,
    target_ambiguity: np.ndarray,
    direction_strength: np.ndarray,
    reliability_bin_count: int,
) -> dict[str, Any]:
    category = np.asarray(category, dtype=np.int64).reshape(-1)
    n = len(category)
    for name, value in (
        ("predicted_potential_mm", predicted_potential_mm),
        ("predicted_step_mm", predicted_step_mm),
        ("predicted_ambiguity", predicted_ambiguity),
        ("target_potential_mm", target_potential_mm),
        ("target_step_mm", target_step_mm),
        ("target_ambiguity", target_ambiguity),
        ("direction_strength", direction_strength),
    ):
        _validated_vector(value, n, name)
    if np.asarray(predicted_direction).shape != (n, 3):
        raise ValueError("predicted_direction must have shape [N,3]")
    if np.asarray(target_direction).shape != (n, 3):
        raise ValueError("target_direction must have shape [N,3]")

    masks: dict[str, np.ndarray] = {"all": np.ones(n, dtype=bool)}
    for value in np.unique(category):
        name = CATEGORY_NAMES.get(int(value), f"category_{int(value)}")
        masks[name] = category == value
    masks["near_clear"] = (
        (category == 0)
        & (np.asarray(target_ambiguity) <= 0.2)
        & (np.asarray(direction_strength) >= 0.9)
    )

    result: dict[str, Any] = {}
    for name, mask in masks.items():
        result[name] = query_metric_group(
            np.asarray(predicted_potential_mm)[mask],
            np.asarray(predicted_step_mm)[mask],
            np.asarray(predicted_direction)[mask],
            np.asarray(predicted_ambiguity)[mask],
            np.asarray(target_potential_mm)[mask],
            np.asarray(target_step_mm)[mask],
            np.asarray(target_direction)[mask],
            np.asarray(target_ambiguity)[mask],
            np.asarray(direction_strength)[mask],
            reliability_bin_count,
        )
    return result


def _bin_label(lower: float, upper: float) -> str:
    return f"[{lower:g},{upper:g})" if np.isfinite(upper) else f"[{lower:g},inf)"


def distance_calibration_bins(
    true_distance_mm: np.ndarray,
    abs_potential_mm: np.ndarray,
    step_distance_mm: np.ndarray,
    edges_mm: Sequence[float] = DEFAULT_DISTANCE_BINS_MM,
) -> list[dict[str, Any]]:
    truth = np.asarray(true_distance_mm, dtype=np.float64).reshape(-1)
    potential = np.asarray(abs_potential_mm, dtype=np.float64).reshape(-1)
    step = np.asarray(step_distance_mm, dtype=np.float64).reshape(-1)
    if truth.shape != potential.shape or truth.shape != step.shape:
        raise ValueError("calibration arrays must have identical shapes")
    edges = np.asarray(edges_mm, dtype=np.float64)
    if len(edges) < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("edges_mm must be strictly increasing")
    rows: list[dict[str, Any]] = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (truth >= lower) & (truth < upper)
        rows.append(
            {
                "range_mm": _bin_label(float(lower), float(upper)),
                "lower_mm": float(lower),
                "upper_mm": float(upper) if np.isfinite(upper) else None,
                "count": int(mask.sum()),
                "true_distance_mm": distribution_summary(truth[mask]),
                "abs_potential_mm": distribution_summary(potential[mask]),
                "step_distance_mm": distribution_summary(step[mask]),
                "abs_potential_error": regression_metrics(
                    potential[mask], truth[mask]
                ),
                "step_distance_error": regression_metrics(step[mask], truth[mask]),
            }
        )
    return rows


def ghost_missed_metrics(
    predicted_distance_mm: np.ndarray,
    true_distance_mm: np.ndarray,
    *,
    near_threshold_mm: float,
    ghost_distance_mm: float,
    missed_prediction_mm: float,
) -> dict[str, Any]:
    prediction = np.asarray(predicted_distance_mm, dtype=np.float64).reshape(-1)
    truth = np.asarray(true_distance_mm, dtype=np.float64).reshape(-1)
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth must have identical shapes")
    ghost = (prediction <= near_threshold_mm) & (truth > ghost_distance_mm)
    true_near = truth <= near_threshold_mm
    missed = true_near & (prediction > missed_prediction_mm)
    predicted_near = prediction <= near_threshold_mm
    return {
        "ghost_count": int(ghost.sum()),
        "ghost_fraction_all": float(ghost.mean()) if len(ghost) else None,
        "ghost_fraction_predicted_near": (
            float(ghost.sum() / predicted_near.sum())
            if predicted_near.sum()
            else None
        ),
        "missed_near_count": int(missed.sum()),
        "missed_near_fraction_all": float(missed.mean()) if len(missed) else None,
        "missed_near_fraction_true_near": (
            float(missed.sum() / true_near.sum()) if true_near.sum() else None
        ),
        "thresholds_mm": {
            "predicted_near": float(near_threshold_mm),
            "ghost_true_distance": float(ghost_distance_mm),
            "missed_prediction": float(missed_prediction_mm),
        },
    }


def input_support_bin_metrics(
    input_support_mm: np.ndarray,
    true_distance_mm: np.ndarray,
    abs_potential_mm: np.ndarray,
    step_distance_mm: np.ndarray,
    *,
    edges_mm: Sequence[float] = DEFAULT_SUPPORT_BINS_MM,
    near_threshold_mm: float,
    ghost_distance_mm: float,
    missed_prediction_mm: float,
) -> list[dict[str, Any]]:
    support = np.asarray(input_support_mm, dtype=np.float64).reshape(-1)
    truth = np.asarray(true_distance_mm, dtype=np.float64).reshape(-1)
    potential = np.asarray(abs_potential_mm, dtype=np.float64).reshape(-1)
    step = np.asarray(step_distance_mm, dtype=np.float64).reshape(-1)
    if not (support.shape == truth.shape == potential.shape == step.shape):
        raise ValueError("support-bin arrays must have identical shapes")
    edges = np.asarray(edges_mm, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (support >= lower) & (support < upper)
        rows.append(
            {
                "range_mm": _bin_label(float(lower), float(upper)),
                "lower_mm": float(lower),
                "upper_mm": float(upper) if np.isfinite(upper) else None,
                "count": int(mask.sum()),
                "input_support_mm": distribution_summary(support[mask]),
                "abs_potential_error": regression_metrics(
                    potential[mask], truth[mask]
                ),
                "step_distance_error": regression_metrics(step[mask], truth[mask]),
                "abs_potential_ghost_missed": ghost_missed_metrics(
                    potential[mask],
                    truth[mask],
                    near_threshold_mm=near_threshold_mm,
                    ghost_distance_mm=ghost_distance_mm,
                    missed_prediction_mm=missed_prediction_mm,
                ),
                "step_ghost_missed": ghost_missed_metrics(
                    step[mask],
                    truth[mask],
                    near_threshold_mm=near_threshold_mm,
                    ghost_distance_mm=ghost_distance_mm,
                    missed_prediction_mm=missed_prediction_mm,
                ),
            }
        )
    return rows


def input_support_distance_mm(
    xyz_norm: np.ndarray, input_points_norm: np.ndarray, scale_mm: float
) -> np.ndarray:
    xyz = _validated_xyz(xyz_norm, "xyz_norm")
    inputs = _validated_xyz(input_points_norm, "input_points_norm")
    if not np.isfinite(scale_mm) or scale_mm <= 0.0:
        raise ValueError("scale_mm must be finite and positive")
    distance, _ = cKDTree(inputs).query(xyz, k=1)
    return (distance * scale_mm).astype(np.float32)


def _stable_score_order(score: np.ndarray, pool_indices: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    pool = np.asarray(pool_indices, dtype=np.int64)
    return pool[np.argsort(score[pool], kind="stable")]


def spatially_balanced_indices(
    xyz: np.ndarray,
    base_score: np.ndarray,
    pool_indices: np.ndarray,
    candidate_count: int,
) -> np.ndarray:
    """Round-robin score-ranked spatial cells for deterministic coverage."""

    points = _validated_xyz(xyz, "xyz")
    score = _validated_vector(base_score, len(points), "base_score")
    pool = np.asarray(pool_indices, dtype=np.int64).reshape(-1)
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if len(pool) == 0:
        return np.zeros(0, dtype=np.int64)
    if np.any(pool < 0) or np.any(pool >= len(points)):
        raise ValueError("pool_indices contains an out-of-range index")

    ranked = _stable_score_order(score, pool)
    pool_points = points[ranked].astype(np.float64)
    lower = pool_points.min(axis=0)
    span = pool_points.max(axis=0) - lower
    cells_per_axis = max(1, int(round(candidate_count ** (1.0 / 3.0))))
    normalized = np.divide(
        pool_points - lower,
        span,
        out=np.zeros_like(pool_points),
        where=span > 1.0e-12,
    )
    cell_xyz = np.minimum(
        (normalized * cells_per_axis).astype(np.int64), cells_per_axis - 1
    )
    cell_id = (
        cell_xyz[:, 0] * cells_per_axis * cells_per_axis
        + cell_xyz[:, 1] * cells_per_axis
        + cell_xyz[:, 2]
    )

    buckets: dict[int, list[int]] = {}
    for index, cell in zip(ranked, cell_id, strict=True):
        buckets.setdefault(int(cell), []).append(int(index))
    ordered_cells = sorted(
        buckets,
        key=lambda cell: (score[buckets[cell][0]], cell),
    )
    selected: list[int] = []
    depth = 0
    target = min(candidate_count, len(pool))
    while len(selected) < target:
        added = False
        for cell in ordered_cells:
            bucket = buckets[cell]
            if depth < len(bucket):
                selected.append(bucket[depth])
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
        depth += 1
    return np.asarray(selected, dtype=np.int64)


def rank_candidate_indices(
    method: str,
    xyz: np.ndarray,
    soft_potential_mm: np.ndarray,
    step_distance_mm: np.ndarray,
    branch_ambiguity: np.ndarray,
    pool_indices: np.ndarray,
    candidate_count: int,
    *,
    ambiguity_penalty_mm: float = 2.0,
) -> np.ndarray:
    """Select deterministic candidates using one named ranking policy."""

    if method not in RANKING_METHODS:
        raise ValueError(f"unknown ranking method: {method}")
    points = _validated_xyz(xyz, "xyz")
    n = len(points)
    potential = _validated_vector(
        soft_potential_mm, n, "soft_potential_mm"
    ).astype(np.float64)
    step = _validated_vector(step_distance_mm, n, "step_distance_mm").astype(
        np.float64
    )
    ambiguity = _validated_vector(branch_ambiguity, n, "branch_ambiguity").astype(
        np.float64
    )
    pool = np.asarray(pool_indices, dtype=np.int64).reshape(-1)
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if ambiguity_penalty_mm < 0.0 or not np.isfinite(ambiguity_penalty_mm):
        raise ValueError("ambiguity_penalty_mm must be finite and non-negative")
    if len(pool) == 0:
        return np.zeros(0, dtype=np.int64)

    combined = np.abs(potential) + ambiguity_penalty_mm * np.clip(
        ambiguity, 0.0, 1.0
    )
    if method == "raw_signed":
        score = potential
    elif method == "abs_potential":
        score = np.abs(potential)
    elif method == "step_distance":
        score = step
    else:
        score = combined

    if method == "spatial_balanced":
        return spatially_balanced_indices(
            points, score, pool, candidate_count
        )
    ranked = _stable_score_order(score, pool)
    return ranked[: min(candidate_count, len(ranked))]


def point_set_accuracy(
    points_world: np.ndarray,
    gt_distance_mm: np.ndarray,
    gt_surface_samples_world: np.ndarray,
) -> dict[str, Any]:
    points = _validated_xyz(points_world, "points_world")
    distance = _validated_vector(
        gt_distance_mm, len(points), "gt_distance_mm"
    )
    samples = _validated_xyz(
        gt_surface_samples_world, "gt_surface_samples_world"
    )
    coverage_distance, _ = cKDTree(points).query(samples, k=1)
    return {
        "point_to_gt_mm": distribution_summary(distance),
        "gt_sample_to_points_coverage_mm": distribution_summary(coverage_distance),
        "coverage_fraction_within_mm": {
            str(threshold): float(np.mean(coverage_distance <= threshold))
            for threshold in (0.5, 1.0, 2.0, 5.0, 10.0)
        },
    }


def projection_transition_metrics(
    before_world: np.ndarray,
    after_world: np.ndarray,
    before_gt_distance_mm: np.ndarray,
    after_gt_distance_mm: np.ndarray,
) -> dict[str, Any]:
    before = _validated_xyz(before_world, "before_world")
    after = _validated_xyz(after_world, "after_world")
    if before.shape != after.shape:
        raise ValueError("before_world and after_world must have identical shapes")
    before_distance = _validated_vector(
        before_gt_distance_mm, len(before), "before_gt_distance_mm"
    ).astype(np.float64)
    after_distance = _validated_vector(
        after_gt_distance_mm, len(after), "after_gt_distance_mm"
    ).astype(np.float64)
    movement = np.linalg.norm(after - before, axis=1)
    improvement = before_distance - after_distance
    epsilon = 1.0e-6
    return {
        "movement_mm": distribution_summary(movement),
        "distance_improvement_mm": distribution_summary(improvement),
        "improved_fraction": float(np.mean(improvement > epsilon)),
        "unchanged_fraction": float(np.mean(np.abs(improvement) <= epsilon)),
        "worsened_fraction": float(np.mean(improvement < -epsilon)),
        "mean_relative_improvement": float(
            np.mean(improvement / np.maximum(before_distance, 1.0e-6))
        ),
    }


def load_saved_queries(path: pathlib.Path, cloud: RSG.InputCloud) -> SavedQueries:
    """Load every stored Stage-A query through the validated dataset contract."""

    item = SoftminGuidanceDataset(
        path, n_query_sample=None, n_points_sample=None, seed=0
    )[0]

    def numpy(name: str) -> np.ndarray:
        value = item[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        return value.detach().cpu().numpy()

    item_query_norm = numpy("query_xyz")
    item_center = numpy("center")
    item_scale = float(numpy("scale"))
    xyz_world = item_query_norm * item_scale + item_center
    xyz_norm = (xyz_world - cloud.center) / cloud.scale_mm
    category = (
        numpy("query_category").astype(np.int64)
        if "query_category" in item
        else np.full(len(xyz_norm), -1, dtype=np.int64)
    )
    return SavedQueries(
        xyz_norm=xyz_norm.astype(np.float32),
        xyz_world=xyz_world.astype(np.float32),
        category=category,
        potential_mm=numpy("query_soft_potential").astype(np.float32),
        step_mm=numpy("query_step_distance").astype(np.float32),
        direction=numpy("query_soft_direction").astype(np.float32),
        ambiguity=numpy("query_branch_ambiguity").astype(np.float32),
        direction_strength=numpy("query_direction_strength").astype(np.float32),
    )


def load_gt_surface(path: pathlib.Path) -> GroundTruthSurface:
    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.geometry.values())
        triangle_meshes = [
            geometry
            for geometry in geometries
            if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces)
        ]
        if triangle_meshes:
            loaded = trimesh.util.concatenate(triangle_meshes)
        else:
            vertices = [
                np.asarray(geometry.vertices)
                for geometry in geometries
                if hasattr(geometry, "vertices") and len(geometry.vertices)
            ]
            if not vertices:
                raise ValueError(f"{path}: GT PLY has no vertices")
            loaded = trimesh.points.PointCloud(np.concatenate(vertices, axis=0))

    if not hasattr(loaded, "vertices") or len(loaded.vertices) == 0:
        raise ValueError(f"{path}: GT PLY has no vertices")
    vertices = _validated_xyz(np.asarray(loaded.vertices), "GT vertices").astype(
        np.float64
    )
    mesh = (
        loaded
        if isinstance(loaded, trimesh.Trimesh) and len(loaded.faces) > 0
        else None
    )
    return GroundTruthSurface(
        vertices=vertices,
        mesh=mesh,
        vertex_tree=cKDTree(vertices),
        distance_mode="exact_triangle" if mesh is not None else "nearest_vertex",
    )


def decode_potential_gradients_chunked(
    model: Any,
    latent: torch.Tensor,
    xyz_norm: np.ndarray,
    device: str | torch.device,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Autograd d(potential)/d(normalized xyz) and decoded toward direction."""

    xyz = _validated_xyz(xyz_norm, "xyz_norm").astype(np.float32, copy=False)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    gradients = np.empty_like(xyz)
    directions = np.empty_like(xyz)
    with torch.enable_grad():
        for start in range(0, len(xyz), chunk_size):
            stop = min(start + chunk_size, len(xyz))
            query = (
                torch.from_numpy(xyz[start:stop])
                .unsqueeze(0)
                .to(device)
                .requires_grad_(True)
            )
            prediction = model.decode(query, latent)
            gradient = torch.autograd.grad(
                prediction.soft_potential.sum(),
                query,
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )[0]
            gradients[start:stop] = gradient[0].detach().cpu().numpy()
            directions[start:stop] = (
                prediction.direction_toward_surface[0].detach().cpu().numpy()
            )
    if not np.all(np.isfinite(gradients)) or not np.all(np.isfinite(directions)):
        raise ValueError("autograd gradient analysis produced non-finite values")
    return gradients, directions


def gradient_consistency_metrics(
    potential_gradient: np.ndarray,
    predicted_direction: np.ndarray,
    target_direction: np.ndarray | None = None,
) -> dict[str, Any]:
    gradient = _validated_xyz(
        potential_gradient, "potential_gradient"
    ).astype(np.float64)
    toward_from_gradient = -gradient
    result = {
        "gradient_norm": distribution_summary(np.linalg.norm(gradient, axis=1)),
        "negative_gradient_vs_predicted_direction": direction_metrics(
            toward_from_gradient, predicted_direction
        ),
    }
    if target_direction is not None:
        result["negative_gradient_vs_gt_direction"] = direction_metrics(
            toward_from_gradient, target_direction
        )
    return result


def grouped_gradient_metrics(
    category: np.ndarray,
    potential_gradient: np.ndarray,
    predicted_direction: np.ndarray,
    target_direction: np.ndarray,
) -> dict[str, Any]:
    category = np.asarray(category, dtype=np.int64)
    masks: dict[str, np.ndarray] = {"all": np.ones(len(category), dtype=bool)}
    for value in np.unique(category):
        masks[CATEGORY_NAMES.get(int(value), f"category_{int(value)}")] = (
            category == value
        )
    return {
        name: gradient_consistency_metrics(
            potential_gradient[mask],
            predicted_direction[mask],
            target_direction[mask],
        )
        for name, mask in masks.items()
        if np.any(mask)
    }


def _resolve_device(option: str) -> str:
    if option == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return option


def _set_deterministic_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inspect_checkpoint_evidence(path: pathlib.Path) -> dict[str, Any]:
    """Return the validation provenance needed to interpret model certainty."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        return {"status": "invalid_payload"}
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return {"status": "metadata_missing"}
    validation = metadata.get("validation_split")
    selection = metadata.get("best_checkpoint_selection")
    return {
        "status": "ok",
        "epoch": payload.get("epoch"),
        "validation_split": (
            dict(validation) if isinstance(validation, Mapping) else None
        ),
        "best_checkpoint_selection": (
            dict(selection) if isinstance(selection, Mapping) else None
        ),
    }


def provisional_quality_assessment(
    input_cloud_report: Mapping[str, Any],
    saved_query_report: Mapping[str, Any],
    dense_grid_report: Mapping[str, Any],
    ranking_reports: Mapping[str, Any],
    checkpoint_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply explicit PoC gates without claiming production certification."""

    near = saved_query_report["by_category"]["near"]
    near_clear = saved_query_report["by_category"]["near_clear"]
    ambiguity_detection = near["ambiguity"]["high_ambiguity_detection"]
    final_projection = ranking_reports["with_ood_gate"]["spatial_balanced"]
    if final_projection["status"] == "ok":
        final = final_projection["projection"][-1]
        transition = final.get("from_initial")
    else:
        final = None
        transition = None

    observed = {
        "near_potential_mae_mm": near["potential"]["mae"],
        "near_potential_p95_mm": near["potential"]["absolute_error"][
            "quantiles"
        ]["q95"],
        "near_step_mae_mm": near["step"]["mae"],
        "near_step_p95_mm": near["step"]["absolute_error"]["quantiles"]["q95"],
        "near_clear_direction_p95_deg": near_clear["direction"]["angle_deg"][
            "quantiles"
        ].get("q95"),
        "near_clear_direction_over_90_fraction": near_clear["direction"][
            "over_90_deg_fraction"
        ],
        "near_ambiguity_mae": near["ambiguity"]["mae"],
        "near_ambiguity_spearman": near["ambiguity"]["spearman_correlation"],
        "near_high_ambiguity_auroc": ambiguity_detection["auroc"],
        "near_high_ambiguity_false_safe_fraction": ambiguity_detection[
            "false_safe_fraction_of_positive"
        ],
        "near_head_consistency_violation_fraction": near["head_consistency"][
            "potential_above_step_plus_0_1mm_fraction"
        ],
        "input_cloud_coverage_p95_mm": input_cloud_report[
            "gt_sample_to_points_coverage_mm"
        ]["quantiles"]["q95"],
        "input_cloud_coverage_within_5mm_fraction": input_cloud_report[
            "coverage_fraction_within_mm"
        ]["5.0"],
        "dense_grid_ghost_fraction": dense_grid_report["ghost_missed_near"][
            "abs_potential"
        ]["ghost_fraction_all"],
        "projected_point_p95_mm": (
            final["point_to_gt_mm"]["quantiles"]["q95"] if final else None
        ),
        "coverage_p95_mm": (
            final["gt_sample_to_points_coverage_mm"]["quantiles"]["q95"]
            if final
            else None
        ),
        "projection_worsened_fraction": (
            transition["worsened_fraction"] if transition else None
        ),
    }
    minimum_is_better = {
        "near_ambiguity_spearman",
        "near_high_ambiguity_auroc",
        "input_cloud_coverage_within_5mm_fraction",
    }
    checks: dict[str, Any] = {}
    for name, threshold in PROVISIONAL_QUALITY_GATES.items():
        value = observed[name]
        passed = (
            value is not None
            and (
                float(value) >= threshold
                if name in minimum_is_better
                else float(value) <= threshold
            )
        )
        checks[name] = {
            "value": value,
            "threshold": threshold,
            "comparison": ">=" if name in minimum_is_better else "<=",
            "passed": bool(passed),
        }

    validation = checkpoint_evidence.get("validation_split")
    has_part_holdout = bool(
        isinstance(validation, Mapping)
        and validation.get("strategy") == "part_holdout"
        and validation.get("has_validation")
    )
    passed_count = sum(bool(item["passed"]) for item in checks.values())
    return {
        "verdict": (
            "conditional_pass"
            if passed_count == len(checks) and has_part_holdout
            else "not_ready"
        ),
        "passed_gate_count": passed_count,
        "total_gate_count": len(checks),
        "checks": checks,
        "evidence_scope": (
            "part_holdout"
            if has_part_holdout
            else "single_part_or_non_holdout; generalization is unproven"
        ),
        "notes": [
            "These thresholds are provisional PoC gates, not a production guarantee.",
            "branch_ambiguity measures geometric branch competition; input support "
            "distance is reported separately as an OOD proxy.",
            "Formal confidence requires unseen-part evaluation across multiple "
            "families and seeds.",
        ],
    }
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _projection_history(
    model: Any,
    latent: torch.Tensor,
    initial_xyz_norm: np.ndarray,
    cloud: RSG.InputCloud,
    gt_surface: GroundTruthSurface,
    gt_samples_world: np.ndarray,
    device: str,
    config: AnalysisConfig,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    current = np.asarray(initial_xyz_norm, dtype=np.float32).copy()
    trajectories = [current.copy()]
    current_world = current * cloud.scale_mm + cloud.center
    _, current_distance = gt_surface.closest(current_world, config.chunk_size)
    history: list[dict[str, Any]] = [
        {
            "iteration": 0,
            **point_set_accuracy(current_world, current_distance, gt_samples_world),
            "transition": None,
        }
    ]
    initial_world = current_world.copy()
    initial_distance = current_distance.copy()

    for iteration in range(1, config.projection_iterations + 1):
        potential, step, direction, ambiguity = RSG.decode_guidance_chunked(
            model,
            latent,
            current,
            device,
            config.chunk_size,
        )
        next_xyz = RSG.bounded_project(
            current,
            soft_potential=potential,
            step_distance=step,
            toward_direction=direction,
            scale_mm=cloud.scale_mm,
            max_step_mm=config.max_step_mm,
            branch_ambiguity=ambiguity,
            ambiguity_damping=config.ambiguity_damping,
        )
        next_world = next_xyz * cloud.scale_mm + cloud.center
        _, next_distance = gt_surface.closest(next_world, config.chunk_size)
        row = {
            "iteration": iteration,
            **point_set_accuracy(next_world, next_distance, gt_samples_world),
            "transition": projection_transition_metrics(
                current_world, next_world, current_distance, next_distance
            ),
            "from_initial": projection_transition_metrics(
                initial_world, next_world, initial_distance, next_distance
            ),
        }
        history.append(row)
        trajectories.append(next_xyz.copy())
        current = next_xyz
        current_world = next_world
        current_distance = next_distance
    return history, trajectories


def _ranking_report(
    method: str,
    selected_indices: np.ndarray,
    grid: RSG.GuidanceGrid,
    grid_true_distance_mm: np.ndarray,
    grid_support_mm: np.ndarray,
    cloud: RSG.InputCloud,
    gt_surface: GroundTruthSurface,
    gt_samples_world: np.ndarray,
    model: Any,
    latent: torch.Tensor,
    device: str,
    config: AnalysisConfig,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    selected = np.asarray(selected_indices, dtype=np.int64)
    if len(selected) == 0:
        return {
            "method": method,
            "status": "insufficient_evidence",
            "selected_count": 0,
        }, []
    history, trajectories = _projection_history(
        model,
        latent,
        grid.xyz[selected],
        cloud,
        gt_surface,
        gt_samples_world,
        device,
        config,
    )
    return {
        "method": method,
        "status": "ok",
        "selected_count": int(len(selected)),
        "selected_index_summary": distribution_summary(selected),
        "selected_grid_fields": {
            "soft_potential_mm": distribution_summary(
                grid.soft_potential[selected] * cloud.scale_mm
            ),
            "abs_potential_mm": distribution_summary(
                np.abs(grid.soft_potential[selected]) * cloud.scale_mm
            ),
            "step_distance_mm": distribution_summary(
                grid.step_distance[selected] * cloud.scale_mm
            ),
            "branch_ambiguity": distribution_summary(
                grid.branch_ambiguity[selected]
            ),
            "true_distance_mm": distribution_summary(
                grid_true_distance_mm[selected]
            ),
            "input_support_mm": distribution_summary(grid_support_mm[selected]),
        },
        "projection": history,
    }, trajectories


def compare_ood_gate_reports(
    without_gate: Mapping[str, Any], with_gate: Mapping[str, Any]
) -> dict[str, Any]:
    if without_gate.get("status") != "ok" or with_gate.get("status") != "ok":
        return {
            "status": "unavailable",
            "without_gate_status": without_gate.get("status"),
            "with_gate_status": with_gate.get("status"),
        }

    def value(report: Mapping[str, Any], iteration: int, section: str) -> float | None:
        projection = report["projection"][iteration]
        return projection[section]["mean"]

    last_without = len(without_gate["projection"]) - 1
    last_with = len(with_gate["projection"]) - 1
    final_p2gt_without = value(
        without_gate, last_without, "point_to_gt_mm"
    )
    final_p2gt_with = value(with_gate, last_with, "point_to_gt_mm")
    final_coverage_without = value(
        without_gate, last_without, "gt_sample_to_points_coverage_mm"
    )
    final_coverage_with = value(
        with_gate, last_with, "gt_sample_to_points_coverage_mm"
    )
    return {
        "status": "ok",
        "selected_count_without_gate": int(without_gate["selected_count"]),
        "selected_count_with_gate": int(with_gate["selected_count"]),
        "final_point_to_gt_mean_mm": {
            "without_gate": final_p2gt_without,
            "with_gate": final_p2gt_with,
            "with_minus_without": (
                final_p2gt_with - final_p2gt_without
                if final_p2gt_with is not None and final_p2gt_without is not None
                else None
            ),
        },
        "final_coverage_mean_mm": {
            "without_gate": final_coverage_without,
            "with_gate": final_coverage_with,
            "with_minus_without": (
                final_coverage_with - final_coverage_without
                if final_coverage_with is not None
                and final_coverage_without is not None
                else None
            ),
        },
    }


def analyze(
    config: AnalysisConfig, *, model: Any | None = None
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run the complete analysis and return JSON report plus NPZ arrays."""

    config.validate()
    _set_deterministic_seed(config.seed)
    device = _resolve_device(config.device)
    checkpoint_evidence = (
        inspect_checkpoint_evidence(config.checkpoint)
        if model is None
        else {"status": "external_model_not_inspected"}
    )
    if model is None:
        model = RSG.load_model(config.checkpoint, device)
    elif hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    cloud = RSG.load_input_h5(config.data_h5)
    saved = load_saved_queries(config.data_h5, cloud)
    gt_surface = load_gt_surface(config.gt_ply)
    gt_samples = gt_surface.sample(config.coverage_sample_count, config.seed)
    input_cloud_world = cloud.points_norm * cloud.scale_mm + cloud.center
    _, input_cloud_gt_distance = gt_surface.closest(
        input_cloud_world, config.chunk_size
    )
    input_cloud_report = point_set_accuracy(
        input_cloud_world,
        input_cloud_gt_distance,
        gt_samples,
    )
    latent = RSG.encode_input(model, cloud, device)

    (
        query_pred_potential_norm,
        query_pred_step_norm,
        query_pred_direction,
        query_pred_ambiguity,
    ) = RSG.decode_guidance_chunked(
        model, latent, saved.xyz_norm, device, config.chunk_size
    )
    query_pred_potential_mm = query_pred_potential_norm * cloud.scale_mm
    query_pred_step_mm = query_pred_step_norm * cloud.scale_mm
    query_gradient, query_gradient_direction = (
        decode_potential_gradients_chunked(
            model, latent, saved.xyz_norm, device, config.chunk_size
        )
    )

    saved_query_report = {
        "query_count": int(len(saved.xyz_norm)),
        "category_mapping": {
            str(key): value for key, value in CATEGORY_NAMES.items()
        },
        "by_category": grouped_query_metrics(
            saved.category,
            query_pred_potential_mm,
            query_pred_step_mm,
            query_pred_direction,
            query_pred_ambiguity,
            saved.potential_mm,
            saved.step_mm,
            saved.direction,
            saved.ambiguity,
            saved.direction_strength,
            config.reliability_bins,
        ),
        "potential_autograd": grouped_gradient_metrics(
            saved.category,
            query_gradient,
            query_gradient_direction,
            saved.direction,
        ),
    }

    grid = RSG.evaluate_grid(
        model,
        latent,
        cloud.points_norm,
        config.grid_res,
        device,
        bbox_pad_fraction=config.bbox_pad_fraction,
        chunk_size=config.chunk_size,
    )
    grid_world = grid.xyz * cloud.scale_mm + cloud.center
    _, grid_true_distance = gt_surface.closest(grid_world, config.chunk_size)
    grid_support = input_support_distance_mm(
        grid.xyz, cloud.points_norm, cloud.scale_mm
    )
    grid_potential_mm = grid.soft_potential * cloud.scale_mm
    grid_abs_potential_mm = np.abs(grid_potential_mm)
    grid_step_mm = grid.step_distance * cloud.scale_mm

    dense_grid_report = {
        "grid_res": int(config.grid_res),
        "grid_count": int(len(grid.xyz)),
        "gt_distance_mode": gt_surface.distance_mode,
        "distributions": {
            "true_distance_mm": distribution_summary(grid_true_distance),
            "soft_potential_mm": distribution_summary(grid_potential_mm),
            "abs_potential_mm": distribution_summary(grid_abs_potential_mm),
            "step_distance_mm": distribution_summary(grid_step_mm),
            "branch_ambiguity": distribution_summary(grid.branch_ambiguity),
            "input_support_mm": distribution_summary(grid_support),
        },
        "distance_accuracy": {
            "raw_signed_potential": regression_metrics(
                grid_potential_mm, grid_true_distance
            ),
            "abs_potential": regression_metrics(
                grid_abs_potential_mm, grid_true_distance
            ),
            "step_distance": regression_metrics(grid_step_mm, grid_true_distance),
            "correlation_abs_potential_vs_true": safe_correlation(
                grid_abs_potential_mm, grid_true_distance
            ),
            "correlation_step_vs_true": safe_correlation(
                grid_step_mm, grid_true_distance
            ),
        },
        "calibration_by_true_distance": distance_calibration_bins(
            grid_true_distance, grid_abs_potential_mm, grid_step_mm
        ),
        "ghost_missed_near": {
            "abs_potential": ghost_missed_metrics(
                grid_abs_potential_mm,
                grid_true_distance,
                near_threshold_mm=config.near_threshold_mm,
                ghost_distance_mm=config.ghost_distance_mm,
                missed_prediction_mm=config.missed_prediction_mm,
            ),
            "step_distance": ghost_missed_metrics(
                grid_step_mm,
                grid_true_distance,
                near_threshold_mm=config.near_threshold_mm,
                ghost_distance_mm=config.ghost_distance_mm,
                missed_prediction_mm=config.missed_prediction_mm,
            ),
        },
        "input_support_bins": input_support_bin_metrics(
            grid_support,
            grid_true_distance,
            grid_abs_potential_mm,
            grid_step_mm,
            near_threshold_mm=config.near_threshold_mm,
            ghost_distance_mm=config.ghost_distance_mm,
            missed_prediction_mm=config.missed_prediction_mm,
        ),
    }

    ranking_reports: dict[str, dict[str, Any]] = {
        "without_ood_gate": {},
        "with_ood_gate": {},
    }
    arrays: dict[str, np.ndarray] = {
        "query_xyz_world": saved.xyz_world,
        "query_category": saved.category,
        "query_gt_potential_mm": saved.potential_mm,
        "query_gt_step_mm": saved.step_mm,
        "query_gt_direction": saved.direction,
        "query_gt_ambiguity": saved.ambiguity,
        "query_pred_potential_mm": query_pred_potential_mm,
        "query_pred_step_mm": query_pred_step_mm,
        "query_pred_direction": query_pred_direction,
        "query_pred_ambiguity": query_pred_ambiguity,
        "query_potential_autograd_gradient": query_gradient,
        "input_cloud_world": input_cloud_world,
        "input_cloud_gt_distance_mm": input_cloud_gt_distance,
        "grid_xyz_world": grid_world,
        "grid_true_distance_mm": grid_true_distance,
        "grid_input_support_mm": grid_support,
        "grid_soft_potential_mm": grid_potential_mm,
        "grid_step_distance_mm": grid_step_mm,
        "grid_toward_direction": grid.toward_direction,
        "grid_branch_ambiguity": grid.branch_ambiguity,
        "gt_surface_samples_world": gt_samples,
    }

    helper_ranking_modes = {
        "raw_signed": "raw_signed",
        "abs_potential": "abs_potential",
        "step_distance": "step_distance",
        "abs_potential_plus_ambiguity": "abs_potential_ambiguity",
        "spatial_balanced": "spatial_balanced",
    }
    for gate_name, gate_threshold in (
        ("without_ood_gate", None),
        ("with_ood_gate", config.max_input_distance_mm),
    ):
        for method in RANKING_METHODS:
            try:
                # Reuse the production selector for every comparison; the
                # report-facing ambiguity method name is intentionally more
                # explicit than the helper's historical mode name.
                selected = RSG.select_ranked_candidates(
                    grid.xyz,
                    grid.soft_potential,
                    input_points_norm=cloud.points_norm,
                    scale_mm=cloud.scale_mm,
                    max_input_distance_mm=gate_threshold,
                    candidate_count=config.candidate_count,
                    minimum_required=1,
                    step_distance=grid.step_distance,
                    branch_ambiguity=grid.branch_ambiguity,
                    ranking_mode=helper_ranking_modes[method],
                    ambiguity_penalty_mm=config.ambiguity_penalty_mm,
                )
            except RSG.InsufficientEvidenceError:
                selected = np.zeros(0, dtype=np.int64)
            report, trajectories = _ranking_report(
                method,
                selected,
                grid,
                grid_true_distance,
                grid_support,
                cloud,
                gt_surface,
                gt_samples,
                model,
                latent,
                device,
                config,
            )
            ranking_reports[gate_name][method] = report
            arrays[f"{gate_name}_{method}_indices"] = selected
            for iteration, trajectory in enumerate(trajectories):
                arrays[
                    f"{gate_name}_{method}_projection_{iteration}_world"
                ] = trajectory * cloud.scale_mm + cloud.center

    ood_comparison = {
        method: compare_ood_gate_reports(
            ranking_reports["without_ood_gate"][method],
            ranking_reports["with_ood_gate"][method],
        )
        for method in RANKING_METHODS
    }
    report = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "inputs": {
            "checkpoint": str(config.checkpoint),
            "data_h5": str(config.data_h5),
            "gt_ply": str(config.gt_ply),
        },
        "configuration": _config_to_json(config),
        "normalization": {
            "center_world_mm": cloud.center.tolist(),
            "scale_mm": float(cloud.scale_mm),
            "thickness_mm": float(cloud.thickness_mm),
        },
        "input_cloud_reference": input_cloud_report,
        "saved_queries": saved_query_report,
        "dense_grid": dense_grid_report,
        "candidate_rankings": ranking_reports,
        "ood_gate_comparison": ood_comparison,
        "checkpoint_evidence": checkpoint_evidence,
    }
    report["provisional_quality_assessment"] = provisional_quality_assessment(
        input_cloud_report,
        saved_query_report,
        dense_grid_report,
        ranking_reports,
        checkpoint_evidence,
    )
    return _json_safe(report), arrays


def _config_to_json(config: AnalysisConfig) -> dict[str, Any]:
    result = asdict(config)
    for key in ("checkpoint", "data_h5", "gt_ply", "output_prefix"):
        result[key] = str(result[key])
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable companion to the complete JSON."""

    lines = [
        "# Softmin guidance field analysis",
        "",
        f"- Schema: `{report['schema_version']}`",
        f"- H5: `{report['inputs']['data_h5']}`",
        f"- GT PLY: `{report['inputs']['gt_ply']}`",
        f"- Grid: {report['dense_grid']['grid_res']}³ "
        f"({report['dense_grid']['grid_count']} points)",
        f"- Provisional verdict: "
        f"`{report['provisional_quality_assessment']['verdict']}` "
        f"({report['provisional_quality_assessment']['passed_gate_count']}/"
        f"{report['provisional_quality_assessment']['total_gate_count']} gates)",
        f"- Evidence scope: "
        f"{report['provisional_quality_assessment']['evidence_scope']}",
        f"- Input-cloud GT coverage mean/p95: "
        f"{_format_number(report['input_cloud_reference']['gt_sample_to_points_coverage_mm']['mean'])} / "
        f"{_format_number(report['input_cloud_reference']['gt_sample_to_points_coverage_mm']['quantiles']['q95'])} mm",
        "",
        "## Saved query accuracy",
        "",
        "| Category | N | Potential MAE/RMSE/bias (mm) | Step MAE/RMSE/bias (mm) | "
        "Direction angle mean (deg) | Ambiguity MAE/Brier/Spearman |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in report["saved_queries"]["by_category"].items():
        potential = metrics["potential"]
        step = metrics["step"]
        direction = metrics["direction"]["angle_deg"]
        ambiguity = metrics["ambiguity"]
        lines.append(
            f"| {name} | {metrics['count']} | "
            f"{_format_number(potential['mae'])} / "
            f"{_format_number(potential['rmse'])} / "
            f"{_format_number(potential['bias'])} | "
            f"{_format_number(step['mae'])} / "
            f"{_format_number(step['rmse'])} / "
            f"{_format_number(step['bias'])} | "
            f"{_format_number(direction['mean'])} | "
            f"{_format_number(ambiguity['mae'])} / "
            f"{_format_number(ambiguity['brier'])} / "
            f"{_format_number(ambiguity['spearman_correlation'])} |"
        )

    dense = report["dense_grid"]
    lines.extend(
        [
            "",
            "## Dense grid",
            "",
            "| Metric | abs potential | step distance |",
            "|---|---:|---:|",
            f"| MAE to true distance (mm) | "
            f"{_format_number(dense['distance_accuracy']['abs_potential']['mae'])} | "
            f"{_format_number(dense['distance_accuracy']['step_distance']['mae'])} |",
            f"| RMSE to true distance (mm) | "
            f"{_format_number(dense['distance_accuracy']['abs_potential']['rmse'])} | "
            f"{_format_number(dense['distance_accuracy']['step_distance']['rmse'])} |",
            f"| Ghost fraction (all) | "
            f"{_format_number(dense['ghost_missed_near']['abs_potential']['ghost_fraction_all'])} | "
            f"{_format_number(dense['ghost_missed_near']['step_distance']['ghost_fraction_all'])} |",
            f"| Missed-near fraction (true-near) | "
            f"{_format_number(dense['ghost_missed_near']['abs_potential']['missed_near_fraction_true_near'])} | "
            f"{_format_number(dense['ghost_missed_near']['step_distance']['missed_near_fraction_true_near'])} |",
            "",
            "## Candidate ranking and projection",
            "",
            "| OOD gate | Ranking | N | Initial point→GT mean | Final point→GT mean | "
            "Final coverage mean | Final worsened fraction |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for gate_name, methods in report["candidate_rankings"].items():
        for method, metrics in methods.items():
            if metrics["status"] != "ok":
                lines.append(
                    f"| {gate_name} | {method} | 0 | n/a | n/a | n/a | n/a |"
                )
                continue
            initial = metrics["projection"][0]
            final = metrics["projection"][-1]
            transition = final.get("from_initial") or final.get("transition")
            lines.append(
                f"| {gate_name} | {method} | {metrics['selected_count']} | "
                f"{_format_number(initial['point_to_gt_mm']['mean'])} | "
                f"{_format_number(final['point_to_gt_mm']['mean'])} | "
                f"{_format_number(final['gt_sample_to_points_coverage_mm']['mean'])} | "
                f"{_format_number(transition['worsened_fraction'] if transition else None)} |"
            )

    gradient = report["saved_queries"]["potential_autograd"]["all"]
    lines.extend(
        [
            "",
            "## Potential autograd consistency",
            "",
            f"- Gradient norm mean: "
            f"{_format_number(gradient['gradient_norm']['mean'])}",
            f"- cos(-∇potential, predicted direction) mean: "
            f"{_format_number(gradient['negative_gradient_vs_predicted_direction']['cosine']['mean'])}",
            f"- angle(-∇potential, predicted direction) mean: "
            f"{_format_number(gradient['negative_gradient_vs_predicted_direction']['angle_deg']['mean'])} deg",
            "",
        ]
    )
    artifacts = report.get("artifacts")
    assessment = report["provisional_quality_assessment"]
    lines.extend(
        [
            "## Provisional quality gates",
            "",
            "| Gate | Value | Condition | Result |",
            "|---|---:|---:|---|",
        ]
    )
    for name, check in assessment["checks"].items():
        lines.append(
            f"| {name} | {_format_number(check['value'])} | "
            f"{check['comparison']} {_format_number(check['threshold'])} | "
            f"{'PASS' if check['passed'] else 'FAIL'} |"
        )
    lines.append("")
    if isinstance(artifacts, Mapping):
        lines.extend(
            [
                "## Diagnostic plots",
                "",
                f"![Distance-field calibration]({artifacts['distance_plot']})",
                "",
                f"![Projection trajectories]({artifacts['projection_plot']})",
                "",
                f"![Projected-point positional relationship]"
                f"({artifacts['spatial_plot']})",
                "",
                f"![Ambiguity calibration]({artifacts['ambiguity_plot']})",
                "",
            ]
        )
    return "\n".join(lines)


def _plot_sample_indices(length: int, limit: int = 20_000) -> np.ndarray:
    if length <= limit:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, limit, dtype=np.int64)


def _principal_plane(
    reference_xyz: np.ndarray, query_xyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    reference = _validated_xyz(reference_xyz, "reference_xyz").astype(np.float64)
    query = _validated_xyz(query_xyz, "query_xyz").astype(np.float64)
    center = reference.mean(axis=0)
    _, _, right = np.linalg.svd(reference - center, full_matrices=False)
    basis = right[:2].T
    return (reference - center) @ basis, (query - center) @ basis


def write_diagnostic_plots(
    paths: Mapping[str, pathlib.Path],
    report: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    """Write visual references for field calibration and point relationships."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    truth = np.asarray(arrays["grid_true_distance_mm"], dtype=np.float64)
    potential = np.abs(
        np.asarray(arrays["grid_soft_potential_mm"], dtype=np.float64)
    )
    step = np.asarray(arrays["grid_step_distance_mm"], dtype=np.float64)
    sample = _plot_sample_indices(len(truth))
    display_limit = max(
        1.0,
        float(np.quantile(np.concatenate([truth, potential, step]), 0.99)),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for predicted, label, color in (
        (potential, "|potential|", "tab:blue"),
        (step, "step distance", "tab:orange"),
    ):
        axes[0].scatter(
            truth[sample],
            predicted[sample],
            s=3,
            alpha=0.15,
            color=color,
            label=label,
            rasterized=True,
        )
    axes[0].plot([0, display_limit], [0, display_limit], "k--", linewidth=1)
    axes[0].set(xlim=(0, display_limit), ylim=(0, display_limit))
    axes[0].set_xlabel("True distance (mm)")
    axes[0].set_ylabel("Predicted distance proxy (mm)")
    axes[0].set_title("Dense-grid distance field")
    axes[0].legend()

    calibration = report["dense_grid"]["calibration_by_true_distance"]
    labels = [row["range_mm"] for row in calibration]
    x = np.arange(len(labels))
    axes[1].plot(
        x,
        [row["abs_potential_error"]["mae"] for row in calibration],
        marker="o",
        label="|potential| MAE",
    )
    axes[1].plot(
        x,
        [row["step_distance_error"]["mae"] for row in calibration],
        marker="o",
        label="step MAE",
    )
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].set_ylabel("MAE (mm)")
    axes[1].set_title("Error by true-distance band")
    axes[1].legend()
    fig.savefig(paths["distance_plot"], dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    colors = {
        method: plt.cm.tab10(index)
        for index, method in enumerate(RANKING_METHODS)
    }
    for gate_name, linestyle in (
        ("without_ood_gate", "--"),
        ("with_ood_gate", "-"),
    ):
        for method, metrics in report["candidate_rankings"][gate_name].items():
            if metrics["status"] != "ok":
                continue
            history = metrics["projection"]
            iterations = [row["iteration"] for row in history]
            label = (
                f"{method} "
                f"({'gate' if gate_name.startswith('with_') else 'no gate'})"
            )
            axes[0].plot(
                iterations,
                [row["point_to_gt_mm"]["mean"] for row in history],
                color=colors[method],
                linestyle=linestyle,
                label=label,
            )
            axes[1].plot(
                iterations,
                [
                    row["gt_sample_to_points_coverage_mm"]["mean"]
                    for row in history
                ],
                color=colors[method],
                linestyle=linestyle,
                label=label,
            )
    axes[0].set_title("Projected points to GT")
    axes[0].set_ylabel("Mean distance (mm)")
    axes[1].set_title("GT samples to projected points (coverage)")
    axes[1].set_ylabel("Mean distance (mm)")
    for axis in axes:
        axis.set_xlabel("Projection iteration")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)
    fig.savefig(paths["projection_plot"], dpi=160)
    plt.close(fig)

    gt_samples = np.asarray(arrays["gt_surface_samples_world"], dtype=np.float64)
    spatial_cases = (
        ("without_ood_gate", "abs_potential", "No gate: |potential|"),
        ("with_ood_gate", "spatial_balanced", "Gate: spatial balanced"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    gt_tree = cKDTree(gt_samples)
    points = None
    input_cloud = np.asarray(arrays["input_cloud_world"], dtype=np.float64)
    gt_2d, input_2d = _principal_plane(gt_samples, input_cloud)
    input_distance, _ = gt_tree.query(input_cloud, k=1)
    axes[0].scatter(
        gt_2d[:, 0],
        gt_2d[:, 1],
        s=1,
        color="0.75",
        alpha=0.25,
        rasterized=True,
    )
    points = axes[0].scatter(
        input_2d[:, 0],
        input_2d[:, 1],
        c=np.clip(input_distance, 0.0, 5.0),
        s=5,
        cmap="viridis",
        vmin=0.0,
        vmax=5.0,
        rasterized=True,
    )
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title("Model input cloud")
    axes[0].set_xlabel("GT principal axis 1")
    axes[0].set_ylabel("GT principal axis 2")

    for axis, (gate, method, title) in zip(axes[1:], spatial_cases):
        projection_count = len(
            report["candidate_rankings"][gate][method]["projection"]
        )
        key = (
            f"{gate}_{method}_projection_{projection_count - 1}_world"
        )
        projected = np.asarray(arrays[key], dtype=np.float64)
        gt_2d, projected_2d = _principal_plane(gt_samples, projected)
        approximate_distance, _ = gt_tree.query(projected, k=1)
        axis.scatter(
            gt_2d[:, 0],
            gt_2d[:, 1],
            s=1,
            color="0.75",
            alpha=0.25,
            rasterized=True,
        )
        points = axis.scatter(
            projected_2d[:, 0],
            projected_2d[:, 1],
            c=np.clip(approximate_distance, 0.0, 5.0),
            s=5,
            cmap="viridis",
            vmin=0.0,
            vmax=5.0,
            rasterized=True,
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(title)
        axis.set_xlabel("GT principal axis 1")
        axis.set_ylabel("GT principal axis 2")
    if points is not None:
        fig.colorbar(
            points, ax=axes, label="Approx. point-to-GT distance (mm)"
        )
    fig.savefig(paths["spatial_plot"], dpi=160)
    plt.close(fig)

    gt_ambiguity = np.asarray(arrays["query_gt_ambiguity"], dtype=np.float64)
    pred_ambiguity = np.asarray(
        arrays["query_pred_ambiguity"], dtype=np.float64
    )
    reliability = report["saved_queries"]["by_category"]["all"]["ambiguity"][
        "reliability_bins"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    valid_rows = [row for row in reliability if row["count"] > 0]
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].plot(
        [row["mean_prediction"] for row in valid_rows],
        [row["mean_target"] for row in valid_rows],
        marker="o",
    )
    axes[0].set(xlim=(0, 1), ylim=(0, 1))
    axes[0].set_xlabel("Predicted ambiguity")
    axes[0].set_ylabel("Mean GT ambiguity")
    axes[0].set_title("Ambiguity reliability")
    axes[1].hist(
        gt_ambiguity, bins=30, range=(0, 1), alpha=0.6, label="GT"
    )
    axes[1].hist(
        pred_ambiguity,
        bins=30,
        range=(0, 1),
        alpha=0.6,
        label="prediction",
    )
    axes[1].set_xlabel("Branch ambiguity")
    axes[1].set_ylabel("Query count")
    axes[1].set_title("Ambiguity distribution")
    axes[1].legend()
    fig.savefig(paths["ambiguity_plot"], dpi=160)
    plt.close(fig)


def output_paths(prefix: pathlib.Path) -> dict[str, pathlib.Path]:
    base = pathlib.Path(prefix)
    if base.suffix.lower() in {".json", ".md", ".npz"}:
        base = base.with_suffix("")
    return {
        "json": base.with_suffix(".json"),
        "markdown": base.with_suffix(".md"),
        "npz": base.with_suffix(".npz"),
        "distance_plot": base.with_name(f"{base.name}_distance_field.png"),
        "projection_plot": base.with_name(f"{base.name}_projection.png"),
        "spatial_plot": base.with_name(f"{base.name}_spatial.png"),
        "ambiguity_plot": base.with_name(f"{base.name}_ambiguity.png"),
    }


def write_outputs(
    prefix: pathlib.Path,
    report: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, pathlib.Path]:
    paths = output_paths(prefix)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    report_with_artifacts = dict(report)
    report_with_artifacts["artifacts"] = {
        key: path.name
        for key, path in paths.items()
        if key.endswith("_plot")
    }
    write_diagnostic_plots(paths, report_with_artifacts, arrays)
    paths["json"].write_text(
        json.dumps(
            _json_safe(report_with_artifacts),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["markdown"].write_text(
        render_markdown(report_with_artifacts), encoding="utf-8"
    )
    np.savez_compressed(
        paths["npz"], **{key: np.asarray(value) for key, value in arrays.items()}
    )
    return paths


def process(
    config: AnalysisConfig, *, model: Any | None = None
) -> dict[str, pathlib.Path]:
    report, arrays = analyze(config, model=model)
    return write_outputs(config.output_prefix, report, arrays)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--ckpt", type=pathlib.Path, required=True)
    parser.add_argument("--data", dest="data_h5", type=pathlib.Path, required=True)
    parser.add_argument("--gt-ply", type=pathlib.Path, required=True)
    parser.add_argument("--out-prefix", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk", "--chunk-size", dest="chunk_size", type=int, default=100_000)
    parser.add_argument("--grid-res", type=int, default=48)
    parser.add_argument("--candidate-count", type=int, default=4_096)
    parser.add_argument("--projection-iters", dest="projection_iterations", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bbox-pad-fraction", type=float, default=0.15)
    parser.add_argument("--input-dist-threshold-mm", type=float, default=10.0)
    parser.add_argument("--max-step-mm", type=float, default=5.0)
    parser.add_argument("--ambiguity-damping", type=float, default=1.0)
    parser.add_argument("--ambiguity-penalty-mm", type=float, default=2.0)
    parser.add_argument("--coverage-samples", type=int, default=10_000)
    parser.add_argument("--reliability-bins", type=int, default=10)
    parser.add_argument("--near-threshold-mm", type=float, default=2.0)
    parser.add_argument("--ghost-distance-mm", type=float, default=10.0)
    parser.add_argument("--missed-prediction-mm", type=float, default=5.0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = AnalysisConfig(
        checkpoint=args.checkpoint,
        data_h5=args.data_h5,
        gt_ply=args.gt_ply,
        output_prefix=args.out_prefix,
        seed=args.seed,
        chunk_size=args.chunk_size,
        grid_res=args.grid_res,
        candidate_count=args.candidate_count,
        projection_iterations=args.projection_iterations,
        device=args.device,
        bbox_pad_fraction=args.bbox_pad_fraction,
        max_input_distance_mm=args.input_dist_threshold_mm,
        max_step_mm=args.max_step_mm,
        ambiguity_damping=args.ambiguity_damping,
        ambiguity_penalty_mm=args.ambiguity_penalty_mm,
        coverage_sample_count=args.coverage_samples,
        reliability_bins=args.reliability_bins,
        near_threshold_mm=args.near_threshold_mm,
        ghost_distance_mm=args.ghost_distance_mm,
        missed_prediction_mm=args.missed_prediction_mm,
    )
    paths = process(config)
    for name, path in paths.items():
        print(f"saved {name}: {path}", flush=True)


if __name__ == "__main__":
    main()

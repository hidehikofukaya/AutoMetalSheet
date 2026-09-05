"""Train the independent Softmin Guidance model.

The four supervised fields remain separate throughout optimization:

* signed softmin potential regression,
* positive hard-step distance regression,
* soft toward-surface direction cosine loss, weighted by direction strength,
* branch ambiguity regression.

Potential targets are never floored or passed through softplus.  Scalar
distance losses use category-specific clipping, while checkpoint selection
uses honest, unclipped near-category MAE for both potential and step distance.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
import pathlib
import random
import uuid
from typing import Any, Mapping

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .softmin_guidance import (
        SoftminGuidanceModel,
        build_checkpoint_metadata,
    )
    from .softmin_guidance_dataset import (
        SoftminGuidanceDataset,
        collate_softmin_guidance,
    )
except ImportError:  # Support direct execution from src/model.
    from softmin_guidance import (  # type: ignore[no-redef]
        SoftminGuidanceModel,
        build_checkpoint_metadata,
    )
    from softmin_guidance_dataset import (  # type: ignore[no-redef]
        SoftminGuidanceDataset,
        collate_softmin_guidance,
    )


CATEGORY_NEAR = 0
CATEGORY_FAR = 1
CATEGORY_BOUNDARY = 2
CATEGORY_NAMES = ("near", "far", "boundary")
TRAINER_SCHEMA_VERSION = "train.softmin_guidance.v1"


@dataclass(frozen=True)
class GuidanceLossConfig:
    """Weights and clips for the independent guidance objectives."""

    category_weights: tuple[float, float, float] = (1.0, 0.2, 0.5)
    distance_clips_mm: tuple[float, float, float] = (5.0, 5.0, 40.0)
    lambda_potential: float = 1.0
    lambda_step: float = 1.0
    lambda_direction: float = 0.5
    lambda_ambiguity: float = 0.3
    lambda_consistency: float = 0.0
    lambda_head_consistency: float = 0.0
    head_consistency_margin_mm: float = 0.1

    def __post_init__(self) -> None:
        if len(self.category_weights) != 3 or len(self.distance_clips_mm) != 3:
            raise ValueError("category_weights and distance_clips_mm require 3 values")
        if any(weight < 0 for weight in self.category_weights):
            raise ValueError("category weights must be non-negative")
        if not any(weight > 0 for weight in self.category_weights):
            raise ValueError("at least one category weight must be positive")
        if any(clip <= 0 for clip in self.distance_clips_mm):
            raise ValueError("distance clips must be positive")
        lambdas = (
            self.lambda_potential,
            self.lambda_step,
            self.lambda_direction,
            self.lambda_ambiguity,
            self.lambda_consistency,
            self.lambda_head_consistency,
        )
        if any(value < 0 for value in lambdas):
            raise ValueError("loss lambdas must be non-negative")
        if self.head_consistency_margin_mm < 0:
            raise ValueError("head_consistency_margin_mm must be non-negative")


@dataclass(frozen=True)
class TrainConfig:
    """Serializable configuration for a Softmin Guidance training run."""

    data: str | pathlib.Path
    ckpt_dir: str | pathlib.Path = "checkpoints_softmin_guidance"
    epochs: int = 200
    batch_size: int = 2
    lr: float = 2.0e-4
    weight_decay: float = 1.0e-4
    n_query_sample: int | None = 4096
    n_points_sample: int | None = None
    num_workers: int = 0
    seed: int = 0
    device: str = "auto"
    near_weight: float = 1.0
    far_weight: float = 0.2
    boundary_weight: float = 0.5
    near_clip_dist_mm: float = 5.0
    far_clip_dist_mm: float = 5.0
    boundary_clip_dist_mm: float = 40.0
    lambda_potential: float = 1.0
    lambda_step: float = 1.0
    lambda_direction: float = 0.5
    lambda_ambiguity: float = 0.3
    lambda_consistency: float = 0.0
    lambda_head_consistency: float = 0.0
    head_consistency_margin_mm: float = 0.1
    best_potential_weight: float = 1.0
    best_step_weight: float = 1.0
    best_direction_weight_mm: float = 0.5
    best_ambiguity_weight_mm: float = 0.5
    token_dim: int = 384
    n_tokens: int = 256
    k_neighbors: int = 32
    n_latents: int = 128
    enc_layers: int = 6
    dec_layers: int = 2
    n_heads: int = 6
    n_freqs: int = 8
    dropout: float = 0.0
    use_thickness_in_encoder: bool = True
    ckpt_every: int = 50
    log_every: int = 10
    grad_clip_norm: float = 1.0
    val_data: str | pathlib.Path | None = None
    val_fraction: float = 0.0

    def loss_config(self) -> GuidanceLossConfig:
        return GuidanceLossConfig(
            category_weights=(
                self.near_weight,
                self.far_weight,
                self.boundary_weight,
            ),
            distance_clips_mm=(
                self.near_clip_dist_mm,
                self.far_clip_dist_mm,
                self.boundary_clip_dist_mm,
            ),
            lambda_potential=self.lambda_potential,
            lambda_step=self.lambda_step,
            lambda_direction=self.lambda_direction,
            lambda_ambiguity=self.lambda_ambiguity,
            lambda_consistency=self.lambda_consistency,
            lambda_head_consistency=self.lambda_head_consistency,
            head_consistency_margin_mm=self.head_consistency_margin_mm,
        )

    def validate(self) -> None:
        self.loss_config()
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must lie in [0, 1)")
        if self.val_data is not None and self.val_fraction > 0:
            raise ValueError("val_data and val_fraction are mutually exclusive")
        if any(
            value < 0
            for value in (
                self.best_potential_weight,
                self.best_step_weight,
                self.best_direction_weight_mm,
                self.best_ambiguity_weight_mm,
            )
        ):
            raise ValueError("best metric weights must be non-negative")
        if (
            self.best_potential_weight
            + self.best_step_weight
            + self.best_direction_weight_mm
            + self.best_ambiguity_weight_mm
            <= 0
        ):
            raise ValueError("at least one best metric weight must be positive")
        if self.ckpt_every < 0 or self.log_every < 0:
            raise ValueError("checkpoint/log intervals must be non-negative")
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive")


def _resolve_dataset_paths(source: str | pathlib.Path) -> list[pathlib.Path]:
    root = pathlib.Path(source)
    paths = sorted(root.rglob("*_dataset.h5")) if root.is_dir() else [root]
    if not paths:
        raise ValueError("No *_dataset.h5 files found")
    return paths


def _part_identity(path: pathlib.Path) -> str:
    """Resolve a stable part key so copied/tessellated H5 files stay together."""

    with h5py.File(path, "r") as h5_file:
        raw = h5_file.attrs.get("source_part_id")
        if raw is None:
            raw = h5_file.attrs.get("source_stp")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="strict")
    if raw is None or not str(raw).strip():
        return str(path.resolve()).replace("\\", "/").casefold()
    return str(raw).strip().replace("\\", "/").casefold()


def split_query_indices(
    n_query: int,
    *,
    val_fraction: float,
    seed: int,
    categories: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic, exhaustive, category-stratified split."""

    if n_query < 2:
        raise ValueError("query holdout requires at least two queries")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("query holdout val_fraction must lie in (0, 1)")
    if categories is None:
        categories = np.zeros(n_query, dtype=np.int64)
    categories = np.asarray(categories, dtype=np.int64)
    if categories.shape != (n_query,):
        raise ValueError(f"categories must have shape [{n_query}]")

    rng = np.random.default_rng(seed)
    train_groups: list[np.ndarray] = []
    val_groups: list[np.ndarray] = []
    for category in np.unique(categories):
        group = np.flatnonzero(categories == category)
        if len(group) < 2:
            raise ValueError(
                "stratified query holdout requires at least two queries "
                f"for category {int(category)}"
            )
        group = group[rng.permutation(len(group))]
        val_count = int(np.floor(len(group) * val_fraction + 0.5))
        val_count = min(max(val_count, 1), len(group) - 1)
        val_groups.append(group[:val_count])
        train_groups.append(group[val_count:])

    train_indices = np.concatenate(train_groups).astype(np.int64, copy=False)
    val_indices = np.concatenate(val_groups).astype(np.int64, copy=False)
    train_indices = train_indices[rng.permutation(len(train_indices))]
    val_indices = val_indices[rng.permutation(len(val_indices))]
    if np.intersect1d(train_indices, val_indices).size:
        raise RuntimeError("query holdout produced overlapping indices")
    return train_indices, val_indices


def _index_digest(indices: np.ndarray) -> str:
    canonical = np.sort(indices).astype("<i8", copy=False)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _build_datasets(
    config: TrainConfig,
) -> tuple[
    SoftminGuidanceDataset,
    SoftminGuidanceDataset | None,
    dict[str, object],
]:
    """Build leak-free train/validation datasets and auditable split metadata."""

    train_paths = _resolve_dataset_paths(config.data)
    if config.val_data is not None:
        val_paths = _resolve_dataset_paths(config.val_data)
        train_resolved = {str(path.resolve()).casefold() for path in train_paths}
        val_resolved = {str(path.resolve()).casefold() for path in val_paths}
        overlap = train_resolved & val_resolved
        if overlap:
            raise ValueError(
                "train and validation part paths overlap; part holdout requires "
                "disjoint HDF5 files"
            )
        train_part_ids = {_part_identity(path) for path in train_paths}
        val_part_ids = {_part_identity(path) for path in val_paths}
        identity_overlap = train_part_ids & val_part_ids
        if identity_overlap:
            raise ValueError(
                "train and validation source part identities overlap; copied "
                "or tessellated variants must remain in the same fold"
            )
        train_dataset = SoftminGuidanceDataset(
            train_paths,
            n_query_sample=config.n_query_sample,
            n_points_sample=config.n_points_sample,
            seed=config.seed,
        )
        val_dataset = SoftminGuidanceDataset(
            val_paths,
            n_query_sample=config.n_query_sample,
            n_points_sample=config.n_points_sample,
            seed=config.seed,
            fixed_sampling=True,
        )
        metadata = {
            "strategy": "part_holdout",
            "has_validation": True,
            "seed": config.seed,
            "train_part_count": len(train_dataset),
            "val_part_count": len(val_dataset),
            "part_overlap_count": 0,
            "source_part_overlap_count": 0,
            "train_query_count": sum(train_dataset.query_counts),
            "val_query_count": sum(val_dataset.query_counts),
            "query_overlap_count": 0,
            "shared_point_cloud_input": False,
        }
        return train_dataset, val_dataset, metadata

    if config.val_fraction > 0:
        if len(train_paths) != 1:
            raise ValueError(
                "val_fraction query holdout requires data to resolve to exactly "
                "one HDF5 part; use val_data for part holdout"
            )
        source = train_paths[0]
        with h5py.File(source, "r") as h5_file:
            if "query_xyz" not in h5_file:
                raise KeyError(f"{source} is missing query_xyz")
            n_query = int(h5_file["query_xyz"].shape[0])
            if "query_category" not in h5_file:
                raise KeyError(f"{source} is missing query_category")
            categories = h5_file["query_category"][:]
        train_indices, val_indices = split_query_indices(
            n_query,
            val_fraction=config.val_fraction,
            seed=config.seed,
            categories=categories,
        )
        train_dataset = SoftminGuidanceDataset(
            [source],
            n_query_sample=config.n_query_sample,
            n_points_sample=config.n_points_sample,
            seed=config.seed,
            query_indices=[train_indices],
        )
        val_dataset = SoftminGuidanceDataset(
            [source],
            n_query_sample=config.n_query_sample,
            n_points_sample=config.n_points_sample,
            seed=config.seed,
            query_indices=[val_indices],
            fixed_sampling=True,
        )
        overlap_count = int(np.intersect1d(train_indices, val_indices).size)
        if overlap_count:
            raise RuntimeError("train/validation query leakage detected")
        metadata = {
            "strategy": "query_holdout",
            "has_validation": True,
            "seed": config.seed,
            "val_fraction": config.val_fraction,
            "train_part_count": 1,
            "val_part_count": 1,
            "part_overlap_count": 1,
            "train_query_count": len(train_indices),
            "val_query_count": len(val_indices),
            "query_overlap_count": overlap_count,
            "shared_point_cloud_input": True,
            "train_query_index_sha256": _index_digest(train_indices),
            "val_query_index_sha256": _index_digest(val_indices),
        }
        return train_dataset, val_dataset, metadata

    train_dataset = SoftminGuidanceDataset(
        train_paths,
        n_query_sample=config.n_query_sample,
        n_points_sample=config.n_points_sample,
        seed=config.seed,
    )
    metadata = {
        "strategy": "none",
        "has_validation": False,
        "seed": config.seed,
        "train_part_count": len(train_dataset),
        "val_part_count": 0,
        "part_overlap_count": 0,
        "train_query_count": sum(train_dataset.query_counts),
        "val_query_count": 0,
        "query_overlap_count": 0,
        "shared_point_cloud_input": False,
    }
    return train_dataset, None, metadata


def _as_tensor(
    batch: Mapping[str, object], key: str, device: torch.device
) -> torch.Tensor:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"batch requires tensor field {key!r}")
    return value.to(device)


def _thickness_norm(batch: Mapping[str, object], device: torch.device) -> torch.Tensor:
    """Per-part thickness in the model's normalized coordinate system.

    Mirrors the points/query_xyz convention (``value / scale``); the model
    ignores this tensor internally when ``use_thickness_in_encoder=False``.
    """

    thickness_mm = _as_tensor(batch, "thickness_mm", device)
    scale = _as_tensor(batch, "scale", device)
    return thickness_mm / scale.clamp_min(1.0e-6)


def _category_values(
    values: tuple[float, float, float],
    category: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    if category.dtype != torch.long:
        category = category.to(torch.long)
    if category.numel() and (
        int(category.min().item()) < CATEGORY_NEAR
        or int(category.max().item()) > CATEGORY_BOUNDARY
    ):
        raise ValueError("query_category values must be 0=near, 1=far, or 2=boundary")
    table = torch.tensor(values, device=category.device, dtype=dtype)
    return table[category]


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denominator = weights.sum()
    if not torch.isfinite(denominator):
        raise ValueError("loss weights must be finite")
    return (values * weights).sum() / denominator.clamp_min(1.0e-12)


def _validate_guidance_targets(
    step_distance: torch.Tensor,
    ambiguity: torch.Tensor,
    direction_strength: torch.Tensor,
) -> None:
    for name, value in (
        ("query_step_distance", step_distance),
        ("query_branch_ambiguity", ambiguity),
        ("query_direction_strength", direction_strength),
    ):
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
    if torch.any(step_distance < 0):
        raise ValueError("query_step_distance must be non-negative")
    if torch.any((ambiguity < 0) | (ambiguity > 1)):
        raise ValueError("query_branch_ambiguity must lie in [0, 1]")
    if torch.any((direction_strength < 0) | (direction_strength > 1)):
        raise ValueError("query_direction_strength must lie in [0, 1]")


def potential_gradient_consistency_loss(
    signed_potential_normalized: torch.Tensor,
    query_xyz: torch.Tensor,
    predicted_toward_surface: torch.Tensor,
    weights: torch.Tensor,
    *,
    create_graph: bool,
) -> torch.Tensor:
    """Align ``-grad(potential)`` with the model's toward-surface direction."""

    potential_gradient = torch.autograd.grad(
        outputs=signed_potential_normalized,
        inputs=query_xyz,
        grad_outputs=torch.ones_like(signed_potential_normalized),
        create_graph=create_graph,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradient_norm = potential_gradient.norm(dim=-1)
    valid = gradient_norm > 1.0e-4
    if not bool(valid.any()):
        # Preserve a differentiable zero without evaluating cosine at the
        # exactly-zero initialization, whose derivative scales as 1/eps.
        return signed_potential_normalized.sum() * 0.0
    cosine = F.cosine_similarity(
        -potential_gradient[valid],
        predicted_toward_surface[valid],
        dim=-1,
        eps=1.0e-6,
    )
    return _weighted_mean(1.0 - cosine, weights[valid])


@contextmanager
def _math_attention_for_higher_order_gradients(enabled: bool):
    """Use the SDPA math backend when consistency needs double backward.

    PyTorch's optimized CPU flash-attention backward does not implement the
    second derivative required by ``autograd(grad(potential))``.  The math
    backend is slower but supports this optional regularizer on both CPU and
    CUDA.  Ordinary training keeps the default optimized backend.
    """

    if not enabled:
        yield
        return

    attention_module = getattr(torch.nn, "attention", None)
    if attention_module is not None and hasattr(attention_module, "sdpa_kernel"):
        with attention_module.sdpa_kernel(attention_module.SDPBackend.MATH):
            yield
        return

    # Compatibility fallback for PyTorch releases predating torch.nn.attention.
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_math=True,
        enable_mem_efficient=False,
    ):
        yield


def compute_guidance_losses(
    prediction: Any,
    batch: Mapping[str, object],
    query_xyz: torch.Tensor,
    config: GuidanceLossConfig,
    *,
    create_graph: bool,
) -> dict[str, torch.Tensor]:
    """Compute independent losses and unclipped near-category error sums."""

    device = query_xyz.device
    gt_potential = _as_tensor(batch, "query_soft_potential", device)
    gt_step = _as_tensor(batch, "query_step_distance", device)
    gt_direction = _as_tensor(batch, "query_soft_direction", device)
    gt_ambiguity = _as_tensor(batch, "query_branch_ambiguity", device)
    direction_strength = _as_tensor(batch, "query_direction_strength", device)
    category = _as_tensor(batch, "query_category", device).to(torch.long)
    scale = _as_tensor(batch, "scale", device)
    _validate_guidance_targets(gt_step, gt_ambiguity, direction_strength)
    if not torch.isfinite(gt_potential).all():
        raise ValueError("query_soft_potential must contain only finite values")
    if not torch.isfinite(gt_direction).all():
        raise ValueError("query_soft_direction must contain only finite values")

    if scale.ndim != 1:
        raise ValueError("scale must have shape [B]")
    scale_per_query = scale.unsqueeze(-1)
    predicted_potential_mm = prediction.soft_potential * scale_per_query
    predicted_step_mm = prediction.step_distance * scale_per_query

    category_weight = _category_values(
        config.category_weights, category, predicted_potential_mm.dtype
    )
    clip_mm = _category_values(
        config.distance_clips_mm, category, predicted_potential_mm.dtype
    )

    # Signed potential remains signed. Clip the TARGET, not the prediction:
    # clamping both sides gives exactly zero gradient when a randomly
    # initialized prediction starts outside the band after mm rescaling.
    # The unclamped prediction therefore always receives a gradient back
    # toward the signed truncated target.
    potential_error = (
        predicted_potential_mm
        - torch.clamp(gt_potential, min=-clip_mm, max=clip_mm)
    ).abs()
    potential_loss = _weighted_mean(potential_error, category_weight)

    # The model's step head is positive by construction.  Targets are checked
    # rather than silently floored, then only upper-clipped for robust fitting.
    step_error = (
        predicted_step_mm
        - torch.minimum(gt_step, clip_mm)
    ).abs()
    step_loss = _weighted_mean(step_error, category_weight)

    direction_cosine = F.cosine_similarity(
        prediction.direction_toward_surface,
        gt_direction,
        dim=-1,
        eps=1.0e-8,
    )
    direction_weight = category_weight * direction_strength
    direction_loss = _weighted_mean(1.0 - direction_cosine, direction_weight)

    ambiguity_error = (prediction.branch_ambiguity - gt_ambiguity).abs()
    ambiguity_loss = _weighted_mean(ambiguity_error, category_weight)

    # Penalize the same |potential| > step + margin violation the post-hoc
    # `head_consistency` eval metric flags, so training directly discourages
    # it instead of only measuring it after the fact. Pure forward-pass
    # quantities (no autograd.grad through query_xyz), so unlike
    # consistency_loss this is cheap enough to always compute and is safe to
    # evaluate under torch.no_grad() during validation.
    head_consistency_violation = F.relu(
        predicted_potential_mm.abs()
        - (predicted_step_mm + config.head_consistency_margin_mm)
    )
    head_consistency_loss = _weighted_mean(
        head_consistency_violation, category_weight
    )

    consistency_loss = predicted_potential_mm.new_zeros(())
    if config.lambda_consistency > 0:
        if not query_xyz.requires_grad:
            raise ValueError(
                "query_xyz must require gradients when lambda_consistency > 0"
            )
        consistency_loss = potential_gradient_consistency_loss(
            prediction.soft_potential,
            query_xyz,
            prediction.direction_toward_surface,
            direction_weight,
            create_graph=create_graph,
        )

    total = (
        config.lambda_potential * potential_loss
        + config.lambda_step * step_loss
        + config.lambda_direction * direction_loss
        + config.lambda_ambiguity * ambiguity_loss
        + config.lambda_consistency * consistency_loss
        + config.lambda_head_consistency * head_consistency_loss
    )

    near_mask = category == CATEGORY_NEAR
    near_potential_abs = (predicted_potential_mm - gt_potential).abs()
    near_step_abs = (predicted_step_mm - gt_step).abs()
    near_direction_weight = direction_strength[near_mask]
    return {
        "total": total,
        "potential": potential_loss,
        "step": step_loss,
        "direction": direction_loss,
        "ambiguity": ambiguity_loss,
        "consistency": consistency_loss,
        "head_consistency": head_consistency_loss,
        "near_potential_abs_sum": near_potential_abs[near_mask].sum(),
        "near_step_abs_sum": near_step_abs[near_mask].sum(),
        "near_direction_error_sum": (
            (1.0 - direction_cosine[near_mask]) * near_direction_weight
        ).sum(),
        "near_direction_weight_sum": near_direction_weight.sum(),
        "near_ambiguity_abs_sum": ambiguity_error[near_mask].sum(),
        "near_count": near_mask.sum(),
    }


def run_epoch(
    model: SoftminGuidanceModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> dict[str, float]:
    """Run one training epoch and return scalar loss/selection metrics."""

    model.train()
    loss_config = config.loss_config()
    totals = {
        "total_loss": 0.0,
        "potential_loss": 0.0,
        "step_loss": 0.0,
        "direction_loss": 0.0,
        "ambiguity_loss": 0.0,
        "consistency_loss": 0.0,
        "head_consistency_loss": 0.0,
    }
    part_count = 0
    near_potential_abs_sum = 0.0
    near_step_abs_sum = 0.0
    near_direction_error_sum = 0.0
    near_direction_weight_sum = 0.0
    near_ambiguity_abs_sum = 0.0
    near_count = 0

    for batch in loader:
        points = _as_tensor(batch, "points", device)
        normals = _as_tensor(batch, "normals", device)
        query_xyz = _as_tensor(batch, "query_xyz", device)
        thickness_norm = _thickness_norm(batch, device)
        if loss_config.lambda_consistency > 0:
            query_xyz.requires_grad_(True)

        optimizer.zero_grad(set_to_none=True)
        with _math_attention_for_higher_order_gradients(
            loss_config.lambda_consistency > 0
        ):
            prediction = model(points, normals, query_xyz, thickness_norm=thickness_norm)
            losses = compute_guidance_losses(
                prediction,
                batch,
                query_xyz,
                loss_config,
                create_graph=loss_config.lambda_consistency > 0,
            )
            losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()

        batch_parts = int(points.shape[0])
        part_count += batch_parts
        for metric_name, loss_name in (
            ("total_loss", "total"),
            ("potential_loss", "potential"),
            ("step_loss", "step"),
            ("direction_loss", "direction"),
            ("ambiguity_loss", "ambiguity"),
            ("consistency_loss", "consistency"),
            ("head_consistency_loss", "head_consistency"),
        ):
            totals[metric_name] += (
                float(losses[loss_name].detach().item()) * batch_parts
            )
        near_potential_abs_sum += float(
            losses["near_potential_abs_sum"].detach().item()
        )
        near_step_abs_sum += float(losses["near_step_abs_sum"].detach().item())
        near_direction_error_sum += float(
            losses["near_direction_error_sum"].detach().item()
        )
        near_direction_weight_sum += float(
            losses["near_direction_weight_sum"].detach().item()
        )
        near_ambiguity_abs_sum += float(
            losses["near_ambiguity_abs_sum"].detach().item()
        )
        near_count += int(losses["near_count"].detach().item())

    if part_count == 0:
        raise RuntimeError("training loader produced no batches")
    if near_count == 0:
        raise RuntimeError(
            "best-checkpoint selection requires at least one near-category query"
        )

    metrics = {name: value / part_count for name, value in totals.items()}
    metrics["near_potential_mae_mm"] = near_potential_abs_sum / near_count
    metrics["near_step_mae_mm"] = near_step_abs_sum / near_count
    metrics["near_direction_cosine_error"] = (
        near_direction_error_sum / max(near_direction_weight_sum, 1.0e-12)
    )
    metrics["near_ambiguity_mae"] = near_ambiguity_abs_sum / near_count
    metrics["near_composite"] = (
        config.best_potential_weight * metrics["near_potential_mae_mm"]
        + config.best_step_weight * metrics["near_step_mae_mm"]
        + config.best_direction_weight_mm
        * metrics["near_direction_cosine_error"]
        + config.best_ambiguity_weight_mm * metrics["near_ambiguity_mae"]
    )
    return metrics


def evaluate_epoch(
    model: SoftminGuidanceModel,
    loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
) -> dict[str, float]:
    """Evaluate without optimizer state, parameter gradients, or consistency."""

    model.eval()
    # Only lambda_consistency is forced off: it requires autograd.grad through
    # query_xyz, which is unavailable under torch.no_grad(). head_consistency
    # is a pure forward-pass quantity (see compute_guidance_losses), so it is
    # left at its configured weight and evaluated normally here.
    loss_config = replace(config.loss_config(), lambda_consistency=0.0)
    totals = {
        "total_loss": 0.0,
        "potential_loss": 0.0,
        "step_loss": 0.0,
        "direction_loss": 0.0,
        "ambiguity_loss": 0.0,
        "consistency_loss": 0.0,
        "head_consistency_loss": 0.0,
    }
    part_count = 0
    near_potential_abs_sum = 0.0
    near_step_abs_sum = 0.0
    near_direction_error_sum = 0.0
    near_direction_weight_sum = 0.0
    near_ambiguity_abs_sum = 0.0
    near_count = 0

    with torch.no_grad():
        for batch in loader:
            points = _as_tensor(batch, "points", device)
            normals = _as_tensor(batch, "normals", device)
            query_xyz = _as_tensor(batch, "query_xyz", device)
            thickness_norm = _thickness_norm(batch, device)
            prediction = model(points, normals, query_xyz, thickness_norm=thickness_norm)
            losses = compute_guidance_losses(
                prediction,
                batch,
                query_xyz,
                loss_config,
                create_graph=False,
            )

            batch_parts = int(points.shape[0])
            part_count += batch_parts
            for metric_name, loss_name in (
                ("total_loss", "total"),
                ("potential_loss", "potential"),
                ("step_loss", "step"),
                ("direction_loss", "direction"),
                ("ambiguity_loss", "ambiguity"),
                ("consistency_loss", "consistency"),
            ):
                totals[metric_name] += (
                    float(losses[loss_name].detach().item()) * batch_parts
                )
            near_potential_abs_sum += float(
                losses["near_potential_abs_sum"].detach().item()
            )
            near_step_abs_sum += float(
                losses["near_step_abs_sum"].detach().item()
            )
            near_direction_error_sum += float(
                losses["near_direction_error_sum"].detach().item()
            )
            near_direction_weight_sum += float(
                losses["near_direction_weight_sum"].detach().item()
            )
            near_ambiguity_abs_sum += float(
                losses["near_ambiguity_abs_sum"].detach().item()
            )
            near_count += int(losses["near_count"].detach().item())

    if part_count == 0:
        raise RuntimeError("validation loader produced no batches")
    if near_count == 0:
        raise RuntimeError(
            "best-checkpoint selection requires at least one validation "
            "near-category query"
        )

    metrics = {name: value / part_count for name, value in totals.items()}
    metrics["near_potential_mae_mm"] = near_potential_abs_sum / near_count
    metrics["near_step_mae_mm"] = near_step_abs_sum / near_count
    metrics["near_direction_cosine_error"] = (
        near_direction_error_sum / max(near_direction_weight_sum, 1.0e-12)
    )
    metrics["near_ambiguity_mae"] = near_ambiguity_abs_sum / near_count
    metrics["near_composite"] = (
        config.best_potential_weight * metrics["near_potential_mae_mm"]
        + config.best_step_weight * metrics["near_step_mae_mm"]
        + config.best_direction_weight_mm
        * metrics["near_direction_cosine_error"]
        + config.best_ambiguity_weight_mm * metrics["near_ambiguity_mae"]
    )
    return metrics


def _jsonable_config(config: TrainConfig) -> dict[str, object]:
    serialized = asdict(config)
    serialized["data"] = str(pathlib.Path(config.data))
    serialized["val_data"] = (
        str(pathlib.Path(config.val_data)) if config.val_data is not None else None
    )
    serialized["ckpt_dir"] = str(pathlib.Path(config.ckpt_dir))
    # Fail before training if an accidentally non-serializable field is added.
    json.dumps(serialized)
    return serialized


def _trainer_metadata(
    model: SoftminGuidanceModel,
    config: TrainConfig,
    validation_split: Mapping[str, object],
) -> dict[str, object]:
    metadata = build_checkpoint_metadata(model)
    metadata["trainer_schema_version"] = TRAINER_SCHEMA_VERSION
    metadata["loss_contract"] = {
        "potential": {
            "type": "category_weighted_l1",
            "target": "signed_world_mm",
            "clip": "target_only_symmetric_per_category",
            "softplus_or_floor": False,
        },
        "step_distance": {
            "type": "category_weighted_l1",
            "target": "non_negative_world_mm",
            "clip": "target_only_upper_per_category",
        },
        "direction": {
            "type": "one_minus_cosine",
            "weight": "category_weight * query_direction_strength",
        },
        "ambiguity": {
            "type": "category_weighted_l1",
            "target_range": "[0, 1]",
        },
        "potential_gradient_consistency": {
            "type": "one_minus_cosine",
            "vectors": "-autograd(soft_potential, query_xyz) vs toward_surface",
            "minimum_gradient_norm": 1.0e-4,
            "lambda": config.lambda_consistency,
            "validation_evaluation": "disabled_under_no_grad",
        },
        "head_consistency": {
            "type": "category_weighted_hinge_l1",
            "formula": "relu(|soft_potential_mm| - (step_distance_mm + margin))",
            "margin_mm": config.head_consistency_margin_mm,
            "lambda": config.lambda_head_consistency,
            "validation_evaluation": "enabled_under_no_grad",
            "rationale": (
                "directly penalizes the same |potential| > step + 0.1mm "
                "violation reported post-hoc by the "
                "near_head_consistency_violation_fraction eval gate"
            ),
        },
    }
    has_validation = bool(validation_split["has_validation"])
    selection_prefix = "val" if has_validation else "train"
    validation_strategy = str(validation_split["strategy"])
    metadata["best_checkpoint_selection"] = {
        "metric": f"{selection_prefix}_near_composite",
        "source": "validation" if has_validation else "train_fallback",
        "formula": (
            "best_potential_weight * near_potential_mae_mm + "
            "best_step_weight * near_step_mae_mm + "
            "best_direction_weight_mm * near_direction_cosine_error + "
            "best_ambiguity_weight_mm * near_ambiguity_mae"
        ),
        "best_potential_weight": config.best_potential_weight,
        "best_step_weight": config.best_step_weight,
        "best_direction_weight_mm": config.best_direction_weight_mm,
        "best_ambiguity_weight_mm": config.best_ambiguity_weight_mm,
        "mae_is_unclipped": True,
        "role": "epoch_selection_proxy",
        "requires_post_training_field_gate": True,
        "post_training_gate_includes": [
            "ambiguity_calibration",
            "projection_convergence",
            "bidirectional_coverage",
            "ghost_rate",
            "input_cloud_coverage",
        ],
    }
    metadata["validation_split"] = dict(validation_split)
    metadata["evidence_scope"] = {
        "generalization_valid": validation_strategy == "part_holdout",
        "query_holdout_is_same_part_interpolation_only": (
            validation_strategy == "query_holdout"
        ),
        "train_fallback": validation_strategy == "none",
        "certification_eligible": False,
        "reason": (
            "Formal model confidence requires post-training field gates and "
            "unseen part/family evaluation."
        ),
    }
    metadata["determinism"] = {
        "seed": config.seed,
        "torch_deterministic_algorithms": True,
        "warn_only": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "scope": "same software, OS, and device stack",
    }
    return metadata


def save_checkpoint(payload: dict[str, object], path: pathlib.Path) -> None:
    """Atomically replace a checkpoint without touching unrelated files."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_payload(
    model: SoftminGuidanceModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: TrainConfig,
    metadata: dict[str, object],
    epoch: int,
    metrics: dict[str, float],
    best_metrics: dict[str, object],
    history: list[dict[str, float]],
) -> dict[str, object]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "metadata": metadata,
        "cfg": _jsonable_config(config),
        "metrics": {
            **metrics,
            "best_epoch": int(best_metrics["epoch"]),
            "best_selection_split": best_metrics["selection_split"],
            "best_near_composite": best_metrics["near_composite"],
            "best_near_potential_mae_mm": best_metrics[
                "near_potential_mae_mm"
            ],
            "best_near_step_mae_mm": best_metrics["near_step_mae_mm"],
            "best_near_direction_cosine_error": best_metrics[
                "near_direction_cosine_error"
            ],
            "best_near_ambiguity_mae": best_metrics["near_ambiguity_mae"],
        },
        "history": list(history),
    }


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _set_deterministic_training(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_softmin_guidance(config: TrainConfig) -> dict[str, object]:
    """Train, save best/last checkpoints, and return the run summary."""

    config.validate()
    _set_deterministic_training(config.seed)
    device = _resolve_device(config.device)

    train_dataset, val_dataset, validation_split = _build_datasets(config)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.batch_size, len(train_dataset)),
        shuffle=True,
        collate_fn=collate_softmin_guidance,
        drop_last=False,
        num_workers=config.num_workers,
        generator=generator,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=min(config.batch_size, len(val_dataset)),
            shuffle=False,
            collate_fn=collate_softmin_guidance,
            drop_last=False,
            num_workers=config.num_workers,
        )
        if val_dataset is not None
        else None
    )

    model = SoftminGuidanceModel(
        token_dim=config.token_dim,
        n_tokens=config.n_tokens,
        k_neighbors=config.k_neighbors,
        n_latents=config.n_latents,
        enc_layers=config.enc_layers,
        dec_layers=config.dec_layers,
        n_heads=config.n_heads,
        dropout=config.dropout,
        n_freqs=config.n_freqs,
        use_thickness_in_encoder=config.use_thickness_in_encoder,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    checkpoint_dir = pathlib.Path(config.ckpt_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata = _trainer_metadata(model, config, validation_split)
    history: list[dict[str, float]] = []
    best_metrics: dict[str, object] | None = None
    selection_split = "val" if val_loader is not None else "train"

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, device, optimizer, config
        )
        val_metrics = (
            evaluate_epoch(model, val_loader, device, config)
            if val_loader is not None
            else None
        )
        scheduler.step()
        epoch_metrics = {
            "epoch": float(epoch),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            **{
                f"train_{name}": value
                for name, value in train_metrics.items()
            },
        }
        if val_metrics is not None:
            epoch_metrics.update(
                {f"val_{name}": value for name, value in val_metrics.items()}
            )
        history.append(epoch_metrics)
        selection_metrics = val_metrics or train_metrics

        if config.log_every and (
            epoch == 1
            or epoch == config.epochs
            or epoch % config.log_every == 0
        ):
            val_log = (
                f" val_composite={val_metrics['near_composite']:.5f}"
                if val_metrics is not None
                else " val=none(train fallback)"
            )
            print(
                f"epoch {epoch:4d}/{config.epochs} "
                f"train_total={train_metrics['total_loss']:.5f} "
                f"train_potential={train_metrics['potential_loss']:.5f} "
                f"train_step={train_metrics['step_loss']:.5f} "
                f"train_direction={train_metrics['direction_loss']:.5f} "
                f"train_ambiguity={train_metrics['ambiguity_loss']:.5f} "
                f"train_consistency={train_metrics['consistency_loss']:.5f} "
                f"train_near_potential_mae="
                f"{train_metrics['near_potential_mae_mm']:.5f}mm "
                f"train_near_step_mae="
                f"{train_metrics['near_step_mae_mm']:.5f}mm "
                f"train_composite={train_metrics['near_composite']:.5f}"
                f"{val_log}"
            )

        if (
            best_metrics is None
            or selection_metrics["near_composite"]
            < float(best_metrics["near_composite"])
        ):
            best_metrics = {
                "epoch": float(epoch),
                "selection_split": selection_split,
                **selection_metrics,
            }
            save_checkpoint(
                _checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    config,
                    metadata,
                    epoch,
                    epoch_metrics,
                    best_metrics,
                    history,
                ),
                checkpoint_dir / "best.pt",
            )

        if config.ckpt_every and epoch % config.ckpt_every == 0:
            assert best_metrics is not None
            save_checkpoint(
                _checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    config,
                    metadata,
                    epoch,
                    epoch_metrics,
                    best_metrics,
                    history,
                ),
                checkpoint_dir / f"epoch{epoch:04d}.pt",
            )

    assert best_metrics is not None
    last_metrics = history[-1]
    save_checkpoint(
        _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            config,
            metadata,
            config.epochs,
            last_metrics,
            best_metrics,
            history,
        ),
        checkpoint_dir / "last.pt",
    )
    return {
        "device": str(device),
        "history": history,
        "best_metrics": best_metrics,
        "validation_split": validation_split,
        "best_checkpoint": checkpoint_dir / "best.pt",
        "last_checkpoint": checkpoint_dir / "last.pt",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the independent Softmin Guidance model."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--val-data",
        default=None,
        help="Optional disjoint HDF5 part/file tree used for validation.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help="Deterministic query holdout fraction for a single input HDF5.",
    )
    parser.add_argument("--ckpt-dir", default="checkpoints_softmin_guidance")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--n-query-sample", type=int, default=4096)
    parser.add_argument("--n-points-sample", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--near-weight", type=float, default=1.0)
    parser.add_argument("--far-weight", type=float, default=0.2)
    parser.add_argument("--boundary-weight", type=float, default=0.5)
    parser.add_argument("--near-clip-dist-mm", type=float, default=5.0)
    parser.add_argument("--far-clip-dist-mm", type=float, default=5.0)
    parser.add_argument("--boundary-clip-dist-mm", type=float, default=40.0)
    parser.add_argument("--lambda-potential", type=float, default=1.0)
    parser.add_argument("--lambda-step", type=float, default=1.0)
    parser.add_argument("--lambda-direction", type=float, default=0.5)
    parser.add_argument("--lambda-ambiguity", type=float, default=0.3)
    parser.add_argument("--lambda-consistency", type=float, default=0.0)
    parser.add_argument(
        "--lambda-head-consistency",
        type=float,
        default=0.0,
        help=(
            "Weight for the |potential| > step + margin hinge loss "
            "(potential-vs-step head consistency). 0.0 disables it."
        ),
    )
    parser.add_argument(
        "--head-consistency-margin-mm",
        type=float,
        default=0.1,
        help="Margin mm for the head-consistency hinge loss (matches the eval gate).",
    )
    parser.add_argument("--best-potential-weight", type=float, default=1.0)
    parser.add_argument("--best-step-weight", type=float, default=1.0)
    parser.add_argument("--best-direction-weight-mm", type=float, default=0.5)
    parser.add_argument("--best-ambiguity-weight-mm", type=float, default=0.5)
    parser.add_argument("--token-dim", type=int, default=384)
    parser.add_argument("--n-tokens", type=int, default=256)
    parser.add_argument("--k-neighbors", type=int, default=32)
    parser.add_argument("--n-latents", type=int, default=128)
    parser.add_argument("--enc-layers", type=int, default=6)
    parser.add_argument("--dec-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-freqs", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--no-thickness-in-encoder",
        dest="use_thickness_in_encoder",
        action="store_false",
        default=True,
        help="Disable thickness conditioning in the encoder (architecture ablation).",
    )
    parser.add_argument("--ckpt-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    return parser


def main() -> None:
    config = TrainConfig(**vars(_build_parser().parse_args()))
    summary = train_softmin_guidance(config)
    best = summary["best_metrics"]
    assert isinstance(best, dict)
    print(
        "done: "
        f"best epoch={int(best['epoch'])}, "
        f"near composite={best['near_composite']:.5f} -> "
        f"{summary['best_checkpoint']}"
    )


if __name__ == "__main__":
    main()

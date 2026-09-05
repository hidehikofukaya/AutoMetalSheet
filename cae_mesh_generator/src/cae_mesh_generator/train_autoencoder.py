"""Train a small hierarchical midsurface autoencoder prototype."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .data.fill_volume_dataset import MidsurfacePointCloudDataset, PartRecord, discover_fill_parts
from .data.source_fingerprint import fingerprint_records
from .data.step_tessellate import TessellationConfig, load_or_tessellate
from .model.hierarchical_ae import (
    HierarchicalMidsurfaceAutoencoder,
    StructuredScaffoldAutoencoder,
    chamfer_components,
    normal_chamfer_loss,
    prediction_surface_fraction,
    prediction_surface_loss,
    scaffold_chamfer_loss,
    spread_loss,
    target_coverage_fraction,
    target_coverage_loss,
)


BEST_METRIC_CHOICES = [
    "auto",
    "train_loss",
    "val_loss",
    "train_chamfer",
    "val_chamfer",
    "train_coverage",
    "val_coverage",
    "train_target_within_threshold",
    "val_target_within_threshold",
    "train_target_p95_mm",
    "val_target_p95_mm",
    "train_recon_p95_mm",
    "val_recon_p95_mm",
    "train_pred_within_threshold",
    "val_pred_within_threshold",
    "train_boundary_coverage",
    "val_boundary_coverage",
    "train_boundary_within_threshold",
    "val_boundary_within_threshold",
    "train_boundary_p95_mm",
    "val_boundary_p95_mm",
    "train_cae_score",
    "val_cae_score",
    "train_feature_scaffold_p95_mm",
    "val_feature_scaffold_p95_mm",
    "train_boundary_scaffold_p95_mm",
    "val_boundary_scaffold_p95_mm",
    "train_crease_scaffold_p95_mm",
    "val_crease_scaffold_p95_mm",
    "train_corner_scaffold_p95_mm",
    "val_corner_scaffold_p95_mm",
    "train_refinement_occupancy",
    "val_refinement_occupancy",
]

BOUNDARY_WEIGHTED_METRICS = {
    "boundary_coverage",
    "boundary_within_threshold",
    "boundary_p95_mm",
}

SCAFFOLD_TARGET_WEIGHTED_METRICS = {
    "feature_scaffold": "feature_scaffold_target_count",
    "feature_scaffold_mean_mm": "feature_scaffold_target_count",
    "feature_scaffold_p95_mm": "feature_scaffold_target_count",
    "boundary_scaffold": "boundary_scaffold_target_count",
    "boundary_scaffold_mean_mm": "boundary_scaffold_target_count",
    "boundary_scaffold_p95_mm": "boundary_scaffold_target_count",
    "crease_scaffold": "crease_scaffold_target_count",
    "crease_scaffold_mean_mm": "crease_scaffold_target_count",
    "crease_scaffold_p95_mm": "crease_scaffold_target_count",
    "corner_scaffold": "corner_scaffold_target_count",
    "corner_scaffold_mean_mm": "corner_scaffold_target_count",
    "corner_scaffold_p95_mm": "corner_scaffold_target_count",
}
SCAFFOLD_TARGET_COUNT_METRICS = set(SCAFFOLD_TARGET_WEIGHTED_METRICS.values())


@dataclass(frozen=True)
class PartSizeProfile:
    canonical_part_id: str
    stp_path: str
    bbox_diagonal: float
    bbox_max_extent: float


@dataclass(frozen=True)
class SplitManifest:
    strategy: str
    split_seed: int
    train_indices: list[int]
    val_indices: list[int]
    train_ids: list[str]
    val_ids: list[str]


def main() -> None:
    args = parse_args()
    validate_boundary_args(args)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.resample_train_each_epoch and args.preload_data:
        raise SystemExit("--resample-train-each-epoch cannot be combined with --preload-data; preloading freezes samples.")
    resume_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location=device, weights_only=False)
        apply_resume_model_args(args, resume_payload)
        validate_boundary_args(args)
    tessellation = TessellationConfig(
        linear_deflection=args.linear_deflection,
        angular_deflection=args.angular_deflection,
    )
    tess_cache_dir = out_dir / "tess_cache"

    records = discover_fill_parts(
        args.fill_volume_root,
        assemblies=args.assemblies,
        max_file_mb=args.max_file_mb,
    )
    size_profile: list[PartSizeProfile] | None = None
    if args.size_quantile_min is not None or args.size_quantile_max is not None:
        validate_quantile_args(args.size_quantile_min, args.size_quantile_max)
        size_profile = build_size_profile(records, tess_cache_dir, tessellation)
        selected_profile = filter_size_profile_by_quantile(
            size_profile,
            args.size_metric,
            args.size_quantile_min,
            args.size_quantile_max,
        )
        if args.max_parts:
            selected_profile = take_evenly_spaced_by_size(selected_profile, args.max_parts, args.size_metric)
        records = records_from_profile(records, selected_profile)
        print_size_selection_summary(size_profile, selected_profile, args.size_metric)
    elif args.max_parts:
        records = records[: args.max_parts]
    if not records:
        raise SystemExit("No filled midsurface STEP files discovered.")

    split = build_split_manifest(
        records=records,
        val_count=args.val_count,
        val_fraction=args.val_fraction,
        strategy=args.split_strategy,
        split_seed=args.split_seed,
    )
    train_records = [records[i] for i in split.train_indices]
    val_records = [records[i] for i in split.val_indices]
    if not train_records:
        raise SystemExit("Train split is empty.")
    print_split_summary(split)

    train_dataset = MidsurfacePointCloudDataset(
        records=train_records,
        cache_dir=tess_cache_dir,
        n_points=args.n_points,
        seed=args.seed,
        tessellation=tessellation,
        use_joint_distance=args.use_joint_distance,
        boundary_sample_fraction=args.boundary_sample_fraction,
        use_boundary_feature=args.use_boundary_feature,
        scaffold_target_points=args.scaffold_target_points,
        scaffold_target_boundary_fraction=args.scaffold_target_boundary_fraction,
        scaffold_target_crease_fraction=args.scaffold_target_crease_fraction,
        scaffold_target_corner_fraction=args.scaffold_target_corner_fraction,
        mirror_axes=args.train_mirror_axes,
        resample_each_epoch=args.resample_train_each_epoch,
    )
    if args.resample_train_each_epoch:
        print("train_resampling: enabled per epoch for train split only; validation samples remain fixed")
    if train_dataset.augmentation_factor > 1:
        print(
            "train_augmentation: "
            f"mirror_axes={args.train_mirror_axes} "
            f"base_records={len(train_records)} samples={len(train_dataset)}"
        )
    train_source = preload_dataset(train_dataset, "train") if args.preload_data else Subset(
        train_dataset,
        list(range(len(train_dataset))),
    )
    train_loader = DataLoader(
        train_source,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0 and not args.resample_train_each_epoch,
    )
    val_dataset = None
    val_loader = None
    if val_records:
        val_dataset = MidsurfacePointCloudDataset(
            records=val_records,
            cache_dir=tess_cache_dir,
            n_points=args.n_points,
            seed=args.seed,
            tessellation=tessellation,
            use_joint_distance=args.use_joint_distance,
            boundary_sample_fraction=args.boundary_sample_fraction,
            use_boundary_feature=args.use_boundary_feature,
            scaffold_target_points=args.scaffold_target_points,
            scaffold_target_boundary_fraction=args.scaffold_target_boundary_fraction,
            scaffold_target_crease_fraction=args.scaffold_target_crease_fraction,
            scaffold_target_corner_fraction=args.scaffold_target_corner_fraction,
        )
        val_source = preload_dataset(val_dataset, "val") if args.preload_data else Subset(
            val_dataset,
            list(range(len(val_dataset))),
        )
        val_loader = DataLoader(
            val_source,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
        )

    model = build_model(args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    best_metric_name = resolve_best_metric(args.best_metric, has_validation=val_loader is not None)
    best_metric_mode = best_metric_mode_for(best_metric_name)
    extra_best = build_extra_best_trackers(args.extra_best_metrics, best_metric_name, val_loader is not None)

    history: list[dict[str, float]] = []
    best_metric_value = initial_best_metric_value(best_metric_mode)
    best_epoch = 0
    best_state = None
    start_epoch = 1
    if args.resume:
        assert resume_payload is not None
        model.load_state_dict(resume_payload["model_state"])
        if "optimizer_state" in resume_payload:
            opt.load_state_dict(resume_payload["optimizer_state"])
        history = list(resume_payload.get("history", []))
        best = resume_payload.get("best", {})
        best_metric_name = str(best.get("metric", best_metric_name))
        best_metric_mode = str(best.get("mode", best_metric_mode_for(best_metric_name)))
        best_metric_value = float(best.get("value", best_metric_value))
        best_epoch = int(best.get("epoch", best_epoch))
        extra_best = restore_extra_best_trackers(
            resume_payload.get("extra_best"),
            extra_best,
        )
        start_epoch = int(resume_payload.get("epoch", 0)) + 1
        print(
            "resume: "
            f"path={args.resume} start_epoch={start_epoch} "
            f"best_{best_metric_name}=epoch{best_epoch}:{best_metric_value:.6f}"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        if args.resample_train_each_epoch:
            train_dataset.set_resample_step(epoch)
        train_metrics = run_train_epoch(model, train_loader, opt, args, device)
        row = dict(train_metrics)
        row.update(prefix_metrics(train_metrics, "train"))
        row["epoch"] = epoch
        if should_run_validation(epoch, args.epochs, args.eval_every, val_loader is not None):
            assert val_loader is not None
            val_metrics = evaluate_loss(model, val_loader, args, device)
            row.update(prefix_metrics(val_metrics, "val"))
        epoch_state = None
        extra_best_updated = False
        if best_metric_name not in row:
            validate_missing_best_metric(best_metric_name, row, epoch)
        elif is_better_metric(row[best_metric_name], best_metric_value, best_metric_mode):
            best_metric_value = row[best_metric_name]
            best_epoch = epoch
            epoch_state = cpu_model_state(model)
            best_state = epoch_state
            save_training_checkpoint(
                out_dir / "best.pt",
                model_state=best_state,
                model=model,
                args=args,
                size_profile=size_profile,
                records=records,
                split=split,
                history=history + [row],
                best_epoch=best_epoch,
                best_metric_name=best_metric_name,
                best_metric_value=best_metric_value,
                best_metric_mode=best_metric_mode,
                extra_best=extra_best,
                epoch=epoch,
            )
        for metric_name, tracker in extra_best.items():
            if metric_name not in row:
                validate_missing_best_metric(metric_name, row, epoch)
                continue
            metric_value = row[metric_name]
            if not is_better_metric(metric_value, float(tracker["value"]), str(tracker["mode"])):
                continue
            if epoch_state is None:
                epoch_state = cpu_model_state(model)
            tracker["value"] = metric_value
            tracker["epoch"] = epoch
            extra_best_updated = True
            save_training_checkpoint(
                out_dir / checkpoint_name_for_best_metric(metric_name),
                model_state=epoch_state,
                model=model,
                args=args,
                size_profile=size_profile,
                records=records,
                split=split,
                history=history + [row],
                best_epoch=epoch,
                best_metric_name=metric_name,
                best_metric_value=metric_value,
                best_metric_mode=str(tracker["mode"]),
                extra_best=extra_best,
                epoch=epoch,
            )
        if extra_best_updated and best_state is not None:
            save_training_checkpoint(
                out_dir / "best.pt",
                model_state=best_state,
                model=model,
                args=args,
                size_profile=size_profile,
                records=records,
                split=split,
                history=history + [row],
                best_epoch=best_epoch,
                best_metric_name=best_metric_name,
                best_metric_value=best_metric_value,
                best_metric_mode=best_metric_mode,
                extra_best=extra_best,
                epoch=best_epoch,
            )
        history.append(row)
        if args.save_every > 0 and (epoch % args.save_every == 0 or epoch == args.epochs):
            save_training_checkpoint(
                out_dir / "last.pt",
                model_state=cpu_model_state(model),
                model=model,
                args=args,
                size_profile=size_profile,
                records=records,
                split=split,
                history=history,
                best_epoch=best_epoch,
                best_metric_name=best_metric_name,
                best_metric_value=best_metric_value,
                best_metric_mode=best_metric_mode,
                extra_best=extra_best,
                epoch=epoch,
                optimizer_state=opt.state_dict(),
            )
            (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(format_epoch_log(epoch, row, best_metric_name, best_metric_value, best_epoch))

    if best_state is not None:
        model.load_state_dict(best_state)
    elif (out_dir / "best.pt").exists():
        best_payload = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best_payload["model_state"])
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    write_reconstruction_preview(model, train_dataset, device, out_dir)
    print(f"wrote {out_dir / 'best.pt'}")


def resolve_best_metric(requested: str, has_validation: bool) -> str:
    if requested == "auto":
        return "val_loss" if has_validation else "train_loss"
    if requested.startswith("val_") and not has_validation:
        raise SystemExit(f"--best-metric {requested} requires a non-empty validation split.")
    return requested


def build_extra_best_trackers(
    requested_metrics: list[str],
    primary_metric: str,
    has_validation: bool,
) -> dict[str, dict[str, float | int | str]]:
    trackers: dict[str, dict[str, float | int | str]] = {}
    for requested in requested_metrics:
        metric_name = resolve_best_metric(requested, has_validation=has_validation)
        if metric_name == primary_metric or metric_name in trackers:
            continue
        mode = best_metric_mode_for(metric_name)
        trackers[metric_name] = {
            "mode": mode,
            "value": initial_best_metric_value(mode),
            "epoch": 0,
            "path": checkpoint_name_for_best_metric(metric_name),
        }
    return trackers


def restore_extra_best_trackers(
    saved: dict | None,
    requested: dict[str, dict[str, float | int | str]],
) -> dict[str, dict[str, float | int | str]]:
    if not saved:
        return requested
    restored = dict(requested)
    for metric_name, tracker in saved.items():
        if metric_name not in restored:
            continue
        current = dict(restored[metric_name])
        current.update(tracker)
        restored[metric_name] = current
    return restored


def checkpoint_name_for_best_metric(metric_name: str) -> str:
    safe_metric = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in metric_name)
    return f"best_by_{safe_metric}.pt"


def best_metric_mode_for(metric_name: str) -> str:
    """Return whether a checkpoint metric should be minimized or maximized."""

    if metric_name.endswith("_within_threshold"):
        return "max"
    return "min"


def initial_best_metric_value(mode: str) -> float:
    if mode == "max":
        return -float("inf")
    if mode == "min":
        return float("inf")
    raise ValueError(f"unsupported best metric mode: {mode}")


def is_better_metric(value: float, best_value: float, mode: str) -> bool:
    if mode == "max":
        return value > best_value
    if mode == "min":
        return value < best_value
    raise ValueError(f"unsupported best metric mode: {mode}")


def validate_missing_best_metric(metric_name: str, row: dict[str, float], epoch: int) -> None:
    if metric_name.startswith("val_"):
        return
    available = ", ".join(sorted(k for k in row if k != "epoch"))
    raise SystemExit(
        f"--best-metric {metric_name} is not available at epoch {epoch}. "
        f"Available metrics: {available}"
    )


def preload_dataset(dataset: MidsurfacePointCloudDataset, label: str) -> list[dict]:
    items = [dataset[i] for i in range(len(dataset))]
    print(f"preload_data: split={label} items={len(items)}")
    return items


def should_run_validation(epoch: int, max_epochs: int, eval_every: int, has_validation: bool) -> bool:
    if not has_validation:
        return False
    interval = max(1, eval_every)
    return epoch == 1 or epoch == max_epochs or epoch % interval == 0


def prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def run_train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals = zero_metric_totals()
    count = 0
    boundary_point_count = 0.0
    for batch in loader:
        loss, metrics, bsz = compute_loss(model, batch, args, device)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        boundary_point_count += accumulate_metric_totals(totals, metrics, bsz)
        count += bsz
    metrics = finalize_metric_totals(totals, count, boundary_point_count)
    add_composite_metrics(metrics, args)
    return metrics


def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = zero_metric_totals()
    count = 0
    boundary_point_count = 0.0
    with torch.no_grad():
        for batch in loader:
            _, metrics, bsz = compute_loss(model, batch, args, device)
            boundary_point_count += accumulate_metric_totals(totals, metrics, bsz)
            count += bsz
    metrics = finalize_metric_totals(totals, count, boundary_point_count)
    add_composite_metrics(metrics, args)
    return metrics


def accumulate_metric_totals(totals: dict[str, float], metrics: dict[str, float], batch_size: int) -> float:
    boundary_points = float(metrics.get("boundary_point_count", 0.0))
    for key, value in metrics.items():
        if key in BOUNDARY_WEIGHTED_METRICS:
            totals[key] += value * boundary_points
        elif key in SCAFFOLD_TARGET_WEIGHTED_METRICS:
            target_count = float(metrics.get(SCAFFOLD_TARGET_WEIGHTED_METRICS[key], 0.0))
            totals[key] += value * target_count
        elif key == "boundary_point_count":
            totals[key] += value
        elif key in SCAFFOLD_TARGET_COUNT_METRICS:
            totals[key] += value
        else:
            totals[key] += value * batch_size
    return boundary_points


def finalize_metric_totals(
    totals: dict[str, float],
    sample_count: int,
    boundary_point_count: float,
) -> dict[str, float]:
    averaged: dict[str, float] = {}
    for key, value in totals.items():
        if key in BOUNDARY_WEIGHTED_METRICS:
            averaged[key] = value / max(boundary_point_count, 1.0)
        elif key in SCAFFOLD_TARGET_WEIGHTED_METRICS:
            target_count = totals.get(SCAFFOLD_TARGET_WEIGHTED_METRICS[key], 0.0)
            averaged[key] = value / max(target_count, 1.0)
        elif key == "boundary_point_count":
            averaged[key] = value / max(sample_count, 1)
        elif key in SCAFFOLD_TARGET_COUNT_METRICS:
            averaged[key] = value / max(sample_count, 1)
        else:
            averaged[key] = value / max(sample_count, 1)
    return averaged


def add_composite_metrics(metrics: dict[str, float], args: argparse.Namespace) -> None:
    metrics["cae_score"] = composite_cae_score(metrics, args)


def composite_cae_score(metrics: dict[str, float], args: argparse.Namespace) -> float:
    """Single validation score for CAE-oriented checkpoint comparison.

    Lower is better. p95 and Chamfer terms are measured in world millimeters;
    within-threshold fractions are rewards.
    """

    chamfer_mm = metrics.get("recon_mean_mm", 0.0) + metrics.get("target_mean_mm", 0.0)
    score = (
        args.cae_score_target_p95_weight * metrics.get("target_p95_mm", 0.0)
        + getattr(args, "cae_score_recon_p95_weight", 0.0) * metrics.get("recon_p95_mm", 0.0)
        + args.cae_score_chamfer_weight * chamfer_mm
        - args.cae_score_target_within_weight * metrics.get("target_within_threshold", 0.0)
        - getattr(args, "cae_score_pred_within_weight", 0.0) * metrics.get("pred_within_threshold", 0.0)
    )
    if metrics.get("boundary_point_count", 0.0) > 0.0:
        score += (
            args.cae_score_boundary_p95_weight * metrics.get("boundary_p95_mm", 0.0)
            - args.cae_score_boundary_within_weight * metrics.get("boundary_within_threshold", 0.0)
        )
    return float(score)


def compute_loss(
    model: torch.nn.Module,
    batch: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float], int]:
    non_blocking = device.type == "cuda"
    points = batch["points"].to(device, non_blocking=non_blocking)
    normals = batch["normals"].to(device, non_blocking=non_blocking)
    features = batch["features"].to(device, non_blocking=non_blocking)
    scale = batch["scale"].to(device, non_blocking=non_blocking)
    scaffold_targets = batch.get("scaffold_targets")
    scaffold_target_mask = batch.get("scaffold_target_mask")
    if scaffold_targets is not None:
        scaffold_targets = scaffold_targets.to(device, non_blocking=non_blocking)
    if scaffold_target_mask is not None:
        scaffold_target_mask = scaffold_target_mask.to(device, non_blocking=non_blocking).bool()
    scaffold_boundary_targets = batch.get("scaffold_boundary_targets")
    scaffold_boundary_mask = batch.get("scaffold_boundary_target_mask")
    scaffold_crease_targets = batch.get("scaffold_crease_targets")
    scaffold_crease_mask = batch.get("scaffold_crease_target_mask")
    scaffold_corner_targets = batch.get("scaffold_corner_targets")
    scaffold_corner_mask = batch.get("scaffold_corner_target_mask")
    if scaffold_boundary_targets is not None:
        scaffold_boundary_targets = scaffold_boundary_targets.to(device, non_blocking=non_blocking)
        scaffold_boundary_mask = scaffold_boundary_mask.to(device, non_blocking=non_blocking).bool()
    if scaffold_crease_targets is not None:
        scaffold_crease_targets = scaffold_crease_targets.to(device, non_blocking=non_blocking)
        scaffold_crease_mask = scaffold_crease_mask.to(device, non_blocking=non_blocking).bool()
    if scaffold_corner_targets is not None:
        scaffold_corner_targets = scaffold_corner_targets.to(device, non_blocking=non_blocking)
        scaffold_corner_mask = scaffold_corner_mask.to(device, non_blocking=non_blocking).bool()
    boundary_mask = batch.get("boundary_mask")
    if boundary_mask is not None:
        boundary_mask = boundary_mask.to(device, non_blocking=non_blocking).bool()
    pred = model(points, features)
    loss_pred_to_target, loss_target_to_pred = chamfer_components(pred["points"], points)
    loss_ch = loss_pred_to_target + loss_target_to_pred
    loss_norm = normal_chamfer_loss(pred["points"], pred["normals"], points, normals)
    loss_sp = spread_loss(pred["points"], points)
    loss_cov = target_coverage_loss(
        pred["points"],
        points,
        scale,
        args.coverage_threshold_mm,
    )
    loss_pred_surface = prediction_surface_loss(
        pred["points"],
        points,
        scale,
        args.pred_surface_threshold_mm,
    )
    refinement_occupancy = refinement_occupancy_metrics(
        pred["points"],
        points,
        pred.get("refinement_logits"),
        scale,
        args.occupancy_positive_threshold_mm,
        args.occupancy_negative_threshold_mm,
    )
    loss_scaffold = torch.zeros((), device=device)
    feature_scaffold_metrics = zero_feature_scaffold_metrics(device, pred["points"])
    boundary_scaffold_metrics = zero_feature_scaffold_metrics(device, pred["points"])
    crease_scaffold_metrics = zero_feature_scaffold_metrics(device, pred["points"])
    corner_scaffold_metrics = zero_feature_scaffold_metrics(device, pred["points"])
    if "scaffold_points" in pred:
        loss_scaffold = scaffold_chamfer_loss(
            pred["scaffold_points"],
            points,
            pred.get("active_scaffold_mask"),
        )
        feature_scaffold_metrics = feature_scaffold_target_metrics(
            pred["scaffold_points"],
            scaffold_targets,
            pred.get("active_scaffold_mask"),
            scale,
            target_mask=scaffold_target_mask,
        )
        boundary_scaffold_metrics = feature_scaffold_target_metrics(
            pred["scaffold_points"],
            scaffold_boundary_targets,
            pred.get("active_scaffold_mask"),
            scale,
            target_mask=scaffold_boundary_mask,
        )
        crease_scaffold_metrics = feature_scaffold_target_metrics(
            pred["scaffold_points"],
            scaffold_crease_targets,
            pred.get("active_scaffold_mask"),
            scale,
            target_mask=scaffold_crease_mask,
        )
        corner_scaffold_metrics = feature_scaffold_target_metrics(
            pred["scaffold_points"],
            scaffold_corner_targets,
            pred.get("active_scaffold_mask"),
            scale,
            target_mask=scaffold_corner_mask,
        )
    coverage_fraction = target_coverage_fraction(
        pred["points"],
        points,
        scale,
        args.coverage_threshold_mm,
    )
    pred_surface_fraction = prediction_surface_fraction(
        pred["points"],
        points,
        scale,
        args.pred_surface_threshold_mm,
    )
    distance_metrics = nearest_distance_metrics_mm(pred["points"], points, scale)
    boundary_metrics = boundary_distance_metrics(
        pred["points"],
        points,
        boundary_mask,
        scale,
        args.boundary_threshold_mm,
    )
    loss = (
        loss_ch
        + args.lambda_spread * loss_sp
        + args.lambda_normal * loss_norm
        + args.lambda_target_coverage * loss_cov
        + args.lambda_pred_surface * loss_pred_surface
        + args.lambda_refinement_occupancy * refinement_occupancy["loss"]
        + args.lambda_scaffold * loss_scaffold
        + args.lambda_feature_scaffold * feature_scaffold_metrics["loss"]
        + args.lambda_boundary_scaffold * boundary_scaffold_metrics["loss"]
        + args.lambda_crease_scaffold * crease_scaffold_metrics["loss"]
        + args.lambda_corner_scaffold * corner_scaffold_metrics["loss"]
        + args.lambda_boundary_coverage * boundary_metrics["coverage_loss"]
    )
    metrics = {
        "loss": float(loss.detach().cpu()),
        "chamfer": float(loss_ch.detach().cpu()),
        "pred_to_target": float(loss_pred_to_target.detach().cpu()),
        "target_to_pred": float(loss_target_to_pred.detach().cpu()),
        "normal": float(loss_norm.detach().cpu()),
        "spread": float(loss_sp.detach().cpu()),
        "coverage": float(loss_cov.detach().cpu()),
        "target_within_threshold": float(coverage_fraction.detach().cpu()),
        "pred_surface": float(loss_pred_surface.detach().cpu()),
        "pred_within_threshold": float(pred_surface_fraction.detach().cpu()),
        "refinement_occupancy": float(refinement_occupancy["loss"].detach().cpu()),
        "refinement_occupancy_labeled_fraction": float(refinement_occupancy["labeled_fraction"].detach().cpu()),
        "refinement_occupancy_positive_fraction": float(refinement_occupancy["positive_fraction"].detach().cpu()),
        "refinement_occupancy_active_fraction": float(refinement_occupancy["active_fraction"].detach().cpu()),
        "refinement_occupancy_accuracy": float(refinement_occupancy["accuracy"].detach().cpu()),
        "recon_mean_mm": distance_metrics["recon_mean_mm"],
        "recon_p95_mm": distance_metrics["recon_p95_mm"],
        "target_mean_mm": distance_metrics["target_mean_mm"],
        "target_p95_mm": distance_metrics["target_p95_mm"],
        "boundary_coverage": float(boundary_metrics["coverage_loss"].detach().cpu()),
        "boundary_within_threshold": float(boundary_metrics["within_threshold"].detach().cpu()),
        "boundary_p95_mm": float(boundary_metrics["p95_mm"].detach().cpu()),
        "boundary_point_count": float(boundary_metrics["point_count"].detach().cpu()),
        "scaffold": float(loss_scaffold.detach().cpu()),
        "feature_scaffold": float(feature_scaffold_metrics["loss"].detach().cpu()),
        "feature_scaffold_mean_mm": float(feature_scaffold_metrics["mean_mm"].detach().cpu()),
        "feature_scaffold_p95_mm": float(feature_scaffold_metrics["p95_mm"].detach().cpu()),
        "feature_scaffold_target_count": float(feature_scaffold_metrics["target_count"].detach().cpu()),
        "boundary_scaffold": float(boundary_scaffold_metrics["loss"].detach().cpu()),
        "boundary_scaffold_mean_mm": float(boundary_scaffold_metrics["mean_mm"].detach().cpu()),
        "boundary_scaffold_p95_mm": float(boundary_scaffold_metrics["p95_mm"].detach().cpu()),
        "boundary_scaffold_target_count": float(boundary_scaffold_metrics["target_count"].detach().cpu()),
        "crease_scaffold": float(crease_scaffold_metrics["loss"].detach().cpu()),
        "crease_scaffold_mean_mm": float(crease_scaffold_metrics["mean_mm"].detach().cpu()),
        "crease_scaffold_p95_mm": float(crease_scaffold_metrics["p95_mm"].detach().cpu()),
        "crease_scaffold_target_count": float(crease_scaffold_metrics["target_count"].detach().cpu()),
        "corner_scaffold": float(corner_scaffold_metrics["loss"].detach().cpu()),
        "corner_scaffold_mean_mm": float(corner_scaffold_metrics["mean_mm"].detach().cpu()),
        "corner_scaffold_p95_mm": float(corner_scaffold_metrics["p95_mm"].detach().cpu()),
        "corner_scaffold_target_count": float(corner_scaffold_metrics["target_count"].detach().cpu()),
    }
    return loss, metrics, int(points.shape[0])


def nearest_distance_metrics_mm(
    pred: torch.Tensor,
    target: torch.Tensor,
    scale_mm: torch.Tensor,
) -> dict[str, float]:
    """CAE-oriented nearest-distance summaries in world millimeters."""

    dist = torch.cdist(pred, target)
    scale = scale_mm.reshape(-1, 1)
    recon_to_target_mm = dist.min(dim=2).values * scale
    target_to_recon_mm = dist.min(dim=1).values * scale
    return {
        "recon_mean_mm": float(recon_to_target_mm.mean().detach().cpu()),
        "recon_p95_mm": float(torch.quantile(recon_to_target_mm.reshape(-1), 0.95).detach().cpu()),
        "target_mean_mm": float(target_to_recon_mm.mean().detach().cpu()),
        "target_p95_mm": float(torch.quantile(target_to_recon_mm.reshape(-1), 0.95).detach().cpu()),
    }


def boundary_distance_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    boundary_mask: torch.Tensor | None,
    scale_mm: torch.Tensor,
    threshold_mm: float,
) -> dict[str, torch.Tensor]:
    """Boundary target-to-reconstruction metrics in normalized loss and world mm."""

    zero = pred.sum() * 0.0
    if boundary_mask is None or not bool(boundary_mask.any()):
        return {
            "coverage_loss": zero,
            "within_threshold": zero,
            "p95_mm": zero,
            "point_count": zero,
        }
    losses: list[torch.Tensor] = []
    within: list[torch.Tensor] = []
    dist_mm: list[torch.Tensor] = []
    point_count = 0
    for batch_idx in range(pred.shape[0]):
        mask = boundary_mask[batch_idx]
        if not bool(mask.any()):
            continue
        boundary_target = target[batch_idx][mask]
        dist = torch.cdist(boundary_target[None, :, :], pred[batch_idx : batch_idx + 1])[0].min(dim=1).values
        threshold = threshold_mm / scale_mm[batch_idx].clamp_min(1.0e-6)
        losses.append(torch.relu(dist - threshold))
        within.append((dist <= threshold).float())
        dist_mm.append(dist * scale_mm[batch_idx])
        point_count += int(boundary_target.shape[0])
    if not losses:
        return {
            "coverage_loss": zero,
            "within_threshold": zero,
            "p95_mm": zero,
            "point_count": zero,
        }
    loss_values = torch.cat(losses)
    within_values = torch.cat(within)
    dist_mm_values = torch.cat(dist_mm)
    return {
        "coverage_loss": loss_values.mean(),
        "within_threshold": within_values.mean(),
        "p95_mm": torch.quantile(dist_mm_values, 0.95),
        "point_count": torch.tensor(float(point_count), device=pred.device),
    }


def zero_feature_scaffold_metrics(device: torch.device, reference: torch.Tensor) -> dict[str, torch.Tensor]:
    zero = reference.sum() * 0.0
    return {
        "loss": zero,
        "mean_mm": zero,
        "p95_mm": zero,
        "target_count": torch.zeros((), device=device),
    }


def feature_scaffold_target_metrics(
    scaffold: torch.Tensor,
    targets: torch.Tensor | None,
    active_mask: torch.Tensor | None,
    scale_mm: torch.Tensor,
    target_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """One-way feature-target-to-scaffold supervision for scaffold placement."""

    if targets is None or targets.shape[1] == 0:
        return zero_feature_scaffold_metrics(scaffold.device, scaffold)
    losses: list[torch.Tensor] = []
    dist_mm: list[torch.Tensor] = []
    target_count = 0
    for batch_idx in range(scaffold.shape[0]):
        scaffold_b = scaffold[batch_idx]
        if active_mask is not None:
            scaffold_b = scaffold_b[active_mask[batch_idx].bool()]
        if scaffold_b.shape[0] == 0:
            continue
        target_b = targets[batch_idx]
        if target_mask is not None:
            target_b = target_b[target_mask[batch_idx].bool()]
        if target_b.shape[0] == 0:
            continue
        dist = torch.cdist(target_b[None, :, :], scaffold_b[None, :, :])[0].min(dim=1).values
        losses.append(dist)
        dist_mm.append(dist * scale_mm[batch_idx])
        target_count += int(target_b.shape[0])
    if not losses:
        return zero_feature_scaffold_metrics(scaffold.device, scaffold)
    dist_values = torch.cat(losses)
    dist_mm_values = torch.cat(dist_mm)
    return {
        "loss": dist_values.mean(),
        "mean_mm": dist_mm_values.mean(),
        "p95_mm": torch.quantile(dist_mm_values, 0.95),
        "target_count": torch.tensor(float(target_count), device=scaffold.device),
    }


def zero_refinement_occupancy_metrics(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    zero = reference.sum() * 0.0
    return {
        "loss": zero,
        "labeled_fraction": zero,
        "positive_fraction": zero,
        "active_fraction": zero,
        "accuracy": zero,
    }


def refinement_occupancy_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    logits: torch.Tensor | None,
    scale_mm: torch.Tensor,
    positive_threshold_mm: float,
    negative_threshold_mm: float,
) -> dict[str, torch.Tensor]:
    """Distance-derived weak supervision for point-level refinement activity logits."""

    if logits is None:
        return zero_refinement_occupancy_metrics(pred)
    if negative_threshold_mm < positive_threshold_mm:
        raise ValueError("occupancy_negative_threshold_mm must be >= occupancy_positive_threshold_mm")
    logits_flat = logits.reshape(logits.shape[0], -1)[:, : pred.shape[1]]
    dist = torch.cdist(pred, target).min(dim=2).values * scale_mm.reshape(-1, 1)
    positive = dist <= positive_threshold_mm
    negative = dist >= negative_threshold_mm
    labeled = positive | negative
    active_fraction = (torch.sigmoid(logits_flat) >= 0.5).float().mean()
    if not bool(labeled.any()):
        out = zero_refinement_occupancy_metrics(pred)
        out["active_fraction"] = active_fraction
        return out
    labels = positive.float()
    selected_logits = logits_flat[labeled]
    selected_labels = labels[labeled]
    predicted_labels = (torch.sigmoid(selected_logits) >= 0.5).float()
    return {
        "loss": F.binary_cross_entropy_with_logits(selected_logits, selected_labels),
        "labeled_fraction": labeled.float().mean(),
        "positive_fraction": selected_labels.mean(),
        "active_fraction": active_fraction,
        "accuracy": (predicted_labels == selected_labels).float().mean(),
    }


def zero_metric_totals() -> dict[str, float]:
    return {
        "loss": 0.0,
        "chamfer": 0.0,
        "pred_to_target": 0.0,
        "target_to_pred": 0.0,
        "normal": 0.0,
        "spread": 0.0,
        "coverage": 0.0,
        "target_within_threshold": 0.0,
        "pred_surface": 0.0,
        "pred_within_threshold": 0.0,
        "refinement_occupancy": 0.0,
        "refinement_occupancy_labeled_fraction": 0.0,
        "refinement_occupancy_positive_fraction": 0.0,
        "refinement_occupancy_active_fraction": 0.0,
        "refinement_occupancy_accuracy": 0.0,
        "recon_mean_mm": 0.0,
        "recon_p95_mm": 0.0,
        "target_mean_mm": 0.0,
        "target_p95_mm": 0.0,
        "boundary_coverage": 0.0,
        "boundary_within_threshold": 0.0,
        "boundary_p95_mm": 0.0,
        "boundary_point_count": 0.0,
        "scaffold": 0.0,
        "feature_scaffold": 0.0,
        "feature_scaffold_mean_mm": 0.0,
        "feature_scaffold_p95_mm": 0.0,
        "feature_scaffold_target_count": 0.0,
        "boundary_scaffold": 0.0,
        "boundary_scaffold_mean_mm": 0.0,
        "boundary_scaffold_p95_mm": 0.0,
        "boundary_scaffold_target_count": 0.0,
        "crease_scaffold": 0.0,
        "crease_scaffold_mean_mm": 0.0,
        "crease_scaffold_p95_mm": 0.0,
        "crease_scaffold_target_count": 0.0,
        "corner_scaffold": 0.0,
        "corner_scaffold_mean_mm": 0.0,
        "corner_scaffold_p95_mm": 0.0,
        "corner_scaffold_target_count": 0.0,
        "cae_score": 0.0,
    }


def format_epoch_log(
    epoch: int,
    row: dict[str, float],
    best_metric_name: str,
    best_metric_value: float,
    best_epoch: int,
) -> str:
    parts = [
        f"epoch={epoch:04d}",
        f"train_loss={row['train_loss']:.6f}",
        f"train_chamfer={row['train_chamfer']:.6f}",
        f"train_cov={row['train_coverage']:.6f}",
        f"train_tgt_p95={row['train_target_p95_mm']:.3f}mm",
        f"train_tgt5={row['train_target_within_threshold']:.3f}",
        f"train_pred5={row['train_pred_within_threshold']:.3f}",
        f"train_scaffold={row['train_scaffold']:.6f}",
        f"train_cae={row['train_cae_score']:.3f}",
    ]
    if row.get("train_feature_scaffold_target_count", 0.0) > 0:
        parts.append(f"train_feat_scaf_p95={row['train_feature_scaffold_p95_mm']:.3f}mm")
    if row.get("train_boundary_scaffold_target_count", 0.0) > 0:
        parts.append(f"train_bnd_scaf_p95={row['train_boundary_scaffold_p95_mm']:.3f}mm")
    if row.get("train_refinement_occupancy_labeled_fraction", 0.0) > 0:
        parts.append(
            "train_occ="
            f"{row['train_refinement_occupancy']:.6f}/"
            f"{row['train_refinement_occupancy_active_fraction']:.3f}"
        )
    if row.get("train_boundary_point_count", 0.0) > 0:
        parts.extend(
            [
                f"train_bnd_p95={row['train_boundary_p95_mm']:.3f}mm",
                f"train_bnd5={row['train_boundary_within_threshold']:.3f}",
            ]
        )
    if "val_loss" in row:
        parts.extend(
            [
                f"val_loss={row['val_loss']:.6f}",
                f"val_chamfer={row['val_chamfer']:.6f}",
                f"val_cov={row['val_coverage']:.6f}",
                f"val_tgt_p95={row['val_target_p95_mm']:.3f}mm",
                f"val_tgt5={row['val_target_within_threshold']:.3f}",
                f"val_pred5={row['val_pred_within_threshold']:.3f}",
                f"val_scaffold={row['val_scaffold']:.6f}",
                f"val_cae={row['val_cae_score']:.3f}",
            ]
        )
        if row.get("val_feature_scaffold_target_count", 0.0) > 0:
            parts.append(f"val_feat_scaf_p95={row['val_feature_scaffold_p95_mm']:.3f}mm")
        if row.get("val_boundary_scaffold_target_count", 0.0) > 0:
            parts.append(f"val_bnd_scaf_p95={row['val_boundary_scaffold_p95_mm']:.3f}mm")
        if row.get("val_refinement_occupancy_labeled_fraction", 0.0) > 0:
            parts.append(
                "val_occ="
                f"{row['val_refinement_occupancy']:.6f}/"
                f"{row['val_refinement_occupancy_active_fraction']:.3f}"
            )
        if row.get("val_boundary_point_count", 0.0) > 0:
            parts.extend(
                [
                    f"val_bnd_p95={row['val_boundary_p95_mm']:.3f}mm",
                    f"val_bnd5={row['val_boundary_within_threshold']:.3f}",
                ]
            )
    parts.append(f"best_{best_metric_name}=epoch{best_epoch}:{best_metric_value:.6f}")
    return " ".join(parts)


def cpu_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def save_training_checkpoint(
    path: Path,
    model_state: dict[str, torch.Tensor],
    model: torch.nn.Module,
    args: argparse.Namespace,
    size_profile: list[PartSizeProfile] | None,
    records: list[PartRecord],
    split: SplitManifest,
    history: list[dict[str, float]],
    best_epoch: int,
    best_metric_name: str,
    best_metric_value: float,
    best_metric_mode: str,
    extra_best: dict[str, dict[str, float | int | str]],
    epoch: int,
    optimizer_state: dict | None = None,
) -> None:
    payload = {
        "model_state": model_state,
        "args": vars(args),
        "size_profile": [asdict(p) for p in size_profile] if size_profile is not None else None,
        "record_fingerprints": fingerprint_records(records),
        "split": asdict(split),
        "records": [
            {
                "assembly_id": r.assembly_id,
                "part_id_raw": r.part_id_raw,
                "canonical_part_id": r.canonical_part_id,
                "stp_path": str(r.stp_path),
            }
            for r in records
        ],
        "history": history,
        "model_config": model_config_from_model(model),
        "best": {
            "epoch": best_epoch,
            "metric": best_metric_name,
            "mode": best_metric_mode,
            "value": best_metric_value,
        },
        "extra_best": extra_best,
        "epoch": epoch,
    }
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    atomic_torch_save(payload, path)


def atomic_torch_save(payload: dict, path: Path, retries: int = 5, delay_seconds: float = 0.5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    last_error: OSError | RuntimeError | None = None
    for _ in range(retries):
        try:
            os.replace(tmp_path, path)
            return
        except (OSError, RuntimeError) as exc:
            last_error = exc
            time.sleep(delay_seconds)
    if tmp_path.exists():
        tmp_path.unlink(missing_ok=True)
    raise RuntimeError(f"Could not replace checkpoint {path}") from last_error


def validate_quantile_args(q_min: float | None, q_max: float | None) -> None:
    if q_min is None or q_max is None:
        raise SystemExit("--size-quantile-min and --size-quantile-max must be provided together.")
    if not (0.0 <= q_min < q_max <= 1.0):
        raise SystemExit("size quantiles must satisfy 0.0 <= min < max <= 1.0.")


def build_size_profile(
    records: list[PartRecord],
    cache_dir: Path,
    tessellation: TessellationConfig,
) -> list[PartSizeProfile]:
    profile: list[PartSizeProfile] = []
    for record in records:
        mesh = load_or_tessellate(record.stp_path, cache_dir, tessellation)
        extents = mesh.bounds_max - mesh.bounds_min
        profile.append(
            PartSizeProfile(
                canonical_part_id=record.canonical_part_id,
                stp_path=str(record.stp_path),
                bbox_diagonal=float(np.linalg.norm(extents)),
                bbox_max_extent=float(np.max(extents)),
            )
        )
    return profile


def filter_size_profile_by_quantile(
    profile: list[PartSizeProfile],
    metric: str,
    q_min: float,
    q_max: float,
) -> list[PartSizeProfile]:
    if not profile:
        return []
    values = np.asarray([getattr(p, metric) for p in profile], dtype=np.float64)
    low, high = np.quantile(values, [q_min, q_max])
    selected = [p for p in profile if low <= getattr(p, metric) <= high]
    return sorted(selected, key=lambda p: (getattr(p, metric), p.canonical_part_id))


def take_evenly_spaced_by_size(
    profile: list[PartSizeProfile],
    max_parts: int,
    metric: str,
) -> list[PartSizeProfile]:
    if max_parts <= 0 or len(profile) <= max_parts:
        return profile
    ordered = sorted(profile, key=lambda p: (getattr(p, metric), p.canonical_part_id))
    if max_parts == 1:
        return [ordered[len(ordered) // 2]]
    indices = np.linspace(0, len(ordered) - 1, max_parts).round().astype(int)
    return [ordered[int(i)] for i in indices]


def records_from_profile(records: list[PartRecord], profile: list[PartSizeProfile]) -> list[PartRecord]:
    by_id = {record.canonical_part_id: record for record in records}
    return [by_id[p.canonical_part_id] for p in profile if p.canonical_part_id in by_id]


def print_size_selection_summary(
    full_profile: list[PartSizeProfile],
    selected_profile: list[PartSizeProfile],
    metric: str,
) -> None:
    if not full_profile:
        print("size_filter: no records available")
        return
    values = np.asarray([getattr(p, metric) for p in full_profile], dtype=np.float64)
    selected_values = np.asarray([getattr(p, metric) for p in selected_profile], dtype=np.float64)
    print(
        "size_filter: "
        f"metric={metric} selected={len(selected_profile)}/{len(full_profile)} "
        f"all_min={values.min():.3f} all_p20={np.quantile(values, 0.2):.3f} "
        f"all_p40={np.quantile(values, 0.4):.3f} all_max={values.max():.3f}"
    )
    if len(selected_values):
        print(
            "size_filter_selected: "
            f"min={selected_values.min():.3f} max={selected_values.max():.3f} "
            f"ids={[p.canonical_part_id for p in selected_profile]}"
        )


def build_split_manifest(
    records: list[PartRecord],
    val_count: int,
    val_fraction: float,
    strategy: str,
    split_seed: int,
) -> SplitManifest:
    train_indices, val_indices = split_indices(
        n_items=len(records),
        val_count=val_count,
        val_fraction=val_fraction,
        strategy=strategy,
        split_seed=split_seed,
    )
    return SplitManifest(
        strategy=strategy,
        split_seed=split_seed,
        train_indices=train_indices,
        val_indices=val_indices,
        train_ids=[records[i].canonical_part_id for i in train_indices],
        val_ids=[records[i].canonical_part_id for i in val_indices],
    )


def split_indices(
    n_items: int,
    val_count: int = 0,
    val_fraction: float = 0.0,
    strategy: str = "even",
    split_seed: int = 13,
) -> tuple[list[int], list[int]]:
    if n_items <= 0:
        return [], []
    if val_count < 0:
        raise SystemExit("--val-count must be non-negative.")
    if not (0.0 <= val_fraction < 1.0):
        raise SystemExit("--val-fraction must satisfy 0.0 <= val_fraction < 1.0.")
    requested_val = val_count
    if requested_val == 0 and val_fraction > 0.0:
        requested_val = int(np.ceil(n_items * val_fraction))
    requested_val = min(requested_val, n_items - 1)
    if requested_val <= 0:
        return list(range(n_items)), []

    if strategy == "even":
        val_indices = unique_even_indices(n_items, requested_val)
    elif strategy == "random":
        rng = np.random.default_rng(split_seed)
        val_indices = sorted(rng.choice(n_items, size=requested_val, replace=False).astype(int).tolist())
    else:
        raise SystemExit(f"Unsupported split strategy: {strategy}")
    val_set = set(val_indices)
    train_indices = [i for i in range(n_items) if i not in val_set]
    return train_indices, val_indices


def unique_even_indices(n_items: int, count: int) -> list[int]:
    if count <= 0:
        return []
    raw = np.linspace(0, n_items - 1, count).round().astype(int).tolist()
    selected: list[int] = []
    seen: set[int] = set()
    for idx in raw:
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)
    if len(selected) < count:
        for idx in range(n_items):
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)
            if len(selected) == count:
                break
    return sorted(selected)


def print_split_summary(split: SplitManifest) -> None:
    print(
        "split: "
        f"strategy={split.strategy} train={len(split.train_indices)} val={len(split.val_indices)} "
        f"train_ids={split.train_ids} val_ids={split.val_ids}"
    )


def apply_resume_model_args(args: argparse.Namespace, payload: dict) -> None:
    """Restore checkpoint run arguments before constructing a resumed model."""

    ckpt_args = payload.get("args", {})
    model_config = payload.get("model_config", {})
    for key in (
        "fill_volume_root",
        "assemblies",
        "max_file_mb",
        "max_parts",
        "size_quantile_min",
        "size_quantile_max",
        "size_metric",
        "val_count",
        "val_fraction",
        "split_strategy",
        "split_seed",
        "model_kind",
        "use_joint_distance",
        "use_boundary_feature",
        "n_points",
        "batch_size",
        "resample_train_each_epoch",
        "token_dim",
        "n_coarse",
        "n_patches",
        "k_neighbors",
        "n_latents",
        "n_scaffold",
        "points_per_scaffold",
        "n_local_tokens",
        "train_mirror_axes",
        "boundary_token_count",
        "lattice_layers",
        "lattice_heads",
        "refinement_mode",
        "tangent_offset_scale",
        "normal_offset_scale",
        "patch_type_count",
        "lambda_spread",
        "lambda_normal",
        "lambda_target_coverage",
        "coverage_threshold_mm",
        "lambda_pred_surface",
        "pred_surface_threshold_mm",
        "lambda_refinement_occupancy",
        "occupancy_positive_threshold_mm",
        "occupancy_negative_threshold_mm",
        "lambda_scaffold",
        "lambda_feature_scaffold",
        "lambda_boundary_scaffold",
        "lambda_crease_scaffold",
        "lambda_corner_scaffold",
        "scaffold_target_points",
        "scaffold_target_boundary_fraction",
        "scaffold_target_crease_fraction",
        "scaffold_target_corner_fraction",
        "boundary_sample_fraction",
        "lambda_boundary_coverage",
        "boundary_threshold_mm",
        "cae_score_target_p95_weight",
        "cae_score_recon_p95_weight",
        "cae_score_boundary_p95_weight",
        "cae_score_chamfer_weight",
        "cae_score_target_within_weight",
        "cae_score_pred_within_weight",
        "cae_score_boundary_within_weight",
        "linear_deflection",
        "angular_deflection",
        "seed",
    ):
        if key in ckpt_args:
            setattr(args, key, ckpt_args[key])
    if "feature_dim" in model_config and int(model_config["feature_dim"]) >= 8:
        args.use_boundary_feature = True
    config_to_arg = {
        "n_points_out": "n_points",
        "n_scaffold": "n_scaffold",
        "points_per_scaffold": "points_per_scaffold",
        "boundary_token_count": "boundary_token_count",
        "lattice_layers": "lattice_layers",
        "lattice_heads": "lattice_heads",
        "refinement_mode": "refinement_mode",
        "tangent_offset_scale": "tangent_offset_scale",
        "normal_offset_scale": "normal_offset_scale",
        "patch_type_count": "patch_type_count",
        "scaffold_mode": "scaffold_mode",
        "scaffold_anchor_source": "scaffold_anchor_source",
        "scaffold_anchor_residual_scale": "scaffold_anchor_residual_scale",
    }
    for config_key, arg_key in config_to_arg.items():
        if config_key in model_config:
            setattr(args, arg_key, model_config[config_key])


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    feature_dim = feature_dim_from_args(args)
    boundary_feature_index = boundary_feature_index_from_args(args)
    if args.model_kind == "point":
        return HierarchicalMidsurfaceAutoencoder(
            feature_dim=feature_dim,
            token_dim=args.token_dim,
            n_points_out=args.n_points,
            n_coarse=args.n_coarse,
            n_patches=args.n_patches,
            k_neighbors=args.k_neighbors,
            n_latents=args.n_latents,
            boundary_feature_index=boundary_feature_index,
            boundary_token_count=args.boundary_token_count,
        )
    if args.model_kind == "structured":
        return StructuredScaffoldAutoencoder(
            feature_dim=feature_dim,
            token_dim=args.token_dim,
            n_points_out=args.n_points,
            n_coarse=args.n_coarse,
            n_patches=args.n_patches,
            k_neighbors=args.k_neighbors,
            n_latents=args.n_latents,
            n_scaffold=args.n_scaffold,
            points_per_scaffold=args.points_per_scaffold,
            n_local_tokens=args.n_local_tokens,
            boundary_feature_index=boundary_feature_index,
            boundary_token_count=args.boundary_token_count,
            lattice_layers=args.lattice_layers,
            lattice_heads=args.lattice_heads,
            refinement_mode=args.refinement_mode,
            tangent_offset_scale=args.tangent_offset_scale,
            normal_offset_scale=args.normal_offset_scale,
            patch_type_count=args.patch_type_count,
            scaffold_mode=args.scaffold_mode,
            scaffold_anchor_source=args.scaffold_anchor_source,
            scaffold_anchor_residual_scale=args.scaffold_anchor_residual_scale,
        )
    raise SystemExit(f"Unsupported model kind: {args.model_kind}")


def feature_dim_from_args(args: argparse.Namespace) -> int:
    return 8 if args.use_boundary_feature else 7


def boundary_feature_index_from_args(args: argparse.Namespace) -> int | None:
    return 7 if args.use_boundary_feature else None


def model_config_from_model(model: torch.nn.Module) -> dict[str, int | float | str]:
    config: dict[str, int | float | str] = {"class": type(model).__name__}
    if hasattr(model, "feature_dim"):
        config["feature_dim"] = int(getattr(model, "feature_dim"))
    if hasattr(model, "boundary_feature_index") and getattr(model, "boundary_feature_index") is not None:
        config["boundary_feature_index"] = int(getattr(model, "boundary_feature_index"))
    if hasattr(model, "boundary_token_count"):
        config["boundary_token_count"] = int(getattr(model, "boundary_token_count"))
    for name in ("n_points_out", "n_scaffold", "points_per_scaffold", "n_generated_points"):
        if hasattr(model, name):
            config[name] = int(getattr(model, name))
    if hasattr(model, "n_lattice_layers"):
        config["lattice_layers"] = int(getattr(model, "n_lattice_layers"))
    if hasattr(model, "lattice_heads"):
        config["lattice_heads"] = int(getattr(model, "lattice_heads"))
    if hasattr(model, "refinement_mode"):
        config["refinement_mode"] = str(getattr(model, "refinement_mode"))
    if hasattr(model, "tangent_offset_scale"):
        config["tangent_offset_scale"] = float(getattr(model, "tangent_offset_scale"))
    if hasattr(model, "normal_offset_scale"):
        config["normal_offset_scale"] = float(getattr(model, "normal_offset_scale"))
    if hasattr(model, "patch_type_count"):
        config["patch_type_count"] = int(getattr(model, "patch_type_count"))
    if hasattr(model, "scaffold_mode"):
        config["scaffold_mode"] = str(getattr(model, "scaffold_mode"))
    if hasattr(model, "scaffold_anchor_source"):
        config["scaffold_anchor_source"] = str(getattr(model, "scaffold_anchor_source"))
    if hasattr(model, "scaffold_anchor_residual_scale"):
        config["scaffold_anchor_residual_scale"] = float(getattr(model, "scaffold_anchor_residual_scale"))
    return config


def validate_boundary_args(args: argparse.Namespace) -> None:
    if args.occupancy_negative_threshold_mm < args.occupancy_positive_threshold_mm:
        raise SystemExit("--occupancy-negative-threshold-mm must be >= --occupancy-positive-threshold-mm")
    if args.refinement_mode != "tangent" and (
        args.tangent_offset_scale != 0.20 or args.normal_offset_scale != 0.02
    ):
        print(
            "[warn] tangent/normal offset scales only affect --refinement-mode tangent.",
            file=sys.stderr,
        )
    requested_metrics = [args.best_metric, *args.extra_best_metrics]
    boundary_sample_metrics = {
        "train_boundary_coverage",
        "val_boundary_coverage",
        "train_boundary_within_threshold",
        "val_boundary_within_threshold",
        "train_boundary_p95_mm",
        "val_boundary_p95_mm",
    }
    uses_boundary_metric = any(metric in boundary_sample_metrics for metric in requested_metrics)
    if args.boundary_sample_fraction <= 0.0:
        if args.use_boundary_feature:
            print(
                "[warn] --use-boundary-feature has no boundary-marked input points while "
                "--boundary-sample-fraction is 0.",
                file=sys.stderr,
            )
        if args.lambda_boundary_coverage > 0.0:
            print(
                "[warn] --lambda-boundary-coverage has no effect while --boundary-sample-fraction is 0.",
                file=sys.stderr,
            )
        if uses_boundary_metric:
            print(
                "[warn] boundary best metrics are constant while --boundary-sample-fraction is 0.",
                file=sys.stderr,
            )


def write_reconstruction_preview(
    model: torch.nn.Module,
    dataset: MidsurfacePointCloudDataset,
    device: torch.device,
    out_dir: Path,
) -> None:
    model.eval()
    with torch.no_grad():
        item = dataset[0]
        points = item["points"][None, ...].to(device)
        features = item["features"][None, ...].to(device)
        pred = model(points, features)["points"][0].cpu()
    target = item["points"]
    write_ply(out_dir / "preview_target.ply", target)
    write_ply(out_dir / "preview_recon.ply", pred)


def write_ply(path: Path, points: torch.Tensor) -> None:
    pts = points.detach().cpu().numpy()
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(pts)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    lines.extend(f"{x:.7f} {y:.7f} {z:.7f}" for x, y, z in pts)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fill-volume-root",
        default=r"C:\Users\hide2\IdeaBox\fill_volume",
    )
    parser.add_argument("--assemblies", nargs="*", default=["A0072600002_AllCATPart", "A0072601285_AllCATPart"])
    parser.add_argument("--max-file-mb", type=float, default=1.0)
    parser.add_argument("--max-parts", type=int, default=6)
    parser.add_argument("--size-quantile-min", type=float, default=None)
    parser.add_argument("--size-quantile-max", type=float, default=None)
    parser.add_argument("--size-metric", choices=["bbox_diagonal", "bbox_max_extent"], default="bbox_diagonal")
    parser.add_argument("--val-count", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--split-strategy", choices=["even", "random"], default="even")
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--best-metric",
        choices=BEST_METRIC_CHOICES,
        default="auto",
        help=(
            "Checkpoint selection metric. target_within_threshold is maximized; "
            "all other metrics are minimized."
        ),
    )
    parser.add_argument(
        "--extra-best-metrics",
        nargs="*",
        choices=BEST_METRIC_CHOICES,
        default=[],
        help="Also save best_by_<metric>.pt snapshots for these metrics.",
    )
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-dir", default="runs/cae_mesh_ae")
    parser.add_argument("--n-points", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preload-data", action="store_true")
    parser.add_argument(
        "--resample-train-each-epoch",
        action="store_true",
        help="Resample train-only surface/boundary/scaffold points every epoch. Validation remains fixed.",
    )
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--model-kind", choices=["point", "structured"], default="point")
    parser.add_argument("--token-dim", type=int, default=96)
    parser.add_argument("--n-coarse", type=int, default=96)
    parser.add_argument("--n-patches", type=int, default=48)
    parser.add_argument("--k-neighbors", type=int, default=24)
    parser.add_argument("--n-latents", type=int, default=48)
    parser.add_argument("--n-scaffold", type=int, default=64)
    parser.add_argument("--points-per-scaffold", type=int, default=None)
    parser.add_argument("--n-local-tokens", type=int, default=8)
    parser.add_argument(
        "--scaffold-mode",
        choices=["learned", "anchored"],
        default="learned",
        help="Structured scaffold generation mode. anchored predicts bounded residuals from encoder centers.",
    )
    parser.add_argument(
        "--scaffold-anchor-source",
        choices=["coarse", "fine", "coarse_fine"],
        default="coarse_fine",
        help="Encoder center stream used by --scaffold-mode anchored.",
    )
    parser.add_argument(
        "--scaffold-anchor-residual-scale",
        type=float,
        default=0.10,
        help="Maximum normalized scaffold residual from the selected anchor center.",
    )
    parser.add_argument(
        "--train-mirror-axes",
        nargs="*",
        choices=["x", "y", "z"],
        default=[],
        help="Augment only the training split with mirrored copies across the given normalized axes.",
    )
    parser.add_argument(
        "--refinement-mode",
        choices=["free", "tangent"],
        default="free",
        help="Local decoder offset parameterization. 'tangent' constrains offsets in scaffold-local frames.",
    )
    parser.add_argument(
        "--tangent-offset-scale",
        type=float,
        default=0.20,
        help="Maximum normalized tangent-plane patch radius for tangent refinement mode.",
    )
    parser.add_argument(
        "--normal-offset-scale",
        type=float,
        default=0.02,
        help="Maximum normalized normal-direction offset for tangent refinement mode.",
    )
    parser.add_argument(
        "--patch-type-count",
        type=int,
        default=4,
        help="Number of unsupervised patch-type logits emitted per scaffold in tangent mode.",
    )
    parser.add_argument(
        "--boundary-token-count",
        type=int,
        default=1,
        help="Number of boundary stream tokens when --use-boundary-feature is enabled.",
    )
    parser.add_argument(
        "--lattice-layers",
        type=int,
        default=0,
        help="Enable typed global/local/boundary cross-attention lattice with this many layers.",
    )
    parser.add_argument("--lattice-heads", type=int, default=4)
    parser.add_argument("--lambda-spread", type=float, default=0.1)
    parser.add_argument("--lambda-normal", type=float, default=0.02)
    parser.add_argument("--lambda-target-coverage", type=float, default=0.0)
    parser.add_argument("--coverage-threshold-mm", type=float, default=5.0)
    parser.add_argument("--lambda-pred-surface", type=float, default=0.0)
    parser.add_argument("--pred-surface-threshold-mm", type=float, default=5.0)
    parser.add_argument("--lambda-refinement-occupancy", type=float, default=0.0)
    parser.add_argument("--occupancy-positive-threshold-mm", type=float, default=5.0)
    parser.add_argument("--occupancy-negative-threshold-mm", type=float, default=15.0)
    parser.add_argument("--lambda-scaffold", type=float, default=0.0)
    parser.add_argument("--lambda-feature-scaffold", type=float, default=0.0)
    parser.add_argument("--lambda-boundary-scaffold", type=float, default=0.0)
    parser.add_argument("--lambda-crease-scaffold", type=float, default=0.0)
    parser.add_argument("--lambda-corner-scaffold", type=float, default=0.0)
    parser.add_argument("--scaffold-target-points", type=int, default=0)
    parser.add_argument("--scaffold-target-boundary-fraction", type=float, default=0.35)
    parser.add_argument("--scaffold-target-crease-fraction", type=float, default=0.35)
    parser.add_argument("--scaffold-target-corner-fraction", type=float, default=0.10)
    parser.add_argument(
        "--boundary-sample-fraction",
        type=float,
        default=0.0,
        help="Fraction of target points sampled from open boundary edges. Disabled by default.",
    )
    parser.add_argument("--lambda-boundary-coverage", type=float, default=0.0)
    parser.add_argument("--boundary-threshold-mm", type=float, default=5.0)
    parser.add_argument(
        "--use-joint-distance",
        action="store_true",
        help="Append nearest joint distance from joints.json. Disabled by default for shape-only AE tests.",
    )
    parser.add_argument(
        "--use-boundary-feature",
        action="store_true",
        help="Append boundary indicator to each input point and enable the boundary summary token.",
    )
    parser.add_argument("--cae-score-target-p95-weight", type=float, default=1.0)
    parser.add_argument("--cae-score-recon-p95-weight", type=float, default=0.0)
    parser.add_argument("--cae-score-boundary-p95-weight", type=float, default=1.0)
    parser.add_argument("--cae-score-chamfer-weight", type=float, default=0.25)
    parser.add_argument("--cae-score-target-within-weight", type=float, default=10.0)
    parser.add_argument("--cae-score-pred-within-weight", type=float, default=0.0)
    parser.add_argument("--cae-score-boundary-within-weight", type=float, default=10.0)
    parser.add_argument("--linear-deflection", type=float, default=2.0)
    parser.add_argument("--angular-deflection", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    main()

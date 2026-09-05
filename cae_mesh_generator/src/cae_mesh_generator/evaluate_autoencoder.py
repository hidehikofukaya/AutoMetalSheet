"""Visual evaluation for midsurface point-cloud autoencoder checkpoints."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - optional visualization dependency
    make_subplots = None
    go = None

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - torch fallback is tested indirectly
    cKDTree = None

from .data.fill_volume_dataset import MidsurfacePointCloudDataset, PartRecord
from .data.source_fingerprint import compare_fingerprint_rows, fingerprint_records
from .data.step_tessellate import TessellationConfig
from .model.hierarchical_ae import HierarchicalMidsurfaceAutoencoder, StructuredScaffoldAutoencoder


@dataclass(frozen=True)
class DistanceSummary:
    mean: float
    p50: float
    p95: float
    p99: float
    max: float


@dataclass(frozen=True)
class PartEvalMetrics:
    part_name: str
    record_index: int
    split: str
    scale_mm: float
    bbox_diagonal_mm: float
    n_target: int
    n_recon: int
    n_boundary: int
    chamfer_mm: float
    chamfer_bbox_diag_pct: float
    sampling_floor_chamfer_mm: float
    sampling_floor_bbox_diag_pct: float
    chamfer_to_sampling_floor: float
    recon_to_target_mm: DistanceSummary
    target_to_recon_mm: DistanceSummary
    boundary_to_recon_mm: DistanceSummary
    recon_within_1mm: float
    recon_within_2mm: float
    recon_within_5mm: float
    target_within_1mm: float
    target_within_2mm: float
    target_within_5mm: float
    boundary_within_1mm: float
    boundary_within_2mm: float
    boundary_within_5mm: float


def main() -> None:
    args = parse_args()
    if args.eval_n_points is not None and args.eval_n_points <= 0:
        raise SystemExit("--eval-n-points must be positive when provided.")
    if args.metric_target_n_points is not None and args.metric_target_n_points <= 0:
        raise SystemExit("--metric-target-n-points must be positive when provided.")
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "visual_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = payload.get("args", {})
    records = records_from_checkpoint(payload)
    if not records:
        raise SystemExit(f"No records found in checkpoint: {checkpoint_path}")
    fingerprint_report = compare_fingerprint_rows(
        payload.get("record_fingerprints"),
        fingerprint_records(records),
    )
    if fingerprint_report["status"] != "match":
        print(f"warning: {fingerprint_report['message']}")

    split_by_index = split_labels_from_checkpoint(payload, len(records))
    selected_indexes = select_indexes(
        n_records=len(records),
        requested=args.part_indexes,
        max_parts=args.max_parts,
        split_filter=args.split,
        split_by_index=split_by_index,
    )
    model = load_model_from_checkpoint(payload, device)
    input_dataset = build_dataset_from_checkpoint(
        records=records,
        ckpt_args=ckpt_args,
        output_dir=output_dir,
        eval_n_points=args.eval_n_points,
        seed=args.seed,
        joint_distance_mode=args.joint_distance,
        boundary_sample_fraction=args.boundary_sample_fraction,
    )
    metric_target_n_points = args.metric_target_n_points or int(args.eval_n_points or ckpt_args.get("n_points", 512))
    target_dataset = build_dataset_from_checkpoint(
        records=records,
        ckpt_args=ckpt_args,
        output_dir=output_dir,
        eval_n_points=metric_target_n_points,
        seed=args.seed,
        joint_distance_mode=args.joint_distance,
        boundary_sample_fraction=args.boundary_sample_fraction,
    )
    calibration_seed = int(args.seed if args.seed is not None else ckpt_args.get("seed", 7)) + args.sampling_floor_seed_offset
    calibration_dataset = build_dataset_from_checkpoint(
        records=records,
        ckpt_args=ckpt_args,
        output_dir=output_dir,
        eval_n_points=metric_target_n_points,
        seed=calibration_seed,
        joint_distance_mode=args.joint_distance,
        boundary_sample_fraction=args.boundary_sample_fraction,
    )

    all_metrics: list[PartEvalMetrics] = []
    part_reports: list[dict[str, str | PartEvalMetrics]] = []
    for record_index in selected_indexes:
        item = input_dataset[record_index]
        target_item = target_dataset[record_index]
        calibration_item = calibration_dataset[record_index]
        part_name = str(item["name"])
        part_dir = output_dir / f"{record_index:03d}_{safe_name(part_name)}"
        part_dir.mkdir(parents=True, exist_ok=True)
        metrics = evaluate_part(
            model=model,
            item=item,
            target_item=target_item,
            calibration_item=calibration_item,
            record_index=record_index,
            split=split_by_index.get(record_index, "unknown"),
            device=device,
            part_dir=part_dir,
            plot_point_limit=args.plot_point_limit,
            error_vmax_percentile=args.error_vmax_percentile,
            write_html=args.write_html,
        )
        all_metrics.append(metrics)
        part_reports.append(
            {
                "part_dir": str(part_dir),
                "png": str(part_dir / "comparison.png"),
                "html": str(part_dir / "comparison.html"),
                "metrics": metrics,
            }
        )
        print(
            f"{part_name}: chamfer_mm={metrics.chamfer_mm:.4f} "
            f"floor_x={metrics.chamfer_to_sampling_floor:.2f} "
            f"recon_p95={metrics.recon_to_target_mm.p95:.4f} "
            f"target_p95={metrics.target_to_recon_mm.p95:.4f}"
        )

    write_metrics(output_dir, all_metrics)
    write_eval_manifest(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        ckpt_args=ckpt_args,
        eval_settings={
            "input_n_points": int(args.eval_n_points or ckpt_args.get("n_points", 512)),
            "metric_target_n_points": int(metric_target_n_points),
            "sampling_floor_seed_offset": int(args.sampling_floor_seed_offset),
        },
        selected_indexes=selected_indexes,
        fingerprint_report=fingerprint_report,
        metrics=all_metrics,
    )
    write_index_html(output_dir, checkpoint_path, part_reports, write_html=args.write_html)
    print(f"wrote visual evaluation to {output_dir}")


def records_from_checkpoint(payload: dict) -> list[PartRecord]:
    records: list[PartRecord] = []
    for row in payload.get("records", []):
        stp_path = Path(row["stp_path"])
        joints_path = stp_path.parent.parent / "annotations" / "joints.json"
        records.append(
            PartRecord(
                assembly_id=str(row["assembly_id"]),
                part_id_raw=str(row["part_id_raw"]),
                canonical_part_id=str(row["canonical_part_id"]),
                stp_path=stp_path,
                joints_path=joints_path if joints_path.exists() else None,
            )
        )
    return records


def split_labels_from_checkpoint(payload: dict, n_records: int) -> dict[int, str]:
    split = payload.get("split")
    if not split:
        return {i: "train" for i in range(n_records)}
    labels = {i: "unused" for i in range(n_records)}
    for i in split.get("train_indices", []):
        labels[int(i)] = "train"
    for i in split.get("val_indices", []):
        labels[int(i)] = "val"
    return labels


def select_indexes(
    n_records: int,
    requested: list[int] | None,
    max_parts: int | None,
    split_filter: str,
    split_by_index: dict[int, str],
) -> list[int]:
    if requested:
        indexes = requested
    else:
        indexes = [
            i
            for i in range(n_records)
            if split_filter == "all" or split_by_index.get(i, "unknown") == split_filter
        ]
        count = len(indexes) if max_parts is None or max_parts <= 0 else min(max_parts, len(indexes))
        indexes = indexes[:count]
    bad = [i for i in indexes if i < 0 or i >= n_records]
    if bad:
        raise SystemExit(f"Part index out of range: {bad}; checkpoint has {n_records} records.")
    return indexes


def load_model_from_checkpoint(payload: dict, device: torch.device) -> torch.nn.Module:
    ckpt_args = payload.get("args", {})
    model_config = payload.get("model_config", {})
    model_kind = ckpt_args.get("model_kind", "point")
    feature_dim = int(model_config.get("feature_dim", 8 if ckpt_args.get("use_boundary_feature", False) else 7))
    boundary_feature_index = model_config.get("boundary_feature_index")
    if boundary_feature_index is None and ckpt_args.get("use_boundary_feature", False) and feature_dim >= 8:
        boundary_feature_index = feature_dim - 1
    if boundary_feature_index is not None:
        boundary_feature_index = int(boundary_feature_index)
    common = {
        "feature_dim": feature_dim,
        "token_dim": int(ckpt_args.get("token_dim", 96)),
        "n_points_out": int(model_config.get("n_points_out", ckpt_args.get("n_points", 512))),
        "n_coarse": int(ckpt_args.get("n_coarse", 96)),
        "n_patches": int(ckpt_args.get("n_patches", 48)),
        "k_neighbors": int(ckpt_args.get("k_neighbors", 24)),
        "n_latents": int(ckpt_args.get("n_latents", 48)),
        "boundary_feature_index": boundary_feature_index,
        "boundary_token_count": int(model_config.get("boundary_token_count", ckpt_args.get("boundary_token_count", 1))),
    }
    if model_kind == "structured":
        model = StructuredScaffoldAutoencoder(
            **common,
            n_scaffold=int(model_config.get("n_scaffold", ckpt_args.get("n_scaffold", 64))),
            points_per_scaffold=model_config.get("points_per_scaffold", ckpt_args.get("points_per_scaffold")),
            n_local_tokens=int(ckpt_args.get("n_local_tokens", 8)),
            lattice_layers=int(model_config.get("lattice_layers", ckpt_args.get("lattice_layers", 0))),
            lattice_heads=int(model_config.get("lattice_heads", ckpt_args.get("lattice_heads", 4))),
            refinement_mode=str(model_config.get("refinement_mode", ckpt_args.get("refinement_mode", "free"))),
            tangent_offset_scale=float(
                model_config.get("tangent_offset_scale", ckpt_args.get("tangent_offset_scale", 0.20))
            ),
            normal_offset_scale=float(
                model_config.get("normal_offset_scale", ckpt_args.get("normal_offset_scale", 0.02))
            ),
            patch_type_count=int(model_config.get("patch_type_count", ckpt_args.get("patch_type_count", 4))),
            scaffold_mode=str(model_config.get("scaffold_mode", ckpt_args.get("scaffold_mode", "learned"))),
            scaffold_anchor_source=str(
                model_config.get("scaffold_anchor_source", ckpt_args.get("scaffold_anchor_source", "coarse_fine"))
            ),
            scaffold_anchor_residual_scale=float(
                model_config.get(
                    "scaffold_anchor_residual_scale",
                    ckpt_args.get("scaffold_anchor_residual_scale", 0.10),
                )
            ),
        ).to(device)
    else:
        model = HierarchicalMidsurfaceAutoencoder(**common).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def build_dataset_from_checkpoint(
    records: list[PartRecord],
    ckpt_args: dict,
    output_dir: Path,
    eval_n_points: int | None,
    seed: int | None,
    joint_distance_mode: str,
    boundary_sample_fraction: float | None,
) -> MidsurfacePointCloudDataset:
    use_joint_distance = bool(ckpt_args.get("use_joint_distance", False))
    if joint_distance_mode == "on":
        use_joint_distance = True
    elif joint_distance_mode == "off":
        use_joint_distance = False
    n_points = int(eval_n_points or ckpt_args.get("n_points", 512))
    boundary_fraction = (
        float(boundary_sample_fraction)
        if boundary_sample_fraction is not None
        else float(ckpt_args.get("boundary_sample_fraction", 0.0))
    )
    return MidsurfacePointCloudDataset(
        records=records,
        cache_dir=output_dir / "tess_cache",
        n_points=n_points,
        seed=int(seed if seed is not None else ckpt_args.get("seed", 7)),
        tessellation=TessellationConfig(
            linear_deflection=float(ckpt_args.get("linear_deflection", 2.0)),
            angular_deflection=float(ckpt_args.get("angular_deflection", 0.5)),
        ),
        use_joint_distance=use_joint_distance,
        boundary_sample_fraction=boundary_fraction,
        use_boundary_feature=bool(ckpt_args.get("use_boundary_feature", False)),
        scaffold_target_points=int(ckpt_args.get("scaffold_target_points", 0)),
        scaffold_target_boundary_fraction=float(ckpt_args.get("scaffold_target_boundary_fraction", 0.35)),
        scaffold_target_crease_fraction=float(ckpt_args.get("scaffold_target_crease_fraction", 0.35)),
        scaffold_target_corner_fraction=float(ckpt_args.get("scaffold_target_corner_fraction", 0.10)),
    )


def evaluate_part(
    model: torch.nn.Module,
    item: dict,
    target_item: dict,
    calibration_item: dict,
    record_index: int,
    device: torch.device,
    part_dir: Path,
    split: str,
    plot_point_limit: int,
    error_vmax_percentile: float,
    write_html: bool,
) -> PartEvalMetrics:
    with torch.no_grad():
        points = item["points"][None, ...].to(device)
        features = item["features"][None, ...].to(device)
        pred = model(points, features)

    target_norm = target_item["points"].detach().cpu().numpy()
    calibration_norm = calibration_item["points"].detach().cpu().numpy()
    recon_norm = pred["points"][0].detach().cpu().numpy()
    input_center = item["center"].detach().cpu().numpy()
    input_scale = float(item["scale"].detach().cpu())
    target_center = target_item["center"].detach().cpu().numpy()
    target_scale = float(target_item["scale"].detach().cpu())
    active_recon_world = None
    if "refinement_logits" in pred:
        logits = pred["refinement_logits"][0].detach().cpu().numpy()
        active_recon_norm = recon_norm[logits >= 0.0]
        if len(active_recon_norm):
            active_recon_world = active_recon_norm * input_scale + input_center[None, :]
    target_world = target_norm * target_scale + target_center[None, :]
    calibration_world = calibration_norm * target_scale + target_center[None, :]
    recon_world = recon_norm * input_scale + input_center[None, :]
    scaffold_world = None
    if "scaffold_points" in pred:
        scaffold_norm = pred["scaffold_points"][0].detach().cpu().numpy()
        if "active_scaffold_mask" in pred:
            active = pred["active_scaffold_mask"][0].detach().cpu().numpy().astype(bool)
            scaffold_norm = scaffold_norm[active]
        scaffold_world = scaffold_norm * input_scale + input_center[None, :]
    scaffold_targets = target_item.get("scaffold_targets")
    if scaffold_targets is not None:
        scaffold_targets_norm = scaffold_targets.detach().cpu().numpy()
        scaffold_target_mask = target_item.get("scaffold_target_mask")
        if scaffold_target_mask is not None:
            mask = scaffold_target_mask.detach().cpu().numpy().astype(bool)
            scaffold_targets_norm = scaffold_targets_norm[mask]
        scaffold_targets_world = scaffold_targets_norm * target_scale + target_center[None, :]
    else:
        scaffold_targets_world = np.zeros((0, 3), dtype=np.float32)
    typed_scaffold_targets_world: dict[str, np.ndarray] = {}
    for target_name in ("boundary", "crease", "corner"):
        pts_tensor = target_item.get(f"scaffold_{target_name}_targets")
        mask_tensor = target_item.get(f"scaffold_{target_name}_target_mask")
        if pts_tensor is None or mask_tensor is None:
            typed_scaffold_targets_world[target_name] = np.zeros((0, 3), dtype=np.float32)
            continue
        pts_norm = pts_tensor.detach().cpu().numpy()
        mask = mask_tensor.detach().cpu().numpy().astype(bool)
        typed_scaffold_targets_world[target_name] = pts_norm[mask] * target_scale + target_center[None, :]

    recon_to_target, _ = nearest_distances(recon_world, target_world)
    target_to_recon, _ = nearest_distances(target_world, recon_world)
    floor_target_to_calibration, _ = nearest_distances(target_world, calibration_world)
    floor_calibration_to_target, _ = nearest_distances(calibration_world, target_world)
    sampling_floor_chamfer_mm = float(np.mean(floor_target_to_calibration) + np.mean(floor_calibration_to_target))
    boundary_mask = target_item.get("boundary_mask")
    if boundary_mask is not None:
        boundary_mask_np = boundary_mask.detach().cpu().numpy().astype(bool)
    else:
        boundary_mask_np = np.zeros((len(target_world),), dtype=bool)
    boundary_world = target_world[boundary_mask_np]
    if len(boundary_world):
        boundary_to_recon, _ = nearest_distances(boundary_world, recon_world)
    else:
        boundary_to_recon = np.zeros((0,), dtype=np.float64)
    metrics = make_metrics(
        part_name=str(item["name"]),
        record_index=record_index,
        split=split,
        scale_mm=target_scale,
        bbox_diagonal_mm=bbox_diagonal(target_world),
        sampling_floor_chamfer_mm=sampling_floor_chamfer_mm,
        recon_to_target=recon_to_target,
        target_to_recon=target_to_recon,
        boundary_to_recon=boundary_to_recon,
        n_target=len(target_world),
        n_recon=len(recon_world),
        n_boundary=len(boundary_world),
    )

    recon_colors = colors_from_errors(recon_to_target, error_vmax_percentile)
    target_colors = colors_from_errors(target_to_recon, error_vmax_percentile)
    write_colored_ply(part_dir / "target.ply", target_world, np.full((len(target_world), 3), [31, 119, 180]))
    write_colored_ply(part_dir / "recon.ply", recon_world, np.full((len(recon_world), 3), [255, 127, 14]))
    if active_recon_world is not None:
        write_colored_ply(
            part_dir / "recon_refinement_active.ply",
            active_recon_world,
            np.full((len(active_recon_world), 3), [23, 190, 207]),
        )
    write_colored_ply(part_dir / "recon_error.ply", recon_world, recon_colors)
    write_colored_ply(part_dir / "target_miss_error.ply", target_world, target_colors)
    write_overlay_ply(part_dir / "overlay_target_recon.ply", target_world, recon_world)
    if len(boundary_world):
        boundary_colors = colors_from_errors(boundary_to_recon, error_vmax_percentile)
        write_colored_ply(part_dir / "boundary_miss_error.ply", boundary_world, boundary_colors)
        write_colored_ply(part_dir / "boundary_points.ply", boundary_world, np.full((len(boundary_world), 3), [148, 103, 189]))
    if scaffold_world is not None:
        write_colored_ply(
            part_dir / "scaffold.ply",
            scaffold_world,
            np.full((len(scaffold_world), 3), [44, 160, 44]),
        )
    if len(scaffold_targets_world):
        write_colored_ply(
            part_dir / "scaffold_targets.ply",
            scaffold_targets_world,
            np.full((len(scaffold_targets_world), 3), [214, 39, 40]),
        )
    for target_name, pts_world in typed_scaffold_targets_world.items():
        if len(pts_world):
            write_colored_ply(
                part_dir / f"scaffold_{target_name}_targets.ply",
                pts_world,
                np.full((len(pts_world), 3), [214, 39, 40]),
            )
    write_png_report(
        path=part_dir / "comparison.png",
        part_name=str(item["name"]),
        target=target_world,
        recon=recon_world,
        recon_errors=recon_to_target,
        target_errors=target_to_recon,
        metrics=metrics,
        plot_point_limit=plot_point_limit,
        error_vmax_percentile=error_vmax_percentile,
    )
    write_projection_png(
        path=part_dir / "projection_overlay.png",
        part_name=str(item["name"]),
        target=target_world,
        recon=recon_world,
        scaffold=scaffold_world,
        metrics=metrics,
        plot_point_limit=plot_point_limit,
    )
    if write_html:
        write_plotly_html(
            path=part_dir / "comparison.html",
            part_name=str(item["name"]),
            target=target_world,
            recon=recon_world,
            recon_errors=recon_to_target,
            target_errors=target_to_recon,
            metrics=metrics,
            plot_point_limit=plot_point_limit,
            error_vmax_percentile=error_vmax_percentile,
        )
    (part_dir / "metrics.json").write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
    return metrics


def nearest_distances(query: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if len(query) == 0 or len(reference) == 0:
        raise ValueError("query and reference point clouds must be non-empty")
    if cKDTree is not None:
        distances, indexes = cKDTree(reference).query(query, k=1)
        return distances.astype(np.float64), indexes.astype(np.int64)
    dist = torch.cdist(
        torch.from_numpy(query[None, ...].astype(np.float32)),
        torch.from_numpy(reference[None, ...].astype(np.float32)),
    )[0]
    values, indexes = dist.min(dim=1)
    return values.numpy().astype(np.float64), indexes.numpy().astype(np.int64)


def make_metrics(
    part_name: str,
    record_index: int,
    split: str,
    scale_mm: float,
    bbox_diagonal_mm: float,
    sampling_floor_chamfer_mm: float,
    recon_to_target: np.ndarray,
    target_to_recon: np.ndarray,
    boundary_to_recon: np.ndarray,
    n_target: int,
    n_recon: int,
    n_boundary: int,
) -> PartEvalMetrics:
    chamfer_mm = float(np.mean(recon_to_target) + np.mean(target_to_recon))
    bbox_diag = max(float(bbox_diagonal_mm), 1.0e-12)
    floor_mm = max(float(sampling_floor_chamfer_mm), 0.0)
    return PartEvalMetrics(
        part_name=part_name,
        record_index=record_index,
        split=split,
        scale_mm=float(scale_mm),
        bbox_diagonal_mm=bbox_diag,
        n_target=int(n_target),
        n_recon=int(n_recon),
        n_boundary=int(n_boundary),
        chamfer_mm=chamfer_mm,
        chamfer_bbox_diag_pct=100.0 * chamfer_mm / bbox_diag,
        sampling_floor_chamfer_mm=floor_mm,
        sampling_floor_bbox_diag_pct=100.0 * floor_mm / bbox_diag,
        chamfer_to_sampling_floor=chamfer_mm / floor_mm if floor_mm > 1.0e-12 else 0.0,
        recon_to_target_mm=summarize_distances(recon_to_target),
        target_to_recon_mm=summarize_distances(target_to_recon),
        boundary_to_recon_mm=summarize_distances_or_zero(boundary_to_recon),
        recon_within_1mm=fraction_within(recon_to_target, 1.0),
        recon_within_2mm=fraction_within(recon_to_target, 2.0),
        recon_within_5mm=fraction_within(recon_to_target, 5.0),
        target_within_1mm=fraction_within(target_to_recon, 1.0),
        target_within_2mm=fraction_within(target_to_recon, 2.0),
        target_within_5mm=fraction_within(target_to_recon, 5.0),
        boundary_within_1mm=fraction_within_or_zero(boundary_to_recon, 1.0),
        boundary_within_2mm=fraction_within_or_zero(boundary_to_recon, 2.0),
        boundary_within_5mm=fraction_within_or_zero(boundary_to_recon, 5.0),
    )


def bbox_diagonal(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return 0.0
    return float(np.linalg.norm(np.max(pts, axis=0) - np.min(pts, axis=0)))


def summarize_distances(distances: np.ndarray) -> DistanceSummary:
    distances = np.asarray(distances, dtype=np.float64)
    return DistanceSummary(
        mean=float(np.mean(distances)),
        p50=float(np.percentile(distances, 50)),
        p95=float(np.percentile(distances, 95)),
        p99=float(np.percentile(distances, 99)),
        max=float(np.max(distances)),
    )


def summarize_distances_or_zero(distances: np.ndarray) -> DistanceSummary:
    if len(distances) == 0:
        return DistanceSummary(mean=0.0, p50=0.0, p95=0.0, p99=0.0, max=0.0)
    return summarize_distances(distances)


def fraction_within(distances: np.ndarray, threshold: float) -> float:
    return float(np.mean(np.asarray(distances) <= threshold))


def fraction_within_or_zero(distances: np.ndarray, threshold: float) -> float:
    if len(distances) == 0:
        return 0.0
    return fraction_within(distances, threshold)


def colors_from_errors(errors: np.ndarray, vmax_percentile: float) -> np.ndarray:
    errors = np.asarray(errors, dtype=np.float64)
    vmax = float(np.percentile(errors, vmax_percentile))
    if vmax <= 1.0e-12:
        vmax = float(np.max(errors))
    if vmax <= 1.0e-12:
        scaled = np.zeros_like(errors)
    else:
        scaled = np.clip(errors / vmax, 0.0, 1.0)
    cmap = plt.get_cmap("turbo")
    return (cmap(scaled)[:, :3] * 255.0).astype(np.uint8)


def write_colored_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    colors = np.asarray(colors, dtype=np.uint8)
    if colors.shape[0] != points.shape[0]:
        raise ValueError("points and colors must have the same length")
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    lines.extend(
        f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {int(c[0])} {int(c[1])} {int(c[2])}"
        for p, c in zip(points, colors)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_overlay_ply(path: Path, target: np.ndarray, recon: np.ndarray) -> None:
    target_colors = np.full((len(target), 3), [31, 119, 180], dtype=np.uint8)
    recon_colors = np.full((len(recon), 3), [255, 127, 14], dtype=np.uint8)
    write_colored_ply(
        path,
        np.concatenate([target, recon], axis=0),
        np.concatenate([target_colors, recon_colors], axis=0),
    )


def write_png_report(
    path: Path,
    part_name: str,
    target: np.ndarray,
    recon: np.ndarray,
    recon_errors: np.ndarray,
    target_errors: np.ndarray,
    metrics: PartEvalMetrics,
    plot_point_limit: int,
    error_vmax_percentile: float,
) -> None:
    target_plot, target_errors_plot = downsample_pair(target, target_errors, plot_point_limit, seed=11)
    recon_plot, recon_errors_plot = downsample_pair(recon, recon_errors, plot_point_limit, seed=17)
    all_points = np.concatenate([target_plot, recon_plot], axis=0)
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    fig.suptitle(part_name, fontsize=14)

    ax = fig.add_subplot(2, 3, 1, projection="3d")
    scatter3d(ax, target_plot, c="#1f77b4", title="target")
    set_axes_equal(ax, all_points)

    ax = fig.add_subplot(2, 3, 2, projection="3d")
    scatter3d(ax, recon_plot, c="#ff7f0e", title="reconstruction")
    set_axes_equal(ax, all_points)

    ax = fig.add_subplot(2, 3, 3, projection="3d")
    scatter3d(ax, target_plot, c="#1f77b4", title="overlay", alpha=0.55)
    scatter3d(ax, recon_plot, c="#ff7f0e", title="", alpha=0.55)
    set_axes_equal(ax, all_points)

    vmax_recon = percentile_vmax(recon_errors, error_vmax_percentile)
    ax = fig.add_subplot(2, 3, 4, projection="3d")
    scatter3d(ax, recon_plot, c=recon_errors_plot, cmap="turbo", title="recon -> target error", vmax=vmax_recon)
    set_axes_equal(ax, all_points)

    vmax_target = percentile_vmax(target_errors, error_vmax_percentile)
    ax = fig.add_subplot(2, 3, 5, projection="3d")
    scatter3d(ax, target_plot, c=target_errors_plot, cmap="turbo", title="target -> recon miss", vmax=vmax_target)
    set_axes_equal(ax, all_points)

    ax = fig.add_subplot(2, 3, 6)
    ax.hist(recon_errors, bins=32, alpha=0.65, label="recon -> target", color="#ff7f0e")
    ax.hist(target_errors, bins=32, alpha=0.65, label="target -> recon", color="#1f77b4")
    ax.set_title("nearest distance histogram [mm]")
    ax.set_xlabel("distance [mm]")
    ax.set_ylabel("count")
    ax.legend()
    ax.text(
        0.02,
        0.96,
        metric_text(metrics),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_projection_png(
    path: Path,
    part_name: str,
    target: np.ndarray,
    recon: np.ndarray,
    scaffold: np.ndarray | None,
    metrics: PartEvalMetrics,
    plot_point_limit: int,
) -> None:
    target_plot, _ = downsample_pair(target, np.zeros(len(target)), plot_point_limit, seed=31)
    recon_plot, _ = downsample_pair(recon, np.zeros(len(recon)), plot_point_limit, seed=37)
    scaffold_plot = None
    if scaffold is not None:
        scaffold_plot, _ = downsample_pair(scaffold, np.zeros(len(scaffold)), plot_point_limit, seed=41)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    fig.suptitle(f"{part_name} orthographic overlay", fontsize=14)
    projections = [(0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ")]
    for ax, (a, b, title) in zip(axes, projections):
        ax.scatter(target_plot[:, a], target_plot[:, b], s=7, c="#1f77b4", alpha=0.55, label="target")
        ax.scatter(recon_plot[:, a], recon_plot[:, b], s=7, c="#ff7f0e", alpha=0.55, label="reconstruction")
        if scaffold_plot is not None:
            ax.scatter(scaffold_plot[:, a], scaffold_plot[:, b], s=16, c="#2ca02c", alpha=0.9, label="scaffold")
        ax.set_title(title)
        ax.set_xlabel("xyz"[a])
        ax.set_ylabel("xyz"[b])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    axes[2].text(
        0.02,
        0.98,
        metric_text(metrics),
        transform=axes[2].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    fig.savefig(path, dpi=170)
    plt.close(fig)


def scatter3d(
    ax,
    points: np.ndarray,
    c,
    title: str,
    alpha: float = 0.85,
    cmap: str | None = None,
    vmax: float | None = None,
) -> None:
    mappable = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=5,
        c=c,
        alpha=alpha,
        cmap=cmap,
        vmin=0.0 if cmap else None,
        vmax=vmax if cmap else None,
        linewidths=0,
    )
    ax.set_title(title)
    ax.view_init(elev=24, azim=-58)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if cmap:
        plt.colorbar(mappable, ax=ax, shrink=0.65, pad=0.02)


def set_axes_equal(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.55)
    if radius <= 1.0e-12:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def downsample_pair(
    points: np.ndarray,
    values: np.ndarray,
    limit: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if limit <= 0 or len(points) <= limit:
        return points, values
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=limit, replace=False)
    return points[idx], values[idx]


def percentile_vmax(values: np.ndarray, percentile: float) -> float:
    vmax = float(np.percentile(values, percentile))
    return vmax if vmax > 1.0e-12 else float(np.max(values) + 1.0e-12)


def metric_text(metrics: PartEvalMetrics) -> str:
    return "\n".join(
        [
            f"Chamfer: {metrics.chamfer_mm:.3f} mm",
            f"Chamfer / bbox diag: {metrics.chamfer_bbox_diag_pct:.2f}%",
            f"Sampling floor: {metrics.sampling_floor_chamfer_mm:.3f} mm ({metrics.chamfer_to_sampling_floor:.2f}x)",
            f"Recon->Target mean/p95: {metrics.recon_to_target_mm.mean:.3f}/{metrics.recon_to_target_mm.p95:.3f} mm",
            f"Target->Recon mean/p95: {metrics.target_to_recon_mm.mean:.3f}/{metrics.target_to_recon_mm.p95:.3f} mm",
            f"Target within 5mm: {metrics.target_within_5mm:.1%}",
            f"Recon within 5mm: {metrics.recon_within_5mm:.1%}",
            f"Boundary p95/within5: {metrics.boundary_to_recon_mm.p95:.3f} mm/{metrics.boundary_within_5mm:.1%}",
        ]
    )


def write_plotly_html(
    path: Path,
    part_name: str,
    target: np.ndarray,
    recon: np.ndarray,
    recon_errors: np.ndarray,
    target_errors: np.ndarray,
    metrics: PartEvalMetrics,
    plot_point_limit: int,
    error_vmax_percentile: float,
) -> None:
    if make_subplots is None or go is None:
        return
    target_plot, target_errors_plot = downsample_pair(target, target_errors, plot_point_limit, seed=23)
    recon_plot, recon_errors_plot = downsample_pair(recon, recon_errors, plot_point_limit, seed=29)
    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("overlay", "recon -> target error", "target -> recon miss"),
    )
    add_points3d(fig, target_plot, row=1, col=1, name="target", color="#1f77b4", opacity=0.6)
    add_points3d(fig, recon_plot, row=1, col=1, name="reconstruction", color="#ff7f0e", opacity=0.6)
    add_points3d(
        fig,
        recon_plot,
        row=1,
        col=2,
        name="recon error",
        values=recon_errors_plot,
        cmax=percentile_vmax(recon_errors, error_vmax_percentile),
    )
    add_points3d(
        fig,
        target_plot,
        row=1,
        col=3,
        name="target miss",
        values=target_errors_plot,
        cmax=percentile_vmax(target_errors, error_vmax_percentile),
    )
    fig.update_layout(
        title=f"{part_name}<br>{metric_text(metrics).replace(chr(10), ' | ')}",
        width=1680,
        height=720,
        margin={"l": 0, "r": 0, "t": 90, "b": 0},
    )
    fig.update_scenes(aspectmode="data")
    fig.write_html(path, include_plotlyjs="cdn")


def add_points3d(
    fig,
    points: np.ndarray,
    row: int,
    col: int,
    name: str,
    color: str | None = None,
    opacity: float = 0.85,
    values: np.ndarray | None = None,
    cmax: float | None = None,
) -> None:
    marker = {"size": 2.5, "opacity": opacity}
    if values is not None:
        marker.update(
            {
                "color": values,
                "colorscale": "Turbo",
                "cmin": 0.0,
                "cmax": cmax,
                "colorbar": {"title": "mm"},
            }
        )
    else:
        marker["color"] = color
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=marker,
            name=name,
        ),
        row=row,
        col=col,
    )


def write_metrics(output_dir: Path, metrics: list[PartEvalMetrics]) -> None:
    (output_dir / "metrics.json").write_text(
        json.dumps([asdict(m) for m in metrics], indent=2),
        encoding="utf-8",
    )
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "record_index",
                "split",
                "part_name",
                "chamfer_mm",
                "chamfer_bbox_diag_pct",
                "sampling_floor_chamfer_mm",
                "sampling_floor_bbox_diag_pct",
                "chamfer_to_sampling_floor",
                "recon_mean_mm",
                "recon_p95_mm",
                "target_mean_mm",
                "target_p95_mm",
                "boundary_count",
                "boundary_p95_mm",
                "boundary_within_5mm",
                "target_within_5mm",
                "recon_within_5mm",
            ],
        )
        writer.writeheader()
        for m in metrics:
            writer.writerow(
                {
                    "record_index": m.record_index,
                    "split": m.split,
                    "part_name": m.part_name,
                    "chamfer_mm": m.chamfer_mm,
                    "chamfer_bbox_diag_pct": m.chamfer_bbox_diag_pct,
                    "sampling_floor_chamfer_mm": m.sampling_floor_chamfer_mm,
                    "sampling_floor_bbox_diag_pct": m.sampling_floor_bbox_diag_pct,
                    "chamfer_to_sampling_floor": m.chamfer_to_sampling_floor,
                    "recon_mean_mm": m.recon_to_target_mm.mean,
                    "recon_p95_mm": m.recon_to_target_mm.p95,
                    "target_mean_mm": m.target_to_recon_mm.mean,
                    "target_p95_mm": m.target_to_recon_mm.p95,
                    "boundary_count": m.n_boundary,
                    "boundary_p95_mm": m.boundary_to_recon_mm.p95,
                    "boundary_within_5mm": m.boundary_within_5mm,
                    "target_within_5mm": m.target_within_5mm,
                    "recon_within_5mm": m.recon_within_5mm,
                }
            )
    write_aggregate_metrics(output_dir, metrics)


def write_aggregate_metrics(output_dir: Path, metrics: list[PartEvalMetrics]) -> None:
    groups: dict[str, list[PartEvalMetrics]] = {}
    groups["all"] = list(metrics)
    for metric in metrics:
        groups.setdefault(metric.split, []).append(metric)
    rows = [aggregate_group(label, rows) for label, rows in groups.items() if rows]
    (output_dir / "aggregate_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (output_dir / "aggregate_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "split",
            "count",
            "chamfer_mean_mm",
            "chamfer_bbox_diag_pct_mean",
            "sampling_floor_chamfer_mean_mm",
            "sampling_floor_bbox_diag_pct_mean",
            "chamfer_to_sampling_floor_mean",
            "recon_p95_mean_mm",
            "target_p95_mean_mm",
            "boundary_p95_mean_mm",
            "boundary_within_5mm_mean",
            "target_within_5mm_mean",
            "recon_within_5mm_mean",
            "worst_chamfer_part",
            "worst_chamfer_mm",
            "worst_target_p95_part",
            "worst_target_p95_mm",
            "worst_boundary_p95_part",
            "worst_boundary_p95_mm",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_group(label: str, metrics: list[PartEvalMetrics]) -> dict[str, object]:
    worst_chamfer = max(metrics, key=lambda m: m.chamfer_mm)
    worst_target = max(metrics, key=lambda m: m.target_to_recon_mm.p95)
    worst_boundary = max(metrics, key=lambda m: m.boundary_to_recon_mm.p95)
    return {
        "split": label,
        "count": len(metrics),
        "chamfer_mean_mm": float(np.mean([m.chamfer_mm for m in metrics])),
        "chamfer_bbox_diag_pct_mean": float(np.mean([m.chamfer_bbox_diag_pct for m in metrics])),
        "sampling_floor_chamfer_mean_mm": float(np.mean([m.sampling_floor_chamfer_mm for m in metrics])),
        "sampling_floor_bbox_diag_pct_mean": float(np.mean([m.sampling_floor_bbox_diag_pct for m in metrics])),
        "chamfer_to_sampling_floor_mean": float(np.mean([m.chamfer_to_sampling_floor for m in metrics])),
        "recon_p95_mean_mm": float(np.mean([m.recon_to_target_mm.p95 for m in metrics])),
        "target_p95_mean_mm": float(np.mean([m.target_to_recon_mm.p95 for m in metrics])),
        "boundary_p95_mean_mm": float(np.mean([m.boundary_to_recon_mm.p95 for m in metrics])),
        "boundary_within_5mm_mean": float(np.mean([m.boundary_within_5mm for m in metrics])),
        "target_within_5mm_mean": float(np.mean([m.target_within_5mm for m in metrics])),
        "recon_within_5mm_mean": float(np.mean([m.recon_within_5mm for m in metrics])),
        "worst_chamfer_part": worst_chamfer.part_name,
        "worst_chamfer_mm": worst_chamfer.chamfer_mm,
        "worst_target_p95_part": worst_target.part_name,
        "worst_target_p95_mm": worst_target.target_to_recon_mm.p95,
        "worst_boundary_p95_part": worst_boundary.part_name,
        "worst_boundary_p95_mm": worst_boundary.boundary_to_recon_mm.p95,
    }


def write_eval_manifest(
    output_dir: Path,
    checkpoint_path: Path,
    ckpt_args: dict,
    eval_settings: dict[str, int],
    selected_indexes: list[int],
    fingerprint_report: dict[str, object],
    metrics: list[PartEvalMetrics],
) -> None:
    manifest = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_args": ckpt_args,
        "eval_settings": eval_settings,
        "selected_indexes": selected_indexes,
        "source_fingerprint_check": fingerprint_report,
        "metrics_count": len(metrics),
        "split_counts": split_counts(metrics),
        "metrics_json": str(output_dir / "metrics.json"),
        "metrics_csv": str(output_dir / "metrics.csv"),
        "aggregate_metrics_json": str(output_dir / "aggregate_metrics.json"),
        "aggregate_metrics_csv": str(output_dir / "aggregate_metrics.csv"),
        "note": (
            "Targets are reconstructed from the current STEP files unless the source_fingerprint_check "
            "status is match against fingerprints saved in the checkpoint."
        ),
    }
    (output_dir / "eval_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def split_counts(metrics: list[PartEvalMetrics]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for metric in metrics:
        counts[metric.split] = counts.get(metric.split, 0) + 1
    return counts


def write_index_html(
    output_dir: Path,
    checkpoint_path: Path,
    reports: list[dict[str, str | PartEvalMetrics]],
    write_html: bool,
) -> None:
    rows = []
    for report in reports:
        metrics = report["metrics"]
        assert isinstance(metrics, PartEvalMetrics)
        part_dir = Path(str(report["part_dir"]))
        rel_png = Path(str(report["png"])).relative_to(output_dir).as_posix()
        rel_projection = (part_dir / "projection_overlay.png").relative_to(output_dir).as_posix()
        html_path = Path(str(report["html"]))
        link_html = ""
        if write_html and html_path.exists():
            rel_html = html_path.relative_to(output_dir).as_posix()
            link_html = f'<a href="{html.escape(rel_html)}">interactive</a>'
        rows.append(
            f"""
            <section>
              <h2>{html.escape(metrics.part_name)} ({html.escape(metrics.split)})</h2>
              <p>{html.escape(metric_text(metrics)).replace(chr(10), '<br>')} {link_html}</p>
              <img src="{html.escape(rel_png)}" alt="{html.escape(metrics.part_name)} comparison" />
              <img src="{html.escape(rel_projection)}" alt="{html.escape(metrics.part_name)} orthographic overlay" />
              <p><a href="{html.escape(part_dir.relative_to(output_dir).as_posix())}/metrics.json">metrics.json</a></p>
            </section>
            """
        )
    content = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Autoencoder Visual Evaluation</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
        img {{ max-width: 100%; border: 1px solid #ddd; }}
        section {{ margin-bottom: 36px; }}
      </style>
    </head>
    <body>
      <h1>Autoencoder Visual Evaluation</h1>
      <p>Checkpoint: {html.escape(str(checkpoint_path))}</p>
      {''.join(rows)}
    </body>
    </html>
    """
    (output_dir / "index.html").write_text(content, encoding="utf-8")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "part"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-parts", type=int, default=3)
    parser.add_argument("--part-indexes", nargs="*", type=int, default=None)
    parser.add_argument("--split", choices=["all", "train", "val", "unused"], default="all")
    parser.add_argument("--eval-n-points", type=int, default=None)
    parser.add_argument(
        "--metric-target-n-points",
        type=int,
        default=None,
        help="Use this many target samples for metrics/visuals while keeping model input at --eval-n-points/checkpoint n_points.",
    )
    parser.add_argument(
        "--sampling-floor-seed-offset",
        type=int,
        default=7919,
        help="Seed offset for the independent target sample used to estimate the per-part sampling floor.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--plot-point-limit", type=int, default=5000)
    parser.add_argument("--error-vmax-percentile", type=float, default=95.0)
    parser.add_argument(
        "--joint-distance",
        choices=["checkpoint", "on", "off"],
        default="checkpoint",
        help="Whether to use joints.json distance features at evaluation time.",
    )
    parser.add_argument(
        "--boundary-sample-fraction",
        type=float,
        default=None,
        help="Override checkpoint boundary sampling fraction for boundary-aware evaluation.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--write-html", dest="write_html", action="store_true", default=True)
    parser.add_argument("--no-html", dest="write_html", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    main()

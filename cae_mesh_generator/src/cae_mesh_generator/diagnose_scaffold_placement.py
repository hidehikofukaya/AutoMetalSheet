"""Diagnose scaffold placement quality for structured midsurface AE checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from .evaluate_autoencoder import (
    DistanceSummary,
    bbox_diagonal,
    build_dataset_from_checkpoint,
    fraction_within,
    load_model_from_checkpoint,
    nearest_distances,
    records_from_checkpoint,
    select_indexes,
    split_labels_from_checkpoint,
    summarize_distances,
    summarize_distances_or_zero,
    write_colored_ply,
)


@dataclass(frozen=True)
class ScaffoldPlacementMetrics:
    part_name: str
    record_index: int
    split: str
    scale_mm: float
    bbox_diagonal_mm: float
    n_scaffold: int
    n_target: int
    n_boundary: int
    n_crease: int
    n_corner: int
    scaffold_to_target_mm: DistanceSummary
    target_to_scaffold_mm: DistanceSummary
    boundary_to_scaffold_mm: DistanceSummary
    crease_to_scaffold_mm: DistanceSummary
    corner_to_scaffold_mm: DistanceSummary
    scaffold_within_5mm: float
    scaffold_within_10mm: float
    target_within_5mm: float
    target_within_10mm: float
    boundary_within_5mm: float
    boundary_within_10mm: float
    scaffold_p95_bbox_diag_pct: float
    target_p95_bbox_diag_pct: float


def main() -> None:
    args = parse_args()
    if args.eval_n_points is not None and args.eval_n_points <= 0:
        raise SystemExit("--eval-n-points must be positive when provided.")
    if args.metric_target_n_points <= 0:
        raise SystemExit("--metric-target-n-points must be positive.")

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "scaffold_diagnostic"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    records = records_from_checkpoint(payload)
    if not records:
        raise SystemExit(f"No records found in checkpoint: {checkpoint_path}")

    ckpt_args = payload.get("args", {})
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
    target_dataset = build_dataset_from_checkpoint(
        records=records,
        ckpt_args=ckpt_args,
        output_dir=output_dir,
        eval_n_points=args.metric_target_n_points,
        seed=args.seed,
        joint_distance_mode=args.joint_distance,
        boundary_sample_fraction=args.boundary_sample_fraction,
    )

    all_metrics: list[ScaffoldPlacementMetrics] = []
    for record_index in selected_indexes:
        item = input_dataset[record_index]
        target_item = target_dataset[record_index]
        part_name = str(item["name"])
        part_dir = output_dir / f"{record_index:03d}_{safe_name(part_name)}"
        part_dir.mkdir(parents=True, exist_ok=True)
        metrics = diagnose_part(
            model=model,
            item=item,
            target_item=target_item,
            record_index=record_index,
            split=split_by_index.get(record_index, "unknown"),
            device=device,
            part_dir=part_dir,
            write_ply=args.write_ply,
        )
        all_metrics.append(metrics)
        print(
            f"{part_name}: "
            f"scaf_p95={metrics.scaffold_to_target_mm.p95:.3f}mm "
            f"target_p95={metrics.target_to_scaffold_mm.p95:.3f}mm "
            f"boundary_p95={metrics.boundary_to_scaffold_mm.p95:.3f}mm"
        )

    write_metrics(output_dir, all_metrics)
    write_manifest(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        ckpt_args=ckpt_args,
        selected_indexes=selected_indexes,
        eval_settings={
            "input_n_points": int(args.eval_n_points or ckpt_args.get("n_points", 512)),
            "metric_target_n_points": int(args.metric_target_n_points),
            "boundary_sample_fraction": (
                float(args.boundary_sample_fraction)
                if args.boundary_sample_fraction is not None
                else float(ckpt_args.get("boundary_sample_fraction", 0.0))
            ),
        },
        metrics=all_metrics,
    )
    print(f"wrote scaffold diagnostic to {output_dir}")


def diagnose_part(
    model: torch.nn.Module,
    item: dict,
    target_item: dict,
    record_index: int,
    split: str,
    device: torch.device,
    part_dir: Path,
    write_ply: bool,
) -> ScaffoldPlacementMetrics:
    with torch.no_grad():
        points = item["points"][None, ...].to(device)
        features = item["features"][None, ...].to(device)
        pred = model(points, features)
    if "scaffold_points" not in pred:
        raise SystemExit("Checkpoint model does not emit scaffold_points; use a structured checkpoint.")

    input_center = item["center"].detach().cpu().numpy()
    input_scale = float(item["scale"].detach().cpu())
    target_center = target_item["center"].detach().cpu().numpy()
    target_scale = float(target_item["scale"].detach().cpu())

    scaffold_norm = pred["scaffold_points"][0].detach().cpu().numpy()
    if "active_scaffold_mask" in pred:
        active = pred["active_scaffold_mask"][0].detach().cpu().numpy().astype(bool)
        scaffold_norm = scaffold_norm[active]
    scaffold_world = scaffold_norm * input_scale + input_center[None, :]
    target_world = target_item["points"].detach().cpu().numpy() * target_scale + target_center[None, :]

    boundary_world = masked_target_world(target_item, "boundary", target_center, target_scale)
    crease_world = masked_target_world(target_item, "crease", target_center, target_scale)
    corner_world = masked_target_world(target_item, "corner", target_center, target_scale)
    if len(boundary_world) == 0:
        boundary_mask = target_item.get("boundary_mask")
        if boundary_mask is not None:
            mask = boundary_mask.detach().cpu().numpy().astype(bool)
            boundary_world = target_world[mask]

    scaffold_to_target, _ = nearest_distances(scaffold_world, target_world)
    target_to_scaffold, _ = nearest_distances(target_world, scaffold_world)
    boundary_to_scaffold = nearest_or_empty(boundary_world, scaffold_world)
    crease_to_scaffold = nearest_or_empty(crease_world, scaffold_world)
    corner_to_scaffold = nearest_or_empty(corner_world, scaffold_world)

    diag = max(bbox_diagonal(target_world), 1.0e-12)
    metrics = ScaffoldPlacementMetrics(
        part_name=str(item["name"]),
        record_index=int(record_index),
        split=split,
        scale_mm=target_scale,
        bbox_diagonal_mm=diag,
        n_scaffold=int(len(scaffold_world)),
        n_target=int(len(target_world)),
        n_boundary=int(len(boundary_world)),
        n_crease=int(len(crease_world)),
        n_corner=int(len(corner_world)),
        scaffold_to_target_mm=summarize_distances(scaffold_to_target),
        target_to_scaffold_mm=summarize_distances(target_to_scaffold),
        boundary_to_scaffold_mm=summarize_distances_or_zero(boundary_to_scaffold),
        crease_to_scaffold_mm=summarize_distances_or_zero(crease_to_scaffold),
        corner_to_scaffold_mm=summarize_distances_or_zero(corner_to_scaffold),
        scaffold_within_5mm=fraction_within(scaffold_to_target, 5.0),
        scaffold_within_10mm=fraction_within(scaffold_to_target, 10.0),
        target_within_5mm=fraction_within(target_to_scaffold, 5.0),
        target_within_10mm=fraction_within(target_to_scaffold, 10.0),
        boundary_within_5mm=fraction_within_or_zero(boundary_to_scaffold, 5.0),
        boundary_within_10mm=fraction_within_or_zero(boundary_to_scaffold, 10.0),
        scaffold_p95_bbox_diag_pct=100.0 * float(np.percentile(scaffold_to_target, 95)) / diag,
        target_p95_bbox_diag_pct=100.0 * float(np.percentile(target_to_scaffold, 95)) / diag,
    )

    if write_ply:
        write_colored_ply(part_dir / "target.ply", target_world, np.full((len(target_world), 3), [31, 119, 180]))
        write_colored_ply(
            part_dir / "scaffold.ply",
            scaffold_world,
            np.full((len(scaffold_world), 3), [44, 160, 44]),
        )
        if len(boundary_world):
            write_colored_ply(
                part_dir / "boundary_targets.ply",
                boundary_world,
                np.full((len(boundary_world), 3), [148, 103, 189]),
            )
    (part_dir / "metrics.json").write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
    return metrics


def masked_target_world(item: dict, name: str, center: np.ndarray, scale: float) -> np.ndarray:
    pts = item.get(f"scaffold_{name}_targets")
    mask = item.get(f"scaffold_{name}_target_mask")
    if pts is None or mask is None:
        return np.zeros((0, 3), dtype=np.float32)
    pts_np = pts.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy().astype(bool)
    if len(pts_np) == 0 or not mask_np.any():
        return np.zeros((0, 3), dtype=np.float32)
    return pts_np[mask_np] * scale + center[None, :]


def nearest_or_empty(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if len(query) == 0:
        return np.zeros((0,), dtype=np.float64)
    distances, _ = nearest_distances(query, reference)
    return distances


def fraction_within_or_zero(distances: np.ndarray, threshold: float) -> float:
    if len(distances) == 0:
        return 0.0
    return fraction_within(distances, threshold)


def write_metrics(output_dir: Path, metrics: list[ScaffoldPlacementMetrics]) -> None:
    (output_dir / "metrics.json").write_text(json.dumps([asdict(m) for m in metrics], indent=2), encoding="utf-8")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "record_index",
            "split",
            "part_name",
            "n_scaffold",
            "n_target",
            "n_boundary",
            "scaffold_p95_mm",
            "target_p95_mm",
            "boundary_p95_mm",
            "crease_p95_mm",
            "corner_p95_mm",
            "scaffold_within_5mm",
            "target_within_5mm",
            "boundary_within_5mm",
            "scaffold_p95_bbox_diag_pct",
            "target_p95_bbox_diag_pct",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics:
            writer.writerow(
                {
                    "record_index": m.record_index,
                    "split": m.split,
                    "part_name": m.part_name,
                    "n_scaffold": m.n_scaffold,
                    "n_target": m.n_target,
                    "n_boundary": m.n_boundary,
                    "scaffold_p95_mm": m.scaffold_to_target_mm.p95,
                    "target_p95_mm": m.target_to_scaffold_mm.p95,
                    "boundary_p95_mm": m.boundary_to_scaffold_mm.p95,
                    "crease_p95_mm": m.crease_to_scaffold_mm.p95,
                    "corner_p95_mm": m.corner_to_scaffold_mm.p95,
                    "scaffold_within_5mm": m.scaffold_within_5mm,
                    "target_within_5mm": m.target_within_5mm,
                    "boundary_within_5mm": m.boundary_within_5mm,
                    "scaffold_p95_bbox_diag_pct": m.scaffold_p95_bbox_diag_pct,
                    "target_p95_bbox_diag_pct": m.target_p95_bbox_diag_pct,
                }
            )
    write_aggregate_metrics(output_dir, metrics)


def write_aggregate_metrics(output_dir: Path, metrics: list[ScaffoldPlacementMetrics]) -> None:
    groups: dict[str, list[ScaffoldPlacementMetrics]] = {"all": list(metrics)}
    for metric in metrics:
        groups.setdefault(metric.split, []).append(metric)
    rows = [aggregate_group(label, rows) for label, rows in groups.items() if rows]
    (output_dir / "aggregate_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (output_dir / "aggregate_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else ["split", "count"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_group(label: str, metrics: list[ScaffoldPlacementMetrics]) -> dict[str, object]:
    worst_scaffold = max(metrics, key=lambda m: m.scaffold_to_target_mm.p95)
    worst_target = max(metrics, key=lambda m: m.target_to_scaffold_mm.p95)
    worst_boundary = max(metrics, key=lambda m: m.boundary_to_scaffold_mm.p95)
    return {
        "split": label,
        "count": len(metrics),
        "scaffold_p95_mean_mm": float(np.mean([m.scaffold_to_target_mm.p95 for m in metrics])),
        "target_p95_mean_mm": float(np.mean([m.target_to_scaffold_mm.p95 for m in metrics])),
        "boundary_p95_mean_mm": float(np.mean([m.boundary_to_scaffold_mm.p95 for m in metrics])),
        "crease_p95_mean_mm": float(np.mean([m.crease_to_scaffold_mm.p95 for m in metrics])),
        "corner_p95_mean_mm": float(np.mean([m.corner_to_scaffold_mm.p95 for m in metrics])),
        "scaffold_within_5mm_mean": float(np.mean([m.scaffold_within_5mm for m in metrics])),
        "target_within_5mm_mean": float(np.mean([m.target_within_5mm for m in metrics])),
        "boundary_within_5mm_mean": float(np.mean([m.boundary_within_5mm for m in metrics])),
        "scaffold_p95_bbox_diag_pct_mean": float(np.mean([m.scaffold_p95_bbox_diag_pct for m in metrics])),
        "target_p95_bbox_diag_pct_mean": float(np.mean([m.target_p95_bbox_diag_pct for m in metrics])),
        "worst_scaffold_p95_part": worst_scaffold.part_name,
        "worst_scaffold_p95_mm": worst_scaffold.scaffold_to_target_mm.p95,
        "worst_target_p95_part": worst_target.part_name,
        "worst_target_p95_mm": worst_target.target_to_scaffold_mm.p95,
        "worst_boundary_p95_part": worst_boundary.part_name,
        "worst_boundary_p95_mm": worst_boundary.boundary_to_scaffold_mm.p95,
    }


def write_manifest(
    output_dir: Path,
    checkpoint_path: Path,
    ckpt_args: dict,
    selected_indexes: list[int],
    eval_settings: dict[str, int | float],
    metrics: list[ScaffoldPlacementMetrics],
) -> None:
    manifest = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_args": ckpt_args,
        "selected_indexes": selected_indexes,
        "eval_settings": eval_settings,
        "metrics_count": len(metrics),
        "split_counts": split_counts(metrics),
        "metrics_json": str(output_dir / "metrics.json"),
        "metrics_csv": str(output_dir / "metrics.csv"),
        "aggregate_metrics_json": str(output_dir / "aggregate_metrics.json"),
        "aggregate_metrics_csv": str(output_dir / "aggregate_metrics.csv"),
    }
    (output_dir / "diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def split_counts(metrics: list[ScaffoldPlacementMetrics]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for metric in metrics:
        counts[metric.split] = counts.get(metric.split, 0) + 1
    return counts


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_") or "part"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-parts", type=int, default=0)
    parser.add_argument("--part-indexes", nargs="*", type=int, default=None)
    parser.add_argument("--split", choices=["all", "train", "val", "unused"], default="all")
    parser.add_argument("--eval-n-points", type=int, default=None)
    parser.add_argument("--metric-target-n-points", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--boundary-sample-fraction", type=float, default=None)
    parser.add_argument("--joint-distance", choices=["checkpoint", "on", "off"], default="checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--write-ply", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

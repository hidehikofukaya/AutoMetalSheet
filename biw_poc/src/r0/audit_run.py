"""Create a manifest-backed R0 audit bundle from stage mesh outputs."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import sys
import time
import uuid
from typing import Any

import numpy as np
import scipy
import trimesh
import yaml

from metrics import (
    load_triangle_mesh,
    mesh_metrics,
    point_to_surface_stats,
    sha256_file,
)


SCHEMA_VERSION = "r0_run_bundle_v1"


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def verify_event_chain(events: list[dict[str, Any]]) -> bool:
    previous: str | None = None
    for expected_sequence, stored in enumerate(events, start=1):
        item = dict(stored)
        event_hash = item.pop("event_hash", None)
        if item.get("sequence_no") != expected_sequence:
            return False
        if item.get("previous_event_hash") != previous:
            return False
        calculated = hashlib.sha256(_canonical_json_bytes(item)).hexdigest()
        if calculated != event_hash:
            return False
        previous = event_hash
    return True


def load_profile(path: pathlib.Path) -> dict[str, Any]:
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "profile_id",
        "profile_version",
        "usage",
        "eligible_for_phase_go",
        "reconstruction",
        "evaluation",
        "quality_gates",
        "phase_gate",
    }
    missing = sorted(required - set(profile or {}))
    if missing:
        raise ValueError(f"Profile missing required fields: {', '.join(missing)}")
    evaluation = profile["evaluation"]
    for key in ("sample_count", "sample_seed", "area_numeric_floor_mm2"):
        if key not in evaluation:
            raise ValueError(f"Profile evaluation missing {key}")
    if evaluation.get("formal_evaluation", True) and int(evaluation["sample_count"]) < 100_000:
        raise ValueError("Formal R0 evaluation requires at least 100,000 samples per direction")
    return profile


def parse_stage(value: str) -> tuple[str, pathlib.Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Stage must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Stage label is empty")
    return label, pathlib.Path(raw_path).expanduser().resolve()


def evaluate_quality_gates(
    metrics: dict[str, Any], quality_gates: dict[str, Any]
) -> list[dict[str, Any]]:
    checks = [
        (
            "non_manifold_edge_count",
            metrics["non_manifold_edge_count"],
            quality_gates["non_manifold_edge_count_max"],
            "<=",
        ),
        (
            "degenerate_face_count",
            metrics["degenerate_face_count"],
            quality_gates["degenerate_face_count_max"],
            "<=",
        ),
        (
            "duplicate_face_count",
            metrics["duplicate_face_count"],
            quality_gates["duplicate_face_count_max"],
            "<=",
        ),
        (
            "zero_length_edge_count",
            metrics["zero_length_edge_count"],
            quality_gates["zero_length_edge_count_max"],
            "<=",
        ),
        (
            "minimum_angle_deg",
            metrics["minimum_angle_deg"],
            quality_gates["minimum_angle_deg_min"],
            ">=",
        ),
        (
            "altitude_aspect_ratio_p95",
            metrics["altitude_aspect_ratio_p95"],
            quality_gates["altitude_aspect_ratio_p95_max"],
            "<=",
        ),
        (
            "altitude_aspect_ratio_max",
            metrics["altitude_aspect_ratio_max"],
            quality_gates["altitude_aspect_ratio_max_max"],
            "<=",
        ),
    ]
    results = []
    for metric_id, actual, threshold, operator in checks:
        passed = actual <= threshold if operator == "<=" else actual >= threshold
        results.append(
            {
                "gate_id": f"R0-QUALITY-{metric_id}",
                "metric_id": metric_id,
                "actual": actual,
                "operator": operator,
                "threshold": threshold,
                "severity": "HARD",
                "passed": bool(passed),
            }
        )
    return results


def audit(
    profile_path: pathlib.Path,
    reference_path: pathlib.Path,
    stages: list[tuple[str, pathlib.Path]],
    output_dir: pathlib.Path,
    part_id: str | None = None,
    part_family: str | None = None,
) -> pathlib.Path:
    profile = load_profile(profile_path)
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)
    for _, stage_path in stages:
        if not stage_path.exists():
            raise FileNotFoundError(stage_path)
    if not stages:
        raise ValueError("At least one --stage is required")

    output_dir.mkdir(parents=True, exist_ok=False)
    started = dt.datetime.now(dt.timezone.utc)
    run_id = f"r0-{started.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    audit_events: list[dict[str, Any]] = []
    previous_event_hash: str | None = None

    def event(event_type: str, **payload: Any) -> None:
        nonlocal previous_event_hash
        item = {
            "sequence_no": len(audit_events) + 1,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "event_type": event_type,
            "previous_event_hash": previous_event_hash,
            **payload,
        }
        item["event_hash"] = hashlib.sha256(_canonical_json_bytes(item)).hexdigest()
        previous_event_hash = item["event_hash"]
        audit_events.append(item)

    event("RUN_STARTED", run_id=run_id)
    reference_mesh = load_triangle_mesh(reference_path)
    event("REFERENCE_LOADED", path=str(reference_path))

    sample_count = int(profile["evaluation"]["sample_count"])
    seed = int(profile["evaluation"]["sample_seed"])
    area_floor = float(profile["evaluation"]["area_numeric_floor_mm2"])
    stage_records = []

    for index, (label, stage_path) in enumerate(stages):
        t0 = time.perf_counter()
        event("STAGE_EVALUATION_STARTED", stage=label, path=str(stage_path))
        mesh = load_triangle_mesh(stage_path)
        metrics = mesh_metrics(mesh, area_floor)
        stage_to_reference_samples = (
            output_dir / f"stage_{index:02d}_{label}_stage_to_reference_samples.npz"
        )
        reference_to_stage_samples = (
            output_dir / f"stage_{index:02d}_{label}_reference_to_stage_samples.npz"
        )
        metrics["stage_to_reference"] = point_to_surface_stats(
            mesh,
            reference_mesh,
            sample_count,
            seed + index * 2,
            stage_to_reference_samples,
        )
        metrics["reference_to_stage"] = point_to_surface_stats(
            reference_mesh,
            mesh,
            sample_count,
            seed + index * 2 + 1,
            reference_to_stage_samples,
        )
        gates = evaluate_quality_gates(metrics, profile["quality_gates"])
        stage_status = "PASS" if all(g["passed"] for g in gates) else "HARD_GATE_FAILED"
        metrics_path = output_dir / f"stage_{index:02d}_{label}_metrics.json"
        _write_json(metrics_path, metrics)
        record = {
            "order": index,
            "label": label,
            "artifact": {
                "path": str(stage_path),
                "sha256": sha256_file(stage_path),
            },
            "metrics_path": metrics_path.name,
            "metrics_sha256": sha256_file(metrics_path),
            "status": stage_status,
            "gates": gates,
            "elapsed_seconds": time.perf_counter() - t0,
        }
        stage_records.append(record)
        event(
            "STAGE_EVALUATION_FINISHED",
            stage=label,
            status=stage_status,
            failed_gate_ids=[g["gate_id"] for g in gates if not g["passed"]],
        )

    minimum_parts = int(profile["phase_gate"]["minimum_out_of_fold_parts"])
    represented_parts = 1
    hard_failed_stages = [
        stage["label"] for stage in stage_records if stage["status"] != "PASS"
    ]
    if not bool(profile["eligible_for_phase_go"]):
        phase_status = "NO_GO"
        phase_reasons = ["Profile is comparison-only and not eligible for phase Go"]
    elif hard_failed_stages:
        phase_status = "NO_GO"
        phase_reasons = [
            "Hard quality gates failed for stages: " + ", ".join(hard_failed_stages)
        ]
    elif represented_parts < minimum_parts:
        phase_status = "PHASE_INSUFFICIENT_VALIDATION_DATA"
        phase_reasons = [
            f"Only {represented_parts} part audited; {minimum_parts} out-of-fold parts required"
        ]
    else:
        phase_status = "TECHNICAL_GO"
        phase_reasons = []

    event("RUN_FINISHED", phase_gate_status=phase_status)
    audit_log_path = output_dir / "audit_events.jsonl"
    with audit_log_path.open("w", encoding="utf-8") as handle:
        for item in audit_events:
            handle.write(json.dumps(_json_value(item), ensure_ascii=False) + "\n")
    if not verify_event_chain(audit_events):
        raise RuntimeError("Internal AuditEvent chain verification failed")

    completed = dt.datetime.now(dt.timezone.utc)
    profile_snapshot = copy.deepcopy(profile)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "run_kind": "R0_OUTPUT_AUDIT",
        "part_id": part_id or reference_path.stem,
        "part_family": part_family or "UNASSIGNED",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "profile": profile_snapshot,
        "profile_source": {
            "path": str(profile_path),
            "sha256": sha256_file(profile_path),
        },
        "reference": {
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
        },
        "stages": stage_records,
        "refinement_run_status": (
            "PASS"
            if all(stage["status"] == "PASS" for stage in stage_records)
            else "MANUAL_REVIEW"
        ),
        "phase_gate_status": phase_status,
        "phase_gate_reasons": phase_reasons,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "trimesh": trimesh.__version__,
            "pid": os.getpid(),
        },
        "audit": {
            "log_path": audit_log_path.name,
            "log_sha256": sha256_file(audit_log_path),
            "final_sequence_no": len(audit_events),
            "final_event_hash": previous_event_hash,
            "chain_verified": True,
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=pathlib.Path, required=True)
    parser.add_argument("--reference", type=pathlib.Path, required=True)
    parser.add_argument("--stage", type=parse_stage, action="append", default=[])
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--part-id")
    parser.add_argument("--part-family")
    args = parser.parse_args()
    manifest = audit(
        args.profile.resolve(),
        args.reference.resolve(),
        args.stage,
        args.output_dir.resolve(),
        part_id=args.part_id,
        part_family=args.part_family,
    )
    print(manifest)


if __name__ == "__main__":
    main()

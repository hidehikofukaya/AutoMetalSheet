"""Aggregate R0 run manifests into a family-aware cohort Go/No-Go report."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
from collections import defaultdict
from typing import Any

import numpy as np
import yaml

from audit_run import verify_event_chain
from metrics import nearest_rank, sha256_file


SCHEMA_VERSION = "r0_cohort_manifest_v1"


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def verify_part_bundle(manifest_path: pathlib.Path, manifest: dict[str, Any]) -> list[str]:
    errors = []
    run_dir = manifest_path.parent

    def verify(path: pathlib.Path, expected: str, label: str) -> None:
        if not path.exists():
            errors.append(f"{label} missing")
        elif sha256_file(path) != expected:
            errors.append(f"{label} hash mismatch")

    verify(
        pathlib.Path(manifest["profile_source"]["path"]),
        manifest["profile_source"]["sha256"],
        "profile",
    )
    verify(
        pathlib.Path(manifest["reference"]["path"]),
        manifest["reference"]["sha256"],
        "reference",
    )
    for stage in manifest.get("stages", []):
        verify(
            pathlib.Path(stage["artifact"]["path"]),
            stage["artifact"]["sha256"],
            f"stage {stage['label']}",
        )
        verify(
            run_dir / stage["metrics_path"],
            stage["metrics_sha256"],
            f"metrics {stage['label']}",
        )
    audit = manifest.get("audit", {})
    audit_path = run_dir / audit.get("log_path", "")
    if audit.get("log_sha256"):
        verify(audit_path, audit["log_sha256"], "audit log")
    if audit_path.exists():
        events = [
            json.loads(line)
            for line in audit_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not verify_event_chain(events):
            errors.append("audit event chain invalid")
        elif events and events[-1]["event_hash"] != audit.get("final_event_hash"):
            errors.append("audit final hash mismatch")
    return errors


def stage_by_label(manifest: dict[str, Any], label: str) -> dict[str, Any]:
    matches = [stage for stage in manifest.get("stages", []) if stage["label"] == label]
    if len(matches) != 1:
        raise ValueError(
            f"Run {manifest.get('run_id')} requires exactly one stage {label!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def stage_metrics(
    manifest_path: pathlib.Path, stage: dict[str, Any]
) -> dict[str, Any]:
    path = manifest_path.parent / stage["metrics_path"]
    if not path.exists():
        raise FileNotFoundError(path)
    if sha256_file(path) != stage["metrics_sha256"]:
        raise ValueError(f"Metrics SHA-256 mismatch: {path}")
    return load_json(path)


def cluster_bootstrap(
    records: list[dict[str, Any]], iterations: int, seed: int
) -> dict[str, float]:
    by_family: dict[str, list[float]] = defaultdict(list)
    for record in records:
        by_family[record["part_family"]].append(record["primary_improvement"])
    families = sorted(by_family)
    if not families:
        return {"estimate": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
    family_means = np.array(
        [np.mean(by_family[family]) for family in families], dtype=np.float64
    )
    estimate = float(family_means.mean())
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        selected = rng.integers(0, len(family_means), size=len(family_means))
        bootstrap[index] = family_means[selected].mean()
    return {
        "estimate": estimate,
        "ci95_low": nearest_rank(bootstrap, 2.5),
        "ci95_high": nearest_rank(bootstrap, 97.5),
    }


def leave_one_family_out(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    families = sorted({record["part_family"] for record in records})
    results = []
    for omitted in families:
        included = [record for record in records if record["part_family"] != omitted]
        by_family: dict[str, list[float]] = defaultdict(list)
        for record in included:
            by_family[record["part_family"]].append(record["primary_improvement"])
        estimate = (
            float(np.mean([np.mean(values) for values in by_family.values()]))
            if by_family
            else math.nan
        )
        results.append({"omitted_family": omitted, "estimate": estimate})
    return results


def aggregate(
    manifest_paths: list[pathlib.Path],
    output_path: pathlib.Path,
    baseline_label: str,
    candidate_label: str,
    primary_metric_path: tuple[str, str],
    minimum_parts: int,
    minimum_families: int,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    expected_parts: dict[str, str] | None = None,
    power_analysis_satisfied: bool = True,
    minimum_improvement_ratio: float = 0.20,
    baseline_numeric_floor: float = 1.0e-9,
    verify_integrity: bool = True,
) -> pathlib.Path:
    if not manifest_paths:
        raise ValueError("At least one run manifest is required")

    part_records = []
    seen_parts: set[str] = set()
    profile_hashes: set[str] = set()
    evaluator_ids: set[tuple[str, str]] = set()
    for manifest_path in sorted(path.resolve() for path in manifest_paths):
        manifest = load_json(manifest_path)
        integrity_errors = (
            verify_part_bundle(manifest_path, manifest) if verify_integrity else []
        )
        part_id = manifest.get("part_id")
        family = manifest.get("part_family")
        if not part_id or not family or family == "UNASSIGNED":
            raise ValueError(f"Run lacks assigned part/family: {manifest_path}")
        if part_id in seen_parts:
            raise ValueError(f"Duplicate part_id in cohort: {part_id}")
        seen_parts.add(part_id)
        profile_hashes.add(manifest["profile_source"]["sha256"])

        baseline = stage_by_label(manifest, baseline_label)
        candidate = stage_by_label(manifest, candidate_label)
        baseline_metrics = stage_metrics(manifest_path, baseline)
        candidate_metrics = stage_metrics(manifest_path, candidate)
        direction, metric_name = primary_metric_path
        baseline_value = float(baseline_metrics[direction][metric_name])
        candidate_value = float(candidate_metrics[direction][metric_name])
        absolute_improvement = baseline_value - candidate_value
        improvement = (
            absolute_improvement / baseline_value
            if abs(baseline_value) > baseline_numeric_floor
            else absolute_improvement
        )
        evaluator_ids.add(
            (
                str(baseline_metrics[direction].get("metric_definition_id")),
                str(candidate_metrics[direction].get("metric_definition_id")),
            )
        )
        failed_gates = [
            gate["gate_id"] for gate in candidate.get("gates", []) if not gate["passed"]
        ]
        part_records.append(
            {
                "part_id": part_id,
                "part_family": family,
                "run_id": manifest["run_id"],
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "baseline_label": baseline_label,
                "candidate_label": candidate_label,
                "baseline_primary_value": baseline_value,
                "candidate_primary_value": candidate_value,
                "absolute_primary_improvement": absolute_improvement,
                "primary_improvement": improvement,
                "improvement_mode": (
                    "ratio"
                    if abs(baseline_value) > baseline_numeric_floor
                    else "absolute"
                ),
                "candidate_status": candidate["status"],
                "candidate_failed_gate_ids": failed_gates,
                "integrity_errors": integrity_errors,
            }
        )

    families = sorted({record["part_family"] for record in part_records})
    hard_failures = [
        record["part_id"] for record in part_records if record["candidate_failed_gate_ids"]
    ]
    profile_consistent = len(profile_hashes) == 1
    evaluator_consistent = len(evaluator_ids) == 1
    integrity_failures = [
        record["part_id"] for record in part_records if record["integrity_errors"]
    ]
    observed_parts = {record["part_id"]: record["part_family"] for record in part_records}
    missing_parts = (
        sorted(set(expected_parts) - set(observed_parts)) if expected_parts else []
    )
    unexpected_parts = (
        sorted(set(observed_parts) - set(expected_parts)) if expected_parts else []
    )
    family_mismatches = (
        sorted(
            part_id
            for part_id in set(observed_parts) & set(expected_parts)
            if observed_parts[part_id] != expected_parts[part_id]
        )
        if expected_parts
        else []
    )
    bootstrap = cluster_bootstrap(part_records, bootstrap_iterations, bootstrap_seed)

    reasons = []
    if integrity_failures:
        status = "NO_GO"
        reasons.append("Bundle integrity failed: " + ", ".join(integrity_failures))
    elif missing_parts or unexpected_parts or family_mismatches:
        status = "PHASE_INSUFFICIENT_VALIDATION_DATA"
        reasons.append(
            f"Cohort mismatch; missing={missing_parts}, unexpected={unexpected_parts}, "
            f"family_mismatch={family_mismatches}"
        )
    elif not profile_consistent:
        status = "NO_GO"
        reasons.append("Cohort contains multiple profile hashes")
    elif not evaluator_consistent:
        status = "NO_GO"
        reasons.append("Cohort contains multiple evaluator definitions")
    elif hard_failures:
        status = "NO_GO"
        reasons.append("Candidate hard gates failed: " + ", ".join(hard_failures))
    elif len(part_records) < minimum_parts or len(families) < minimum_families:
        status = "PHASE_INSUFFICIENT_VALIDATION_DATA"
        reasons.append(
            f"Need >= {minimum_parts} parts / {minimum_families} families; "
            f"have {len(part_records)} / {len(families)}"
        )
    elif not power_analysis_satisfied:
        status = "PHASE_INSUFFICIENT_VALIDATION_DATA"
        reasons.append("A-priori family-cluster power analysis is not satisfied")
    elif bootstrap["estimate"] < minimum_improvement_ratio:
        status = "NO_GO"
        reasons.append(
            f"Family-macro improvement {bootstrap['estimate']:.4g} is below "
            f"{minimum_improvement_ratio:.4g}"
        )
    elif bootstrap["ci95_low"] <= 0:
        status = "NO_GO"
        reasons.append("Family-cluster bootstrap 95% CI does not exclude zero")
    else:
        status = "TECHNICAL_GO"

    family_records = []
    for family in families:
        members = [r for r in part_records if r["part_family"] == family]
        family_records.append(
            {
                "part_family": family,
                "part_count": len(members),
                "mean_primary_improvement": float(
                    np.mean([r["primary_improvement"] for r in members])
                ),
                "hard_failure_count": sum(
                    bool(r["candidate_failed_gate_ids"]) for r in members
                ),
                "worst_part_improvement": float(
                    min(r["primary_improvement"] for r in members)
                ),
            }
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "run_kind": "R0_COHORT_AGGREGATE",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "primary_metric": {
            "direction": primary_metric_path[0],
            "metric": primary_metric_path[1],
            "improvement_definition": (
                "(baseline - candidate) / baseline; positive is better; "
                "absolute fallback near zero"
            ),
            "minimum_family_macro_improvement": minimum_improvement_ratio,
        },
        "requirements": {
            "minimum_parts": minimum_parts,
            "minimum_families": minimum_families,
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_seed": bootstrap_seed,
        },
        "part_count": len(part_records),
        "family_count": len(families),
        "profile_hash_consistent": profile_consistent,
        "evaluator_consistent": evaluator_consistent,
        "cohort_integrity": {
            "expected_part_count": len(expected_parts) if expected_parts else None,
            "missing_parts": missing_parts,
            "unexpected_parts": unexpected_parts,
            "family_mismatches": family_mismatches,
            "bundle_integrity_failures": integrity_failures,
        },
        "power_analysis_satisfied": power_analysis_satisfied,
        "phase_gate_status": status,
        "phase_gate_reasons": reasons,
        "cluster_bootstrap": bootstrap,
        "leave_one_family_out": leave_one_family_out(part_records),
        "families": family_records,
        "parts": part_records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, action="append", default=[])
    parser.add_argument("--root", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--baseline-label", default="coarse")
    parser.add_argument("--candidate-label", default="refine3")
    parser.add_argument("--primary-direction", default="reference_to_stage")
    parser.add_argument("--primary-metric", default="p95_mm")
    parser.add_argument("--minimum-parts", type=int, default=8)
    parser.add_argument("--minimum-families", type=int, default=3)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260621)
    parser.add_argument("--cohort-config", type=pathlib.Path)
    args = parser.parse_args()
    manifests = list(args.manifest)
    if args.root:
        manifests.extend(args.root.rglob("run_manifest.json"))
    expected_parts = None
    power_satisfied = True
    minimum_improvement_ratio = 0.20
    if args.cohort_config:
        cohort_config = yaml.safe_load(
            args.cohort_config.read_text(encoding="utf-8")
        )
        expected_parts = {
            item["part_id"]: item["part_family"]
            for item in cohort_config["members"]
        }
        power_satisfied = bool(cohort_config["power_analysis_satisfied"])
        minimum_improvement_ratio = float(
            cohort_config["minimum_family_macro_improvement_ratio"]
        )
    output = aggregate(
        manifests,
        args.output.resolve(),
        args.baseline_label,
        args.candidate_label,
        (args.primary_direction, args.primary_metric),
        args.minimum_parts,
        args.minimum_families,
        args.bootstrap_iterations,
        args.bootstrap_seed,
        expected_parts=expected_parts,
        power_analysis_satisfied=power_satisfied,
        minimum_improvement_ratio=minimum_improvement_ratio,
    )
    print(output)


if __name__ == "__main__":
    main()

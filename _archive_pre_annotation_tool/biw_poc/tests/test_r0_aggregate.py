from __future__ import annotations

import json
import pathlib
import sys

R0_DIR = pathlib.Path(__file__).parents[1] / "src" / "r0"
sys.path.insert(0, str(R0_DIR))

from aggregate_runs import aggregate
from metrics import sha256_file


def create_run(
    root: pathlib.Path,
    part_id: str,
    family: str,
    baseline: float,
    candidate: float,
    candidate_pass: bool = True,
) -> pathlib.Path:
    run_dir = root / part_id
    run_dir.mkdir()
    stages = []
    for index, (label, value, passed) in enumerate(
        [("coarse", baseline, True), ("candidate", candidate, candidate_pass)]
    ):
        metrics = {
            "reference_to_stage": {"p95_mm": value},
            "stage_to_reference": {"p95_mm": value},
        }
        metrics_path = run_dir / f"{label}.json"
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        stages.append(
            {
                "label": label,
                "metrics_path": metrics_path.name,
                "metrics_sha256": sha256_file(metrics_path),
                "status": "PASS" if passed else "HARD_GATE_FAILED",
                "gates": [
                    {
                        "gate_id": "QUALITY",
                        "passed": passed,
                    }
                ],
            }
        )
    manifest = {
        "run_id": f"run-{part_id}",
        "part_id": part_id,
        "part_family": family,
        "profile_source": {"sha256": "same-profile"},
        "stages": stages,
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_aggregate_requires_parts_and_families(tmp_path):
    manifests = [
        create_run(tmp_path, "p1", "f1", 4.0, 2.0),
        create_run(tmp_path, "p2", "f2", 5.0, 3.0),
    ]
    output = tmp_path / "cohort.json"
    aggregate(
        manifests,
        output,
        "coarse",
        "candidate",
        ("reference_to_stage", "p95_mm"),
        minimum_parts=8,
        minimum_families=3,
        bootstrap_iterations=100,
        bootstrap_seed=1,
        power_analysis_satisfied=True,
        verify_integrity=False,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["phase_gate_status"] == "PHASE_INSUFFICIENT_VALIDATION_DATA"


def test_aggregate_hard_failure_is_no_go(tmp_path):
    manifests = [
        create_run(tmp_path, "p1", "f1", 4.0, 2.0, candidate_pass=False),
        create_run(tmp_path, "p2", "f2", 5.0, 3.0),
        create_run(tmp_path, "p3", "f3", 6.0, 4.0),
    ]
    output = tmp_path / "cohort.json"
    aggregate(
        manifests,
        output,
        "coarse",
        "candidate",
        ("reference_to_stage", "p95_mm"),
        minimum_parts=3,
        minimum_families=3,
        bootstrap_iterations=100,
        bootstrap_seed=1,
        power_analysis_satisfied=True,
        verify_integrity=False,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["phase_gate_status"] == "NO_GO"


def test_aggregate_can_reach_technical_go(tmp_path):
    manifests = []
    for family_index in range(3):
        family = f"f{family_index}"
        for part_index in range(3):
            manifests.append(
                create_run(
                    tmp_path,
                    f"p{family_index}{part_index}",
                    family,
                    baseline=5.0 + part_index,
                    candidate=2.0 + part_index,
                )
            )
    output = tmp_path / "cohort.json"
    aggregate(
        manifests,
        output,
        "coarse",
        "candidate",
        ("reference_to_stage", "p95_mm"),
        minimum_parts=8,
        minimum_families=3,
        bootstrap_iterations=1000,
        bootstrap_seed=1,
        power_analysis_satisfied=True,
        minimum_improvement_ratio=0.20,
        verify_integrity=False,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["phase_gate_status"] == "TECHNICAL_GO"
    assert result["cluster_bootstrap"]["ci95_low"] > 0

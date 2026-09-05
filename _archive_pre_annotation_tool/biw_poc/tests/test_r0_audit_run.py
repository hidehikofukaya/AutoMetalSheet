from __future__ import annotations

import json
import pathlib
import sys

import trimesh
import yaml

R0_DIR = pathlib.Path(__file__).parents[1] / "src" / "r0"
sys.path.insert(0, str(R0_DIR))

from audit_run import audit, verify_event_chain


def test_audit_bundle_is_created_and_fail_closed_on_sample_count(tmp_path):
    reference = trimesh.creation.icosphere(subdivisions=1, radius=10.0)
    stage = trimesh.creation.icosphere(subdivisions=1, radius=10.2)
    reference_path = tmp_path / "reference.ply"
    stage_path = tmp_path / "stage.ply"
    reference.export(reference_path)
    stage.export(stage_path)
    profile = {
        "profile_id": "TEST",
        "profile_version": 1,
        "usage": "baseline",
        "eligible_for_phase_go": True,
        "reconstruction": {
            "grid_res": 16,
            "band_factor": 1.0,
            "n_proj_iters": 1,
            "nbr_sz": 4,
            "refine_rounds": 0,
            "conf_threshold": 0.0,
            "input_dist_threshold_mm": 10.0,
            "prune_dist_threshold_mm": 20.0,
        },
        "evaluation": {
            "formal_evaluation": False,
            "sample_count": 500,
            "sample_seed": 11,
            "area_numeric_floor_mm2": 1.0e-12,
        },
        "quality_gates": {
            "non_manifold_edge_count_max": 0,
            "degenerate_face_count_max": 0,
            "duplicate_face_count_max": 0,
            "zero_length_edge_count_max": 0,
            "minimum_angle_deg_min": 1.0,
            "altitude_aspect_ratio_p95_max": 100.0,
            "altitude_aspect_ratio_max_max": 100.0,
        },
        "phase_gate": {
            "minimum_out_of_fold_parts": 8,
            "minimum_part_families": 3,
        },
    }
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump(profile), encoding="utf-8")
    output = tmp_path / "run"
    manifest_path = audit(
        profile_path, reference_path, [("candidate", stage_path)], output
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["phase_gate_status"] == "PHASE_INSUFFICIENT_VALIDATION_DATA"
    assert manifest["part_family"] == "UNASSIGNED"
    assert manifest["stages"][0]["status"] == "PASS"
    assert (output / manifest["stages"][0]["metrics_path"]).exists()
    assert (output / "audit_events.jsonl").exists()
    events = [
        json.loads(line)
        for line in (output / "audit_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert verify_event_chain(events)

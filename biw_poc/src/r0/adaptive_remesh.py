"""Deterministic bounded longest-edge refinement baseline for triangle meshes."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from dataclasses import asdict, dataclass

import numpy as np
import trimesh

from metrics import load_triangle_mesh, mesh_metrics


@dataclass(frozen=True)
class RefinementConfig:
    target_edge_mm: float
    split_ratio: float = 4.0 / 3.0
    h_floor_mm: float = 0.5
    max_sweeps: int = 8
    max_split_edge_fraction_per_sweep: float = 0.10
    max_vertex_growth_ratio_per_sweep: float = 1.5
    hard_max_vertices: int = 500_000
    area_ratio_tolerance: float = 1.0e-6
    quality_epsilon: float = 1.0e-9


def canonical_mesh_hash(
    vertices: np.ndarray, faces: np.ndarray, coordinate_quantum_mm: float = 1.0e-6
) -> str:
    quantized = np.rint(np.asarray(vertices) / coordinate_quantum_mm).astype("<i8")
    canonical_faces = []
    for face in np.asarray(faces, dtype=np.int64):
        rotations = [
            tuple(face),
            tuple(np.roll(face, -1)),
            tuple(np.roll(face, -2)),
        ]
        canonical_faces.append(min(rotations))
    canonical_faces = np.asarray(sorted(canonical_faces), dtype="<i8")
    digest = hashlib.sha256()
    digest.update(quantized.tobytes(order="C"))
    digest.update(canonical_faces.tobytes(order="C"))
    return digest.hexdigest()


def short_edge_count(
    vertices: np.ndarray, faces: np.ndarray, threshold_mm: float
) -> int:
    edges, _ = unique_edges_with_faces(faces)
    if len(edges) == 0:
        return 0
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    return int(np.count_nonzero(lengths < threshold_mm))


def unique_edges_with_faces(faces: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, (v0, v1, v2) in enumerate(np.asarray(faces, dtype=np.int64)):
        for a, b in ((v0, v1), (v1, v2), (v2, v0)):
            edge = (int(min(a, b)), int(max(a, b)))
            edge_to_faces.setdefault(edge, []).append(face_index)
    edges = np.array(sorted(edge_to_faces), dtype=np.int64)
    return edges, [edge_to_faces[tuple(edge)] for edge in edges]


def select_non_conflicting_edges(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: RefinementConfig,
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    edges, incident_faces = unique_edges_with_faces(faces)
    if len(edges) == 0:
        return np.empty((0, 2), dtype=np.int64), {"reason": "NO_EDGES"}
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    basic_eligible = np.where(
        (lengths > config.split_ratio * config.target_edge_mm)
        & (lengths / 2.0 >= config.h_floor_mm)
    )[0]
    eligible = []
    rejected_child_floor = 0
    for edge_index in basic_eligible:
        edge = edges[edge_index]
        midpoint = (vertices[edge[0]] + vertices[edge[1]]) / 2.0
        child_lengths = [lengths[edge_index] / 2.0]
        for face_index in incident_faces[edge_index]:
            opposite = next(
                vertex
                for vertex in faces[face_index]
                if vertex != edge[0] and vertex != edge[1]
            )
            child_lengths.append(float(np.linalg.norm(midpoint - vertices[opposite])))
        if min(child_lengths) >= config.h_floor_mm:
            eligible.append(int(edge_index))
        else:
            rejected_child_floor += 1
    eligible = np.asarray(eligible, dtype=np.int64)
    order = sorted(
        eligible.tolist(),
        key=lambda index: (-float(lengths[index]), int(edges[index, 0]), int(edges[index, 1])),
    )
    fraction_limit = max(
        0, int(np.floor(config.max_split_edge_fraction_per_sweep * len(edges)))
    )
    growth_limit = max(
        0,
        int(
            np.floor(
                (config.max_vertex_growth_ratio_per_sweep - 1.0) * len(vertices)
            )
        ),
    )
    hard_limit = max(0, config.hard_max_vertices - len(vertices))
    selection_limit = min(fraction_limit, growth_limit, hard_limit)
    if selection_limit <= 0:
        return np.empty((0, 2), dtype=np.int64), {
            "reason": "BUDGET_EXHAUSTED",
            "eligible_edge_count": int(len(eligible)),
            "rejected_child_floor_count": int(rejected_child_floor),
        }

    occupied_faces: set[int] = set()
    selected_indices = []
    for edge_index in order:
        faces_for_edge = incident_faces[edge_index]
        if any(face_index in occupied_faces for face_index in faces_for_edge):
            continue
        selected_indices.append(edge_index)
        occupied_faces.update(faces_for_edge)
        if len(selected_indices) >= selection_limit:
            break
    selected = edges[np.asarray(selected_indices, dtype=np.int64)]
    return selected, {
        "reason": "SELECTED" if len(selected) else "NO_NON_CONFLICTING_EDGE",
        "unique_edge_count": int(len(edges)),
        "eligible_edge_count": int(len(eligible)),
        "rejected_child_floor_count": int(rejected_child_floor),
        "selected_edge_count": int(len(selected)),
        "selection_limit": int(selection_limit),
        "max_eligible_edge_mm": (
            float(lengths[eligible].max()) if len(eligible) else 0.0
        ),
    }


def split_edges_conformingly(
    vertices: np.ndarray, faces: np.ndarray, selected_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if len(selected_edges) == 0:
        return vertices.copy(), faces.copy()
    selected = {
        (int(edge[0]), int(edge[1])): len(vertices) + index
        for index, edge in enumerate(selected_edges)
    }
    midpoints = (
        vertices[selected_edges[:, 0]] + vertices[selected_edges[:, 1]]
    ) / 2.0
    new_vertices = np.concatenate([vertices, midpoints], axis=0)
    new_faces: list[list[int]] = []
    for v0, v1, v2 in np.asarray(faces, dtype=np.int64):
        candidates = [
            ((min(v0, v1), max(v0, v1)), (v0, v1, v2)),
            ((min(v1, v2), max(v1, v2)), (v1, v2, v0)),
            ((min(v2, v0), max(v2, v0)), (v2, v0, v1)),
        ]
        matching = [(edge, cyclic) for edge, cyclic in candidates if edge in selected]
        if not matching:
            new_faces.append([int(v0), int(v1), int(v2)])
            continue
        if len(matching) != 1:
            raise RuntimeError("Selection invariant violated: multiple split edges in one face")
        edge, (a, b, opposite) = matching[0]
        midpoint = selected[edge]
        new_faces.append([int(a), midpoint, int(opposite)])
        new_faces.append([midpoint, int(b), int(opposite)])
    return new_vertices, np.asarray(new_faces, dtype=np.int64)


def refine(
    mesh: trimesh.Trimesh, config: RefinementConfig
) -> tuple[list[trimesh.Trimesh], list[dict]]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    stages = []
    reports = []
    current_mesh = trimesh.Trimesh(
        vertices=vertices.copy(), faces=faces.copy(), process=False
    )
    current_metrics = mesh_metrics(current_mesh, area_floor=1.0e-12)
    current_short_edges = short_edge_count(vertices, faces, config.h_floor_mm)
    for sweep in range(1, config.max_sweeps + 1):
        selected, report = select_non_conflicting_edges(vertices, faces, config)
        report = {"sweep": sweep, **report}
        if len(selected) == 0:
            reports.append(report)
            break
        previous_vertices = len(vertices)
        parent_vertices = vertices
        parent_faces = faces
        parent_hash = canonical_mesh_hash(parent_vertices, parent_faces)
        candidate_vertices, candidate_faces = split_edges_conformingly(
            parent_vertices, parent_faces, selected
        )
        candidate_mesh = trimesh.Trimesh(
            vertices=candidate_vertices, faces=candidate_faces, process=False
        )
        candidate_metrics = mesh_metrics(candidate_mesh, area_floor=1.0e-12)
        candidate_short_edges = short_edge_count(
            candidate_vertices, candidate_faces, config.h_floor_mm
        )
        area_ratio = (
            candidate_metrics["surface_area_mm2"] / current_metrics["surface_area_mm2"]
        )
        gate_failures = []
        for metric_name in (
            "non_manifold_edge_count",
            "duplicate_face_count",
            "degenerate_face_count",
        ):
            if candidate_metrics[metric_name] > current_metrics[metric_name]:
                gate_failures.append(f"{metric_name}_INCREASED")
        for metric_name in (
            "vertex_connected_component_count",
            "face_connected_component_count",
        ):
            if candidate_metrics[metric_name] != current_metrics[metric_name]:
                gate_failures.append(f"{metric_name}_CHANGED")
        if (
            candidate_metrics["boundary_component_count"]
            != current_metrics["boundary_component_count"]
        ):
            gate_failures.append("BOUNDARY_COMPONENT_COUNT_CHANGED")
        if candidate_short_edges > current_short_edges:
            gate_failures.append("SHORT_EDGE_COUNT_INCREASED")
        if not candidate_metrics["is_winding_consistent"]:
            gate_failures.append("WINDING_INCONSISTENT")
        if abs(area_ratio - 1.0) > config.area_ratio_tolerance:
            gate_failures.append("SURFACE_AREA_CHANGED")
        if (
            candidate_metrics["minimum_angle_deg"]
            + config.quality_epsilon
            < current_metrics["minimum_angle_deg"]
        ):
            gate_failures.append("MINIMUM_ANGLE_WORSENED")
        if (
            candidate_metrics["altitude_aspect_ratio_p95"]
            > current_metrics["altitude_aspect_ratio_p95"] + config.quality_epsilon
        ):
            gate_failures.append("ASPECT_P95_WORSENED")

        candidate_hash = canonical_mesh_hash(candidate_vertices, candidate_faces)
        report.update(
            {
                "vertex_count_before": previous_vertices,
                "vertex_count_after": int(len(candidate_vertices)),
                "face_count_after": int(len(candidate_faces)),
                "vertex_growth_ratio": float(
                    len(candidate_vertices) / previous_vertices
                ),
                "parent_mesh_hash": parent_hash,
                "candidate_mesh_hash": candidate_hash,
                "area_ratio": float(area_ratio),
                "minimum_angle_before": current_metrics["minimum_angle_deg"],
                "minimum_angle_after": candidate_metrics["minimum_angle_deg"],
                "aspect_p95_before": current_metrics["altitude_aspect_ratio_p95"],
                "aspect_p95_after": candidate_metrics["altitude_aspect_ratio_p95"],
                "short_edge_count_before": current_short_edges,
                "short_edge_count_after": candidate_short_edges,
                "boundary_component_count_before": current_metrics[
                    "boundary_component_count"
                ],
                "boundary_component_count_after": candidate_metrics[
                    "boundary_component_count"
                ],
                "gate_failures": gate_failures,
            }
        )
        if gate_failures:
            report["decision"] = "ROLLBACK"
            report["rollback_result_hash"] = canonical_mesh_hash(
                parent_vertices, parent_faces
            )
            reports.append(report)
            break

        vertices, faces = candidate_vertices, candidate_faces
        current_metrics = candidate_metrics
        current_short_edges = candidate_short_edges
        report["decision"] = "COMMIT"
        report["committed_mesh_hash"] = candidate_hash
        reports.append(report)
        stages.append(candidate_mesh)
    return stages, reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--target-edge-mm", type=float, required=True)
    parser.add_argument("--h-floor-mm", type=float, default=0.5)
    parser.add_argument("--max-sweeps", type=int, default=8)
    parser.add_argument("--max-split-fraction", type=float, default=0.10)
    parser.add_argument("--max-growth-ratio", type=float, default=1.5)
    parser.add_argument("--hard-max-vertices", type=int, default=500_000)
    args = parser.parse_args()
    config = RefinementConfig(
        target_edge_mm=args.target_edge_mm,
        h_floor_mm=args.h_floor_mm,
        max_sweeps=args.max_sweeps,
        max_split_edge_fraction_per_sweep=args.max_split_fraction,
        max_vertex_growth_ratio_per_sweep=args.max_growth_ratio,
        hard_max_vertices=args.hard_max_vertices,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    mesh = load_triangle_mesh(args.input.resolve())
    stages, reports = refine(mesh, config)
    for index, stage in enumerate(stages, start=1):
        stage.export(output_dir / f"adaptive_sweep{index}.ply")
    final_mesh = stages[-1] if stages else mesh
    unresolved_short_edges = short_edge_count(
        np.asarray(final_mesh.vertices),
        np.asarray(final_mesh.faces),
        config.h_floor_mm,
    )
    rolled_back = bool(reports and reports[-1].get("decision") == "ROLLBACK")
    budget_exhausted = bool(
        reports and reports[-1].get("reason") == "BUDGET_EXHAUSTED"
    )
    if budget_exhausted:
        terminal_status = "INFEASIBLE_MESH_BUDGET"
    elif rolled_back or unresolved_short_edges:
        terminal_status = "STALLED_SAFE"
    else:
        terminal_status = "PASS_WITH_WARNINGS"
    (output_dir / "adaptive_report.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "input_mesh_hash": canonical_mesh_hash(
                    np.asarray(mesh.vertices), np.asarray(mesh.faces)
                ),
                "output_mesh_hash": canonical_mesh_hash(
                    np.asarray(final_mesh.vertices), np.asarray(final_mesh.faces)
                ),
                "refinement_run_status": terminal_status,
                "unresolved_short_edge_count": unresolved_short_edges,
                "sweeps": reports,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

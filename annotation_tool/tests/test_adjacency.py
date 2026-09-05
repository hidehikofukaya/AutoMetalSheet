import json
import pathlib

import pytest
import pyvista as pv

from annotation_tool.adjacency import bboxes_intersect, classify_adjacency, nearest_distance_mm
from annotation_tool.assembly import AssemblyStore


def _body_entry(part_id: str, body_idx: int, bbox_min, bbox_max) -> dict:
    return {
        "body_idx": body_idx,
        "catia_name": part_id,
        "tag": "sheet_metal",
        "stp_file": f"bodies/{part_id}.stp",
        "volume_mm3": 1000.0,
        "total_face_area_mm2": 600.0,
        "thickness_est_mm": 1.5,
        "face_count": 12,
        "centroid_xyz": [0.0, 0.0, 0.0],
        "bbox_min_xyz": list(bbox_min),
        "bbox_max_xyz": list(bbox_max),
    }


def _write_cube(assembly_dir: pathlib.Path, part_id: str, center) -> tuple:
    cube = pv.Cube(center=center, x_length=10, y_length=10, z_length=10)
    (assembly_dir / "bodies" / f"{part_id}.vtp").parent.mkdir(parents=True, exist_ok=True)
    cube.save(str(assembly_dir / "bodies" / f"{part_id}.vtp"))
    bounds = cube.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    bbox_min = (bounds[0], bounds[2], bounds[4])
    bbox_max = (bounds[1], bounds[3], bounds[5])
    return bbox_min, bbox_max


@pytest.fixture
def assembly_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / "bodies").mkdir()
    bodies = []

    # focus: 10mm cube at origin -> bbox [-5,5]^3
    bbox_min, bbox_max = _write_cube(tmp_path, "bodyF", (0, 0, 0))
    bodies.append(_body_entry("bodyF", 1, bbox_min, bbox_max))

    # adjacent: 10mm cube 10mm away along x -> surface gap = 10mm (< 15mm threshold)
    bbox_min, bbox_max = _write_cube(tmp_path, "bodyA", (20, 0, 0))
    bodies.append(_body_entry("bodyA", 2, bbox_min, bbox_max))

    # far: 10mm cube 20mm away along x -> bbox doesn't intersect even after 15mm margin
    bbox_min, bbox_max = _write_cube(tmp_path, "bodyB", (50, 0, 0))
    bodies.append(_body_entry("bodyB", 3, bbox_min, bbox_max))

    # diagonal: passes the bbox prefilter (axis-aligned overlap after margin) but the true
    # corner-to-corner mesh distance exceeds the 15mm threshold -> must be classified by
    # the real nearest-distance step, not just the prefilter.
    bbox_min, bbox_max = _write_cube(tmp_path, "bodyD", (20, 20, 20))
    bodies.append(_body_entry("bodyD", 4, bbox_min, bbox_max))

    # missing_vtp: passes bbox prefilter but has no .vtp cache on disk
    bbox_min, bbox_max = _write_cube(tmp_path, "bodyM", (10, 0, 0))
    (tmp_path / "bodies" / "bodyM.vtp").unlink()
    bodies.append(_body_entry("bodyM", 5, bbox_min, bbox_max))

    hierarchy = {
        "source_catpart": "X.CATPart",
        "full_assembly_stp": "full_assembly.stp",
        "classification": {},
        "summary": {},
        "bodies": bodies,
    }
    (tmp_path / "hierarchy.json").write_text(json.dumps(hierarchy), encoding="utf-8")
    return tmp_path


def test_bboxes_intersect():
    import numpy as np

    assert bboxes_intersect(np.array([0, 0, 0]), np.array([10, 10, 10]), np.array([5, 5, 5]), np.array([15, 15, 15]))
    assert not bboxes_intersect(np.array([0, 0, 0]), np.array([10, 10, 10]), np.array([20, 20, 20]), np.array([30, 30, 30]))


def test_nearest_distance_mm():
    import numpy as np

    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[3.0, 4.0, 0.0]])
    assert nearest_distance_mm(a, b) == pytest.approx(5.0)


def test_classify_adjacency_marks_close_part_as_adjacent(assembly_dir: pathlib.Path) -> None:
    store = AssemblyStore.load(assembly_dir)
    focus = store.hierarchy.body_by_part_id("bodyF")

    results = classify_adjacency(store, focus, gap_threshold_mm=15.0)
    by_id = {r.part_id: r for r in results}

    assert by_id["bodyA"].tier == "adjacent"
    assert by_id["bodyA"].gap_distance_mm == pytest.approx(10.0, abs=0.01)


def test_classify_adjacency_marks_far_part_as_distant_via_bbox_prefilter(assembly_dir: pathlib.Path) -> None:
    store = AssemblyStore.load(assembly_dir)
    focus = store.hierarchy.body_by_part_id("bodyF")

    results = classify_adjacency(store, focus, gap_threshold_mm=15.0)
    by_id = {r.part_id: r for r in results}

    assert by_id["bodyB"].tier == "distant"
    assert by_id["bodyB"].gap_distance_mm is None  # bbox-prefiltered out, never measured


def test_classify_adjacency_marks_diagonal_part_as_distant_via_real_distance(assembly_dir: pathlib.Path) -> None:
    store = AssemblyStore.load(assembly_dir)
    focus = store.hierarchy.body_by_part_id("bodyF")

    results = classify_adjacency(store, focus, gap_threshold_mm=15.0)
    by_id = {r.part_id: r for r in results}

    assert by_id["bodyD"].tier == "distant"
    assert by_id["bodyD"].gap_distance_mm is not None  # passed prefilter, measured, then excluded
    assert by_id["bodyD"].gap_distance_mm > 15.0


def test_classify_adjacency_treats_missing_vtp_as_distant(assembly_dir: pathlib.Path) -> None:
    store = AssemblyStore.load(assembly_dir)
    focus = store.hierarchy.body_by_part_id("bodyF")

    results = classify_adjacency(store, focus, gap_threshold_mm=15.0)
    by_id = {r.part_id: r for r in results}

    assert by_id["bodyM"].tier == "distant"


def test_classify_adjacency_excludes_focus_itself(assembly_dir: pathlib.Path) -> None:
    store = AssemblyStore.load(assembly_dir)
    focus = store.hierarchy.body_by_part_id("bodyF")

    results = classify_adjacency(store, focus, gap_threshold_mm=15.0)

    assert "bodyF" not in {r.part_id for r in results}

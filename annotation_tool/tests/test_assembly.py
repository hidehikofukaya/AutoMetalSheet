import json
import pathlib

import pytest
import pyvista as pv

from annotation_tool.assembly import AssemblyStore

HIERARCHY = {
    "source_catpart": "X.CATPart",
    "full_assembly_stp": "full_assembly.stp",
    "classification": {},
    "summary": {"empty": 1, "sheet_metal": 2, "small_hardware": 1},
    "bodies": [
        {
            "body_idx": 1,
            "catia_name": "n1",
            "tag": "empty",
            "stp_file": "bodies/body001.stp",
            "volume_mm3": 0.0,
            "total_face_area_mm2": 0.0,
            "thickness_est_mm": 0.0,
            "face_count": 0,
            "centroid_xyz": [0, 0, 0],
            "bbox_min_xyz": [0, 0, 0],
            "bbox_max_xyz": [0, 0, 0],
        },
        {
            "body_idx": 5,
            "catia_name": "n5",
            "tag": "sheet_metal",
            "stp_file": "bodies/body005.stp",
            "volume_mm3": 100.0,
            "total_face_area_mm2": 100.0,
            "thickness_est_mm": 1.5,
            "face_count": 10,
            "centroid_xyz": [0, 0, 0],
            "bbox_min_xyz": [0, 0, 0],
            "bbox_max_xyz": [0, 0, 0],
        },
        {
            "body_idx": 2,
            "catia_name": "n2",
            "tag": "sheet_metal",
            "stp_file": "bodies/body002.stp",
            "volume_mm3": 100.0,
            "total_face_area_mm2": 100.0,
            "thickness_est_mm": 1.7,
            "face_count": 10,
            "centroid_xyz": [0, 0, 0],
            "bbox_min_xyz": [0, 0, 0],
            "bbox_max_xyz": [0, 0, 0],
        },
        {
            "body_idx": 17,
            "catia_name": "n17",
            "tag": "small_hardware",
            "stp_file": "bodies/body017.stp",
            "volume_mm3": 4000.0,
            "total_face_area_mm2": 1000.0,
            "thickness_est_mm": 0.0,
            "face_count": 40,
            "centroid_xyz": [0, 0, 0],
            "bbox_min_xyz": [0, 0, 0],
            "bbox_max_xyz": [0, 0, 0],
        },
    ],
}


@pytest.fixture
def assembly_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / "hierarchy.json").write_text(json.dumps(HIERARCHY), encoding="utf-8")
    bodies_dir = tmp_path / "bodies"
    bodies_dir.mkdir()
    pv.Sphere().save(str(bodies_dir / "body005.vtp"))
    return tmp_path


def test_load_reads_hierarchy(assembly_dir: pathlib.Path) -> None:
    store = AssemblyStore.load(assembly_dir)
    assert store.assembly_dir == assembly_dir
    assert len(store.hierarchy.bodies) == 4


def test_visible_bodies_excludes_empty_tag(assembly_dir: pathlib.Path) -> None:
    store = AssemblyStore.load(assembly_dir)
    assert {b.part_id for b in store.visible_bodies} == {"body005", "body002", "body017"}


def test_tags_returns_sorted_unique_visible_tags(assembly_dir: pathlib.Path) -> None:
    store = AssemblyStore.load(assembly_dir)
    assert store.tags() == ["sheet_metal", "small_hardware"]


def test_bodies_by_tag_groups_and_sorts_by_body_idx(assembly_dir: pathlib.Path) -> None:
    store = AssemblyStore.load(assembly_dir)
    groups = store.bodies_by_tag()
    assert [b.part_id for b in groups["sheet_metal"]] == ["body002", "body005"]
    assert [b.part_id for b in groups["small_hardware"]] == ["body017"]
    assert "empty" not in groups


def test_vtp_and_stp_path_resolve_relative_to_assembly_dir(assembly_dir: pathlib.Path) -> None:
    store = AssemblyStore.load(assembly_dir)
    body = store.hierarchy.body_by_part_id("body005")
    assert store.vtp_path(body) == assembly_dir / "bodies" / "body005.vtp"
    assert store.stp_path(body) == assembly_dir / "bodies" / "body005.stp"


def test_load_mesh_reads_vtp_cache(assembly_dir: pathlib.Path) -> None:
    store = AssemblyStore.load(assembly_dir)
    body = store.hierarchy.body_by_part_id("body005")
    mesh = store.load_mesh(body)
    assert mesh.n_points > 0


def test_load_mesh_raises_when_vtp_missing(assembly_dir: pathlib.Path) -> None:
    store = AssemblyStore.load(assembly_dir)
    body = store.hierarchy.body_by_part_id("body002")  # no .vtp written for this one
    with pytest.raises(FileNotFoundError):
        store.load_mesh(body)

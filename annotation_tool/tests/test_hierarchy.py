import json
import pathlib

import pytest

from annotation_tool.hierarchy import Hierarchy, HierarchyBody

SAMPLE = {
    "source_catpart": "A0072601285_AllCATPart.CATPart",
    "full_assembly_stp": "full_assembly.stp",
    "classification": {"sheet_metal_t_max_mm": 4.0, "small_hardware_vol_mm3": 1000.0},
    "summary": {"empty": 1, "sheet_metal": 2, "small_hardware": 1, "thick_part": 0},
    "bodies": [
        {
            "body_idx": 2,
            "catia_name": "A0072600344.1\\A0072600345.1\\0\\MANIFOLD_SOLID_BREP #101990",
            "tag": "sheet_metal",
            "stp_file": "bodies/body002.stp",
            "volume_mm3": 582743.42,
            "total_face_area_mm2": 692763.9,
            "thickness_est_mm": 1.682,
            "face_count": 1290,
            "centroid_xyz": [2613.43, 552.36, 326.97],
            "bbox_min_xyz": [1988.68, 323.4, 128.21],
            "bbox_max_xyz": [3238.18, 781.32, 525.72],
        },
        {
            "body_idx": 5,
            "catia_name": "some.other.path",
            "tag": "sheet_metal",
            "stp_file": "bodies/body005.stp",
            "volume_mm3": 100000.0,
            "total_face_area_mm2": 50000.0,
            "thickness_est_mm": 1.41,
            "face_count": 800,
            "centroid_xyz": [2600.0, 540.0, 320.0],
            "bbox_min_xyz": [2400.0, 400.0, 200.0],
            "bbox_max_xyz": [2800.0, 700.0, 450.0],
            "shapes_count": 1,
            "step_solid_idx": 0,
            "largest_face_area_mm2": 12000.0,
        },
        {
            "body_idx": 17,
            "catia_name": "A007S030431",
            "tag": "small_hardware",
            "stp_file": "bodies/body017.stp",
            "volume_mm3": 4200.0,
            "total_face_area_mm2": 1800.0,
            "thickness_est_mm": 0.0,
            "face_count": 40,
            "centroid_xyz": [2610.0, 550.0, 330.0],
            "bbox_min_xyz": [2605.0, 545.0, 325.0],
            "bbox_max_xyz": [2615.0, 555.0, 335.0],
        },
    ],
}


@pytest.fixture
def hierarchy() -> Hierarchy:
    return Hierarchy.from_dict(SAMPLE)


def test_from_dict_parses_top_level_fields(hierarchy: Hierarchy) -> None:
    assert hierarchy.source_catpart == "A0072601285_AllCATPart.CATPart"
    assert hierarchy.full_assembly_stp == "full_assembly.stp"
    assert hierarchy.summary["sheet_metal"] == 2
    assert len(hierarchy.bodies) == 3


def test_body_part_id_derived_from_stp_file(hierarchy: Hierarchy) -> None:
    body = hierarchy.bodies[0]
    assert body.part_id == "body002"
    assert body.vtp_file == "bodies/body002.vtp"


def test_body_optional_fields_default_to_none(hierarchy: Hierarchy) -> None:
    body = hierarchy.bodies[0]
    assert body.shapes_count is None
    assert body.step_solid_idx is None
    assert body.largest_face_area_mm2 is None


def test_body_optional_fields_round_trip_when_present(hierarchy: Hierarchy) -> None:
    body = hierarchy.bodies[1]
    assert body.shapes_count == 1
    assert body.step_solid_idx == 0
    assert body.largest_face_area_mm2 == 12000.0


def test_body_by_part_id(hierarchy: Hierarchy) -> None:
    body = hierarchy.body_by_part_id("body005")
    assert body.body_idx == 5

    with pytest.raises(KeyError):
        hierarchy.body_by_part_id("body999")


def test_bodies_with_tag(hierarchy: Hierarchy) -> None:
    sheet_metal = hierarchy.bodies_with_tag("sheet_metal")
    assert {b.part_id for b in sheet_metal} == {"body002", "body005"}

    hardware = hierarchy.bodies_with_tag("small_hardware")
    assert [b.part_id for b in hardware] == ["body017"]


def test_centroid_and_bbox_are_float_tuples(hierarchy: Hierarchy) -> None:
    body = hierarchy.bodies[0]
    assert body.centroid_xyz == (2613.43, 552.36, 326.97)
    assert all(isinstance(v, float) for v in body.centroid_xyz)


def test_from_dict_skips_bodies_with_no_stp_file() -> None:
    # Reproduces a real hierarchy.json record: tag="empty", shapes_count=0, and every
    # geometry field (stp_file, centroid_xyz, bbox_*, thickness_est_mm) is null.
    data = {
        **SAMPLE,
        "bodies": [
            {
                "body_idx": 1,
                "catia_name": "root",
                "tag": "empty",
                "shapes_count": 0,
                "step_solid_idx": None,
                "stp_file": None,
                "volume_mm3": 0,
                "total_face_area_mm2": 0,
                "largest_face_area_mm2": 0,
                "thickness_est_mm": None,
                "face_count": 0,
                "centroid_xyz": None,
                "bbox_min_xyz": None,
                "bbox_max_xyz": None,
            },
            *SAMPLE["bodies"],
        ],
    }

    hierarchy = Hierarchy.from_dict(data)

    assert len(hierarchy.bodies) == len(SAMPLE["bodies"])
    assert "body001" not in {b.part_id for b in hierarchy.bodies}


def test_load_reads_json_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "hierarchy.json"
    path.write_text(json.dumps(SAMPLE), encoding="utf-8")

    hierarchy = Hierarchy.load(path)

    assert hierarchy.source_catpart == SAMPLE["source_catpart"]
    assert len(hierarchy.bodies) == 3

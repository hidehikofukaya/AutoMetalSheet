import pathlib

import pytest

from annotation_tool.hierarchy import Hierarchy
from annotation_tool.schema import (
    SCHEMA_VERSION,
    AnnotationDocument,
    Axis,
    Joint,
    PartEntry,
    PartRef,
)


def make_bolt_joint(joint_id: str = "j0001") -> Joint:
    return Joint(
        joint_id=joint_id,
        type="bolt",
        parts=["body002", "body014"],
        axis=Axis(
            start_xyz=(2613.4, 552.3, 326.9),
            direction_xyz=(0.0, 0.0, 1.0),
            length_mm=4.6,
            direction_user_selected=True,
        ),
        per_part=[
            PartRef(
                part_id="body002",
                hole_center_xyz=(2613.4, 552.3, 326.9),
                hole_diameter_mm=12.0,
                detection_method="auto_hardware",
            ),
            PartRef(
                part_id="body014",
                hole_center_xyz=(2613.5, 552.4, 322.3),
                hole_diameter_mm=12.1,
                detection_method="auto_circle",
            ),
        ],
        source_hardware_part_id="body017",
        confidence="auto_suggested",
    )


def make_weld_joint(joint_id: str = "j0002") -> Joint:
    return Joint(
        joint_id=joint_id,
        type="weld",
        parts=["body002", "body005"],
        axis=Axis(
            start_xyz=(2601.1, 548.9, 327.4),
            direction_xyz=(0.0, 0.0, 1.0),
            length_mm=3.09,
            direction_user_selected=True,
        ),
        per_part=[
            PartRef(part_id="body002", contact_xyz=(2601.1, 548.9, 327.4), local_thickness_mm=1.682),
            PartRef(part_id="body005", contact_xyz=(2601.1, 548.9, 325.7), local_thickness_mm=1.41),
        ],
        confidence="user_confirmed",
    )


def make_mounting_hole_joint(joint_id: str = "j0003") -> Joint:
    return Joint(
        joint_id=joint_id,
        type="mounting_hole",
        parts=["body008"],
        axis=Axis(start_xyz=(0.0, 0.0, 0.0), direction_xyz=(0.0, 0.0, 1.0), length_mm=1.6),
        per_part=[PartRef(part_id="body008", hole_center_xyz=(0.0, 0.0, 0.0), hole_diameter_mm=8.0)],
        confidence="manual",
    )


def make_synthetic_joint(joint_id: str = "j0004") -> Joint:
    """A single-part joint on a rule-based-generated part (see synthetic_generator)."""
    return Joint(
        joint_id=joint_id,
        type="mounting_hole",
        parts=["SYN_parallel_same_offset_0001"],
        axis=Axis(start_xyz=(0.0, 0.0, 0.0), direction_xyz=(0.0, 0.0, 1.0), length_mm=1.5),
        per_part=[
            PartRef(
                part_id="SYN_parallel_same_offset_0001",
                hole_center_xyz=(0.0, 0.0, 0.0),
                hole_diameter_mm=8.0,
                detection_method="manual",
            )
        ],
        confidence="synthetic",
    )


# ---- Axis ----


def test_axis_flipped_negates_direction_and_marks_user_selected() -> None:
    axis = Axis(start_xyz=(1.0, 2.0, 3.0), direction_xyz=(0.0, 0.0, 1.0), length_mm=5.0)

    flipped = axis.flipped()

    assert flipped.direction_xyz == (0.0, 0.0, -1.0)
    assert flipped.direction_user_selected is True
    assert flipped.start_xyz == axis.start_xyz
    assert flipped.length_mm == axis.length_mm
    # original is untouched
    assert axis.direction_xyz == (0.0, 0.0, 1.0)


def test_axis_round_trip() -> None:
    axis = Axis(start_xyz=(1.0, 2.0, 3.0), direction_xyz=(0.0, 0.0, 1.0), length_mm=5.0, direction_user_selected=True)
    restored = Axis.from_dict(axis.to_dict())
    assert restored == axis


# ---- Joint validation ----


def test_joint_requires_at_least_one_part() -> None:
    with pytest.raises(ValueError):
        Joint(
            joint_id="j_bad",
            type="mounting_hole",
            parts=[],
            axis=Axis(start_xyz=(0, 0, 0), direction_xyz=(0, 0, 1), length_mm=1.0),
            per_part=[],
        )


def test_joint_requires_parts_and_per_part_same_length() -> None:
    with pytest.raises(ValueError):
        Joint(
            joint_id="j_bad",
            type="weld",
            parts=["body002", "body005"],
            axis=Axis(start_xyz=(0, 0, 0), direction_xyz=(0, 0, 1), length_mm=1.0),
            per_part=[PartRef(part_id="body002", contact_xyz=(0, 0, 0), local_thickness_mm=1.0)],
        )


def test_mounting_hole_joint_allows_single_part() -> None:
    joint = make_mounting_hole_joint()
    assert joint.parts == ["body008"]
    assert len(joint.per_part) == 1


# ---- Joint / PartRef round-trip ----


@pytest.mark.parametrize(
    "factory", [make_bolt_joint, make_weld_joint, make_mounting_hole_joint, make_synthetic_joint]
)
def test_joint_round_trip(factory) -> None:
    joint = factory()
    restored = Joint.from_dict(joint.to_dict())
    assert restored == joint


def test_synthetic_confidence_is_a_valid_literal() -> None:
    joint = make_synthetic_joint()
    assert joint.confidence == "synthetic"
    assert joint.to_dict()["confidence"] == "synthetic"


def test_weld_per_part_uses_contact_xyz_not_hole_fields() -> None:
    joint = make_weld_joint()
    data = joint.to_dict()
    for per_part in data["per_part"]:
        assert "contact_xyz" in per_part
        assert "local_thickness_mm" in per_part
        assert "hole_center_xyz" not in per_part
        assert "hole_diameter_mm" not in per_part


def test_bolt_per_part_uses_hole_fields_not_contact_xyz() -> None:
    joint = make_bolt_joint()
    data = joint.to_dict()
    for per_part in data["per_part"]:
        assert "hole_center_xyz" in per_part
        assert "hole_diameter_mm" in per_part
        assert "contact_xyz" not in per_part


# ---- AnnotationDocument ----


@pytest.fixture
def doc(tmp_path: pathlib.Path) -> AnnotationDocument:
    return AnnotationDocument(
        assembly_dir=tmp_path,
        full_assembly_stp="full_assembly.stp",
        parts={
            "body002": PartEntry("body002", "bodies/body002.stp", "bodies/body002.vtp", "sheet_metal", 1.682),
            "body005": PartEntry("body005", "bodies/body005.stp", "bodies/body005.vtp", "sheet_metal", 1.41),
        },
    )


def test_add_joint_updates_reverse_index(doc: AnnotationDocument) -> None:
    joint = make_weld_joint()
    doc.add_joint(joint)

    assert doc.joints_for_part("body002") == [joint]
    assert doc.joints_for_part("body005") == [joint]
    assert doc.joints_for_part("body999") == []


def test_remove_joint_updates_reverse_index(doc: AnnotationDocument) -> None:
    joint = make_weld_joint()
    doc.add_joint(joint)

    doc.remove_joint(joint.joint_id)

    assert doc.joints_for_part("body002") == []
    assert joint.joint_id not in doc.joints


def test_joint_referencing_multiple_parts_is_visible_from_either_focus(doc: AnnotationDocument) -> None:
    bolt = make_bolt_joint()
    doc.parts["body014"] = PartEntry("body014", "bodies/body014.stp", "bodies/body014.vtp", "sheet_metal", 2.0)
    doc.add_joint(bolt)

    assert doc.joints_for_part("body002") == [bolt]
    assert doc.joints_for_part("body014") == [bolt]


def test_next_joint_id_skips_existing(doc: AnnotationDocument) -> None:
    assert doc.next_joint_id() == "j0001"
    doc.add_joint(make_weld_joint(joint_id="j0001"))
    assert doc.next_joint_id() == "j0002"


def test_save_writes_to_annotations_subdir(doc: AnnotationDocument) -> None:
    doc.add_joint(make_weld_joint())
    doc.add_joint(make_mounting_hole_joint())

    doc.save()

    expected_path = doc.assembly_dir / "annotations" / "joints.json"
    assert expected_path.exists()


def test_save_then_load_round_trips(doc: AnnotationDocument) -> None:
    doc.add_joint(make_bolt_joint())
    doc.add_joint(make_weld_joint())
    doc.parts["body014"] = PartEntry("body014", "bodies/body014.stp", "bodies/body014.vtp", "sheet_metal", 2.0)
    doc.save()

    reloaded = AnnotationDocument.load(doc.assembly_dir)

    assert reloaded.schema_version == SCHEMA_VERSION
    assert reloaded.full_assembly_stp == "full_assembly.stp"
    assert set(reloaded.parts.keys()) == set(doc.parts.keys())
    assert set(reloaded.joints.keys()) == set(doc.joints.keys())
    assert reloaded.joints["j0001"] == doc.joints["j0001"]
    # reverse index must be rebuilt on load
    assert reloaded.joints_for_part("body002") != []


def test_new_for_assembly_skips_empty_tag_bodies() -> None:
    hierarchy = Hierarchy.from_dict(
        {
            "source_catpart": "x.CATPart",
            "full_assembly_stp": "full_assembly.stp",
            "classification": {},
            "summary": {},
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
                    "body_idx": 2,
                    "catia_name": "n2",
                    "tag": "sheet_metal",
                    "stp_file": "bodies/body002.stp",
                    "volume_mm3": 100.0,
                    "total_face_area_mm2": 100.0,
                    "thickness_est_mm": 1.5,
                    "face_count": 10,
                    "centroid_xyz": [0, 0, 0],
                    "bbox_min_xyz": [0, 0, 0],
                    "bbox_max_xyz": [0, 0, 0],
                },
            ],
        }
    )

    document = AnnotationDocument.new_for_assembly(pathlib.Path("dummy"), hierarchy)

    assert set(document.parts.keys()) == {"body002"}

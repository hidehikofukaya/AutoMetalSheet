"""``joints.json`` data model (see docs/annotation_tool_design.md SS5, schema v1.1).

All three joint types (``weld`` / ``bolt`` / ``mounting_hole``) share the same
``axis`` structure (a fastening axis line through the contact point, along the
sheet-thickness direction) -- this was an explicit user decision ("weld also
gets an axis") made after the initial design draft, which is why ``Joint`` has
a single ``axis`` field rather than per-type point/axis variants.

Output is written to ``<assembly_dir>/annotations/joints.json``, i.e.
alongside the source data in Mesh_Generater/output/<assembly_id>/, per the
confirmed storage-location decision in SS7 of the design doc.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Literal

from annotation_tool.hierarchy import Hierarchy, HierarchyBody

SCHEMA_VERSION = "1.1"

JointType = Literal["weld", "bolt", "mounting_hole"]
Confidence = Literal["auto_suggested", "user_confirmed", "manual", "synthetic"]
DetectionMethod = Literal["auto_hardware", "auto_circle", "manual"]

Vec3 = tuple[float, float, float]


def _vec3(values) -> Vec3:
    x, y, z = values
    return (float(x), float(y), float(z))


@dataclasses.dataclass
class Axis:
    start_xyz: Vec3
    direction_xyz: Vec3
    length_mm: float
    direction_user_selected: bool = False

    def flipped(self) -> "Axis":
        """Return a copy with direction_xyz negated (the user-facing "反転" action).

        Endpoints are never re-picked -- only the direction flips, per the
        confirmed UX decision in SS5/SS6 of the design doc.
        """
        dx, dy, dz = self.direction_xyz
        return dataclasses.replace(
            self, direction_xyz=(-dx, -dy, -dz), direction_user_selected=True
        )

    def to_dict(self) -> dict:
        return {
            "start_xyz": list(self.start_xyz),
            "direction_xyz": list(self.direction_xyz),
            "length_mm": self.length_mm,
            "direction_user_selected": self.direction_user_selected,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Axis":
        return cls(
            start_xyz=_vec3(data["start_xyz"]),
            direction_xyz=_vec3(data["direction_xyz"]),
            length_mm=data["length_mm"],
            direction_user_selected=data.get("direction_user_selected", False),
        )


@dataclasses.dataclass
class PartRef:
    """Per-part info for one joint. Field usage depends on the joint type:

    - bolt / mounting_hole: hole_center_xyz + hole_diameter_mm
    - weld: contact_xyz + local_thickness_mm (no diameter -- not a hole)
    """

    part_id: str
    detection_method: DetectionMethod = "manual"
    hole_center_xyz: Vec3 | None = None
    hole_diameter_mm: float | None = None
    contact_xyz: Vec3 | None = None
    local_thickness_mm: float | None = None

    def to_dict(self) -> dict:
        data: dict = {"part_id": self.part_id, "detection_method": self.detection_method}
        if self.hole_center_xyz is not None:
            data["hole_center_xyz"] = list(self.hole_center_xyz)
        if self.hole_diameter_mm is not None:
            data["hole_diameter_mm"] = self.hole_diameter_mm
        if self.contact_xyz is not None:
            data["contact_xyz"] = list(self.contact_xyz)
        if self.local_thickness_mm is not None:
            data["local_thickness_mm"] = self.local_thickness_mm
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "PartRef":
        return cls(
            part_id=data["part_id"],
            detection_method=data.get("detection_method", "manual"),
            hole_center_xyz=_vec3(data["hole_center_xyz"]) if "hole_center_xyz" in data else None,
            hole_diameter_mm=data.get("hole_diameter_mm"),
            contact_xyz=_vec3(data["contact_xyz"]) if "contact_xyz" in data else None,
            local_thickness_mm=data.get("local_thickness_mm"),
        )


@dataclasses.dataclass
class Joint:
    joint_id: str
    type: JointType
    parts: list[str]
    axis: Axis
    per_part: list[PartRef]
    confidence: Confidence = "manual"
    source_hardware_part_id: str | None = None
    created_by: str | None = None
    created_at: str | None = None

    def __post_init__(self) -> None:
        if not self.parts:
            raise ValueError("joints[].parts must contain at least 1 part (N>=1)")
        if len(self.parts) != len(self.per_part):
            raise ValueError(
                f"joint {self.joint_id!r}: parts ({len(self.parts)}) and "
                f"per_part ({len(self.per_part)}) must have the same length"
            )

    def to_dict(self) -> dict:
        data: dict = {
            "joint_id": self.joint_id,
            "type": self.type,
            "parts": list(self.parts),
            "axis": self.axis.to_dict(),
            "per_part": [p.to_dict() for p in self.per_part],
            "confidence": self.confidence,
        }
        if self.source_hardware_part_id is not None:
            data["source_hardware_part_id"] = self.source_hardware_part_id
        if self.created_by is not None:
            data["created_by"] = self.created_by
        if self.created_at is not None:
            data["created_at"] = self.created_at
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Joint":
        return cls(
            joint_id=data["joint_id"],
            type=data["type"],
            parts=list(data["parts"]),
            axis=Axis.from_dict(data["axis"]),
            per_part=[PartRef.from_dict(p) for p in data["per_part"]],
            confidence=data.get("confidence", "manual"),
            source_hardware_part_id=data.get("source_hardware_part_id"),
            created_by=data.get("created_by"),
            created_at=data.get("created_at"),
        )


@dataclasses.dataclass
class PartEntry:
    part_id: str
    stp_file: str
    vtp_file: str
    tag: str
    thickness_mm: float
    thickness_source: str = "hierarchy_est"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PartEntry":
        return cls(**data)

    @classmethod
    def from_hierarchy_body(cls, body: HierarchyBody) -> "PartEntry":
        return cls(
            part_id=body.part_id,
            stp_file=body.stp_file,
            vtp_file=body.vtp_file,
            tag=body.tag,
            thickness_mm=body.thickness_est_mm,
            thickness_source="hierarchy_est",
        )


class AnnotationDocument:
    """In-memory ``joints.json`` document with a part_id -> joint_id[] reverse index.

    The reverse index (SS6.6 of the design doc) is what lets the UI reliably
    surface every joint referencing the currently-focused part, even when
    that joint was created while a *different* part was focused.
    """

    def __init__(
        self,
        assembly_dir: pathlib.Path,
        full_assembly_stp: str,
        parts: dict[str, PartEntry] | None = None,
        joints: dict[str, Joint] | None = None,
        schema_version: str = SCHEMA_VERSION,
    ) -> None:
        self.assembly_dir = pathlib.Path(assembly_dir)
        self.full_assembly_stp = full_assembly_stp
        self.schema_version = schema_version
        self.parts: dict[str, PartEntry] = dict(parts or {})
        self.joints: dict[str, Joint] = {}
        self._joints_by_part: dict[str, set[str]] = {}
        for joint in (joints or {}).values():
            self.add_joint(joint)

    @classmethod
    def new_for_assembly(cls, assembly_dir: pathlib.Path, hierarchy: Hierarchy) -> "AnnotationDocument":
        doc = cls(assembly_dir=assembly_dir, full_assembly_stp=hierarchy.full_assembly_stp)
        for body in hierarchy.bodies:
            if body.tag == "empty":
                continue
            doc.parts[body.part_id] = PartEntry.from_hierarchy_body(body)
        return doc

    def add_joint(self, joint: Joint) -> None:
        self.joints[joint.joint_id] = joint
        for part_id in joint.parts:
            self._joints_by_part.setdefault(part_id, set()).add(joint.joint_id)

    def remove_joint(self, joint_id: str) -> None:
        joint = self.joints.pop(joint_id)
        for part_id in joint.parts:
            self._joints_by_part.get(part_id, set()).discard(joint_id)

    def joints_for_part(self, part_id: str) -> list[Joint]:
        return [self.joints[jid] for jid in self._joints_by_part.get(part_id, ())]

    def next_joint_id(self) -> str:
        n = len(self.joints) + 1
        candidate = f"j{n:04d}"
        while candidate in self.joints:
            n += 1
            candidate = f"j{n:04d}"
        return candidate

    @property
    def annotations_dir(self) -> pathlib.Path:
        return self.assembly_dir / "annotations"

    @property
    def joints_json_path(self) -> pathlib.Path:
        return self.annotations_dir / "joints.json"

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "source": {
                "assembly_dir": str(self.assembly_dir.as_posix()),
                "full_assembly_stp": self.full_assembly_stp,
            },
            "parts": [p.to_dict() for p in self.parts.values()],
            "joints": [j.to_dict() for j in self.joints.values()],
        }

    @classmethod
    def from_dict(cls, data: dict, assembly_dir: pathlib.Path) -> "AnnotationDocument":
        parts = {p["part_id"]: PartEntry.from_dict(p) for p in data["parts"]}
        joints = {j["joint_id"]: Joint.from_dict(j) for j in data["joints"]}
        return cls(
            assembly_dir=assembly_dir,
            full_assembly_stp=data["source"]["full_assembly_stp"],
            parts=parts,
            joints=joints,
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )

    def save(self) -> None:
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        with open(self.joints_json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, assembly_dir: pathlib.Path) -> "AnnotationDocument":
        assembly_dir = pathlib.Path(assembly_dir)
        path = assembly_dir / "annotations" / "joints.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, assembly_dir)

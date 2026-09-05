"""Loader for ``hierarchy.json`` (see docs/annotation_tool_design.md SS3).

``hierarchy.json`` is produced upstream (Mesh_Generater) and is read-only
input to this tool. It carries pre-computed per-body classification,
geometry stats, and bbox -- this module exposes that data as typed objects
instead of raw dicts.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

Vec3 = tuple[float, float, float]


def _vec3(values: list[float]) -> Vec3:
    x, y, z = values
    return (float(x), float(y), float(z))


@dataclasses.dataclass(frozen=True)
class HierarchyBody:
    body_idx: int
    catia_name: str
    tag: str  # "sheet_metal" | "small_hardware" | "thick_part" | "empty"
    stp_file: str
    volume_mm3: float
    total_face_area_mm2: float
    thickness_est_mm: float
    face_count: int
    centroid_xyz: Vec3
    bbox_min_xyz: Vec3
    bbox_max_xyz: Vec3
    shapes_count: int | None = None
    step_solid_idx: int | None = None
    largest_face_area_mm2: float | None = None

    @property
    def part_id(self) -> str:
        return pathlib.PurePosixPath(self.stp_file).stem

    @property
    def vtp_file(self) -> str:
        return str(pathlib.PurePosixPath(self.stp_file).with_suffix(".vtp"))

    @classmethod
    def from_dict(cls, data: dict) -> "HierarchyBody":
        return cls(
            body_idx=data["body_idx"],
            catia_name=data["catia_name"],
            tag=data["tag"],
            stp_file=data["stp_file"],
            volume_mm3=data["volume_mm3"],
            total_face_area_mm2=data["total_face_area_mm2"],
            thickness_est_mm=data["thickness_est_mm"],
            face_count=data["face_count"],
            centroid_xyz=_vec3(data["centroid_xyz"]),
            bbox_min_xyz=_vec3(data["bbox_min_xyz"]),
            bbox_max_xyz=_vec3(data["bbox_max_xyz"]),
            shapes_count=data.get("shapes_count"),
            step_solid_idx=data.get("step_solid_idx"),
            largest_face_area_mm2=data.get("largest_face_area_mm2"),
        )


@dataclasses.dataclass(frozen=True)
class Hierarchy:
    source_catpart: str
    full_assembly_stp: str
    classification: dict
    summary: dict
    bodies: tuple[HierarchyBody, ...]

    def body_by_part_id(self, part_id: str) -> HierarchyBody:
        for body in self.bodies:
            if body.part_id == part_id:
                return body
        raise KeyError(f"no body with part_id={part_id!r} in hierarchy")

    def bodies_with_tag(self, tag: str) -> tuple[HierarchyBody, ...]:
        return tuple(b for b in self.bodies if b.tag == tag)

    @classmethod
    def from_dict(cls, data: dict) -> "Hierarchy":
        return cls(
            source_catpart=data["source_catpart"],
            full_assembly_stp=data["full_assembly_stp"],
            classification=data["classification"],
            summary=data["summary"],
            # bodies with no stp_file (e.g. tag="empty", shapes_count=0) carry no geometry
            # at all -- centroid/bbox/thickness are null too, so they can't become a usable
            # HierarchyBody and are dropped here rather than threading None-checks downstream.
            bodies=tuple(
                HierarchyBody.from_dict(b) for b in data["bodies"] if b.get("stp_file") is not None
            ),
        )

    @classmethod
    def load(cls, hierarchy_json_path: pathlib.Path) -> "Hierarchy":
        with open(hierarchy_json_path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

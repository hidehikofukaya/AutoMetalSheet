"""Assembly-level data access: combines ``hierarchy.json`` with the on-disk
``bodies/*.stp`` / ``*.vtp`` files it references (see design doc SS3).

Kept Qt-free so the grouping / path-resolution logic stays unit-testable
without a display server; the Qt widgets in ``ui/`` are thin wrappers around
this.
"""

from __future__ import annotations

import dataclasses
import pathlib

import pyvista as pv

from annotation_tool.hierarchy import Hierarchy, HierarchyBody

# Tags that carry no real geometry and are never shown in the part tree.
HIDDEN_TAGS = {"empty"}


@dataclasses.dataclass
class AssemblyStore:
    assembly_dir: pathlib.Path
    hierarchy: Hierarchy
    _mesh_cache: dict = dataclasses.field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, assembly_dir: pathlib.Path) -> "AssemblyStore":
        assembly_dir = pathlib.Path(assembly_dir)
        hierarchy = Hierarchy.load(assembly_dir / "hierarchy.json")
        return cls(assembly_dir=assembly_dir, hierarchy=hierarchy)

    @property
    def visible_bodies(self) -> tuple[HierarchyBody, ...]:
        return tuple(b for b in self.hierarchy.bodies if b.tag not in HIDDEN_TAGS)

    def tags(self) -> list[str]:
        return sorted({b.tag for b in self.visible_bodies})

    def bodies_by_tag(self) -> dict[str, list[HierarchyBody]]:
        groups: dict[str, list[HierarchyBody]] = {}
        for body in sorted(self.visible_bodies, key=lambda b: b.body_idx):
            groups.setdefault(body.tag, []).append(body)
        return groups

    def vtp_path(self, body: HierarchyBody) -> pathlib.Path:
        return self.assembly_dir / body.vtp_file

    def stp_path(self, body: HierarchyBody) -> pathlib.Path:
        return self.assembly_dir / body.stp_file

    def load_mesh(self, body: HierarchyBody) -> pv.PolyData:
        """Read the pre-tessellated .vtp mesh cache (already in assembly world coords, SS3).

        Results are cached in memory so repeated renders of the same body (e.g.
        when toggling checkboxes or the adjacent-only filter) avoid redundant disk reads.
        """
        if body.part_id not in self._mesh_cache:
            path = self.vtp_path(body)
            if not path.exists():
                raise FileNotFoundError(f"no .vtp cache for {body.part_id}: {path}")
            self._mesh_cache[body.part_id] = pv.read(str(path))
        return self._mesh_cache[body.part_id]

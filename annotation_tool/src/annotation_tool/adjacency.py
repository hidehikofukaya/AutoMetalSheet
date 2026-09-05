"""Geometric adjacency classification via face-gap distance (design doc SS6.2).

Two-stage, for speed:
1. Expand the focus part's bbox by the gap threshold and keep only other
   parts whose (unexpanded) bbox intersects it -- a cheap, conservative
   prefilter (it can only over-include candidates, never miss a true
   adjacency, since any point within ``gap_threshold_mm`` of the focus mesh
   must lie inside the focus bbox expanded by that same margin).
2. For the surviving candidates, compute the true nearest point-to-point
   distance between the focus mesh and the candidate mesh (.vtp point
   clouds, via KD-tree) and compare against the threshold.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree

from annotation_tool.assembly import AssemblyStore
from annotation_tool.hierarchy import HierarchyBody

DEFAULT_GAP_THRESHOLD_MM = 15.0

Tier = Literal["focus", "adjacent", "distant"]

TIER_OPACITY: dict[Tier, float] = {"focus": 1.0, "adjacent": 0.5, "distant": 0.0}


@dataclasses.dataclass(frozen=True)
class AdjacencyResult:
    part_id: str
    tier: Tier
    gap_distance_mm: float | None  # None when bbox-prefiltered out (distance never computed)


def _bbox_arrays(body: HierarchyBody) -> tuple[np.ndarray, np.ndarray]:
    return np.array(body.bbox_min_xyz), np.array(body.bbox_max_xyz)


def bboxes_intersect(lo_a: np.ndarray, hi_a: np.ndarray, lo_b: np.ndarray, hi_b: np.ndarray) -> bool:
    return bool(np.all(lo_a <= hi_b) and np.all(lo_b <= hi_a))


def bbox_candidates(
    focus: HierarchyBody, others: list[HierarchyBody], margin_mm: float
) -> list[HierarchyBody]:
    flo, fhi = _bbox_arrays(focus)
    flo, fhi = flo - margin_mm, fhi + margin_mm
    candidates = []
    for other in others:
        olo, ohi = _bbox_arrays(other)
        if bboxes_intersect(flo, fhi, olo, ohi):
            candidates.append(other)
    return candidates


def nearest_distance_mm(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """Nearest point-to-point distance between two point clouds (KD-tree approximation of mesh gap)."""
    tree = cKDTree(points_b)
    dists, _ = tree.query(points_a, k=1)
    return float(dists.min())


def classify_adjacency(
    store: AssemblyStore,
    focus: HierarchyBody,
    gap_threshold_mm: float = DEFAULT_GAP_THRESHOLD_MM,
) -> list[AdjacencyResult]:
    """Classify every other visible part's adjacency tier relative to *focus*."""
    others = [b for b in store.visible_bodies if b.part_id != focus.part_id]
    candidates = {c.part_id for c in bbox_candidates(focus, others, gap_threshold_mm)}

    results: list[AdjacencyResult] = []
    focus_points = None  # lazily loaded only if there's at least one candidate

    for body in others:
        if body.part_id not in candidates:
            results.append(AdjacencyResult(body.part_id, "distant", None))
            continue

        try:
            if focus_points is None:
                focus_points = store.load_mesh(focus).points
            candidate_points = store.load_mesh(body).points
        except FileNotFoundError:
            results.append(AdjacencyResult(body.part_id, "distant", None))
            continue

        gap = nearest_distance_mm(focus_points, candidate_points)
        tier: Tier = "adjacent" if gap <= gap_threshold_mm else "distant"
        results.append(AdjacencyResult(body.part_id, tier, gap))

    return results

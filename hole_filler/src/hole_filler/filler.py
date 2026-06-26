"""Core hole-filling operations on PyVista meshes.

Mesh states
-----------
original mesh  (``self._mesh``)
    The VTP from Mesh_Generater.  Bolt/mounting holes are represented as
    cylindrical voids: hole walls (cylindrical faces) + fill caps (flat discs
    at each end).  Visually the holes are still "visible" as cylinders.

patched mesh   (``self._filled``)
    Created by ``create_patched_output()``.  For each selected hole the entire
    cylindrical region (walls + caps) is removed and both openings are
    fan-triangulated with flat patches → truly hole-free sheet metal surface.
    This is what gets saved as the AI-training target.
"""
from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Low-level cell-mask helpers
# ---------------------------------------------------------------------------

def _cap_face_mask(
    mesh: pv.PolyData,
    hole: dict,
    thickness_mm: float = 3.0,
) -> np.ndarray:
    """Boolean face mask of fill-cap triangles belonging to *hole*."""
    center = np.asarray(hole["center"], dtype=float)
    normal = np.asarray(hole["normal"], dtype=float)
    radius = float(hole["radius_mm"])

    try:
        centroids = mesh.cell_centers().points
    except Exception:
        return np.zeros(mesh.n_cells, dtype=bool)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            n_pd = mesh.compute_normals(
                cell_normals=True, point_normals=False, inplace=False
            )
        face_nrm = np.asarray(n_pd.cell_data["Normals"])
    except Exception:
        face_nrm = np.zeros((mesh.n_cells, 3))

    diff = centroids - center
    along = diff @ normal
    in_plane = np.linalg.norm(diff - np.outer(along, normal), axis=1)
    cos_angle = np.abs(face_nrm @ normal)

    return (
        (np.abs(along) <= thickness_mm)
        & (in_plane <= radius * 1.1)
        & (cos_angle >= 0.5)
    )


def _extract_cells_as_polydata(mesh: pv.PolyData, keep_mask: np.ndarray) -> pv.PolyData:
    """Keep cells where *keep_mask* is True, returning a PolyData."""
    keep_ids = np.where(keep_mask)[0]
    if len(keep_ids) == 0:
        return pv.PolyData()
    if keep_mask.all():
        return mesh.copy()

    result = mesh.extract_cells(keep_ids)

    if not isinstance(result, pv.PolyData):
        try:
            result = result.extract_surface(algorithm="dataset_surface")
        except Exception:
            pass
    if not isinstance(result, pv.PolyData):
        result = pv.PolyData(result.points, np.asarray(result.cells))

    return result


# ---------------------------------------------------------------------------
# Boundary loop extraction
# ---------------------------------------------------------------------------

def _extract_ordered_boundary_loops(mesh: pv.PolyData) -> list[np.ndarray]:
    """Return ordered boundary vertex arrays from all open-edge loops in *mesh*.

    Uses edge-based traversal so that T-junction vertices (degree 4+, caused by
    two loops sharing a single mesh vertex) are handled correctly: each edge is
    consumed exactly once, yielding distinct simple loops.

    Each returned array lists the loop vertices in consecutive order, suitable
    for fan triangulation.  Only loops with ≥ 3 vertices are returned.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        be = mesh.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
        )

    if be.n_cells == 0 or be.n_points == 0:
        return []

    pts = be.points
    n_pts = len(pts)

    # Build vertex adjacency and full edge set from LINE cells
    adj: dict[int, list[int]] = {i: [] for i in range(n_pts)}
    lines = be.lines
    ptr = 0
    while ptr < len(lines):
        n = int(lines[ptr])
        if n == 2:
            a, b = int(lines[ptr + 1]), int(lines[ptr + 2])
            if b not in adj[a]:
                adj[a].append(b)
            if a not in adj[b]:
                adj[b].append(a)
        ptr += n + 1

    # Edge-based traversal: mark edges as consumed so T-junctions split cleanly
    # into separate loops rather than being traversed as one tangled path.
    used_edges: set[tuple[int, int]] = set()
    loops: list[np.ndarray] = []

    for start in range(n_pts):
        while True:
            # Find any unused edge from start
            next_nb = None
            for nb in adj[start]:
                key = (min(start, nb), max(start, nb))
                if key not in used_edges:
                    next_nb = nb
                    break
            if next_nb is None:
                break  # No unused edges remain from this start vertex

            # Walk a new loop beginning start -> next_nb -> ...
            ordered: list[int] = [start]
            key = (min(start, next_nb), max(start, next_nb))
            used_edges.add(key)
            ordered.append(next_nb)
            current = next_nb

            for _ in range(n_pts):
                # Prefer continuing along an unused edge back toward start
                # (closes the loop) or forward to the next unused neighbour.
                chosen = None
                for nb in adj[current]:
                    k = (min(current, nb), max(current, nb))
                    if k not in used_edges:
                        if nb == start and len(ordered) >= 3:
                            chosen = nb  # close the loop
                            break
                        if chosen is None:
                            chosen = nb
                if chosen is None:
                    break
                k = (min(current, chosen), max(current, chosen))
                used_edges.add(k)
                if chosen == start and len(ordered) >= 3:
                    break  # loop closed
                ordered.append(chosen)
                current = chosen

            if len(ordered) >= 3:
                loops.append(pts[ordered])

    return loops


def _loop_normal(loop_pts: np.ndarray) -> np.ndarray:
    """Compute the plane normal of a polygon loop via Newell's method."""
    n = len(loop_pts)
    normal = np.zeros(3, dtype=float)
    for i in range(n):
        c = loop_pts[i]
        nx = loop_pts[(i + 1) % n]
        normal[0] += (c[1] - nx[1]) * (c[2] + nx[2])
        normal[1] += (c[2] - nx[2]) * (c[0] + nx[0])
        normal[2] += (c[0] - nx[0]) * (c[1] + nx[1])
    mag = np.linalg.norm(normal)
    return normal / mag if mag > 1e-10 else normal


# ---------------------------------------------------------------------------
# Public mesh-state builders
# ---------------------------------------------------------------------------

def create_open_mesh(
    mesh: pv.PolyData,
    holes: list[dict],
    thickness_mm: float = 3.0,
) -> pv.PolyData:
    """Remove fill-cap faces for feature holes (kept for utility; not used in UI)."""
    if mesh.n_cells == 0:
        return mesh.copy()

    cap_mask = np.zeros(mesh.n_cells, dtype=bool)
    for h in holes:
        if h.get("_source") == "feature":
            cap_mask |= _cap_face_mask(mesh, h, thickness_mm)

    if not cap_mask.any():
        return mesh.copy()

    return _extract_cells_as_polydata(mesh, ~cap_mask)


def _ear_clip_xz(pts3d: np.ndarray, expected: np.ndarray) -> list[tuple[int, int, int]] | None:
    """Ear-clip triangulation using XZ projection.

    Returns a list of (i, j, k) index triples into pts3d, or None if the
    polygon cannot be fully triangulated (e.g. because its XZ projection is
    non-simple or degenerate).  Both CW and CCW orientations are tried; the
    one with fewer wrong-direction 3D normals is returned.
    """
    n = len(pts3d)
    pts2d = pts3d[:, [0, 2]]  # XZ slice

    def _cross2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    def _in_tri2d(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        d1 = _cross2d(a, b, p)
        d2 = _cross2d(b, c, p)
        d3 = _cross2d(c, a, p)
        has_neg = d1 < -1e-9 or d2 < -1e-9 or d3 < -1e-9
        has_pos = d1 > 1e-9 or d2 > 1e-9 or d3 > 1e-9
        return not (has_neg and has_pos)

    def _clip(ccw: bool) -> list:
        idxs = list(range(n))
        tris: list[tuple[int, int, int]] = []
        limit = n * n + 10
        for _ in range(limit):
            if len(idxs) <= 3:
                break
            ni = len(idxs)
            found = False
            for il in range(ni):
                p0 = idxs[(il - 1) % ni]
                p1 = idxs[il]
                p2 = idxs[(il + 1) % ni]
                a, b, c = pts2d[p0], pts2d[p1], pts2d[p2]
                cp = _cross2d(a, b, c)
                if (ccw and cp <= 0) or (not ccw and cp >= 0):
                    continue
                ok = all(
                    j_pos in ((il - 1) % ni, il, (il + 1) % ni)
                    or not _in_tri2d(pts2d[idxs[j_pos]], a, b, c)
                    for j_pos in range(ni)
                )
                if ok:
                    tris.append((p0, p1, p2))
                    idxs.pop(il)
                    found = True
                    break
            if not found:
                break
        if len(idxs) == 3:
            tris.append(tuple(idxs))  # type: ignore[arg-type]
        return tris

    area2 = sum(
        pts2d[i][0] * pts2d[(i + 1) % n][1] - pts2d[(i + 1) % n][0] * pts2d[i][1]
        for i in range(n)
    )
    fwd = _clip(area2 > 0)
    rev = _clip(area2 <= 0)

    def _n_bad(tris: list) -> int:
        return sum(
            1
            for i, j, k in tris
            if np.dot(np.cross(pts3d[j] - pts3d[i], pts3d[k] - pts3d[i]), expected) < 0
        )

    if len(fwd) == n - 2 and len(rev) == n - 2:
        return fwd if _n_bad(fwd) <= _n_bad(rev) else rev
    if len(fwd) == n - 2:
        return fwd
    if len(rev) == n - 2:
        return rev
    return None


def _fill_loop_ear_xz(
    mesh: pv.PolyData, lp: np.ndarray, expected: np.ndarray
) -> pv.PolyData | None:
    """Close a boundary loop via XZ-projected ear-clipping.

    Returns the updated mesh on success, or None if ear-clipping fails or the
    loop vertices cannot be matched to existing mesh points.
    """
    pts3d = np.array(lp, dtype=float)
    tree = cKDTree(mesh.points)
    dists, idxs = tree.query(pts3d)
    if (dists > 0.5).any():
        return None
    loop_idxs = idxs.astype(int)

    tris = _ear_clip_xz(pts3d, expected)
    if tris is None:
        return None

    # Explicitly orient every triangle so its normal matches `expected`.
    # Ear-clip minimises wrong-direction triangles but cannot eliminate them
    # for non-planar 3-D loops; per-triangle flipping ensures the fill is
    # correctly lit even when adjacent triangles share an edge in opposite
    # winding (the visual discontinuity is sub-pixel on a nearly-flat surface).
    new_faces: list[int] = []
    for (i, j, k) in tris:
        v0 = pts3d[j] - pts3d[i]
        v1 = pts3d[k] - pts3d[i]
        if np.dot(np.cross(v0, v1), expected) < 0:
            j, k = k, j  # flip this triangle
        new_faces += [3, int(loop_idxs[i]), int(loop_idxs[j]), int(loop_idxs[k])]

    all_faces = np.concatenate([mesh.faces, np.array(new_faces, dtype=int)])
    return pv.PolyData(mesh.points.copy(), all_faces)


_SLOT_CIRCULARITY_THRESHOLD = 0.7
"""Holes with circularity below this are treated as non-circular (slot-like)."""


def _fill_loop_pca_delaunay(
    mesh: pv.PolyData, lp: np.ndarray, expected: np.ndarray
) -> pv.PolyData | None:
    """Fill a boundary loop with PCA-projected Delaunay triangulation.

    Projects the loop to its best-fit plane (PCA/SVD), triangulates via
    Delaunay in 2-D (all simplices kept — works for convex/near-convex
    loops, i.e. circularity ≥ 0.7), then **explicitly orients every
    resulting triangle** so its normal agrees with *expected*.  This
    guarantees zero wrong-direction triangles for nearly-flat loops.

    Returns the updated mesh, or None if the approach fails.
    """
    from scipy.spatial import Delaunay as _Delaunay

    pts3d = np.array(lp, dtype=float)
    n = len(pts3d)
    if n < 3:
        return None

    # Match loop vertices to existing mesh points.
    tree = cKDTree(mesh.points)
    dists, idxs = tree.query(pts3d)
    if (dists > 0.5).any():
        return None
    loop_idxs = idxs.astype(int)

    # PCA: best-fit plane via SVD.
    centroid = pts3d.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts3d - centroid, full_matrices=False)
    plane_normal = Vt[2]  # least-variance axis ≈ patch normal

    if np.dot(plane_normal, expected) < 0:
        plane_normal = -plane_normal

    # Project to 2-D.
    pts2d = np.column_stack(
        [(pts3d - centroid) @ Vt[0], (pts3d - centroid) @ Vt[1]]
    )

    # Delaunay triangulation.
    try:
        tri = _Delaunay(pts2d)
    except Exception:
        return None

    # Orient every simplex explicitly — no wrong-direction triangles.
    new_faces: list[int] = []
    for simplex in tri.simplices:
        i, j, k = simplex
        v0 = pts3d[j] - pts3d[i]
        v1 = pts3d[k] - pts3d[i]
        if np.dot(np.cross(v0, v1), plane_normal) < 0:
            j, k = k, j  # flip winding to match plane_normal
        new_faces += [3, int(loop_idxs[i]), int(loop_idxs[j]), int(loop_idxs[k])]

    if not new_faces:
        return None

    all_faces = np.concatenate([mesh.faces, np.array(new_faces, dtype=int)])
    return pv.PolyData(mesh.points.copy(), all_faces)


def replace_hole_with_patch(
    mesh: pv.PolyData,
    hole: dict,
    margin_mm: float = 1.5,
    thickness_mm: float = 6.0,
    force_fill: bool = False,
) -> pv.PolyData:
    """Remove the cylindrical void (walls + caps) and close both openings.

    Algorithm
    ---------
    1. Build a cylindrical removal mask around the hole axis.
       Threshold = max(radius + margin_mm, radius * 1.15) so that oval holes
       (where the wall centroid can sit up to ~12 % beyond the nominal radius)
       are fully captured without over-removing faces on neighbouring panels.
    2. Try PyVista ``fill_holes`` to patch the exposed openings.
    3. For slot-shaped holes (circularity < 0.7): try XZ ear-clip triangulation.
    4. Try PCA-projected Delaunay triangulation (explicitly orients every triangle).
    5. Fall back to edge-based fan triangulation as a last resort.

    Special case — already-filled slot holes
    -----------------------------------------
    When a non-circular hole was pre-filled by Mesh_Generater (_source='feature')
    the existing curved fill matches the panel surface and is superior to any flat
    patch the fan can produce.  Automatic batch processing skips such holes to
    avoid degrading mesh quality.  Pass ``force_fill=True`` to override this and
    explicitly replace the existing fill with a flat patch (e.g. when creating
    targeted individual fills for ML training data).

    The result is a visually hole-free mesh suitable for AI training.
    """
    center = np.asarray(hole["center"], dtype=float)
    normal = np.asarray(hole["normal"], dtype=float)
    radius = float(hole["radius_mm"])
    circularity = float(hole.get("circularity", 1.0))

    # Pre-filled slot holes: keep the Mesh_Generater surface as-is unless
    # the caller explicitly requests a re-fill.
    if (
        not force_fill
        and hole.get("_source") == "feature"
        and circularity < _SLOT_CIRCULARITY_THRESHOLD
    ):
        return mesh

    if mesh.n_cells == 0:
        return mesh

    def _be_count(m: pv.PolyData) -> int:
        be = m.extract_feature_edges(
            boundary_edges=True, feature_edges=False,
            manifold_edges=False, non_manifold_edges=False,
        )
        return int(be.n_cells)

    try:
        centroids = mesh.cell_centers().points
    except Exception:
        return mesh

    diff = centroids - center
    along = diff @ normal
    in_plane = np.linalg.norm(diff - np.outer(along, normal), axis=1)

    # Adaptive threshold: oval holes can have wall centroids up to ~r*1.12;
    # using r*1.15 as the upper bound keeps small holes unchanged (r*1.15 < r+1.5
    # when r < 10 mm) while correctly capturing the outer wall ring of large holes.
    threshold = max(radius + margin_mm, radius * 1.15)
    remove_mask = (np.abs(along) <= thickness_mm) & (in_plane <= threshold)

    if not remove_mask.any():
        return mesh
    if remove_mask.sum() >= mesh.n_cells:
        return mesh

    open_mesh = _extract_cells_as_polydata(mesh, ~remove_mask)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filled = open_mesh.fill_holes(radius * 3.0)
    if _be_count(filled) == 0:
        return filled

    # Edge-based fan (or ear-clip for slot holes) fallback
    all_loops = _extract_ordered_boundary_loops(open_mesh)
    result = open_mesh
    for lp in all_loops:
        if len(lp) < 3:
            continue
        lp_center = lp.mean(axis=0)
        lp_diff = lp_center - center
        lp_in_plane = np.linalg.norm(lp_diff - np.dot(lp_diff, normal) * normal)
        if lp_in_plane > radius * 2.5:
            continue
        along_c = np.dot(lp_diff, normal)
        # Determine which panel face this loop is on.
        # If along_c ≈ 0 (helical loop spanning both faces), default to outward.
        # Otherwise use the sign: along_c > 0 means front face → fill toward +normal;
        #                          along_c < 0 means back face  → fill toward -normal.
        if abs(along_c) < thickness_mm * 0.3:
            expected = normal  # helical/centered loop — fill outward
        else:
            expected = normal if along_c >= 0 else -normal
        if circularity < _SLOT_CIRCULARITY_THRESHOLD:
            ear_result = _fill_loop_ear_xz(result, lp, expected)
            if ear_result is not None:
                result = ear_result
                continue
        result = fill_hole_fan(result, {"loop_points": lp, "center": lp_center},
                               expected=expected)
    return result


def create_patched_output(
    mesh: pv.PolyData,
    holes: list[dict],
    selected_indices: set[int],
    margin_mm: float = 1.5,
    thickness_mm: float = 6.0,
) -> pv.PolyData:
    """Build the hole-free AI-training output mesh.

    For each **selected** hole: the entire cylindrical void (walls + caps) is
    removed and replaced with a flat patch.
    For **deselected** holes: geometry is left unchanged (hole remains visible).
    """
    result = mesh.copy()
    for i in selected_indices:
        if 0 <= i < len(holes):
            result = replace_hole_with_patch(result, holes[i], margin_mm, thickness_mm)
    return result


def create_filled_output(
    mesh_orig: pv.PolyData,
    holes: list[dict],
    selected_indices: set[int],
    thickness_mm: float = 3.0,
) -> pv.PolyData:
    """Legacy helper: remove fill caps for deselected feature holes only.

    Prefer ``create_patched_output`` for new code.
    """
    desel_feature = [
        i for i in range(len(holes))
        if i not in selected_indices and holes[i].get("_source") == "feature"
    ]
    result = mesh_orig.copy()
    if desel_feature:
        cap_mask = np.zeros(result.n_cells, dtype=bool)
        for i in desel_feature:
            cap_mask |= _cap_face_mask(result, holes[i], thickness_mm)
        if cap_mask.any():
            result = _extract_cells_as_polydata(result, ~cap_mask)
    for i in selected_indices:
        if holes[i].get("_source") == "boundary":
            result = fill_hole_fan(result, holes[i])
    return result


# ---------------------------------------------------------------------------
# Fan triangulation
# ---------------------------------------------------------------------------

def fill_hole_fan(
    mesh: pv.PolyData,
    hole: dict,
    expected: np.ndarray | None = None,
) -> pv.PolyData:
    """Close one open boundary by fan-triangulating from the loop center.

    When *expected* (a unit normal) is supplied every fan triangle is
    explicitly oriented so its computed normal agrees with *expected*.
    This eliminates inverted-normal triangles on 3-D helical loops.
    """
    loop_pts = np.asarray(hole["loop_points"], dtype=float)
    center = np.asarray(hole["center"], dtype=float)
    n_loop = len(loop_pts)
    if n_loop < 3:
        return mesh

    tree = cKDTree(mesh.points)
    dists, idxs = tree.query(loop_pts)

    if (dists > 0.5).any():
        base = mesh.n_points
        ctr_idx = base + n_loop
        new_pts = np.vstack([mesh.points, loop_pts, center.reshape(1, 3)])
        loop_idxs = np.arange(base, base + n_loop, dtype=int)
    else:
        ctr_idx = mesh.n_points
        new_pts = np.vstack([mesh.points, center.reshape(1, 3)])
        loop_idxs = idxs.astype(int)

    new_faces: list[int] = []
    if expected is not None:
        for i in range(n_loop):
            j = (i + 1) % n_loop
            pi = new_pts[loop_idxs[i]]
            pj = new_pts[loop_idxs[j]]
            pc = new_pts[ctr_idx]
            face_n = np.cross(pj - pi, pc - pi)
            if np.dot(face_n, expected) < 0:
                new_faces += [3, int(loop_idxs[i]), ctr_idx, int(loop_idxs[j])]
            else:
                new_faces += [3, int(loop_idxs[i]), int(loop_idxs[j]), ctr_idx]
    else:
        for i in range(n_loop):
            j = (i + 1) % n_loop
            new_faces += [3, int(loop_idxs[i]), int(loop_idxs[j]), ctr_idx]

    all_faces = np.concatenate([mesh.faces, np.array(new_faces, dtype=int)])
    return pv.PolyData(new_pts, all_faces)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def build_holes_record(
    part_id: str,
    assembly_id: str,
    holes: list[dict],
    selected_indices: set[int],
) -> dict:
    """Return the holes.json dict for the AI training dataset."""
    records = []
    for i, h in enumerate(holes):
        status = "filled" if i in selected_indices else "open"
        records.append({
            "hole_id": f"h{i + 1:03d}",
            "status": status,
            "was_open": h.get("_source") == "boundary",
            "shape": h["shape"],
            "radius_mm": round(float(h["radius_mm"]), 4),
            "diameter_mm": round(float(h["radius_mm"]) * 2.0, 4),
            "circularity": round(float(h["circularity"]), 4),
            "center_xyz": [round(float(v), 4) for v in h["center"]],
            "normal_xyz": [round(float(v), 6) for v in h["normal"]],
        })
    return {
        "schema_version": _SCHEMA_VERSION,
        "assembly_id": assembly_id,
        "part_id": part_id,
        "source_vtp": f"bodies/{part_id}.vtp",
        "filled_vtp": f"filled/{part_id}.vtp",
        "holes": records,
    }


def save_result(
    assembly_dir: pathlib.Path,
    part_id: str,
    filled_mesh: pv.PolyData,
    holes: list[dict],
    selected_indices: set[int],
) -> tuple[pathlib.Path, pathlib.Path]:
    """Save the patched mesh and hole metadata to filled/."""
    out_dir = assembly_dir / "filled"
    out_dir.mkdir(parents=True, exist_ok=True)

    vtp_path = out_dir / f"{part_id}.vtp"
    json_path = out_dir / f"{part_id}_holes.json"

    filled_mesh.save(str(vtp_path))

    data = build_holes_record(
        part_id, assembly_dir.name, holes, selected_indices
    )
    json_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return vtp_path, json_path

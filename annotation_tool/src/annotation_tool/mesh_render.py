"""Mesh-rendering helpers kept Qt-free so they stay unit-testable.

Feature-edge extraction policy: hide an edge only when the two triangles on
either side of it are truly coplanar (dihedral angle ~0 deg, i.e. an
arbitrary diagonal cut through one flat face). Any non-zero curvature is a
genuine direction change and stays visible -- this means cylinders and other
curved surfaces show their full tessellation as edges (expected, since no
two neighboring facets on a curved surface are ever exactly coplanar), and a
flat region's perimeter always shows because boundary edges have no
opposite-face angle to compare against. The threshold is kept just above
zero (not exactly zero) to absorb mesh-export floating-point noise on
faces that are flat in the source CAD geometry.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull, cKDTree

DEFAULT_FEATURE_ANGLE_DEG = 1.0
DEFAULT_CONTACT_THRESHOLD_MM = 2.0


def feature_edges(mesh: pv.PolyData, feature_angle_deg: float = DEFAULT_FEATURE_ANGLE_DEG) -> pv.PolyData:
    """Boundary / non-manifold / sharp-dihedral-angle edges only -- no flat-face diagonals."""
    return mesh.extract_feature_edges(
        feature_angle=feature_angle_deg,
        boundary_edges=True,
        non_manifold_edges=True,
        manifold_edges=False,
        feature_edges=True,
    )


def contact_zone_labels(
    focus_mesh: pv.PolyData,
    adjacent_meshes: list[tuple[pv.PolyData, int]],
    threshold_mm: float = DEFAULT_CONTACT_THRESHOLD_MM,
) -> np.ndarray:
    """Per-cell integer label array for focus_mesh.

    Returns an (n_cells,) int32 array: 0 = no contact (focus base color),
    1..N = the label of the adjacent part whose surface is within threshold_mm.

    A cell is labeled when ANY of its vertices is within threshold_mm of the
    adjacent mesh -- not just the centroid.  This ensures that large flat faces
    whose centroid is far from the contact edge are still fully captured: any
    cell touching the boundary of the contact zone has at least one vertex
    close to the adjacent part.
    """
    n_cells = focus_mesh.n_cells
    labels = np.zeros(n_cells, dtype=np.int32)

    # Fast path: only usable when the mesh is purely triangulated AND the face
    # connectivity count matches n_cells exactly.  VTP files from STEP
    # tessellators sometimes embed extra edge/vertex cells that inflate n_cells
    # beyond the triangle count, which breaks the reshape trick.
    face_vtx = None
    if focus_mesh.is_all_triangles:
        raw = focus_mesh.faces
        if len(raw) == n_cells * 4:            # each triangle = [3, v0, v1, v2]
            face_vtx = raw.reshape(-1, 4)[:, 1:]   # (n_cells, 3) vertex indices

    _TMP = "__cz"

    for adj_mesh, label in adjacent_meshes:
        tree = cKDTree(adj_mesh.points)
        point_dists, _ = tree.query(focus_mesh.points)  # (n_points,)
        close_verts = point_dists < threshold_mm          # (n_points,) boolean

        if face_vtx is not None:
            # Vectorised: cell is in contact if ANY of its 3 vertices is close
            labels[close_verts[face_vtx].any(axis=1)] = label
        else:
            # Fallback for meshes with extra edge/vertex cells beyond the faces.
            # Temporarily borrow the cached mesh's point_data slot and restore
            # it in the finally block so the cache stays clean.
            focus_mesh.point_data[_TMP] = close_verts.astype(np.float32)
            try:
                cell_mesh = focus_mesh.point_data_to_cell_data(pass_point_data=False)
            finally:
                if _TMP in focus_mesh.point_data:
                    del focus_mesh.point_data[_TMP]
            labels[cell_mesh.cell_data[_TMP] > 0] = label

    return labels


def _deduplicate_fill_cap_pairs(holes: list[dict]) -> list[dict]:
    """Merge fill-cap loop pairs from opposite ends of the same closed hole.

    When ``detect_holes`` falls back to feature edges on a closed (filled) mesh,
    each cylindrical hole produces TWO loops — one per fill cap.  Pairs share the
    same radius (±5%), have centres separated by at most 3× that radius (covers
    BIW plate thickness), and have (anti-)parallel normals.

    The merged record uses the cap-to-cap midpoint as centre and the cap-to-cap
    axis as the hole normal.
    """
    used: set[int] = set()
    out: list[dict] = []

    for i, h1 in enumerate(holes):
        if i in used:
            continue
        r1 = h1["radius_mm"]
        c1 = np.asarray(h1["center"], dtype=float)
        # Use PCA normal (hole axis) for matching — mesh surface normals are
        # unreliable at cap-wall junctions because find_closest_cell can return
        # either a wall cell (radial normal) or a cap cell (axial normal).
        n1 = np.asarray(h1.get("_pca_normal", h1["normal"]), dtype=float)

        best_j: int | None = None
        best_dist = float("inf")

        for j, h2 in enumerate(holes):
            if j <= i or j in used:
                continue
            r2 = h2["radius_mm"]
            if abs(r1 - r2) / max(r1, r2) > 0.05:
                continue
            c2 = np.asarray(h2["center"], dtype=float)
            dist = float(np.linalg.norm(c1 - c2))
            if dist >= r1 * 3.0:
                continue
            n2 = np.asarray(h2.get("_pca_normal", h2["normal"]), dtype=float)
            if abs(float(np.dot(n1, n2))) < 0.85:
                continue
            if dist < best_dist:
                best_dist = dist
                best_j = j

        if best_j is not None:
            h2 = holes[best_j]
            c2 = np.asarray(h2["center"], dtype=float)
            n2 = np.asarray(h2["normal"], dtype=float)
            axis = c2 - c1
            axis_len = float(np.linalg.norm(axis))
            if axis_len > 1e-6:
                axis = axis / axis_len
                if float(np.dot(axis, n1)) < 0:
                    axis = -axis
            else:
                axis = n1
            out.append({
                "center": (c1 + c2) / 2.0,
                "normal": axis,
                "_pca_normal": axis,
                "_source": "feature",
                "radius_mm": (r1 + h2["radius_mm"]) / 2.0,
                "circularity": (h1["circularity"] + h2["circularity"]) / 2.0,
                "shape": h1["shape"],
                "loop_points": h1["loop_points"],
            })
            used.add(best_j)
        else:
            out.append(h1)
        used.add(i)

    return out


def _order_boundary_loop(labeled: pv.PolyData, region_mask: np.ndarray) -> np.ndarray:
    """Return boundary-loop points in edge-traversal order.

    Falls back to the unordered point array if traversal fails (e.g. when
    ``labeled.lines`` is empty or the loop has non-manifold topology).
    The ordered result is used by hole-filling pipelines.
    """
    loop_pts = labeled.points[region_mask]
    n = len(loop_pts)
    if n < 2:
        return loop_pts

    global_indices = np.where(region_mask)[0]
    in_region = set(int(g) for g in global_indices)
    g2l = {int(g): l for l, g in enumerate(global_indices)}

    # Build adjacency from edge connectivity stored in labeled.lines.
    # Format: [2, a, b,  2, c, d, ...]  (each edge is a 2-point line cell).
    adj: dict[int, list[int]] = {l: [] for l in range(n)}
    raw = labeled.lines
    i = 0
    while i < len(raw):
        cell_n = int(raw[i])
        if cell_n == 2 and i + 2 < len(raw):
            a, b = int(raw[i + 1]), int(raw[i + 2])
            if a in in_region and b in in_region:
                la, lb = g2l[a], g2l[b]
                if lb not in adj[la]:
                    adj[la].append(lb)
                if la not in adj[lb]:
                    adj[lb].append(la)
        i += cell_n + 1

    # Greedy traversal from an arbitrary start vertex
    start = next(iter(adj), 0)
    path = [start]
    visited = {start}
    curr = start
    while True:
        nbs = [v for v in adj.get(curr, []) if v not in visited]
        if not nbs:
            break
        nxt = nbs[0]
        path.append(nxt)
        visited.add(nxt)
        curr = nxt

    if len(path) >= max(3, n // 2):
        return loop_pts[np.array(path, dtype=int)]
    return loop_pts  # fallback: unordered


def detect_holes(
    mesh: pv.PolyData,
    min_radius_mm: float = 1.5,
    max_radius_mm: float = 100.0,
    planarity_tol: float = 0.30,
    min_points: int = 4,
    feature_angle_deg: float = 30.0,
) -> list[dict]:
    """Detect holes of any shape by analyzing open boundary-edge loops.

    Unlike the previous implementation, all hole shapes — circular, slot /
    stadium, polygon, irregular — are returned.  Holes are **not** filtered by
    circularity; only by equivalent radius (sqrt of enclosed area divided by π).

    **Open vs closed meshes** — the algorithm detects holes in both:

    * *Open mesh* (boundary edges present): boundary-edge loops identify the
      hole rims directly.
    * *Closed mesh* (no boundary edges, e.g. Mesh_Generater output where holes
      are filled during tessellation): the algorithm falls back to feature edges
      at ``feature_angle_deg``.  Each filled hole creates two fill-cap loops
      (one per side); ``_deduplicate_fill_cap_pairs`` merges them into one.

    Parameters
    ----------
    min_radius_mm
        Minimum equivalent circular radius ``sqrt(A / π)`` in mm.
    max_radius_mm
        Maximum equivalent circular radius in mm.
    planarity_tol
        Maximum allowed ratio of out-of-plane RMS to in-plane RMS for the
        boundary loop.  0.30 accommodates holes on moderately curved BIW
        panels where the previous 0.05 limit was too strict.
    min_points
        Minimum boundary-edge vertex count to consider a loop (default 4).
    feature_angle_deg
        Dihedral-angle threshold for the feature-edge fallback.  30° is
        empirically reliable for BIW sheet-metal fill caps (≈90° transition
        from flat cap to hole wall).

    Returns
    -------
    list of dicts with keys:
        center       -- (3,) ndarray, hole centroid in world coordinates
        normal       -- (3,) ndarray, unit surface normal at hole centre
        radius_mm    -- float, equivalent radius ``sqrt(area / π)``
        circularity  -- float in [0, 1], isoperimetric quotient ``4πA / P²``
                        (1 = perfect circle; kept as metadata, not used as filter)
        shape        -- str, ``'circle' | 'slot' | 'polygon' | 'irregular'``
        loop_points  -- (N, 3) ndarray, boundary vertices in traversal order
    """
    # Always use feature+boundary edges to capture both open holes (boundary
    # edges) and closed/filled holes (feature edges at 90° cap-wall junctions).
    # This handles mixed meshes that have some open and some filled holes.
    boundary = mesh.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=True,
        feature_edges=True,
        feature_angle=feature_angle_deg,
        manifold_edges=False,
    )

    if boundary.n_points < min_points:
        return []

    # Classify each loop: "boundary" (open hole rim) vs "feature" (fill-cap edge).
    # Extract only the raw open boundary edges; any loop whose points all sit on
    # this edge set is an open hole, otherwise it is a closed fill-cap rim.
    _open_be = mesh.extract_feature_edges(
        boundary_edges=True, non_manifold_edges=False,
        feature_edges=False, manifold_edges=False)
    _bnd_tree: cKDTree | None = None
    if _open_be.n_points >= 1:
        _bnd_tree = cKDTree(_open_be.points)

    # Label each connected boundary-edge loop with a region id
    labeled = boundary.connectivity(largest=False)
    region_ids = labeled.point_data.get("RegionId")
    if region_ids is None or len(region_ids) == 0:
        return []

    # Pre-compute cell normals for surface-normal estimation
    try:
        n_mesh = mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)
    except Exception:
        n_mesh = None

    holes: list[dict] = []
    n_regions = int(region_ids.max()) + 1

    for rid in range(n_regions):
        mask = region_ids == rid
        loop_pts = labeled.points[mask]
        n_pts = int(mask.sum())
        if n_pts < min_points:
            continue

        # ---- PCA plane fit ------------------------------------------------
        centroid = loop_pts.mean(axis=0)
        centered = loop_pts - centroid
        cov = (centered.T @ centered) / n_pts
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # eigh: ascending order → [0]=min (out-of-plane), [2]=max (in-plane)

        if eigenvalues[2] < 1e-12:
            continue  # degenerate

        out_rms = float(np.sqrt(max(eigenvalues[0], 0.0)))
        in_rms = float(np.sqrt((eigenvalues[1] + eigenvalues[2]) / 2.0))
        if in_rms < 1e-8:
            continue
        # Reject only strongly non-planar loops (e.g. seams on sharp cylinders)
        if out_rms / in_rms > planarity_tol:
            continue

        # ---- Project to 2-D PCA plane -------------------------------------
        u = eigenvectors[:, 1]
        v = eigenvectors[:, 2]
        pts_2d = centered @ np.column_stack([u, v])  # (N, 2)

        # ---- Convex hull → area, perimeter --------------------------------
        try:
            hull = ConvexHull(pts_2d)
            area = float(hull.volume)     # scipy 2-D: volume = enclosed area
            perimeter = float(hull.area)  # scipy 2-D: area = perimeter
        except Exception:
            continue  # degenerate (collinear points)

        if area < 1e-8:
            continue

        equiv_radius = float(np.sqrt(area / np.pi))
        if not (min_radius_mm <= equiv_radius <= max_radius_mm):
            continue

        # ---- Outer-perimeter rejection -------------------------------------
        # The part's own outer contour appears as a large feature-edge loop
        # whose projected area covers nearly the entire mesh.  Reject it by
        # comparing the loop's convex-hull area to the mesh's total projected
        # footprint in the same PCA direction.  Coverage > 70 % → outer edge.
        # Threshold 15 mm keeps bolt holes (typ. r=3-12 mm) while catching
        # outer perimeters even on small parts (body014: r=22 mm, cov=97 %).
        if equiv_radius > 15.0:
            try:
                mesh_proj = (mesh.points - centroid) @ np.column_stack([u, v])
                mesh_ph = ConvexHull(mesh_proj)
                if area / mesh_ph.volume > 0.70:
                    continue
            except Exception:
                pass

        # ---- Shape metrics ------------------------------------------------
        # Isoperimetric quotient: 1.0 for a perfect circle
        circularity = float(min(1.0, 4.0 * np.pi * area / (perimeter ** 2)))

        if circularity > 0.85:
            shape = "circle"
        elif circularity > 0.65:
            shape = "slot"
        elif circularity > 0.50:
            shape = "polygon"
        else:
            shape = "irregular"

        # ---- Hole centre --------------------------------------------------
        # Algebraic circle fit for circular holes (better than centroid);
        # for other shapes the PCA centroid (origin of the centered coords) is used.
        cx_2d, cy_2d = 0.0, 0.0
        if shape == "circle" and n_pts >= 6:
            A_mat = np.column_stack([2 * pts_2d[:, 0], 2 * pts_2d[:, 1], np.ones(n_pts)])
            b_vec = pts_2d[:, 0] ** 2 + pts_2d[:, 1] ** 2
            try:
                res, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                r_sq = res[2] + res[0] ** 2 + res[1] ** 2
                if r_sq > 0:
                    cx_2d, cy_2d = float(res[0]), float(res[1])
            except np.linalg.LinAlgError:
                pass

        center_3d = centroid + cx_2d * u + cy_2d * v

        # ---- Surface normal -----------------------------------------------
        # Average face normals around the rim — more accurate than the PCA
        # normal for holes on curved surfaces.
        pca_normal = eigenvectors[:, 0].copy()
        face_normal = pca_normal
        if n_mesh is not None:
            try:
                stride = max(1, n_pts // 12)  # sample ~12 rim points
                near_normals = []
                for pt in loop_pts[::stride]:
                    cid = n_mesh.find_closest_cell(pt.astype(float))
                    if cid >= 0:
                        near_normals.append(np.array(n_mesh.cell_data["Normals"][cid]))
                if near_normals:
                    avg = np.mean(near_normals, axis=0)
                    if np.dot(avg, pca_normal) < 0:
                        avg = -avg
                    nlen = np.linalg.norm(avg)
                    if nlen > 1e-8:
                        face_normal = avg / nlen
            except Exception:
                pass

        nlen = np.linalg.norm(face_normal)
        if nlen > 1e-8:
            face_normal = face_normal / nlen

        # ---- Ordered boundary loop for hole-filling -----------------------
        loop_ordered = _order_boundary_loop(labeled, mask)

        # Classify loop source for downstream hole-filling
        if _bnd_tree is not None:
            q_d, _ = _bnd_tree.query(loop_pts)
            _source = "boundary" if float(q_d.max()) < 0.5 else "feature"
        else:
            _source = "feature"

        holes.append({
            "center": center_3d,
            "normal": face_normal,
            "_pca_normal": pca_normal,
            "_source": _source,
            "radius_mm": equiv_radius,
            "circularity": circularity,
            "shape": shape,
            "loop_points": loop_ordered,
        })

    holes = _deduplicate_fill_cap_pairs(holes)

    return holes

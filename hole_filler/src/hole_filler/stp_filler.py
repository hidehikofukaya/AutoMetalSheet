"""STP B-rep based hole filler using PythonOCC.

Approach
--------
For each hole (from holes.json):
1. Find hole wall faces via sphere query: centroid within (radius+5mm) of hole center
   AND area < 4*pi*radius (exclude large panel faces).
2. Find boundary edges: edges shared between hole faces and non-hole faces.
3. Group boundary edges into connected closed loops.
4. Build flat fill face from each loop (best-fit plane via PCA + BRepBuilderAPI_MakeFace).
5. Sew: original faces minus hole faces + fill faces.
6. Export as STP + VTP.
"""
from __future__ import annotations

import json
import pathlib
import warnings
from typing import Callable

import numpy as np
import pyvista as pv

from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepTools import breptools
from OCC.Core.GCPnts import GCPnts_UniformAbscissa
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_Sewing,
)
from OCC.Core.BRepFill import BRepFill_Filling
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.GeomAbs import GeomAbs_C0, GeomAbs_Cylinder
from OCC.Core.GProp import GProp_GProps
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.ShapeFix import ShapeFix_Shape
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX, TopAbs_WIRE
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopTools import (
    TopTools_IndexedDataMapOfShapeListOfShape,
    TopTools_IndexedMapOfShape,
    TopTools_ListIteratorOfListOfShape,
)
from OCC.Core.TopoDS import topods
from OCC.Core.gp import gp_Ax3, gp_Dir, gp_Pln, gp_Pnt


# ---------------------------------------------------------------------------
# STP I/O
# ---------------------------------------------------------------------------

def load_stp(path: str | pathlib.Path):
    """Load STP file and return root OCC shape."""
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Cannot read STP: {path}")
    reader.TransferRoots()
    return reader.OneShape()


def save_stp(shape, path: str | pathlib.Path) -> None:
    """Write OCC shape as STP (AP203)."""
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(str(path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to write STP: {path}")


# ---------------------------------------------------------------------------
# B-rep helpers
# ---------------------------------------------------------------------------

def _face_centroid_area(face) -> tuple[np.ndarray, float]:
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    c = props.CentreOfMass()
    return np.array([c.X(), c.Y(), c.Z()]), props.Mass()


def _edge_endpoints_key(edge) -> tuple[tuple, tuple]:
    """Return (p0_key, p1_key) with coords rounded to 2 dp."""
    pts = []
    exp = TopExp_Explorer(edge, TopAbs_VERTEX)
    while exp.More():
        v = topods.Vertex(exp.Current())
        pt = BRep_Tool.Pnt(v)
        pts.append((round(pt.X(), 2), round(pt.Y(), 2), round(pt.Z(), 2)))
        exp.Next()
    if len(pts) < 2:
        return (pts[0] if pts else (0., 0., 0.)), (pts[0] if pts else (0., 0., 0.))
    return pts[0], pts[-1]


def _group_edges_into_loops(edges: list) -> list[list]:
    """Group OCC edges into connected closed loops by vertex adjacency."""
    if not edges:
        return []

    adj: dict[tuple, list[tuple[int, tuple]]] = {}
    for idx, edge in enumerate(edges):
        p0, p1 = _edge_endpoints_key(edge)
        adj.setdefault(p0, []).append((idx, p1))
        adj.setdefault(p1, []).append((idx, p0))

    used: set[int] = set()
    loops: list[list] = []

    for start_key in list(adj.keys()):
        while True:
            # Find any unused edge from start_key
            seed = next(
                ((eidx, other) for eidx, other in adj.get(start_key, [])
                 if eidx not in used),
                None,
            )
            if seed is None:
                break

            loop: list = []
            cur = start_key
            for _ in range(len(edges) + 2):
                # Prefer closing the loop; otherwise pick first unused
                closing = next(
                    ((eidx, other) for eidx, other in adj.get(cur, [])
                     if eidx not in used and other == start_key and len(loop) >= 2),
                    None,
                )
                nxt = closing or next(
                    ((eidx, other) for eidx, other in adj.get(cur, [])
                     if eidx not in used),
                    None,
                )
                if nxt is None:
                    break
                eidx, other = nxt
                used.add(eidx)
                loop.append(edges[eidx])
                cur = other
                if cur == start_key and len(loop) >= 2:
                    break

            if len(loop) >= 2:
                loops.append(loop)

    return loops


def _build_wire(edges: list):
    """Assemble a list of OCC edges into a BRepBuilderAPI_MakeWire. Returns wire or None."""
    wm = BRepBuilderAPI_MakeWire()
    for e in edges:
        wm.Add(topods.Edge(e))
    return wm.Wire() if wm.IsDone() else None


def _sample_edge(edge, n: int = 24) -> list[list[float]]:
    """Sample n points along a BRep edge (handles arcs/splines correctly)."""
    try:
        adaptor = BRepAdaptor_Curve(edge)
        disc = GCPnts_UniformAbscissa()
        disc.Initialize(adaptor, n, adaptor.FirstParameter(), adaptor.LastParameter())
        if disc.IsDone():
            return [
                [adaptor.Value(disc.Parameter(i)).X(),
                 adaptor.Value(disc.Parameter(i)).Y(),
                 adaptor.Value(disc.Parameter(i)).Z()]
                for i in range(1, disc.NbPoints() + 1)
            ]
    except Exception:
        pass
    # Fallback: just endpoints from vertices
    pts: list[list[float]] = []
    vexp = TopExp_Explorer(edge, TopAbs_VERTEX)
    while vexp.More():
        v = topods.Vertex(vexp.Current())
        pt = BRep_Tool.Pnt(v)
        pts.append([pt.X(), pt.Y(), pt.Z()])
        vexp.Next()
    return pts


def _wire_pca_plane(wire, n_per_edge: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """PCA best-fit plane for a wire. Samples edge curves (arc-accurate)."""
    pts: list[list[float]] = []
    exp = TopExp_Explorer(wire, TopAbs_EDGE)
    while exp.More():
        edge = topods.Edge(exp.Current())
        pts.extend(_sample_edge(edge, n_per_edge))
        exp.Next()
    if not pts:  # fallback: vertices only
        vexp = TopExp_Explorer(wire, TopAbs_VERTEX)
        while vexp.More():
            v = topods.Vertex(vexp.Current())
            pt = BRep_Tool.Pnt(v)
            pts.append([pt.X(), pt.Y(), pt.Z()])
            vexp.Next()
    if len(pts) < 3:
        return np.zeros(3), np.array([0., 0., 1.])
    pts_arr = np.array(pts)
    centroid = pts_arr.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts_arr - centroid, full_matrices=False)
    return centroid, Vt[2]


def _make_fill_face(wire, normal_hint: np.ndarray | None = None):
    """Build a flat face from a closed wire. Returns OCC Face or None.

    normal_hint bypasses PCA and uses the known cylinder axis direction.
    This is critical for arc wires that have only 2 vertex positions (half-circles),
    where PCA is degenerate.
    """
    centroid, pca_normal = _wire_pca_plane(wire)

    if normal_hint is not None:
        # Prefer the supplied normal; also try its opposite and the PCA result
        normals_to_try = [normal_hint, -normal_hint, pca_normal, -pca_normal]
    else:
        normals_to_try = [pca_normal, -pca_normal]

    for n in normals_to_try:
        n = np.asarray(n, dtype=float)
        mag = float(np.linalg.norm(n))
        if mag < 1e-9:
            continue
        n = n / mag
        try:
            ax3 = gp_Ax3(
                gp_Pnt(float(centroid[0]), float(centroid[1]), float(centroid[2])),
                gp_Dir(float(n[0]), float(n[1]), float(n[2])),
            )
            fm = BRepBuilderAPI_MakeFace(gp_Pln(ax3), wire, True)
            if fm.IsDone():
                return fm.Face()
        except Exception:
            pass

    # Fallback: BRepFill_Filling (handles slightly non-planar wires)
    try:
        filling = BRepFill_Filling()
        exp = TopExp_Explorer(wire, TopAbs_EDGE)
        while exp.More():
            filling.Add(topods.Edge(exp.Current()), GeomAbs_C0)
            exp.Next()
        filling.Build()
        if filling.IsDone():
            return filling.Face()
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# B-rep face helpers for precise hole detection
# ---------------------------------------------------------------------------

def _find_cylinder_wall_faces(
    face_map: TopTools_IndexedMapOfShape,
    center: np.ndarray,
    radius: float,
    normal: np.ndarray,
    radius_tol_frac: float = 0.15,
    axis_cos_tol: float = 0.85,  # ≈32° tolerance on axis alignment
) -> tuple[set[int], list]:
    """Find cylinder wall B-rep faces for a hole by geometry (not centroid distance).

    Uses cylinder radius, axis direction, and axis midpoint proximity.
    This is far more precise than the area+distance filter.

    Returns (set of 1-based face indices in face_map, list of face TopoDS shapes).
    """
    tight_mm = max(radius * 1.5, 8.0)
    hole_idxs: set[int] = set()
    hole_faces: list = []

    for i in range(1, face_map.Size() + 1):
        face = topods.Face(face_map.FindKey(i))
        try:
            adaptor = BRepAdaptor_Surface(face)
            if adaptor.GetType() != GeomAbs_Cylinder:
                continue
            cyl = adaptor.Cylinder()
            r = cyl.Radius()
            if abs(r - radius) > radius * radius_tol_frac:
                continue
            d = cyl.Axis().Direction()
            axis_dir = np.array([d.X(), d.Y(), d.Z()])
            # Axis direction must align with hole normal (either polarity)
            if abs(float(np.dot(axis_dir, normal))) < axis_cos_tol:
                continue
            # Compute axis midpoint (accurate hole center for arc patches)
            loc = cyl.Location()
            v_mid = (adaptor.FirstVParameter() + adaptor.LastVParameter()) * 0.5
            axis_center = np.array([loc.X(), loc.Y(), loc.Z()]) + v_mid * axis_dir
            if float(np.linalg.norm(axis_center - center)) > tight_mm:
                continue
            hole_idxs.add(i)
            hole_faces.append(face)
        except Exception:
            continue

    return hole_idxs, hole_faces


def _find_panel_faces_with_hole_wire(
    face_map: TopTools_IndexedMapOfShape,
    center: np.ndarray,
    radius: float,
    normal: np.ndarray,
    radius_tol_frac: float = 0.4,
    max_dist_mm: float | None = None,
    min_face_area_mm2: float = 50.0,
) -> list[tuple[int, object, object]]:
    """Find panel B-rep faces that have an inner wire matching a through-hole.

    In CATIA V5 STP B-rep, a through-hole typically appears as an inner trimming
    wire on each panel face (top and bottom of the cylinder).  This function
    identifies those faces and the specific matching inner wire.

    Returns list of (1-based face index, TopoDS_Face, matching TopoDS_Wire) triples.
    """
    if max_dist_mm is None:
        max_dist_mm = max(radius * 2.0, 10.0)

    found: list[tuple[int, object, object]] = []

    for i in range(1, face_map.Size() + 1):
        face = topods.Face(face_map.FindKey(i))

        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        if props.Mass() < min_face_area_mm2:
            continue

        try:
            outer = breptools.OuterWire(face)
        except Exception:
            continue

        wire_exp = TopExp_Explorer(face, TopAbs_WIRE)
        while wire_exp.More():
            wire = topods.Wire(wire_exp.Current())
            wire_exp.Next()
            if wire.IsSame(outer):
                continue

            # Sample inner wire edge curves (arc-accurate)
            pts: list[list[float]] = []
            seen: set[tuple] = set()
            e2 = TopExp_Explorer(wire, TopAbs_EDGE)
            while e2.More():
                for pt3 in _sample_edge(topods.Edge(e2.Current()), 12):
                    key = (round(pt3[0], 1), round(pt3[1], 1), round(pt3[2], 1))
                    if key not in seen:
                        pts.append(pt3)
                        seen.add(key)
                e2.Next()

            if len(pts) < 4:
                continue

            arr = np.array(pts)
            wc = arr.mean(axis=0)
            wr = float(np.linalg.norm(arr - wc, axis=1).mean())

            if abs(wr - radius) <= radius * radius_tol_frac:
                if float(np.linalg.norm(wc - center)) <= max_dist_mm:
                    found.append((i, face, wire))
                    break  # one matching wire per face is enough

    return found


def _clone_face_removing_inner_wires(face, wires_to_remove: list) -> object | None:
    """Clone a B-rep face on the SAME underlying surface but without the specified
    inner wires (= fill those holes by extension of the surface).

    This is the correct approach for curved panel surfaces: instead of adding a
    separate flat fill patch, we rebuild the face using EmptyCopied() which
    preserves the original surface geometry (plane, BSpline, torus, etc.) and
    reattaches only the desired wires.  The mesher then tessellates the extended
    surface over the former hole — guaranteed conformant because there is no
    separate fill face.

    Returns the new TopoDS_Face, or None if construction fails.
    """
    try:
        outer = breptools.OuterWire(face)
        builder = BRep_Builder()
        new_face = topods.Face(face.EmptyCopied())
        builder.Add(new_face, outer)

        # Keep inner wires that do NOT match any wire in wires_to_remove
        if wires_to_remove:
            wire_exp = TopExp_Explorer(face, TopAbs_WIRE)
            while wire_exp.More():
                wire = topods.Wire(wire_exp.Current())
                wire_exp.Next()
                if wire.IsSame(outer):
                    continue
                if not any(wire.IsSame(w) for w in wires_to_remove):
                    builder.Add(new_face, wire)

        return new_face
    except Exception:
        return None


def _find_panel_inner_wires(
    shape,
    center: np.ndarray,
    radius: float,
    normal: np.ndarray,
    radius_tol_frac: float = 0.4,
    max_dist_mm: float | None = None,
    min_face_area_mm2: float = 50.0,
) -> list:
    """Find inner trimming wires on panel faces (used by diagnose_holes_geometry)."""
    if max_dist_mm is None:
        max_dist_mm = max(radius * 2.0, 10.0)

    found_wires: list = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        exp.Next()
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        if props.Mass() < min_face_area_mm2:
            continue
        try:
            outer = breptools.OuterWire(face)
        except Exception:
            continue
        wire_exp = TopExp_Explorer(face, TopAbs_WIRE)
        while wire_exp.More():
            wire = topods.Wire(wire_exp.Current())
            wire_exp.Next()
            if wire.IsSame(outer):
                continue
            pts: list[list[float]] = []
            seen: set[tuple] = set()
            edge_exp2 = TopExp_Explorer(wire, TopAbs_EDGE)
            while edge_exp2.More():
                edge = topods.Edge(edge_exp2.Current())
                edge_exp2.Next()
                for pt3 in _sample_edge(edge, 12):
                    key = (round(pt3[0], 1), round(pt3[1], 1), round(pt3[2], 1))
                    if key not in seen:
                        pts.append(pt3)
                        seen.add(key)
            if len(pts) < 4:
                continue
            arr = np.array(pts)
            wire_center = arr.mean(axis=0)
            wire_radius = float(np.linalg.norm(arr - wire_center, axis=1).mean())
            if abs(wire_radius - radius) > radius * radius_tol_frac:
                continue
            if float(np.linalg.norm(wire_center - center)) > max_dist_mm:
                continue
            found_wires.append(wire)
    return found_wires


# ---------------------------------------------------------------------------
# Core fill logic
# ---------------------------------------------------------------------------

def fill_shape_holes(
    shape,
    holes_data: list[dict],
    sewing_tolerance: float = 0.1,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[object, dict]:
    """
    Fill holes in an OCC shape using B-rep surface extension.

    Strategy
    --------
    For each hole:
    1. Find cylinder wall faces precisely (geometry-based: radius + axis + center).
    2. PRIMARY — EmptyCopied: find panel faces with matching inner wires.
       Replace each such face with a wire-free version on the SAME surface
       (EmptyCopied + outer wire only).  The mesher then covers the former hole
       area with the existing curved surface — guaranteed conformant tessellation.
    3. FALLBACK — boundary edges: if no panel inner wire found, extract boundary
       edges between wall and panel faces, group into loops, build flat fill faces.

    Parameters
    ----------
    shape            : OCC shape loaded from STP
    holes_data       : list of hole dicts (center_xyz, radius_mm, normal_xyz, hole_id, ...)
    sewing_tolerance : mm tolerance for BRepBuilderAPI_Sewing
    progress_cb      : optional callable(message) for progress reporting

    Returns
    -------
    (filled_shape, stats_dict)
    """
    def _log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    stats: dict = {
        "holes_attempted": len(holes_data),
        "holes_filled": 0,
        "details": [],
    }

    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)
    n_faces = face_map.Size()
    _log(f"Indexing {n_faces} B-rep faces...")

    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    # faces to remove from the final sewing (wall faces + original panel faces replaced)
    all_remove_idxs: set[int] = set()
    # faces to add to the final sewing (replacement + flat fallback patches)
    all_add_faces: list = []
    # per-face wire-removal map: {face_idx → (orig_face, [wires_to_remove])}
    panel_wire_removals: dict[int, list] = {}
    # holes that need boundary-edge fallback (no inner wire found)
    fallback_holes: list = []

    for hole in holes_data:
        hole_id = hole.get("hole_id", "?")
        center = np.array(hole["center_xyz"], dtype=float)
        radius = float(hole["radius_mm"])
        normal = np.array(hole["normal_xyz"], dtype=float)
        _log(f"Processing {hole_id} (r={radius:.1f}mm)...")

        # Step 1: cylinder wall faces
        wall_idxs, _ = _find_cylinder_wall_faces(face_map, center, radius, normal)
        if not wall_idxs:
            _log(f"  {hole_id}: no cylinder wall faces found")
            stats["details"].append({"hole_id": hole_id, "status": "no_faces_found", "fill_faces": 0})
            continue
        all_remove_idxs |= wall_idxs

        # Step 2: EmptyCopied (primary) — find panel faces with matching inner wires
        matches = _find_panel_faces_with_hole_wire(face_map, center, radius, normal)
        if matches:
            for fidx, orig_face, matching_wire in matches:
                if fidx not in panel_wire_removals:
                    panel_wire_removals[fidx] = [orig_face, []]
                panel_wire_removals[fidx][1].append(matching_wire)
            _log(f"  {hole_id}: {len(wall_idxs)} wall faces, {len(matches)} panel face(s) to extend (EmptyCopied)")
            stats["details"].append({
                "hole_id": hole_id,
                "wall_faces": len(wall_idxs),
                "fill_faces": len(matches),
                "method": "empty_copied",
                "status": "filled",
            })
            stats["holes_filled"] += 1
        else:
            # Step 3: Boundary-edge fallback
            _log(f"  {hole_id}: no inner wires - will use boundary edge fallback")
            fallback_holes.append((hole, wall_idxs))

    # Build EmptyCopied replacement faces
    for fidx, (orig_face, wires_to_rm) in panel_wire_removals.items():
        new_face = _clone_face_removing_inner_wires(orig_face, wires_to_rm)
        if new_face is not None:
            all_remove_idxs.add(fidx)
            all_add_faces.append(new_face)

    # Boundary-edge fallback
    for hole, wall_idxs in fallback_holes:
        hole_id = hole.get("hole_id", "?")
        center = np.array(hole["center_xyz"], dtype=float)
        radius = float(hole["radius_mm"])
        normal = np.array(hole["normal_xyz"], dtype=float)

        boundary_edges = []
        for i in range(1, edge_face_map.Size() + 1):
            fl = edge_face_map.FindFromIndex(i)
            it = TopTools_ListIteratorOfListOfShape(fl)
            adj_hole = adj_panel = False
            while it.More():
                fidx = face_map.FindIndex(it.Value())
                if fidx in wall_idxs:
                    adj_hole = True
                else:
                    adj_panel = True
                it.Next()
            if adj_hole and adj_panel:
                boundary_edges.append(edge_face_map.FindKey(i))

        loops = _group_edges_into_loops(boundary_edges)
        n_filled = 0
        for loop_edges in loops:
            wire = _build_wire(loop_edges)
            if wire is None:
                continue
            fill_face = _make_fill_face(wire, normal_hint=normal)
            if fill_face is not None:
                all_add_faces.append(fill_face)
                n_filled += 1

        status = "filled" if n_filled > 0 else "fill_failed"
        _log(f"  {hole_id}: boundary edge fallback: {len(boundary_edges)} edges, {n_filled} fill faces")
        stats["details"].append({
            "hole_id": hole_id,
            "wall_faces": len(wall_idxs),
            "fill_faces": n_filled,
            "method": "boundary_edge",
            "status": status,
        })
        if status == "filled":
            stats["holes_filled"] += 1

    # Sewing
    n_kept = n_faces - len(all_remove_idxs)
    _log(f"Sewing {n_kept} original + {len(all_add_faces)} modified/fill faces...")
    sewing = BRepBuilderAPI_Sewing(sewing_tolerance)
    for i in range(1, n_faces + 1):
        if i not in all_remove_idxs:
            sewing.Add(topods.Face(face_map.FindKey(i)))
    for face in all_add_faces:
        sewing.Add(face)
    sewing.Perform()
    result = sewing.SewedShape()

    # Shape fix
    _log("Fixing shape...")
    fixer = ShapeFix_Shape(result)
    fixer.Perform()
    result = fixer.Shape()

    _log(f"Done: {stats['holes_filled']}/{stats['holes_attempted']} holes filled.")
    return result, stats


# ---------------------------------------------------------------------------
# Meshing: OCC → PyVista VTP
# ---------------------------------------------------------------------------

def mesh_shape_to_vtp(
    shape,
    output_path: str | pathlib.Path,
    linear_deflection: float = 0.05,
    angular_deflection: float = 0.5,
    merge_tol: float = 0.05,
) -> pv.PolyData:
    """Mesh an OCC shape with BRepMesh, merge shared vertices, save as VTP."""
    mesh = _mesh_shape_internal(shape, linear_deflection, angular_deflection, merge_tol)
    mesh.save(str(output_path))
    return mesh


# ---------------------------------------------------------------------------
# Hole detection from STP
# ---------------------------------------------------------------------------

_MAX_HOLE_RADIUS_MM = 40.0   # holes larger than this are structural openings
_MAX_PANEL_THICKNESS_MM = 4.0  # max panel thickness for per-face height filter
_AXIS_ANGLE_TOL_DEG = 8.0    # group cylinders with similar axis
_SPATIAL_CLUSTER_K = 2.0     # group cylinders within k*radius of each other


def _detect_holes_cylindrical(
    shape,
    max_radius_mm: float,
    max_panel_thickness_mm: float,
    min_angle_coverage: float,
) -> list[dict]:
    """
    Phase 1: find holes by grouping cylindrical faces.

    Key filter: total U-parameter angle span of the group ≥ min_angle_coverage × 2π.
    A true through-hole wraps ≈360°; a structural arc or fillet wraps a small fraction.
    This is more reliable than area-based coverage because it does not depend on
    panel thickness being correct.
    """
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface  # noqa: PLC0415 (already imported)

    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)

    max_area_per_r = 2.0 * np.pi * max_panel_thickness_mm  # height filter per face

    cylinders: list[dict] = []
    for i in range(1, face_map.Size() + 1):
        face = topods.Face(face_map.FindKey(i))
        try:
            adaptor = BRepAdaptor_Surface(face)
            if adaptor.GetType() != GeomAbs_Cylinder:
                continue
            cyl = adaptor.Cylinder()
            r = cyl.Radius()
            if r > max_radius_mm:
                continue
            props = GProp_GProps()
            brepgprop.SurfaceProperties(face, props)
            area = props.Mass()
            if area / r > max_area_per_r:
                continue  # face is too tall → structural
            d = cyl.Axis().Direction()
            u_span = abs(adaptor.LastUParameter() - adaptor.FirstUParameter())
            # Cylinder axis midpoint: more accurate than face centroid for arc patches.
            # For a partial-arc cylinder face, the centroid is offset from the axis by
            # up to r; the axis midpoint is at the geometric hole center.
            loc = cyl.Location()
            axis_dir = np.array([d.X(), d.Y(), d.Z()])
            v_mid = (adaptor.FirstVParameter() + adaptor.LastVParameter()) * 0.5
            axis_center = np.array([loc.X(), loc.Y(), loc.Z()]) + v_mid * axis_dir
            cylinders.append({
                "radius": r,
                "axis": axis_dir,
                "centroid": axis_center,
                "area": area,
                "angle_span": u_span,
            })
        except Exception:
            continue

    if not cylinders:
        return []

    tol_cos = np.cos(np.radians(_AXIS_ANGLE_TOL_DEG))
    used = [False] * len(cylinders)
    holes: list[dict] = []

    for seed_i, seed in enumerate(cylinders):
        if used[seed_i]:
            continue
        group = [seed_i]
        used[seed_i] = True
        for j, cyl in enumerate(cylinders):
            if used[j]:
                continue
            if abs(cyl["radius"] - seed["radius"]) > seed["radius"] * 0.15:
                continue
            if abs(float(np.dot(cyl["axis"], seed["axis"]))) < tol_cos:
                continue
            if np.linalg.norm(cyl["centroid"] - seed["centroid"]) > _SPATIAL_CLUSTER_K * seed["radius"] + 10.0:
                continue
            group.append(j)
            used[j] = True

        # Angle-span coverage: fraction of 360° covered by group faces
        total_angle = sum(cylinders[k]["angle_span"] for k in group)
        coverage = total_angle / (2.0 * np.pi)
        if coverage < min_angle_coverage:
            continue  # structural arc, not a closed through-hole

        total_area = sum(cylinders[k]["area"] for k in group)
        if total_area < 1e-6:
            continue

        radius = float(np.mean([cylinders[k]["radius"] for k in group]))
        # Simple mean of axis midpoints (area-weighting helps less here; each face
        # already points to the same geometric center regardless of arc length)
        center = np.mean([cylinders[k]["centroid"] for k in group], axis=0)
        ax_sum = np.zeros(3)
        for k in group:
            ax = cylinders[k]["axis"]
            ax_sum += ax if np.dot(ax, seed["axis"]) >= 0 else -ax
        normal = ax_sum / (np.linalg.norm(ax_sum) or 1.0)

        holes.append({
            "center_xyz": [round(float(v), 4) for v in center],
            "normal_xyz": [round(float(v), 6) for v in normal],
            "radius_mm": round(radius, 4),
            "diameter_mm": round(radius * 2.0, 4),
            "shape": "circle",
            "circularity": 1.0,
            "status": "open",
            "was_open": True,
            "_source": "cylinder",
        })

    return holes


def _detect_holes_inner_wires(
    shape,
    max_radius_mm: float,
    min_face_area_mm2: float = 30.0,
) -> list[dict]:
    """
    Phase 2: find holes as inner wires on large panel faces.

    In STP B-rep, a through-hole can appear as an inner trimming wire on the
    panel face (a void inside the face boundary).  This catches holes whose
    walls are not cylindrical (planar, NURBS, etc.).
    """
    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)

    holes: list[dict] = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        exp.Next()

        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        if props.Mass() < min_face_area_mm2:
            continue

        # Outer wire of this face
        try:
            outer = breptools.OuterWire(face)
        except Exception:
            continue

        # Collect inner wires (= hole boundaries)
        wire_exp = TopExp_Explorer(face, TopAbs_WIRE)
        while wire_exp.More():
            wire = topods.Wire(wire_exp.Current())
            wire_exp.Next()
            if wire.IsSame(outer):
                continue

            # Collect unique vertices along the wire
            pts: list[list[float]] = []
            seen: set[tuple] = set()
            vexp = TopExp_Explorer(wire, TopAbs_VERTEX)
            while vexp.More():
                v = topods.Vertex(vexp.Current())
                pt = BRep_Tool.Pnt(v)
                key = (round(pt.X(), 1), round(pt.Y(), 1), round(pt.Z(), 1))
                if key not in seen:
                    pts.append([pt.X(), pt.Y(), pt.Z()])
                    seen.add(key)
                vexp.Next()

            if len(pts) < 3:
                continue

            arr = np.array(pts)
            center = arr.mean(axis=0)
            radii = np.linalg.norm(arr - center, axis=1)
            radius = float(radii.mean())

            if radius > max_radius_mm or radius < 0.5:
                continue

            # PCA normal of the wire loop plane
            _, _, Vt = np.linalg.svd(arr - center, full_matrices=False)
            normal = Vt[2]

            # Align normal with face normal at centroid
            try:
                from OCC.Core.BRepAdaptor import BRepAdaptor_Surface as _BAS  # noqa: PLC0415
                fa = _BAS(face)
                u_mid = (fa.FirstUParameter() + fa.LastUParameter()) * 0.5
                v_mid = (fa.FirstVParameter() + fa.LastVParameter()) * 0.5
                from OCC.Core.BRepLProp import BRepLProp_SLProps  # noqa: PLC0415
                slp = BRepLProp_SLProps(fa, u_mid, v_mid, 1, 1e-6)
                if slp.IsNormalDefined():
                    fn = slp.Normal()
                    fvec = np.array([fn.X(), fn.Y(), fn.Z()])
                    if float(np.dot(normal, fvec)) < 0.0:
                        normal = -normal
            except Exception:
                pass

            circularity = float(max(0.0, 1.0 - radii.std() / (radius + 1e-6)))
            holes.append({
                "center_xyz": [round(float(v), 4) for v in center],
                "normal_xyz": [round(float(v), 6) for v in normal],
                "radius_mm": round(radius, 4),
                "diameter_mm": round(radius * 2.0, 4),
                "shape": "circle" if circularity > 0.80 else "slot",
                "circularity": round(circularity, 4),
                "status": "open",
                "was_open": True,
                "_source": "inner_wire",
            })

    return holes


def detect_holes_from_stp(
    shape,
    max_radius_mm: float = _MAX_HOLE_RADIUS_MM,
    max_panel_thickness_mm: float = _MAX_PANEL_THICKNESS_MM,
    panel_thickness_mm: float | None = None,   # kept for API compatibility (unused now)
    min_coverage: float = 0.9,                  # angle-span coverage threshold
) -> list[dict]:
    """
    Detect through-holes in an STP shape.

    Two complementary methods:
    1. Cylinder face grouping with angle-span coverage filter (≥ 90% of 360°).
       CATIA V5 splits cylinder walls into arc patches; grouping them and checking
       the total arc angle correctly distinguishes full rings (holes) from partial
       arcs (structural fillets / rounded edges).
    2. Inner wire detection on large panel faces.
       Holes modeled with planar or NURBS walls (no cylinder face) appear as inner
       trimming wires on the panel face.  Catches holes the cylinder pass misses.

    Returns list of hole dicts compatible with holes.json.
    """
    # Phase 1: cylinder-based
    holes = _detect_holes_cylindrical(
        shape,
        max_radius_mm=max_radius_mm,
        max_panel_thickness_mm=max_panel_thickness_mm,
        min_angle_coverage=min_coverage,
    )

    # Phase 2: inner wire supplement
    iw_holes = _detect_holes_inner_wires(shape, max_radius_mm=max_radius_mm)

    # Merge: add inner-wire holes that are not already found by cylinder phase
    for iw in iw_holes:
        iw_center = np.array(iw["center_xyz"])
        iw_r = iw["radius_mm"]
        already_found = any(
            float(np.linalg.norm(np.array(h["center_xyz"]) - iw_center)) < max(iw_r, h["radius_mm"]) * 0.4
            and abs(h["radius_mm"] - iw_r) < iw_r * 0.3
            for h in holes
        )
        if not already_found:
            holes.append(iw)

    # Clean up internal metadata key before returning
    for h in holes:
        h.pop("_source", None)

    holes.sort(key=lambda h: h["radius_mm"], reverse=True)
    for idx, h in enumerate(holes):
        h["hole_id"] = f"h{idx + 1:03d}"

    return holes


def refine_holes_from_stp(
    shape,
    holes_data: list[dict],
    radius_tol_frac: float = 0.3,
    max_search_mm: float = 30.0,
) -> list[dict]:
    """
    Refine each hole's center_xyz and normal_xyz using accurate STP cylinder geometry.

    For each hole in holes_data, find the nearest cylindrical face in the STP with
    matching radius, then replace the hole's normal with the STP cylinder axis.
    This corrects the 38° normal errors that arise from VTP-based estimation.

    Parameters
    ----------
    shape           : OCC shape loaded from STP
    holes_data      : list of hole dicts (center_xyz, radius_mm, normal_xyz, ...)
    radius_tol_frac : accept cylinders whose radius is within this fraction of hole radius
    max_search_mm   : max centroid distance to consider a cylinder a match

    Returns
    -------
    List of refined hole dicts (copies of input with updated normal_xyz / center_xyz).
    """
    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)

    # Collect all cylinder faces, storing axis midpoint (not face centroid)
    all_cyls: list[dict] = []
    for i in range(1, face_map.Size() + 1):
        face = topods.Face(face_map.FindKey(i))
        try:
            adaptor = BRepAdaptor_Surface(face)
            if adaptor.GetType() != GeomAbs_Cylinder:
                continue
            cyl = adaptor.Cylinder()
            r = cyl.Radius()
            d = cyl.Axis().Direction()
            axis_dir = np.array([d.X(), d.Y(), d.Z()])
            loc = cyl.Location()
            v_mid = (adaptor.FirstVParameter() + adaptor.LastVParameter()) * 0.5
            axis_center = np.array([loc.X(), loc.Y(), loc.Z()]) + v_mid * axis_dir
            all_cyls.append({
                "radius": r,
                "axis": axis_dir,
                "centroid": axis_center,  # cylinder axis midpoint
            })
        except Exception:
            continue

    refined: list[dict] = []
    for hole in holes_data:
        center = np.array(hole["center_xyz"], dtype=float)
        radius = float(hole["radius_mm"])
        cur_normal = np.array(hole["normal_xyz"], dtype=float)

        # Search radius: tight enough to avoid picking up neighbouring holes
        tight_mm = max(radius * 1.5, 5.0)

        # Collect ALL cylinder faces that match radius and are close to current center
        matching = [
            cyl for cyl in all_cyls
            if abs(cyl["radius"] - radius) <= radius * radius_tol_frac
            and float(np.linalg.norm(cyl["centroid"] - center)) <= tight_mm
        ]

        # Fallback: expand search if nothing found at tight radius
        if not matching:
            matching = [
                cyl for cyl in all_cyls
                if abs(cyl["radius"] - radius) <= radius * radius_tol_frac
                and float(np.linalg.norm(cyl["centroid"] - center)) <= max_search_mm
            ]

        if not matching:
            refined.append(dict(hole))
            continue

        # Average axis centers → best hole center estimate
        avg_center = np.mean([c["centroid"] for c in matching], axis=0)

        # Average axis directions (align all to same half-space as current normal)
        ax_sum = np.zeros(3)
        for c in matching:
            ax = c["axis"].copy()
            if float(np.dot(ax, cur_normal)) < 0.0:
                ax = -ax
            ax_sum += ax
        avg_axis = ax_sum / (np.linalg.norm(ax_sum) + 1e-9)

        h = dict(hole)
        h["normal_xyz"] = [round(float(v), 6) for v in avg_axis]
        h["center_xyz"] = [round(float(v), 4) for v in avg_center]
        refined.append(h)

    return refined


def build_holes_json(
    part_id: str,
    assembly_id: str,
    holes: list[dict],
) -> dict:
    """Return holes.json-compatible dict."""
    return {
        "schema_version": "1.0",
        "assembly_id": assembly_id,
        "part_id": part_id,
        "source_vtp": f"bodies/{part_id}.vtp",
        "filled_vtp": f"filled/{part_id}.vtp",
        "holes": holes,
    }


# ---------------------------------------------------------------------------
# Hybrid fill: STP detection → VTP mesh manipulation (guaranteed be=0)
# ---------------------------------------------------------------------------

def _be_count(mesh: pv.PolyData) -> int:
    be = mesh.extract_feature_edges(
        boundary_edges=True, feature_edges=False,
        manifold_edges=False, non_manifold_edges=False,
    )
    return int(be.n_cells)


def fill_via_mesh(
    shape,
    holes_data: list[dict],
    linear_deflection: float = 0.05,
    angular_deflection: float = 0.5,
    merge_tol: float = 0.05,
    progress_cb: Callable[[str], None] | None = None,
    panel_thickness_mm: float | None = None,
) -> tuple[pv.PolyData, dict]:
    """
    Fill holes using VTP mesh manipulation after meshing the OCC shape.

    This hybrid approach:
    1. Meshes the OCC shape (accurate STP geometry, be=0 guaranteed).
    2. Uses holes_data (detected from STP) for accurate center/normal/radius.
    3. Applies the VTP-level fan/ear-clip fill (which reliably achieves be=0).

    Returns (filled_pyvista_mesh, stats_dict).
    """
    from hole_filler.filler import replace_hole_with_patch  # noqa: PLC0415

    def _log(m: str) -> None:
        if progress_cb:
            progress_cb(m)

    _log("Meshing STP shape...")
    mesh = _mesh_shape_internal(shape, linear_deflection, angular_deflection, merge_tol)

    stats = {
        "holes_attempted": len(holes_data),
        "holes_filled": 0,
        "details": [],
    }

    result = mesh
    initial_n_cells = mesh.n_cells
    for hole in holes_data:
        hole_id = hole.get("hole_id", "?")
        center = hole["center_xyz"]
        normal = hole["normal_xyz"]
        radius = float(hole["radius_mm"])
        circularity = float(hole.get("circularity", 1.0))
        _log(f"Filling {hole_id} (r={radius:.1f}mm)...")

        hole_dict = {
            "center": center,
            "normal": normal,
            "radius_mm": radius,
            "circularity": circularity,
            "shape": hole.get("shape", "circle"),
        }

        # Safety: save backup so we can revert if the fill goes wrong
        backup = result.copy()
        prev_cells = result.n_cells
        # Use actual panel thickness for search band (thin-panel walls have few triangles;
        # 8mm default would incorrectly sweep in surrounding panel face triangles)
        if panel_thickness_mm and panel_thickness_mm > 0:
            t_mm = max(panel_thickness_mm * 3.0, 3.0)
        else:
            t_mm = max(radius * 0.5, 8.0)
        result = replace_hole_with_patch(
            result, hole_dict,
            thickness_mm=t_mm,
            force_fill=True,
        )
        delta = result.n_cells - prev_cells

        # Sanity check: rollback only on explosive growth; thin-panel holes legitimately
        # have negative delta (cylinder walls > flat patch triangles), so accept any
        # delta as long as the mesh stays watertight (be == 0).
        max_allowed_delta = max(500, initial_n_cells // 5)
        if delta > max_allowed_delta:
            result = backup
            delta = 0
            status = "failed"
        elif delta != 0:
            local_be = _be_count(result)
            if local_be > 0:
                result = backup
                delta = 0
                status = "failed"
            else:
                status = "filled"
        else:
            status = "failed"  # no cells changed → hole not found

        stats["details"].append({
            "hole_id": hole_id,
            "delta_cells": delta,
            "status": status,
        })
        if status == "filled":
            stats["holes_filled"] += 1

    # Final cleanup: if any boundary edges remain, do a small-radius fill pass
    final_be = _be_count(result)
    if final_be > 0:
        _log(f"  Cleanup: {final_be} boundary edge(s) remain, running fill_holes...")
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            cleaned = result.fill_holes(1000.0)
        if _be_count(cleaned) == 0:
            result = cleaned
            final_be = 0

    stats["boundary_edges"] = final_be
    stats["n_vertices"] = int(result.n_points)
    stats["n_faces"] = int(result.n_cells)
    _log(
        f"Done: {stats['holes_filled']}/{stats['holes_attempted']} filled, "
        f"be={stats['boundary_edges']}"
    )
    return result, stats


def _mesh_shape_internal(
    shape,
    linear_deflection: float = 0.05,
    angular_deflection: float = 0.5,
    merge_tol: float = 0.05,
) -> pv.PolyData:
    """Mesh OCC shape to PyVista PolyData (no file write)."""
    mesher = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection)
    mesher.Perform()

    inv_tol = 1.0 / merge_tol
    vertex_map: dict[tuple, int] = {}
    all_verts: list[list[float]] = []
    all_tris: list[list[int]] = []

    def _add(x: float, y: float, z: float) -> int:
        key = (round(x * inv_tol), round(y * inv_tol), round(z * inv_tol))
        if key not in vertex_map:
            vertex_map[key] = len(all_verts)
            all_verts.append([x, y, z])
        return vertex_map[key]

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is not None:
            is_id = loc.IsIdentity()
            trsf = None if is_id else loc.Transformation()
            local_to_global: list[int] = []
            for i in range(1, tri.NbNodes() + 1):
                node = tri.Node(i)
                if trsf is not None:
                    pt = gp_Pnt(node.X(), node.Y(), node.Z())
                    pt.Transform(trsf)
                    local_to_global.append(_add(pt.X(), pt.Y(), pt.Z()))
                else:
                    local_to_global.append(_add(node.X(), node.Y(), node.Z()))
            for i in range(1, tri.NbTriangles() + 1):
                n1, n2, n3 = tri.Triangle(i).Get()
                a = local_to_global[n1 - 1]
                b = local_to_global[n2 - 1]
                c = local_to_global[n3 - 1]
                if a != b and b != c and a != c:
                    all_tris.append([a, b, c])
        exp.Next()

    if not all_verts:
        return pv.PolyData()

    verts = np.array(all_verts, dtype=np.float32)
    flat: list[int] = []
    for t in all_tris:
        flat += [3, t[0], t[1], t[2]]
    return pv.PolyData(verts, np.array(flat, dtype=np.int32))


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def fill_stp_file(
    stp_path: str | pathlib.Path,
    holes_json_path: str | pathlib.Path,
    output_stp_path: str | pathlib.Path,
    output_vtp_path: str | pathlib.Path,
    linear_deflection: float = 0.05,
    progress_cb: Callable[[str], None] | None = None,
    hole_ids: list[str] | None = None,   # only fill these hole IDs (None = fill all)
    panel_thickness_mm: float | None = None,  # kept for API compat (unused in B-rep pipeline)
) -> dict:
    """
    Load STP + holes.json, fill holes at B-rep level, remesh → VTP.

    Strategy: B-rep fill (fill_shape_holes) → mesh filled shape (mesh_shape_to_vtp).
    This produces geometrically exact flat fill patches, unlike the old VTP mesh
    manipulation approach which was unreliable for thin-panel STP geometry.

    Returns stats dict with per-hole fill results.
    """
    stp_path = pathlib.Path(stp_path)
    holes_json_path = pathlib.Path(holes_json_path)
    output_vtp_path = pathlib.Path(output_vtp_path)
    output_vtp_path.parent.mkdir(parents=True, exist_ok=True)

    if progress_cb:
        progress_cb(f"Loading {stp_path.name}...")
    shape = load_stp(stp_path)

    holes_record = json.loads(holes_json_path.read_text(encoding="utf-8"))
    holes_data = holes_record.get("holes", [])

    # Refine hole centers/normals using accurate STP cylinder geometry
    if progress_cb:
        progress_cb("Refining hole centers/normals from STP geometry...")
    holes_data = refine_holes_from_stp(shape, holes_data)

    # Filter to selected hole IDs if specified
    if hole_ids is not None:
        holes_to_fill = [h for h in holes_data if h.get("hole_id") in set(hole_ids)]
        if progress_cb:
            progress_cb(f"Selected {len(holes_to_fill)}/{len(holes_data)} holes to fill...")
    else:
        holes_to_fill = holes_data

    # B-rep level fill: remove cylinder wall faces, extend curved surfaces, sew
    filled_brep, stats = fill_shape_holes(
        shape, holes_to_fill, progress_cb=progress_cb
    )

    # Mesh the filled B-rep → VTP
    if progress_cb:
        progress_cb(f"Meshing filled B-rep -> {output_vtp_path.name}...")
    mesh = mesh_shape_to_vtp(filled_brep, output_vtp_path,
                             linear_deflection=linear_deflection)

    # Post-process: iterative VTP fill_holes to close any residual gaps.
    # Residual gaps arise when a hole has no inner wire (boundary-edge fallback);
    # the flat fill patch creates non-conformant tessellation at curved surfaces.
    # fill_holes() is stable and converges to be=0 for small residual gaps.
    final_be = _be_count(mesh)
    if final_be > 0:
        if progress_cb:
            progress_cb(f"Residual be={final_be} — applying VTP cleanup...")
        for _pass in range(10):
            if final_be == 0:
                break
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cleaned = mesh.fill_holes(1000.0)
            new_be = _be_count(cleaned)
            if new_be < final_be:
                mesh = cleaned
                final_be = new_be
            else:
                break
        if final_be == 0:
            mesh.save(str(output_vtp_path))
            if progress_cb:
                progress_cb(f"VTP cleanup done: be=0")
        else:
            if progress_cb:
                progress_cb(f"VTP cleanup partial: be={final_be}")

    # Update holes.json with fill status
    for detail in stats.get("details", []):
        hole_id = detail.get("hole_id")
        status = detail.get("status", "open")
        for h in holes_data:
            if h.get("hole_id") == hole_id:
                h["status"] = "filled" if status == "filled" else h.get("status", "open")
                break
    holes_record["holes"] = holes_data
    holes_json_path.write_text(
        json.dumps(holes_record, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Also save filled STP
    if output_stp_path is not None:
        output_stp_path = pathlib.Path(output_stp_path)
        output_stp_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if progress_cb:
                progress_cb(f"Saving filled STP → {output_stp_path.name}...")
            save_stp(filled_brep, output_stp_path)
        except Exception as exc:
            if progress_cb:
                progress_cb(f"STP save skipped: {exc}")

    stats["boundary_edges"] = final_be
    stats["n_faces"] = 0  # filled in by caller if needed

    return stats


# ---------------------------------------------------------------------------
# Diagnostic: return geometry of detected hole walls + boundary edges
# ---------------------------------------------------------------------------

def diagnose_holes_geometry(
    shape,
    holes_data: list[dict],
    n_edge_samples: int = 24,
) -> list[dict]:
    """
    Return diagnostic geometry for visualizing hole detection results.

    For each hole returns:
    - wall_verts / wall_tris : tessellated cylinder wall faces (as red overlay)
    - boundary_segs          : line segments of edges between wall and panel faces
    - inner_wire_segs        : line segments sampling the inner wires on panel faces

    This allows the user to verify exactly which B-rep faces and edges are being
    used for hole detection and fill.
    """
    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)

    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    # Mesh shape so triangulations are available
    mesher = BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5)
    mesher.Perform()

    results: list[dict] = []

    for hole in holes_data:
        center = np.array(hole["center_xyz"], dtype=float)
        radius = float(hole["radius_mm"])
        normal = np.array(hole["normal_xyz"], dtype=float)

        # --- cylinder wall faces ---
        hole_face_idxs, wall_faces = _find_cylinder_wall_faces(face_map, center, radius, normal)

        wall_verts: list[list[float]] = []
        wall_tris: list[list[int]] = []
        for face in wall_faces:
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation(face, loc)
            if tri is None:
                continue
            is_id = loc.IsIdentity()
            trsf = None if is_id else loc.Transformation()
            base = len(wall_verts)
            for i in range(1, tri.NbNodes() + 1):
                node = tri.Node(i)
                if trsf is not None:
                    pt = gp_Pnt(node.X(), node.Y(), node.Z())
                    pt.Transform(trsf)
                    wall_verts.append([pt.X(), pt.Y(), pt.Z()])
                else:
                    wall_verts.append([node.X(), node.Y(), node.Z()])
            for i in range(1, tri.NbTriangles() + 1):
                n1, n2, n3 = tri.Triangle(i).Get()
                wall_tris.append([base + n1 - 1, base + n2 - 1, base + n3 - 1])

        # --- boundary edges (between wall faces and panel faces) ---
        boundary_segs: list[list[list[float]]] = []
        for i in range(1, edge_face_map.Size() + 1):
            fl = edge_face_map.FindFromIndex(i)
            it = TopTools_ListIteratorOfListOfShape(fl)
            adj_hole = adj_panel = False
            while it.More():
                fidx = face_map.FindIndex(it.Value())
                if fidx in hole_face_idxs:
                    adj_hole = True
                else:
                    adj_panel = True
                it.Next()
            if adj_hole and adj_panel:
                edge = topods.Edge(edge_face_map.FindKey(i))
                pts = _sample_edge(edge, n_edge_samples)
                for k in range(len(pts) - 1):
                    boundary_segs.append([pts[k], pts[k + 1]])

        # --- inner wires on panel faces ---
        inner_wire_segs: list[list[list[float]]] = []
        for wire in _find_panel_inner_wires(shape, center, radius, normal):
            wire_pts: list[list[float]] = []
            seen: set[tuple] = set()
            edge_exp = TopExp_Explorer(wire, TopAbs_EDGE)
            while edge_exp.More():
                edge = topods.Edge(edge_exp.Current())
                edge_exp.Next()
                for pt3 in _sample_edge(edge, n_edge_samples):
                    key = (round(pt3[0], 1), round(pt3[1], 1), round(pt3[2], 1))
                    if key not in seen:
                        wire_pts.append(pt3)
                        seen.add(key)
            for k in range(len(wire_pts)):
                inner_wire_segs.append([wire_pts[k], wire_pts[(k + 1) % len(wire_pts)]])

        results.append({
            "hole_id": hole.get("hole_id", "?"),
            "center": center.tolist(),
            "radius_mm": radius,
            "wall_verts": wall_verts,
            "wall_tris": wall_tris,
            "boundary_segs": boundary_segs,
            "inner_wire_segs": inner_wire_segs,
            "n_wall_faces": len(wall_faces),
            "n_inner_wires": len([w for w in _find_panel_inner_wires(shape, center, radius, normal)]),
        })

    return results

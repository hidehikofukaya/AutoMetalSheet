"""
hole_filler.py -- Detect and fill holes in sheet metal B-Rep solids.

detect_holes() is comprehensive: it finds ANY closed inner-wire surface
(any shape, any size -- not just true cylinders), by BFS-walking the wall
faces between an inner wire and its matching far cap. This is necessary
because real sheet-metal holes are often non-cylindrical (stamped slots,
stepped/chamfered bores, multi-facet B-spline bores) and a cylinder-only
filter silently misses them.

Hole classification by diameter (applied to ALL detected holes, for
reporting; only jig/bolt holes are auto-filled in batch mode):
  <= JIG_DIAM_MM  (default 5 mm)  : jig / pilot holes -> fill silently
  <= BOLT_DIAM_MM (default 15 mm) : bolt / fastener holes -> fill + record in JSON
  >  BOLT_DIAM_MM                 : design features (lightening holes, etc.) -> keep, record only

R6-03 invariant (see CLAUDE.md): batch auto-fill must never remove a real
design feature, so process_body() filters detected holes to
diameter <= BOLT_DIAM_MM before calling fill_holes(), even though
detect_holes() itself reports every hole found, of any size.

A small number of BFS-walked clusters are excluded entirely (not reported
as holes at all): clusters whose merged wall-face count exceeds
CLUSTER_FACE_COUNT_LIMIT are almost certainly a BFS leak into the part's
own outer skin (a known algorithm limitation -- see CLUSTER_FACE_COUNT_LIMIT
docstring below), not a real hole.

The cylindrical/non-cylindrical hole-wall faces are removed using
BRepAlgoAPI_Defeaturing, which automatically closes the void by
extending/trimming the adjacent planar faces.

Usage
-----
Single file:
    python hole_filler.py bodies/body003.stp

Batch from hierarchy.json (all sheet_metal bodies):
    python hole_filler.py --hierarchy C:/.../output/A0072600002/hierarchy.json

Batch across entire output directory:
    python hole_filler.py --batch-dir C:/.../Mesh_Generater/output

Dry-run (detect only, no output files):
    python hole_filler.py --hierarchy hierarchy.json --dry-run

Output per processed file
--------------------------
bodies/body003_filled.stp          -- defeature solid
bodies/body003_filled_holes.json   -- hole metadata (center, radius, category)
"""

import argparse
import json
import math
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

# ── pythonocc ────────────────────────────────────────────────────────────────
from OCC.Core.STEPControl import (STEPControl_Reader,
                                   STEPControl_Writer,
                                   STEPControl_AsIs)
from OCC.Core.IFSelect       import IFSelect_RetDone
from OCC.Core.TopAbs         import TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE
from OCC.Core.TopExp         import TopExp_Explorer, topexp
from OCC.Core.TopTools       import (TopTools_IndexedDataMapOfShapeListOfShape,
                                      TopTools_ListIteratorOfListOfShape)
from OCC.Core.TopoDS         import topods
from OCC.Core.BRep           import BRep_Tool
from OCC.Core.BRepTools      import breptools
from OCC.Core.BRepGProp      import brepgprop
from OCC.Core.GProp          import GProp_GProps
from OCC.Core.GeomAdaptor    import GeomAdaptor_Surface
from OCC.Core.GeomAbs        import GeomAbs_Cylinder
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.BRepBndLib     import brepbndlib
from OCC.Core.Bnd            import Bnd_Box
from OCC.Core.BRepAlgoAPI    import BRepAlgoAPI_Defeaturing

# ── thresholds ────────────────────────────────────────────────────────────────
JIG_DIAM_MM  =  5.0   # diameter: fill silently  (jig / pilot holes)
BOLT_DIAM_MM = 15.0   # diameter: fill + record  (bolt / fastener holes)

# Post-hoc cluster-level cutoff for the BFS-based detector below: a full
# 107-body hierarchy sweep found every legitimate hole (incl. a 50-face
# stamped slot and a 29-face stepped hole) at n_wall<=50 merged faces,
# while every confirmed BFS-runaway cluster (the part's own outer skin
# leaking through one bad opposite-cap match) starts at n_wall=153 -- a
# >3x empirical gap. 60 sits in the middle. Clusters exceeding this are
# excluded as "indeterminate" rather than reported as a bogus giant hole.
# This targets the symptom (cluster size), not the root cause (the
# matched_cap-detection gap that lets BFS leak past a real hole's wall);
# revisit with proper concave/convex edge classification (AAG) if a body
# is found whose genuine hole exceeds the threshold or whose runaway leak
# stays under it.
CLUSTER_FACE_COUNT_LIMIT = 60

# BRepAlgoAPI_Defeaturing.Build() has no native timeout and is known to loop
# indefinitely on certain hole geometries even when removing small faces --
# an unfixed OCCT defect (dev.opencascade.org/content/brepalgoapidefeaturing-never-finishes),
# reproduced in this project on a body with only <=15mm "bolt" holes (no
# >15mm "design_feature" holes involved). Each hole is therefore defeatured
# in its own subprocess (see _fill_one_worker) so a hang can be killed by
# wall-clock timeout without losing every other hole in the same body.
DEFAULT_FILL_TIMEOUT_S = 60.0

# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_step(stp: pathlib.Path):
    reader = STEPControl_Reader()
    if reader.ReadFile(str(stp)) != IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed: {stp}")
    reader.TransferRoots()
    return reader.OneShape()


def _write_step(shape, out: pathlib.Path):
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    if writer.Write(str(out)) != IFSelect_RetDone:
        raise RuntimeError(f"STEP write failed: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Topology helpers (BFS hole-wall walk)
# ─────────────────────────────────────────────────────────────────────────────

def _all_faces(shape):
    seen, out = [], []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        f = topods.Face(exp.Current())
        if not any(f.IsSame(s) for s in seen):
            seen.append(f)
            out.append(f)
        exp.Next()
    return out


def _wires_of(face):
    out = []
    wexp = TopExp_Explorer(face, TopAbs_WIRE)
    while wexp.More():
        out.append(topods.Wire(wexp.Current()))
        wexp.Next()
    return out


def _inner_wires(face):
    outer = breptools.OuterWire(face)
    return [w for w in _wires_of(face) if not w.IsSame(outer)]


def _wire_edges(wire):
    out = []
    eexp = TopExp_Explorer(wire, TopAbs_EDGE)
    while eexp.More():
        out.append(topods.Edge(eexp.Current()))
        eexp.Next()
    return out


def _all_face_wire_edges(face):
    out = []
    for w in _wires_of(face):
        out.extend(_wire_edges(w))
    return out


def _build_edge_face_map(shape):
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, m)
    return m


def _faces_of_edge(edge_face_map, edge):
    idx = edge_face_map.FindIndex(edge)
    if idx == 0:
        return []
    lst = edge_face_map.FindFromIndex(idx)
    it = TopTools_ListIteratorOfListOfShape(lst)
    out = []
    while it.More():
        out.append(topods.Face(it.Value()))
        it.Next()
    return out


def _face_has_edge_in_own_inner_wire(face, edge):
    for iw in _inner_wires(face):
        for e in _wire_edges(iw):
            if e.IsSame(edge):
                return True
    return False


def _wire_planar_props(wire):
    mk = BRepBuilderAPI_MakeFace(wire, True)
    if not mk.IsDone():
        return None
    gp = GProp_GProps()
    brepgprop.SurfaceProperties(mk.Face(), gp)
    c = gp.CentreOfMass()
    return gp.Mass(), (c.X(), c.Y(), c.Z())


def _wire_bbox_metrics(wire):
    box = Bnd_Box()
    brepbndlib.Add(wire, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    center = ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
    extent = max(xmax - xmin, ymax - ymin, zmax - zmin)
    return extent, center


def _collect_hole_faces(edge_face_map, cap_face, inner_wire):
    """BFS-walk wall faces from *inner_wire* outward until each branch hits
    a face that contains the traversal edge in its OWN inner wire (the
    matching opposite cap). Early-exits past CLUSTER_FACE_COUNT_LIMIT since
    a cluster that large is going to be excluded as indeterminate anyway."""
    visited_edges = list(_wire_edges(inner_wire))
    frontier = list(visited_edges)
    wall_faces = []
    matched_cap = None
    while frontier:
        e = frontier.pop()
        for f in _faces_of_edge(edge_face_map, e):
            if f.IsSame(cap_face):
                continue
            if any(f.IsSame(wf) for wf in wall_faces):
                continue
            if _face_has_edge_in_own_inner_wire(f, e):
                matched_cap = f
                continue
            wall_faces.append(f)
            for e2 in _all_face_wire_edges(f):
                if not any(e2.IsSame(ve) for ve in visited_edges):
                    visited_edges.append(e2)
                    frontier.append(e2)
            if len(wall_faces) > CLUSTER_FACE_COUNT_LIMIT:
                return wall_faces, matched_cap
    return wall_faces, matched_cap


def _hole_geometry(inner_wire, wall_faces):
    planar = _wire_planar_props(inner_wire)
    if planar is not None:
        area, center = planar
        diam = math.sqrt(4.0 * area / math.pi)
    else:
        diam, center = _wire_bbox_metrics(inner_wire)

    axis = None
    is_simple_cylinder = False
    if len(wall_faces) == 1:
        adaptor = GeomAdaptor_Surface(BRep_Tool.Surface(wall_faces[0]))
        if adaptor.GetType() == GeomAbs_Cylinder:
            d = adaptor.Cylinder().Axis().Direction()
            axis = (d.X(), d.Y(), d.Z())
            is_simple_cylinder = True
    if axis is None:
        axis = (0.0, 0.0, 1.0)
    return diam, center, axis, is_simple_cylinder

# ─────────────────────────────────────────────────────────────────────────────
# Hole detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_holes(shape, max_diam_mm: float = None) -> list:
    """
    Find every closed inner-wire surface in *shape* -- any shape, any size,
    not just true cylinders -- by BFS-walking each inner wire's wall faces
    to its matching opposite cap (see _collect_hole_faces). The same
    physical hole is discovered once per cap/step-boundary it has;
    candidates that share >=1 wall face are merged into one cluster, and
    the cluster reports geometry from its largest-diameter member (this
    correctly handles multi-step/chamfered holes).

    Clusters whose merged face count exceeds CLUSTER_FACE_COUNT_LIMIT are
    dropped entirely (see its docstring) -- they are BFS leaks, not holes.

    *max_diam_mm*, if given, filters the returned list to diameter <= that
    value (used by older call sites that want only small holes). Default
    None returns every detected hole regardless of size.

    Returns list of dicts:
        faces       : list[OCC TopoDS_Face]  (removed from the returned JSON)
        n_faces     : int
        radius_mm   : float
        diameter_mm : float
        center_xyz  : [x, y, z]
        axis_xyz    : [dx, dy, dz]  (meaningful only if is_simple_cylinder)
        is_simple_cylinder : bool
        through     : bool  (True if BFS found a matching opposite cap)
        category    : 'jig' | 'bolt' | 'design_feature'
    """
    edge_face_map = _build_edge_face_map(shape)
    all_faces = _all_faces(shape)

    candidates = []
    for cap_face in all_faces:
        for iw in _inner_wires(cap_face):
            wall_faces, matched_cap = _collect_hole_faces(edge_face_map, cap_face, iw)
            if not wall_faces:
                continue
            diam, center, axis, is_simple_cyl = _hole_geometry(iw, wall_faces)
            candidates.append(dict(wall_faces=wall_faces, diam=diam, center=center,
                                    axis=axis, is_simple_cylinder=is_simple_cyl,
                                    through=matched_cap is not None))

    # Cluster candidates that share >=1 wall face.
    clusters = []
    for cand in candidates:
        hit = None
        for cl in clusters:
            if any(any(wf.IsSame(cf) for cf in cl["faces"]) for wf in cand["wall_faces"]):
                hit = cl
                break
        if hit is None:
            clusters.append({"faces": list(cand["wall_faces"]), "members": [cand]})
        else:
            hit["members"].append(cand)
            for wf in cand["wall_faces"]:
                if not any(wf.IsSame(cf) for cf in hit["faces"]):
                    hit["faces"].append(wf)

    holes = []
    for cl in clusters:
        if len(cl["faces"]) > CLUSTER_FACE_COUNT_LIMIT:
            # Almost certainly a BFS leak into the part's own outer skin
            # (see CLUSTER_FACE_COUNT_LIMIT docstring) -- not a real hole.
            continue
        best = max(cl["members"], key=lambda c: c["diam"])
        diam = best["diam"]
        if max_diam_mm is not None and diam > max_diam_mm:
            continue
        holes.append({
            "faces":        cl["faces"],
            "n_faces":      len(cl["faces"]),
            "radius_mm":    round(diam / 2, 3),
            "diameter_mm":  round(diam, 3),
            "center_xyz":   [round(v, 3) for v in best["center"]],
            "axis_xyz":     [round(v, 3) for v in best["axis"]],
            "is_simple_cylinder": best["is_simple_cylinder"],
            "through":      best["through"],
            "category":     ("jig" if diam <= JIG_DIAM_MM
                              else "bolt" if diam <= BOLT_DIAM_MM
                              else "design_feature"),
        })
    return holes

# ─────────────────────────────────────────────────────────────────────────────
# Defeaturing
# ─────────────────────────────────────────────────────────────────────────────

def _defeature_match(shape, target_center, tol_mm: float = 5.0):
    """Re-detect holes in *shape* and return the one whose center_xyz is
    closest to *target_center* (within tol_mm), or None. Used by the
    subprocess worker to re-locate a hole after earlier holes in the same
    body have already been defeatured (which shifts face indices/identity,
    so a pre-fill face list can't be reused directly)."""
    best, best_d = None, None
    for h in detect_holes(shape, max_diam_mm=None):
        d = math.dist(target_center, h["center_xyz"])
        if d > tol_mm:
            continue
        if best is None or d < best_d:
            best, best_d = h, d
    return best


def _fill_one_worker(in_stp_str: str, target_json: str, out_stp_str: str):
    """Subprocess entry point: defeature exactly ONE hole (matched by
    center_xyz against *target_json*) and write the result to out_stp_str.
    Invoked as: python hole_filler.py _fill_one_worker <in.stp> <target_json> <out.stp>

    Exit codes:
      0  success, out_stp_str written
      2  Build() ran but IsDone() was False (clean defeaturing failure)
      3  no matching hole found in the current shape (topology drift)
    """
    target = json.loads(target_json)
    shape = _read_step(pathlib.Path(in_stp_str))
    match = _defeature_match(shape, target["center_xyz"])
    if match is None:
        print(f"_fill_one_worker: no hole found near {target['center_xyz']} "
              f"(target diam={target['diameter_mm']}mm)", file=sys.stderr)
        sys.exit(3)

    df = BRepAlgoAPI_Defeaturing()
    df.SetShape(shape)
    df.SetRunParallel(False)
    for f in match["faces"]:
        df.AddFaceToRemove(f)
    df.Build()

    if not df.IsDone():
        try:
            if df.HasErrors():
                df.DumpErrors(sys.stderr)
            if df.HasWarnings():
                df.DumpWarnings(sys.stderr)
        except Exception:
            pass
        sys.exit(2)

    _write_step(df.Shape(), pathlib.Path(out_stp_str))
    sys.exit(0)


def fill_holes(shape, holes: list, timeout_s: float = DEFAULT_FILL_TIMEOUT_S):
    """
    Defeature *holes* ONE AT A TIME, each in its own subprocess bounded by
    *timeout_s* wall-clock seconds (see DEFAULT_FILL_TIMEOUT_S docstring
    above for why: BRepAlgoAPI_Defeaturing.Build() can hang indefinitely
    and offers no native timeout). A hole that times out, fails cleanly, or
    can no longer be located is skipped -- every other hole in the same
    call is still attempted independently.

    Returns (result_shape, results) where results is a list of dicts
    aligned 1:1 with *holes*:
        {"diameter_mm": float, "category": str,
         "fill_status": "ok" | "failed" | "timeout" | "not_found"}
    """
    if not holes:
        return shape, []

    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="hole_fill_"))
    worker_script = str(pathlib.Path(__file__).resolve())
    cur_stp = tmpdir / "step_00.stp"
    _write_step(shape, cur_stp)

    results = []
    try:
        for i, h in enumerate(holes, start=1):
            target = json.dumps({"center_xyz": h["center_xyz"], "diameter_mm": h["diameter_mm"]})
            next_stp = tmpdir / f"step_{i:02d}.stp"
            try:
                proc = subprocess.run(
                    [sys.executable, worker_script, "_fill_one_worker",
                     str(cur_stp), target, str(next_stp)],
                    timeout=timeout_s, capture_output=True,
                )
            except subprocess.TimeoutExpired:
                results.append({"diameter_mm": h["diameter_mm"], "category": h["category"],
                                 "fill_status": "timeout"})
                continue

            if proc.returncode == 0 and next_stp.exists():
                cur_stp = next_stp
                results.append({"diameter_mm": h["diameter_mm"], "category": h["category"],
                                 "fill_status": "ok"})
            elif proc.returncode == 3:
                results.append({"diameter_mm": h["diameter_mm"], "category": h["category"],
                                 "fill_status": "not_found"})
            else:
                results.append({"diameter_mm": h["diameter_mm"], "category": h["category"],
                                 "fill_status": "failed"})

        result_shape = _read_step(cur_stp)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return result_shape, results

# ─────────────────────────────────────────────────────────────────────────────
# Per-body processing
# ─────────────────────────────────────────────────────────────────────────────

def process_body(stp_path: pathlib.Path, dry_run: bool = False) -> dict:
    """
    Process a single body###.stp.
    Outputs body###_filled.stp and body###_filled_holes.json alongside the input.
    Returns a metadata dict.

    Detection is comprehensive (any shape/size, see detect_holes()) so the
    JSON report covers every real hole found. Auto-fill is restricted to
    diameter <= BOLT_DIAM_MM (R6-03 invariant: never auto-remove a real
    design feature larger than the bolt-hole ceiling).
    """
    t0 = time.time()

    out_stp  = stp_path.parent / (stp_path.stem + "_filled.stp")
    out_json = stp_path.parent / (stp_path.stem + "_filled_holes.json")

    print(f"  [{stp_path.stem}] reading ...", end=" ", flush=True)
    try:
        shape = _read_step(stp_path)
    except Exception as e:
        print(f"ERROR: {e}")
        return {"source": str(stp_path), "status": "read_error", "error": str(e)}

    all_holes = detect_holes(shape, max_diam_mm=None)
    fillable  = [h for h in all_holes if h["diameter_mm"] <= BOLT_DIAM_MM]
    kept      = [h for h in all_holes if h["diameter_mm"] >  BOLT_DIAM_MM]
    for h in kept:
        h["fill_status"] = "not_attempted_gt_bolt_diam"
    n_jig  = sum(1 for h in fillable if h["category"] == "jig")
    n_bolt = sum(1 for h in fillable if h["category"] == "bolt")
    print(f"holes={len(all_holes)} (jig={n_jig} bolt={n_bolt} kept={len(kept)})", flush=True)

    status = "ok"
    if not dry_run:
        if fillable:
            result, fill_results = fill_holes(shape, fillable)
            for h, r in zip(fillable, fill_results):
                h["fill_status"] = r["fill_status"]
            n_bad = sum(1 for r in fill_results if r["fill_status"] != "ok")
            if n_bad:
                bad_kinds = ", ".join(sorted({r["fill_status"] for r in fill_results
                                               if r["fill_status"] != "ok"}))
                print(f"    WARNING: {n_bad}/{len(fillable)} hole(s) not filled ({bad_kinds})")
                status = "defeaturing_partial" if n_bad < len(fillable) else "defeaturing_failed"
        else:
            result = shape
            status = "no_holes"
    else:
        for h in fillable:
            h["fill_status"] = "not_attempted_dry_run"

    meta = {
        "source_stp":     str(stp_path),
        "filled_stp":     str(out_stp),
        "jig_diam_mm":    JIG_DIAM_MM,
        "bolt_diam_mm":   BOLT_DIAM_MM,
        "n_holes_total":  len(all_holes),
        "n_holes_filled": len(fillable),
        "n_holes_kept":   len(kept),
        "holes": [
            {k: v for k, v in h.items() if k != "faces"}
            for h in all_holes
        ],
        "status":         status,
        "elapsed_s":      0.0,
    }

    if not dry_run:
        _write_step(result, out_stp)
        out_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"    -> {out_stp.name}  ({time.time()-t0:.1f}s)")

    meta["elapsed_s"] = round(time.time() - t0, 2)
    return meta

# ─────────────────────────────────────────────────────────────────────────────
# Batch helpers
# ─────────────────────────────────────────────────────────────────────────────

def batch_from_hierarchy(hierarchy_json: pathlib.Path, dry_run: bool = False) -> list:
    """
    Process all sheet_metal bodies listed in a hierarchy.json produced by
    extract_solid_step.py.

    Supports the actual field names used by the extractor:
      tag       : 'sheet_metal' | 'thick_part' | 'small_hardware' | 'empty' | 'adhesive'
      stp_file  : relative path  'bodies/body002.stp'  (may be None for empty bodies)
      body_idx  : 1-based integer body index
    """
    data    = json.loads(hierarchy_json.read_text("utf-8"))
    root    = hierarchy_json.parent
    bodies  = data.get("bodies", [])

    # Support both 'tag' (actual field) and 'classification' (alias)
    sm_bodies = [b for b in bodies
                 if b.get("tag", b.get("classification", "")) == "sheet_metal"]

    print(f"\n{'DRY RUN -- ' if dry_run else ''}"
          f"{root.name}: {len(sm_bodies)} sheet_metal / {len(bodies)} total bodies")

    results = []
    for b in sm_bodies:
        # Resolve the STP file path
        rel = b.get("stp_file")
        if rel:
            stp = root / rel
        else:
            # Fallback: derive from body_idx
            idx = b.get("body_idx", b.get("index"))
            if idx is None:
                print(f"  SKIP: no stp_file and no body_idx in {b}")
                continue
            stp = root / "bodies" / f"body{int(idx):03d}.stp"

        if not stp.exists():
            print(f"  SKIP (not found): {stp}")
            continue
        try:
            meta = process_body(stp, dry_run=dry_run)
        except Exception as e:
            print(f"  ERROR: {stp.name}: {e}")
            meta = {"source": str(stp), "status": "error", "error": str(e)}
        results.append(meta)

    print(f"  Done ({len(results)} processed)")
    return results


def batch_from_dir(root: pathlib.Path, dry_run: bool = False) -> list:
    hfiles = sorted(root.rglob("hierarchy.json"))
    print(f"Found {len(hfiles)} hierarchy.json files under {root}")
    all_results = []
    for hf in hfiles:
        all_results.extend(batch_from_hierarchy(hf, dry_run=dry_run))
    return all_results

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _jig_default  = JIG_DIAM_MM
    _bolt_default = BOLT_DIAM_MM

    ap = argparse.ArgumentParser(
        description="Detect and fill jig/bolt holes in sheet metal STEP solids.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("stp",          nargs="?",
                    help="Single body###.stp to process")
    ap.add_argument("--hierarchy",  metavar="PATH",
                    help="hierarchy.json -- process all sheet_metal bodies")
    ap.add_argument("--batch-dir",  metavar="PATH",
                    help="Root output dir -- process every hierarchy.json found")
    ap.add_argument("--dry-run",    action="store_true",
                    help="Detect holes only; write no output files")
    ap.add_argument("--jig-diam",   type=float, default=_jig_default,
                    help=f"Max jig-hole diameter mm (default {_jig_default})")
    ap.add_argument("--bolt-diam",  type=float, default=_bolt_default,
                    help=f"Max bolt-hole diameter mm (default {_bolt_default})")
    args = ap.parse_args()

    # Thread the CLI thresholds into the processing functions
    import functools
    _proc = functools.partial(
        _process_body_with_thresholds,
        jig_diam=args.jig_diam,
        bolt_diam=args.bolt_diam,
    )

    if args.stp:
        _proc(pathlib.Path(args.stp), dry_run=args.dry_run)
    elif args.hierarchy:
        _batch_hierarchy_with_thresholds(
            pathlib.Path(args.hierarchy),
            dry_run=args.dry_run,
            jig_diam=args.jig_diam,
            bolt_diam=args.bolt_diam,
        )
    elif args.batch_dir:
        root   = pathlib.Path(args.batch_dir)
        hfiles = sorted(root.rglob("hierarchy.json"))
        print(f"Found {len(hfiles)} hierarchy.json files under {root}")
        for hf in hfiles:
            _batch_hierarchy_with_thresholds(
                hf,
                dry_run=args.dry_run,
                jig_diam=args.jig_diam,
                bolt_diam=args.bolt_diam,
            )
    else:
        ap.print_help()
        sys.exit(1)


def _process_body_with_thresholds(stp_path, dry_run=False,
                                   jig_diam=JIG_DIAM_MM, bolt_diam=BOLT_DIAM_MM):
    """Thin wrapper around process_body that accepts explicit thresholds."""
    _orig_jig, _orig_bolt = JIG_DIAM_MM, BOLT_DIAM_MM
    import sys as _sys
    _this = _sys.modules[__name__]
    _this.JIG_DIAM_MM  = jig_diam
    _this.BOLT_DIAM_MM = bolt_diam
    try:
        return process_body(stp_path, dry_run=dry_run)
    finally:
        _this.JIG_DIAM_MM  = _orig_jig
        _this.BOLT_DIAM_MM = _orig_bolt


def _batch_hierarchy_with_thresholds(hierarchy_json, dry_run=False,
                                      jig_diam=JIG_DIAM_MM, bolt_diam=BOLT_DIAM_MM):
    import sys as _sys
    _this = _sys.modules[__name__]
    _orig_jig, _orig_bolt = _this.JIG_DIAM_MM, _this.BOLT_DIAM_MM
    _this.JIG_DIAM_MM  = jig_diam
    _this.BOLT_DIAM_MM = bolt_diam
    try:
        return batch_from_hierarchy(hierarchy_json, dry_run=dry_run)
    finally:
        _this.JIG_DIAM_MM  = _orig_jig
        _this.BOLT_DIAM_MM = _orig_bolt


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "_fill_one_worker":
        _fill_one_worker(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main()

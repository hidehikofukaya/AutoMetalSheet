"""Phase 0 of final_wireframe_theory.md: convert v4.4 typed wireframes into the
frozen sparse representation W = (P, T, E) with quantization, canonical order,
sequence codec sigma/delta, and a round-trip test.

Scope (per theory §1/§10): edge classes {outer_boundary, hole_boundary,
bend_line, crease(+feature_rim override)} are converted; surface_frame is out
of scope (fillet interiors are recovered from face assumptions A), seam is noise.

Pipeline per part:
  polylines -> welded chains per class -> recursive LINE/ARC fitting
  hole circles -> CIRCLE, or CIRCLE_C when matched to a joint (concentricity)
  -> quantize (b=14, coarse/fine) -> merge same (T,bin) -> canonical sort
  -> sigma(W) token sequence -> delta round-trip check -> fidelity Chamfer
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..wireflow.dataset import load_joints, part_id_from_stem
from .constants import BITS, E_ARITY, ETYPES, HI_BITS, SparseW, VTYPES  # noqa: F401

SCOPE_CLASSES = {"outer_boundary", "hole_boundary", "bend_line", "crease", "feature_rim"}
FIT_TOL_MM = 0.35          # max deviation polyline -> fitted primitive
WELD_MM = 0.05             # endpoint welding when chaining
DEGEN_K = 4                # sagitta < k*Delta -> LINE (theory def.2)
CIRCLE_JOINT_TOL_MM = 3.0  # hole-circle center to joint distance for CIRCLE_C
CIRCLE_AXIS_DOT = 0.85



# ---------------------------------------------------------------- fitting

def point_segment_dev(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    tt = np.clip((pts - a) @ ab / max(ab @ ab, 1e-12), 0.0, 1.0)
    return np.linalg.norm(pts - (a + tt[:, None] * ab), axis=1)


def fit_plane(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[2]
    return c, n, float(np.abs((pts - c) @ n).max())


def fit_circle(pts: np.ndarray):
    """Fit circle in the best plane. Returns (center, normal, radius, max_dev)."""
    c0, n, plane_dev = fit_plane(pts)
    e1 = np.cross(n, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(n, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    x, y = (pts - c0) @ e1, (pts - c0) @ e2
    A = np.c_[x, y, np.ones_like(x)]
    try:
        (D, E, F), *_ = np.linalg.lstsq(A, -(x**2 + y**2), rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = -D / 2, -E / 2
    r2 = cx * cx + cy * cy - F
    if r2 <= 1e-12:
        return None
    r = math.sqrt(r2)
    center = c0 + cx * e1 + cy * e2
    dev = float(np.abs(np.hypot(x - cx, y - cy) - r).max())
    return center, n, r, max(dev, plane_dev)


def arc_from_endpoints(center, normal, radius, p0, p1, probe):
    """Arc through p0->p1 on the fitted circle whose sweep contains `probe`.
    Returns (mid_point_at_parameter_midpoint, sweep_angle)."""
    e1 = p0 - center
    e1 = e1 - normal * (e1 @ normal)
    e1 /= max(np.linalg.norm(e1), 1e-12)
    e2 = np.cross(normal, e1)

    def ang(p):
        v = p - center
        return math.atan2(v @ e2, v @ e1)

    a1 = ang(p1) % (2 * math.pi)
    ap = ang(probe) % (2 * math.pi)
    if ap <= a1:            # ccw sweep 0 -> a1 contains probe
        sweep = a1
    else:                   # take the complementary (cw) arc
        sweep = a1 - 2 * math.pi
    amid = sweep / 2.0
    mid = center + radius * (math.cos(amid) * e1 + math.sin(amid) * e2)
    return mid, abs(sweep)


def arc_dev(pts, center, normal, radius) -> float:
    """Distance of points to the full circle (arc-span errors are caught by
    the probe-based sweep choice; adequate as a fitting gate)."""
    v = pts - center
    v_in = v - np.outer(v @ normal, normal)
    rho = np.linalg.norm(v_in, axis=1)
    axial = np.abs(v @ normal)
    return float(np.hypot(rho - radius, axial).max())


def fit_edge_primitive(pts: np.ndarray) -> tuple[dict, float]:
    """One B-rep edge -> ONE primitive, never split (KB 18).

    Measured on 60 parts / 1084 outer edges: every CATIA edge already fits a
    single LINE or ARC at ~0 residual, so the recursive refit was re-cutting
    primitives at places CATIA never put a vertex -- 62% of its corners had no
    CATIA vertex within 1mm, and those invented corners are the outline's
    corner jitter. Returns (segment, residual_mm); the residual is reported,
    not gated, so a genuine spline shows up in the stats instead of growing
    new corners.
    """
    dev_line = float(point_segment_dev(pts, pts[0], pts[-1]).max())
    if len(pts) < 3 or dev_line < FIT_TOL_MM:
        return {"kind": "LINE", "p0": pts[0], "p1": pts[-1]}, dev_line
    fc = fit_circle(pts)
    if fc is not None and fc[2] < 1e5:
        center, normal, r, _ = fc
        adev = float(arc_dev(pts, center, normal, r))
        if adev < dev_line:
            probe = pts[len(pts) // 2]
            mid, sweep = arc_from_endpoints(center, normal, r, pts[0], pts[-1], probe)
            return {"kind": "ARC", "p0": pts[0], "p1": pts[-1], "mid": mid,
                    "center": center, "normal": normal, "radius": r,
                    "sagitta": r * (1 - math.cos(sweep / 2))}, adev
    return {"kind": "LINE", "p0": pts[0], "p1": pts[-1]}, dev_line


def _prim_tangent(seg: dict, poly: np.ndarray, at_end: bool) -> np.ndarray:
    """Unit tangent of the fitted primitive at one endpoint, pointing along
    the p0 -> p1 travel direction. The primitive fits at ~0 residual, so this
    IS the raw geometry's tangent -- unlike a chord over a 2mm span, it does
    not read an arc's own curvature as a corner."""
    if seg["kind"] == "ARC":
        pt = seg["p1"] if at_end else seg["p0"]
        t = np.cross(seg["normal"], pt - seg["center"])
        n = np.linalg.norm(t)
        if n > 1e-9:
            t = t / n
            # the fitted normal's sign is arbitrary: orient by the polyline
            ref = (poly[-1] - poly[-2]) if at_end else (poly[1] - poly[0])
            return t if t @ ref >= 0 else -t
    d = seg["p1"] - seg["p0"]
    return d / max(np.linalg.norm(d), 1e-12)


def add_outline_primitives(W: SparseW, polys: list[np.ndarray], cls: str,
                           delta_est: float, stats: dict) -> list[np.ndarray]:
    """The CATIA decomposition as the teacher: one vertex per CATIA vertex,
    one primitive per edge, and the junction's tangent angle stored on the
    shared vertex (`jt`, degrees) so the representation can carry G1/G0.

    Computed BEFORE quantization -- the +-0.3mm bin snap would fold a tangent
    junction into a false corner (KB 18.4).

    Returns the polylines it could NOT take (closed single-edge loops), for
    the caller's chain-and-fit fallback.
    """
    fitted, leftover = [], []
    for poly in polys:
        length = float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())
        if length < WELD_MM:
            continue                       # extractor junk: a zero-length edge
        if np.linalg.norm(poly[0] - poly[-1]) < WELD_MM:
            leftover.append(poly)          # a closed edge has no junctions
            continue
        seg, dev = fit_edge_primitive(poly)
        fitted.append((seg, dev, poly))
    if not fitted:
        return leftover

    # weld shared endpoints by CLUSTER, not by grid cell: two copies of one
    # CATIA vertex can differ by 0.01mm and a rounded grid puts that straddle
    # in different cells, which broke the loop on the first part tried
    from scipy.spatial import cKDTree
    ends = np.concatenate([[s["p0"], s["p1"]] for s, _, _ in fitted])
    parent = list(range(len(ends)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in cKDTree(ends).query_pairs(WELD_MM):
        parent[find(i)] = find(j)
    # cluster id per fitted endpoint: fitted edge k owns rows 2k (p0), 2k+1 (p1)
    cid = [find(i) for i in range(len(ends))]

    # junction turn angle where exactly two edges meet: the angle between the
    # tangent arriving and the tangent leaving, 0 = tangent-continuous (G1)
    at_junction: dict = {}
    for k, (seg, _, poly) in enumerate(fitted):
        at_junction.setdefault(cid[2 * k], []).append(
            _prim_tangent(seg, poly, at_end=False))          # leaving
        at_junction.setdefault(cid[2 * k + 1], []).append(
            -_prim_tangent(seg, poly, at_end=True))          # arriving, flipped
    jt = {}
    for c, ts in at_junction.items():
        if len(ts) == 2:
            jt[c] = math.degrees(math.acos(np.clip(-(ts[0] @ ts[1]), -1.0, 1.0)))

    resid = []
    for k, (seg, dev, _) in enumerate(fitted):
        resid.append(dev)
        if seg["kind"] == "ARC" and seg["sagitta"] < DEGEN_K * delta_est:
            seg = {"kind": "LINE", "p0": seg["p0"], "p1": seg["p1"]}
        idx = []
        for c in (cid[2 * k], cid[2 * k + 1]):
            i = W.add_vertex(ends[c], "END")     # the cluster root's position
            if c in jt:
                W.vertices[i]["jt"] = round(jt[c], 2)
            idx.append(i)
        if seg["kind"] == "LINE":
            W.edges.append({"tau": "LINE", "refs": idx, "cls": cls})
            stats["n_line"] += 1
        else:
            im = W.add_vertex(seg["mid"], "MID")
            W.edges.append({"tau": "ARC", "refs": [idx[0], im, idx[1]], "cls": cls})
            stats["n_arc"] += 1
    stats["n_outline_edges"] = stats.get("n_outline_edges", 0) + len(fitted)
    stats["outline_resid_max_mm"] = max(stats.get("outline_resid_max_mm", 0.0),
                                        float(max(resid)))
    return leftover


def segment_polyline(pts: np.ndarray, tol: float, depth: int = 0) -> list[dict]:
    """Recursive LINE/ARC decomposition of an open chain."""
    if len(pts) < 3 or point_segment_dev(pts, pts[0], pts[-1]).max() < tol:
        return [{"kind": "LINE", "p0": pts[0], "p1": pts[-1]}]
    if depth < 24:
        fc = fit_circle(pts)
        if fc is not None:
            center, normal, r, dev = fc
            if dev < tol and r < 1e5:
                probe = pts[len(pts) // 2]
                mid, sweep = arc_from_endpoints(center, normal, r, pts[0], pts[-1], probe)
                chord = float(np.linalg.norm(pts[-1] - pts[0]))
                # guard (theory finding IV): near-closed or huge-sweep arcs are
                # branch-unstable under quantization -> split instead
                if (arc_dev(pts, center, normal, r) < tol and 0.05 < sweep < 5.0
                        and chord > max(2.0, 0.05 * r)):
                    return [{"kind": "ARC", "p0": pts[0], "p1": pts[-1], "mid": mid,
                             "sagitta": r * (1 - math.cos(sweep / 2))}]
    k = int(np.argmax(point_segment_dev(pts, pts[0], pts[-1])))
    k = min(max(k, 1), len(pts) - 2)
    return (segment_polyline(pts[: k + 1], tol, depth + 1)
            + segment_polyline(pts[k:], tol, depth + 1))


# ---------------------------------------------------------------- chaining

def build_chains(polylines: list[np.ndarray]) -> list[tuple[np.ndarray, bool]]:
    """Weld polyline endpoints and walk maximal chains. Returns (points, closed)."""
    def key(p):
        return (round(p[0] / WELD_MM), round(p[1] / WELD_MM), round(p[2] / WELD_MM))

    adjacency: dict = {}
    for idx, poly in enumerate(polylines):
        for end, k in ((0, key(poly[0])), (1, key(poly[-1]))):
            adjacency.setdefault(k, []).append((idx, end))

    used = [False] * len(polylines)
    chains = []

    def walk(start_idx: int, start_end: int) -> np.ndarray:
        pts = polylines[start_idx] if start_end == 0 else polylines[start_idx][::-1]
        used[start_idx] = True
        pts = [pts]
        while True:
            tail = key(pts[-1][-1])
            nxt = [(i, e) for (i, e) in adjacency.get(tail, []) if not used[i]]
            if len(nxt) != 1 or len(adjacency.get(tail, [])) > 2:
                break  # junction or dead end
            i, e = nxt[0]
            used[i] = True
            seg = polylines[i] if e == 0 else polylines[i][::-1]
            pts.append(seg[1:])
        return np.concatenate(pts)

    # open chains first (start at odd-degree/junction nodes)
    for k, ends in adjacency.items():
        if len(ends) != 2:
            for i, e in ends:
                if not used[i]:
                    chains.append((walk(i, e), False))
    # remaining loops
    for i in range(len(polylines)):
        if not used[i]:
            pts = walk(i, 0)
            chains.append((pts, np.linalg.norm(pts[0] - pts[-1]) < WELD_MM * 4))
    return chains


# ---------------------------------------------------------------- W assembly

def circle_canonical_points(center, normal, radius):
    ref = np.array([1.0, 0.0, 0.0])
    if abs(ref @ normal) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(normal, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    return [center + radius * (math.cos(a) * e1 + math.sin(a) * e2)
            for a in (0.0, 2 * math.pi / 3, 4 * math.pi / 3)]


def convert_part(wf_json: Path, joints: list[dict]) -> tuple[SparseW, dict]:
    data = json.loads(wf_json.read_text(encoding="utf-8"))
    ov_path = wf_json.with_name(wf_json.stem + ".overrides.json")
    overrides = json.loads(ov_path.read_text(encoding="utf-8")) if ov_path.exists() else {}

    by_class: dict[str, list[np.ndarray]] = {}
    for e in data["edges"]:
        cls = overrides.get(e.get("fingerprint"), e["type"])
        if cls not in SCOPE_CLASSES:
            continue
        cls = "bend_line" if cls in ("crease", "feature_rim") else cls
        poly = np.asarray(e["polyline"], dtype=np.float64)
        if len(poly) >= 2:
            by_class.setdefault(cls, []).append(poly)

    W = SparseW()
    fix_idx = []
    for j in joints:
        fix_idx.append(W.add_vertex(j["xyz"], "FIX", np.asarray(j["dir"], dtype=np.float64)))

    stats = {"n_chains": 0, "n_circle": 0, "n_circle_c": 0, "n_line": 0, "n_arc": 0,
             "fit_fail_chains": 0}
    delta_est = float(max(np.concatenate([p for ps in by_class.values() for p in ps]
                                         ).max(0) - np.concatenate(
        [p for ps in by_class.values() for p in ps]).min(0))) / (1 << BITS)

    for cls, polys in by_class.items():
        if cls == "outer_boundary":
            # KB 18: the CATIA edges ARE the primitives; chain-and-refit only
            # what this path cannot take (closed single-edge loops)
            polys = add_outline_primitives(W, polys, cls, delta_est, stats)
            if not polys:
                continue
        for pts, closed in build_chains(polys):
            stats["n_chains"] += 1
            if closed and len(pts) >= 8:
                fc = fit_circle(pts)
                chain_len = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
                # a genuine circle has chain length ~ its circumference; a
                # degenerate out-and-back loop of near-collinear points fits a
                # huge circle with tiny residual and must be rejected here
                if (fc is not None and fc[3] < FIT_TOL_MM and fc[2] < 1000.0
                        and 0.8 < chain_len / (2 * math.pi * fc[2]) < 1.25):
                    center, normal, r, _ = fc
                    matched = None
                    for fi, j in zip(fix_idx, joints):
                        jx = np.asarray(j["xyz"])
                        nd = np.asarray(j["dir"])
                        if (np.linalg.norm(jx - center) < max(CIRCLE_JOINT_TOL_MM, 0.3 * r)
                                and abs(nd @ normal) > CIRCLE_AXIS_DOT):
                            matched = fi
                            break
                    if matched is not None:
                        rv = W.add_vertex(center + r * _circle_dir(normal), "END")
                        W.edges.append({"tau": "CIRCLE_C", "refs": [matched, rv], "cls": cls})
                        stats["n_circle_c"] += 1
                    else:
                        c3 = circle_canonical_points(center, normal, r)
                        refs = [W.add_vertex(p, "END") for p in c3]
                        W.edges.append({"tau": "CIRCLE", "refs": refs, "cls": cls})
                        stats["n_circle"] += 1
                    continue
            # open chain (or non-circular loop): LINE/ARC decomposition
            for seg in segment_polyline(pts, FIT_TOL_MM):
                if seg["kind"] == "ARC" and seg["sagitta"] < DEGEN_K * delta_est:
                    seg = {"kind": "LINE", "p0": seg["p0"], "p1": seg["p1"]}
                i0 = W.add_vertex(seg["p0"], "END")
                i1 = W.add_vertex(seg["p1"], "END")
                if seg["kind"] == "LINE":
                    W.edges.append({"tau": "LINE", "refs": [i0, i1], "cls": cls})
                    stats["n_line"] += 1
                else:
                    im = W.add_vertex(seg["mid"], "MID")
                    W.edges.append({"tau": "ARC", "refs": [i0, im, i1], "cls": cls})
                    stats["n_arc"] += 1
    return W, stats


def _circle_dir(normal):
    ref = np.array([1.0, 0.0, 0.0])
    if abs(ref @ normal) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(normal, ref)
    return e1 / np.linalg.norm(e1)

"""P2 realizer: (theta, c | condition) -> sampled midsurface point cloud.

Single source of truth: imports PartMaker's pure-Python planner
(`plan_general_two_point`) so panel layout, feasibility checks and fold math are
exactly the generator's own. This module only *samples* the planned surface:

  base panels (sheared quads, fold-tangent trimmed)
  + bend fillets (quadratic-Bezier approximation of the arc, error << 1mm)
  + fastening holes (constructive: samples removed around p1/p2)
  + flange (root quarter-fillet + wall strip, backside direction)
  + bead (trapezoid profile swept along the centreline, end caps, base cutout)

An infeasible theta raises the planner's ValueError -> caller counts it
(constructive validity check on generated parameters).
"""
from __future__ import annotations

import math
import sys

import numpy as np

GEN_SRC = "C:/Users/hide2/IdeaBox/PartMaker/synthetic_generator/src"
if GEN_SRC not in sys.path:
    sys.path.insert(0, GEN_SRC)

from synthetic_generator.general_geometry import plan_general_two_point  # noqa: E402
from synthetic_generator.classify import FasteningPoint  # noqa: E402
from synthetic_generator.bead import BeadParams  # noqa: E402
from synthetic_generator.flange import (  # noqa: E402
    FLANGE_EDGE_INSET_MM, FLANGE_EXTENSION_MARGIN_MM)

STEP_MM = 2.0


def _pt(frame, run, width):
    return np.asarray(frame.origin) + run * np.asarray(frame.u) + width * np.asarray(frame.v)


def realize_points(spec: dict, theta: np.ndarray, cls: int,
                   flange_bits: tuple[int, int] = (1, 1),
                   bead_rise_sign: int = 1) -> np.ndarray:
    """spec: {'point1':{'position_xyz','normal_xyz'},'point2':..., 'thickness_mm',
    'hole_diameter_mm','min_bearing_radius_mm'}. Raises ValueError if infeasible."""
    half_width, bend_r, s1, s2, tilt_pert = (float(x) for x in theta[:5])
    flange = None
    bead = None
    if cls == 0:
        flange = {"height": float(theta[5]), "root_r": float(theta[6]),
                  "side": 1 if flange_bits[0] else -1,
                  "direction": 1 if flange_bits[1] else -1}
        ext = flange["root_r"] + FLANGE_EXTENSION_MARGIN_MM
        side_ext = (ext, 0.0) if flange["side"] < 0 else (0.0, ext)
    else:
        bead = BeadParams(
            depth_mm=float(theta[7]), top_width_mm=float(theta[8]),
            wall_angle_deg=float(theta[9]), ridge_radius_mm=float(theta[10]),
            corner_radius_mm=float(theta[11]))
        side_ext = (0.0, 0.0)

    p1 = FasteningPoint(tuple(spec["point1"]["position_xyz"]),
                        tuple(spec["point1"]["normal_xyz"]))
    p2 = FasteningPoint(tuple(spec["point2"]["position_xyz"]),
                        tuple(spec["point2"]["normal_xyz"]))
    margin = float(spec["min_bearing_radius_mm"])
    plan = plan_general_two_point(
        p1, p2, min_bearing_radius_mm=margin, half_width_mm=half_width,
        bend_radius_mm=bend_r, fold1_slack_mm=s1, fold2_slack_mm=s2,
        fold1_tilt_perturbation_rad=tilt_pert, side_extension_mm=side_ext)

    frames = plan.panel_frames
    tilts = plan.fold_tilts
    tangents = plan.fold_tangents
    ext_neg, ext_pos = side_ext
    hw = half_width
    w_lo, w_hi = -(hw + ext_neg), hw + ext_pos
    if flange is not None:
        # base stops where the root fillet leaves the surface
        edge_off = hw + flange["root_r"] + FLANGE_EXTENSION_MARGIN_MM - FLANGE_EDGE_INSET_MM
        root_start = edge_off - flange["root_r"]
        if flange["side"] > 0:
            w_hi = root_start
        else:
            w_lo = -root_start

    # bead footprint cutout (along the centreline span)
    bead_span = None
    if bead is not None:
        total = sum(f.far_run_mm - f.near_run_mm for f in frames)
        inset = 2.0 * margin
        bead_span = (inset, total - inset)
        half_foot = bead.half_footprint_mm

    pts: list[np.ndarray] = []
    run_offset = 0.0  # cumulative run coordinate across the chain
    centreline: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    for i, frame in enumerate(frames):
        near_cut, far_cut = tangents[i]
        near_tilt, far_tilt = tilts[i]
        u = np.asarray(frame.u)
        v = np.asarray(frame.v)
        n = np.cross(u, v)
        for w in np.arange(w_lo, w_hi + 1e-6, STEP_MM):
            lo = frame.near_run_mm + w * math.tan(near_tilt) + near_cut
            hi = frame.far_run_mm + w * math.tan(far_tilt) - far_cut
            if hi <= lo:
                continue
            for r in np.arange(lo, hi + 1e-6, STEP_MM):
                chain_r = run_offset + (r - frame.near_run_mm)
                if (bead_span is not None and abs(w) < half_foot
                        and bead_span[0] < chain_r < bead_span[1]):
                    continue
                pts.append(_pt(frame, r, w))
        # centreline samples (width 0) for bead sweep
        lo0 = frame.near_run_mm + near_cut
        hi0 = frame.far_run_mm - far_cut
        for r in np.arange(lo0, hi0 + 1e-6, STEP_MM):
            centreline.append((run_offset + (r - frame.near_run_mm),
                               _pt(frame, r, 0.0), v.copy(), n.copy()))
        run_offset += frame.far_run_mm - frame.near_run_mm

    # ---- bend fillets: quadratic Bezier between the tangent-trim edges ----
    for i in range(len(frames) - 1):
        fa, fb = frames[i], frames[i + 1]
        cut_a = tangents[i][1]
        cut_b = tangents[i + 1][0]
        tilt = tilts[i][1]
        for w in np.arange(w_lo, w_hi + 1e-6, STEP_MM):
            a = _pt(fa, fa.far_run_mm + w * math.tan(tilt) - cut_a, w)
            c = _pt(fa, fa.far_run_mm + w * math.tan(tilt), w)
            b = _pt(fb, fb.near_run_mm + w * math.tan(tilt) + cut_b, w)
            for t in np.linspace(0.15, 0.85, 5):
                pts.append((1 - t) ** 2 * a + 2 * t * (1 - t) * c + t ** 2 * b)

    # ---- flange: root quarter-fillet + wall strip ----
    if flange is not None:
        s = flange["side"]
        d = flange["direction"]
        rr = flange["root_r"]
        h_wall = max(flange["height"] - rr, 0.0)
        for i, frame in enumerate(frames):
            near_cut, far_cut = tangents[i]
            near_tilt, far_tilt = tilts[i]
            u = np.asarray(frame.u)
            v = np.asarray(frame.v)
            n = np.cross(u, v) * d
            lo = frame.near_run_mm + s * (edge_off) * math.tan(near_tilt) + near_cut
            hi = frame.far_run_mm + s * (edge_off) * math.tan(far_tilt) - far_cut
            for r in np.arange(lo, hi + 1e-6, STEP_MM):
                base = _pt(frame, r, s * (edge_off - rr))
                centre = base + n * rr
                for phi in np.linspace(0.0, math.pi / 2, 5):
                    pts.append(centre - n * rr * math.cos(phi)
                               + s * v * rr * math.sin(phi))
                top = base + s * v * rr + n * rr
                for z in np.arange(STEP_MM, h_wall + 1e-6, STEP_MM):
                    pts.append(top + n * z)

    # ---- bead: trapezoid profile swept along the centreline ----
    if bead is not None and centreline:
        wall_run = bead.wall_run_mm
        for chain_r, p0, v, n in centreline:
            lo_span, hi_span = bead_span
            # end caps: scale depth down linearly over wall_run outside the span
            if chain_r < lo_span - wall_run or chain_r > hi_span + wall_run:
                continue
            scale = 1.0
            if chain_r < lo_span:
                scale = 1.0 - (lo_span - chain_r) / wall_run
            elif chain_r > hi_span:
                scale = 1.0 - (chain_r - hi_span) / wall_run
            depth = bead.depth_mm * scale
            rise = bead_rise_sign * n
            half_top = bead.top_width_mm / 2.0
            for f in np.linspace(0.0, 1.0, 4)[1:]:
                for sgn in (-1.0, 1.0):
                    w = sgn * (half_foot - f * wall_run)
                    pts.append(p0 + v * w + rise * depth * f)
            for w in np.arange(-half_top, half_top + 1e-6, STEP_MM):
                pts.append(p0 + v * w + rise * depth)

    out = np.asarray(pts)
    # ---- fastening holes (constructive removal) ----
    hole_r = float(spec["hole_diameter_mm"]) / 2.0
    for key in ("point1", "point2"):
        c = np.asarray(spec[key]["position_xyz"])
        out = out[np.linalg.norm(out - c, axis=1) > hole_r]
    return out

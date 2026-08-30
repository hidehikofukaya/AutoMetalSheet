"""Rationality: can every feature of a generated wire be explained?

Three tiers, in order; a failure at a higher tier makes the lower ones moot.

    1 manufacturability   can it be made          pass / fail
    2 parsimony           nothing without reason  relative to the teacher
    3 function            does it do its job      equivalence, not similarity

Nothing here depends on the part class, the number of fasteners or the number
of features. The seat is derived from each fastener's own required bearing
radius, the bend limit from the sheet thickness, and parsimony from the turning
a closed planar loop must have (360 degrees) against what it does have.

None of it enters the model. It is the definition of the goal, used to evaluate
and to choose among draws.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .codec import circle_through
from .frame import realize_frame

RT_MIN = 2.0            # forming limit: the teacher satisfies it on 99.5% of faces
AMBIG = (3.0, 15.0)     # a junction turning this much is neither tangent nor a corner


def seats(part):
    """Each fastener's centre, axis and required bearing radius, in mm."""
    from .meshgen import fix_points_mm
    from .sidecar import SPEC_KEYS, SPEC_SCALE, load_spec

    P, A = fix_points_mm(part)
    sp = load_spec(part)
    r = 10.0
    t = 1.5
    if sp is not None:
        i = SPEC_KEYS.index("min_bearing_radius_mm")
        r = float(sp[i] / SPEC_SCALE[i]) or r
        j = SPEC_KEYS.index("thickness_mm")
        t = float(sp[j] / SPEC_SCALE[j]) or t
    return P, A, r, t


def score(x, frame, seat_pts, seat_r, t_mm):
    """All tiers for one frame tensor. Distances in mm."""
    from .meshgen import to_frame

    u = frame[2]
    live = x[:, 7] < 0.5
    P, B, A = x[live, 0:3], x[live, 3:6], x[live, 6] > 0.5
    n = len(P)
    if n < 4:
        return None
    d = np.roll(P, -1, 0) - P
    L = np.linalg.norm(d, axis=1) * u
    T = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    turn = np.degrees(np.arccos(np.clip(
        np.einsum("ij,ij->i", np.roll(T, 1, 0), T), -1, 1)))
    rt = []
    for i in range(n):
        if A[i]:
            a, b = P[i], P[(i + 1) % n]
            cc = circle_through(a, 0.5 * (a + b) + B[i], b)
            if cc:
                rt.append(cc[2] * u / t_mm)
    poly = realize_frame(x, 60)
    if len(poly) < 10:
        return None
    sf, _ = to_frame(seat_pts, np.zeros_like(seat_pts), frame)
    clear = u * cKDTree(poly).query(sf)[0]
    return {
        # tier 1
        "sliver": float(np.mean(L < t_mm)),
        "sharp": float(np.mean(np.array(rt) < RT_MIN)) if rt else 0.0,
        # tier 2
        "excess_turn": float(turn.sum() - 360.0),
        "ambiguous": float(np.mean((turn > AMBIG[0]) & (turn < AMBIG[1]))),
        "edges": int(n),
        # tier 3
        "seat": float((clear / seat_r).min()),
    }


def passes_tier1(s):
    return s is not None and s["sliver"] == 0.0 and s["sharp"] == 0.0


def rank(frames, frame, seat_pts, seat_r, t_mm, teacher=None):
    """Order draws by rationality: tier 1 first, then seat, then parsimony.

    Returns indices best-first, with the scores. Replaces the medoid, which
    ranked by distance to the other draws and was blind to all of this.
    """
    sc = [score(f, frame, seat_pts, seat_r, t_mm) for f in frames]
    ref_turn = teacher["excess_turn"] if teacher else 0.0

    def key(i):
        s = sc[i]
        if s is None:
            return (2, 0.0, 0.0)
        t1 = 0 if passes_tier1(s) else 1
        seat_pen = max(0.0, 1.0 - s["seat"])            # 0 once the seat is clear
        pars = abs(s["excess_turn"] - ref_turn)
        return (t1, seat_pen, pars)

    order = sorted(range(len(frames)), key=key)
    return order, sc


def seat_project(x, frame, seat_pts, seat_axes, seat_r, margin=1.0):
    """Push every corner out of any fastener seat it intrudes on.

    A seat is a disc of the required bearing radius, normal to the fastener
    axis. A corner inside that disc (radially, in the seat's plane) is moved
    radially to its rim. This is the same kind of guarantee as anchor freezing
    (FIX error 0.00mm): the constraint is input-derived and holds for any number
    of fasteners, so honouring it exactly is not a rule about shape.

    Applied at sampling time it acts as a constraint on the path, not a term in
    a loss -- nothing is added to what the model is asked to learn.
    """
    from .meshgen import to_frame

    x = x.copy()
    u = frame[2]
    live = np.flatnonzero(x[:, 7] < 0.5)
    if not len(live):
        return x
    P = x[live, 0:3]
    sf, af = to_frame(seat_pts, seat_axes, frame)
    r = (seat_r * margin) / u
    for c, a in zip(sf, af):
        a = a / max(np.linalg.norm(a), 1e-12)
        w = P - c
        h = w @ a
        radial = w - np.outer(h, a)
        dist = np.linalg.norm(radial, axis=1)
        inside = (dist < r) & (np.abs(h) < r)     # within the seat's slab too
        for i in np.flatnonzero(inside):
            dirn = radial[i] / dist[i] if dist[i] > 1e-9 else np.array([1.0, 0, 0])
            P[i] = c + h[i] * a + dirn * r
    x[live, 0:3] = P
    return x


def bend_score(curves, outline, u, t_mm, attach_t: float = 2.0):
    """The same question for bend lines: is every feature explainable?

    `curves`: polylines in frame units (what tokens_to_curves returns);
    `outline`: the outline polyline in the same units.

      excess    turning a curve has beyond the net rotation of its tangent
                (a straight fold turns 0; a smooth bead ridge turns exactly
                its end-to-end angle; zigzag turns without going anywhere).
                Closed curves owe 360. Degrees, summed over the part.
      dangling  open ends that reach neither the outline nor another curve
                within `attach_t` thicknesses -- a bend that ends in the
                middle of a panel has no reason to exist.
      curves    how many came out. An observation, not a target.
    """
    if not curves:
        return None
    tol = attach_t * t_mm / u
    # junctions are tested against the curves as LINES, not their vertices:
    # corner tokens leave a straight fold with two vertices 100mm apart
    from .tokens import _arc, _resample
    dense = [_resample(c, max(int(_arc(c)[-1] / (0.5 * tol)) + 2, 2)) for c in curves]
    owner = np.concatenate([np.full(len(d), i) for i, d in enumerate(dense)])
    allpts = np.concatenate(dense)
    tree = cKDTree(allpts)
    excess = 0.0
    ends_open, ends_dangling = 0, 0
    for ci, c in enumerate(curves):
        d = np.diff(c, axis=0)
        L = np.linalg.norm(d, axis=1)
        T = d[L > 1e-9] / L[L > 1e-9, None]
        if len(T) < 2:
            continue
        turn = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", T[:-1], T[1:]), -1, 1)))
        closed = np.linalg.norm(c[0] - c[-1]) < 1e-6
        if closed:
            excess += max(turn.sum() - 360.0, 0.0)
        else:
            net = np.degrees(np.arccos(np.clip(T[0] @ T[-1], -1, 1)))
            excess += max(turn.sum() - net, 0.0)
            for e in (c[0], c[-1]):
                ends_open += 1
                near_out = cKDTree(outline).query(e)[0] < tol
                near_curve = any(owner[j] != ci for j in tree.query_ball_point(e, tol))
                if not (near_out or near_curve):
                    ends_dangling += 1
    return {"excess": float(excess),
            "dangling": ends_dangling / ends_open if ends_open else 0.0,
            "curves": len(curves)}


def bend_rank(curve_sets, outline, u, t_mm):
    """Order draws: fewest dangling ends, then least excess turning."""
    sc = [bend_score(c, outline, u, t_mm) for c in curve_sets]
    order = sorted(range(len(sc)), key=lambda i: (
        (1, 0.0, 0.0) if sc[i] is None else (0, sc[i]["dangling"], sc[i]["excess"])))
    return order, sc


def demo():
    """Teacher must pass tier 1 and sit at the seat; a squeezed copy must not."""
    import pathlib

    from .dataset_curve import load_curve_parts
    from .frame import frame_target
    from .meshgen import fastener_frame

    R = pathlib.Path(__file__).resolve().parents[4]
    have = {p.stem for p in (R / "runs" / "mesh_synth" / "parts").glob("*.npz")}
    parts = [p for p in load_curve_parts(R / "runs" / "wtok_synth")
             if p.name in have][:40]
    seat_t, seat_s, fixed, t1 = [], [], [], []
    for p in parts:
        fr = fastener_frame(p)
        x = frame_target(p, fr)
        if x is None:
            continue
        P, A, r, t = seats(p)
        s = score(x, fr, P, r, t)
        if s is None:
            continue
        t1.append(passes_tier1(s))       # the teacher itself has slivers: a floor, not a law
        seat_t.append(s["seat"])
        y = x.copy()
        y[:, 0:3] *= 0.85                              # squeeze toward the frame origin
        s2 = score(y, fr, P, r, t)
        seat_s.append(s2["seat"])
        z = seat_project(y, fr, P, A, r)
        fixed.append(score(z, fr, P, r, t)["seat"])
    print(f"{len(seat_t)} parts")
    print(f"  teacher passes tier 1     {100*np.mean(t1):.0f}%")
    print(f"  teacher seat ratio        median {np.median(seat_t):.3f}")
    print(f"  squeezed 0.85             median {np.median(seat_s):.3f}")
    print(f"  after seat_project        median {np.median(fixed):.3f}  (must be >= 1)")
    assert np.median(seat_t) > 0.95 and np.median(fixed) >= 0.99
    print("ok")


if __name__ == "__main__":
    demo()

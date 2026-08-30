"""Bend features anchored to the outline, so a dangling end cannot be expressed.

The previous bend representation generated 24 free line segments, and 90.6% of
what it produced were bead ridge fragments. A part has exactly TWO folds plus
one bead or one flange, and the sidecar says so exactly. Generating the
tessellated consequence instead of the design intent is why the folds came out
short and fragmented.

Measured on 150 parts against the PartMaker feature ground truth:

    bead centreline ends on the outline      100.0% within 2mm  (110/110 open)
    flange root_curve ends on the outline    100.0% within 2mm
    fold tangent-line ends on the outline     86.7% within 2mm
                          ... on the flange   13.3%
                          ... nowhere          0.0%

Every feature end lands on the outline or on another feature, never in free
space. Encoding an end AS a position along the outline makes that a property of
the representation rather than something the model has to learn -- the same move
that made a closed loop free (ordered ring) and a straight line free
(parametric edges).
"""
from __future__ import annotations

import numpy as np

KINDS = ("fold", "bead", "flange")
FEAT_SLOTS = 6          # 2 folds + 1 bead-or-flange, with headroom
N_CTRL = 2              # interior control points; see the table below
FEAT_CH = 4 + 4 + 3 * N_CTRL + 5        # = 19
# 0:3 kind one-hot | 3 unused | 4:6 anchor A (cos,sin of s) | 6:8 anchor B
# 8:8+3*N_CTRL interior control points, as offsets from the chord
# then 5 parameters, read according to kind
A0, A1, CTRL = 4, 6, 8
PARAM = CTRL + 3 * N_CTRL

# Error of the fitted curve against the true feature curve, 60 curves. The
# anchors are exact either way; this is only the shape between them, and the
# task's own floor is 14.9mm.
#
#     interior points   channels   error
#     1                 3          6.90mm
#     2                 6          3.94mm     <- adopted
#     3                 9          3.65mm
#     4                 12         2.40mm


def arc_params(loop):
    """Cumulative arc length of a closed loop, normalised to [0,1)."""
    q = np.concatenate([loop, loop[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
    return s[:-1] / max(s[-1], 1e-9)


def to_param(loop, pt):
    """Where along the loop a point sits, as a fraction in [0,1)."""
    s = arc_params(loop)
    return float(s[int(np.argmin(np.linalg.norm(loop - pt, axis=1)))])


def from_param(loop, u):
    """The point at fraction u along the loop. Always ON the loop, by construction."""
    s = np.concatenate([arc_params(loop), [1.0]])
    q = np.concatenate([loop, loop[:1]])
    u = float(u) % 1.0
    return np.array([np.interp(u, s, q[:, d]) for d in range(3)])


def _cs(u):
    a = 2 * np.pi * float(u)
    return np.array([np.cos(a), np.sin(a)])


def _un(c, s):
    return float(np.arctan2(s, c) / (2 * np.pi)) % 1.0


def _bezier(ctrl, n=96):
    """De Casteljau over any number of control points."""
    t = np.linspace(0, 1, n)[:, None]
    P = [np.repeat(c[None], n, 0) for c in ctrl]
    while len(P) > 1:
        P = [(1 - t) * P[i] + t * P[i + 1] for i in range(len(P) - 1)]
    return P[0]


def _fit_ctrl(ref, a, b, k=N_CTRL):
    """Least-squares interior control points for a Bezier pinned at a and b."""
    from math import comb
    n, m = len(ref), k + 2
    t = np.linspace(0, 1, n)
    B = np.stack([comb(m - 1, i) * t ** i * (1 - t) ** (m - 1 - i)
                  for i in range(m)], 1)
    rhs = ref - B[:, [0]] * a - B[:, [-1]] * b
    return np.linalg.lstsq(B[:, 1:-1], rhs, rcond=None)[0]


def feature_target(part, frame, loop, slots: int = FEAT_SLOTS):
    """(slots, FEAT_CH) from the PartMaker sidecar, anchored to `loop`.

    `loop` is the outline in the same frame -- the true one at training time,
    the generated one at inference. The anchors are fractions along it, so a
    feature follows whatever outline it is given instead of floating free of it.
    """
    from .meshgen import to_frame
    from .sidecar import load_features

    x = load_features(part)
    if x is None:
        return None
    out = np.zeros((slots, FEAT_CH))
    out[:, 3] = 1.0                                    # unused
    state = {"at": 0}

    def put(kind, curve, params):
        at = state["at"]
        if at >= slots:
            return
        c = np.asarray(curve, float)
        p, _ = to_frame(c, np.zeros_like(c), frame)
        u0, u1 = to_param(loop, p[0]), to_param(loop, p[-1])
        out[at, KINDS.index(kind)] = 1.0
        out[at, 3] = 0.0
        out[at, A0:A0 + 2] = _cs(u0)
        out[at, A1:A1 + 2] = _cs(u1)
        # the curve's shape between its two anchors, as control-point offsets
        # from the chord -- zero for a straight feature
        A, B = from_param(loop, u0), from_param(loop, u1)
        c = _fit_ctrl(p if len(p) > 2 else np.stack([p[0], 0.5 * (p[0] + p[-1]),
                                                     p[-1]]), A, B)
        base = np.stack([A + (B - A) * (i + 1) / (N_CTRL + 1)
                         for i in range(N_CTRL)])
        out[at, CTRL:CTRL + 3 * N_CTRL] = (c - base).ravel()
        out[at, PARAM:PARAM + len(params)] = params
        state["at"] = at + 1

    for f in (x.get("folds") or [])[:2]:
        put("fold", f["sharp_line"],
            [f["angle_deg"] / 180.0, f["radius_mm"] / 50.0, f["tilt_deg"] / 10.0, 0, 0])
    b = x.get("bead")
    if b and b.get("centreline"):
        put("bead", b["centreline"],
            [b.get("depth_mm", 0) / 50.0, b.get("top_width_mm", 0) / 50.0,
             b.get("wall_angle_deg", 0) / 90.0, b.get("ridge_radius_mm", 0) / 50.0,
             b.get("corner_radius_mm", 0) / 50.0])
    fl = x.get("flange")
    if fl and fl.get("root_curve"):
        put("flange", fl["root_curve"],
            [fl.get("height_mm", 0) / 50.0, fl.get("root_radius_mm", 0) / 50.0,
             0, 0, 0])
    return out if state["at"] else None


def realize_features(x, loop, per=48):
    """Back to curves. Every curve starts and ends ON `loop` -- there is no other option.

    Returns [(kind, polyline, params), ...]. A dangling end is not something this
    can emit: both ends are looked up on the loop by their fraction.
    """
    out = []
    for r in x:
        if r[3] >= 0.5:
            continue
        k = KINDS[int(np.argmax(r[:3]))]
        a = from_param(loop, _un(r[A0], r[A0 + 1]))
        b = from_param(loop, _un(r[A1], r[A1 + 1]))
        base = np.stack([a + (b - a) * (i + 1) / (N_CTRL + 1)
                         for i in range(N_CTRL)])
        c = base + r[CTRL:CTRL + 3 * N_CTRL].reshape(N_CTRL, 3)
        poly = _bezier([a] + list(c) + [b], per)
        out.append((k, poly, r[PARAM:PARAM + 5].copy()))
    return out


def demo():
    """Anchors must land on the loop exactly, and the curve must track the truth."""
    import collections
    import pathlib

    from scipy.spatial import cKDTree

    from .dataset_curve import load_curve_parts
    from .frame import frame_target, realize_frame
    from .meshgen import fastener_frame, to_frame
    from .sidecar import load_features

    R = pathlib.Path(__file__).resolve().parents[4]
    parts = load_curve_parts(R / "runs" / "wtok_synth")[:40]
    anch, err, kinds = [], [], []
    for p in parts:
        fr = fastener_frame(p)
        ft = frame_target(p, fr)
        if ft is None:
            continue
        loop = realize_frame(ft, per_edge=60)
        x = feature_target(p, fr, loop)
        sc = load_features(p)
        if x is None or sc is None:
            continue
        got = realize_features(x, loop)
        kinds.append(tuple(k for k, _, _ in got))
        lt = cKDTree(loop)
        for _k, poly, _q in got:
            anch += [fr[2] * float(lt.query(poly[0])[0]),
                     fr[2] * float(lt.query(poly[-1])[0])]
        b = sc.get("bead")
        if b and b.get("centreline"):
            c = np.asarray(b["centreline"], float)
            ref, _ = to_frame(c, np.zeros_like(c), fr)
            for k, poly, _q in got:
                if k == "bead":
                    err.append(fr[2] * float(cKDTree(poly).query(ref)[0].mean()))
    print(f"{len(kinds)} parts, feature sets: "
          f"{dict(collections.Counter(kinds).most_common(3))}")
    print(f"anchor -> outline distance: max {max(anch):.6f}mm  (must be 0)")
    if err:
        print(f"bead curve vs true centreline: median {np.median(err):.1f}mm")
    assert max(anch) < 1e-6, "an anchor left the outline"
    print("ok")


if __name__ == "__main__":
    demo()

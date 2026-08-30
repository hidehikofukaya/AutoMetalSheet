"""The outline as a parametric wireframe: corners and edge types, not samples.

Sampled points have to LEARN to lie on a line, and the bend experiment showed
they do not: a plain cloud of bend points came out a diffuse smear, and the
32x20 slot scaffolding only bought the line-ness back by brute force. An edge
cannot smear. A line is two corners; an arc is two corners and a bulge. The
one-dimensional structure is a property of the representation instead of
something the model has to discover.

Measured on the 2300-part set, the outline is far smaller than its sampled
form:

    edges per part      median 18, p90 21, max 26      (vs 300 sampled points)
    corners per part    median 26, p90 35, max 43
    edge types          LINE 48.3%, ARC 51.7%, no splines at all

and it is exactly one closed loop on every part. So edge i runs from corner i
to corner i+1 and the connectivity never has to be generated -- which is what
made the earlier autoregressive route fail, with its counts, references and
stop tokens.
"""
from __future__ import annotations

import numpy as np

EDGE_SLOTS = 32         # max measured 26
FRAME_CH = 9            # corner 3, sagitta 1, is_arc 1, unused 1, normal 3
SAG, ARC_F, UNUSED_F = 3, 4, 5      # channel indices after the corner
NRM_WIN = 1             # corners each side of an edge used to fit its plane
NRM_TOL = 0.06          # third singular value over the first; above it, not flat


def outline_edges(part):
    """The outline walked edge by edge: [(corner, bulge, is_arc), ...].

    `bulge` is the arc's midpoint measured FROM THE CHORD MIDPOINT, not in
    absolute coordinates. It is then zero for a straight edge and small for a
    gentle one, so the channel carries only the departure from straightness
    instead of re-encoding where the part is.
    """
    from .codec import bin_center, realize_edge
    from .staged import _edge_ends
    from .train_curve import realized_q

    q = realized_q(part, part.vertices, part.edges)
    adj = {}
    for e in part.edges:
        if e.get("cls") != "outer_boundary":
            continue
        ends = _edge_ends(e)
        if ends is None:
            continue
        mid = None
        if e["tau"] == "ARC":
            mid = bin_center(q, q["vertices"][e["refs"][1]])
        adj.setdefault(ends[0], []).append((ends[1], mid, e["tau"] == "ARC"))
        adj.setdefault(ends[1], []).append((ends[0], mid, e["tau"] == "ARC"))
    if not adj:
        return None

    pos = {k: bin_center(q, q["vertices"][k]) for k in adj}
    start = min(adj)
    cur, prev, out = start, None, []
    for _ in range(len(adj) + 1):
        nxt = [(v, m, a) for v, m, a in adj[cur] if v != prev]
        if not nxt:
            break
        v, m, a = nxt[0]
        out.append((pos[cur], m, a))
        prev, cur = cur, v
        if cur == start:
            break
    if len(out) < 3 or cur != start:          # must come back to where it began
        return None
    return out


def frame_target(part, frame, slots: int = EDGE_SLOTS):
    """(slots, FRAME_CH) in the fastener frame, canonicalised, zero-padded."""
    from .meshgen import to_frame

    ed = outline_edges(part)
    if ed is None or len(ed) > slots:
        return None
    corners = np.stack([c for c, _, _ in ed])
    is_arc = np.array([a for _, _, a in ed], float)
    mids = np.stack([m if m is not None else c for c, m, a in ed])

    p, _ = to_frame(corners, np.zeros_like(corners), frame)
    mid, _ = to_frame(mids, np.zeros_like(mids), frame)
    p, mid, is_arc = _canonicalise(p, mid, is_arc)

    chord = 0.5 * (p + np.roll(p, -1, 0))     # edge i runs corner i -> i+1
    bulge = (mid - chord) * is_arc[:, None]
    out = loop_outward(p)
    sag = np.einsum("ij,ij->i", bulge, out)   # signed, along the outward normal

    x = np.zeros((slots, FRAME_CH))
    n = len(p)
    x[:n, 0:3] = p
    x[:n, SAG] = sag
    x[:n, ARC_F] = is_arc
    x[n:, UNUSED_F] = 1.0                     # unused slots
    x[:n, 6:9] = edge_normals(p)
    return x


def edge_normals(P, win: int = NRM_WIN, tol: float = NRM_TOL):
    """The normal of the panel each edge bounds; a zero vector where there is none.

    A panel is flat, so its bounding edges are coplanar -- and in the true data
    that is exact: every corner lies on a plane, with a 0.16mm residual that is
    the quantisation. Generated corners sit 0.94mm off, which is a panel that
    should be flat coming out twisted. Carrying the normal per EDGE rather than
    as a list of planes is what makes it scale: a panel is a run of edges sharing
    a normal, so the plane count is emergent and nothing caps it. Measured on the
    true outlines this window defines a normal for 100% of edges, finds 4
    distinct ones per part (max 12), and puts 47% of edges on the largest panel.

    A wider window was tried first and covered only 44% of edges: on an 18-edge
    outline it spans a third of the loop and straddles folds.
    """
    P = np.asarray(P, float)
    n = len(P)
    N = np.zeros((n, 3))
    if n < win * 2 + 2:
        return N
    for i in range(n):
        idx = [(i + d) % n for d in range(-win, win + 2)]
        w = P[idx] - P[idx].mean(0)
        _, s, V = np.linalg.svd(w)
        if s[0] < 1e-12 or s[2] / s[0] > tol:
            continue                      # straddles a fold: no single normal
        v = V[-1]
        N[i] = -v if v[int(np.argmax(np.abs(v)))] < 0 else v   # undirected
    return N


def loop_outward(P):
    """In-plane outward direction for each edge of a closed corner loop.

    An arc on an outline bows across the loop, not out of its plane: measured
    over 1764 true arcs the bulge's component along the loop normal is 0.337
    while 83% of it points outward. Carrying only a SIGNED SAGITTA along this
    direction makes an out-of-plane bulge inexpressible and leaves the model one
    number instead of a free 3-vector -- of which it got the direction wrong on
    23% of edges.

    The centre was considered instead. It is the natural CAD parameter but it
    diverges: a gentle arc has a huge radius, and the measured centre reaches
    2116mm on a part a tenth that size, while the sagitta stays at 1.1mm median
    and 22.8mm worst. The centre is recovered from the sagitta by
    `codec.circle_through`, so nothing is lost downstream.
    """
    P = np.asarray(P, float)
    c = P.mean(0)
    n = np.linalg.svd(P - c)[2][-1]
    t = np.roll(P, -1, 0) - P
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    o = np.cross(np.tile(n, (len(P), 1)), t)
    no = np.linalg.norm(o, axis=1, keepdims=True)
    o = np.divide(o, np.maximum(no, 1e-12))
    mid = 0.5 * (P + np.roll(P, -1, 0))
    s = np.sign(np.einsum("ij,ij->i", o, mid - c))
    s[s == 0] = 1.0
    return o * s[:, None]


def _canonicalise(p, mid, is_arc):
    """Same rule as the sampled ring: a fixed start corner and turn direction.

    A closed loop has no natural first edge, and the traversal picks the lowest
    vertex id, which lands somewhere different on every part. Slot 0 has to mean
    the same thing everywhere or the model fits an arbitrary rotation per part
    on top of the shape.
    """
    area = np.cross(p - p.mean(0), np.roll(p, -1, 0) - p.mean(0)).sum(0)
    if area[2] < 0:
        # reversing a loop of EDGES shifts them: edge i of the reversed loop
        # starts at what was corner i+1, so the arrays roll by one after the
        # flip. Getting this wrong pairs each corner with the wrong bulge.
        p = p[::-1].copy()
        mid = np.roll(mid[::-1].copy(), -1, 0)
        is_arc = np.roll(is_arc[::-1].copy(), -1, 0)
    s = int(np.argmax(p[:, 0]))
    return np.roll(p, -s, 0), np.roll(mid, -s, 0), np.roll(is_arc, -s, 0)


def realize_frame(x, per_edge: int = 24, arc_thresh: float = 0.5):
    """Slots back to a polyline, for drawing and for measuring against the truth.

    An arc is drawn as the circle through its two corners and its bulged
    midpoint -- the same construction `codec.realize_edge` uses -- so a frame
    that round-trips here is a frame the existing CAD writer can consume.

    Points come out evenly spaced ALONG THE LOOP, not evenly per edge. Placing
    the same count on every edge makes a short edge dense and a long one sparse,
    and anything that reads the result as a point cloud then sees the density
    jump as a break: a square that is closed by construction measured 54%
    endpoints that way, and 2% once respaced.
    """
    from .codec import circle_through

    live = x[:, UNUSED_F] < 0.5
    p = x[live, 0:3]
    if len(p) < 3:
        return np.zeros((0, 3))
    is_arc = x[live, ARC_F] > arc_thresh
    nxt = np.roll(p, -1, 0)
    mid = 0.5 * (p + nxt) + x[live, SAG][:, None] * loop_outward(p)

    out = []
    for i in range(len(p)):
        if not is_arc[i]:
            out.append(np.linspace(p[i], nxt[i], per_edge, endpoint=False))
            continue
        cc = circle_through(p[i], mid[i], nxt[i])
        if cc is None:
            out.append(np.linspace(p[i], nxt[i], per_edge, endpoint=False))
            continue
        c, n, r = cc
        e1 = (p[i] - c) / max(np.linalg.norm(p[i] - c), 1e-12)
        e2 = np.cross(n, e1)
        a_m = np.arctan2((mid[i] - c) @ e2, (mid[i] - c) @ e1) % (2 * np.pi)
        a_e = np.arctan2((nxt[i] - c) @ e2, (nxt[i] - c) @ e1) % (2 * np.pi)
        sweep = a_e if a_m <= a_e else a_e - 2 * np.pi
        t = np.linspace(0.0, sweep, per_edge, endpoint=False)
        out.append(c + r * (np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2))
    loop = np.concatenate(out)
    q = np.concatenate([loop, loop[:1]])
    seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-12:
        return loop
    want = np.linspace(0.0, s[-1], len(loop), endpoint=False)
    return np.stack([np.interp(want, s, q[:, d]) for d in range(3)], axis=1)


def demo():
    """Round-trip: the target must rebuild the outline it was made from."""
    import pathlib

    from scipy.spatial import cKDTree

    from .dataset_curve import load_curve_parts
    from .meshgen import fastener_frame
    from .staged import ordered_outline, to_frame

    R = pathlib.Path(__file__).resolve().parents[4]
    parts = load_curve_parts(R / "runs" / "wtok_synth")[:60]
    err, ne, arc, dropped = [], [], [], 0
    for p in parts:
        fr = fastener_frame(p)
        x = frame_target(p, fr)
        if x is None:
            dropped += 1
            continue
        loop = ordered_outline(p)
        if loop is None:
            continue
        ref, _ = to_frame(loop, np.zeros_like(loop), fr)
        got = realize_frame(x)
        # both directions: the frame must cover the outline AND stay on it
        d1 = cKDTree(got).query(ref)[0].mean()
        d2 = cKDTree(ref).query(got)[0].mean()
        err.append(fr[2] * max(d1, d2))       # frame unit -> mm
        ne.append(int((x[:, UNUSED_F] < 0.5).sum()))
        arc.append(float(x[:len(ne)][:, 6].mean()))
    print(f"{len(err)} parts round-tripped, {dropped} over {EDGE_SLOTS} slots")
    print(f"edges per part: median {np.median(ne):.0f}  max {max(ne)}")
    print(f"rebuild error vs the sampled outline: median {np.median(err):.3f}mm"
          f"  p90 {np.percentile(err, 90):.3f}mm  max {max(err):.3f}mm")
    assert np.median(err) < 1.0, "the parametric frame does not rebuild the outline"
    assert dropped == 0, f"{dropped} parts need more than {EDGE_SLOTS} slots"
    print("ok")


if __name__ == "__main__":
    demo()


CONSIST_LAMBDA = 2.0    # swept, not tuned: see below


def consistent_corners(P, N, lam: float = CONSIST_LAMBDA):
    """Move the corners onto the planes the model's own normal channel declares.

    The model knows the panels -- its emitted normals sit 3.8 deg from the true
    ones and find the same number of them -- but its corners sit 6.7 deg off
    those same normals, nearly twice the error against the truth. The two
    outputs disagree with each other, and this solves for the corners that the
    output already implies. No knowledge is added from outside.

    Nothing here is capped or fitted to this dataset: the planes are whatever
    the normal field says, so their number grows with the part, and there is no
    fastener count anywhere in it. An earlier attempt fitted planes to the
    corners with a fixed 4-plane cap and a hand-set tolerance; it drove the
    twist to zero and moved the shape 0.34mm further from the part, and was
    dropped.

    lam trades consistency against staying near the emitted corners. Median
    over THREE independent draws x 30 val parts:

        lam    twist mm    shape error mm
        0        0.367         7.31
        1        0.103         7.30
        2        0.067         7.22     <- best shape of any setting
        3        0.051         7.54
        5        0.041         7.45

    2 gives the largest twist reduction whose shape cost is zero or better. An
    earlier two-draw sweep picked 3, and 10 looked free in one draw and cost
    0.85mm in the other -- the shape numbers carry roughly +-0.2mm of sampling
    noise, so anything chosen on a single draw is not chosen.
    """
    P = np.asarray(P, float)
    n = len(P)
    rows, rhs = [], []
    for i in range(n):
        for d in range(3):
            r = np.zeros(3 * n)
            r[3 * i + d] = 1.0
            rows.append(r)
            rhs.append(P[i, d])
    for i in range(n):
        nb = float(np.linalg.norm(N[i]))
        if nb < 0.5:                       # no panel normal here (a fold)
            continue
        ni = np.asarray(N[i], float) / nb
        win = [(i + k) % n for k in (-1, 0, 1, 2)]
        for a, b in zip(win, win[1:]):
            # every corner in the window must share one n.q -- that IS coplanar
            r = np.zeros(3 * n)
            r[3 * a:3 * a + 3] = lam * ni
            r[3 * b:3 * b + 3] = -lam * ni
            rows.append(r)
            rhs.append(0.0)
    q = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)[0]
    return q.reshape(n, 3)


def apply_consistency(x, lam: float = CONSIST_LAMBDA):
    """`consistent_corners` over a frame tensor, live slots only."""
    x = np.asarray(x, float).copy()
    live = x[:, 7] < 0.5
    if live.sum() < 6 or x.shape[1] < FRAME_CH:
        return x
    idx = np.flatnonzero(live)
    x[idx, 0:3] = consistent_corners(x[live, 0:3], x[live, 6:9], lam)
    return x


def consist_demo():
    """A flat ring with one corner lifted, and normals that say it is flat."""
    t = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    P = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], 1) * 10.0
    P[3, 2] = 2.0                                  # pushed off the panel
    N = np.tile([0.0, 0.0, 1.0], (12, 1))          # the normal says: flat
    before = float(np.abs(P[:, 2]).max())
    Q = consistent_corners(P, N, CONSIST_LAMBDA)
    after = float(np.abs(Q[:, 2] - np.median(Q[:, 2])).max())
    moved = float(np.linalg.norm(Q[:, :2] - P[:, :2], axis=1).max())
    print(f"off-plane {before:.3f} -> {after:.3f}   in-plane drift {moved:.4f}")
    assert after < before / 3, (before, after)
    assert moved < 1e-6, "the solve must not slide corners within the plane"
    print("ok")


MEDOID_K = 9            # draws to choose among; 9 measured, see frame_eval


def medoid(frames, per_edge: int = 60):
    """Pick the draw closest to all the others. Returns (index, frames[index]).

    A CHOICE, not a blend: the winner is one of the generated frames, unchanged,
    so it keeps every structural guarantee the representation gives -- one closed
    loop of lines and arcs. Averaging K frames would destroy that, since corner i
    of one draw is not corner i of another.

    Worth 7.77mm -> 5.47mm on 30 val parts at K=9. The model already draws a
    4.12mm frame among those 9; it just cannot tell which one, and the middle of
    its own answers is the best guess available without new information.
    """
    from scipy.spatial import cKDTree

    polys = [realize_frame(f, per_edge) for f in frames]
    keep = [i for i, o in enumerate(polys) if len(o) >= 10]
    if not keep:
        return 0, frames[0]
    D = np.zeros((len(keep), len(keep)))
    for a, i in enumerate(keep):
        for b, j in enumerate(keep):
            if a != b:
                D[a, b] = float(cKDTree(polys[j]).query(polys[i])[0].mean())
    w = keep[int(D.sum(1).argmin())]
    return w, frames[w]


# ----------------------------------------------------------- bend wireframe

BEND_CH = 11            # this layout is its own; it does not share the outline's
BEND_SLOTS = 24         # folds per part: median 15, max 22 measured
BEND_MERGE_DEG = 8.0    # parallel tolerance when pairing a fold's two tangents
BEND_MERGE_MM = 25.0    # and how far apart the pair may sit


def fold_lines(part, min_mm: float = 10.0):
    """The part's FOLDS, as polylines -- not the raw bend_line edges.

    bend_line marks where a bend's cylinder meets a flat panel, so one fold
    contributes TWO of them, parallel and about 5.5mm apart. Merging each pair
    into its centreline is what turns the class into the thing a wireframe
    actually wants: 24 segments become 15 folds, and the per-part maximum drops
    from 40 to 22. Segments under 10mm are tessellation slivers -- 61% of them,
    holding 11.9% of the total length.
    """
    from .codec import realize_edge
    from .train_curve import realized_q

    q = realized_q(part, part.vertices, part.edges)
    S = []
    for e in part.edges:
        if e.get("cls") != "bend_line":
            continue
        poly = realize_edge(q, e)
        if poly is None or len(poly) < 2:
            continue
        poly = np.asarray(poly, float)
        if float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum()) >= min_mm:
            S.append(poly)
    if not S:
        return []
    ax = [((s[0] + s[-1]) / 2,
           (s[-1] - s[0]) / max(np.linalg.norm(s[-1] - s[0]), 1e-9),
           float(np.linalg.norm(s[-1] - s[0]))) for s in S]
    taken, out = set(), []
    for i in range(len(S)):
        if i in taken:
            continue
        best, bj = None, -1
        for j in range(len(S)):
            if j == i or j in taken:
                continue
            if abs(float(ax[i][1] @ ax[j][1])) <= np.cos(np.radians(BEND_MERGE_DEG)):
                continue
            d = float(np.linalg.norm(ax[i][0] - ax[j][0]))
            lr = min(ax[i][2], ax[j][2]) / max(ax[i][2], ax[j][2])
            if d < BEND_MERGE_MM and lr > 0.6 and (best is None or d < best):
                best, bj = d, j
        taken.add(i)
        if bj < 0:
            out.append(S[i])
            continue
        taken.add(bj)
        n = min(len(S[i]), len(S[bj]))
        a = S[i][np.round(np.linspace(0, len(S[i]) - 1, n)).astype(int)]
        b = S[bj][np.round(np.linspace(0, len(S[bj]) - 1, n)).astype(int)]
        if float(np.linalg.norm(a[0] - b[0])) > float(np.linalg.norm(a[0] - b[-1])):
            b = b[::-1]
        out.append(0.5 * (a + b))
    out.sort(key=lambda s: -float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum()))
    return out[:BEND_SLOTS]


def bend_frame_target(part, frame, slots: int = BEND_SLOTS, ch: int = BEND_CH):
    """(slots, FRAME_CH), the same layout as the outline frame.

        [0:3] first endpoint      [3:6] second endpoint
        [6]   is_arc              [7]   unused
        [8:11] bulge -- the mid point measured from the chord midpoint

    Endpoints are FREE, not tied to the outline: measured over 200 parts a fold
    end sits 11.4mm from the boundary and only 3% of them come within 2mm, so
    parametrising a fold by two positions along the perimeter -- which would
    have made a floating wire impossible by construction -- does not fit the
    data. Whatever holds the wires down has to come from somewhere else.

    Slots are ordered longest-first, which is a canonical order the model can
    learn ("slots past here are usually unused"), the same role the strand index
    played in the point-based bend stage.
    """
    from .meshgen import to_frame

    F = fold_lines(part)
    if not F:
        return None
    x = np.zeros((slots, ch))
    x[len(F):, 7] = 1.0
    for i, f in enumerate(F):
        pts, _ = to_frame(f, np.zeros_like(f), frame)
        a, b, mid = pts[0], pts[-1], _halfway(pts)
        chord = 0.5 * (a + b)
        sag = float(np.linalg.norm(mid - chord))
        span = max(float(np.linalg.norm(b - a)), 1e-9)
        x[i, 0:3], x[i, 3:6] = a, b
        x[i, 6] = 1.0 if sag / span > 0.02 else 0.0     # bowed enough to be an arc
        x[i, 8:11] = (mid - chord) * x[i, 6]
    return x


def _densify(pts, n):
    """Resample a polyline to n points, evenly by arc length."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-12:
        return np.repeat(pts[:1], n, axis=0)
    want = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(want, s, pts[:, d]) for d in range(3)], axis=1)


def _halfway(pts):
    """The point half way ALONG the polyline, by arc length.

    Not pts[len(pts)//2]: a LINE edge realises as exactly two points, so the
    index midpoint is the END point, the sag comes out as half the chord, and
    every fold gets flagged as an arc bulging by half its own length.
    """
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-12:
        return pts[0]
    return np.array([np.interp(s[-1] / 2, s, pts[:, d]) for d in range(3)])


def realize_bend(x, per_edge: int = 40, arc_thresh: float = 0.5):
    """Bend slots back to a list of polylines, one per live fold."""
    from .codec import circle_through

    out = []
    for r in x:
        if r[7] >= 0.5:
            continue
        a, b = r[0:3], r[3:6]
        if float(np.linalg.norm(b - a)) < 1e-9:
            continue
        if r[6] <= arc_thresh:
            out.append(np.linspace(a, b, per_edge))
            continue
        mid = 0.5 * (a + b) + r[8:11]
        cc = circle_through(a, mid, b)
        if cc is None:
            out.append(np.linspace(a, b, per_edge))
            continue
        c, n, rad = cc
        e1 = (a - c) / max(np.linalg.norm(a - c), 1e-12)
        e2 = np.cross(n, e1)
        a_m = np.arctan2((mid - c) @ e2, (mid - c) @ e1) % (2 * np.pi)
        a_e = np.arctan2((b - c) @ e2, (b - c) @ e1) % (2 * np.pi)
        sweep = a_e if a_m <= a_e else a_e - 2 * np.pi
        t = np.linspace(0.0, sweep, per_edge)
        out.append(c + rad * (np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2))
    return out


def bend_demo():
    """Round-trip: the target must rebuild the folds it was made from."""
    import pathlib

    from scipy.spatial import cKDTree

    from .dataset_curve import load_curve_parts
    from .meshgen import fastener_frame, to_frame

    R = pathlib.Path(__file__).resolve().parents[4]
    parts = load_curve_parts(R / "runs" / "wtok_synth")[:40]
    err, nf, arc = [], [], []
    for p in parts:
        fr = fastener_frame(p)
        x = bend_frame_target(p, fr)
        F = fold_lines(p)
        if x is None or not F:
            continue
        # Densify the TRUE folds to the same resolution before comparing. A
        # LINE realises as two points; measuring 40 reconstructed points against
        # those two charges a quarter of the chord for the densification alone,
        # which reads as a 12mm error on a fold that is reproduced exactly.
        ref = np.concatenate([_densify(to_frame(f, np.zeros_like(f), fr)[0], 40)
                              for f in F])
        got = realize_bend(x)
        if not got:
            continue
        got = np.concatenate(got)
        err.append(fr[2] * max(float(cKDTree(got).query(ref)[0].mean()),
                               float(cKDTree(ref).query(got)[0].mean())))
        nf.append(int((x[:, 7] < 0.5).sum()))
        arc.append(float(x[x[:, 7] < 0.5][:, 6].mean()))
    print(f"{len(err)} parts, folds median {np.median(nf):.0f} max {max(nf)}, "
          f"arc share {np.mean(arc):.2f}")
    print(f"rebuild error: median {np.median(err):.3f}mm  max {max(err):.3f}mm")
    assert max(nf) <= BEND_SLOTS and np.median(err) < 1.5
    print("ok")


# ------------------------------------------------- multi-loop generalisation

MULTI_CH = FRAME_CH + 1     # frame channels + one loop-end flag
LOOP_END = FRAME_CH         # channel: this slot is the LAST edge of its loop


def boundary_loops(part):
    """Every boundary loop of the part, outer first, each walked into order.

    The single-ring representation cannot express a part with a hole: the ring
    IS the loop, and two loops have no single traversal. Measured on the current
    2300 parts every boundary is one loop, so nothing here changes their
    encoding -- this is the shape the data will take once the extraction emits
    inner boundaries, and it degrades exactly to the old behaviour until then.

    Reads an explicit `loop` field when the extraction provides one, and falls
    back to walking the adjacency when it does not.
    """
    from .codec import bin_center, realize_edge
    from .staged import _edge_ends
    from .train_curve import realized_q

    q = realized_q(part, part.vertices, part.edges)
    groups = {}
    for e in part.edges:
        cls = e.get("cls")
        if cls not in ("outer_boundary", "inner_boundary"):
            continue
        ends = _edge_ends(e)
        if ends is None:
            continue
        key = e.get("loop", 0 if cls == "outer_boundary" else 1)
        mid = bin_center(q, q["vertices"][e["refs"][1]]) if e["tau"] == "ARC" else None
        groups.setdefault(key, []).append((ends, mid, e["tau"] == "ARC", cls))
    if not groups:
        return []

    out = []
    for key in sorted(groups):
        adj, pos = {}, {}
        for ends, mid, is_arc, cls in groups[key]:
            adj.setdefault(ends[0], []).append((ends[1], mid, is_arc))
            adj.setdefault(ends[1], []).append((ends[0], mid, is_arc))
        for k in adj:
            pos[k] = bin_center(q, q["vertices"][k])
        seen = set()
        for start in sorted(adj):
            if start in seen:
                continue
            cur, prev, chain = start, None, []
            for _ in range(len(adj) + 1):
                seen.add(cur)
                nxt = [(v, m, a) for v, m, a in adj[cur] if v != prev]
                if not nxt:
                    break
                v, m, a = nxt[0]
                chain.append((pos[cur], m, a))
                prev, cur = cur, v
                if cur == start:
                    break
            if len(chain) >= 3 and cur == start:
                out.append((chain, groups[key][0][3]))
    # outer boundary first, then holes largest to smallest -- a canonical order,
    # so slot 0 means the same thing on every part
    def size(c):
        p = np.stack([x[0] for x in c[0]])
        return float(np.linalg.norm(p.max(0) - p.min(0)))
    outer = [c for c in out if c[1] == "outer_boundary"]
    inner = sorted([c for c in out if c[1] != "outer_boundary"], key=size, reverse=True)
    if not outer and out:                       # no class says outer: take the biggest
        out.sort(key=size, reverse=True)
        return out
    return outer + inner


def multi_frame_target(part, frame, slots: int = EDGE_SLOTS):
    """(slots, MULTI_CH). Loops laid end to end, each closed by its own flag.

    Slot i still joins corner i to corner i+1, EXCEPT where `LOOP_END` is set:
    there the edge closes back to that loop's first corner. Closure therefore
    stays structural for every loop, not just for one, which is the property the
    ordered ring was adopted for in the first place.
    """
    from .meshgen import to_frame

    L = boundary_loops(part)
    if not L:
        return None
    x = np.zeros((slots, MULTI_CH))
    x[:, UNUSED_F] = 1.0                             # unused until filled
    at = 0
    for chain, _cls in L:
        n = len(chain)
        if at + n > slots:
            break
        corners = np.stack([c for c, _, _ in chain])
        is_arc = np.array([a for _, _, a in chain], float)
        mids = np.stack([m if m is not None else c for c, m, a in chain])
        p, _ = to_frame(corners, np.zeros_like(corners), frame)
        mid, _ = to_frame(mids, np.zeros_like(mids), frame)
        p, mid, is_arc = _canonicalise(p, mid, is_arc)
        chord = 0.5 * (p + np.roll(p, -1, 0))
        x[at:at + n, 0:3] = p
        x[at:at + n, SAG] = np.einsum(
            "ij,ij->i", (mid - chord) * is_arc[:, None], loop_outward(p))
        x[at:at + n, ARC_F] = is_arc
        x[at:at + n, UNUSED_F] = 0.0
        x[at:at + n, 6:9] = edge_normals(p)
        x[at + n - 1, LOOP_END] = 1.0                # close this loop here
        at += n
    return x if at else None


def realize_multi(x, per_edge: int = 24, arc_thresh: float = 0.5):
    """Multi-loop slots back to a list of closed polylines, one per loop."""
    live = np.flatnonzero(x[:, UNUSED_F] < 0.5)
    if len(live) < 3:
        return []
    out, start = [], 0
    for i in range(len(live)):
        if x[live[i], LOOP_END] > 0.5 or i == len(live) - 1:
            idx = live[start:i + 1]
            if len(idx) >= 3:
                sub = np.zeros((len(idx), FRAME_CH))
                sub[:, :FRAME_CH] = x[idx, :FRAME_CH]
                out.append(realize_frame(sub, per_edge, arc_thresh))
            start = i + 1
    return [o for o in out if len(o)]


def multi_demo():
    """Two loops must come back as two closed loops, and one as one."""
    import pathlib

    from scipy.spatial import cKDTree

    from .dataset_curve import load_curve_parts
    from .meshgen import fastener_frame
    from .validity import outline_closed

    R = pathlib.Path(__file__).resolve().parents[4]
    parts = load_curve_parts(R / "runs" / "wtok_synth")[:40]
    nl, err = [], []
    for p in parts:
        fr = fastener_frame(p)
        x = multi_frame_target(p, fr)
        y = frame_target(p, fr)
        if x is None or y is None:
            continue
        loops = realize_multi(x)
        nl.append(len(loops))
        # with one loop it must agree with the single-ring encoding exactly
        if len(loops) == 1:
            err.append(fr[2] * float(cKDTree(realize_frame(y)).query(loops[0])[0].mean()))
    print(f"{len(nl)} parts, loops per part {sorted(set(nl))}")
    print(f"single-loop parts agree with the ring encoding to "
          f"{np.median(err):.4f}mm")
    assert np.median(err) < 0.05, np.median(err)

    # synthetic two-loop part: a square with a square hole
    x = np.zeros((EDGE_SLOTS, MULTI_CH)); x[:, UNUSED_F] = 1.0
    x[:4, 0:3] = [[0,0,0],[40,0,0],[40,40,0],[0,40,0]]
    x[4:8, 0:3] = [[15,15,0],[25,15,0],[25,25,0],[15,25,0]]
    x[:8, UNUSED_F] = 0.0
    x[3, LOOP_END] = 1.0; x[7, LOOP_END] = 1.0
    loops = realize_multi(x)
    ends = [outline_closed(o)[0] for o in loops]
    print(f"square with a square hole -> {len(loops)} loops, "
          f"endpoint fractions {[round(e,3) for e in ends]}")
    assert len(loops) == 2 and max(ends) < 0.05
    print("ok")


BEND_CLASSES = ("bend_line", "bead", "inflection", "corner_relief")


def bend_features(part, min_mm: float = 10.0):
    """Every bend-like feature, as (polyline, class, closed).

    `bend_line` still goes through the pair merge that turns two tangent lines
    into one fold. The other classes do not exist in the data yet -- the
    extraction emits only outer_boundary and bend_line, and 7.8% of the surface
    is curved more than 15mm from anything it marks. They are read here so that
    the day they appear nothing has to change, and until then this returns
    exactly what fold_lines returns.

    `closed` matters because a bead is a ring, not a segment: the one-slot-per
    -segment encoding cannot express it, and a chain closed by LOOP_END can.
    """
    from .codec import realize_edge
    from .train_curve import realized_q

    out = [(f, "bend_line", False) for f in fold_lines(part, min_mm)]
    q = realized_q(part, part.vertices, part.edges)
    for cls in BEND_CLASSES[1:]:
        segs = []
        for e in part.edges:
            if e.get("cls") != cls:
                continue
            poly = realize_edge(q, e)
            if poly is not None and len(poly) >= 2:
                segs.append(np.asarray(poly, float))
        for s in segs:
            closed = bool(np.linalg.norm(s[0] - s[-1]) < 1e-6) or cls == "bead"
            out.append((s, cls, closed))
    return out


def bend_multi_target(part, frame, slots: int = BEND_SLOTS,
                      n_cls: int = len(BEND_CLASSES)):
    """(slots, BEND_MULTI_CH): folds, beads and the rest in one tensor.

        [0:3] endpoint 1   [3:6] endpoint 2   [6] is_arc   [7] unused
        [8:11] bulge       [11] loop_end      [12:12+n_cls] class one-hot

    A straight or arc fold is one slot. A bead is a chain of slots whose last
    one sets loop_end, exactly the mechanism the multi-loop outline uses -- so a
    closed bend feature is closed by construction, the same way a hole is.

    On today's data this fills only the bend_line channel and sets no loop_end,
    so it encodes the same thing bend_frame_target does.
    """
    from .meshgen import to_frame

    F = bend_features(part)
    if not F:
        return None
    ch = 12 + n_cls
    x = np.zeros((slots, ch))
    x[:, 7] = 1.0
    at = 0
    for poly, cls, closed in F:
        if at >= slots:
            break
        pts, _ = to_frame(poly, np.zeros_like(poly), frame)
        ci = BEND_CLASSES.index(cls) if cls in BEND_CLASSES else 0
        if not closed:
            a, b, mid = pts[0], pts[-1], _halfway(pts)
            chord = 0.5 * (a + b)
            span = max(float(np.linalg.norm(b - a)), 1e-9)
            x[at, 0:3], x[at, 3:6] = a, b
            x[at, 6] = 1.0 if float(np.linalg.norm(mid - chord)) / span > 0.02 else 0.0
            x[at, 7] = 0.0
            x[at, 8:11] = (mid - chord) * x[at, 6]
            x[at, 12 + ci] = 1.0
            at += 1
            continue
        # closed: lay the ring down as a chain of straight slots and close it
        k = min(8, slots - at)
        if k < 3:
            break
        r = _densify(np.concatenate([pts, pts[:1]]), k + 1)[:k]
        for j in range(k):
            x[at + j, 0:3] = r[j]
            x[at + j, 3:6] = r[(j + 1) % k]
            x[at + j, 7] = 0.0
            x[at + j, 12 + ci] = 1.0
        x[at + k - 1, LOOP_END] = 1.0
        at += k
    return x if at else None


def bend_multi_demo():
    """Degrades to the current encoding today, and holds a bead when given one."""
    import pathlib

    from scipy.spatial import cKDTree

    from .dataset_curve import load_curve_parts
    from .meshgen import fastener_frame

    R = pathlib.Path(__file__).resolve().parents[4]
    parts = load_curve_parts(R / "runs" / "wtok_synth")[:40]
    err, cls_seen = [], set()
    for p in parts:
        fr = fastener_frame(p)
        a = bend_multi_target(p, fr)
        b = bend_frame_target(p, fr)
        if a is None or b is None:
            continue
        for i in np.flatnonzero(a[:, 7] < 0.5):
            cls_seen.add(BEND_CLASSES[int(np.argmax(a[i, 12:]))])
        A = realize_bend(a[:, :FRAME_CH])
        B = realize_bend(b)
        if A and B:
            err.append(fr[2] * float(cKDTree(np.concatenate(B)).query(
                np.concatenate(A))[0].mean()))
    print(f"{len(err)} parts, classes present today: {sorted(cls_seen)}")
    print(f"agrees with bend_frame_target to {np.median(err):.4f}mm")
    assert np.median(err) < 0.01 and cls_seen == {"bend_line"}

    # a synthetic bead: a closed ring must come back closed
    from types import SimpleNamespace
    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = np.stack([np.cos(t) * 12, np.sin(t) * 12, np.zeros_like(t)], 1)
    x = np.zeros((BEND_SLOTS, 12 + len(BEND_CLASSES)))
    x[:, 7] = 1.0
    k = 8
    r = _densify(np.concatenate([ring, ring[:1]]), k + 1)[:k]
    for j in range(k):
        x[j, 0:3], x[j, 3:6] = r[j], r[(j + 1) % k]
        x[j, 7] = 0.0
        x[j, 12 + BEND_CLASSES.index("bead")] = 1.0
    x[k - 1, LOOP_END] = 1.0
    seg = realize_bend(x[:, :FRAME_CH])
    ends = np.concatenate([[s[0], s[-1]] for s in seg])
    gap = float(np.linalg.norm(x[k - 1, 3:6] - x[0, 0:3]))
    print(f"bead: {len(seg)} slots, chain closes to {gap:.6f}")
    assert len(seg) == k and gap < 1e-9
    print("ok")

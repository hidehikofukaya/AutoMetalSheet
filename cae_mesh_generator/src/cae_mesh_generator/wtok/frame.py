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
FRAME_CH = 11           # corner 3, arc bulge 3, is_arc 1, unused 1, normal 3
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

    x = np.zeros((slots, FRAME_CH))
    n = len(p)
    x[:n, 0:3] = p
    x[:n, 3:6] = bulge
    x[:n, 6] = is_arc
    x[n:, 7] = 1.0                            # unused slots
    x[:n, 8:11] = edge_normals(p)
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

    live = x[:, 7] < 0.5
    p = x[live, 0:3]
    if len(p) < 3:
        return np.zeros((0, 3))
    bulge, is_arc = x[live, 3:6], x[live, 6] > arc_thresh
    nxt = np.roll(p, -1, 0)
    mid = 0.5 * (p + nxt) + bulge

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
        ne.append(int((x[:, 7] < 0.5).sum()))
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
    if live.sum() < 6 or x.shape[1] < 11:
        return x
    idx = np.flatnonzero(live)
    x[idx, 0:3] = consistent_corners(x[live, 0:3], x[live, 8:11], lam)
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


def bend_frame_target(part, frame, slots: int = BEND_SLOTS):
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
    x = np.zeros((slots, FRAME_CH))
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

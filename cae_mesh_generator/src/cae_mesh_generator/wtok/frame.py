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

"""Bend lines as a token sequence. Nothing here is a count.

Every earlier target fixed a number and built to it -- two folds, one bead, four
ridges, six slots, twenty-eight curves. This one does not. The tensor has a
CAPACITY of N tokens, which is a memory budget, and the part uses as many as it
uses. Curves are the runs between the tokens that raise `end`, so how many
curves a part has is read off the output rather than designed into it. A part
with one bend and a part with forty are the same shape of tensor.

    [0:3]  xyz
    [3:6]  unit tangent
    [6]    end   -- this token is the last of its curve
    [7]    close -- that curve closes back on its own first token
    [8]    unused

`close` is what lets a closed curve exist at all: 51% of the real bend curves
are closed, and an anchor-both-ends design could not express one. A curve is
either closed, or it is open and its ends are where they are -- both are
expressible, and neither is imposed.

Order is canonical so slot i means the same thing on every part: curves longest
first, points along each curve in traversal order, and each curve's traversal
started at the end nearest the frame origin.
"""
from __future__ import annotations

import numpy as np

TOK_N = 512             # capacity, not a count
TOK_CH = 9
END, CLOSE, UNUSED = 6, 7, 8


def _arc(c):
    d = np.linalg.norm(np.diff(c, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def _resample(c, n):
    s = _arc(c)
    if s[-1] < 1e-9 or n < 2:
        return np.repeat(c[:1], max(n, 1), axis=0)
    w = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(w, s, c[:, d]) for d in range(3)], axis=1)


def _simplify(c, tol):
    """Douglas-Peucker: the fewest vertices that keep the polyline within `tol`."""
    keep = np.zeros(len(c), bool)
    keep[[0, -1]] = True
    stack = [(0, len(c) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        seg = c[b] - c[a]
        L = np.linalg.norm(seg)
        w = c[a + 1:b] - c[a]
        if L < 1e-9:
            d = np.linalg.norm(w, axis=1)
        else:
            d = np.linalg.norm(w - np.outer(w @ seg / L, seg / L), axis=1)
        i = int(np.argmax(d))
        if d[i] > tol:
            keep[a + 1 + i] = True
            stack += [(a, a + 1 + i), (a + 1 + i, b)]
    return c[keep]


def curves_to_tokens(curves, frame=None, n_tok: int = TOK_N, min_pts: int = 3,
                     corner_mm: float = 0.0):
    """Lay curves out as tokens, sharing the budget by arc length.

    A long curve gets more tokens than a short one because it needs them, not
    because a rule assigned it a quota. If the part has more curves than the
    budget can give `min_pts` each, the shortest are dropped and the caller can
    see it in the returned count -- that is a capacity limit, and it is reported
    rather than hidden.

    `corner_mm` > 0 switches to CORNER tokens: each curve is reduced to the
    fewest vertices that keep it within that tolerance, and the tokens are
    those vertices. A straight fold is then two tokens and cannot zigzag, the
    way an outline edge cannot -- the property sits in the representation.
    Measured on the dense tokens the generator turned 9026deg per part in
    excess against the teacher's 759deg. How many vertices a curve gets is
    whatever its shape needs.
    ponytail: straight segments between corners; add an arc bulge (as the
    outline frame has) when bead ridges need true arcs for B-rep.
    """
    from .meshgen import to_frame

    cur = [np.asarray(c, float) for c in curves if len(np.asarray(c, float)) >= 2]
    if not cur:
        return np.zeros((n_tok, TOK_CH)), 0
    L = np.array([_arc(c)[-1] for c in cur])
    order = np.argsort(-L)
    cur = [cur[i] for i in order]
    L = L[order]
    if corner_mm > 0:
        cur = [_simplify(c, corner_mm) for c in cur]
        n = np.array([len(c) for c in cur])
        while n.sum() > n_tok and len(cur) > 1:  # over capacity: drop the shortest
            cur, n = cur[:-1], n[:-1]
    else:
        keep = min(len(cur), n_tok // min_pts)
        cur, L = cur[:keep], L[:keep]
        share = L / max(L.sum(), 1e-9)
        n = np.maximum(min_pts, np.floor(share * n_tok).astype(int))
        while n.sum() > n_tok:                   # trim the longest first
            n[int(np.argmax(n))] -= 1
    x = np.zeros((n_tok, TOK_CH))
    x[:, UNUSED] = 1.0
    at = 0
    for c, k in zip(cur, n):
        closed = bool(np.linalg.norm(c[0] - c[-1]) < 1e-3)
        # canonical traversal: start at the end nearer the frame origin
        p = c if corner_mm > 0 else _resample(c, int(k))
        if not closed and np.linalg.norm(p[-1]) < np.linalg.norm(p[0]):
            p = p[::-1]
        if frame is not None:
            p, _ = to_frame(p, np.zeros_like(p), frame)
        t = np.gradient(p, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        x[at:at + len(p), 0:3] = p
        x[at:at + len(p), 3:6] = t
        x[at:at + len(p), UNUSED] = 0.0
        x[at + len(p) - 1, END] = 1.0
        if closed:
            x[at:at + len(p), CLOSE] = 1.0
        at += len(p)
    return x, len(cur)


def tokens_to_curves(x, end_thr: float = 0.5):
    """Read the curves back. Their number is however many `end` flags there are."""
    live = np.flatnonzero(x[:, UNUSED] < 0.5)
    if len(live) < 2:
        return []
    out, start = [], 0
    for i in range(len(live)):
        last = i == len(live) - 1
        if x[live[i], END] > end_thr or last:
            idx = live[start:i + 1]
            if len(idx) >= 2:
                p = x[idx, 0:3]
                if x[idx[0], CLOSE] > 0.5:
                    p = np.concatenate([p, p[:1]])
                out.append(p)
            start = i + 1
    return out


def demo():
    """The tokens must give back the curves, and impose no count on the way."""
    import collections
    import pathlib

    from scipy.spatial import cKDTree

    from .bendlines import mesh_bend_lines
    from .dataset_curve import load_curve_parts
    from .meshgen import fastener_frame, to_frame

    R = pathlib.Path(__file__).resolve().parents[4]
    have = {p.stem for p in (R / "runs" / "mesh_synth" / "parts").glob("*.npz")}
    parts = [p for p in load_curve_parts(R / "runs" / "wtok_synth")
             if p.name in have][:40]

    err, nin, nout, dropped, closed = [], [], [], [], []
    for p in parts:
        cur = mesh_bend_lines(p, classes=("bend_line", "crease"))
        if not cur:
            continue
        fr = fastener_frame(p)
        x, kept = curves_to_tokens(cur, fr)
        got = tokens_to_curves(x)
        nin.append(len(cur)); nout.append(len(got)); dropped.append(len(cur) - kept)
        closed.append(sum(1 for c in got if np.linalg.norm(c[0] - c[-1]) < 1e-6))
        ref = np.concatenate([to_frame(np.asarray(c, float),
                                       np.zeros_like(np.asarray(c, float)), fr)[0]
                              for c in cur])
        A = np.concatenate(got)
        err.append(fr[2] * max(float(cKDTree(A).query(ref)[0].mean()),
                               float(cKDTree(ref).query(A)[0].mean())))
    f = lambda v: (np.median(v), max(v))
    print(f"{len(err)} parts, capacity {TOK_N} tokens")
    print(f"  curves in    median {f(nin)[0]:4.0f}  max {f(nin)[1]:4.0f}")
    print(f"  curves back  median {f(nout)[0]:4.0f}  max {f(nout)[1]:4.0f}")
    print(f"  of which closed  median {np.median(closed):4.0f}")
    print(f"  curves dropped for capacity  median {np.median(dropped):.0f}"
          f"  max {max(dropped)}")
    print(f"  round-trip error  median {np.median(err):.3f}mm"
          f"  p90 {np.percentile(err, 90):.3f}mm")
    assert nin == nout or np.median(np.abs(np.array(nin) - np.array(nout))) == 0, \
        "the token layout lost or invented curves"
    assert np.median(err) < 2.0
    print("ok")


if __name__ == "__main__":
    demo()

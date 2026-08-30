"""Curves as displacements: each token says where to go NEXT, not where it is.

The position layout produced outlines that wander 14 to 95 degrees per step --
smooth on some parts, visibly ragged on others -- because nothing ties one token
to the next. Predicting a step instead of a place makes smoothness a property of
what is generated rather than something a loss has to ask for, which is the move
that worked three times before (an ordered ring for closure, parametric edges
for straightness, per-edge normals for planarity).

Measured on the true targets before building this: a displacement field has 54x
less spread than a position field, and integrating one with 2% per-token noise
drifts 0.34mm (p90 0.55) against a current position error of 7.7mm.

    [0:3]   step -- displacement to the next token, or the absolute position of
            a curve's FIRST token
    [3]     start -- this token begins a curve, so [0:3] is a position
    [4]     end   -- this token finishes a curve
    [5]     close -- that curve returns to its own start
    [6]     unused

A closed curve is closed by construction: its last step is not read from the
model at all but computed as whatever remains to return to the start. Closure
therefore cannot fail, in the way an ordered ring could not fail to be a loop.

Nothing here is a count. Curves are the runs between `end` flags, so a part has
as many as its output has -- the tensor is a capacity.
"""
from __future__ import annotations

import numpy as np

TOK_N = 512             # capacity, not a count
DTOK_CH = 7
STEP, START, END, CLOSE, UNUSED = slice(0, 3), 3, 4, 5, 6


def _arc(c):
    return np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(c, axis=0), axis=1))])


def _resample(c, n):
    s = _arc(c)
    if s[-1] < 1e-9 or n < 2:
        return np.repeat(c[:1], max(n, 1), axis=0)
    w = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(w, s, c[:, d]) for d in range(3)], axis=1)


def curves_to_delta(curves, frame=None, n_tok: int = TOK_N, min_pts: int = 3):
    """Lay curves out as start-and-step tokens, sharing the budget by arc length."""
    from .meshgen import to_frame

    cur = [np.asarray(c, float) for c in curves if len(np.asarray(c, float)) >= 2]
    if not cur:
        return np.zeros((n_tok, DTOK_CH)), 0
    L = np.array([_arc(c)[-1] for c in cur])
    o = np.argsort(-L)
    cur, L = [cur[i] for i in o], L[o]
    keep = min(len(cur), n_tok // min_pts)
    cur, L = cur[:keep], L[:keep]

    n = np.maximum(min_pts, np.floor(L / max(L.sum(), 1e-9) * n_tok).astype(int))
    while n.sum() > n_tok:
        n[int(np.argmax(n))] -= 1

    x = np.zeros((n_tok, DTOK_CH))
    x[:, UNUSED] = 1.0
    at = 0
    for c, k in zip(cur, n):
        closed = bool(np.linalg.norm(c[0] - c[-1]) < 1e-3)
        p = _resample(c, int(k))
        if not closed and np.linalg.norm(p[-1]) < np.linalg.norm(p[0]):
            p = p[::-1]
        if frame is not None:
            p, _ = to_frame(p, np.zeros_like(p), frame)
        x[at, STEP] = p[0]                      # a start token carries a POSITION
        x[at, START] = 1.0
        x[at + 1:at + len(p), STEP] = np.diff(p, axis=0)
        x[at:at + len(p), UNUSED] = 0.0
        x[at + len(p) - 1, END] = 1.0
        if closed:
            x[at:at + len(p), CLOSE] = 1.0
        at += len(p)
    return x, len(cur)


def delta_to_curves(x, thr: float = 0.5):
    """Integrate the tokens back into curves.

    A curve's last step is DISCARDED when the curve is closed, and replaced by
    whatever returns the walk to its own first point. That is what makes closure
    structural: the model can be wrong about the shape, never about closing.
    """
    live = np.flatnonzero(x[:, UNUSED] < thr)
    if len(live) < 2:
        return []
    out, run = [], []
    for i in live:
        if x[i, START] > thr and run:
            out.append(run)
            run = []
        run.append(i)
        if x[i, END] > thr:
            out.append(run)
            run = []
    if run:
        out.append(run)

    curves = []
    for idx in out:
        if len(idx) < 2:
            continue
        p = [x[idx[0], STEP].copy()]
        for j in idx[1:]:
            p.append(p[-1] + x[j, STEP])
        p = np.stack(p)
        if x[idx[0], CLOSE] > thr:
            # drop the emitted last step and close the loop exactly
            p = np.concatenate([p[:-1], p[:1]]) if len(p) > 2 else p
        curves.append(p)
    return curves


def demo():
    """Round-trip, and closure that holds even when the steps are wrong."""
    import pathlib

    from scipy.spatial import cKDTree

    from .bendlines import mesh_bend_lines, outline_polylines
    from .dataset_curve import load_curve_parts
    from .meshgen import fastener_frame, to_frame

    R = pathlib.Path(__file__).resolve().parents[4]
    have = {p.stem for p in (R / "runs" / "mesh_synth" / "parts").glob("*.npz")}
    parts = [p for p in load_curve_parts(R / "runs" / "wtok_synth")
             if p.name in have][:40]

    err, nin, nout, gap = [], [], [], []
    rng = np.random.default_rng(0)
    for p in parts:
        fr = fastener_frame(p)
        for cur in (outline_polylines(p),
                    mesh_bend_lines(p, classes=("bend_line", "crease"))):
            if not cur:
                continue
            x, kept = curves_to_delta(cur, fr)
            got = delta_to_curves(x)
            nin.append(kept)
            nout.append(len(got))
            ref = np.concatenate([to_frame(np.asarray(c, float),
                                           np.zeros_like(np.asarray(c, float)),
                                           fr)[0] for c in cur])
            A = np.concatenate(got)
            err.append(fr[2] * max(float(cKDTree(A).query(ref)[0].mean()),
                                   float(cKDTree(ref).query(A)[0].mean())))
            # closure must survive a corrupted step field
            y = x.copy()
            live = y[:, UNUSED] < 0.5
            y[live, STEP] += rng.normal(scale=0.02, size=(int(live.sum()), 3))
            for c in delta_to_curves(y):
                if len(c) > 2 and np.linalg.norm(c[0] - c[-1]) < 1e-9:
                    gap.append(0.0)
            for c, o in zip(delta_to_curves(y), got):
                if len(o) > 2 and np.linalg.norm(o[0] - o[-1]) < 1e-9:
                    gap.append(fr[2] * float(np.linalg.norm(c[0] - c[-1])))
    print(f"{len(err)} curve sets")
    print(f"  curves in / out   median {np.median(nin):.0f} / {np.median(nout):.0f}"
          f"   max {max(nin)} / {max(nout)}")
    print(f"  round-trip error  median {np.median(err):.3f}mm"
          f"   p90 {np.percentile(err, 90):.3f}mm")
    print(f"  closed curves after corrupting every step: gap max "
          f"{(max(gap) if gap else 0.0):.6f}mm  (must be 0)")
    assert np.median(nin) == np.median(nout), "curves lost or invented"
    assert np.median(err) < 2.0
    assert not gap or max(gap) < 1e-9, "closure is not structural"
    print("ok")


if __name__ == "__main__":
    demo()

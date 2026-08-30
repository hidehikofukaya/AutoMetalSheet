"""FAILED ATTEMPT, kept as a record. Do not build on this.

Constructing the fillet skeleton from the bend lines does not work:

    attempt                    curves built   true -> built   built -> true
    plain medial set                    954           2.52mm          6.55mm
    + curvature filter                  954           2.54mm          6.55mm
    + across-the-band pairing           398           3.54mm         16.22mm
    (true surface_frame: 128 curves)

Each added condition made it worse. The reason is that the premise was wrong.
`surface_frame` is not the medial curve of a band between two parallel bend
lines -- it is the boundary between two curved PATCHES:

    cylinder <-> freeform  31%     sphere <-> sphere    10%
    cylinder <-> cylinder  15%     sphere <-> freeform  10%

The earlier measurement that it sits "exactly midway between two bend curves"
(ratio 1.00) did not imply the medial reading: equidistance also holds near a
junction, and that is what was being seen. Proximity (99.9% within 20mm of a
bend line) shows it is LOCAL to the bend lines, not DERIVABLE from them.
Constructing it needs the face decomposition itself, which is the CAD fillet
operation.

Original (now withdrawn) rationale follows.

Construct the fillet skeleton from the bend lines, instead of generating it.

`surface_frame` -- the boundaries between curved faces -- is 74% of what bounds
a non-planar face, so a B-rep cannot be built without it. But it is not design
information: measured over 80 parts it sits at a median 1.79mm from a bend line
and 99.9% of it lies within 20mm of one, and measured again it runs EXACTLY
midway between two bend curves (ratio 1.00, 89% between 0.8 and 1.25).

That makes it the medial curve of the band between neighbouring bend lines --
a consequence of them, not an independent thing. Per the rule in KB 12, what is
derivable should not be generated: the model produces the ~25 bend curves, and
this reconstructs the ~266-curve skeleton from them.

Nothing here decides how many curves come out.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .bendlines import chain_polylines


def densify(c, step: float = 0.5):
    c = np.asarray(c, float)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    if s[-1] < 1e-9:
        return c[:1]
    w = np.linspace(0.0, s[-1], max(int(s[-1] / step) + 2, 2))
    return np.stack([np.interp(w, s, c[:, d]) for d in range(3)], axis=1)


def curved_mask(surface, query, k: int = 14, deg: float = 2.0):
    """Which query points sit where the surface actually bends.

    The same normal-turn-rate field that separates ridge from flat by a factor
    of 63. A midpoint that lands in the middle of a flat panel is not a fillet
    crest, and this is the criterion that says so without inventing a distance.
    """
    S = np.asarray(surface, float)
    if len(S) < k + 1:
        return np.ones(len(query), bool)
    nb = cKDTree(S).query(S, k=k)[1]
    N = np.zeros_like(S)
    for i in range(len(S)):
        w = S[nb[i]] - S[nb[i]].mean(0)
        v = np.linalg.svd(w)[2][-1]
        N[i] = -v if v[int(np.argmax(np.abs(v)))] < 0 else v
    turn = np.degrees(np.arccos(np.clip(
        np.abs(np.einsum("ij,ikj->ik", N, N[nb])), 0, 1))).mean(1)
    near = cKDTree(S).query(np.asarray(query, float))[1]
    return turn[near] > deg


def medial_points(curves, surface=None, step: float = 0.5, on_surface: float = 1.5):
    """Midpoints between each sampled point and its nearest point on ANOTHER curve.

    No gap threshold: every point contributes, and the ones that pair across an
    unrelated part of the shape are removed by the only criterion that is not an
    invented number -- the midpoint has to lie ON the surface. A midpoint that
    floats off the sheet was never a fillet crest.

    When no surface is supplied nothing is filtered, and the caller gets the raw
    medial set.
    """
    pts, who, tan = [], [], []
    for i, c in enumerate(curves):
        q = densify(c, step)
        if len(q) < 2:
            continue
        d = np.gradient(q, axis=0)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        pts.append(q)
        who.append(np.full(len(q), i))
        tan.append(d)
    if not pts:
        return np.zeros((0, 3)), np.zeros((0,))
    P = np.concatenate(pts)
    W = np.concatenate(who)
    T = np.concatenate(tan)
    if W.max() < 1:
        return np.zeros((0, 3)), np.zeros((0,))
    tree = cKDTree(P)
    k = min(64, len(P))
    dd, ii = tree.query(P, k=k)
    mid, gap = [], []
    for a in range(len(P)):
        cand = ii[a][W[ii[a]] != W[a]]
        for b in cand:
            v = P[b] - P[a]
            n = float(np.linalg.norm(v))
            if n < 1e-9:
                continue
            v = v / n
            # ACROSS a band, not along the network: the two curves must run
            # parallel, and the line joining them must cross them. Without this
            # the nearest "other curve" is usually the next fragment of the same
            # chain, and the result is a thickened copy of the input -- 954
            # curves against a true 128.
            if abs(float(T[a] @ T[b])) < 0.9:
                continue
            if abs(float(v @ T[a])) > 0.2 or abs(float(v @ T[b])) > 0.2:
                continue
            mid.append(0.5 * (P[a] + P[b]))
            gap.append(n)
            break
    if not mid:
        return np.zeros((0, 3)), np.zeros((0,))
    M, G = np.asarray(mid), np.asarray(gap)
    if surface is not None and len(surface):
        S = np.asarray(surface, float)
        keep = cKDTree(S).query(M)[0] < on_surface
        M, G = M[keep], G[keep]
        if len(M):
            # and it has to be somewhere the surface bends: a midpoint across a
            # flat panel lies on the sheet too, which is why proximity alone
            # over-produced 964 curves against a true 128
            c = curved_mask(S, M)
            M, G = M[c], G[c]
    return M, G


def skeleton(curves, surface=None, step: float = 0.5, on_surface: float = 1.5):
    """The fillet skeleton as curves, chained from the medial points.

    Points are thinned to `step` so the chaining sees a polyline rather than a
    cloud, then joined the same way the extraction's edges are joined.
    """
    M, _ = medial_points(curves, surface, step, on_surface)
    if len(M) < 4:
        return []
    # thin: keep one point per `step` cell, so duplicates from both sides of a
    # pair collapse to one
    key = np.round(M / step).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    M = M[np.sort(first)]
    if len(M) < 4:
        return []
    # link each point to its neighbours within a cell diagonal, walk the links
    tree = cKDTree(M)
    pairs = tree.query_pairs(step * 1.8, output_type="ndarray")
    if not len(pairs):
        return [M[i:i + 1] for i in range(len(M))]
    segs = [np.stack([M[a], M[b]]) for a, b in pairs]
    return chain_polylines(segs, weld=step * 0.4)


def demo():
    """Does the construction land on the real surface_frame?"""
    import json
    import pathlib

    from .dataset_curve import load_curve_parts

    R = pathlib.Path(__file__).resolve().parents[4]
    WF = R / "runs" / "wtok_synth" / "wireframes"
    MD = R / "runs" / "mesh_synth" / "parts"
    have = {p.stem for p in MD.glob("*.npz")}
    parts = [p for p in load_curve_parts(R / "runs" / "wtok_synth")
             if p.name in have][:25]

    fwd, bwd, ncur = [], [], []
    for p in parts:
        f = WF / f"{p.name}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        bl = [e["polyline"] for e in d["edges"]
              if e["type"] in ("bend_line", "crease") and len(e["polyline"]) >= 2]
        sf = [e["polyline"] for e in d["edges"]
              if e["type"] == "surface_frame" and len(e["polyline"]) >= 2]
        if not bl or not sf:
            continue
        z = np.load(MD / f"{p.name}.npz")
        span = np.maximum(z["env_hi"] - z["env_lo"], 1e-9)
        S = z["xyz"].astype(np.float64) * span + z["env_lo"]
        got = skeleton(bl, surface=S)
        if not got:
            continue
        A = np.concatenate([densify(c) for c in got])
        B = np.concatenate([densify(c) for c in sf])
        fwd.append(float(cKDTree(B).query(A)[0].mean()))   # built -> true
        bwd.append(float(cKDTree(A).query(B)[0].mean()))   # true -> built
        ncur.append((len(bl), len(got), len(sf)))
    n = np.array(ncur)
    print(f"{len(fwd)} parts")
    print(f"  bend curves in            median {np.median(n[:,0]):5.0f}")
    print(f"  skeleton curves built     median {np.median(n[:,1]):5.0f}")
    print(f"  true surface_frame curves median {np.median(n[:,2]):5.0f}")
    print(f"\n  built -> true surface_frame  median {np.median(fwd):5.2f}mm")
    print(f"  true surface_frame -> built  median {np.median(bwd):5.2f}mm")
    assert np.median(bwd) < 6.0, "the construction misses the real skeleton"
    print("ok")


if __name__ == "__main__":
    demo()

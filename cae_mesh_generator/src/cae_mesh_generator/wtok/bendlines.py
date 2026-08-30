"""The bend lines a mesh needs, taken from the part. No classes, no counts.

Every earlier attempt here decided a number first -- two folds, one bead, four
ridges, twenty-eight slots -- and then built to it. That is backwards. A part
has whatever bend lines it has; the count is an observation, not a parameter.

So this takes the curves the extraction found, joins the ones that meet, and
returns them. Nothing is dropped for being short, nothing is merged because a
rule said a pair belongs together, nothing is labelled. What comes out is the
set of curves along which the midsurface is not smooth -- which is exactly the
set a mesher must respect, because a mesh element may not straddle one.

The classes read are the ones the extractor emits for that condition:

    bend_line      cylinder meets plane
    surface_frame  surface meets surface (fillet skeletons, panel transitions)
    crease         a sharp edge between faces

`surface_frame` is 30% of all extracted length and the converter used to drop
it entirely. `seam` is left out: it is a parametrisation artefact of a closed
face, not a place the surface bends.
"""
from __future__ import annotations

import collections
import json
import pathlib

import numpy as np

MESH_CLASSES = ("bend_line", "surface_frame", "crease")
WELD_MM = 0.05          # endpoints this close are the same point


def _wireframe(part_name: str, root: pathlib.Path | None = None):
    root = root or pathlib.Path(__file__).resolve().parents[4]
    f = root / "runs" / "wtok_synth" / "wireframes" / f"{part_name}.json"
    return json.loads(f.read_text()) if f.exists() else None


def chain_polylines(polys, weld: float = WELD_MM):
    """Join polylines that share an endpoint into maximal curves.

    A curve stops where nothing continues it, or where it closes on itself.
    Junctions where three or more curves meet are left as separate curves -- a
    mesher needs the junction to be a node, and forcing a choice of which pair
    to continue would invent structure that is not in the part.
    """
    at = collections.defaultdict(list)

    def key(p):
        return tuple(np.round(np.asarray(p, float) / weld).astype(np.int64))

    for i, p in enumerate(polys):
        at[key(p[0])].append((i, 0))
        at[key(p[-1])].append((i, 1))

    used, out = set(), []
    for i0 in range(len(polys)):
        if i0 in used:
            continue
        used.add(i0)
        cur = [np.asarray(polys[i0], float)]
        for forward in (True, False):
            while True:
                tip = cur[-1][-1] if forward else cur[0][0]
                here = at[key(tip)]
                if len(here) != 2:          # an end, or a junction: stop
                    break
                nxt = [(i, e) for i, e in here if i not in used]
                if not nxt:
                    break
                i, e = nxt[0]
                used.add(i)
                seg = np.asarray(polys[i], float)
                seg = seg if e == 0 else seg[::-1]
                if forward:
                    cur.append(seg[1:])
                else:
                    cur.insert(0, seg[:-1])
        out.append(np.concatenate(cur))
    return out


def mesh_bend_lines(part, classes=MESH_CLASSES, root=None):
    """Every curve along which the midsurface bends. Count-free.

    Returns a list of polylines in part coordinates. How many there are is
    whatever the part has -- this function has no cap, no minimum length and no
    notion of what kind of feature a curve belongs to.
    """
    d = _wireframe(getattr(part, "name", str(part)), root)
    if d is None:
        return []
    polys = [np.asarray(e["polyline"], float) for e in d["edges"]
             if e["type"] in classes and len(e["polyline"]) >= 2]
    return chain_polylines(polys) if polys else []


def outline_polylines(part, root=None):
    """The outer boundary, for the same part, chained the same way."""
    d = _wireframe(getattr(part, "name", str(part)), root)
    if d is None:
        return []
    polys = [np.asarray(e["polyline"], float) for e in d["edges"]
             if e["type"] == "outer_boundary" and len(e["polyline"]) >= 2]
    return chain_polylines(polys) if polys else []


def describe(curves):
    """What a part turned out to have. An observation, not a specification."""
    if not curves:
        return {}
    L = [float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum()) for c in curves]
    closed = [bool(np.linalg.norm(c[0] - c[-1]) < WELD_MM * 4) for c in curves]
    return {"curves": len(curves), "closed": int(sum(closed)),
            "total_mm": float(sum(L)), "median_mm": float(np.median(L)),
            "points": int(sum(len(c) for c in curves))}


def demo():
    """Does this cover the surface's curvature, and what does it come out as?"""
    import pathlib as _pl

    from scipy.spatial import cKDTree

    from .dataset_curve import load_curve_parts

    R = _pl.Path(__file__).resolve().parents[4]
    MD = R / "runs" / "mesh_synth" / "parts"
    have = {p.stem for p in MD.glob("*.npz")}
    parts = [p for p in load_curve_parts(R / "runs" / "wtok_synth")
             if p.name in have][:40]

    def dens(c, step=1.0):
        c = np.asarray(c, float)
        s = np.concatenate([[0], np.cumsum(
            np.linalg.norm(np.diff(c, axis=0), axis=1))])
        if s[-1] < 1e-9:
            return c[:1]
        w = np.linspace(0, s[-1], max(int(s[-1] / step) + 2, 2))
        return np.stack([np.interp(w, s, c[:, d]) for d in range(3)], 1)

    stats, miss = [], []
    for p in parts:
        cur = mesh_bend_lines(p)
        if not cur:
            continue
        stats.append(describe(cur))
        z = np.load(MD / f"{p.name}.npz")
        span = np.maximum(z["env_hi"] - z["env_lo"], 1e-9)
        P = z["xyz"].astype(np.float64) * span + z["env_lo"]
        P = P[np.random.default_rng(0).choice(len(P), min(1500, len(P)),
                                              replace=False)]
        _, nb = cKDTree(P).query(P, k=14)
        N = np.zeros_like(P)
        for i in range(len(P)):
            w = P[nb[i]] - P[nb[i]].mean(0)
            v = np.linalg.svd(w)[2][-1]
            N[i] = -v if v[int(np.argmax(np.abs(v)))] < 0 else v
        turn = np.degrees(np.arccos(np.clip(
            np.abs(np.einsum("ij,ikj->ik", N, N[nb])), 0, 1))).mean(1)
        t = cKDTree(np.concatenate([dens(c) for c in cur]))
        far = t.query(P)[0] > 15.0
        miss.append(float(((turn > 8) & far).mean()))
    k = lambda f: np.array([s[f] for s in stats])
    print(f"{len(stats)} parts -- what they turned out to have:")
    for f in ("curves", "closed", "points"):
        v = k(f)
        print(f"  {f:<9} median {np.median(v):6.0f}   p90 {np.percentile(v,90):6.0f}"
              f"   max {v.max():6.0f}")
    print(f"  total_mm  median {np.median(k('total_mm')):6.0f}")
    m = np.array(miss)
    print(f"\ncurved surface further than 15mm from every extracted line: "
          f"median {100*np.median(m):.1f}%   p90 {100*np.percentile(m,90):.1f}%")
    assert np.median(m) < 0.02, "the extracted lines miss the part's curvature"
    print("ok")


if __name__ == "__main__":
    demo()

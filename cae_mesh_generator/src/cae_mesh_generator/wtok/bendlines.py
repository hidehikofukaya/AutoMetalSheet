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


def chain_polylines(polys, weld: float = WELD_MM, through_junctions: bool = False):
    """Join polylines that share an endpoint into maximal curves.

    A curve stops where nothing continues it, or where it closes on itself.
    Junctions where three or more curves meet are left as separate curves by
    default -- a mesher needs the junction to be a node, and forcing a choice of
    which branch continues would invent structure the part does not have.

    `through_junctions` continues anyway, taking the branch that turns least.
    That is right where the curve is known in advance to be a single loop: an
    outline touches itself at a few points (measured: 2 of 22 junctions on a
    typical part have degree 4) and stopping there splits one boundary into five
    pieces. It is wrong for bend lines, where a junction is a real node.
    """
    at = collections.defaultdict(list)

    def key(p):
        return tuple(np.round(np.asarray(p, float) / weld).astype(np.int64))

    for i, p in enumerate(polys):
        at[key(p[0])].append((i, 0))
        at[key(p[-1])].append((i, 1))

    def heading(seg, at_end):
        if len(seg) < 2:                # a one-point piece has no direction
            return np.zeros(3)
        d = (seg[-1] - seg[-2]) if at_end else (seg[0] - seg[1])
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-12 else np.zeros(3)

    used, out = set(), []
    for i0 in range(len(polys)):
        if i0 in used:
            continue
        used.add(i0)
        cur = [np.asarray(polys[i0], float)]
        for forward in (True, False):
            while True:
                seg_now = cur[-1] if forward else cur[0]
                tip = seg_now[-1] if forward else seg_now[0]
                here = at[key(tip)]
                if len(here) < 2:
                    break
                if len(here) != 2 and not through_junctions:
                    break               # a junction: a mesher needs it as a node
                nxt = [(i, e) for i, e in here if i not in used]
                if not nxt:
                    break
                if len(nxt) > 1:
                    # straightest continuation, so a self-touching outline
                    # carries on around instead of turning up a side branch
                    h = heading(seg_now, forward)
                    def align(ie):
                        s = np.asarray(polys[ie[0]], float)
                        s = s if ie[1] == 0 else s[::-1]
                        d = s[1] - s[0]
                        n = float(np.linalg.norm(d))
                        return float(h @ (d / n)) if n > 1e-12 else -1.0
                    nxt.sort(key=align, reverse=True)
                i, e = nxt[0]
                used.add(i)
                seg = np.asarray(polys[i], float)
                seg = seg if e == 0 else seg[::-1]
                if forward:
                    cur.append(seg[1:])
                else:
                    cur.insert(0, seg[:-1])
                if key(cur[-1][-1]) == key(cur[0][0]):
                    break                       # closed on itself
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
    # an outline is one loop, so walk through the points where it touches itself
    return chain_polylines(polys, through_junctions=True) if polys else []


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


def face_properties(part, root=None):
    """Per-face geometry the extractor already records and convert.py throws away.

    Two of these are ground truth for quantities that are mechanically meaningful
    and not otherwise available:

      face_geo      planar / single_curved / double_curved
                    Sheet metal is made by isometric bending, and Gauss's
                    theorema egregium then forces K=0 on the midsurface, so a
                    face is planar or single_curved. A double_curved face is
                    where that breaks -- a bead's corner sphere, formed with
                    stretch. This is the developability label, exactly.
      face_radius   the bend radius of a curved face. With the thickness it
                    gives R/t, the one absolute pass condition available in a
                    task whose distance-to-truth floor is 14.9mm.

    Estimating either from a point cloud does not work: measured over 30 parts,
    |Gaussian curvature| appears to find bend lines at AUC 0.896, but a
    developable surface has K=0 everywhere and what the estimator is picking up
    is fitting residual correlated with the larger principal curvature. The
    extractor's label is not an estimate.
    """
    d = _wireframe(getattr(part, "name", str(part)), root)
    if d is None:
        return {}
    return {
        "face_geo": d.get("face_geo") or [],
        "face_types": d.get("face_types") or [],
        "face_radius_mm": d.get("face_radius_mm") or [],
        "face_wall": d.get("face_wall") or [],
        "n_faces": d.get("n_faces", 0),
        "clusters": d.get("clusters") or [],
    }


def edge_face_context(part, root=None):
    """Each mesh bend line with the faces it separates and their radii.

    An edge in the extraction records the pair of faces it lies between, so a
    curve carries the geometry of both sides without any estimation: which of
    them is planar, and what radius the curved one has.
    """
    d = _wireframe(getattr(part, "name", str(part)), root)
    if d is None:
        return []
    out = []
    for e in d["edges"]:
        if e.get("type") not in MESH_CLASSES:
            continue
        poly = np.asarray(e["polyline"], float)
        if len(poly) < 2:
            continue
        out.append({
            "polyline": poly,
            "type": e["type"],
            "face_types": tuple(e.get("face_types") or ()),
            "face_wall": tuple(e.get("face_wall") or ()),
            "length_mm": e.get("length_mm"),
            "closed": bool(e.get("closed", False)),
        })
    return out


def developability(part, root=None):
    """What share of the part is developable, by face count and by type.

    Reported rather than assumed: the textbook statement that sheet metal is
    developable is an approximation, and the fraction that is not is the size of
    the error a developable-by-construction representation would make.
    """
    fp = face_properties(part, root)
    geo = fp.get("face_geo") or []
    if not geo:
        return {}
    c = collections.Counter(geo)
    r = [x for x in (fp.get("face_radius_mm") or []) if x]
    return {
        "n_faces": len(geo),
        "planar": c.get("planar", 0),
        "single_curved": c.get("single_curved", 0),
        "double_curved": c.get("double_curved", 0),
        "developable_frac": (c.get("planar", 0) + c.get("single_curved", 0)) / len(geo),
        "radius_min_mm": float(min(r)) if r else None,
        "radius_median_mm": float(np.median(r)) if r else None,
    }


def face_demo():
    """What the discarded fields actually contain, over many parts."""
    import pathlib as _pl

    from .dataset_curve import load_curve_parts

    R = _pl.Path(__file__).resolve().parents[4]
    have = {p.stem for p in (R / "runs" / "mesh_synth" / "parts").glob("*.npz")}
    parts = [p for p in load_curve_parts(R / "runs" / "wtok_synth")
             if p.name in have][:200]
    rows, radii, pairs = [], [], collections.Counter()
    for p in parts:
        d = developability(p)
        if d:
            rows.append(d)
            if d["radius_min_mm"]:
                radii.append(d["radius_min_mm"])
        for e in edge_face_context(p)[:400]:
            pairs[e["face_types"]] += 1
    k = lambda f: np.array([r[f] for r in rows], float)
    print(f"{len(rows)} parts")
    print(f"  faces per part          median {np.median(k('n_faces')):5.0f}")
    print(f"  planar                  median {np.median(k('planar')):5.0f}")
    print(f"  single_curved           median {np.median(k('single_curved')):5.0f}")
    print(f"  double_curved           median {np.median(k('double_curved')):5.0f}")
    print(f"  DEVELOPABLE fraction    median {np.median(k('developable_frac')):5.3f}"
          f"   min {k('developable_frac').min():.3f}")
    print(f"  smallest bend radius    median {np.median(radii):5.2f}mm"
          f"   min {min(radii):.2f}mm")
    print(f"\n  most common face pairs an edge separates:")
    for kk, v in pairs.most_common(5):
        print(f"    {str(kk):<26}{v:>7}")
    assert np.median(k("developable_frac")) > 0.5
    print("ok")

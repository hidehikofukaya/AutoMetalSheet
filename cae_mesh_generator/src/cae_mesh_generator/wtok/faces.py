"""Face boundary loops as CATIA primitives -- the two-level targets of KB 21.

The bend network is not a set of chains (7,414 of 7,901 junctions have degree
3-4, KB 20.8); it is the boundary graph of the midsurface FACES. Each edge
bounds exactly two faces, so generating one closed boundary ring per face
reconstructs the whole network structurally -- and the rings ARE the face
patch boundaries a B-rep needs.

Level 2a: which faces exist -- one slot per non-wall face (boundary centroid,
          loop-plane normal, log perimeter), perimeter-ranked. 112 slots is
          capacity (measured max 98, KB 21.5), not a count.
Level 2b: one face's boundary ring -- the outline-frame machinery at a smaller
          size (measured max 11 edges vs the outline's 24). Closed by
          construction, so the open-end snap constraint of KB 20 has nothing
          to attach to: it does not exist here.

The G1 channel carries the KB 18.8 smoothness (1 - jt/45), not a bit -- the
binary flag collapsed to the prior on the outline and there is no reason to
re-learn that lesson on faces.

Data: wireframe v4.5 (edges carry face_ids), runs/wtok_synth_v45/wireframes.
"""
from __future__ import annotations

import json
import os
import pathlib

import numpy as np
from scipy.spatial import cKDTree

from .chains import MIN_EDGE_MM, WELD_MM, _fit_edge, realize_chain
from .frame import G1_SMOOTH, _canonicalise, _edge_tangent

FACE_SLOTS = 112        # capacity: measured max 98 non-wall faces (KB 21.5)
FACE_CH = 8             # centroid 3 | loop-plane normal 3 | log-perimeter 1 | unused 1
FRING_SLOTS = 16        # capacity: measured max 11 edges/face (KB 21.5)
FRING_CH = 10           # corner 3 | bulge 3 | is_arc 1 | unused 1 | g1 1 | spare 1
FR_G1 = 8               # channel index of the smoothness value
SELF_LOOP_MM = 0.2      # an edge whose endpoints weld to the SAME node and is
                        # this short is extractor garbage (607 measured, all
                        # <= 0.05mm): it pinches the ring with a degree-4 node.
                        # A genuine closed-curve edge is orders longer.

_ROOT = pathlib.Path(__file__).resolve().parents[4]


def _wireframe(part_name: str, root=None):
    root = pathlib.Path(root or os.environ.get(
        "WTOK_FACES_WF", _ROOT / "runs" / "wtok_synth_v45" / "wireframes"))
    f = root / f"{part_name}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def face_loops(part, root=None):
    """Every non-wall face's boundary as an ordered closed ring of primitives.

    Returns (loops, stats): loops is a list of dicts
      {face_id, corners (n,3), mids (n,3), is_arc (n,), jt (n,), perimeter}
    (closed ring: n corners = n edges), stats counts the faces that could not
    be encoded -- 'unclosed' (an endpoint node without degree 2: the 0.66%
    tail of KB 21.5, a dropped degenerate/seam edge), 'multi_loop' (edges form
    more than one cycle), 'single_edge' (a lone closed curve, no corners).
    """
    d = _wireframe(getattr(part, "name", str(part)), root)
    if d is None:
        return [], {}
    wall = d["face_wall"]
    by_face: dict[int, list] = {}
    for e in d["edges"]:
        if len(e["polyline"]) < 2:
            continue
        pl = np.asarray(e["polyline"], float)
        if np.linalg.norm(np.diff(pl, axis=0), axis=1).sum() < MIN_EDGE_MM:
            continue
        for k in e["face_ids"]:
            if not wall[k]:
                by_face.setdefault(k, []).append(pl)

    loops, stats = [], {"unclosed": 0, "multi_loop": 0, "single_edge": 0,
                        "skipped_perimeter_mm": 0.0}

    def skip(kind, polys):
        stats[kind] += 1
        stats["skipped_perimeter_mm"] += float(sum(
            np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in polys))

    for k, polys in by_face.items():
        if len(polys) == 1:
            skip("single_edge", polys)
            continue
        prim = [_fit_edge(pl) for pl in polys]
        ends = np.array([q for p0, _, p1, _ in prim for q in (p0, p1)])
        tree = cKDTree(ends)
        parent = list(range(len(ends)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, b in tree.query_pairs(WELD_MM):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        node = [find(i) for i in range(len(ends))]
        live = [i for i, pl in enumerate(polys)
                if not (node[2 * i] == node[2 * i + 1]
                        and np.linalg.norm(np.diff(pl, axis=0), axis=1).sum()
                        < SELF_LOOP_MM)]
        if len(live) < 2:
            skip("single_edge", polys)
            continue
        adj: dict[int, list] = {}
        for i in live:
            adj.setdefault(node[2 * i], []).append((i, node[2 * i + 1]))
            adj.setdefault(node[2 * i + 1], []).append((i, node[2 * i]))
        if any(len(v) != 2 for v in adj.values()):
            skip("unclosed", polys)
            continue

        # walk the cycle from the first live edge
        e0 = live[0]
        seq = [(e0, node[2 * e0], node[2 * e0 + 1])]
        used = {e0}
        cur_e, cur_n = e0, node[2 * e0 + 1]
        while True:
            nxt = [(e, m) for e, m in adj[cur_n] if e != cur_e]
            e, m = nxt[0]
            if e in used:
                break
            used.add(e)
            seq.append((e, cur_n, m))
            cur_e, cur_n = e, m
        if len(seq) != len(live):
            skip("multi_loop", polys)
            continue

        corners, mids, is_arc = [], [], []
        for e, n_in, _ in seq:
            p0, m, p1, a = prim[e]
            if node[2 * e] != n_in:               # traversed backwards
                p0, p1 = p1, p0
            corners.append(p0)
            mids.append(m)
            is_arc.append(a)
        corners = np.asarray(corners)
        mids = np.asarray(mids)
        is_arc = np.asarray(is_arc, float)
        n = len(corners)
        jt = np.empty(n)
        for i in range(n):                        # junction with the PREVIOUS edge
            j = (i - 1) % n
            t_in = _edge_tangent(corners[j], mids[j], corners[i],
                                 is_arc[j] > 0.5, at_end=True)
            t_out = _edge_tangent(corners[i], mids[i], corners[(i + 1) % n],
                                  is_arc[i] > 0.5, at_end=False)
            jt[i] = np.degrees(np.arccos(np.clip(t_in @ t_out, -1, 1)))
        per = float(sum(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
                        for p in polys))
        loops.append({"face_id": k, "corners": corners, "mids": mids,
                      "is_arc": is_arc, "jt": jt, "perimeter": per})
    return loops, stats


def _cache_dir():
    d = os.environ.get("WTOK_FACES", "")
    return pathlib.Path(d) if d else None


def export_cache(out_dir, wtok="runs/wtok_synth_g1", limit=0):
    """Precompute face targets into compact npz files so a machine without the
    v4.5 wireframes (Kaggle) can train the face stages -- the chains.py move."""
    from .dataset_curve import load_curve_parts
    from .meshgen import fastener_frame

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    parts = load_curve_parts(_ROOT / wtok)
    if limit:
        parts = parts[:limit]
    n_ok = n_none = 0
    for k, p in enumerate(parts):
        f = out / f"{p.name}.npz"
        if f.exists():
            n_ok += 1
            continue
        t = face_targets(p, fastener_frame(p))
        if t is None:
            np.savez_compressed(f, none=np.array([1]))
            n_none += 1
            continue
        x2a, r2b, _ = t
        np.savez_compressed(f, x2a=x2a.astype(np.float32),
                            x2b=np.stack(r2b).astype(np.float32))
        n_ok += 1
        if (k + 1) % 200 == 0:
            print(f"{k + 1}/{len(parts)}", flush=True)
    print(f"exported {n_ok} parts ({n_none} without faces) -> {out}")


def load_cached_targets(part_name, cache=None):
    cache = cache or _cache_dir()
    if cache is None:
        return False, None
    f = pathlib.Path(cache) / f"{part_name}.npz"
    if not f.exists():
        return False, None
    d = np.load(f)
    if "none" in d:
        return True, None
    return True, (d["x2a"].astype(np.float64),
                  [x.astype(np.float64) for x in d["x2b"]], {})


def face_targets(part, frame, root=None,
                 face_slots: int = FACE_SLOTS, ring_slots: int = FRING_SLOTS):
    """(x2a, rings2b, stats) in the fastener frame, canonicalised, or None.

    x2a: (face_slots, FACE_CH), faces perimeter-ranked largest first.
    rings2b: list of (ring_slots, FRING_CH) aligned with the live rows of x2a.
    A part over either capacity is dropped and reported by the caller.
    """
    from .meshgen import to_frame

    hit, cached = load_cached_targets(getattr(part, "name", str(part)))
    if hit:
        return cached
    raw, stats = face_loops(part, root)
    if not raw:
        return None
    if len(raw) > face_slots or max(len(c["corners"]) for c in raw) > ring_slots:
        return None

    rings = []
    for c in raw:
        p, _ = to_frame(c["corners"], np.zeros_like(c["corners"]), frame)
        m, _ = to_frame(c["mids"], np.zeros_like(c["mids"]), frame)
        p, m, ia, jt = _canonicalise(p, m, c["is_arc"].copy(), c["jt"].copy())
        rings.append({"p": p, "m": m, "ia": ia, "jt": jt,
                      "per": c["perimeter"]})
    rings.sort(key=lambda c: -c["per"])

    x2a = np.zeros((face_slots, FACE_CH))
    x2a[:, FACE_CH - 1] = 1.0
    out2b = []
    for i, c in enumerate(rings):
        cen = c["p"].mean(0)
        w = c["p"] - cen
        _, _, V = np.linalg.svd(w) if len(w) > 2 else (0, 0, np.eye(3))
        v = V[-1]                                 # loop-plane normal, unoriented
        v = -v if v[int(np.argmax(np.abs(v)))] < 0 else v
        x2a[i, 0:3] = cen
        x2a[i, 3:6] = v
        x2a[i, 6] = np.log(max(c["per"], 1e-6))
        x2a[i, FACE_CH - 1] = 0.0

        n = len(c["p"])
        x = np.zeros((ring_slots, FRING_CH))
        x[:, 7] = 1.0
        x[:n, 0:3] = c["p"]
        chord = 0.5 * (c["p"] + np.roll(c["p"], -1, 0))
        x[:n, 3:6] = (c["m"] - chord) * c["ia"][:, None]
        x[:n, 6] = c["ia"]
        x[:n, 7] = 0.0
        x[:n, FR_G1] = 1.0 - np.clip(c["jt"], 0.0, 45.0) / 45.0
        out2b.append(x)
    return x2a, out2b, stats


def realize_face_ring(x, per_edge: int = 24):
    """One 2b ring back to a closed polyline (frame units)."""
    return realize_chain(x, closed=True, per_edge=per_edge, g1_thresh=G1_SMOOTH)


def demo():
    """KB 21.3 step 2: capacity, skipped-face share, and the round trip of the
    realized rings against the raw edge polylines (both resampled -- A1)."""
    from .dataset_curve import load_curve_parts
    from .meshgen import fastener_frame, to_frame

    def dens(c, step=1.0):
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
        w = np.linspace(0, s[-1], max(int(s[-1] / step) + 2, 2))
        return np.stack([np.interp(w, s, c[:, d]) for d in range(3)], 1)

    parts = load_curve_parts(_ROOT / "runs" / "wtok_synth_g1")[:60]
    ok = drop = 0
    rt, nf, skip_share = [], [], []
    tot_stats = {"unclosed": 0, "multi_loop": 0, "single_edge": 0}
    for p in parts:
        fr = fastener_frame(p)
        t = face_targets(p, fr)
        if t is None:
            drop += 1
            continue
        x2a, rings2b, stats = t
        ok += 1
        nf.append(len(rings2b))
        for k in tot_stats:
            tot_stats[k] += stats[k]
        got = [realize_face_ring(x) for x in rings2b]
        got = [g for g in got if g is not None]

        d = _wireframe(p.name)
        wall = d["face_wall"]
        raw = [np.asarray(e["polyline"], float) for e in d["edges"]
               if any(not wall[k] for k in e["face_ids"])
               and len(e["polyline"]) >= 2]
        per_all = sum(np.linalg.norm(np.diff(c, axis=0), axis=1).sum() for c in raw)
        skip_share.append(stats["skipped_perimeter_mm"] / max(per_all, 1e-9))
        if not got or not raw:
            continue
        u = fr[2]
        A = np.concatenate([dens(c, step=1.0 / u) for c in got])
        B = np.concatenate([to_frame(dens(c), np.zeros((len(dens(c)), 3)), fr)[0]
                            for c in raw])
        rt.append(u * 0.5 * (cKDTree(B).query(A)[0].mean()
                             + cKDTree(A).query(B)[0].mean()))
    rt = np.array(rt)
    print(f"{ok} parts encoded, {drop} over capacity / without loops "
          f"({100 * drop / max(ok + drop, 1):.0f}%)")
    print(f"faces/part median {np.median(nf):.0f}  max {max(nf)}")
    print(f"skipped faces: {tot_stats}  perimeter share "
          f"median {100 * np.median(skip_share):.2f}%  "
          f"max {100 * max(skip_share):.2f}%")
    print(f"round trip vs raw polylines: median {np.median(rt):.3f}mm  "
          f"p90 {np.percentile(rt, 90):.3f}mm  max {rt.max():.3f}mm")
    assert np.median(rt) < 0.6, "round trip worse than the outline's 0.45mm level"
    print("ok")


if __name__ == "__main__":
    demo()

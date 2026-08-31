"""Bend chains as CATIA primitives -- the two-level targets of KB 20.

A part's bend lines and bead ridges are a SET OF CHAINS. Each chain is an
ordered run of B-rep edges; measured over 6674 edges every one of them is a
single LINE or ARC (residual ~0 at 0.25t), so the edge is the slot, exactly as
the outline learned in KB 18.

Level 2a: which chains exist -- one slot per chain (centroid, direction,
          length, closed), length-ranked. 96 slots is capacity (measured max
          90), not a count.
Level 2b: what one chain looks like -- one slot per corner, edges implied by
          slot adjacency, arcs as bulges, G1 flags on junctions. The chain
          boundary needs no flag at all: a chain IS its own tensor, so a
          missed delimiter cannot join two curves (the corner-token failure,
          KB 17.5). Geometric delimiters were measured impossible first:
          distinct endpoints sit 0.2t apart and junctions share points.

Open-chain ends lie on the outline for 100% of 474 measured ends, so snapping
them to the generated outline at sampling time is an input-derived constraint
with the same status as the fastener seat.
"""
from __future__ import annotations

import json
import os
import pathlib

import numpy as np
from scipy.spatial import cKDTree

from .frame import _bulge_from_tangent, _canonicalise, _edge_tangent

CHAIN_SLOTS = 96        # capacity: measured max 90 chains/part over 200 parts
CHAIN_CH = 9            # centroid 3 | direction 3 | log-length 1 | closed 1 | unused 1
CEDGE_SLOTS = 40        # capacity: measured max 40 edges/chain
CEDGE_CH = 10           # corner 3 | bulge 3 | is_arc 1 | unused 1 | g1 1 | spare 1
CE_G1 = 8               # channel index of the G1 flag
G1_DEG = 3.0            # same junction threshold as the outline (KB 18)
WELD_MM = 0.05          # cluster weld; grid rounding broke a loop at 0.008mm
MIN_EDGE_MM = 0.01      # the extractor emits 0.008mm garbage edges (KB 18.5)
CLASSES = ("bend_line", "crease")

_ROOT = pathlib.Path(__file__).resolve().parents[4]


def _wireframe(part_name: str, root=None):
    root = pathlib.Path(root or os.environ.get(
        "WTOK_WIREFRAMES", _ROOT / "runs" / "wtok_synth" / "wireframes"))
    f = root / f"{part_name}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def _fit_edge(pl):
    """One CATIA edge -> (p0, mid, p1, is_arc). Measured: always one primitive."""
    a, b = pl[0], pl[-1]
    ch = b - a
    L = np.linalg.norm(ch)
    if L < 1e-9 or len(pl) < 3:
        return a, 0.5 * (a + b), b, False
    w = pl - a
    d = np.linalg.norm(w - np.outer(w @ ch / L, ch / L), axis=1)
    if d.max() < WELD_MM:
        return a, 0.5 * (a + b), b, False
    return a, pl[int(np.argmax(d))], b, True


def chain_primitives(part, root=None):
    """The part's bend chains in mm: list of dicts
       {closed, corners (n,3), mids (n_e,3), is_arc (n_e,), jt (n,)}
    where jt is the junction tangent angle at each corner in degrees
    (nan where there is no junction: open-chain ends)."""
    d = _wireframe(getattr(part, "name", str(part)), root)
    if d is None:
        return []
    E, prim = [], []
    for e in d["edges"]:
        if e["type"] not in CLASSES or len(e["polyline"]) < 2:
            continue
        pl = np.asarray(e["polyline"], float)
        if np.linalg.norm(np.diff(pl, axis=0), axis=1).sum() < MIN_EDGE_MM:
            continue
        E.append(pl)
        prim.append(_fit_edge(pl))
    if not E:
        return []
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
    adj: dict = {}
    for i in range(len(E)):
        n0, n1 = node[2 * i], node[2 * i + 1]
        adj.setdefault(n0, []).append((i, n1))
        adj.setdefault(n1, []).append((i, n0))

    used, out = set(), []
    for i in range(len(E)):
        if i in used:
            continue
        used.add(i)
        seq = [(i, node[2 * i], node[2 * i + 1])]     # (edge, entry node, exit node)
        for grow_front in (False, True):
            cur_e, cur_n = (seq[-1][0], seq[-1][2]) if grow_front else (seq[0][0], seq[0][1])
            while len(adj[cur_n]) == 2:
                nxt = [(e, m) for e, m in adj[cur_n] if e != cur_e]
                if not nxt or nxt[0][0] in used:
                    break
                e, m = nxt[0]
                used.add(e)
                if grow_front:
                    seq.append((e, cur_n, m))
                else:
                    seq.insert(0, (e, m, cur_n))
                cur_e, cur_n = e, m
        closed = seq[0][1] == seq[-1][2] and len(seq) > 1
        corners, mids, is_arc = [], [], []
        for e, n_in, n_out in seq:
            p0, m, p1, a = prim[e]
            if node[2 * e] != n_in:                   # traversed backwards
                p0, p1 = p1, p0
            corners.append(p0)
            mids.append(m)
            is_arc.append(a)
        if not closed:
            e, n_in, n_out = seq[-1]
            p0, _, p1, _ = prim[e]
            corners.append(p1 if node[2 * e] == n_in else p0)
        corners = np.asarray(corners)
        mids = np.asarray(mids)
        is_arc = np.asarray(is_arc, float)
        n, n_e = len(corners), len(mids)
        jt = np.full(n, np.nan)
        for k in range(n):
            if not closed and (k == 0 or k == n - 1):
                continue
            j = (k - 1) % n_e
            t_in = _edge_tangent(corners[j], mids[j],
                                 corners[(j + 1) % n], is_arc[j] > 0.5, at_end=True)
            t_out = _edge_tangent(corners[k % n_e], mids[k % n_e],
                                  corners[(k + 1) % n], is_arc[k % n_e] > 0.5,
                                  at_end=False)
            jt[k] = np.degrees(np.arccos(np.clip(t_in @ t_out, -1, 1)))
        out.append({"closed": bool(closed), "corners": corners,
                    "mids": mids, "is_arc": is_arc, "jt": jt})
    return out


def _arclen(c):
    return float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())


def _cache_dir():
    d = os.environ.get("WTOK_CHAINS", "")
    return pathlib.Path(d) if d else None


def export_cache(out_dir, wtok="runs/wtok_synth_g1", limit=0):
    """Precompute chain targets for every part into compact npz files, so a
    machine without the 244MB wireframes (Kaggle) can train the chain stages.
    One file per part: the live 2a rows, the stacked 2b tensors, closed flags."""
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
        t = chain_targets(p, fastener_frame(p))
        if t is None:
            np.savez_compressed(f, none=np.array([1]))
            n_none += 1
            continue
        x2a, c2b = t
        np.savez_compressed(
            f, x2a=x2a.astype(np.float32),
            x2b=np.stack([x for x, _ in c2b]).astype(np.float32),
            closed=np.array([c for _, c in c2b], np.uint8))
        n_ok += 1
        if (k + 1) % 200 == 0:
            print(f"{k + 1}/{len(parts)}", flush=True)
    print(f"exported {n_ok} parts ({n_none} without chains) -> {out}")


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
    x2a = d["x2a"].astype(np.float64)
    c2b = [(x.astype(np.float64), bool(c)) for x, c in zip(d["x2b"], d["closed"])]
    return True, (x2a, c2b)


def chain_targets(part, frame, root=None,
                  chain_slots: int = CHAIN_SLOTS, edge_slots: int = CEDGE_SLOTS):
    """(x2a, chains2b) in the fastener frame, canonicalised, or None.

    x2a: (chain_slots, CHAIN_CH), chains length-ranked longest first.
    chains2b: list of (x2b (edge_slots, CEDGE_CH), closed) aligned with the
    live rows of x2a. A part whose chains exceed either capacity is dropped
    and reported by the caller -- capacity, not a count.
    """
    from .meshgen import to_frame

    hit, cached = load_cached_targets(getattr(part, "name", str(part)))
    if hit:
        return cached
    raw = chain_primitives(part, root)
    if not raw:
        return None
    if len(raw) > chain_slots or max(len(c["mids"]) for c in raw) + 1 > edge_slots:
        return None
    ch = []
    for c in raw:
        p, _ = to_frame(c["corners"], np.zeros_like(c["corners"]), frame)
        m, _ = to_frame(c["mids"], np.zeros_like(c["mids"]), frame)
        ia, jt = c["is_arc"].copy(), c["jt"].copy()
        if c["closed"]:
            p, m, ia, jt = _canonicalise(p, m, ia, jt)
        elif tuple(np.round(p[0], 6)) < tuple(np.round(p[-1], 6)):
            p = p[::-1].copy()
            m = m[::-1].copy()
            ia = ia[::-1].copy()
            jt = jt[::-1].copy()
        ch.append({"closed": c["closed"], "p": p, "m": m, "ia": ia, "jt": jt,
                   "len": _arclen(np.vstack([p, p[:1]]) if c["closed"] else p)})
    ch.sort(key=lambda c: -c["len"])

    x2a = np.zeros((chain_slots, CHAIN_CH))
    x2a[:, CHAIN_CH - 1] = 1.0
    out2b = []
    for i, c in enumerate(ch):
        cen = c["p"].mean(0)
        w = c["p"] - cen
        _, _, V = np.linalg.svd(w) if len(w) > 1 else (0, 0, np.eye(3))
        v = V[0]
        v = -v if v[int(np.argmax(np.abs(v)))] < 0 else v
        x2a[i, 0:3] = cen
        x2a[i, 3:6] = v
        x2a[i, 6] = np.log(max(c["len"], 1e-6))
        x2a[i, 7] = 1.0 if c["closed"] else 0.0
        x2a[i, CHAIN_CH - 1] = 0.0

        x = np.zeros((edge_slots, CEDGE_CH))
        x[:, 7] = 1.0
        n, n_e = len(c["p"]), len(c["m"])
        x[:n, 0:3] = c["p"]
        nxt = np.roll(c["p"], -1, 0)[:n_e] if c["closed"] else c["p"][1:n_e + 1]
        chord = 0.5 * (c["p"][:n_e] + nxt)
        x[:n_e, 3:6] = (c["m"] - chord) * c["ia"][:n_e, None]
        x[:n_e, 6] = c["ia"][:n_e]
        x[:n, 7] = 0.0
        x[:n, CE_G1] = np.where(np.isfinite(c["jt"]) & (c["jt"] < G1_DEG), 1.0, 0.0)
        out2b.append((x, c["closed"]))
    return x2a, out2b


def enforce_g1_chain(p, bulge, is_arc, g1, closed):
    """frame.enforce_g1 for a chain: open chains have n-1 edges and no wrap.
    Same local solve, same blow-up guard (a junction whose solution needs a
    sagitta beyond half the chord is a conflict, not a fix)."""
    n = len(p)
    n_e = n if closed else n - 1
    nxt = np.roll(p, -1, 0)
    mid = 0.5 * (p + nxt) + np.vstack([bulge, np.zeros((n - len(bulge), 3))])[:n]
    want_s = [None] * n_e
    want_e = [None] * n_e
    residual = 0.0
    for i in range(n):
        if not g1[i] or (not closed and (i == 0 or i >= n_e)):
            continue
        j = (i - 1) % n_e
        t_in = _edge_tangent(p[j], mid[j], nxt[j], is_arc[j] > 0.5, at_end=True)
        t_out = _edge_tangent(p[i % n_e], mid[i % n_e], nxt[i % n_e],
                              is_arc[i % n_e] > 0.5, at_end=False)
        if is_arc[j] < 0.5 and is_arc[i % n_e] < 0.5:
            residual += np.degrees(np.arccos(np.clip(t_in @ t_out, -1, 1)))
            continue
        t = t_in if is_arc[j] < 0.5 else (t_out if is_arc[i % n_e] < 0.5
                                          else t_in + t_out)
        nt = np.linalg.norm(t)
        if nt < 1e-6:
            continue
        t = t / nt
        if is_arc[i % n_e] > 0.5:
            want_s[i % n_e] = t
        if is_arc[j] > 0.5:
            want_e[j] = t
    out = bulge.copy()
    for k in range(n_e):
        if want_s[k] is None and want_e[k] is None:
            continue
        cand = []
        for want, at_end in ((want_s[k], False), (want_e[k], True)):
            if want is None:
                continue
            b = _bulge_from_tangent(p[k], nxt[k], want, at_end)
            if b is not None:
                cand.append(b)
        if not cand:
            continue
        b = np.mean(cand, axis=0)
        if np.linalg.norm(b) > 0.5 * np.linalg.norm(nxt[k] - p[k]):
            residual += 90.0
            continue
        out[k] = b
    return out, float(residual)


def realize_chain(x, closed, per_edge: int = 24):
    """One 2b tensor back to a polyline (frame units)."""
    from .codec import realize_edge

    live = x[:, 7] < 0.5
    p = x[live, 0:3]
    n = len(p)
    if n < 2:
        return None
    n_e = n if closed else n - 1
    bulge = x[live, 3:6][:n_e]
    is_arc = x[live, 6][:n_e]
    g1 = x[live, CE_G1] > 0.5
    bulge, _ = enforce_g1_chain(p, bulge, is_arc, g1, closed)
    pts = []
    for i in range(n_e):
        a, b = p[i], p[(i + 1) % n]
        seg = None
        if is_arc[i] > 0.5:
            m = 0.5 * (a + b) + bulge[i]
            try:
                seg = realize_edge(a, m, b, per_edge)
            except Exception:
                seg = None
        if seg is None:
            seg = np.linspace(a, b, per_edge)
        pts.append(seg if not pts else seg[1:])
    poly = np.concatenate(pts)
    return np.vstack([poly, poly[:1]]) if closed else poly


def realize_chains(x2a, chains2b, per_edge: int = 24):
    """All chains of a part; the 2a closed flag picks each chain's topology."""
    out = []
    for (x, _), row in zip(chains2b, x2a):
        c = realize_chain(x, closed=row[7] > 0.5, per_edge=per_edge)
        if c is not None:
            out.append(c)
    return out


def demo():
    """Coverage, capacity and the round trip against the raw polylines."""
    from .bendlines import mesh_bend_lines
    from .meshgen import fastener_frame
    from .dataset_curve import load_curve_parts

    def dens(c, step=1.0):
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
        w = np.linspace(0, s[-1], max(int(s[-1] / step) + 2, 2))
        return np.stack([np.interp(w, s, c[:, d]) for d in range(3)], 1)

    parts = load_curve_parts(_ROOT / "runs" / "wtok_synth_g1")[:60]
    ok = drop = 0
    rt, nch = [], []
    for p in parts:
        fr = fastener_frame(p)
        t = chain_targets(p, fr)
        if t is None:
            drop += 1
            continue
        x2a, c2b = t
        ok += 1
        nch.append(len(c2b))
        got = realize_chains(x2a, c2b, per_edge=24)
        raw = [np.asarray(c) for c in _raw_polys(p)]
        if not got or not raw:
            continue
        from .meshgen import to_frame
        u = fr[2]
        A = np.concatenate([dens(c, step=1.0 / u) for c in got])
        B = np.concatenate([to_frame(dens(c), np.zeros((len(dens(c)), 3)), fr)[0]
                            for c in raw])
        rt.append(u * 0.5 * (cKDTree(B).query(A)[0].mean()
                             + cKDTree(A).query(B)[0].mean()))
    rt = np.array(rt)
    print(f"{ok} parts encoded, {drop} over capacity "
          f"({100 * drop / max(ok + drop, 1):.0f}%)")
    print(f"chains/part median {np.median(nch):.0f}  max {max(nch)}")
    print(f"round trip vs raw polylines: median {np.median(rt):.3f}mm  "
          f"p90 {np.percentile(rt, 90):.3f}mm  max {rt.max():.3f}mm")
    assert np.median(rt) < 0.6, "round trip worse than the outline's 0.45mm level"
    print("ok")


def _raw_polys(part, root=None):
    d = _wireframe(getattr(part, "name", str(part)), root)
    if d is None:
        return []
    return [np.asarray(e["polyline"], float) for e in d["edges"]
            if e["type"] in CLASSES and len(e["polyline"]) >= 2
            and np.linalg.norm(np.diff(np.asarray(e["polyline"], float),
                                       axis=0), axis=1).sum() >= MIN_EDGE_MM]


if __name__ == "__main__":
    demo()

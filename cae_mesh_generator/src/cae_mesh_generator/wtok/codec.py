"""Quantization, canonical ordering, sequence codec sigma/delta and curve
realization for the frozen theory spec (final_wireframe_theory.md §3-4).

Round-trip property (proposition F1): delta(sigma(W), C) == W on the quantized
representation. Verified per part by build_dataset.py.
"""
from __future__ import annotations

import math

import numpy as np

from .constants import BITS, HI_BITS, E_ARITY, SparseW

LO_MASK = (1 << HI_BITS) - 1
T_ORDER = {"FIX": 0, "END": 1, "MID": 2}
TAU_ORDER = {"LINE": 0, "ARC": 1, "CIRCLE": 2, "CIRCLE_C": 3}


def envelope(W: SparseW, margin: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    pts = np.stack([v["xyz"] for v in W.vertices])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    return lo - margin * span, hi + margin * span


def quantize_W(W: SparseW) -> dict:
    """Quantize, merge same (T,bin), canonicalize order. Returns the frozen
    discrete form: {env, vertices: [(T, cz, cy, cx, nf)], edges: [(tau, refs, cls)]}."""
    lo, hi = envelope(W)
    span = hi - lo
    n_bins = 1 << BITS

    def qbin(xyz):
        c = np.floor((xyz - lo) / span * n_bins).astype(np.int64)
        return tuple(int(x) for x in np.clip(c, 0, n_bins - 1))

    merged: dict = {}
    new_vertices = []
    remap = {}
    for i, v in enumerate(W.vertices):
        key = (v["T"], qbin(v["xyz"]))
        if key not in merged:
            merged[key] = len(new_vertices)
            nf = None if v["nf"] is None else [round(float(x), 6) for x in v["nf"]]
            new_vertices.append({"T": v["T"], "bin": key[1], "nf": nf})
        remap[i] = merged[key]

    # canonical vertex order: FIX block first, then (T, z, y, x) lexicographic
    order = sorted(range(len(new_vertices)),
                   key=lambda k: (T_ORDER[new_vertices[k]["T"]], *new_vertices[k]["bin"]))
    pos = {old: new for new, old in enumerate(order)}
    vertices = [new_vertices[old] for old in order]
    remap = {i: pos[m] for i, m in remap.items()}

    edges = []
    seen = set()
    for e in W.edges:
        refs = [remap[r] for r in e["refs"]]
        tau = e["tau"]
        if tau == "LINE":
            if refs[0] == refs[1]:
                continue
            refs = sorted(refs)
        elif tau == "ARC":
            if refs[0] == refs[2]:
                continue
            if refs[1] in (refs[0], refs[2]):
                tau, refs = "LINE", sorted([refs[0], refs[2]])
            else:
                refs = [min(refs[0], refs[2]), refs[1], max(refs[0], refs[2])]
        elif tau == "CIRCLE":
            if len(set(refs)) < 3:
                continue
            refs = sorted(refs)
        elif tau == "CIRCLE_C":
            if refs[0] == refs[1]:
                continue
        sig = (tau, tuple(refs))
        if sig in seen:
            continue
        seen.add(sig)
        edges.append({"tau": tau, "refs": refs, "cls": e["cls"]})
    # prune orphan non-FIX vertices (left behind by dropped degenerate edges;
    # they contribute nothing to realization and break curve-major round trips)
    used = {r for e in edges for r in e["refs"]}
    keep = [i for i, v in enumerate(vertices) if v["T"] == "FIX" or i in used]
    pos2 = {old: new for new, old in enumerate(keep)}
    vertices = [vertices[i] for i in keep]
    for e in edges:
        e["refs"] = [pos2[r] for r in e["refs"]]

    edges.sort(key=lambda e: (min(e["refs"]), max(e["refs"]),
                              TAU_ORDER[e["tau"]], tuple(e["refs"])))
    return {"env_lo": lo.tolist(), "env_hi": hi.tolist(), "vertices": vertices,
            "edges": edges}


# ---------------------------------------------------------------- sigma/delta

def sigma(Q: dict) -> dict:
    """Serialize the quantized W. FIX vertices are condition (not generated):
    the vertex token block covers non-FIX vertices only."""
    vt, coords = [], []
    for v in Q["vertices"]:
        if v["T"] == "FIX":
            continue
        vt.append(v["T"])
        c = []
        for axis in range(3):
            b = v["bin"][axis]
            c += [b >> HI_BITS, b & LO_MASK]
        coords.append(c)
    return {"vertex_types": vt, "vertex_coords": coords,
            "edge_types": [e["tau"] for e in Q["edges"]],
            "edge_refs": [list(e["refs"]) for e in Q["edges"]],
            "edge_cls": [e["cls"] for e in Q["edges"]]}


def delta(seq: dict, fix_vertices: list[dict], env_lo, env_hi) -> dict:
    vertices = list(fix_vertices)
    for t, c in zip(seq["vertex_types"], seq["vertex_coords"]):
        b = tuple((c[2 * a] << HI_BITS) | c[2 * a + 1] for a in range(3))
        vertices.append({"T": t, "bin": b, "nf": None})
    edges = [{"tau": t, "refs": r, "cls": c} for t, r, c in
             zip(seq["edge_types"], seq["edge_refs"], seq["edge_cls"])]
    return {"env_lo": env_lo, "env_hi": env_hi, "vertices": vertices, "edges": edges}


def roundtrip_ok(Q: dict) -> bool:
    fix = [v for v in Q["vertices"] if v["T"] == "FIX"]
    R = delta(sigma(Q), fix, Q["env_lo"], Q["env_hi"])
    if [(v["T"], v["bin"]) for v in R["vertices"]] != \
       [(v["T"], v["bin"]) for v in Q["vertices"]]:
        return False
    return [(e["tau"], tuple(e["refs"])) for e in R["edges"]] == \
           [(e["tau"], tuple(e["refs"])) for e in Q["edges"]]


def n_tokens(Q: dict) -> int:
    n_v = sum(1 for v in Q["vertices"] if v["T"] != "FIX")
    n_e = sum(1 + E_ARITY[e["tau"]] for e in Q["edges"])
    return n_v * 7 + n_e + 2  # +stop tokens for both blocks


# ---------------------------------------------------------------- realization

def bin_center(Q: dict, v: dict) -> np.ndarray:
    lo = np.asarray(Q["env_lo"])
    span = np.asarray(Q["env_hi"]) - lo
    return lo + (np.asarray(v["bin"], dtype=np.float64) + 0.5) / (1 << BITS) * span


def circle_through(p0, p1, p2):
    """Circumcircle of 3 points. Returns (center, normal, radius) or None."""
    n = np.cross(p1 - p0, p2 - p0)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return None
    n /= nn
    # solve for center in the plane: |c-p0|=|c-p1|=|c-p2|
    A = np.stack([2 * (p1 - p0), 2 * (p2 - p0), n])
    b = np.array([p1 @ p1 - p0 @ p0, p2 @ p2 - p0 @ p0, n @ p0])
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return c, n, float(np.linalg.norm(p0 - c))


def realize_edge(Q: dict, e: dict, samples: int = 33) -> np.ndarray | None:
    P = [bin_center(Q, Q["vertices"][r]) for r in e["refs"]]
    if e["tau"] == "LINE":
        return np.stack([P[0], P[1]])
    if e["tau"] == "ARC":
        cc = circle_through(P[0], P[1], P[2])
        if cc is None:
            return np.stack([P[0], P[2]])
        c, n, r = cc
        e1 = (P[0] - c) / max(np.linalg.norm(P[0] - c), 1e-12)
        e2 = np.cross(n, e1)
        a_mid = math.atan2((P[1] - c) @ e2, (P[1] - c) @ e1) % (2 * math.pi)
        a_end = math.atan2((P[2] - c) @ e2, (P[2] - c) @ e1) % (2 * math.pi)
        sweep = a_end if a_mid <= a_end else a_end - 2 * math.pi
        ts = np.linspace(0.0, sweep, samples)
        return c + r * (np.cos(ts)[:, None] * e1 + np.sin(ts)[:, None] * e2)
    if e["tau"] == "CIRCLE":
        cc = circle_through(P[0], P[1], P[2])
        if cc is None:
            return None
        c, n, r = cc
    else:  # CIRCLE_C: center FIX (holds axis nf), radius from projected point
        vc = Q["vertices"][e["refs"][0]]
        if vc.get("nf") is None:
            return None
        c = bin_center(Q, vc)
        n = np.asarray(vc["nf"], dtype=np.float64)
        n /= max(np.linalg.norm(n), 1e-12)
        d = P[1] - c
        d_in = d - n * (d @ n)
        r = float(np.linalg.norm(d_in))
        if r < 1e-9:
            return None
    e1 = np.cross(n, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(n, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    ts = np.linspace(0.0, 2 * math.pi, 2 * samples)
    return c + r * (np.cos(ts)[:, None] * e1 + np.sin(ts)[:, None] * e2)


def realize_points(Q: dict, step_mm: float = 2.0) -> np.ndarray:
    out = []
    for e in Q["edges"]:
        poly = realize_edge(Q, e)
        if poly is None or len(poly) < 2:
            continue
        seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        length = float(seg.sum())
        n = max(2, int(length / step_mm) + 1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        tt = np.linspace(0.0, cum[-1], n)
        idx = np.clip(np.searchsorted(cum, tt) - 1, 0, len(seg) - 1)
        frac = (tt - cum[idx]) / np.maximum(seg[idx], 1e-12)
        out.append(poly[idx] + (poly[idx + 1] - poly[idx]) * frac[:, None])
    return np.concatenate(out) if out else np.zeros((0, 3))

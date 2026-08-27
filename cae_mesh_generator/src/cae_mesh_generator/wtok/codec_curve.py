"""Curve-major sequence codec (definition 6', wireframe_theory_rev_curve_major.md).

sigma_c: edges in canonical order; per edge [tau][slots], each slot POINTER(k)
to an already-materialized vertex (FIX block + earlier NEW vertices, in
emission order) or NEW + 6 coarse/fine coordinate tokens (vertex type is
implied by the slot position).

Round trip compares geometric signatures (vertex (T,bin) sets and edge
multisets) because vertex indices are order-dependent by design.
"""
from __future__ import annotations

from .constants import HI_BITS, E_ARITY

LO_MASK = (1 << HI_BITS) - 1

# slot type templates per edge type (FIX slots are pointer-only)
SLOT_TYPES = {
    "LINE": ("END", "END"),
    "ARC": ("END", "MID", "END"),
    "CIRCLE": ("END", "END", "END"),
    "CIRCLE_C": ("FIX", "END"),
}


def _endpoints(e: dict) -> tuple:
    if e["tau"] == "LINE":
        return (e["refs"][0], e["refs"][1])
    if e["tau"] == "ARC":
        return (e["refs"][0], e["refs"][2])
    return ()  # circles are closed: no open endpoints


def traversal_order(edges: list[dict]) -> list[dict]:
    """Deterministic connected-walk ordering (revision of definition 6'):
    consecutive edges share endpoints wherever the graph allows, so 'point at
    the previous wire's end' becomes the dominant pointer pattern. Determinism
    (component order, walk starts, tie-breaks all by canonical edge key)
    preserves the injectivity required by F1."""
    n = len(edges)

    def ekey(k: int):
        e = edges[k]
        return (min(e["refs"]), max(e["refs"]), e["tau"], tuple(e["refs"]))

    incident: dict[int, list[int]] = {}
    for k, e in enumerate(edges):
        for v in _endpoints(e):
            incident.setdefault(v, []).append(k)

    # connected components over endpoint-sharing
    comp = list(range(n))

    def find(a):
        while comp[a] != a:
            comp[a] = comp[comp[a]]
            a = comp[a]
        return a

    for eids in incident.values():
        for k in eids[1:]:
            comp[find(k)] = find(eids[0])
    groups: dict[int, list[int]] = {}
    for k in range(n):
        groups.setdefault(find(k), []).append(k)

    order: list[int] = []
    for members in sorted(groups.values(), key=lambda ms: min(ekey(k) for k in ms)):
        unvis = set(members)
        cur: int | None = None
        while unvis:
            cands = [k for k in incident.get(cur, []) if k in unvis] if cur is not None else []
            if not cands:
                k = min(unvis, key=ekey)  # restart at the smallest unvisited edge
            else:
                k = min(cands, key=ekey)
            order.append(k)
            unvis.discard(k)
            eps = _endpoints(edges[k])
            if not eps:
                cur = None
            elif cur is None or cur not in eps:
                cur = eps[1]
            else:
                cur = eps[1] if eps[0] == cur else eps[0]
    return [edges[k] for k in order]


def sigma_curve(Q: dict) -> dict:
    """Serialize edges in their canonical order with first-occurrence NEW rule."""
    fix_indices = [i for i, v in enumerate(Q["vertices"]) if v["T"] == "FIX"]
    fix_pos = {orig: k for k, orig in enumerate(fix_indices)}
    materialized: dict[int, int] = dict(fix_pos)  # original idx -> pointer id
    n_mat = len(fix_indices)
    edges_out = []
    for e in Q["edges"]:
        slots = []
        for slot_t, orig in zip(SLOT_TYPES[e["tau"]], e["refs"]):
            if orig in materialized:
                slots.append({"kind": "ptr", "id": materialized[orig]})
            else:
                assert slot_t != "FIX", "FIX slots must always resolve to pointers"
                v = Q["vertices"][orig]
                assert v["T"] == slot_t, f"slot type mismatch {v['T']} vs {slot_t}"
                coords = []
                for axis in range(3):
                    b = v["bin"][axis]
                    coords += [b >> HI_BITS, b & LO_MASK]
                slots.append({"kind": "new", "coords": coords})
                materialized[orig] = n_mat
                n_mat += 1
        edges_out.append({"tau": e["tau"], "slots": slots, "cls": e["cls"]})
    return {"edges": edges_out, "n_fix": len(fix_indices)}


def delta_curve(seq: dict, fix_vertices: list[dict], env_lo, env_hi) -> dict:
    """Rebuild a quantized W from the curve-major sequence + FIX condition.
    Vertices carry emission order; same-(T,bin) NEW vertices merge on decode."""
    vertices = [dict(v) for v in fix_vertices]
    bin_index: dict[tuple, int] = {}
    for i, v in enumerate(vertices):
        bin_index[(v["T"], tuple(v["bin"]))] = i
    edges = []
    for e in seq["edges"]:
        refs = []
        for slot_t, s in zip(SLOT_TYPES[e["tau"]], e["slots"]):
            if s["kind"] == "ptr":
                refs.append(s["id"])
            else:
                c = s["coords"]
                b = tuple((c[2 * a] << HI_BITS) | c[2 * a + 1] for a in range(3))
                key = (slot_t, b)
                if key in bin_index:  # decode-time merge (definition 6 rule)
                    refs.append(bin_index[key])
                else:
                    vertices.append({"T": slot_t, "bin": b, "nf": None})
                    bin_index[key] = len(vertices) - 1
                    refs.append(len(vertices) - 1)
        edges.append({"tau": e["tau"], "refs": refs, "cls": e["cls"]})
    # decode-time duplicate-edge removal (guarantee demoted per revision §3)
    seen, uniq = set(), []
    for e in edges:
        sig = (e["tau"], tuple(sorted(e["refs"])) if e["tau"] != "ARC"
               else (min(e["refs"][0], e["refs"][2]), e["refs"][1],
                     max(e["refs"][0], e["refs"][2])))
        if sig not in seen:
            seen.add(sig)
            uniq.append(e)
    return {"env_lo": env_lo, "env_hi": env_hi, "vertices": vertices, "edges": uniq}


def geometric_signature(Q: dict) -> tuple:
    """Order-independent signature: vertex set + edge multiset by geometry."""
    def vkey(i):
        v = Q["vertices"][i]
        return (v["T"], tuple(v["bin"]))

    verts = frozenset(vkey(i) for i in range(len(Q["vertices"])))
    edges = sorted(
        (e["tau"], tuple(sorted(vkey(r) for r in e["refs"])))
        for e in Q["edges"]
    )
    return verts, tuple(edges)


def roundtrip_curve_ok(Q: dict) -> bool:
    fix = [v for v in Q["vertices"] if v["T"] == "FIX"]
    R = delta_curve(sigma_curve(Q), fix, Q["env_lo"], Q["env_hi"])
    return geometric_signature(Q) == geometric_signature(R)


def n_tokens_curve(seq: dict) -> int:
    n = 1  # stop
    for e in seq["edges"]:
        n += 1  # tau
        for s in e["slots"]:
            n += 1 if s["kind"] == "ptr" else 1 + 6  # selector (+ coords)
    return n

"""Phase 0 CLI: convert all parts to the frozen sparse representation, verify
the sigma/delta round trip, measure conversion fidelity (Chamfer vs the source
v4.4 polylines), and export viewer JSONs.

Usage:
  python -m cae_mesh_generator.wtok.build_dataset --output-dir ../runs/wtok_dataset
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from ..wireflow.dataset import ASSEMBLIES, DEFAULT_BASE, load_joints, part_id_from_stem
from .convert import SCOPE_CLASSES, convert_part
from .codec import n_tokens, quantize_W, realize_edge, realize_points, roundtrip_ok, sigma

VIEWER_COLOR_CLS = {"outer_boundary", "hole_boundary", "bend_line"}


def source_scope_points(wf_json: pathlib.Path, step_mm: float = 2.0) -> np.ndarray:
    data = json.loads(wf_json.read_text(encoding="utf-8"))
    ov = wf_json.with_name(wf_json.stem + ".overrides.json")
    overrides = json.loads(ov.read_text(encoding="utf-8")) if ov.exists() else {}
    pts = []
    for e in data["edges"]:
        cls = overrides.get(e.get("fingerprint"), e["type"])
        if cls not in SCOPE_CLASSES:
            continue
        poly = np.asarray(e["polyline"])
        seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        if seg.sum() < 1e-9:
            continue
        n = max(2, int(seg.sum() / step_mm) + 1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        tt = np.linspace(0.0, cum[-1], n)
        idx = np.clip(np.searchsorted(cum, tt) - 1, 0, len(seg) - 1)
        frac = (tt - cum[idx]) / np.maximum(seg[idx], 1e-12)
        pts.append(poly[idx] + (poly[idx + 1] - poly[idx]) * frac[:, None])
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def chamfer(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    # chunked to keep memory bounded
    def one_way(x, y):
        mins = []
        for i in range(0, len(x), 2048):
            d = np.linalg.norm(x[i:i + 2048, None, :] - y[None, :, :], axis=-1)
            mins.append(d.min(axis=1))
        m = np.concatenate(mins)
        return float(m.mean()), float(np.percentile(m, 95))
    m_ab, p_ab = one_way(a, b)
    m_ba, p_ba = one_way(b, a)
    return m_ab + m_ba, max(p_ab, p_ba)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=str(DEFAULT_BASE))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--assemblies", default=",".join(ASSEMBLIES),
                    help="comma list; parts without joints get no FIX/CIRCLE_C")
    args = ap.parse_args()
    base = pathlib.Path(args.base_dir)
    out = pathlib.Path(args.output_dir)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    (out / "viewer").mkdir(exist_ok=True)

    report = []
    for asm in args.assemblies.split(","):
        joints_map = load_joints(base, asm)
        for f in sorted((base / asm / "wireframes").glob("*.json")):
            if f.name.endswith(".overrides.json"):
                continue
            joints = joints_map.get(part_id_from_stem(f.stem), [])
            try:
                W, cstats = convert_part(f, joints)
                Q = quantize_W(W)
                ok = roundtrip_ok(Q)
                gen = realize_points(Q)
                src = source_scope_points(f)
                cd, p95 = chamfer(gen, src)
            except Exception as exc:
                print(f"[FAIL] {asm}:{f.stem}: {exc}", flush=True)
                report.append({"part": f"{asm}:{f.stem}", "error": str(exc)})
                continue
            n_v = len(Q["vertices"])
            n_fix = sum(1 for v in Q["vertices"] if v["T"] == "FIX")
            row = {"part": f"{asm}:{f.stem}", "roundtrip_ok": bool(ok),
                   "n_vertices": n_v, "n_fix": n_fix, "n_edges": len(Q["edges"]),
                   "n_tokens": n_tokens(Q),
                   "fidelity_chamfer_mm": cd, "fidelity_p95_mm": p95, **cstats}
            report.append(row)
            (out / "parts" / f"{asm}__{f.stem}.json").write_text(
                json.dumps({"Q": Q, "sigma": sigma(Q)}, default=str), encoding="utf-8")
            # viewer export of the realized sparse wireframe
            edges_view = []
            for k, e in enumerate(Q["edges"]):
                poly = realize_edge(Q, e)
                if poly is None:
                    continue
                cls = e["cls"] if e["cls"] in VIEWER_COLOR_CLS else "bend_line"
                edges_view.append({"id": k, "type": cls, "length_mm": 0.0,
                                   "closed": False, "face_types": [e["tau"]],
                                   "fingerprint": f"w{k}",
                                   "polyline": np.round(poly, 3).tolist()})
            counts: dict[str, int] = {}
            for e in edges_view:
                counts[e["type"]] = counts.get(e["type"], 0) + 1
            (out / "viewer" / f"{asm}__{f.stem}.json").write_text(
                json.dumps({"schema": "wireframe v4.4", "source_stp": "wtok",
                            "n_faces": 0, "face_types": [], "clusters": [],
                            "edge_counts": counts, "edges": edges_view}), encoding="utf-8")
            print(f"[ok] {asm}:{f.stem}  V={n_v}(fix {n_fix}) E={len(Q['edges'])} "
                  f"tok={row['n_tokens']} rt={'OK' if ok else 'NG'} "
                  f"chamfer={cd:.2f}mm p95={p95:.2f}mm", flush=True)

    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    good = [r for r in report if "error" not in r]
    print(f"--- {len(good)}/{len(report)} parts converted")
    if good:
        print(f"roundtrip pass: {sum(r['roundtrip_ok'] for r in good)}/{len(good)}")
        for key in ("n_vertices", "n_edges", "n_tokens"):
            vals = [r[key] for r in good]
            print(f"{key}: median {int(np.median(vals))}, p90 {int(np.percentile(vals, 90))}, "
                  f"max {max(vals)}")
        cds = [r["fidelity_chamfer_mm"] for r in good if np.isfinite(r["fidelity_chamfer_mm"])]
        p95s = [r["fidelity_p95_mm"] for r in good if np.isfinite(r["fidelity_p95_mm"])]
        print(f"fidelity chamfer: mean {np.mean(cds):.2f}mm, worst {max(cds):.2f}mm")
        print(f"fidelity p95: mean {np.mean(p95s):.2f}mm, worst {max(p95s):.2f}mm")
        print(f"circle_c total: {sum(r['n_circle_c'] for r in good)}, "
              f"circle: {sum(r['n_circle'] for r in good)}, "
              f"arc: {sum(r['n_arc'] for r in good)}, line: {sum(r['n_line'] for r in good)}")


if __name__ == "__main__":
    main()

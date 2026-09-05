"""Constraint Geometric Attention — Phase 0 dataset builder.

For every part: project each constraint point (joints.json) onto the typed
wireframe graph (v4.4), compute pairwise geodesic distances along the graph
via Dijkstra, and save (joints, d_euclid, d_geo) matrices. Also reports the
proposal's guardrail baseline: how well plain Euclidean (+normal) distance
already predicts geodesic ordering — the learned model is only worth building
if there is headroom left.

Usage:
  python -m cae_mesh_generator.cga.build_pairs --output-dir ../runs/cga_dataset
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import spearmanr

from ..data.step_tessellate import TessellationConfig, load_or_tessellate
from ..wireflow.dataset import DEFAULT_BASE, ASSEMBLIES, load_joints, load_wireframe_segments

WELD_MM = 0.5            # node welding tolerance when building the graph
SAMPLE_STEP_MM = 4.0     # graph densification step along polylines
TOP_K = 3                # top-k recall for the baseline report


def build_graph(segments: dict) -> tuple[np.ndarray, coo_matrix]:
    """Wireframe graph: nodes = densified polyline points (welded), edges =
    consecutive samples. Returns (node_xyz, sparse adjacency with mm weights)."""
    nodes: dict[tuple, int] = {}
    xyz: list[np.ndarray] = []
    rows, cols, vals = [], [], []

    def node_id(p: np.ndarray) -> int:
        key = (round(p[0] / WELD_MM), round(p[1] / WELD_MM), round(p[2] / WELD_MM))
        if key not in nodes:
            nodes[key] = len(xyz)
            xyz.append(p)
        return nodes[key]

    for starts, vecs, cum in segments.values():
        lens = np.diff(np.concatenate([[0.0], cum]))
        for s, v, ln in zip(starts, vecs, lens):
            n = max(1, int(ln / SAMPLE_STEP_MM))
            pts = s + v * np.linspace(0.0, 1.0, n + 1)[:, None]
            ids = [node_id(p) for p in pts]
            for a, b in zip(ids[:-1], ids[1:]):
                if a != b:
                    d = float(np.linalg.norm(xyz[a] - xyz[b]))
                    rows += [a, b]
                    cols += [b, a]
                    vals += [d, d]
    node_xyz = np.asarray(xyz)
    adj = coo_matrix((vals, (rows, cols)), shape=(len(node_xyz), len(node_xyz)))
    return node_xyz, adj.tocsr()


def build_mesh_graph(stp_path: pathlib.Path, cache_dir: pathlib.Path
                     ) -> tuple[np.ndarray, coo_matrix]:
    """Midsurface geodesic teacher: triangle-edge graph of the tessellation.

    OCC duplicates vertices per CAD face, so vertices are welded by coordinate
    first — otherwise the graph falls apart at face seams (known pitfall)."""
    mesh = load_or_tessellate(stp_path, cache_dir, TessellationConfig())
    v = np.asarray(mesh.vertices, dtype=np.float64)
    keys = np.round(v / 0.05).astype(np.int64)
    _, weld, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    node_xyz = v[weld]
    f = inverse[np.asarray(mesh.faces, dtype=np.int64)]
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = e[e[:, 0] != e[:, 1]]
    d = np.linalg.norm(node_xyz[e[:, 0]] - node_xyz[e[:, 1]], axis=1)
    adj = coo_matrix((np.concatenate([d, d]),
                      (np.concatenate([e[:, 0], e[:, 1]]),
                       np.concatenate([e[:, 1], e[:, 0]]))),
                     shape=(len(node_xyz), len(node_xyz)))
    return node_xyz, adj.tocsr()


def resolve_stp(base: pathlib.Path, asm: str, stem: str) -> pathlib.Path | None:
    for sub in ("mid", "fill"):
        p = base / asm / sub / f"{stem}.stp"
        if p.exists():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=str(DEFAULT_BASE))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--min-joints", type=int, default=4)
    ap.add_argument("--teacher", choices=("mesh", "wire"), default="mesh",
                    help="geodesic source: midsurface tessellation (connected) or wireframe graph")
    args = ap.parse_args()
    base = pathlib.Path(args.base_dir)
    out = pathlib.Path(args.output_dir)
    (out / "parts").mkdir(parents=True, exist_ok=True)

    report = []
    for asm in ASSEMBLIES:
        joints_map = load_joints(base, asm)
        for f in sorted((base / asm / "wireframes").glob("*.json")):
            if f.name.endswith(".overrides.json"):
                continue
            pid = f.stem.split("_")[0] if "_" in f.stem else f.stem
            joints = joints_map.get(pid, [])
            if len(joints) < args.min_joints:
                continue
            segments, all_pts = load_wireframe_segments(f)
            if not segments:
                continue
            if args.teacher == "mesh":
                stp = resolve_stp(base, asm, f.stem)
                if stp is None:
                    continue
                node_xyz, adj = build_mesh_graph(stp, out / "tess_cache")
            else:
                node_xyz, adj = build_graph(segments)
            jxyz = np.asarray([j["xyz"] for j in joints], dtype=np.float64)
            jdir = np.asarray([j["dir"] for j in joints], dtype=np.float64)
            jtype = [j["type"] for j in joints]
            # project constraints onto graph nodes
            d2 = np.linalg.norm(jxyz[:, None, :] - node_xyz[None, :, :], axis=-1)
            proj_idx = d2.argmin(axis=1)
            proj_dist = d2.min(axis=1)
            geo_nodes = dijkstra(adj, indices=proj_idx)  # (K, n_nodes)
            d_geo = geo_nodes[:, proj_idx] + proj_dist[:, None] + proj_dist[None, :]
            np.fill_diagonal(d_geo, 0.0)
            d_geo = np.minimum(d_geo, d_geo.T)
            d_euc = np.linalg.norm(jxyz[:, None, :] - jxyz[None, :, :], axis=-1)
            scale = float(max(all_pts.max(axis=0) - all_pts.min(axis=0)))

            K = len(jxyz)
            iu = np.triu_indices(K, 1)
            finite = np.isfinite(d_geo[iu])
            rho = spearmanr(d_euc[iu][finite], d_geo[iu][finite]).statistic if finite.sum() > 2 else np.nan
            # top-k recall: does Euclidean top-k match geodesic top-k per point?
            recall = []
            dg = np.where(np.isfinite(d_geo), d_geo, 1e9)
            for i in range(K):
                order_g = np.argsort(dg[i])
                order_e = np.argsort(d_euc[i])
                tg = set(order_g[1: TOP_K + 1].tolist())
                te = set(order_e[1: TOP_K + 1].tolist())
                if tg:
                    recall.append(len(tg & te) / len(tg))
            row = {"part": f"{asm}:{f.stem}", "n_joints": K,
                   "n_graph_nodes": int(len(node_xyz)),
                   "proj_dist_mean_mm": float(proj_dist.mean()),
                   "unreachable_pair_fraction": float(1.0 - finite.mean()),
                   "spearman_euclid_vs_geo": float(rho),
                   f"top{TOP_K}_recall_euclid": float(np.mean(recall)),
                   "geo_over_euclid_median": float(np.nanmedian(
                       (d_geo[iu][finite] / np.maximum(d_euc[iu][finite], 1e-9)))),
                   "scale_mm": scale}
            report.append(row)
            np.savez_compressed(
                out / "parts" / f"{asm}__{f.stem}.npz",
                jxyz=jxyz, jdir=jdir, jtype=np.asarray(jtype),
                d_geo=d_geo, d_euclid=d_euc, center=(all_pts.min(0) + all_pts.max(0)) / 2,
                scale=scale)
            print(f"[ok] {asm}:{f.stem}  K={K} nodes={len(node_xyz)} "
                  f"spearman={rho:.3f} top{TOP_K}={row[f'top{TOP_K}_recall_euclid']:.3f} "
                  f"unreachable={row['unreachable_pair_fraction']:.2f}", flush=True)

    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    rhos = [r["spearman_euclid_vs_geo"] for r in report if np.isfinite(r["spearman_euclid_vs_geo"])]
    recs = [r[f"top{TOP_K}_recall_euclid"] for r in report]
    unr = [r["unreachable_pair_fraction"] for r in report]
    print(f"--- {len(report)} parts | Euclid-vs-geodesic spearman mean {np.mean(rhos):.3f} "
          f"| top{TOP_K} recall mean {np.mean(recs):.3f} | unreachable {np.mean(unr):.3f}")
    print("guardrail: if spearman/top-k are already ~1, a learned A_ij adds little; "
          "the headroom for the model is 1 - these values.")


if __name__ == "__main__":
    main()

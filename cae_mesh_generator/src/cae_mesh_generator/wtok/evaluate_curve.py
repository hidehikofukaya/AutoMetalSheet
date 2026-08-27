"""Curve-major evaluation: infill metrics + rendered realized curves.

Usage:
  python -m cae_mesh_generator.wtok.evaluate_curve --dataset ../runs/wtok_dataset \
      --checkpoint ../runs/wtok_curve_v1/last.pt --output-dir ../runs/wtok_curve_v1_eval
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .codec import realize_edge, realize_points
from .dataset_curve import load_curve_parts
from .model_curve import CurveAR, CurveSampler
from .train_curve import part_cond, realized_q
from .train_ar import chamfer_mm

TAU_COLOR = {"LINE": "#1f77b4", "ARC": "#ff7f0e", "CIRCLE": "#2ca02c", "CIRCLE_C": "#d62728"}


def plot_curves(ax_row, Q, fix_mm, lims, title):
    pairs = (((0, 1), "XY"), ((0, 2), "XZ"), ((1, 2), "YZ"))
    polys = [(e["tau"], realize_edge(Q, e)) for e in Q["edges"]]
    for ax, ((i, j), name) in zip(ax_row, pairs):
        for tau, poly in polys:
            if poly is not None and len(poly) >= 2:
                ax.plot(poly[:, i], poly[:, j], lw=0.7, c=TAU_COLOR.get(tau, "gray"))
        if len(fix_mm):
            ax.scatter(fix_mm[:, i], fix_mm[:, j], s=25, c="red", marker="*", zorder=3)
        ax.set_xlim(lims[i])
        ax.set_ylim(lims[j])
        ax.set_aspect("equal")
        ax.set_title(f"{title} {name}", fontsize=9)
        ax.tick_params(labelsize=6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = pathlib.Path(args.output_dir)
    (out / "renders").mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = CurveAR(ck["args"]["dim"], 8, ck["args"]["layers"], dropout=0.0).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"checkpoint epoch {ck['epoch']}")

    parts = load_curve_parts(pathlib.Path(args.dataset))
    order = np.random.default_rng(ck["args"]["split_seed"]).permutation(len(parts))
    n_val = max(1, int(round(len(parts) * ck["args"]["val_fraction"])))
    val_idx = set(order[:n_val].tolist())
    val_parts = [p for i, p in enumerate(parts) if i in val_idx]

    sampler = CurveSampler(model, args.device)
    rows = []
    for p in val_parts:
        cond, fix = part_cond(p, args.device)
        gt_q = realized_q(p, p.vertices, p.edges)
        gt_pts = realize_points(gt_q)
        fix_mm = np.zeros((0, 3))
        if fix:
            from .train_ar import bins_to_mm
            fix_mm = bins_to_mm(fix, p.env_lo, p.env_hi)

        rng = np.random.default_rng(args.seed + 11)
        observed = [
            {"tau": e["tau"],
             "verts": [(p.vertices[r]["T"], p.vertices[r]["bin"]) for r in e["refs"]]}
            for e in p.edges if rng.uniform() < 0.5
        ]
        vi, ei = sampler.run(cond, fix, observed, seed=args.seed)
        cd_infill = chamfer_mm(realize_points(realized_q(p, vi, ei))[::2], gt_pts[::2])

        vg, eg = sampler.run(cond, fix, [], seed=args.seed)
        gen_q = realized_q(p, vg, eg)
        cd_gen = chamfer_mm(realize_points(gen_q)[::2], gt_pts[::2])

        rows.append({"part": p.name, "n_gt_edges": len(p.edges), "n_fix": len(fix),
                     "infill50_chamfer_mm": cd_infill, "infill50_n_edges": len(ei),
                     "gen_n_edges": len(eg), "gen_chamfer_ref_mm": cd_gen})
        print(f"{p.name[:46]:46s} infill {cd_infill:7.2f}mm (E={len(ei):3d}) "
              f"gen E={len(eg):3d}/{len(p.edges):3d}", flush=True)

        lims = list(zip(p.env_lo, p.env_hi))
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plot_curves(axes[0], gen_q, fix_mm, lims, f"GENERATED E={len(eg)}")
        plot_curves(axes[1], gt_q, fix_mm, lims, f"GT E={len(p.edges)}")
        fig.suptitle(f"{p.name} — curve-major 100% generation (epoch {ck['epoch']})",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "renders" / f"{p.name}.png", dpi=110)
        plt.close(fig)

    agg = {"epoch": ck["epoch"],
           "infill50_chamfer_mean_mm": float(np.nanmean([r["infill50_chamfer_mm"] for r in rows])),
           "gen_n_edges_mean": float(np.mean([r["gen_n_edges"] for r in rows])),
           "gt_n_edges_mean": float(np.mean([r["n_gt_edges"] for r in rows])),
           "gen_chamfer_ref_mean_mm": float(np.nanmean([r["gen_chamfer_ref_mm"] for r in rows]))}
    (out / "metrics.json").write_text(json.dumps({"aggregate": agg, "parts": rows}, indent=1),
                                      encoding="utf-8")
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()

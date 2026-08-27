"""Phase 1 evaluation:
  - 50% infilling: vertex Chamfer metrics (existing axis)
  - 100% generation (mu_3): rendered images for standalone shape judgement —
    a generated layout that differs from GT but is coherent counts as success.

Usage:
  python -m cae_mesh_generator.wtok.evaluate_ar --dataset ../runs/wtok_dataset \
      --checkpoint ../runs/wtok_ar_v2/best.pt --output-dir ../runs/wtok_ar_v2_eval_best
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

from .dataset_ar import cond_features, load_parts
from .model_ar import MonotonicSampler, VertexAR
from .train_ar import bins_to_mm, chamfer_mm

TYPE_COLOR = {"END": "#1f77b4", "MID": "#ff7f0e"}


def scatter_views(ax_row, pts: np.ndarray, types: list[str], fix_mm: np.ndarray,
                  lims, title_prefix: str):
    pairs = (((0, 1), "XY"), ((0, 2), "XZ"), ((1, 2), "YZ"))
    for ax, ((i, j), name) in zip(ax_row, pairs):
        if len(pts):
            colors = [TYPE_COLOR.get(t, "gray") for t in types]
            ax.scatter(pts[:, i], pts[:, j], s=4, c=colors, linewidths=0)
        if len(fix_mm):
            ax.scatter(fix_mm[:, i], fix_mm[:, j], s=30, c="red", marker="*", zorder=3)
        ax.set_xlim(lims[i])
        ax.set_ylim(lims[j])
        ax.set_aspect("equal")
        ax.set_title(f"{title_prefix} {name}", fontsize=9)
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
    model = VertexAR(ck["args"]["dim"], 8, ck["args"]["layers"],
                     dropout=0.0).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"checkpoint: {args.checkpoint} (epoch {ck['epoch']})")

    parts = load_parts(pathlib.Path(args.dataset))
    order = np.random.default_rng(ck["args"]["split_seed"]).permutation(len(parts))
    n_val = max(1, int(round(len(parts) * ck["args"]["val_fraction"])))
    val_idx = set(order[:n_val].tolist())
    val_parts = [p for i, p in enumerate(parts) if i in val_idx]

    sampler = MonotonicSampler(model, args.device)
    rows = []
    for p in val_parts:
        fix, gen_gt = p.variant(False)
        cond = torch.from_numpy(cond_features(fix, p.env_lo, p.env_hi))[None].to(args.device)
        cmask = torch.ones(1, cond.shape[1], dtype=torch.bool, device=args.device)
        gt_mm = bins_to_mm(gen_gt, p.env_lo, p.env_hi)
        fix_mm = bins_to_mm(fix, p.env_lo, p.env_hi)

        # ---- 50% infill: existing metric axis ----
        rng = np.random.default_rng(args.seed + 7)
        observed = [v for v in gen_gt if rng.uniform() < 0.5]
        out_infill = sampler.run(cond, cmask, observed, seed=args.seed)
        infill_mm = bins_to_mm(out_infill, p.env_lo, p.env_hi)
        cd_infill = chamfer_mm(infill_mm, gt_mm)

        # ---- 100% generation: rendered for standalone judgement ----
        out_gen = sampler.run(cond, cmask, [], seed=args.seed)
        gen_mm = bins_to_mm(out_gen, p.env_lo, p.env_hi)
        gen_types = [v["T"] for v in out_gen]
        cd_gen = chamfer_mm(gen_mm, gt_mm)  # reference only, not the verdict

        rows.append({"part": p.name, "n_gt": len(gen_gt), "n_fix": len(fix),
                     "infill50_chamfer_mm": cd_infill,
                     "infill50_n": len(out_infill),
                     "gen_n": len(out_gen), "gen_chamfer_ref_mm": cd_gen})
        print(f"{p.name[:48]:48s} infill50 {cd_infill:6.2f}mm  "
              f"gen n={len(out_gen):4d} (gt {len(gen_gt):4d})", flush=True)

        lims = list(zip(p.env_lo, p.env_hi))
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        scatter_views(axes[0], gen_mm, gen_types, fix_mm, lims,
                      f"GENERATED n={len(out_gen)}")
        scatter_views(axes[1], gt_mm, [v["T"] for v in gen_gt], fix_mm, lims,
                      f"GT n={len(gen_gt)}")
        fig.suptitle(f"{p.name} — 100% generation (FIX={len(fix)}, epoch {ck['epoch']})",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "renders" / f"{p.name}.png", dpi=110)
        plt.close(fig)
        np.savez_compressed(out / "renders" / f"{p.name}.npz",
                            gen_mm=gen_mm, gen_types=np.asarray(gen_types),
                            gt_mm=gt_mm, fix_mm=fix_mm)

    agg = {"checkpoint": args.checkpoint, "epoch": ck["epoch"],
           "infill50_chamfer_mean_mm": float(np.nanmean(
               [r["infill50_chamfer_mm"] for r in rows])),
           "gen_n_mean": float(np.mean([r["gen_n"] for r in rows])),
           "gt_n_mean": float(np.mean([r["n_gt"] for r in rows])),
           "gen_chamfer_ref_mean_mm": float(np.nanmean(
               [r["gen_chamfer_ref_mm"] for r in rows]))}
    (out / "metrics.json").write_text(json.dumps({"aggregate": agg, "parts": rows},
                                                 indent=1), encoding="utf-8")
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()

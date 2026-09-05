"""P2 end-to-end: condition -> flow samples -> realizer -> shape metrics + renders.

Metrics per val part (N=8 samples):
  feasibility rate (planner accepts the generated theta - constructive check)
  min-of-N / mean chamfer vs GT mid STP (with the measured sampling floor)
  diversity: mean pairwise chamfer among feasible samples
  retrieval-geometry baseline: nearest-condition train theta realized under the
  val part's own condition.

Usage:
  python -m cae_mesh_generator.twopoint.generate_eval \
      --model ../runs/twopoint_p1b/model.pt --output-dir ../runs/twopoint_p2_gen \
      --tess-cache ../runs/twopoint_p2_full/tess_cache
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
import torch.nn as nn

from ..wtok.train_ar import chamfer_mm
from .dataset import CLASSES, ThetaNormalizer, load_parts, stratified_split
from .flow import D, FlowMLP
from .baselines import RegressionMLP
from .realize import realize_points
from .realize_eval import gt_points


def render(out_png, gen_pts, gt_pts, title):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    pairs = (((0, 1), "XY"), ((0, 2), "XZ"), ((1, 2), "YZ"))
    lims = [(min(gt_pts[:, k].min(), gen_pts[:, k].min()) - 5,
             max(gt_pts[:, k].max(), gen_pts[:, k].max()) + 5) for k in range(3)]
    for row, (pts, label, color) in enumerate(
            ((gen_pts, "GENERATED(best)", "#1f77b4"), (gt_pts, "GT", "#888888"))):
        for ax, ((i, j), name) in zip(axes[row], pairs):
            ax.scatter(pts[:, i], pts[:, j], s=1.5, c=color, linewidths=0)
            ax.set_xlim(lims[i])
            ax.set_ylim(lims[j])
            ax.set_aspect("equal")
            ax.set_title(f"{label} {name}", fontsize=9)
            ax.tick_params(labelsize=6)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tess-cache", required=True)
    ap.add_argument("--split-seed", type=int, default=13)
    ap.add_argument("--max-parts", type=int, default=40)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--n-renders", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = pathlib.Path(args.output_dir)
    (out / "renders").mkdir(parents=True, exist_ok=True)
    dev = args.device

    ck = torch.load(args.model, map_location=dev, weights_only=False)
    flow = FlowMLP(hidden=ck["args"].get("hidden", 512)).to(dev)
    flow.load_state_dict(ck["flow"])
    flow.eval()
    clf = RegressionMLP().to(dev)
    clf.load_state_dict(ck["clf"])
    clf.eval()
    norm = ThetaNormalizer.__new__(ThetaNormalizer)
    norm.lo, norm.hi = ck["norm_lo"], ck["norm_hi"]
    mu, sd = ck["cond_mu"], ck["cond_sd"]

    parts = load_parts()
    train, val = stratified_split(parts, args.split_seed)
    tr_z = (np.stack([p["cond"] for p in train]) - mu) / sd

    @torch.no_grad()
    def sample_theta(cond_z, cls, bits, seed):
        g = torch.Generator(dev).manual_seed(seed)
        c1 = nn.functional.one_hot(torch.tensor([cls], device=dev), len(CLASSES)).float()
        x = torch.randn(1, D, device=dev, generator=g)
        cz = torch.tensor(cond_z, dtype=torch.float32, device=dev)[None]
        bt = torch.tensor(bits, dtype=torch.float32, device=dev)[None]
        dt = 1.0 / args.steps
        for k in range(args.steps):
            t = torch.full((1,), k * dt, device=dev)
            x = x + flow(x, t, cz, c1, bt) * dt
        return norm.decode(x[0].cpu().numpy())

    rows = []
    for vi, p in enumerate(val[: args.max_parts]):
        path = pathlib.Path(p["path"])
        spec = json.loads(path.read_text(encoding="utf-8"))["spec"]
        gt = gt_points(path, pathlib.Path(args.tess_cache))
        cond_z = (p["cond"] - mu) / sd
        with torch.no_grad():
            _, cl, bits_l = clf(torch.tensor(cond_z, dtype=torch.float32, device=dev)[None])
            probs = torch.softmax(cl[0], dim=0)
            bits = (torch.sigmoid(bits_l[0]) > 0.5).float().cpu().numpy()
        g = torch.Generator(dev).manual_seed(9000 + vi)
        cls_samples = torch.multinomial(probs, args.n_samples, replacement=True,
                                        generator=g).tolist()
        feasible, cds = [], []
        for j, cs in enumerate(cls_samples):
            theta = sample_theta(cond_z, cs, bits, seed=50_000 * vi + 31 * j)
            try:
                pts = realize_points(spec, theta, cs,
                                     (int(bits[0]), int(bits[1])), bead_rise_sign=1)
            except ValueError:
                continue
            feasible.append(pts)
            cds.append(chamfer_mm(pts, gt))
        # retrieval-geometry baseline
        nn_i = int(np.argmin(np.linalg.norm(tr_z - cond_z, axis=1)))
        tp = train[nn_i]
        try:
            rp = realize_points(spec, tp["theta"], tp["cls"], tp["flange_bits"], 1)
            cd_retr = chamfer_mm(rp, gt)
        except ValueError:
            cd_retr = float("nan")
        div = float("nan")
        if len(feasible) >= 2:
            pw = [chamfer_mm(feasible[a][::4], feasible[b][::4])
                  for a in range(len(feasible)) for b in range(a + 1, len(feasible))]
            div = float(np.mean(pw))
        row = {"part": p["part_id"], "cls": p["cls"],
               "feasible": len(feasible), "n_samples": args.n_samples,
               "min_chamfer_mm": float(min(cds)) if cds else float("nan"),
               "mean_chamfer_mm": float(np.mean(cds)) if cds else float("nan"),
               "retrieval_chamfer_mm": cd_retr, "diversity_mm": div}
        rows.append(row)
        print(f"{p['part_id']:36s} feas {len(feasible)}/{args.n_samples} "
              f"min {row['min_chamfer_mm']:6.2f}mm mean {row['mean_chamfer_mm']:6.2f} "
              f"retr {cd_retr:6.2f} div {div:5.1f}", flush=True)
        if vi < args.n_renders and cds:
            best = feasible[int(np.argmin(cds))]
            render(out / "renders" / f"{p['part_id']}.png", best, gt,
                   f"{p['part_id']} min-of-{args.n_samples} chamfer "
                   f"{row['min_chamfer_mm']:.2f}mm (feasible {len(feasible)})")

    ok = [r for r in rows if np.isfinite(r["min_chamfer_mm"])]
    agg = {
        "n": len(rows),
        "feasibility_rate": float(np.mean([r["feasible"] / r["n_samples"] for r in rows])),
        "min_chamfer_mean_mm": float(np.mean([r["min_chamfer_mm"] for r in ok])),
        "mean_chamfer_mean_mm": float(np.mean([r["mean_chamfer_mm"] for r in ok])),
        "retrieval_chamfer_mean_mm": float(np.nanmean([r["retrieval_chamfer_mm"] for r in rows])),
        "diversity_mean_mm": float(np.nanmean([r["diversity_mm"] for r in rows])),
    }
    for c, name in ((0, "flange"), (1, "bead")):
        sub = [r["min_chamfer_mm"] for r in ok if r["cls"] == c]
        if sub:
            agg[f"min_chamfer_{name}_mm"] = float(np.mean(sub))
    (out / "metrics.json").write_text(json.dumps({"aggregate": agg, "parts": rows},
                                                 indent=1), encoding="utf-8")
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()

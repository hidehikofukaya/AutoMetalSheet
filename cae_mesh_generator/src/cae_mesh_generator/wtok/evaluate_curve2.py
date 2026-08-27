"""Integrated evaluation for curve-major models (runs anywhere: no pythonocc).

Covers every model-verdict metric established in the session, so the go/no-go
decision can be made entirely on Kaggle:

  A. teacher forcing : val NLL, per-token-category accuracy, coord digit error
  B. free running    : realized-curve chamfer (best of N seeds), edge-count ratio
  C. error breakdown : GT-outline coverage / GT-bend coverage / false positives
                       (the decomposition that showed the v1 failure is dominated
                        by stray wires, not by misplaced base panels)
  D. infilling       : 50% observed-wire completion chamfer
  E. renders         : generated vs GT curves, 3 orthogonal views

Works with both model generations:
  --arch v2 (default) -> curve2.CurveAR2 / CurveSampler2   (relative coords)
  --arch v1           -> model_curve.CurveAR / CurveSampler (absolute coords)

Usage:
  python -m cae_mesh_generator.wtok.evaluate_curve2 \
      --dataset ../runs/wtok_synth --checkpoint ../runs/wtok_curve2_v1/best.pt \
      --val-list ../runs/wtok_curve_synth_v1/val_names_100.json \
      --output-dir ../runs/wtok_curve2_v1_eval --arch v2
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import collections
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .codec import realize_edge, realize_points
from .dataset_curve import (ADV, COORD0, NEW, STOP, TAU_BASE, VOCAB_C,
                            load_curve_parts)
from .train_ar import bins_to_mm, chamfer_mm
from .train_curve import part_cond, realized_q, to_device

TAU_COLOR = {"LINE": "#1f77b4", "ARC": "#ff7f0e",
             "CIRCLE": "#2ca02c", "CIRCLE_C": "#d62728"}
OUTLINE_CLASSES = {"outer_boundary"}
BEND_CLASSES = {"bend_line", "crease", "feature_rim", "hole_boundary"}


# ---------------------------------------------------------------- arch plumbing

def load_arch(arch: str, ck: dict, device: str):
    """Returns (model, sampler, dataset_cls, is_v2)."""
    a = ck.get("args", {})
    dim, layers = a.get("dim", 256), a.get("layers", 8)
    if arch == "v3":
        from .curve3 import CurveAR3, CurveARDataset3, CurveSampler3
        model = CurveAR3(dim, 8, layers, dropout=0.0).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        return model, CurveSampler3(model, device), CurveARDataset3, False
    if arch == "v2":
        from .curve2 import CurveAR2, CurveARDataset2, CurveSampler2
        model = CurveAR2(dim, 8, layers, dropout=0.0).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        return model, CurveSampler2(model, device), CurveARDataset2, True
    from .dataset_curve import CurveARDataset
    from .model_curve import CurveAR, CurveSampler
    model = CurveAR(dim, 8, layers, dropout=0.0).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, CurveSampler(model, device), CurveARDataset, False


# ---------------------------------------------------------------- metrics

def class_points(p, classes: set, step_mm: float = 2.0) -> np.ndarray:
    """Densely sample GT curves belonging to the given edge classes."""
    Q = realized_q(p, p.vertices, p.edges)
    out = []
    for e in p.edges:
        if e.get("cls") not in classes:
            continue
        poly = realize_edge(Q, e)
        if poly is None or len(poly) < 2:
            continue
        seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        total = float(seg.sum())
        if total < 1e-9:
            continue
        n = max(2, int(total / step_mm))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        tt = np.linspace(0.0, cum[-1], n)
        idx = np.clip(np.searchsorted(cum, tt) - 1, 0, len(seg) - 1)
        fr = (tt - cum[idx]) / np.maximum(seg[idx], 1e-12)
        out.append(poly[idx] + (poly[idx + 1] - poly[idx]) * fr[:, None])
    return np.concatenate(out) if out else np.zeros((0, 3))


def one_way(a: np.ndarray, b: np.ndarray, max_pts: int = 6000) -> tuple[float, float]:
    """mean / p95 of nearest-neighbour distance from a to b (mm)."""
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    if len(a) > max_pts:
        a = a[rng.choice(len(a), max_pts, replace=False)]
    if len(b) > max_pts:
        b = b[rng.choice(len(b), max_pts, replace=False)]
    mins = []
    for i in range(0, len(a), 2048):
        d = np.linalg.norm(a[i:i + 2048, None, :] - b[None, :, :], axis=-1)
        mins.append(d.min(axis=1))
    m = np.concatenate(mins)
    return float(m.mean()), float(np.percentile(m, 95))


@torch.no_grad()
def token_accuracy(model, dataset, device, is_v2: bool, max_parts: int = 40) -> dict:
    """Teacher-forced accuracy per token category (+ coord digit bin error)."""
    from .curve2 import N_OFF, OFF0
    ok, n = collections.Counter(), collections.Counter()
    coord_err = collections.defaultdict(list)
    vocab = OFF0 + N_OFF if is_v2 else VOCAB_C
    for i in range(min(len(dataset), max_parts)):
        item = to_device(dataset[i], device)
        static, ptr = model.forward(item)
        pos = torch.arange(static.shape[0], device=device)
        if ptr.shape[1]:
            avail = item["mat_pos"][None, :] <= pos[:, None]
            st, mt = item["slot_next"], item["mat_type"]
            want = torch.full_like(st, -1)
            want[st == 1], want[st == 2], want[st == 3] = 1, 2, 0
            tm = (mt[None, :] == want[:, None]) & (want[:, None] >= 0)
            ptr = ptr.masked_fill(~(avail & tm), -1e9)
        pred = torch.cat([static, ptr], dim=1).argmax(dim=1).tolist()
        tgt = item["target"].tolist()
        for p_, t_ in zip(pred, tgt):
            if t_ == STOP:
                cat = "STOP"
            elif t_ == ADV:
                cat = "ADV"
            elif TAU_BASE <= t_ < TAU_BASE + 4:
                cat = "TAU"
            elif t_ == NEW:
                cat = "NEW_sel"
            elif t_ >= vocab:
                cat = "PTR_sel"
            elif is_v2 and OFF0 <= t_ < OFF0 + N_OFF:
                cat = "COORD_offset"
                if OFF0 <= p_ < OFF0 + N_OFF:
                    coord_err[cat].append(abs(p_ - t_))
            elif COORD0 <= t_ < COORD0 + 128:
                cat = "COORD_fine"
                if COORD0 <= p_ < COORD0 + 128:
                    coord_err[cat].append(abs(p_ - t_))
            else:
                cat = "other"
            n[cat] += 1
            ok[cat] += (p_ == t_)
    out = {f"acc_{c}": ok[c] / n[c] for c in n if n[c]}
    for c, errs in coord_err.items():
        out[f"binerr_median_{c}"] = float(np.median(errs))
        out[f"binerr_mean_{c}"] = float(np.mean(errs))
    return out


@torch.no_grad()
def val_nll(model, dataset, device) -> float:
    return float(np.mean([float(model.loss(to_device(dataset[i], device)))
                          for i in range(len(dataset))]))


def observed_records(p, rate: float, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [{"tau": e["tau"],
             "verts": [(p.vertices[r]["T"], p.vertices[r]["bin"]) for r in e["refs"]]}
            for e in p.edges if rng.uniform() < rate]


# ---------------------------------------------------------------- rendering

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


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-list", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--arch", choices=("v1", "v2", "v3"), default="v3")
    ap.add_argument("--max-parts", type=int, default=40, help="parts for free-run metrics")
    ap.add_argument("--seeds", type=int, default=2, help="sampling seeds per part")
    ap.add_argument("--n-renders", type=int, default=12)
    ap.add_argument("--baseline", default="", help="metrics.json of a previous run")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = pathlib.Path(args.output_dir)
    (out / "renders").mkdir(parents=True, exist_ok=True)
    dev = args.device

    ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    model, sampler, dataset_cls, is_v2 = load_arch(args.arch, ck, dev)
    print(f"checkpoint epoch {ck.get('epoch')} | arch {args.arch} | device {dev}")

    parts = load_curve_parts(pathlib.Path(args.dataset))
    val_names = set(json.loads(pathlib.Path(args.val_list).read_text(encoding="utf-8")))
    val = [p for p in parts if p.name in val_names]
    print(f"val parts: {len(val)} (of {len(val_names)} requested)")

    ds_gen = dataset_cls(val, augment=False, obs_rate=0.0, base_seed=555)
    ds_half = dataset_cls(val, augment=False, obs_rate=0.5, base_seed=555)

    # ---- A. teacher forcing ----
    tf = {"val_nll_gen": val_nll(model, ds_gen, dev),
          "val_nll_half": val_nll(model, ds_half, dev)}
    tf.update(token_accuracy(model, ds_gen, dev, is_v2, args.max_parts))
    print("[A] teacher forcing:",
          " ".join(f"{k}={v:.3f}" for k, v in tf.items() if k.startswith(("val", "acc"))))

    # ---- B/C/D. free running, decomposition, infill ----
    rows = []
    for vi, p in enumerate(val[: args.max_parts]):
        cond, fix = part_cond(p, dev)
        fix_mm = bins_to_mm(fix, p.env_lo, p.env_hi) if fix else np.zeros((0, 3))
        gt_q = realized_q(p, p.vertices, p.edges)
        gt_pts = realize_points(gt_q)
        gt_outline = class_points(p, OUTLINE_CLASSES)
        gt_bend = class_points(p, BEND_CLASSES)
        gt_all = (np.concatenate([gt_outline, gt_bend])
                  if len(gt_outline) or len(gt_bend) else gt_pts)

        best = None
        for s in range(args.seeds):
            v, e = sampler.run(cond, fix, [], seed=1 + s)
            gen_q = realized_q(p, v, e)
            gen_pts = realize_points(gen_q)
            if len(gen_pts) == 0:
                continue
            cd = chamfer_mm(gen_pts[::2], gt_pts[::2])
            if best is None or cd < best[0]:
                best = (cd, gen_q, gen_pts, len(e))
        if best is None:
            rows.append({"part": p.name, "gen_failed": True})
            continue
        cd_gen, gen_q, gen_pts, n_e = best
        cov_o = one_way(gt_outline, gen_pts)
        cov_b = one_way(gt_bend, gen_pts)
        fp = one_way(gen_pts[::2], gt_all)

        v_i, e_i = sampler.run(cond, fix, observed_records(p, 0.5, 11), seed=1)
        inf_pts = realize_points(realized_q(p, v_i, e_i))
        cd_inf = (chamfer_mm(inf_pts[::2], gt_pts[::2]) if len(inf_pts)
                  else float("nan"))

        rows.append({
            "part": p.name, "n_gt_edges": len(p.edges), "gen_n_edges": n_e,
            "edge_ratio": n_e / max(len(p.edges), 1),
            "gen_chamfer_mm": cd_gen, "infill50_chamfer_mm": cd_inf,
            "cov_outline_mean_mm": cov_o[0], "cov_outline_p95_mm": cov_o[1],
            "cov_bend_mean_mm": cov_b[0], "cov_bend_p95_mm": cov_b[1],
            "falsepos_mean_mm": fp[0], "falsepos_p95_mm": fp[1],
        })
        print(f"  {p.name[:34]:34s} gen {cd_gen:6.1f}mm E={n_e:3d}/{len(p.edges):3d} "
              f"outline {cov_o[0]:5.1f} bend {cov_b[0]:5.1f} fp {fp[0]:5.1f} "
              f"infill {cd_inf:6.1f}", flush=True)

        if vi < args.n_renders:
            lims = list(zip(p.env_lo, p.env_hi))
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            plot_curves(axes[0], gen_q, fix_mm, lims, f"GEN E={n_e}")
            plot_curves(axes[1], gt_q, fix_mm, lims, f"GT E={len(p.edges)}")
            fig.suptitle(f"{p.name} — {args.arch} ep{ck.get('epoch')} "
                         f"chamfer {cd_gen:.1f}mm", fontsize=11)
            fig.tight_layout()
            fig.savefig(out / "renders" / f"{p.name}.png", dpi=110)
            plt.close(fig)

    ok = [r for r in rows if "gen_failed" not in r]
    agg = {"epoch": ck.get("epoch"), "arch": args.arch, "n_val": len(val),
           "n_scored": len(ok), **tf}
    for key in ("gen_chamfer_mm", "infill50_chamfer_mm", "edge_ratio",
                "cov_outline_mean_mm", "cov_outline_p95_mm", "cov_bend_mean_mm",
                "cov_bend_p95_mm", "falsepos_mean_mm", "falsepos_p95_mm"):
        vals = [r[key] for r in ok if np.isfinite(r.get(key, np.nan))]
        if vals:
            agg[f"{key}_median"] = float(np.median(vals))
            agg[f"{key}_mean"] = float(np.mean(vals))
    (out / "metrics.json").write_text(
        json.dumps({"aggregate": agg, "parts": rows}, indent=1), encoding="utf-8")

    print("\n=== aggregate ===")
    print(json.dumps(agg, indent=1))
    if args.baseline:
        base = json.loads(pathlib.Path(args.baseline).read_text(encoding="utf-8"))["aggregate"]
        print("\n=== vs baseline ===")
        print(f"{'metric':32s} {'baseline':>10s} {'this':>10s} {'delta':>10s}")
        for k in sorted(set(base) & set(agg)):
            if isinstance(agg[k], (int, float)) and isinstance(base[k], (int, float)):
                print(f"{k:32s} {base[k]:10.3f} {agg[k]:10.3f} {agg[k]-base[k]:+10.3f}")


if __name__ == "__main__":
    main()

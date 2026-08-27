"""Train the curve-major AR model (definition 6').

Usage:
  python -m cae_mesh_generator.wtok.train_curve --dataset ../runs/wtok_dataset \
      --output-dir ../runs/wtok_curve_v1 --epochs 300 --val-fraction 0.1 --device cuda
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from .codec import realize_points
from .codec_curve import sigma_curve
from .dataset_curve import CurveARDataset, build_curve_item, load_curve_parts
from .model_curve import CurveAR, CurveSampler
from .dataset_ar import cond_features
from .train_ar import chamfer_mm
from .constants import safe_save


def to_device(item: dict, device: str) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in item.items()}


def part_cond(p, device):
    fix = [v for v in p.vertices if v["T"] == "FIX"]
    import numpy as _np
    from .dataset_ar import N_BINS
    rows = cond_features(fix, p.env_lo, p.env_hi)
    return torch.from_numpy(rows).to(device), fix


def realized_q(p, vertices, edges) -> dict:
    return {"env_lo": list(p.env_lo), "env_hi": list(p.env_hi),
            "vertices": vertices, "edges": edges}


@torch.no_grad()
def sample_eval(model, parts, device, obs_rate: float, max_parts: int = 8,
                seed: int = 0) -> dict:
    sampler = CurveSampler(model, device)
    cds, n_edges = [], []
    for p in parts[:max_parts]:
        cond, fix = part_cond(p, device)
        rng = np.random.default_rng(seed + 11)
        observed = [
            {"tau": e["tau"],
             "verts": [(p.vertices[r]["T"], p.vertices[r]["bin"]) for r in e["refs"]]}
            for e in p.edges if rng.uniform() < obs_rate
        ]
        verts, edges = sampler.run(cond, fix, observed, seed=seed)
        gt_pts = realize_points(realized_q(p, p.vertices, p.edges))
        gen_pts = realize_points(realized_q(p, verts, edges))
        cds.append(chamfer_mm(gen_pts[::3], gt_pts[::3]))
        n_edges.append(len(edges))
    return {"curve_chamfer_mm": float(np.nanmean(cds)),
            "n_edges_mean": float(np.mean(n_edges))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--jitter-bins", type=int, default=1)
    ap.add_argument("--stage2-after", type=int, default=120)
    ap.add_argument("--split-seed", type=int, default=13)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--sample-every", type=int, default=75)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", default="")
    ap.add_argument("--val-list", default="",
                    help="JSON file with a fixed list of val part names (keeps the "
                         "val set stable when the dataset grows across a resume)")
    args = ap.parse_args()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    parts = load_curve_parts(pathlib.Path(args.dataset))
    if args.val_list:
        val_names = set(json.loads(pathlib.Path(args.val_list).read_text(encoding="utf-8")))
        train_parts = [p for p in parts if p.name not in val_names]
        val_parts = [p for p in parts if p.name in val_names]
        assert len(val_parts) == len(val_names), "val-list parts missing from dataset"
    else:
        order = np.random.default_rng(args.split_seed).permutation(len(parts))
        n_val = max(1, int(round(len(parts) * args.val_fraction)))
        val_idx = set(order[:n_val].tolist())
        train_parts = [p for i, p in enumerate(parts) if i not in val_idx]
        val_parts = [p for i, p in enumerate(parts) if i in val_idx]
    print(f"parts: train {len(train_parts)} val {len(val_parts)}")
    (out / "split.json").write_text(json.dumps(
        {"train": [p.name for p in train_parts], "val": [p.name for p in val_parts],
         "args": vars(args)}, indent=1), encoding="utf-8")

    train_ds = CurveARDataset(train_parts, augment=True, jitter_bins=args.jitter_bins,
                              stage2_after=args.stage2_after)
    val_gen = CurveARDataset(val_parts, augment=False, obs_rate=0.0, base_seed=555)
    val_half = CurveARDataset(val_parts, augment=False, obs_rate=0.5, base_seed=555)

    model = CurveAR(args.dim, 8, args.layers, dropout=args.dropout).to(args.device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    history, best = [], float("inf")
    start_epoch = 1
    if args.resume:
        ck = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        hist_file = out / "history.json"
        if hist_file.exists():
            history = [r for r in json.loads(hist_file.read_text(encoding="utf-8"))
                       if r["epoch"] < start_epoch]
            best = min((r["val_nll_gen"] for r in history if "val_nll_gen" in r),
                       default=float("inf"))
        print(f"resumed at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()
        t0, tot = time.time(), 0.0
        idx = torch.randperm(len(train_ds)).tolist()
        opt.zero_grad()
        for step, i in enumerate(idx):
            item = to_device(train_ds[i], args.device)
            loss = model.loss(item) / args.accum
            loss.backward()
            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
            tot += float(loss) * args.accum
        row = {"epoch": epoch, "train_nll": tot / len(idx),
               "seconds": round(time.time() - t0, 1)}
        if epoch % args.val_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                vls = []
                for ds in (val_gen, val_half):
                    vls.append(float(np.mean(
                        [float(model.loss(to_device(ds[i], args.device)))
                         for i in range(len(ds))])))
            row["val_nll_gen"], row["val_nll_half"] = vls
            ck = {"model": model.state_dict(), "opt": opt.state_dict(),
                  "args": vars(args), "epoch": epoch}
            if vls[0] < best:
                best = vls[0]
                safe_save(ck, out / "best.pt")
            safe_save(ck, out / "last.pt")
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            model.eval()
            row["gen"] = sample_eval(model, val_parts, args.device, 0.0)
            row["infill50"] = sample_eval(model, val_parts, args.device, 0.5)
        history.append(row)
        (out / "history.json").write_text(json.dumps(history), encoding="utf-8")
        msg = f"epoch {epoch}: nll {row['train_nll']:.4f} ({row['seconds']}s)"
        if "val_nll_gen" in row:
            msg += f"  val gen {row['val_nll_gen']:.4f} half {row['val_nll_half']:.4f}"
        if "gen" in row:
            msg += (f"  | curve-chamfer gen {row['gen']['curve_chamfer_mm']:.1f}mm "
                    f"(E={row['gen']['n_edges_mean']:.0f}) "
                    f"infill {row['infill50']['curve_chamfer_mm']:.1f}mm")
        print(msg, flush=True)


if __name__ == "__main__":
    main()

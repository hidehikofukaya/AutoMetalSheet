"""Train the Phase 1 vertex-stage AR model (mu_1 random-mask infilling).

Usage:
  python -m cae_mesh_generator.wtok.train_ar --dataset ../runs/wtok_dataset \
      --output-dir ../runs/wtok_ar_v1 --device cuda
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
from torch.utils.data import DataLoader

from .dataset_ar import VertexARDataset, collate, load_parts, vertex_key
from .model_ar import MonotonicSampler, VertexAR
from .constants import BITS
from .constants import safe_save


def bins_to_mm(vertices: list[dict], env_lo, env_hi) -> np.ndarray:
    lo = np.asarray(env_lo)
    span = np.asarray(env_hi) - lo
    if not vertices:
        return np.zeros((0, 3))
    b = np.asarray([v["bin"] for v in vertices], dtype=np.float64)
    return lo + (b + 0.5) / (1 << BITS) * span


def chamfer_mm(a: np.ndarray, b: np.ndarray, max_points: int = 8000) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    # subsample (runaway generated curves can realize to 400k+ points) and chunk
    rng = np.random.default_rng(0)
    if len(a) > max_points:
        a = a[rng.choice(len(a), max_points, replace=False)]
    if len(b) > max_points:
        b = b[rng.choice(len(b), max_points, replace=False)]

    def one_way(x, y):
        mins = []
        for i in range(0, len(x), 2048):
            d = np.linalg.norm(x[i:i + 2048, None, :] - y[None, :, :], axis=-1)
            mins.append(d.min(axis=1))
        return float(np.concatenate(mins).mean())

    return one_way(a, b) + one_way(b, a)


@torch.no_grad()
def sample_eval(model, parts, device, obs_rate: float, max_parts: int = 15,
                seed: int = 0) -> dict:
    """Generate vertices (mu_3 when obs_rate=0) and compare vertex sets in mm."""
    from .dataset_ar import cond_features
    sampler = MonotonicSampler(model, device)
    cds, kept = [], []
    for p in parts[:max_parts]:
        fix, gen = p.variant(False)
        rng = np.random.default_rng(seed + 7)
        observed = [v for v in gen if rng.uniform() < obs_rate]
        cond = torch.from_numpy(cond_features(fix, p.env_lo, p.env_hi))[None].to(device)
        cmask = torch.ones(1, cond.shape[1], dtype=torch.bool, device=device)
        out = sampler.run(cond, cmask, observed, seed=seed)
        gen_mm = bins_to_mm([v for v in out], p.env_lo, p.env_hi)
        gt_mm = bins_to_mm(gen, p.env_lo, p.env_hi)
        cds.append(chamfer_mm(gen_mm, gt_mm))
        kept.append(len(out))
    return {"vertex_chamfer_mm": float(np.nanmean(cds)),
            "n_generated_mean": float(np.mean(kept))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--split-seed", type=int, default=13)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--sample-every", type=int, default=50)
    ap.add_argument("--resume", default="")
    # ---- v2 ----
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--mirror-axes", default="x,y,z", help="comma list or empty")
    ap.add_argument("--jitter-bins", type=int, default=1,
                    help="+-N bin coordinate noise on train vertices")
    ap.add_argument("--stage2-after", type=int, default=150,
                    help="epoch after which observation rates shift sparse (mu_3)")
    args = ap.parse_args()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    parts = load_parts(pathlib.Path(args.dataset))
    order = np.random.default_rng(args.split_seed).permutation(len(parts))
    n_val = max(1, int(round(len(parts) * args.val_fraction)))
    val_idx = set(order[:n_val].tolist())
    train_parts = [p for i, p in enumerate(parts) if i not in val_idx]
    val_parts = [p for i, p in enumerate(parts) if i in val_idx]
    print(f"parts: train {len(train_parts)} val {len(val_parts)}")
    (out / "split.json").write_text(json.dumps(
        {"train": [p.name for p in train_parts], "val": [p.name for p in val_parts],
         "args": vars(args)}, indent=1), encoding="utf-8")

    mirror_axes = tuple(a for a in args.mirror_axes.split(",") if a)
    train_ds = VertexARDataset(train_parts, mirror_axes=mirror_axes,
                               jitter_bins=args.jitter_bins,
                               stage2_after=args.stage2_after)
    val_gen = VertexARDataset(val_parts, obs_rate=0.0, base_seed=555)
    val_half = VertexARDataset(val_parts, obs_rate=0.5, base_seed=555)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate)

    model = VertexAR(args.dim, 8, args.layers, dropout=args.dropout).to(args.device)
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
        print(f"resumed at epoch {start_epoch}, best {best:.4f}")
    for epoch in range(start_epoch, args.epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        opt.zero_grad()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            loss = model.loss(batch) / args.accum
            loss.backward()
            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
            tot += float(loss) * args.accum
            n += 1
        row = {"epoch": epoch, "train_nll": tot / max(n, 1),
               "seconds": round(time.time() - t0, 1)}
        if epoch % args.val_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                vls = []
                for ds in (val_gen, val_half):
                    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate)
                    tot_v = sum(float(model.loss(
                        {k: v.to(args.device) if torch.is_tensor(v) else v
                         for k, v in b.items()})) for b in loader)
                    vls.append(tot_v / len(loader))
            row["val_nll_gen"], row["val_nll_half"] = vls
            if vls[0] < best:
                best = vls[0]
                safe_save({"model": model.state_dict(), "args": vars(args),
                           "epoch": epoch}, out / "best.pt")
            safe_save({"model": model.state_dict(), "opt": opt.state_dict(),
                       "args": vars(args), "epoch": epoch}, out / "last.pt")
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            model.eval()
            row["gen"] = sample_eval(model, val_parts, args.device, 0.0)
            row["infill50"] = sample_eval(model, val_parts, args.device, 0.5)
        history.append(row)
        (out / "history.json").write_text(json.dumps(history), encoding="utf-8")
        msg = f"epoch {epoch}: nll {row['train_nll']:.4f}"
        if "val_nll_gen" in row:
            msg += f"  val gen {row['val_nll_gen']:.4f} half {row['val_nll_half']:.4f}"
        if "gen" in row:
            msg += (f"  | sample chamfer gen {row['gen']['vertex_chamfer_mm']:.2f}mm "
                    f"infill {row['infill50']['vertex_chamfer_mm']:.2f}mm")
        print(msg, flush=True)


if __name__ == "__main__":
    main()

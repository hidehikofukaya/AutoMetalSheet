"""Train the wireframe flow AE.

Usage (from cae_mesh_generator/):
  python -m cae_mesh_generator.wireflow.train --output-dir ../runs/wireflow_v1 \
      --epochs 300 --device cuda
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import pathlib
import time

import torch
from torch.utils.data import DataLoader

from .dataset import DEFAULT_BASE, WireflowDataset, discover_parts, split_parts
from .model import WireFlowModel


def safe_save(obj, path: pathlib.Path, retries: int = 5) -> None:
    """Atomic save with retry: Windows AV/indexer can transiently lock the
    destination, which killed a 100-epoch run once. Write to tmp + replace."""
    tmp = path.with_suffix(".tmp")
    for attempt in range(retries):
        try:
            torch.save(obj, tmp)
            os.replace(tmp, path)
            return
        except (RuntimeError, OSError) as exc:
            if attempt == retries - 1:
                print(f"[warn] giving up saving {path.name}: {exc}", flush=True)
                return
            time.sleep(2.0 * (attempt + 1))


def evaluate_fm_loss(model, loader, device, z_dropout: float, ot_coupling: bool = False,
                     relation_mode: str = "none") -> dict:
    model.eval()
    sums, n = {}, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            losses = model.training_losses(batch, z_dropout=z_dropout, joint_dropout=0.0,
                                           ot_coupling=ot_coupling, relation_mode=relation_mode)
            bs = batch["points"].shape[0]
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v) * bs
            n += bs
    model.train()
    return {k: v / max(n, 1) for k, v in sums.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=str(DEFAULT_BASE))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n-points", type=int, default=1024)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--dec-layers", type=int, default=6)
    ap.add_argument("--enc-layers", type=int, default=3)
    ap.add_argument("--n-latent", type=int, default=32)
    ap.add_argument("--split-seed", type=int, default=13)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--mirror-axes", default="y", help="comma list or empty")
    ap.add_argument("--z-dropout", type=float, default=0.5)
    ap.add_argument("--joint-dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--resume", default="", help="checkpoint path to resume from")
    ap.add_argument("--ot-coupling", action="store_true",
                    help="Sinkhorn source-target pairing (straightens flows)")
    ap.add_argument("--relation-bias", choices=("none", "heuristic", "oracle"), default="none",
                    help="constraint relation graph injected into joint self-attention")
    ap.add_argument("--pairs-dir", default="", help="cga npz dir (defaults to cga_dataset_mesh)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from .dataset import DEFAULT_PAIRS
    pairs_dir = pathlib.Path(args.pairs_dir) if args.pairs_dir else DEFAULT_PAIRS
    parts = discover_parts(pathlib.Path(args.base_dir), pairs_dir=pairs_dir)
    n_rel = sum(1 for p in parts if p.d_geo is not None)
    print(f"relation matrices available: {n_rel}/{len(parts)} parts (bias mode: {args.relation_bias})")
    train_parts, val_parts = split_parts(parts, args.split_seed, args.val_fraction)
    mirror = tuple(a for a in args.mirror_axes.split(",") if a)
    train_ds = WireflowDataset(train_parts, args.n_points, mirror, base_seed=args.seed)
    val_ds = WireflowDataset(val_parts, args.n_points, (), base_seed=10_000_019)
    print(f"parts: train {len(train_parts)} (+mirror -> {len(train_ds)}), val {len(val_parts)}")
    (out / "split.json").write_text(json.dumps({
        "train": [p.name for p in train_parts], "val": [p.name for p in val_parts],
        "args": vars(args)}, indent=1), encoding="utf-8")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = WireFlowModel(args.dim, 8, args.enc_layers, args.dec_layers, args.n_latent).to(args.device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_param/1e6:.2f}M, device: {args.device}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best = {"val_loss_zon": float("inf"), "val_loss_zoff": float("inf")}
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        start_epoch = ckpt["epoch"] + 1
        hist_file = out / "history.json"
        if hist_file.exists():
            history = [r for r in json.loads(hist_file.read_text(encoding="utf-8"))
                       if r["epoch"] < start_epoch]
            zon = [r["val_zon_loss"] for r in history if "val_zon_loss" in r]
            zoff = [r["val_zoff_loss"] for r in history if "val_zoff_loss" in r]
            best["val_loss_zon"] = min(zon, default=float("inf"))
            best["val_loss_zoff"] = min(zoff, default=float("inf"))
        print(f"resumed from {args.resume} at epoch {start_epoch}, best={best}")
    for epoch in range(start_epoch, args.epochs + 1):
        train_ds.set_epoch(epoch)
        t0 = time.time()
        sums, n = {}, 0
        for batch in train_loader:
            batch = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            losses = model.training_losses(batch, args.z_dropout, args.joint_dropout,
                                           ot_coupling=args.ot_coupling,
                                           relation_mode=args.relation_bias)
            opt.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = batch["points"].shape[0]
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v) * bs
            n += bs
        row = {"epoch": epoch, **{f"train_{k}": v / n for k, v in sums.items()},
               "seconds": round(time.time() - t0, 1)}
        if epoch % args.val_every == 0 or epoch == args.epochs:
            zon = evaluate_fm_loss(model, val_loader, args.device, 0.0, args.ot_coupling,
                                   args.relation_bias)
            zoff = evaluate_fm_loss(model, val_loader, args.device, 1.0, args.ot_coupling,
                                    args.relation_bias)
            row.update({f"val_zon_{k}": v for k, v in zon.items()})
            row.update({f"val_zoff_{k}": v for k, v in zoff.items()})
            ckpt = {"model": model.state_dict(), "opt": opt.state_dict(),
                    "args": vars(args), "epoch": epoch}
            if zon["loss"] < best["val_loss_zon"]:
                best["val_loss_zon"] = zon["loss"]
                safe_save(ckpt, out / "best_zon.pt")
            if zoff["loss"] < best["val_loss_zoff"]:
                best["val_loss_zoff"] = zoff["loss"]
                safe_save(ckpt, out / "best_zoff.pt")
            safe_save(ckpt, out / "last.pt")
        history.append(row)
        (out / "history.json").write_text(json.dumps(history), encoding="utf-8")
        msg = f"epoch {epoch}: train {row.get('train_loss', 0):.4f}"
        if "val_zon_loss" in row:
            msg += f"  val z-on {row['val_zon_loss']:.4f}  z-off {row['val_zoff_loss']:.4f}"
        print(msg, flush=True)


if __name__ == "__main__":
    main()

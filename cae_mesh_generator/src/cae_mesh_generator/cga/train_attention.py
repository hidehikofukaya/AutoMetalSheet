"""Constraint Geometric Attention — Phase 1: learned pairwise geodesic predictor
vs Euclidean(+normal) baselines on held-out parts.

Success criterion (proposal §11): the learned model must rank geometrically
related constraint pairs better than the Euclidean+normal baseline on val parts.

Usage:
  python -m cae_mesh_generator.cga.train_attention \
      --dataset ../runs/cga_dataset_mesh --output-dir ../runs/cga_attention_v1
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import pathlib

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

TOP_K = 3


def load_parts(dataset_dir: pathlib.Path) -> list[dict]:
    parts = []
    for f in sorted((dataset_dir / "parts").glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        scale = float(d["scale"])
        parts.append({
            "name": f.stem,
            "xyz": ((d["jxyz"] - d["center"]) / scale).astype(np.float32),
            "dir": d["jdir"].astype(np.float32),
            "d_geo": (d["d_geo"] / scale).astype(np.float32),
            "d_euc": (d["d_euclid"] / scale).astype(np.float32),
        })
    return parts


class PairModel(nn.Module):
    """Small transformer over constraint nodes + symmetric pair head."""

    def __init__(self, dim: int = 128, heads: int = 4, layers: int = 2):
        super().__init__()
        self.embed = nn.Linear(6, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        # pair features: euclid dist, |normal dot|, delta along mean normal
        self.head = nn.Sequential(
            nn.Linear(dim * 2 + 3, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, xyz, ndir):
        h = self.encoder(self.embed(torch.cat([xyz, ndir], dim=-1)))  # (K,dim), K nodes
        K = h.shape[0]
        hi = h[:, None, :].expand(K, K, -1)
        hj = h[None, :, :].expand(K, K, -1)
        delta = xyz[:, None, :] - xyz[None, :, :]
        dist = delta.norm(dim=-1, keepdim=True)
        ndot = (ndir[:, None, :] * ndir[None, :, :]).sum(-1, keepdim=True).abs()
        along = (delta * (ndir[:, None, :] + ndir[None, :, :]) * 0.5).sum(-1, keepdim=True).abs()
        pair = torch.cat([hi + hj, (hi - hj).abs(), dist, ndot, along], dim=-1)
        pred = self.head(pair).squeeze(-1)
        return 0.5 * (pred + pred.T)  # symmetric predicted normalized geodesic


def rank_metrics(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    K = len(true)
    iu = np.triu_indices(K, 1)
    rho = spearmanr(pred[iu], true[iu]).statistic if K > 3 else np.nan
    recall = []
    for i in range(K):
        tg = set(np.argsort(true[i])[1: TOP_K + 1].tolist())
        tp = set(np.argsort(pred[i])[1: TOP_K + 1].tolist())
        if tg:
            recall.append(len(tg & tp) / len(tg))
    return float(rho), float(np.mean(recall))


def euclid_normal_baseline(train_parts, eval_part) -> np.ndarray:
    """Linear regression d_geo ~ [d_euc, d_euc*(1-|ndot|), 1] fit on train pairs."""
    feats, ys = [], []
    for p in train_parts:
        K = len(p["xyz"])
        iu = np.triu_indices(K, 1)
        ndot = np.abs(p["dir"] @ p["dir"].T)[iu]
        de = p["d_euc"][iu]
        feats.append(np.c_[de, de * (1 - ndot), np.ones_like(de)])
        ys.append(p["d_geo"][iu])
    X, y = np.concatenate(feats), np.concatenate(ys)
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    ndot = np.abs(eval_part["dir"] @ eval_part["dir"].T)
    de = eval_part["d_euc"]
    return w[0] * de + w[1] * de * (1 - ndot) + w[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--split-seed", type=int, default=13)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rank-weight", type=float, default=0.0,
                    help="InfoNCE weight pulling true top-k neighbours first")
    args = ap.parse_args()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    parts = load_parts(pathlib.Path(args.dataset))
    order = np.random.default_rng(args.split_seed).permutation(len(parts))
    n_val = max(1, int(round(len(parts) * args.val_fraction)))
    val_idx = set(order[:n_val].tolist())
    train = [p for i, p in enumerate(parts) if i not in val_idx]
    val = [p for i, p in enumerate(parts) if i in val_idx]
    print(f"parts: train {len(train)}, val {len(val)}")

    model = PairModel().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(train)
        tot = 0.0
        for p in train:
            xyz = torch.from_numpy(p["xyz"]).to(args.device)
            ndir = torch.from_numpy(p["dir"]).float().to(args.device)
            target = torch.from_numpy(p["d_geo"]).to(args.device)
            pred = model(xyz, ndir)
            loss = nn.functional.smooth_l1_loss(pred, target, beta=0.1)
            if args.rank_weight > 0 and len(xyz) > TOP_K + 2:
                # InfoNCE on -distance: pull true top-k neighbours above the rest
                K = len(xyz)
                eye = torch.eye(K, dtype=torch.bool, device=pred.device)
                logits = (-pred / 0.1).masked_fill(eye, -1e9)
                topk = target.masked_fill(eye, 1e9).argsort(dim=1)[:, :TOP_K]
                logp = torch.log_softmax(logits, dim=1)
                loss = loss - args.rank_weight * logp.gather(1, topk).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        if epoch % 20 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                vloss = float(np.mean([float(nn.functional.smooth_l1_loss(
                    model(torch.from_numpy(p["xyz"]).to(args.device),
                          torch.from_numpy(p["dir"]).float().to(args.device)),
                    torch.from_numpy(p["d_geo"]).to(args.device), beta=0.1)) for p in val]))
            print(f"epoch {epoch}: train {tot/len(train):.4f} val {vloss:.4f}", flush=True)
            if vloss < best_val:
                best_val = vloss
                torch.save(model.state_dict(), out / "best.pt")

    # ---- final evaluation: learned vs baselines on val parts ----
    model.load_state_dict(torch.load(out / "best.pt", map_location=args.device))
    model.eval()
    rows = []
    for p in val:
        with torch.no_grad():
            pred = model(torch.from_numpy(p["xyz"]).to(args.device),
                         torch.from_numpy(p["dir"]).float().to(args.device)).cpu().numpy()
        base_en = euclid_normal_baseline(train, p)
        r = {"part": p["name"], "n_joints": len(p["xyz"])}
        for name, mat in (("euclid", p["d_euc"]), ("euclid_normal", base_en), ("learned", pred)):
            rho, rec = rank_metrics(mat, p["d_geo"])
            r[f"{name}_spearman"] = rho
            r[f"{name}_top{TOP_K}"] = rec
        rows.append(r)
        print(f"{p['name'][:45]:45s} K={r['n_joints']:3d} "
              f"spearman e/en/L = {r['euclid_spearman']:.3f}/{r['euclid_normal_spearman']:.3f}/"
              f"{r['learned_spearman']:.3f}  top3 = {r['euclid_top3']:.2f}/"
              f"{r['euclid_normal_top3']:.2f}/{r['learned_top3']:.2f}", flush=True)

    agg = {"n_val_parts": len(rows)}
    for name in ("euclid", "euclid_normal", "learned"):
        agg[f"{name}_spearman_mean"] = float(np.nanmean([r[f"{name}_spearman"] for r in rows]))
        agg[f"{name}_top{TOP_K}_mean"] = float(np.mean([r[f"{name}_top{TOP_K}"] for r in rows]))
    print(json.dumps(agg, indent=1))
    (out / "eval.json").write_text(json.dumps({"aggregate": agg, "parts": rows}, indent=1),
                                   encoding="utf-8")


if __name__ == "__main__":
    main()

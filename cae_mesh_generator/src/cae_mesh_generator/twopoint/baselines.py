"""P0 baselines: condition-nearest retrieval and point-estimate regression MLP.

The flow model (P1) must beat regression on min-of-N and beat retrieval on
held-out interpolation to justify itself (session evaluation discipline).

Usage:
  python -m cae_mesh_generator.twopoint.baselines --output-dir ../runs/twopoint_p0
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

from .dataset import (CLASSES, THETA_NAMES, ThetaNormalizer, load_parts,
                      stratified_split, theta_mask)


def eval_theta(pred_theta, pred_cls, val_parts) -> dict:
    """Per-dim MAE in physical units over class-relevant dims + class accuracy."""
    per_dim = {k: [] for k in range(len(THETA_NAMES))}
    cls_ok = []
    for pt, pc, p in zip(pred_theta, pred_cls, val_parts):
        cls_ok.append(int(pc == p["cls"]))
        m = theta_mask(p["cls"])
        for k in range(len(THETA_NAMES)):
            if m[k]:
                per_dim[k].append(abs(float(pt[k]) - float(p["theta"][k])))
    out = {"class_accuracy": float(np.mean(cls_ok))}
    for k, name in enumerate(THETA_NAMES):
        if per_dim[k]:
            out[f"mae_{name}"] = float(np.mean(per_dim[k]))
    out["mae_mean_common"] = float(np.mean([out[f"mae_{THETA_NAMES[k]}"] for k in range(5)]))
    return out


class RegressionMLP(nn.Module):
    def __init__(self, cond_dim: int = 10, hidden: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU())
        self.theta_head = nn.Linear(hidden, len(THETA_NAMES))
        self.cls_head = nn.Linear(hidden, len(CLASSES))
        self.bits_head = nn.Linear(hidden, 2)

    def forward(self, cond):
        h = self.body(cond)
        return self.theta_head(h), self.cls_head(h), self.bits_head(h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--split-seed", type=int, default=13)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    parts = load_parts()
    train, val = stratified_split(parts, args.split_seed)
    print(f"parts: train {len(train)} val {len(val)} "
          f"(val flange {sum(p['cls'] == 0 for p in val)} / bead {sum(p['cls'] == 1 for p in val)})")
    norm = ThetaNormalizer(train)

    conds_tr = np.stack([p["cond"] for p in train])
    cond_mu, cond_sd = conds_tr.mean(0), conds_tr.std(0) + 1e-6

    def cz(c):
        return (c - cond_mu) / cond_sd

    # ---- retrieval baseline ----
    tr_z = cz(conds_tr)
    pred_theta, pred_cls = [], []
    for p in val:
        d = np.linalg.norm(tr_z - cz(p["cond"]), axis=1)
        nn_i = int(np.argmin(d))
        pred_theta.append(train[nn_i]["theta"])
        pred_cls.append(train[nn_i]["cls"])
    retrieval = eval_theta(pred_theta, pred_cls, val)

    # ---- regression baseline ----
    dev = args.device
    model = RegressionMLP().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    X = torch.tensor(cz(conds_tr), dtype=torch.float32, device=dev)
    Y = torch.tensor(np.stack([norm.encode(p["theta"]) for p in train]), device=dev)
    M = torch.tensor(np.stack([theta_mask(p["cls"]) for p in train]), device=dev)
    C = torch.tensor([p["cls"] for p in train], device=dev)
    B = torch.tensor([p["flange_bits"] for p in train], dtype=torch.float32, device=dev)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), 128):
            idx = perm[i:i + 128]
            th, cl, bits = model(X[idx])
            loss = (((th - Y[idx]) ** 2) * M[idx]).sum() / M[idx].sum()
            loss = loss + nn.functional.cross_entropy(cl, C[idx])
            fl = C[idx] == 0
            if fl.any():
                loss = loss + nn.functional.binary_cross_entropy_with_logits(
                    bits[fl], B[idx][fl])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        Xv = torch.tensor(np.stack([cz(p["cond"]) for p in val]),
                          dtype=torch.float32, device=dev)
        th, cl, _ = model(Xv)
        pred_theta = [norm.decode(t) for t in th.cpu().numpy()]
        pred_cls = cl.argmax(dim=1).cpu().numpy().tolist()
    regression = eval_theta(pred_theta, pred_cls, val)

    report = {"n_train": len(train), "n_val": len(val),
              "retrieval": retrieval, "regression": regression}
    (out / "baselines.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"{'metric':34s} {'retrieval':>10s} {'regression':>10s}")
    for k in sorted(set(retrieval) | set(regression)):
        print(f"{k:34s} {retrieval.get(k, float('nan')):10.3f} "
              f"{regression.get(k, float('nan')):10.3f}")


if __name__ == "__main__":
    main()

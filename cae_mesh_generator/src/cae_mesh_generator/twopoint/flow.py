"""P1: conditional flow matching over the two-point DOF vector.

p(c | C): small classifier (+flange bits). p(theta | C, c): rectified-flow MLP.
Evaluation: min-of-N masked MAE (vs the regression point estimate), per-dim
sample spread vs data spread (free DOF must reproduce the distribution, not
its mean), class sample distribution.

Usage:
  python -m cae_mesh_generator.twopoint.flow --output-dir ../runs/twopoint_p1 --device cuda
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
from .baselines import RegressionMLP, eval_theta

D = len(THETA_NAMES)


class FlowMLP(nn.Module):
    """v(theta_t, t | cond, class, flange_bits) with FiLM conditioning."""

    def __init__(self, cond_dim: int = 10, hidden: int = 512, blocks: int = 4,
                 **_):
        super().__init__()
        self.cond_net = nn.Sequential(
            nn.Linear(cond_dim + len(CLASSES) + 2 + 1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden))
        self.inp = nn.Linear(D, hidden)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            for _ in range(blocks))
        self.films = nn.ModuleList(nn.Linear(hidden, hidden * 2) for _ in range(blocks))
        self.out = nn.Linear(hidden, D)

    def forward(self, x_t, t, cond, cls_onehot, bits):
        c = self.cond_net(torch.cat([cond, cls_onehot, bits, t[:, None]], dim=1))
        h = self.inp(x_t)
        for blk, film in zip(self.blocks, self.films):
            scale, shift = film(c).chunk(2, dim=1)
            h = h + blk(h * (1 + scale) + shift)
        return self.out(h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--split-seed", type=int, default=13)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--cond-jitter", type=float, default=0.0,
                    help="Gaussian noise sigma on the normalized condition "
                         "(unique continuous conditions fingerprint each part; "
                         "jitter widens the empirical conditional back toward "
                         "the true one instead of a memorized delta)")
    ap.add_argument("--cond-dropout", type=float, default=0.0)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    dev = args.device

    parts = load_parts()
    train, val = stratified_split(parts, args.split_seed)
    norm = ThetaNormalizer(train)
    conds = np.stack([p["cond"] for p in train])
    mu, sd = conds.mean(0), conds.std(0) + 1e-6

    X = torch.tensor((conds - mu) / sd, dtype=torch.float32, device=dev)
    Y = torch.tensor(np.stack([norm.encode(p["theta"]) for p in train]), device=dev)
    M = torch.tensor(np.stack([theta_mask(p["cls"]) for p in train]), device=dev)
    C = torch.tensor([p["cls"] for p in train], device=dev)
    B = torch.tensor([p["flange_bits"] for p in train], dtype=torch.float32, device=dev)
    C1 = nn.functional.one_hot(C, len(CLASSES)).float()

    # ---- class head (reuse regression architecture, class/bits only) ----
    clf = RegressionMLP().to(dev)
    opt_c = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(300):
        perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), 128):
            idx = perm[i:i + 128]
            _, cl, bits = clf(X[idx])
            loss = nn.functional.cross_entropy(cl, C[idx])
            fl = C[idx] == 0
            if fl.any():
                loss = loss + nn.functional.binary_cross_entropy_with_logits(
                    bits[fl], B[idx][fl])
            opt_c.zero_grad()
            loss.backward()
            opt_c.step()

    # ---- flow ----
    flow = FlowMLP(hidden=args.hidden).to(dev)
    opt = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    for epoch in range(args.epochs):
        perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), args.batch):
            idx = perm[i:i + args.batch]
            y = Y[idx]
            x0 = torch.randn_like(y)
            t = torch.rand(len(idx), device=dev)
            x_t = (1 - t[:, None]) * x0 + t[:, None] * y
            xc = X[idx]
            if args.cond_jitter > 0:
                xc = xc + torch.randn_like(xc) * args.cond_jitter
            if args.cond_dropout > 0:
                keep = (torch.rand(len(idx), 1, device=dev) >= args.cond_dropout).float()
                xc = xc * keep
            v = flow(x_t, t, xc, C1[idx], B[idx])
            loss = ((v - (y - x0)) ** 2 * M[idx]).sum() / M[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if (epoch + 1) % 500 == 0:
            print(f"epoch {epoch+1}: fm loss {float(loss):.4f}", flush=True)
    torch.save({"flow": flow.state_dict(), "clf": clf.state_dict(),
                "norm_lo": norm.lo, "norm_hi": norm.hi, "cond_mu": mu, "cond_sd": sd,
                "args": vars(args)}, out / "model.pt")

    # ---- evaluation ----
    @torch.no_grad()
    def sample(cond_z, cls, bits, n, seed):
        g = torch.Generator(dev).manual_seed(seed)
        cz = cond_z[None].repeat(n, 1)
        c1 = nn.functional.one_hot(torch.full((n,), cls, device=dev, dtype=torch.long),
                                   len(CLASSES)).float()
        bt = bits[None].repeat(n, 1)
        x = torch.randn(n, D, device=dev, generator=g)
        dt = 1.0 / args.steps
        for k in range(args.steps):
            t = torch.full((n,), k * dt, device=dev)
            x = x + flow(x, t, cz, c1, bt) * dt
        return x

    flow.eval()
    clf.eval()
    rows = []
    minN_theta, minN_cls = [], []
    spread_ratio = {k: [] for k in range(D)}
    for vi, p in enumerate(val):
        cond_z = torch.tensor((p["cond"] - mu) / sd, dtype=torch.float32, device=dev)
        with torch.no_grad():
            _, cl, bits_l = clf(cond_z[None])
            cls_probs = torch.softmax(cl[0], dim=0)
        # sample classes from p(c|C), then theta from the flow
        g = torch.Generator(dev).manual_seed(1000 + vi)
        cls_samples = torch.multinomial(cls_probs, args.n_samples, replacement=True,
                                        generator=g)
        bits = (torch.sigmoid(bits_l[0]) > 0.5).float()
        thetas, classes = [], []
        for j, cs in enumerate(cls_samples.tolist()):
            z = sample(cond_z, cs, bits, 1, seed=100_000 * vi + 17 * j + cs)[0]
            thetas.append(norm.decode(z.cpu().numpy()))
            classes.append(cs)
        # min-of-N on samples with the GT class (fallback: all samples)
        cands = [(t, c) for t, c in zip(thetas, classes) if c == p["cls"]] or \
                list(zip(thetas, classes))
        m = theta_mask(p["cls"])
        errs = [np.abs(t[m] - p["theta"][m]).mean() for t, _ in cands]
        best = cands[int(np.argmin(errs))]
        minN_theta.append(best[0])
        minN_cls.append(best[1])
        rows.append({"part": p["part_id"], "gt_cls_fraction":
                     float(np.mean([c == p["cls"] for c in classes]))})
        same = [t for t, c in zip(thetas, classes) if c == p["cls"]]
        if len(same) >= 3:
            s = np.stack(same)
            for k in np.where(m)[0]:
                spread_ratio[k].append(float(s[:, k].std()))

    result = {"min_of_N": eval_theta(minN_theta, minN_cls, val),
              "gt_class_sample_fraction": float(np.mean([r["gt_cls_fraction"] for r in rows]))}
    # sample spread vs train data spread per dim (free DOF should be ~1)
    thetas_tr = np.stack([p["theta"] for p in train])
    masks_tr = np.stack([theta_mask(p["cls"]) for p in train])
    for k, name in enumerate(THETA_NAMES):
        vals = thetas_tr[masks_tr[:, k], k]
        if spread_ratio[k] and len(vals):
            result[f"spread_ratio_{name}"] = float(np.mean(spread_ratio[k]) / (vals.std() + 1e-9))
    (out / "eval.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

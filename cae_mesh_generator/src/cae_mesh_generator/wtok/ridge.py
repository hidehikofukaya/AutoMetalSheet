"""Ceiling test: can a model pull surface points onto the feature curves?

Given a PERFECT oriented cloud, predict for every point the displacement to the
nearest point on the outline and on the nearest bend line. Applying it drags the
points off the surface and onto the curves, where they form a traceable 1-D
chain. Geometry is handed in, so what comes out is the ceiling for curve
extraction sitting on top of a generator that is already good.

Why displacements and not labels or level sets -- both were tried and measured:
  labels + bonds on points   15.4mm   a curve is measure-zero in a surface, so a
                                      uniform sample lands on each one ~once
  level set of a distance field 18.3mm  the iso-line of a distance field is the
                                      curve OFFSET by the level; lower it and
                                      nothing crosses, raise it and it shifts in
  geometric nearest-point projection  coverage 9.8mm, spurious 0.27mm
                                      exact where it lands, but biased -- every
                                      point goes to its nearest curve point, so
                                      stretches that are nearest to nothing stay
                                      empty. A learned map does not have to be
                                      the nearest-point map, which is the whole
                                      reason to try this.

Numbers to beat, from the current pipeline: outline 5.73mm, bend 6.62mm.
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
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

from .codec import realize_edge, realize_points
from .connect import densify, trace
from .constants import safe_save, stable_seed
from .dataset_curve import load_curve_parts
from .evaluate_curve2 import one_way
from .plan_g import Block, RelBias
from .train_curve import realized_q

CLASSES = ("outer_boundary", "bend_line")
BAND = 0.12          # frame units: only points this close to a curve are pulled

# Trained on perfect clouds this net reaches 1.51/1.99mm, but on GENERATED ones
# it falls to 7.13/7.65 -- worse than the geometry it was meant to replace,
# because a generated cloud is out of its training distribution. Measured over
# 16 parts, that distribution is not noise at all:
#     total displacement          3.23mm
#     rigid, per-part mean        2.81mm   <- most of it: the cloud sits offset
#     residual                    3.19mm   normal 2.34mm / tangential 1.40mm
#     spatial correlation         0.755 at 5mm, 0.52 still at 80mm
# So the error is a smooth low-frequency warp plus a rigid shift, and the local
# structure the net actually reads survives it. That is reproducible from GT
# alone, without touching the generator.
# The net is a RESTORER, not a follower. Its job: given a cloud that is holed
# and noisy the way a generated one is, put out the wireframe of the shape that
# cloud is a sample OF. So the corruption here is only the recoverable kind --
# missing points, dropped patches, per-point noise, uneven density. It does NOT
# include the rigid offset and smooth warp that the generator also has (measured
# 2.81mm rigid, correlation 0.755 across 5mm): those cannot be inverted from the
# cloud alone, and they should not be. They surface as generator error when the
# restored frame is compared with the ground truth, which is where they belong.
HOLE_COUNT = (0, 6)        # patches removed
HOLE_RADIUS = (8.0, 30.0)  # mm
DROP_RATE = (0.0, 0.35)    # additional points thinned out at random
NOISE_MM = (0.0, 3.0)      # per-point jitter
DENSITY_SKEW = (0.0, 2.0)  # how unevenly the survivors are drawn


def corrupt(xyz, nrm, rng, strength=1.0):
    """Hole and roughen a clean cloud. Returns the surviving indices and the
    jittered positions; the TARGETS are untouched, because the wireframe wanted
    is the one of the original shape."""
    if strength <= 0:
        return np.arange(len(xyz)), xyz
    keep = np.ones(len(xyz), dtype=bool)

    for _ in range(int(rng.integers(*HOLE_COUNT)) if HOLE_COUNT[1] > HOLE_COUNT[0] else 0):
        centre = xyz[rng.integers(len(xyz))]
        r = float(rng.uniform(*HOLE_RADIUS)) * strength
        keep &= np.linalg.norm(xyz - centre, axis=1) > r

    skew = float(rng.uniform(*DENSITY_SKEW)) * strength
    if skew > 0:                       # uneven density: a smooth keep-probability
        ctrl = xyz[rng.choice(len(xyz), 6, replace=False)]
        d = np.linalg.norm(xyz[:, None, :] - ctrl[None], axis=-1)
        w = np.exp(-(d / 60.0) ** 2) @ rng.normal(0, 1, len(ctrl))
        w = (w - w.mean()) / max(w.std(), 1e-9)
        keep &= rng.random(len(xyz)) < 1.0 / (1.0 + np.exp(skew * w))

    drop = float(rng.uniform(*DROP_RATE)) * strength
    keep &= rng.random(len(xyz)) >= drop
    if keep.sum() < 32:                # never starve it completely
        keep = np.ones(len(xyz), dtype=bool)

    idx = np.flatnonzero(keep)
    noisy = xyz[idx] + rng.normal(0.0, float(rng.uniform(*NOISE_MM)) * strength,
                                  (len(idx), 3))
    return idx, noisy


def curve_points(part, cls):
    q = realized_q(part, part.vertices, part.edges)
    polys = [poly for poly in
             (realize_edge(q, e) for e in part.edges if e.get("cls") == cls)
             if poly is not None and len(poly) >= 2]
    return np.concatenate(polys) if polys else np.zeros((0, 3))


class RidgeDataset(Dataset):
    def __init__(self, parts, mesh_dir, n_pts, augment, base_seed=0,
                 degrade_max=0.0):
        self.parts, self.dir, self.n_pts = parts, pathlib.Path(mesh_dir), n_pts
        self.augment, self.base_seed, self.epoch = augment, base_seed, 0
        self.degrade_max = degrade_max
        self.cache: dict = {}

    def set_epoch(self, e):
        self.epoch = int(e)

    def __len__(self):
        return len(self.parts)

    def load(self, part):
        if part.name not in self.cache:
            d = np.load(self.dir / f"{part.name}.npz")
            span = np.maximum(d["env_hi"] - d["env_lo"], 1e-9)
            xyz = d["xyz"].astype(np.float64) * span + d["env_lo"]
            trees = [cKDTree(curve_points(part, c))
                     if len(curve_points(part, c)) else None for c in CLASSES]
            tgt = np.zeros((len(xyz), len(CLASSES), 3))
            for ci, t in enumerate(trees):
                if t is None:
                    tgt[:, ci] = xyz            # no such curve: zero displacement
                    continue
                tgt[:, ci] = t.data[t.query(xyz)[1]]
            self.cache[part.name] = (xyz, d["normal"].astype(np.float64), tgt)
        return self.cache[part.name]

    def __getitem__(self, i):
        p = self.parts[i]
        rng = np.random.default_rng(
            stable_seed(self.base_seed + 999983 * self.epoch, p.name))
        xyz, nrm, tgt = self.load(p)
        take = rng.choice(len(xyz), min(self.n_pts, len(xyz)), replace=False)
        x, n, t = xyz[take], nrm[take], tgt[take]
        c = x.mean(0)
        s = float(np.linalg.norm(x - c, axis=1).max())
        if self.augment:                       # mirror in the local frame
            ax = int(rng.integers(0, 4)) - 1
            if ax >= 0:
                x = x.copy(); n = n.copy(); t = t.copy()
                x[:, ax] = 2 * c[ax] - x[:, ax]
                n[:, ax] = -n[:, ax]
                t[:, :, ax] = 2 * c[ax] - t[:, :, ax]
        if self.degrade_max > 0:
            # holes and noise only; the targets stay on the ORIGINAL curves, so
            # the net learns to restore the shape's wireframe from a damaged
            # view of it
            idx, x = corrupt(x, n, rng, float(rng.uniform(0.0, self.degrade_max)))
            n, t = n[idx], t[idx]
        c = x.mean(0)
        s = float(np.linalg.norm(x - c, axis=1).max())
        disp = (t - x[:, None, :]) / s          # displacement in frame units
        feat = np.concatenate([(x - c) / s, n], 1).astype(np.float32)
        valid = np.ones(len(x), dtype=bool)
        if len(x) < self.n_pts:        # holing leaves fewer points; pad and mask
            pad = self.n_pts - len(x)
            take = rng.choice(len(x), pad, replace=True)
            feat = np.concatenate([feat, feat[take]])
            disp = np.concatenate([disp, disp[take]])
            valid = np.concatenate([valid, np.zeros(pad, dtype=bool)])
        elif len(x) > self.n_pts:
            sel = rng.choice(len(x), self.n_pts, replace=False)
            feat, disp, valid = feat[sel], disp[sel], valid[sel]
        return {"x": torch.from_numpy(feat),
                "disp": torch.from_numpy(disp.astype(np.float32)),
                "valid": torch.from_numpy(valid),
                "scale": np.float32(s), "center": c.astype(np.float32)}


class RidgeNet(nn.Module):
    """Per-point displacement onto each feature curve."""

    def __init__(self, dim=192, layers=6, heads=8):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.inp = nn.Linear(6, dim)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, len(CLASSES) * 3)
        self.rel = RelBias(heads)
        self.register_buffer("zero", torch.zeros(dim))

    def forward(self, x):
        B, N, _ = x.shape
        h = self.inp(x)
        bias = self.rel(x[..., :3]).reshape(B * self.heads, N, N).to(h.dtype)
        c = self.zero.expand(B, -1)
        for blk in self.blocks:
            h = blk(h, c, bias)
        return self.head(self.norm(h)).reshape(B, N, len(CLASSES), 3)


def loss_fn(model, batch):
    pred = model(batch["x"])
    tgt = batch["disp"]
    d = torch.linalg.norm(tgt, dim=-1)
    # a point 200mm from the outline carries no information about where the
    # outline is; only supervise the band that does
    w = (d < BAND).float() * batch["valid"][..., None].float()
    err = ((pred - tgt) ** 2).mean(-1)
    return (err * w).sum() / w.sum().clamp(min=1)


@torch.no_grad()
def score(model, parts, md, n_pts, device, max_parts=12, degrade_max=0.0):
    """Restoration: from a holed and noisy view, did the net put out the
    wireframe of the shape underneath? Corruption of this kind is invertible, so
    the original curves are the right target."""
    ds = RidgeDataset(parts[:max_parts], md, n_pts, False, 555,
                      degrade_max=degrade_max)
    out = {"outline": [], "bend": [], "spur": [], "curves": []}
    for i in range(len(ds)):
        it = ds[i]
        pred = model(it["x"][None].to(device))[0].cpu().numpy()
        s, c = float(it["scale"]), it["center"]
        pts = it["x"][:, :3].numpy() * s + c
        disp_gt = it["disp"].numpy()
        truth = pts[:, None, :] + disp_gt * s   # the ORIGINAL shape's curves
        band = (np.linalg.norm(disp_gt, axis=-1) < BAND) & it["valid"].numpy()[:, None]
        drawn = []
        for ci in range(len(CLASSES)):
            disp = pred[:, ci] * s
            keep = np.linalg.norm(disp, axis=1) < BAND * s
            if keep.sum() < 4:
                drawn.append(np.zeros((0, 3)))
                continue
            proj = pts[keep] + disp[keep]
            kk = min(13, len(proj))
            dd, nb = cKDTree(proj).query(proj, k=kk)
            pi = np.repeat(np.arange(len(proj)), kk - 1)
            pj = nb[:, 1:].reshape(-1)
            bond = (dd[:, 1:].reshape(-1) < s * 0.06).astype(float)
            drawn.append(densify(trace(proj, np.ones(len(proj), int), bond,
                                       np.stack([pi, pj], 1))))
        for ci in range(len(CLASSES)):
            ref = truth[band[:, ci], ci]
            if len(ref) and len(drawn[ci]):
                out["outline" if ci == 0 else "bend"].append(one_way(ref, drawn[ci])[0])
        allp = np.concatenate([d for d in drawn if len(d)])             if any(len(d) for d in drawn) else np.zeros((0, 3))
        ref_all = truth[band.any(1)].reshape(-1, 3)
        if len(allp) and len(ref_all):
            out["spur"].append(one_way(allp, ref_all)[0])
            out["curves"].append(len(allp))
    med = lambda k: float(np.median(out[k])) if out[k] else float("nan")
    return {k: med(k) for k in out}


def build_splits(args):
    parts = load_curve_parts(pathlib.Path(args.wtok))
    have = {f.stem for f in (pathlib.Path(args.dataset) / "parts").glob("*.npz")}
    parts = [p for p in parts if p.name in have]
    val = set(json.loads(pathlib.Path(args.val_list).read_text(encoding="utf-8")))
    return ([p for p in parts if p.name not in val],
            [p for p in parts if p.name in val])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--wtok", required=True)
    ap.add_argument("--val-list", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--points", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--degrade", type=float, default=0.0,
                    help="corrupt training clouds the way the generator does; "
                         "strength is drawn per item from [0, this]")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-parts", type=int, default=12)
    ap.add_argument("--max-hours", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    train_parts, val_parts = build_splits(args)
    md = pathlib.Path(args.dataset) / "parts"
    print(f"parts: train {len(train_parts)} val {len(val_parts)}  N={args.points}")
    tl = DataLoader(RidgeDataset(train_parts, md, args.points, True,
                                 degrade_max=args.degrade),
                    batch_size=args.batch_size, shuffle=True, drop_last=True)
    model = RidgeNet(args.dim, args.layers, args.heads).to(args.device)
    print(f"params: {sum(q.numel() for q in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, args.lr * 0.05)
    hist, best, t0 = [], float("inf"), time.time()
    for epoch in range(1, args.epochs + 1):
        tl.dataset.set_epoch(epoch)
        model.train()
        te, tot, n = time.time(), 0.0, 0
        for b in tl:
            loss = loss_fn(model, {k: (v.to(args.device) if torch.is_tensor(v) else v)
                                   for k, v in b.items()})
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            n += 1
        sched.step()
        row = {"epoch": epoch, "loss": tot / max(n, 1),
               "seconds": round(time.time() - te, 1)}
        msg = f"epoch {epoch}: loss {row['loss']:.5f} ({row['seconds']}s)"
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            row.update(score(model, val_parts, md, args.points, args.device,
                             args.eval_parts, degrade_max=args.degrade))
            msg += (f" | outline {row['outline']:.2f}mm bend {row['bend']:.2f}mm "
                    f"spur {row['spur']:.2f}mm")
            key = np.nanmean([row["outline"], row["bend"]])
            if key < best:
                best = key
                safe_save({"model": model.state_dict(), "args": vars(args),
                           "epoch": epoch}, out / "best.pt")
                msg += " *best*"
        hist.append(row)
        (out / "history.json").write_text(json.dumps(hist), encoding="utf-8")
        print(msg, flush=True)
        if (time.time() - t0) / 3600.0 > args.max_hours:
            print("stopping at the budget", flush=True)
            break
    print(f"\nCEILING: learned displacement on perfect clouds -> "
          f"mean(outline, bend) {best:.2f}mm "
          f"(current pipeline: outline 5.73mm bend 6.62mm)")


if __name__ == "__main__":
    main()

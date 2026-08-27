"""Ceiling test for learned connectivity.

The generator emits an unordered point set, so a wireframe needs edges. A purely
geometric tracer was measured first: even with perfect labels on a perfect
cloud it recovers only 34 curves out of 82 GT edges (coverage 6.22mm at N=512),
because distance-based clustering merges the parallel bead lines that run a few
millimetres apart. That is the number to beat.

This isolates the question: given a PERFECT cloud, how well can a learned model
say which points lie on a feature curve and which pairs are adjacent along it?
Geometry is handed in, so whatever comes out is the ceiling for connectivity on
top of a generator that is already good.

  per point : interior / outline / bend
  per kNN pair : is the segment between them running along one feature curve?

Pairs are scored only over the kNN graph -- at N=512, k=12 that is ~6k pairs,
the same order as the 512x3 coordinates, where a dense N x N would be 262k.
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
from .constants import safe_save, stable_seed
from .dataset_curve import load_curve_parts
from .evaluate_curve2 import one_way
from .plan_g import Block, RelBias
from .train_curve import realized_q

CLASSES = ("interior", "outer_boundary", "bend_line")
K_NN = 12
SEG_SAMPLES = 5


# ---------------------------------------------------------------- data

def gt_curves(part):
    """(polyline, class index) for every GT edge that realizes."""
    q = realized_q(part, part.vertices, part.edges)
    out = []
    for e in part.edges:
        poly = realize_edge(q, e)
        if poly is None or len(poly) < 2:
            continue
        cls = e.get("cls")
        out.append((poly, CLASSES.index(cls) if cls in CLASSES else 0))
    return out


def labels_and_bonds(xyz, curves, tol):
    """Per-point class, and for each kNN pair whether the straight segment
    between the two points runs along a single GT curve."""
    n = len(xyz)
    lab = np.zeros(n, dtype=np.int64)
    owner = np.full(n, -1, dtype=np.int64)
    if curves:
        allp = np.concatenate([c for c, _ in curves])
        who = np.concatenate([[i] * len(c) for i, (c, _) in enumerate(curves)])
        d, near = cKDTree(allp).query(xyz)
        on = d < tol
        owner[on] = who[near[on]]
        lab[on] = [curves[w][1] for w in who[near[on]]]

    kk = min(K_NN + 1, n)
    _, nb = cKDTree(xyz).query(xyz, k=kk)
    nb = nb[:, 1:]
    pi = np.repeat(np.arange(n), nb.shape[1])
    pj = nb.reshape(-1)
    bond = np.zeros(len(pi), dtype=np.float32)
    if curves:
        same = (owner[pi] >= 0) & (owner[pi] == owner[pj])
        idx = np.flatnonzero(same)
        if len(idx):
            a, b = xyz[pi[idx]], xyz[pj[idx]]
            t = np.linspace(0.0, 1.0, SEG_SAMPLES)[None, :, None]
            mid = a[:, None, :] + (b - a)[:, None, :] * t
            flat = mid.reshape(-1, 3)
            # the whole segment must hug the same curve, or a chord across a
            # bend would count as a bond
            keep = np.ones(len(idx), dtype=bool)
            for w in np.unique(owner[pi[idx]]):
                sel = np.flatnonzero(owner[pi[idx]] == w)
                dd, _ = cKDTree(curves[w][0]).query(
                    flat.reshape(len(idx), SEG_SAMPLES, 3)[sel].reshape(-1, 3))
                keep[sel] = dd.reshape(len(sel), SEG_SAMPLES).max(1) < tol * 1.5
            bond[idx[keep]] = 1.0
    return lab, np.stack([pi, pj], 1), bond


class ConnectDataset(Dataset):
    def __init__(self, parts, mesh_dir, n_pts, augment, base_seed=0):
        self.parts, self.dir, self.n_pts = parts, pathlib.Path(mesh_dir), n_pts
        self.augment, self.base_seed, self.epoch = augment, base_seed, 0
        self.cache: dict = {}

    def set_epoch(self, e):
        self.epoch = int(e)

    def __len__(self):
        return len(self.parts)

    def __getitem__(self, i):
        p = self.parts[i]
        rng = np.random.default_rng(
            stable_seed(self.base_seed + 999983 * self.epoch, p.name))
        if p.name not in self.cache:
            d = np.load(self.dir / f"{p.name}.npz")
            span = np.maximum(d["env_hi"] - d["env_lo"], 1e-9)
            self.cache[p.name] = (d["xyz"] * span + d["env_lo"],
                                  d["normal"].astype(np.float64),
                                  gt_curves(p))
        full, nrm, curves = self.cache[p.name]
        take = rng.choice(len(full), self.n_pts, replace=False)
        xyz, nn_ = full[take], nrm[take]
        # scale-free frame: centre and divide by the cloud's own radius, so the
        # model sees shape, not millimetres
        c = xyz.mean(0)
        s = float(np.linalg.norm(xyz - c, axis=1).max())
        tol = s * 0.02
        lab, pairs, bond = labels_and_bonds(xyz, curves, tol)
        x = np.concatenate([(xyz - c) / s, nn_], axis=1).astype(np.float32)
        return {"x": torch.from_numpy(x), "lab": torch.from_numpy(lab),
                "pairs": torch.from_numpy(pairs), "bond": torch.from_numpy(bond),
                "scale": np.float32(s), "center": c.astype(np.float32)}


# ---------------------------------------------------------------- model

class ConnectNet(nn.Module):
    def __init__(self, dim=192, layers=6, heads=8):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.inp = nn.Linear(6, dim)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.cls_head = nn.Linear(dim, len(CLASSES))
        self.pair = nn.Sequential(nn.Linear(2 * dim + 4, dim), nn.GELU(),
                                  nn.Linear(dim, 1))
        self.rel = RelBias(heads)
        self.zero = nn.Parameter(torch.zeros(dim), requires_grad=False)

    def forward(self, x, pairs):
        B, N, _ = x.shape
        h = self.inp(x)
        bias = self.rel(x[..., :3]).reshape(B * self.heads, N, N).to(h.dtype)
        c = self.zero.expand(B, -1)
        for blk in self.blocks:
            h = blk(h, c, bias)
        h = self.norm(h)
        logits = self.cls_head(h)
        bi = torch.arange(B, device=x.device)[:, None].expand(-1, pairs.shape[1])
        hi, hj = h[bi, pairs[..., 0]], h[bi, pairs[..., 1]]
        pi, pj = x[bi, pairs[..., 0], :3], x[bi, pairs[..., 1], :3]
        d = pj - pi
        geo = torch.cat([d, torch.linalg.norm(d, dim=-1, keepdim=True)], -1)
        # symmetric in (i, j): a bond has no direction
        pooled = torch.cat([hi + hj, (hi - hj).abs(), geo], dim=-1)
        return logits, self.pair(pooled).squeeze(-1)


def loss_fn(model, batch):
    logits, bond = model(batch["x"], batch["pairs"])
    ce = F.cross_entropy(logits.reshape(-1, len(CLASSES)), batch["lab"].reshape(-1))
    tgt = batch["bond"]
    pos = tgt.mean().clamp(1e-4, 1 - 1e-4)
    w = torch.where(tgt > 0.5, 1.0 / pos, 1.0 / (1 - pos))
    bce = (F.binary_cross_entropy_with_logits(bond, tgt, reduction="none")
           * w).mean()
    return ce + bce, ce.detach(), bce.detach()


# ---------------------------------------------------------------- tracing

def trace(xyz, cls, bond_p, pairs, thr=0.5):
    """predicted labels + bonds -> polylines, one per connected component"""
    keep = bond_p > thr
    parent = list(range(len(xyz)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (i, j) in pairs[keep]:
        if cls[i] == 0 or cls[j] == 0 or cls[i] != cls[j]:
            continue
        ra, rb = find(int(i)), find(int(j))
        if ra != rb:
            parent[ra] = rb
    groups: dict = {}
    for i in range(len(xyz)):
        if cls[i] != 0:
            groups.setdefault(find(i), []).append(i)
    out = []
    for g in groups.values():
        if len(g) < 3:
            continue
        pts = xyz[np.asarray(g)]
        c = pts - pts.mean(0)
        u = np.linalg.svd(c, full_matrices=False)[2][0]
        out.append(pts[np.argsort(c @ u)])
    return out


def densify(polys, per_seg=6):
    segs = [a + (b - a) * np.linspace(0, 1, per_seg)[:, None]
            for t in polys for a, b in zip(t[:-1], t[1:])]
    return np.concatenate(segs) if segs else np.zeros((0, 3))


# ---------------------------------------------------------------- stages

def build_splits(args):
    parts = load_curve_parts(pathlib.Path(args.wtok))
    have = {f.stem for f in (pathlib.Path(args.dataset) / "parts").glob("*.npz")}
    parts = [p for p in parts if p.name in have]
    val = set(json.loads(pathlib.Path(args.val_list).read_text(encoding="utf-8")))
    return ([p for p in parts if p.name not in val],
            [p for p in parts if p.name in val])


def to_device(b, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


@torch.no_grad()
def score(model, parts, md, n_pts, device, max_parts=20, thr=0.5):
    ds = ConnectDataset(parts[:max_parts], md, n_pts, False, 555)
    cov, spur, ncur, ngt, acc, bacc = [], [], [], [], [], []
    for i in range(len(ds)):
        it = ds[i]
        b = to_device({k: (v[None] if torch.is_tensor(v) else v)
                       for k, v in it.items()}, device)
        logits, bond = model(b["x"], b["pairs"])
        cls = logits[0].argmax(-1).cpu().numpy()
        pb = torch.sigmoid(bond[0]).cpu().numpy()
        acc.append(float((cls == it["lab"].numpy()).mean()))
        bacc.append(float(((pb > thr) == (it["bond"].numpy() > 0.5)).mean()))
        # undo the dataset's centre-and-scale so the polylines land in mm
        xyz = it["x"][:, :3].numpy() * float(it["scale"]) + it["center"]
        polys = trace(xyz, cls, pb, it["pairs"].numpy(), thr)
        p = parts[i]
        gt = realize_points(realized_q(p, p.vertices, p.edges))
        dense = densify(polys)
        if not len(dense):
            continue
        cov.append(one_way(gt, dense)[0])
        spur.append(one_way(dense, gt)[0])
        ncur.append(len(polys))
        ngt.append(len(gt_curves(p)))
    med = lambda a: float(np.median(a)) if a else float("nan")
    return {"coverage_mm": med(cov), "spurious_mm": med(spur),
            "curves": med(ncur), "gt_edges": med(ngt),
            "point_acc": med(acc), "bond_acc": med(bacc)}


def stage_train(args, out: pathlib.Path):
    train_parts, val_parts = build_splits(args)
    md = pathlib.Path(args.dataset) / "parts"
    print(f"parts: train {len(train_parts)} val {len(val_parts)}  N={args.points}")
    tl = DataLoader(ConnectDataset(train_parts, md, args.points, True),
                    batch_size=args.batch_size, shuffle=True, drop_last=True)
    model = ConnectNet(args.dim, args.layers, args.heads).to(args.device)
    print(f"params: {sum(q.numel() for q in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, args.lr * 0.05)
    hist, best, t0 = [], float("inf"), time.time()
    for epoch in range(1, args.epochs + 1):
        tl.dataset.set_epoch(epoch)
        model.train()
        te, tot, n = time.time(), 0.0, 0
        for b in tl:
            loss, ce, bce = loss_fn(model, to_device(b, args.device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            n += 1
        sched.step()
        row = {"epoch": epoch, "loss": tot / max(n, 1),
               "seconds": round(time.time() - te, 1)}
        msg = f"epoch {epoch}: loss {row['loss']:.4f} ({row['seconds']}s)"
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            row.update(score(model, val_parts, md, args.points, args.device,
                             args.eval_parts))
            msg += (f" | cover {row['coverage_mm']:.2f}mm spur "
                    f"{row['spurious_mm']:.2f}mm curves {row['curves']:.0f}/"
                    f"{row['gt_edges']:.0f} ptacc {row['point_acc']:.3f} "
                    f"bondacc {row['bond_acc']:.3f}")
            if row["coverage_mm"] < best:
                best = row["coverage_mm"]
                safe_save({"model": model.state_dict(), "args": vars(args),
                           "epoch": epoch}, out / "best.pt")
                msg += " *best*"
        hist.append(row)
        (out / "history.json").write_text(json.dumps(hist), encoding="utf-8")
        print(msg, flush=True)
        if (time.time() - t0) / 3600.0 > args.max_hours:
            print("stopping cleanly at the budget", flush=True)
            break
    print(f"\nCEILING: learned connectivity on perfect clouds -> "
          f"coverage {best:.2f}mm (geometric tracer: 6.22mm at N=512)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("train",), default="train")
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
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-parts", type=int, default=12)
    ap.add_argument("--max-hours", type=float, default=1.2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    stage_train(args, out)


if __name__ == "__main__":
    main()

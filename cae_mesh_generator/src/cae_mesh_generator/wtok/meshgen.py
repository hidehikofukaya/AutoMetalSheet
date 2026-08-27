"""Generate the midsurface as an oriented point cloud, then extract the wire.

The wire is a sparse combinatorial object, and every failure measured so far was
combinatorial: a third of B-core's edges referenced vertices that did not exist.
A surface sample has no connectivity to get wrong -- only coordinates, which is
the one paradigm that has worked here (flow matching). The wire comes out
afterwards from generic geometry (mesh_extract), measured at ~4mm even with 3mm
of noise, against generators that are at 23-31mm.

  stage "train": conditional flow matching over N points x (xyz, normal)
  stage "eval" : sample -> extract -> the session's standard decomposition
  stage "smoke": self-check

Anchors: the fastening points lie ON the midsurface (measured 0.78mm to the
nearest of 4096 samples), so slots 0..n_fix-1 are pinned to them at zero noise
for every step -- R1, and here it needs no incidence rule because a point IS the
surface.
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
from torch.utils.data import DataLoader, Dataset

from .codec import bin_center, realize_points
from .constants import safe_save, stable_seed
from .dataset_ar import cond_features
from .dataset_curve import load_curve_parts
from .mesh_extract import extract
from .plan_g import COND_ROWS, Block, RelBias, timestep_embedding
from .train_ar import chamfer_mm
from .train_curve import realized_q

CH = 6                       # xyz + normal


# ---------------------------------------------------------------- data

def part_cond_rows(part, vertices=None) -> np.ndarray:
    vs = vertices if vertices is not None else part.vertices
    rows = cond_features([v for v in vs if v["T"] == "FIX"], part.env_lo, part.env_hi)
    out = np.zeros((COND_ROWS, 8), dtype=np.float32)
    out[: min(COND_ROWS, len(rows))] = rows[:COND_ROWS]
    return out


def fix_xyz_norm(part, vertices=None) -> np.ndarray:
    """FIX positions in the envelope frame ([0,1]), the same frame as the cloud."""
    vs = vertices if vertices is not None else part.vertices
    from .plan_g import N_BINS
    return np.asarray([(np.asarray(v["bin"], np.float64) + 0.5) / N_BINS
                       for v in vs if v["T"] == "FIX"], dtype=np.float32)


def mirror(xyz, normal, fix, cond_rows, axis):
    """Mirror in the normalized envelope frame: position reflects, normal flips.
    The condition rows carry FIX xyz in columns 0..2 and the axis in 3..5."""
    xyz, normal, fix = xyz.copy(), normal.copy(), fix.copy()
    xyz[:, axis] = 1.0 - xyz[:, axis]
    normal[:, axis] = -normal[:, axis]
    fix[:, axis] = 1.0 - fix[:, axis]
    c = cond_rows.copy()
    is_fix = c[:, 6] > 0.5
    c[is_fix, axis] = 1.0 - c[is_fix, axis]
    c[is_fix, 3 + axis] = -c[is_fix, 3 + axis]
    return xyz, normal, fix, c


class MeshDataset(Dataset):
    def __init__(self, parts, mesh_dir: pathlib.Path, n_pts: int,
                 augment: bool, base_seed: int = 0):
        self.parts = parts
        self.dir = pathlib.Path(mesh_dir)
        self.n_pts = n_pts
        self.augment = augment
        self.base_seed = base_seed
        self.epoch = 0
        self.cache: dict = {}

    def set_epoch(self, e: int) -> None:
        self.epoch = int(e)

    def __len__(self) -> int:
        return len(self.parts)

    def load(self, name):
        if name not in self.cache:
            d = np.load(self.dir / f"{name}.npz")
            self.cache[name] = (d["xyz"].astype(np.float32),
                                d["normal"].astype(np.float32))
        return self.cache[name]

    def __getitem__(self, i: int) -> dict:
        p = self.parts[i]
        rng = np.random.default_rng(
            stable_seed(self.base_seed + 999983 * self.epoch, p.name))
        xyz, nrm = self.load(p.name)
        fix = fix_xyz_norm(p)
        cond = part_cond_rows(p)
        if self.augment:
            axis = rng.integers(0, 4) - 1
            if axis >= 0:
                xyz, nrm, fix, cond = mirror(xyz, nrm, fix, cond, int(axis))
        n_fix = min(len(fix), self.n_pts)
        take = rng.choice(len(xyz), self.n_pts - n_fix, replace=False)
        # anchors occupy the first slots; every other slot is a random surface
        # sample, so slot order carries no information the model could lean on
        pts = np.concatenate([fix[:n_fix], xyz[take]])
        nn_ = np.concatenate([nrm[cKD(xyz, fix[:n_fix])], nrm[take]])
        return {"x": torch.from_numpy(np.concatenate([pts, nn_], axis=1)),
                "cond": torch.from_numpy(cond),
                "n_fix": n_fix}


def cKD(xyz, q):
    if len(q) == 0:
        return np.zeros(0, dtype=np.int64)
    d = np.linalg.norm(xyz[None, :, :] - q[:, None, :], axis=-1)
    return d.argmin(axis=1)


# ---------------------------------------------------------------- model

class PointFlow(nn.Module):
    """v(x_t, t | condition) over a permutation-equivariant set of points."""

    def __init__(self, dim=256, layers=8, heads=8, rel_attn=True):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.inp = nn.Linear(CH, dim)
        self.cond_proj = nn.Linear(8, dim)
        self.null_cond = nn.Parameter(torch.zeros(COND_ROWS, dim))
        self.seg = nn.Parameter(torch.zeros(2, dim))
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, CH)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.rel = RelBias(heads) if rel_attn else None

    def forward(self, x, t, cond, drop=None):
        B, N, _ = x.shape
        c = self.cond_proj(cond)
        if drop is not None:
            c = torch.where(drop[:, None, None], self.null_cond, c)
        c = c + self.seg[0]
        h = torch.cat([c, self.inp(x) + self.seg[1]], dim=1)
        g = self.t_mlp(timestep_embedding(t, self.dim)) + c.mean(1)
        bias = None
        if self.rel is not None:
            S = h.shape[1]
            m = torch.zeros(B, self.heads, S, S, device=x.device, dtype=h.dtype)
            m[:, :, COND_ROWS:, COND_ROWS:] = self.rel(x[..., :3]).to(h.dtype)
            bias = m.reshape(B * self.heads, S, S)
        for blk in self.blocks:
            h = blk(h, g, bias)
        return self.head(self.norm(h[:, COND_ROWS:]))


def anchor_keep(n_fix: torch.Tensor, N: int) -> torch.Tensor:
    return torch.arange(N, device=n_fix.device)[None] < n_fix[:, None]


def flow_loss(model, batch, anchor: bool, p_drop: float, w_normal: float = 0.2):
    x1 = batch["x"] * 2.0 - 1.0                 # xyz in [0,1] -> [-1,1]
    x1 = torch.cat([x1[..., :3], batch["x"][..., 3:]], dim=-1)   # normals already signed
    x0 = torch.randn_like(x1)
    B, N, _ = x1.shape
    t = torch.rand(B, device=x1.device)
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    keep = anchor_keep(batch["n_fix"], N)
    if anchor:
        pos = torch.where(keep[..., None], x1, xt)
        xt = torch.cat([pos[..., :3], xt[..., 3:]], dim=-1)   # pin position only
    drop = (torch.rand(B, device=x1.device) < p_drop) if (p_drop > 0 and model.training) else None
    v = model(xt, t, batch["cond"], drop)
    target = x1 - x0
    err = (v - target) ** 2
    w = torch.cat([torch.ones(3, device=x1.device),
                   torch.full((3,), w_normal, device=x1.device)])
    err = (err * w).mean(-1)
    if anchor:
        err = err * (~keep)
        return err.sum() / (~keep).sum().clamp(min=1)
    return err.mean()


@torch.no_grad()
def sample(model, cond, fix_xyz, steps=48, gen=None, scale=1.0, n_pts=256):
    dev = cond.device
    B = cond.shape[0]
    x = torch.randn(B, n_pts, CH, device=dev, generator=gen)
    n_fix = fix_xyz.shape[1] if fix_xyz is not None else 0
    if n_fix:
        x[:, :n_fix, :3] = fix_xyz * 2.0 - 1.0
    dt = 1.0 / steps
    for k in range(steps):
        t = torch.full((B,), k * dt, device=dev)
        if scale > 1.0:
            keep = torch.zeros(B, dtype=torch.bool, device=dev)
            v = model(x, t, cond, ~keep)
            v = v + scale * (model(x, t, cond, keep) - v)
        else:
            v = model(x, t, cond)
        x = x + v * dt
        if n_fix:
            x[:, :n_fix, :3] = fix_xyz * 2.0 - 1.0
    return x


def to_world(x, part):
    """model output -> (points mm, unit normals)"""
    lo = np.asarray(part.env_lo, dtype=np.float64)
    span = np.maximum(np.asarray(part.env_hi, dtype=np.float64) - lo, 1e-9)
    xyz = ((x[..., :3].clamp(-1, 1).cpu().numpy() + 1.0) / 2.0) * span + lo
    nrm = x[..., 3:].cpu().numpy()
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-9)
    return xyz, nrm


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


def stage_train(args, out: pathlib.Path) -> None:
    train_parts, val_parts = build_splits(args)
    print(f"parts: train {len(train_parts)} val {len(val_parts)}  N={args.points}")
    md = pathlib.Path(args.dataset) / "parts"
    tl = DataLoader(MeshDataset(train_parts, md, args.points, True),
                    batch_size=args.batch_size, shuffle=True, drop_last=True)
    vl = DataLoader(MeshDataset(val_parts, md, args.points, False, 555),
                    batch_size=args.batch_size)
    model = PointFlow(args.dim, args.layers, args.heads, bool(args.rel_attn)).to(args.device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, args.lr * 0.05)
    hist_path = out / "history.json"
    history, best, start = [], float("inf"), 1
    last = out / "last.pt"
    if args.resume and last.exists():
        ck = torch.load(last, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["epoch"] + 1
        if hist_path.exists():
            history = [r for r in json.loads(hist_path.read_text()) if r["epoch"] < start]
            best = min((r["val"] for r in history if "val" in r), default=float("inf"))
        for _ in range(start - 1):
            sched.step()
        print(f"resumed at {start} (best {best:.5f})", flush=True)
    t_start = time.time()
    for epoch in range(start, args.epochs + 1):
        tl.dataset.set_epoch(epoch)
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for b in tl:
            loss = flow_loss(model, to_device(b, args.device), args.anchor, args.cfg_drop)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            n += 1
        sched.step()
        row = {"epoch": epoch, "train": tot / max(n, 1),
               "seconds": round(time.time() - t0, 1)}
        if epoch % args.val_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                row["val"] = float(np.mean(
                    [float(flow_loss(model, to_device(b, args.device), args.anchor, 0.0))
                     for b in vl]))
            ck = {"model": model.state_dict(), "opt": opt.state_dict(),
                  "args": vars(args), "epoch": epoch}
            if row["val"] < best:
                best = row["val"]
                safe_save(ck, out / "best.pt")
            safe_save(ck, last)
        history.append(row)
        hist_path.write_text(json.dumps(history), encoding="utf-8")
        msg = f"epoch {epoch}: loss {row['train']:.5f} ({row['seconds']}s)"
        if "val" in row:
            msg += f"  val {row['val']:.5f}"
        print(msg, flush=True)
        if (time.time() - t_start) / 3600.0 > args.max_hours:
            print(f"stopping cleanly at the {args.max_hours}h budget", flush=True)
            break


@torch.no_grad()
def generate(model, part, device, steps, seed, scale, n_pts, anchor=True):
    gen = torch.Generator(device=device).manual_seed(seed)
    cond = torch.from_numpy(part_cond_rows(part))[None].to(device)
    fx = torch.from_numpy(fix_xyz_norm(part))[None].to(device) if anchor else None
    x = sample(model, cond, fx, steps, gen, scale, n_pts)
    return to_world(x[0], part)


def stage_eval(args, out: pathlib.Path) -> None:
    from .evaluate_curve2 import class_points, one_way
    _, val_parts = build_splits(args)
    ck = torch.load(out / "best.pt", map_location=args.device, weights_only=False)
    a = argparse.Namespace(**ck["args"])
    model = PointFlow(a.dim, a.layers, a.heads, bool(a.rel_attn)).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    md = pathlib.Path(args.dataset) / "parts"
    rows = []
    for p in val_parts[: args.eval_parts]:
        d = np.load(md / f"{p.name}.npz")
        lo, hi = d["env_lo"], d["env_hi"]
        gt_surf = d["xyz"] * np.maximum(hi - lo, 1e-9) + lo
        gt_wire = realize_points(realized_q(p, p.vertices, p.edges))
        fix_mm = [bin_center(realized_q(p, p.vertices, p.edges), v)
                  for v in p.vertices if v["T"] == "FIX"]
        xyz, nrm = generate(model, p, args.device, args.steps, 7, args.cfg_scale,
                            a.points, args.anchor)
        f = extract(xyz, nrm)
        bp, cp = xyz[f["boundary"]], xyz[f["crease"]]
        wire = np.concatenate([x for x in (bp, cp) if len(x)]) \
            if len(bp) or len(cp) else np.zeros((0, 3))
        row = {"part": p.name,
               "surface_chamfer_mm": chamfer_mm(xyz, gt_surf),
               "n_boundary": len(bp), "n_crease": len(cp),
               "fix_err_mm": float(np.mean(
                   [np.min(np.linalg.norm(xyz - q, axis=1)) for q in fix_mm]))}
        if len(wire):
            row["wire_chamfer_mm"] = chamfer_mm(wire, gt_wire)
            row["outline_mm"] = one_way(class_points(p, {"outer_boundary"}), bp)[0] \
                if len(bp) else float("nan")
            row["bend_mm"] = one_way(class_points(p, {"bend_line"}), cp)[0] \
                if len(cp) else float("nan")
            row["spurious_mm"] = one_way(wire, gt_wire)[0]
        rows.append(row)
        print(f"  {p.name[-8:]}: surf {row['surface_chamfer_mm']:.1f}mm  "
              f"wire {row.get('wire_chamfer_mm', float('nan')):.1f}mm  "
              f"fix {row['fix_err_mm']:.2f}mm  b/c={len(bp)}/{len(cp)}", flush=True)

    def med(k):
        v = [r[k] for r in rows if k in r and np.isfinite(r[k])]
        return float(np.median(v)) if v else float("nan")

    summary = {"parts": len(rows), "points": a.points, "steps": args.steps,
               "cfg_scale": args.cfg_scale,
               **{k: round(med(k), 2) for k in
                  ("surface_chamfer_mm", "wire_chamfer_mm", "outline_mm",
                   "bend_mm", "spurious_mm", "fix_err_mm")}}
    print(json.dumps(summary, indent=1))
    (out / "eval.json").write_text(json.dumps({"summary": summary, "rows": rows},
                                              indent=1), encoding="utf-8")


def stage_smoke(args, out: pathlib.Path) -> None:
    train_parts, _ = build_splits(args)
    parts = train_parts[:24]
    md = pathlib.Path(args.dataset) / "parts"
    ds = MeshDataset(parts, md, args.points, True)
    loader = DataLoader(ds, batch_size=4, shuffle=True, drop_last=True)
    model = PointFlow(args.dim, args.layers, args.heads, bool(args.rel_attn)).to(args.device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for epoch in range(8):
        ds.set_epoch(epoch)
        for b in loader:
            loss = flow_loss(model, to_device(b, args.device), args.anchor, args.cfg_drop)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
    head, tail = np.mean(losses[:5]), np.mean(losses[-5:])
    assert tail < head, "did not learn"
    print(f"loss {head:.4f} -> {tail:.4f}")

    p = parts[0]
    xyz, nrm = generate(model, p, args.device, 8, 1, args.cfg_scale, args.points,
                        args.anchor)
    f = extract(xyz, nrm)
    fixq = [bin_center(realized_q(p, p.vertices, p.edges), v)
            for v in p.vertices if v["T"] == "FIX"]
    err = float(np.mean([np.min(np.linalg.norm(xyz - q, axis=1)) for q in fixq]))
    print(f"generated {len(xyz)} pts, boundary={len(f['boundary'])} "
          f"crease={len(f['crease'])}, fix_err={err:.3f}mm")
    if args.anchor:
        assert err < 1e-3, f"anchor pinning failed ({err:.4f}mm)"
    print("smoke ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("train", "eval", "smoke"), required=True)
    ap.add_argument("--dataset", required=True, help="runs/mesh_synth")
    ap.add_argument("--wtok", required=True, help="runs/wtok_synth (conditions + GT wire)")
    ap.add_argument("--val-list", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--points", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--rel-attn", type=int, default=1)
    ap.add_argument("--anchor", type=int, default=1)
    ap.add_argument("--cfg-drop", type=float, default=0.1)
    ap.add_argument("--cfg-scale", type=float, default=1.5)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--eval-parts", type=int, default=40)
    ap.add_argument("--max-hours", type=float, default=0.55)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    {"train": stage_train, "eval": stage_eval, "smoke": stage_smoke}[args.stage](args, out)


if __name__ == "__main__":
    main()

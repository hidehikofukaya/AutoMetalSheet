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

def fix_points_mm(part) -> tuple[np.ndarray, np.ndarray]:
    """Fastening points and their axes in mm. The bins/envelope here are only the
    storage encoding of a position that is itself legitimate input."""
    from .plan_g import N_BINS
    lo = np.asarray(part.env_lo, dtype=np.float64)
    span = np.asarray(part.env_hi, dtype=np.float64) - lo
    fx = [v for v in part.vertices if v["T"] == "FIX"]
    P = np.stack([lo + (np.asarray(v["bin"], np.float64) + 0.5) / N_BINS * span
                  for v in fx])
    A = np.stack([np.asarray(v["nf"] if v["nf"] is not None else [0, 0, 1],
                             dtype=np.float64) for v in fx])
    return P, A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)


def fastener_frame(part):
    """Canonical frame from the fastening points ALONE -- no part geometry.

      origin : midpoint of the two fasteners
      unit   : the distance between them (part diagonal / this distance is
               1.21-1.69 across the set, so frame coordinates land near +-0.85)
      e1     : fastener-to-fastener direction
      e2     : whichever fastener axis is furthest from e1, orthogonalised
               (measured: at least 17.5 deg for every part, so never degenerate)

    Replaces the envelope frame, which was the ground-truth bounding box and so
    handed the model the part's position, orientation and extent.
    """
    P, A = fix_points_mm(part)
    if len(P) < 2:
        R = np.eye(3)
        return (P[0] if len(P) else np.zeros(3)), R, 1.0
    d = float(np.linalg.norm(P[1] - P[0]))
    e1 = (P[1] - P[0]) / max(d, 1e-12)
    cos = np.abs(A @ e1)
    a = A[int(np.argmin(cos))]
    e2 = a - (a @ e1) * e1
    n2 = np.linalg.norm(e2)
    if n2 < 1e-6:                       # never hit on this data; stay defined
        alt = np.array([1.0, 0.0, 0.0])
        if abs(alt @ e1) > 0.9:
            alt = np.array([0.0, 1.0, 0.0])
        e2 = alt - (alt @ e1) * e1
        n2 = np.linalg.norm(e2)
    e2 /= n2
    return P.mean(0), np.stack([e1, e2, np.cross(e1, e2)]), max(d, 1e-9)


def to_frame(xyz, normal, frame):
    o, R, d = frame
    return ((xyz - o) @ R.T) / d, normal @ R.T


def from_frame(xyz, normal, frame):
    o, R, d = frame
    return (xyz * d) @ R + o, normal @ R


def frame_cond_rows(part) -> np.ndarray:
    """Fastening points only: their position and axis expressed in their own
    frame, plus log of their separation. No envelope, no part geometry."""
    P, A = fix_points_mm(part)
    frame = fastener_frame(part)
    Pf, Af = to_frame(P, A, frame)
    out = np.zeros((COND_ROWS, 8), dtype=np.float32)
    for i in range(min(len(Pf), COND_ROWS - 1)):
        out[i] = np.concatenate([Pf[i], Af[i], [1.0, 0.0]])
    out[COND_ROWS - 1] = np.concatenate(
        [[np.log(frame[2]) / 10.0], np.zeros(6), [1.0]])
    return out


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


ANCHOR_R = 10.0        # mm; measured exact out to here (see fastener_disc)
ANCHOR_PER_FIX = 8


def fastener_disc(part, per_fix: int = ANCHOR_PER_FIX, r: float = ANCHOR_R,
                  rng=None):
    """Points the fastener itself determines, SYNTHESISED -- never read from the
    part.

    A bolt needs a flat seat normal to its axis, and the data agrees exactly: over
    240 fastener sites the real surface within 10mm of a fastening point deviates
    from the ideal disc by 0.00mm at the 95th percentile, with 0.0 deg of normal
    tilt. (At 15mm the 95th percentile tilt is already 49 deg, so 10mm is the
    honest radius.) Feeding these in is the fastening point drawn out, not
    ground truth: they are generated from the point and its axis alone.
    """
    rng = rng or np.random.default_rng(0)
    P, A = fix_points_mm(part)
    pts, nrm = [], []
    for c, ax in zip(P, A):
        e1 = np.array([1.0, 0.0, 0.0])
        if abs(e1 @ ax) > 0.9:
            e1 = np.array([0.0, 1.0, 0.0])
        e1 = e1 - (e1 @ ax) * ax
        e1 /= max(np.linalg.norm(e1), 1e-12)
        e2 = np.cross(ax, e1)
        rad = r * np.sqrt(rng.random(per_fix))       # uniform over the area
        th = rng.random(per_fix) * 2 * np.pi
        rad[0], th[0] = 0.0, 0.0                     # keep the exact centre
        pts.append(c + rad[:, None] * (np.cos(th)[:, None] * e1
                                       + np.sin(th)[:, None] * e2))
        nrm.append(np.tile(ax, (per_fix, 1)))
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(pts), np.concatenate(nrm)


class MeshDataset(Dataset):
    def __init__(self, parts, mesh_dir: pathlib.Path, n_pts: int,
                 augment: bool, base_seed: int = 0,
                 anchor_per_fix: int = ANCHOR_PER_FIX,
                 mask_rate: float = 0.0, mask_max: float = 0.5):
        self.anchor_per_fix = anchor_per_fix
        self.mask_rate = mask_rate
        self.mask_max = mask_max
        self.parts = parts
        self.dir = pathlib.Path(mesh_dir)
        self.n_pts = n_pts
        self.augment = augment
        self.base_seed = base_seed
        self.epoch = 0
        self.cache: dict = {}
        self.outside: dict = {}

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
        xyz_n, nrm = self.load(p.name)
        lo = np.asarray(p.env_lo, dtype=np.float64)
        span = np.maximum(np.asarray(p.env_hi, dtype=np.float64) - lo, 1e-9)
        xyz_mm = xyz_n.astype(np.float64) * span + lo      # envelope: storage only
        frame = fastener_frame(p)
        xyz, nrm = to_frame(xyz_mm, nrm.astype(np.float64), frame)
        dp, dn = fastener_disc(p, self.anchor_per_fix, rng=rng)
        fix, fixn = to_frame(dp, dn, frame)
        cond = frame_cond_rows(p)
        if self.anchor_per_fix:
            # free slots come from outside the disc so the density near a
            # fastener matches the rest of the sheet. The disc is centred on the
            # fasteners, so this only needs the distance to those two points --
            # and it never changes, so cache it per part.
            keep_idx = self.outside.get(p.name)
            if keep_idx is None:
                centres, _ = to_frame(*fix_points_mm(p), frame)
                far = np.linalg.norm(
                    xyz[:, None, :] - centres[None, :, :], axis=-1).min(1)
                keep_idx = np.flatnonzero(far > ANCHOR_R / frame[2])
                self.outside[p.name] = keep_idx
            if len(keep_idx) > self.n_pts:
                xyz, nrm = xyz[keep_idx], nrm[keep_idx]
        if self.augment and rng.random() < 0.5:
            # the frame fixes e1 and e2, so only the e3 component is free to
            # mirror -- flipping the others would just redefine the frame
            xyz, nrm, fix = xyz.copy(), nrm.copy(), fix.copy()
            xyz[:, 2] *= -1.0
            nrm[:, 2] *= -1.0
            fix[:, 2] *= -1.0
            fixn = fixn.copy()
            fixn[:, 2] *= -1.0
            cond = cond.copy()
            cond[:COND_ROWS - 1, 2] *= -1.0
            cond[:COND_ROWS - 1, 5] *= -1.0
        n_fix = min(len(fix), self.n_pts)
        take = rng.choice(len(xyz), self.n_pts - n_fix, replace=False)
        # anchors occupy the first slots; every other slot is a random surface
        # sample, so slot order carries no information the model could lean on
        pts = np.concatenate([fix[:n_fix], xyz[take]])
        nn_ = np.concatenate([fixn[:n_fix], nrm[take]])
        # Masked training: pin a random extra set of surface points as "known"
        # and let the model infer the rest. Pinning is already how anchors work,
        # so a constraint point and a fastening point are the same mechanism --
        # this keeps the model usable with any set of known points at inference.
        if self.mask_rate > 0 and rng.random() < self.mask_rate:
            free = self.n_pts - n_fix
            k = int(rng.integers(1, max(2, int(free * self.mask_max))))
            sel = n_fix + rng.permutation(free)[:k]
            order = np.concatenate([np.arange(n_fix), sel,
                                    np.setdiff1d(np.arange(n_fix, self.n_pts), sel)])
            pts, nn_ = pts[order], nn_[order]
            n_fix += k
        return {"x": torch.from_numpy(
                    np.concatenate([pts, nn_], axis=1).astype(np.float32)),
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
    x1 = batch["x"]          # already in the fastener frame: coords ~ +-0.85
    x0 = torch.randn_like(x1)
    B, N, _ = x1.shape
    t = torch.rand(B, device=x1.device)
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    keep = anchor_keep(batch["n_fix"], N)
    if anchor:
        xt = torch.where(keep[..., None], x1, xt)   # the disc fixes both
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
        x[:, :n_fix] = fix_xyz            # position and normal are both implied
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
            x[:, :n_fix] = fix_xyz
    return x


def to_world(x, part):
    """model output (fastener frame) -> (points mm, unit normals). The envelope
    is never consulted: the frame comes from the fastening points alone."""
    xyz = x[..., :3].cpu().numpy().astype(np.float64)
    nrm = x[..., 3:].cpu().numpy().astype(np.float64)
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-9)
    return from_frame(xyz, nrm, fastener_frame(part))


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
    tl = DataLoader(MeshDataset(train_parts, md, args.points, True,
                                mask_rate=args.mask_rate),
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
            best = min((r["surf_mm"] for r in history if "surf_mm" in r),
                       default=float("inf"))
        for _ in range(start - 1):
            sched.step()
        print(f"resumed at {start} (best {best:.5f})", flush=True)
    # the loss is a conditional variance and plateaus long before the geometry
    # does, so probe the actual surface error during training as well
    probe_gt = {}
    for p in val_parts[: args.probe_parts]:
        d = np.load(md / f"{p.name}.npz")
        probe_gt[p.name] = d["xyz"] * np.maximum(d["env_hi"] - d["env_lo"], 1e-9)             + d["env_lo"]
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
            # flow_loss draws t and x0 at random, so an unseeded val score moves
            # by more than the training signal does and picks best.pt by luck
            with torch.no_grad():
                torch.manual_seed(1234)
                row["val"] = float(np.mean(
                    [float(flow_loss(model, to_device(b, args.device), args.anchor, 0.0))
                     for b in vl]))
                torch.manual_seed(epoch)
            ck = {"model": model.state_dict(), "opt": opt.state_dict(),
                  "args": vars(args), "epoch": epoch}
            safe_save(ck, last)
        history.append(row)
        hist_path.write_text(json.dumps(history), encoding="utf-8")
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            model.eval()
            # select on geometry, never on the loss: the flow-matching loss is a
            # conditional variance whose minimum is noise, and picking best.pt by
            # it once froze a run at epoch 30 while the surface kept improving
            # through epoch 60
            row["surf_mm"] = float(np.median([
                chamfer_mm(generate(model, p, args.device, args.steps, 7,
                                    args.cfg_scale, args.points, args.anchor)[0],
                           probe_gt[p.name])
                for p in val_parts[: args.probe_parts]]))
        msg = f"epoch {epoch}: loss {row['train']:.5f} ({row['seconds']}s)"
        if "val" in row:
            msg += f"  val {row['val']:.5f}"
        if "surf_mm" in row:
            msg += f"  | surf {row['surf_mm']:.1f}mm"
            if row["surf_mm"] < best:
                best = row["surf_mm"]
                safe_save({"model": model.state_dict(), "opt": opt.state_dict(),
                           "args": vars(args), "epoch": epoch}, out / "best.pt")
                msg += " *best*"
        print(msg, flush=True)
        if (time.time() - t_start) / 3600.0 > args.max_hours:
            print(f"stopping cleanly at the {args.max_hours}h budget", flush=True)
            break


@torch.no_grad()
def generate(model, part, device, steps, seed, scale, n_pts, anchor=True,
             per_fix=ANCHOR_PER_FIX, known=None):
    gen = torch.Generator(device=device).manual_seed(seed)
    cond = torch.from_numpy(frame_cond_rows(part))[None].to(device)
    fx = None
    rows = []
    if anchor:
        dp, dn = fastener_disc(part, per_fix, rng=np.random.default_rng(seed))
        f, fn = to_frame(dp, dn, fastener_frame(part))
        rows.append(np.concatenate([f, fn], axis=1))
    if known is not None and len(known):
        # any point the caller already knows, in mm with its normal
        kf, kn = to_frame(known[:, :3], known[:, 3:], fastener_frame(part))
        rows.append(np.concatenate([kf, kn], axis=1))
    if rows:
        fx = torch.from_numpy(np.concatenate(rows).astype(np.float32))[None].to(device)
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


def check_no_envelope_leak(parts) -> None:
    """Nothing but the fastening points may reach the model or the decode.

    The envelope is how a FIX position is *stored* (a quantised bin inside it),
    so the test re-quantises into a different box that keeps every fastening
    point at the same millimetre position. Condition, anchors and decode must
    then be bit-for-bit unmoved. This is the check that would have caught the
    ground-truth bounding box serving as the coordinate frame.
    """
    from .plan_g import N_BINS
    worst = 0.0
    for p in parts[:6]:
        before_cond = frame_cond_rows(p)
        before_fix, _ = to_frame(*fix_points_mm(p), fastener_frame(p))
        x = torch.randn(8, CH)
        before_world = to_world(x, p)[0]

        lo0, hi0 = np.array(p.env_lo), np.array(p.env_hi)
        mm = {id(v): lo0 + (np.asarray(v["bin"], float) + 0.5) / N_BINS * (hi0 - lo0)
              for v in p.vertices if v["T"] == "FIX"}
        pad = 0.37 * (hi0 - lo0) + 5.0
        lo1, hi1 = lo0 - pad, hi0 + pad
        old_bins = {id(v): v["bin"] for v in p.vertices if v["T"] == "FIX"}
        p.env_lo, p.env_hi = lo1, hi1
        for v in p.vertices:
            if v["T"] == "FIX":
                b = (mm[id(v)] - lo1) / (hi1 - lo1) * N_BINS - 0.5
                v["bin"] = tuple(int(round(c)) for c in b)
        try:
            assert np.allclose(before_cond, frame_cond_rows(p), atol=1e-3),                 f"{p.name}: condition moved with the envelope"
            after_fix, _ = to_frame(*fix_points_mm(p), fastener_frame(p))
            assert np.allclose(before_fix, after_fix, atol=1e-3),                 f"{p.name}: anchor coords moved with the envelope"
            # re-quantising into a 1.7x looser box moves each fastening point by
            # up to a bin, which propagates through the frame; a real leak moved
            # the decode by ~100mm, so 1mm separates the two cases cleanly
            drift = float(np.abs(before_world - to_world(x, p)[0]).max())
            worst = max(worst, drift)
            assert drift < 1.0,                 f"{p.name}: decode moved {drift:.2f}mm with the envelope"
        finally:
            p.env_lo, p.env_hi = lo0, hi0
            for v in p.vertices:
                if v["T"] == "FIX":
                    v["bin"] = old_bins[id(v)]
    # the disc must come from the fastener alone: scramble every non-FIX vertex
    # and it has to be unchanged
    for p in parts[:3]:
        before = fastener_disc(p, rng=np.random.default_rng(0))[0]
        saved = [(v, v["bin"]) for v in p.vertices if v["T"] != "FIX"]
        for v, _ in saved:
            v["bin"] = tuple(int(b) ^ 0x2AAA for b in v["bin"])
        try:
            after = fastener_disc(p, rng=np.random.default_rng(0))[0]
            assert np.allclose(before, after),                 f"{p.name}: the anchor disc depends on the part, not the fastener"
        finally:
            for v, b in saved:
                v["bin"] = b
    print(f"no-envelope-leak ok ({min(len(parts), 6)} parts, "
          f"worst decode drift {worst:.3f}mm = re-quantisation noise); "
          f"anchor disc independent of part geometry")


def stage_smoke(args, out: pathlib.Path) -> None:
    train_parts, _ = build_splits(args)
    parts = train_parts[:24]
    check_no_envelope_leak(parts)
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

    # masked training must actually vary how much is pinned
    ds_m = MeshDataset(parts, md, args.points, True, mask_rate=1.0)
    counts = set()
    for e in range(6):
        ds_m.set_epoch(e)
        counts.update(int(ds_m[i]["n_fix"]) for i in range(4))
    assert len(counts) > 1, f"masking pinned the same count every time: {counts}"
    print(f"masked training: pinned-point counts seen {sorted(counts)[:6]}...")

    p = parts[0]
    # a caller-supplied constraint point must be honoured exactly
    d = np.load(md / f"{p.name}.npz")
    span = np.maximum(d["env_hi"] - d["env_lo"], 1e-9)
    surf = d["xyz"] * span + d["env_lo"]
    pick = surf[[7, 900, 3000]]
    known = np.concatenate([pick, d["normal"][[7, 900, 3000]]], axis=1)
    kx, _ = generate(model, p, args.device, 8, 1, args.cfg_scale, args.points,
                     args.anchor, known=known)
    err = float(max(np.min(np.linalg.norm(kx - q, axis=1)) for q in pick))
    print(f"caller constraint points honoured to {err:.4f}mm")
    assert err < 1e-2, f"known points were not pinned ({err:.3f}mm)"

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
    ap.add_argument("--mask-rate", type=float, default=0.5,
                    help="fraction of training items that also pin a random set "
                         "of known surface points (masked / inpainting training)")
    ap.add_argument("--cfg-drop", type=float, default=0.1)
    # swept at epoch 65: wire 14.3 / 13.2 / 12.2 / 13.8 / 15.0mm at scale
    # 1.0 / 1.5 / 2.0 / 3.0 / 4.0, and span ratio 0.92 -> 1.06 across it.
    # Without guidance the model hedges toward the mean and the part comes
    # out too small, which is exactly where the outline metric was losing.
    ap.add_argument("--cfg-scale", type=float, default=2.0)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--sample-every", type=int, default=20)
    ap.add_argument("--probe-parts", type=int, default=6)
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

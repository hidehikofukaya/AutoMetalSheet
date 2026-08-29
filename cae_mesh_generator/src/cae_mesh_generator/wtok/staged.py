"""Class-wise targets for the staged generator: outline, then bends, then surface.

One model generating all 512 points has to spend them on three things at once,
and the split it lands on -- roughly 77 outline, 150 bend, 285 surface -- is
where the outline fragments. Measured on the true shapes (val 50, coverage =
mean distance from the dense truth to the nearest of N sampled points):

    N     outline    bend     surface
    77     0.57mm   1.04mm    4.92mm
    150    0.21mm   0.32mm    3.38mm
    300    0.04mm   0.08mm    2.28mm
    600       -     0.00mm    1.47mm

Curves are described to sub-millimetre by 300 points; the surface is still at
1.47mm with 600 and has not converged. A single point budget cannot serve both.

A curve carries a TANGENT, not a normal: the direction that means something at a
point on a curve is the one along it, and it is what a line or arc fit needs
downstream. Only the surface stage carries normals.
"""
from __future__ import annotations

import pathlib

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from .constants import stable_seed
from .meshgen import (CH as MESH_CH, FIELD_CAP, fastener_frame, fastener_disc,
                      fix_points_mm, fps_order, frame_cond_rows, to_frame)
from .ridge import curve_points

# points each stage generates, from the coverage table above
N_OUTLINE = 300
N_BEND = 300
N_SURFACE = 600

STAGES = ("outline", "bend", "surface")
GUARD_PER_FIX = 8
CH = 7          # xyz 3, tangent-or-normal 3, off-the-part flag 1


def tangents(curve: np.ndarray, k: int = 5) -> np.ndarray:
    """Local direction of a curve, from the neighbourhood's principal axis.

    Sign is arbitrary on an unoriented curve, so it is fixed to point along the
    order the points came in -- otherwise neighbouring points get opposite
    tangents and nothing downstream can use them.
    """
    if len(curve) < 3:
        return np.tile([1.0, 0.0, 0.0], (len(curve), 1))
    kk = min(k, len(curve) - 1)
    _, nb = cKDTree(curve).query(curve, k=kk + 1)
    out = np.empty_like(curve)
    for i in range(len(curve)):
        q = curve[nb[i]] - curve[nb[i]].mean(0)
        _, _, vt = np.linalg.svd(q, full_matrices=False)
        out[i] = vt[0]
    ref = np.gradient(curve, axis=0)
    flip = np.einsum("ij,ij->i", out, ref) < 0
    out[flip] *= -1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-12)


# Bounds read off the true outlines, not chosen. Curvature above 0.2 (radius
# 5mm) is 92% single samples -- a corner polygonised, not a tight arc, and a
# corner is poor sheet-metal design anyway. Bounding |k| alone would still admit
# a zigzag of +0.2, -0.2, so the RATE is bounded too: 91% of the true outline
# sits below 0.01 and the 95th percentile is 0.0443, because an outline is lines
# and arcs and its curvature is piecewise constant. Flipping from +0.2 to -0.2
# over 1.8mm needs 0.22, far outside the bound.
K_MAX = 0.2          # 1/mm
DK_MAX = 0.05        # 1/mm^2


def _edge_ends(e):
    r = list(e["refs"])
    if e["tau"] == "LINE":
        return r[0], r[1]
    if e["tau"] == "ARC":
        return r[0], r[2]          # r[1] is the arc's midpoint, not a junction
    return None


def ordered_outline(part, step=1.8):
    """The outline as ONE ordered loop, walked edge to edge.

    curve_points concatenates edge by edge, and consecutive edges in that list
    are not neighbours on the curve -- so "the next point" in that array is
    usually somewhere else entirely. Anything that reasons about a curve needs
    the traversal: curvature, smoothing, and the ordered representation all do.
    Measured on the source: 2300 of 2300 parts have exactly one closed loop.
    """
    from .codec import realize_edge
    from .train_curve import realized_q
    from .ridge import resample_polyline
    q = realized_q(part, part.vertices, part.edges)
    adj = {}
    for e in part.edges:
        if e.get("cls") != "outer_boundary":
            continue
        ends = _edge_ends(e)
        poly = realize_edge(q, e)
        if ends is None or poly is None or len(poly) < 2:
            continue
        adj.setdefault(ends[0], []).append((ends[1], np.asarray(poly, float)))
        adj.setdefault(ends[1], []).append((ends[0], np.asarray(poly, float)[::-1]))
    if not adj:
        return None
    start = min(adj)
    cur, prev, chain = start, None, []
    for _ in range(len(adj) + 1):
        nxt = [(v, pl) for v, pl in adj[cur] if v != prev]
        if not nxt:
            break
        v, pl = nxt[0]
        chain.append(pl[:-1] if chain else pl)      # drop the shared endpoint
        prev, cur = cur, v
        if cur == start:
            break
    if not chain:
        return None
    loop = np.concatenate(chain)
    return resample_polyline(np.concatenate([loop, loop[:1]]), step)


def smooth_curve(pts, k_max=K_MAX, dk_max=DK_MAX, iters=60):
    """Bring a curve inside the curvature bounds, moving points as little as
    possible.

    Applied to the TARGET rather than added as a loss term: two loss terms have
    already been written here that optimised a proxy and left the thing they
    were meant to fix untouched. A target that already satisfies the constraint
    needs no weight to tune, and a model that fits it inherits the property.
    """
    P = np.asarray(pts, float).copy()
    n = len(P)
    if n < 5:
        return P
    closed = np.linalg.norm(P[0] - P[-1]) < 1e-6
    for _ in range(iters):
        Q = np.roll(P, 1, 0), P, np.roll(P, -1, 0)
        a, b, c = Q
        seg = 0.5 * (np.linalg.norm(b - a, axis=1) + np.linalg.norm(c - b, axis=1))
        seg = np.maximum(seg, 1e-9)
        mid = 0.5 * (a + c)
        off = mid - b                                  # towards the local mean
        kap = 2.0 * np.linalg.norm(off, axis=1) / seg ** 2
        w = np.clip(kap / max(k_max, 1e-9) - 1.0, 0.0, None)
        dk = np.abs(kap - np.roll(kap, -1)) / seg
        w = w + np.clip(dk / max(dk_max, 1e-9) - 1.0, 0.0, None)
        if not closed:
            w[0] = w[-1] = 0.0
        step = off * np.clip(w, 0.0, 1.0)[:, None] * 0.3
        if float(np.abs(step).max()) < 1e-9:
            break
        P = P + step
    return P


def stage_target(part, mesh_dir, stage: str, n_pts: int, rng):
    """The true point set for one stage, in the fastener frame.

    Returns (n_pts, CH) or None when the part has no curve of that class.
    """
    frame = fastener_frame(part)
    if stage == "surface":
        d = np.load(pathlib.Path(mesh_dir) / f"{part.name}.npz")
        span = np.maximum(d["env_hi"] - d["env_lo"], 1e-9)
        pts = d["xyz"].astype(np.float64) * span + d["env_lo"]
        dirs = d["normal"].astype(np.float64)
    else:
        if stage == "outline":
            pts = ordered_outline(part)             # walked into one loop
            if pts is None or len(pts) < 16:
                return None
            pts = smooth_curve(pts)
            dirs = tangents(pts)
        else:
            strands, used = bend_strands(part)
            if strands is None:
                return None
            d = np.zeros_like(strands)
            for i in range(len(strands)):
                d[i] = tangents(strands[i]) if used[i] else 0.0
            p_, v_ = to_frame(strands.reshape(-1, 3), d.reshape(-1, 3), frame)
            flag = np.repeat(~used, strands.shape[1]).astype(float)[:, None]
            return np.concatenate([p_, v_, flag], axis=1)
    if stage == "outline":
        # keep the traversal order: evenly along the loop, so slot i and slot
        # i+1 are neighbours on the curve. Farthest-point order would scramble
        # exactly the thing the ordered representation is for.
        idx = np.linspace(0, len(pts) - 1, n_pts)
        take = np.round(idx).astype(int)
    else:
        take = fps_order(pts.astype(np.float32), min(n_pts, len(pts)))[:n_pts]
        if len(take) < n_pts:
            take = np.resize(take, n_pts)
    p, v = to_frame(pts[take], dirs[take], frame)
    if stage == "outline":
        p, v = canonicalise_loop(p, v)
    flag = np.zeros((n_pts, 1))
    return np.concatenate([p, v, flag], axis=1)


def canonicalise_loop(p, v):
    """Fix where the loop starts and which way it runs.

    A closed loop has no natural first point, so the traversal picks one -- and
    it picked the lowest vertex index, which lands somewhere different on every
    part. With ordering now carrying meaning, slot 0 has to mean the same thing
    everywhere or the model is fitting an arbitrary rotation per part on top of
    the shape. Start at the point furthest along the frame's first axis, and run
    so the loop turns positively about the third.
    """
    area = np.cross(p - p.mean(0), np.roll(p, -1, 0) - p.mean(0)).sum(0)
    if area[2] < 0:
        p, v = p[::-1].copy(), -v[::-1]
    s = int(np.argmax(p[:, 0]))
    return np.roll(p, -s, 0), np.roll(v, -s, 0)


class StageDataset(Dataset):
    """Targets for one stage, with the earlier stages' true output as context.

    `context` is what the previous stages produced -- at training time the true
    version, which the trainer may swap for a real generated one. Keeping the
    two interchangeable is the whole point: a stage trained only on perfect
    context learns to trust it completely, and the stage before it is not
    perfect.
    """

    def __init__(self, parts, mesh_dir, stage: str, n_pts: int = 0,
                 augment: bool = False, base_seed: int = 0,
                 n_context: int = 300, outlier_rate: float = 0.0):
        self.parts = [p for p in parts]
        self.dir = pathlib.Path(mesh_dir)
        self.stage = stage
        self.n_pts = n_pts or {"outline": N_OUTLINE,
                               "bend": BEND_STRANDS * BEND_PER_STRAND,
                               "surface": N_SURFACE}[stage]
        self.augment, self.base_seed, self.epoch = augment, base_seed, 0
        self.n_context = n_context
        self.outlier_rate = outlier_rate
        self.cache: dict = {}
        self.self_bank: dict = {}       # stage name -> {part: (n, CH)}

    def set_epoch(self, e):
        self.epoch = int(e)

    def __len__(self):
        return len(self.parts)

    def _target(self, part, stage, n, rng):
        key = (part.name, stage, n)
        if key not in self.cache:
            self.cache[key] = stage_target(part, self.dir, stage, n, rng)
        return self.cache[key]

    def __getitem__(self, i):
        p = self.parts[i]
        rng = np.random.default_rng(
            stable_seed(self.base_seed + 999983 * self.epoch, p.name))
        x = self._target(p, self.stage, self.n_pts, rng)
        if x is None:                                  # no curve of this class
            x = np.zeros((self.n_pts, CH))
            x[:, 6] = 1.0                              # all flagged off-part
        x = x.copy()

        if self.outlier_rate > 0 and rng.random() < self.outlier_rate:
            k = max(1, int(0.10 * len(x)))
            sel = rng.choice(len(x), k, replace=False)
            pitch = float(np.median(cKDTree(x[:, :3]).query(x[:, :3], k=2)[0][:, 1]))
            v = rng.normal(size=(k, 3))
            v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
            x[sel, :3] += v * (pitch * rng.uniform(1.0, 4.0, size=(k, 1)))
            x[sel, 6] = 1.0

        ctx, ctx_n, warp = self._context(p, rng)
        if warp is not None:
            x = x.copy()
            x[:, :3] = warp(x[:, :3])
        # eight seats per fastener, not one. A bolt needs a flat seat normal to
        # its axis and the data agrees exactly out to 10mm (phi 20): over 240
        # sites the real surface deviates from the ideal disc by 0.00mm at the
        # 95th percentile with 0.0 deg of tilt. They are synthesised from the
        # point and its axis, never read from the part, so they leak nothing.
        fx, fn = fastener_disc(p, GUARD_PER_FIX, rng=rng)
        f, fd = to_frame(fx, fn, fastener_frame(p))
        return {
            "x": torch.from_numpy(x.astype(np.float32)),
            "ctx": torch.from_numpy(ctx.astype(np.float32)),
            "n_ctx": ctx_n,
            "cond": torch.from_numpy(frame_cond_rows(p)),
            "fix": torch.from_numpy(np.concatenate([f, fd], 1).astype(np.float32)),
        }

    def _context(self, part, rng):
        """What the earlier stages produced, and the warp that goes with it.

        When the context is a GENERATED outline, the true bend lines belong to a
        different outline and asking for them is the same contradiction that made
        self-conditioning worthless. The two outlines are ordered rings with the
        same canonical start, so the map between them can be fitted directly --
        a cubic explains 98% of the difference - and applying it to the target
        puts both on the same part.
        """
        earlier = STAGES[: STAGES.index(self.stage)]
        if not earlier:
            return np.zeros((1, CH)), 0, None
        rows, warp = [], None
        for s in earlier:
            bank = self.self_bank.get(s, {}).get(part.name)
            true = self._target(part, s, self.n_context, rng)
            if bank is not None:
                rows.append(bank)
                if s == "outline" and true is not None and len(bank) == len(true):
                    warp = fit_warp(true[:, :3], bank[:, :3])
            else:
                rows.append(true if true is not None else np.zeros((1, CH)))
        return np.concatenate(rows), sum(len(r) for r in rows), warp


def coverage_check(parts, mesh_dir, n=25):
    """Does the class-wise target reproduce the coverage the design assumes?"""
    from .ridge import curve_points as cp
    rng = np.random.default_rng(0)
    md = pathlib.Path(mesh_dir)
    got = {s: [] for s in STAGES}
    for p in parts[:n]:
        for s, npts in (("outline", N_OUTLINE), ("bend", N_BEND),
                        ("surface", N_SURFACE)):
            t = stage_target(p, md, s, npts, rng)
            if t is None:
                continue
            frame = fastener_frame(p)
            world = t[:, :3] @ frame[1] * frame[2] + frame[0]
            if s == "surface":
                d = np.load(md / f"{p.name}.npz")
                span = np.maximum(d["env_hi"] - d["env_lo"], 1e-9)
                dense = d["xyz"].astype(np.float64) * span + d["env_lo"]
            else:
                dense = cp(p, "outer_boundary" if s == "outline" else "bend_line")
            if len(dense) < 16:
                continue
            got[s].append(float(cKDTree(world).query(dense)[0].mean()))
    return {s: float(np.median(v)) if v else float("nan") for s, v in got.items()}


# ------------------------------------------------------------------ model

import torch.nn as nn
import torch.nn.functional as F

from .plan_g import COND_ROWS, timestep_embedding

SCALES = (8, 32, 128)      # neighbour counts the encoder reads the context at


class ContextEncoder(nn.Module):
    """The cloud built so far, read at three neighbourhood sizes.

    Each scale is the same patch normalisation the local kernel uses -- centre,
    divide by that neighbourhood's own mean distance, and carry the neighbours'
    directions -- so the features do not change when the point density does.
    That matters here more than anywhere: every stage hands the next one a cloud
    at a different density.
    """

    def __init__(self, dim=192):
        super().__init__()
        per = dim // len(SCALES)
        self.legs = nn.ModuleList(
            nn.Sequential(nn.Linear(6, per), nn.GELU(), nn.Linear(per, per))
            for _ in SCALES)
        self.out = nn.Linear(per * len(SCALES) + 7, dim)

    def forward(self, ctx):                       # (B, M, CH)
        p, d = ctx[..., :3], ctx[..., 3:6]
        M = p.shape[1]
        dist = torch.cdist(p, p) + torch.eye(M, device=p.device)[None] * 1e9
        feats = []
        for leg, k in zip(self.legs, SCALES):
            kk = max(1, min(k, M - 1))
            nd, ni = dist.topk(kk, dim=-1, largest=False)
            off = torch.gather(p[:, None].expand(-1, M, -1, -1), 2,
                               ni[..., None].expand(-1, -1, -1, 3)) - p[:, :, None]
            scale = nd.mean(-1, keepdim=True).clamp(min=1e-6)[..., None]
            nb_d = torch.gather(d[:, None].expand(-1, M, -1, -1), 2,
                                ni[..., None].expand(-1, -1, -1, 3))
            feats.append(leg(torch.cat([off / scale, nb_d], -1)).amax(dim=2))
        return self.out(torch.cat(feats + [ctx], -1))


class Block(nn.Module):
    """Self-attention over the points being generated, then cross-attention to
    everything known, both gated by the timestep the way AdaLN-Zero does.

    Cross-attention is NOT optional. It is the only path the fastening points
    have into the model, and an earlier version made it conditional on the
    stage having a previous stage to read -- which left the outline stage with
    no conditioning at all beyond the timestep. It generated the dataset's
    average outline for every part, which showed up as a band around roughly
    the right shape instead of a curve. What is optional is the ENCODER, which
    adds the earlier stages' points to what is attended to.
    """

    def __init__(self, dim, heads):
        super().__init__()
        self.n1, self.n2, self.n3 = (nn.LayerNorm(dim, elementwise_affine=False)
                                     for _ in range(3))
        self.self_at = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_at = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                 nn.Linear(dim * 4, dim))
        self.ada = nn.Linear(dim, dim * 6)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, h, c, mem):
        g = self.ada(c)[:, None].chunk(6, dim=-1)
        x = self.n1(h) * (1 + g[0])
        h = h + g[1] * self.self_at(x, x, x, need_weights=False)[0]
        x = self.n2(h) * (1 + g[2])
        h = h + g[3] * self.cross_at(x, mem, mem, need_weights=False)[0]
        return h + g[5] * self.mlp(self.n3(h) * (1 + g[4]))


class StageFlow(nn.Module):
    """One stage: flow matching over its own class of points.

    `cross=False` for the outline. The outline is a design decision with no
    curvature signature -- there is nothing local to read it from, and there is
    no earlier stage to attend to either. Bends and surface are local once the
    boundary is known, and they do have something to attend to.
    """

    def __init__(self, dim=256, layers=8, heads=8, cross=True, ordered=""):
        super().__init__()
        self.cross = cross
        self.ordered = ordered or ""
        self.inp = nn.Linear(CH, dim)
        self.cond = nn.Linear(8, dim)
        self.fixp = nn.Linear(6, dim)
        self.enc = ContextEncoder(dim) if cross else None
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(),
                                   nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, CH)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.dim = dim

    @staticmethod
    def strand_pe(n, dim, device, per=None):
        """Which strand a slot belongs to, and where it sits along that strand.

        Not cyclic: a bend line is an OPEN segment, so its first and last points
        are the two ends and must not encode as neighbours. Half the channels
        carry the position along the strand, half carry the strand's index --
        the strands are sorted longest-first, so that index is canonical and
        the model can learn "slots past here are usually unused" directly.

        Without this the 640 slots are an unordered set and the layout the
        target actually has is invisible, which is the same defect that stopped
        the outline stage from learning.
        """
        per = per or BEND_PER_STRAND
        half = dim // 2
        i = torch.arange(n, device=device)
        within = (i % per).float()[:, None]                # 0..per-1 along a strand
        which = torch.div(i, per, rounding_mode="floor").float()[:, None]

        def enc(pos, width):
            # geometric frequency ladder, as in the usual transformer encoding.
            # A LINEAR ladder was tried first and failed its own check: the top
            # harmonics alias and every slot ends up equidistant from every
            # other, which encodes nothing.
            k = torch.arange(width // 2, device=device).float()
            f = torch.exp(-np.log(10000.0) * 2 * k / width)[None]
            return torch.cat([torch.sin(pos * f), torch.cos(pos * f)], -1)[:, :width]

        return torch.cat([enc(within, half), enc(which, dim - half)], -1)

    @staticmethod
    def ring_pe(n, dim, device):
        """Where a slot sits on the loop, as sin/cos of its angle around it.

        Cyclic rather than linear: slot 0 and slot n-1 are neighbours on a closed
        curve, and a linear encoding would put them at opposite ends.
        """
        i = torch.arange(n, device=device)[:, None] / n * 2 * np.pi
        f = torch.arange(1, dim // 2 + 1, device=device)[None]
        return torch.cat([torch.sin(i * f), torch.cos(i * f)], -1)[:, :dim]

    def forward(self, x, t, cond, fix, ctx=None, drop=None):
        h = self.inp(x)
        if self.ordered == "loop":
            h = h + self.ring_pe(x.shape[1], self.dim, x.device)[None]
        elif self.ordered == "strand":
            h = h + self.strand_pe(x.shape[1], self.dim, x.device)[None]
        c = self.t_mlp(timestep_embedding(t, self.dim))
        # the fastening points and the frame rows are known for every stage
        mem = torch.cat([self.cond(cond), self.fixp(fix)], dim=1)
        if self.cross:
            e = self.enc(ctx)
            if drop is not None:                 # CFG drops the earlier stages
                e = torch.where(drop[:, None, None], torch.zeros_like(e), e)
            mem = torch.cat([mem, e], dim=1)
        elif drop is not None:
            # The first stage has no earlier stage to drop, so guidance has to
            # act on the condition itself -- otherwise it has no unconditional
            # branch at all and never learns one. Without it the stage hedges
            # across every outline the two fastening points allow and returns
            # their average: a band around the right shape rather than a curve.
            mem = torch.where(drop[:, None, None], torch.zeros_like(mem), mem)
        for b in self.blocks:
            h = b(h, c, mem)
        return self.head(self.norm(h))


W_CH = None      # built lazily: position 1.0, direction 0.2, flag 0.5


def flow_loss(model, batch, p_drop: float = 0.0, n_pin: int = 0):
    global W_CH
    x1 = batch["x"]
    B, N, _ = x1.shape
    if W_CH is None or W_CH.device != x1.device:
        W_CH = torch.tensor([1., 1., 1., .2, .2, .2, .5], device=x1.device)
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=x1.device)
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    if n_pin:
        xt = torch.cat([x1[:, :n_pin], xt[:, n_pin:]], dim=1)
    drop = (torch.rand(B, device=x1.device) < p_drop) if p_drop > 0 else None
    v = model(xt, t, batch["cond"], batch["fix"], batch.get("ctx"), drop)
    err = ((v - (x1 - x0)) ** 2 * W_CH).mean(-1)
    if n_pin:
        err = err[:, n_pin:]
    return err.mean()


@torch.no_grad()
def sample(model, cond, fix, ctx, n_pts, steps=24, scale=2.0, gen=None,
           pin=None):
    B = cond.shape[0]
    x = torch.randn(B, n_pts, CH, device=cond.device, generator=gen)
    if pin is not None:
        x[:, :pin.shape[1]] = pin
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((B,), i * dt, device=cond.device)
        if scale > 1.0:
            keep = torch.zeros(B, dtype=torch.bool, device=cond.device)
            v = model(x, t, cond, fix, ctx, ~keep)
            v = v + scale * (model(x, t, cond, fix, ctx, keep) - v)
        else:
            v = model(x, t, cond, fix, ctx)
        x = x + v * dt
        if pin is not None:
            x[:, :pin.shape[1]] = pin
    return x


# ------------------------------------------------------------------ training

import argparse
import json
import time

from torch.utils.data import DataLoader

from .constants import safe_save
from .dataset_curve import load_curve_parts
from .validity import outline_closed


@torch.no_grad()
def probe(model, ds, device, n_parts=12, steps=24, scale=2.0):
    """The gate conditions, not the loss.

    A flow-matching loss bottoms out at the conditional variance, so its minimum
    is noise -- checkpoints picked on it have been frozen at the wrong epoch here
    before. What this stage is for is a curve that closes and is evenly spaced,
    so that is what gets measured. Distance to the true outline is reported but
    is NOT a gate: parts with the same fastening conditions differ by 14.9mm, so
    it cannot go much below that however good the model is.
    """
    model.eval()
    ends, cvs = [], []
    for i in range(min(n_parts, len(ds))):
        it = ds[i]
        cond = it["cond"][None].to(device)
        fix = it["fix"][None].to(device)
        ctx = it["ctx"][None].to(device) if model.cross else None
        x = sample(model, cond, fix, ctx, ds.n_pts, steps, scale)[0]
        p = x[:, :3].cpu().numpy().astype(np.float64)
        if ds.stage == "outline":
            ends.append(outline_closed(p)[0])
            d = cKDTree(p).query(p, k=2)[0][:, 1]
        else:
            # bends are open strands, so closure means nothing. What matters is
            # that the model uses the slots it should: how far the flag is from
            # the truth, and how evenly spaced each live strand is.
            want = it["x"][:, 6].numpy() > 0.5
            got = x[:, 6].cpu().numpy() > 0.5
            ends.append(float(np.mean(want != got)))
            q = p.reshape(BEND_STRANDS, BEND_PER_STRAND, 3)[~want.reshape(
                BEND_STRANDS, BEND_PER_STRAND)[:, 0]]
            if len(q) == 0:
                cvs.append(1.0)
                continue
            s = np.linalg.norm(np.diff(q, axis=1), axis=2)
            cvs.append(float(np.median(s.std(1) / np.maximum(s.mean(1), 1e-9))))
            continue
        cvs.append(float(d.std() / max(d.mean(), 1e-9)))
    model.train()
    return float(np.median(ends)), float(np.median(cvs))


def train(args):
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    md = pathlib.Path(args.dataset) / "parts"
    parts = load_curve_parts(pathlib.Path(args.wtok))
    have = {f.stem for f in md.glob("*.npz")}
    parts = [p for p in parts if p.name in have]
    vn = set(json.loads(pathlib.Path(args.val_list).read_text(encoding="utf-8")))
    train_parts = [p for p in parts if p.name not in vn][: args.train_parts]
    val_parts = [p for p in parts if p.name in vn][: args.val_parts]
    print(f"stage {args.stage}: train {len(train_parts)} val {len(val_parts)}",
          flush=True)

    cross = args.stage != "outline"
    ordered = {"outline": "loop", "bend": "strand"}.get(args.stage, "")
    ds = StageDataset(train_parts, md, args.stage, base_seed=7,
                      outlier_rate=args.outlier_rate)
    vs = StageDataset(val_parts, md, args.stage, base_seed=555)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    model = StageFlow(args.dim, args.layers, args.heads, cross, ordered).to(args.device)
    print(f"params: {sum(q.numel() for q in model.parameters())/1e6:.2f}M",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs,
                                                       args.lr * 0.05)
    start, best, hist = 1, float("inf"), []
    last = out / "last.pt"
    if args.resume and last.exists():
        ck = torch.load(last, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["epoch"] + 1
        best = ck.get("best", best)
        for _ in range(start - 1):
            sched.step()
        print(f"resumed at {start}", flush=True)

    t_start = time.time()
    for epoch in range(start, args.epochs + 1):
        t0 = time.time()
        ds.set_epoch(epoch)
        tot, n = 0.0, 0
        for b in dl:
            b = {k: (v.to(args.device) if torch.is_tensor(v) else v)
                 for k, v in b.items()}
            loss = flow_loss(model, b, args.cfg_drop)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            n += 1
        sched.step()
        msg = f"epoch {epoch}: loss {tot/max(n,1):.5f} ({time.time()-t0:.1f}s)"
        row = {"epoch": epoch, "loss": tot / max(n, 1)}
        if epoch % args.probe_every == 0:
            e, cv = probe(model, vs, args.device, args.probe_parts, args.steps,
                          args.cfg_scale)
            row.update(ends=e, spacing_cv=cv)
            score = e + cv                      # both are "lower is better"
            msg += f"  | ends {100*e:.1f}% spacing_cv {cv:.3f}"
            if score < best:
                best = score
                msg += " *best*"
                safe_save({"model": model.state_dict(), "opt": opt.state_dict(),
                           "epoch": epoch, "best": best, "args": vars(args)},
                          out / "best.pt")
            safe_save({"model": model.state_dict(), "opt": opt.state_dict(),
                       "epoch": epoch, "best": best, "args": vars(args)}, last)
        hist.append(row)
        (out / "history.json").write_text(json.dumps(hist), encoding="utf-8")
        print(msg, flush=True)
        if (time.time() - t_start) / 3600 > args.max_hours:
            print("stopping cleanly at the time budget", flush=True)
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--wtok", required=True)
    ap.add_argument("--val-list", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--train-parts", type=int, default=500)
    ap.add_argument("--val-parts", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--cfg-drop", type=float, default=0.1)
    ap.add_argument("--cfg-scale", type=float, default=1.0,
                    help="guidance at sampling. 1.0 = none, which is what the "
                         "ordered representation wants: the old point-set model "
                         "needed 2.0 because hedging shrank it (span 0.92), but "
                         "an ordered ring stays a ring when it hedges, so "
                         "guidance only inflates it. Measured over 30 val parts, "
                         "perimeter / non-planarity / span against the truth: "
                         "1.0 -> 0.993 / 0.978 / 1.003, 2.0 -> 1.070 / 1.104 / "
                         "1.014, and the distance from the true curve is lowest "
                         "at 1.0 as well (0.0394 against 0.0519).")
    ap.add_argument("--outlier-rate", type=float, default=0.35)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--probe-every", type=int, default=5)
    ap.add_argument("--probe-parts", type=int, default=12)
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train(ap.parse_args())



# ------------------------------------------------------- output conditioning

LOWPASS_KEEP = 25       # frequency bands kept on a closed outline


def clean_loop(P, keep: int = LOWPASS_KEEP, n: int = 0):
    """Take the jitter out of a generated loop without moving the curve.

    A closed ordered loop has a Fourier series, and the true outlines put 84.6%
    of their energy in the first four bands and 2.7% above band 25 -- an outline
    made of lines and arcs has nowhere else to put it. The generated loops carry
    17.9% up there, and that is the visible wobble.

    Dropping those bands and then re-spacing the points along the arc length
    measured, over 30 val parts:

        raw                 step cv 0.609   jumps 2.0%   hf 0.185   curv99 418
        resample only               0.175         0.0%       0.113        279
        lowpass + resample          0.001         0.0%       0.009         24
        (true outline)              0.322         0.0%       0.037          -

    while the distance from the true curve moves 0.0491 -> 0.0507, i.e. the
    shape is kept. That remaining distance is the task's information limit and
    nothing here can reach it.
    """
    P = np.asarray(P, float)
    F = np.fft.rfft(P, axis=0)
    F[keep:] = 0
    P = np.fft.irfft(F, n=len(P), axis=0)
    n = n or len(P)
    Q = np.concatenate([P, P[:1]])
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return P
    want = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.stack([np.interp(want, s, Q[:, d]) for d in range(3)], axis=1)


def clean_demo():
    """A wobbly ring must come back round, and a real corner must survive."""
    t = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    ring = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], 1)
    rng = np.random.default_rng(0)
    noisy = ring + rng.normal(scale=0.02, size=ring.shape)
    out = clean_loop(noisy)
    err = float(np.abs(np.linalg.norm(out[:, :2], axis=1) - 1).mean())
    raw = float(np.abs(np.linalg.norm(noisy[:, :2], axis=1) - 1).mean())
    d = np.linalg.norm(np.roll(out, -1, 0) - out, axis=1)
    print(f"noisy ring: radius error {raw:.4f} -> {err:.4f}, "
          f"step cv {float(d.std()/d.mean()):.4f}")
    assert err < raw / 2 and float(d.std() / d.mean()) < 0.05
    # a square: the corners must not be rounded away entirely
    m = 75
    s = np.linspace(-1, 1, m, endpoint=False)
    sq = np.concatenate([np.stack([s, -np.ones(m), np.zeros(m)], 1),
                         np.stack([np.ones(m), s, np.zeros(m)], 1),
                         np.stack([-s, np.ones(m), np.zeros(m)], 1),
                         np.stack([-np.ones(m), -s, np.zeros(m)], 1)])
    o = clean_loop(sq)
    keep = float(np.abs(o[:, :2]).max(1).max())
    print(f"square: extent {float(np.abs(sq[:, :2]).max()):.3f} -> {keep:.3f} "
          f"(corners softened, not lost)")
    assert keep > 0.9
    print("ok")


# ------------------------------------------------------------ stage 2: bends

BEND_MIN_MM = 10.0      # segments shorter than this are tessellation debris
BEND_STRANDS = 32       # fixed slots; unused ones are flagged off-part
BEND_PER_STRAND = 20


def bend_strands(part, min_mm=BEND_MIN_MM, n_strand=BEND_STRANDS,
                 per=BEND_PER_STRAND):
    """The bend lines as a fixed set of ordered strands.

    A part carries 68 bend_line edges at a median length of 6mm, but they are
    not 68 folds: the class marks where a bend's cylindrical face meets a flat
    panel, so one fold contributes two of them, and most of the rest are slivers
    (61% are under 10mm and hold 11.9% of the total length). Keeping only
    segments of 10mm or more leaves 24 per part and loses 0.42mm of coverage.

    Fixed slots rather than a variable list: a variable count brings back the
    combinatorial structure that killed the earlier autoregressive route --
    counts, ordering, references, a stop token. Here an unused strand is simply
    flagged, and there is nothing to get structurally wrong. Points are ordered
    WITHIN a strand, for the same reason the outline is ordered; strands among
    themselves are exchangeable, which is fine because they are separate curves.
    """
    from .codec import realize_edge
    from .train_curve import realized_q
    from .ridge import resample_polyline
    q = realized_q(part, part.vertices, part.edges)
    keep = []
    for e in part.edges:
        if e.get("cls") != "bend_line":
            continue
        poly = realize_edge(q, e)
        if poly is None or len(poly) < 2:
            continue
        poly = np.asarray(poly, float)
        if float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum()) < min_mm:
            continue
        keep.append(resample_polyline(poly, 1e9)[:0] if False else poly)
    if not keep:
        return None, None
    keep.sort(key=lambda a: -float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum()))
    keep = keep[:n_strand]
    pts = np.zeros((n_strand, per, 3))
    used = np.zeros(n_strand, bool)
    for i, poly in enumerate(keep):
        L = float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())
        pts[i] = resample_polyline(poly, max(L / (per - 1), 1e-6))[:per]
        if len(pts[i]) < per:                       # resample can come up short
            pts[i] = np.resize(pts[i], (per, 3))
        used[i] = True
    return pts, used


def fit_warp(src, dst, deg: int = 3):
    """The smooth map that takes the true outline onto the generated one.

    Stage 2 is handed a GENERATED outline but would otherwise be trained against
    the TRUE bend lines, which belong to a different outline -- the same
    contradiction that made self-conditioning useless. The difference between the
    two outlines is not noise: a cubic explains 98% of it (an affine map already
    explains 88%). Applying that map to the true bend lines gives a target that
    belongs to the outline the stage actually received.

    Both outlines are ordered rings with a canonical start, so the point
    correspondence needed to fit it is free.
    """
    def basis(P):
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        c = [np.ones_like(x)]
        if deg >= 1:
            c += [x, y, z]
        if deg >= 2:
            c += [x * x, y * y, z * z, x * y, y * z, z * x]
        if deg >= 3:
            c += [x ** 3, y ** 3, z ** 3, x * x * y, x * x * z, y * y * x,
                  y * y * z, z * z * x, z * z * y, x * y * z]
        return np.stack(c, 1)
    coef, *_ = np.linalg.lstsq(basis(src), dst, rcond=None)
    return lambda P: basis(np.asarray(P, float)) @ coef


def warp_demo():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(300, 3))
    true = lambda P: P * [1.05, 0.95, 1.0] + 0.02 * P[:, [1, 2, 0]] ** 2 + 0.01
    w = fit_warp(src, true(src))
    q = rng.normal(size=(50, 3))
    err = float(np.abs(w(q) - true(q)).max())
    print(f"cubic warp recovers a quadratic map to {err:.2e}")
    assert err < 1e-6
    print("ok")

def pe_demo():
    """The strand encoding must expose the layout the target actually has."""
    pe = StageFlow.strand_pe(BEND_STRANDS * BEND_PER_STRAND, 256, "cpu").numpy()
    h = 128
    along, across = pe[1] - pe[0], pe[BEND_PER_STRAND] - pe[0]
    print(f"step along a strand -> position ch {np.abs(along[:h]).max():.3f}"
          f", strand ch {np.abs(along[h:]).max():.3f}")
    print(f"step across strands -> position ch {np.abs(across[:h]).max():.3f}"
          f", strand ch {np.abs(across[h:]).max():.3f}")
    assert np.abs(along[h:]).max() < 1e-6 and np.abs(across[:h]).max() < 1e-6
    assert np.linalg.norm(pe[1] - pe[0]) < np.linalg.norm(
        pe[BEND_PER_STRAND - 1] - pe[0]), "strand ends read as neighbours"
    assert len(np.unique(pe.round(4), axis=0)) == len(pe), "slots not distinct"
    print("ok")


if __name__ == "__main__":
    main()

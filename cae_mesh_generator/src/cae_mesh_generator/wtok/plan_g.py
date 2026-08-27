"""Set-based wireframe generation: B-core (frontier baseline) and G-AGF (ours).

Two stages, both permutation-equivariant over slots (no positional encoding on
the vertex/edge blocks; edge->vertex references are pointer attention, so a
reference names the vertex by its embedding, never by an index):

  stage "topo": masked discrete flow matching over the skeleton
                (vertex types, edge types, edge classes, edge->vertex refs,
                 and -- only when --coarse-in-t -- coarse coordinate digits).
  stage "geo" : continuous flow matching over vertex coordinates, conditioned
                on the skeleton (absolute coords for B-core; the fine residual
                inside the coarse cell for G-AGF).
  stage "eval": generate topo -> geo -> Q -> realize, with the session's
                standard error decomposition plus FIX pass-through error,
                topology validity and best-of-N headroom.

Arms (docs/frontier_gaps_and_reform.md):
  B-core : --arm bcore   (I1 shared refs + I2 discrete FM + I3 two stages)
  G-AGF  : --arm gagf    (+ R1 anchor freezing, R2 coarse-in-T)
R3 (learned ranker) is deferred; eval reports the best-of-N oracle headroom it
would compete for.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import math
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .codec import realize_points
from .constants import BITS, HI_BITS, safe_save, stable_seed
from .dataset_ar import cond_features
from .dataset_curve import load_curve_parts, transform_vertices
from .train_ar import chamfer_mm
from .train_curve import realized_q

K_V, K_E = 160, 128            # data maxima are 148 / 124
COND_ROWS = 4
N_COARSE = 1 << HI_BITS                 # 128
N_FINE = 1 << (BITS - HI_BITS)          # 128
N_BINS = 1 << BITS
LO_MASK = N_FINE - 1

VT = ("PAD", "FIX", "END", "MID")
ET = ("PAD", "LINE", "ARC", "CIRCLE", "CIRCLE_C")
CLS = ("outer_boundary", "bend_line")
ARITY = {"LINE": 2, "ARC": 3, "CIRCLE": 3, "CIRCLE_C": 2}
VT_ID = {t: i for i, t in enumerate(VT)}
ET_ID = {t: i for i, t in enumerate(ET)}

MASK_VT, MASK_ET, MASK_CLS, MASK_C = len(VT), len(ET), len(CLS), N_COARSE
NONE_REF, MASK_REF = K_V, K_V + 1       # extra pointer targets


# ---------------------------------------------------------------- data

def build_item(part, vertices, rng, shuffle: bool) -> dict:
    """One part -> fixed-size slot tensors. FIX vertices take the first slots
    (the anchor block); every other slot order is randomized when shuffling, so
    the model can never lean on a canonical order."""
    fix_ids = [i for i, v in enumerate(vertices) if v["T"] == "FIX"]
    other = [i for i in range(len(vertices)) if v_is_free(vertices, i)]
    if shuffle:
        other = list(rng.permutation(other))
    order = fix_ids + other
    if len(order) > K_V:
        order = order[:K_V]
    slot_of = {vi: s for s, vi in enumerate(order)}

    vtype = np.zeros(K_V, dtype=np.int64)             # PAD
    vcoarse = np.zeros((K_V, 3), dtype=np.int64)
    vfine = np.zeros((K_V, 3), dtype=np.float32)
    vabs = np.zeros((K_V, 3), dtype=np.float32)
    for s, vi in enumerate(order):
        v = vertices[vi]
        b = np.asarray(v["bin"], dtype=np.int64)
        vtype[s] = VT_ID[v["T"]]
        vcoarse[s] = b >> HI_BITS
        vfine[s] = ((b & LO_MASK) + 0.5) / N_FINE
        vabs[s] = (b + 0.5) / N_BINS

    eidx = [i for i, e in enumerate(part.edges)
            if all(r in slot_of for r in e["refs"])]
    if shuffle:
        eidx = list(rng.permutation(eidx))
    eidx = eidx[:K_E]
    etype = np.zeros(K_E, dtype=np.int64)
    ecls = np.zeros(K_E, dtype=np.int64)
    erefs = np.full((K_E, 3), NONE_REF, dtype=np.int64)
    for s, ei in enumerate(eidx):
        e = part.edges[ei]
        etype[s] = ET_ID[e["tau"]]
        ecls[s] = CLS.index(e["cls"]) if e["cls"] in CLS else 0
        for k, r in enumerate(e["refs"][:3]):
            erefs[s, k] = slot_of[r]

    cond = np.zeros((COND_ROWS, 8), dtype=np.float32)
    rows = cond_features([vertices[i] for i in fix_ids], part.env_lo, part.env_hi)
    cond[: min(COND_ROWS, len(rows))] = rows[:COND_ROWS]

    return {"vtype": torch.from_numpy(vtype), "vcoarse": torch.from_numpy(vcoarse),
            "vfine": torch.from_numpy(vfine), "vabs": torch.from_numpy(vabs),
            "etype": torch.from_numpy(etype), "ecls": torch.from_numpy(ecls),
            "erefs": torch.from_numpy(erefs), "cond": torch.from_numpy(cond),
            "n_fix": len(fix_ids)}


def v_is_free(vertices, i: int) -> bool:
    return vertices[i]["T"] != "FIX"


class SlotDataset(Dataset):
    def __init__(self, parts, augment: bool, base_seed: int = 0, jitter_bins: int = 1):
        self.parts = parts
        self.augment = augment
        self.base_seed = base_seed
        self.jitter_bins = jitter_bins if augment else 0
        self.epoch = 0

    def set_epoch(self, e: int) -> None:
        self.epoch = int(e)

    def __len__(self) -> int:
        return len(self.parts)

    def __getitem__(self, i: int) -> dict:
        p = self.parts[i]
        rng = np.random.default_rng(
            stable_seed(self.base_seed + 999983 * self.epoch, p.name))
        axis = rng.choice([None, 0, 1, 2]) if self.augment else None
        vs = transform_vertices(p.vertices, axis, self.jitter_bins, rng)
        return build_item(p, vs, rng, shuffle=self.augment)


# ---------------------------------------------------------------- backbone

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * freqs[None] * 1000.0
    return torch.cat([torch.cos(a), torch.sin(a)], dim=1)


class Block(nn.Module):
    """Pre-norm transformer block with AdaLN-Zero conditioning (CLR-Wire style:
    the film layers start at zero, so the block begins as plain pre-norm)."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                nn.Linear(dim * 4, dim))
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.film = nn.Linear(dim, dim * 4)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, c, bias=None):
        s1, b1, s2, b2 = self.film(c)[:, None].chunk(4, dim=-1)
        h = self.n1(x) * (1 + s1) + b1
        x = x + self.attn(h, h, h, attn_mask=bias, need_weights=False)[0]
        h = self.n2(x) * (1 + s2) + b2
        return x + self.ff(h)


class RelBias(nn.Module):
    """Per-head attention bias from the pairwise distance between vertex slots.

    The image-domain lever: a convolution shares weights across positions, so one
    picture teaches a motif everywhere. Absolute coordinate embeddings share
    nothing, which is why 'which vertex connects to which' had to be memorised
    per location. Making attention a function of the relative geometry restores
    that sharing -- and connect-the-dots is exactly a relative-geometry problem.
    """

    def __init__(self, heads: int, n_rbf: int = 16):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, 1.5, n_rbf))
        self.width = 1.5 / n_rbf
        self.proj = nn.Linear(n_rbf, heads)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        d = torch.cdist(xyz, xyz)                              # (B,K_V,K_V)
        rbf = torch.exp(-((d[..., None] - self.centers) / self.width) ** 2)
        return self.proj(rbf).permute(0, 3, 1, 2)              # (B,H,K_V,K_V)


class SlotNet(nn.Module):
    """Shared trunk for every stage: [cond | K_V vertex slots | K_E edge slots].

    Slots carry no positional encoding, so the network is permutation
    equivariant within each block; an edge names its vertices by adding that
    vertex's own embedding (pointer semantics at the input side) and predicts
    references by dot product against vertex states (pointer at the output).

    Modes:
      topo / geo      : the original two-stage split (kept so the Kaggle
                        checkpoints still load).
      points / edges  : the reordered split. `points` places the vertices
                        (existence + coordinates) from the condition alone;
                        `edges` then wires them up *while seeing where they are*.
                        The first run's topology stage was geometry-blind, which
                        is why a third of its references were invalid.
    """

    def __init__(self, dim=256, layers=8, heads=8, coarse_in: bool = True,
                 geo: bool = False, mode: str | None = None,
                 rel_attn: bool = False):
        super().__init__()
        mode = mode or ("geo" if geo else "topo")
        self.mode, self.heads = mode, heads
        self.pred_vtype = mode in ("topo", "points")
        self.pred_edges = mode in ("topo", "edges")
        self.pred_xyz = mode in ("geo", "points")
        self.xyz_in = mode in ("geo", "points", "edges")
        geo = self.pred_xyz                       # head layout follows this
        self.dim, self.coarse_in, self.geo = dim, coarse_in, geo
        self.rel = RelBias(heads) if rel_attn else None
        self.cond_proj = nn.Linear(8, dim)
        self.null_cond = nn.Parameter(torch.zeros(COND_ROWS, dim))
        self.seg = nn.Parameter(torch.zeros(3, dim))
        self.vt_emb = nn.Embedding(len(VT) + 1, dim)          # +MASK
        self.et_emb = nn.Embedding(len(ET) + 1, dim)
        self.cls_emb = nn.Embedding(len(CLS) + 1, dim)
        self.coarse_emb = nn.ModuleList(
            nn.Embedding(N_COARSE + 1, dim) for _ in range(3))
        self.ref_mask_emb = nn.Parameter(torch.zeros(3, dim))
        self.ref_none_emb = nn.Parameter(torch.zeros(3, dim))
        self.ref_proj = nn.ModuleList(nn.Linear(dim, dim) for _ in range(3))
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        if self.xyz_in:
            self.xt_proj = nn.Linear(3, dim)
        if self.pred_xyz:
            self.vel_head = nn.Linear(dim, 3)
            nn.init.zeros_(self.vel_head.weight)
            nn.init.zeros_(self.vel_head.bias)
        if self.pred_vtype:
            self.vt_head = nn.Linear(dim, len(VT))
            self.coarse_head = nn.ModuleList(
                nn.Linear(dim, N_COARSE) for _ in range(3))
        if self.pred_edges:
            self.et_head = nn.Linear(dim, len(ET))
            self.cls_head = nn.Linear(dim, len(CLS))
            self.q_proj = nn.ModuleList(nn.Linear(dim, dim) for _ in range(3))
            self.k_proj = nn.Linear(dim, dim)
            self.none_key = nn.Parameter(torch.randn(dim) * 0.02)

    def embed(self, s: dict, xt: torch.Tensor | None = None):
        v = self.vt_emb(s["vtype"])
        if self.coarse_in:
            for ax in range(3):
                v = v + self.coarse_emb[ax](s["vcoarse"][..., ax])
        if xt is not None:
            v = v + self.xt_proj(xt)
        v = v + self.seg[1]
        e = self.et_emb(s["etype"]) + self.cls_emb(s["ecls"]) + self.seg[2]
        for k in range(3):
            r = s["erefs"][..., k]
            gathered = torch.gather(
                v, 1, r.clamp(max=K_V - 1)[..., None].expand(-1, -1, self.dim))
            gathered = torch.where((r == NONE_REF)[..., None],
                                   self.ref_none_emb[k], gathered)
            gathered = torch.where((r == MASK_REF)[..., None],
                                   self.ref_mask_emb[k], gathered)
            e = e + self.ref_proj[k](gathered)
        return v, e

    def rel_mask(self, xt, B, S, device, dtype):
        """Distance bias over the vertex block, zero elsewhere, as the float
        attn_mask nn.MultiheadAttention takes (B*heads, S, S)."""
        if self.rel is None or xt is None:
            return None
        m = torch.zeros(B, self.heads, S, S, device=device, dtype=dtype)
        lo = COND_ROWS
        m[:, :, lo:lo + K_V, lo:lo + K_V] = self.rel(xt).to(dtype)
        return m.reshape(B * self.heads, S, S)

    def forward(self, s: dict, t: torch.Tensor, xt: torch.Tensor | None = None,
                drop: torch.Tensor | None = None):
        cond = self.cond_proj(s["cond"])
        if drop is not None:                     # classifier-free guidance
            cond = torch.where(drop[:, None, None], self.null_cond, cond)
        cond = cond + self.seg[0]
        v, e = self.embed(s, xt)
        # the points stage has nothing to say about edges: drop that block and
        # its 44% of the sequence rather than feed it MASK for compute's sake
        x = torch.cat([cond, v], dim=1) if self.mode == "points" \
            else torch.cat([cond, v, e], dim=1)
        c = self.t_mlp(timestep_embedding(t, self.dim)) + cond.mean(1)
        bias = self.rel_mask(xt, x.shape[0], x.shape[1], x.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, c, bias)
        x = self.norm(x)
        hv = x[:, COND_ROWS: COND_ROWS + K_V]
        he = x[:, COND_ROWS + K_V:] if self.mode != "points" else None
        if self.mode == "geo":
            return self.vel_head(hv)
        out = {}
        if self.pred_xyz:
            out["vel"] = self.vel_head(hv)
        if self.pred_vtype:
            out["vtype"] = self.vt_head(hv)
            out["coarse"] = [h(hv) for h in self.coarse_head]
        if self.pred_edges:
            keys = torch.cat([self.k_proj(hv),
                              self.none_key.expand(hv.shape[0], 1, self.dim)], dim=1)
            out["refs"] = [torch.einsum("bed,bvd->bev", self.q_proj[k](he), keys)
                           / (self.dim ** 0.5) for k in range(3)]
            out["etype"] = self.et_head(he)
            out["ecls"] = self.cls_head(he)
        return out


# ---------------------------------------------------------------- discrete FM

DISC_KEYS = ("vtype", "etype", "ecls", "erefs")
MASK_OF = {"vtype": MASK_VT, "etype": MASK_ET, "ecls": MASK_CLS, "erefs": MASK_REF}


def corrupt(batch: dict, t: torch.Tensor, coarse_in: bool, anchor: bool,
            gen: torch.Generator | None = None):
    """Masking path of discrete flow matching: each token is independently
    replaced by MASK with probability 1-t. Anchors (the FIX block) stay clean
    when the arm freezes them -- R1."""
    s, masks = {"cond": batch["cond"]}, {}
    B = batch["vtype"].shape[0]
    keep_v = torch.zeros_like(batch["vtype"], dtype=torch.bool)
    if anchor:
        idx = torch.arange(K_V, device=batch["vtype"].device)[None]
        keep_v = idx < batch["n_fix"][:, None]
    for key in DISC_KEYS:
        x = batch[key]
        p = torch.rand(x.shape, device=x.device, generator=gen)
        m = p >= t.view(B, *([1] * (x.dim() - 1)))
        if key == "vtype":
            m = m & ~keep_v
        s[key] = torch.where(m, torch.full_like(x, MASK_OF[key]), x)
        masks[key] = m
    if coarse_in:
        p = torch.rand(batch["vcoarse"].shape, device=t.device, generator=gen)
        m = (p >= t[:, None, None]) & ~keep_v[..., None]
        s["vcoarse"] = torch.where(m, torch.full_like(batch["vcoarse"], MASK_C),
                                   batch["vcoarse"])
        masks["vcoarse"] = m
    else:
        s["vcoarse"] = torch.zeros_like(batch["vcoarse"])
    return s, masks


def masked_ce(logits: torch.Tensor, target: torch.Tensor, m: torch.Tensor):
    if not bool(m.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[m], target[m])


def topo_loss(model: SlotNet, batch: dict, anchor: bool) -> torch.Tensor:
    B = batch["vtype"].shape[0]
    t = torch.rand(B, device=batch["vtype"].device)
    s, m = corrupt(batch, t, model.coarse_in, anchor)
    out = model(s, t)
    loss = (masked_ce(out["vtype"], batch["vtype"], m["vtype"])
            + masked_ce(out["etype"], batch["etype"], m["etype"])
            + masked_ce(out["ecls"], batch["ecls"], m["ecls"]))
    for k in range(3):
        loss = loss + masked_ce(out["refs"][k], batch["erefs"][..., k],
                                m["erefs"][..., k]) / 3.0
    if model.coarse_in:
        for ax in range(3):
            loss = loss + masked_ce(out["coarse"][ax], batch["vcoarse"][..., ax],
                                    m["vcoarse"][..., ax]) / 3.0
    return loss


@torch.no_grad()
def sample_topo(model: SlotNet, cond: torch.Tensor, anchors: dict | None,
                steps: int = 24, gen: torch.Generator | None = None) -> dict:
    """Iteratively unmask at the discrete-FM rate dt/(1-t)."""
    dev = cond.device
    B = cond.shape[0]
    s = {"cond": cond,
         "vtype": torch.full((B, K_V), MASK_VT, dtype=torch.long, device=dev),
         "vcoarse": torch.full((B, K_V, 3), MASK_C, dtype=torch.long, device=dev),
         "etype": torch.full((B, K_E), MASK_ET, dtype=torch.long, device=dev),
         "ecls": torch.full((B, K_E), MASK_CLS, dtype=torch.long, device=dev),
         "erefs": torch.full((B, K_E, 3), MASK_REF, dtype=torch.long, device=dev)}
    if not model.coarse_in:
        s["vcoarse"] = torch.zeros((B, K_V, 3), dtype=torch.long, device=dev)
    if anchors is not None:                      # R1: the answer already holds them
        s["vtype"][:, : anchors["n"]] = VT_ID["FIX"]
        if model.coarse_in:
            s["vcoarse"][:, : anchors["n"]] = anchors["coarse"]

    def draw(logits, temp=1.0):
        p = F.softmax(logits / temp, dim=-1)
        flat = p.reshape(-1, p.shape[-1])
        return torch.multinomial(flat, 1, generator=gen).reshape(p.shape[:-1])

    for k in range(steps):
        t = k / steps
        rate = (1.0 / steps) / max(1.0 - t, 1e-6)
        tt = torch.full((B,), t, device=dev)
        out = model(s, tt)
        cand = {"vtype": draw(out["vtype"]), "etype": draw(out["etype"]),
                "ecls": draw(out["ecls"])}
        for key, logits in (("vtype", None), ("etype", None), ("ecls", None)):
            m = (s[key] == MASK_OF[key]) & (
                torch.rand(s[key].shape, device=dev, generator=gen) < rate)
            s[key] = torch.where(m, cand[key], s[key])
        # references: never point at a slot already decided to be PAD
        dead = (s["vtype"] == VT_ID["PAD"])
        for j in range(3):
            logits = out["refs"][j].clone()
            logits[:, :, :K_V] = logits[:, :, :K_V].masked_fill(
                dead[:, None, :], torch.finfo(logits.dtype).min)
            r = draw(logits)
            m = (s["erefs"][..., j] == MASK_REF) & (
                torch.rand(r.shape, device=dev, generator=gen) < rate)
            s["erefs"][..., j] = torch.where(m, r, s["erefs"][..., j])
        if model.coarse_in:
            for ax in range(3):
                c = draw(out["coarse"][ax])
                m = (s["vcoarse"][..., ax] == MASK_C) & (
                    torch.rand(c.shape, device=dev, generator=gen) < rate)
                s["vcoarse"][..., ax] = torch.where(m, c, s["vcoarse"][..., ax])
    # final pass: fill whatever is still masked with the argmax
    tt = torch.ones(B, device=dev)
    out = model(s, tt)
    for key, logits in (("vtype", out["vtype"]), ("etype", out["etype"]),
                        ("ecls", out["ecls"])):
        s[key] = torch.where(s[key] == MASK_OF[key], logits.argmax(-1), s[key])
    dead = (s["vtype"] == VT_ID["PAD"])
    for j in range(3):
        logits = out["refs"][j].clone()
        logits[:, :, :K_V] = logits[:, :, :K_V].masked_fill(
            dead[:, None, :], torch.finfo(logits.dtype).min)
        s["erefs"][..., j] = torch.where(s["erefs"][..., j] == MASK_REF,
                                         logits.argmax(-1), s["erefs"][..., j])
    if model.coarse_in:
        for ax in range(3):
            s["vcoarse"][..., ax] = torch.where(
                s["vcoarse"][..., ax] == MASK_C, out["coarse"][ax].argmax(-1),
                s["vcoarse"][..., ax])
    return s


# ---------------------------------------------------------------- continuous FM

def geo_target(batch: dict, coarse_in: bool) -> torch.Tensor:
    """[-1,1] target: the fine residual inside the coarse cell (G-AGF, R2) or
    the absolute coordinate (B-core)."""
    x = batch["vfine"] if coarse_in else batch["vabs"]
    return x * 2.0 - 1.0


def geo_loss(model: SlotNet, batch: dict, anchor: bool) -> torch.Tensor:
    B = batch["vtype"].shape[0]
    dev = batch["vtype"].device
    x1 = geo_target(batch, model.coarse_in)
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=dev)
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    live = batch["vtype"] != VT_ID["PAD"]
    if anchor:                                   # R1: anchors are never noised
        idx = torch.arange(K_V, device=dev)[None]
        is_anchor = idx < batch["n_fix"][:, None]
        xt = torch.where(is_anchor[..., None], x1, xt)
        live = live & ~is_anchor
    s = dict(batch)
    s["vcoarse"] = batch["vcoarse"] if model.coarse_in else torch.zeros_like(batch["vcoarse"])
    v = model(s, t, xt)
    return (((v - (x1 - x0)) ** 2).mean(-1) * live).sum() / live.sum().clamp(min=1)


@torch.no_grad()
def sample_geo(model: SlotNet, skel: dict, anchors: dict | None, steps: int = 24,
               gen: torch.Generator | None = None) -> torch.Tensor:
    dev = skel["vtype"].device
    B = skel["vtype"].shape[0]
    x = torch.randn((B, K_V, 3), device=dev, generator=gen)
    if anchors is not None:
        x[:, : anchors["n"]] = anchors["x1"]
    dt = 1.0 / steps
    for k in range(steps):
        t = torch.full((B,), k * dt, device=dev)
        x = x + model(skel, t, x) * dt
        if anchors is not None:
            x[:, : anchors["n"]] = anchors["x1"]
    return x


# ------------------------------------------- points / edges (reordered split)

def dummy_edges(B: int, dev) -> dict:
    return {"etype": torch.full((B, K_E), MASK_ET, dtype=torch.long, device=dev),
            "ecls": torch.full((B, K_E), MASK_CLS, dtype=torch.long, device=dev),
            "erefs": torch.full((B, K_E, 3), MASK_REF, dtype=torch.long, device=dev)}


def anchor_mask(batch: dict) -> torch.Tensor:
    idx = torch.arange(K_V, device=batch["vtype"].device)[None]
    return idx < batch["n_fix"][:, None]


def cfg_drop(B: int, p: float, dev, training: bool):
    if p <= 0 or not training:
        return None
    return torch.rand(B, device=dev) < p


def points_loss(model: SlotNet, batch: dict, anchor: bool, p_drop: float = 0.0):
    """Place the vertices: existence as masked discrete FM, coordinates as
    continuous FM, both from the condition alone."""
    dev = batch["vtype"].device
    B = batch["vtype"].shape[0]
    t = torch.rand(B, device=dev)
    keep = anchor_mask(batch) if anchor else torch.zeros_like(batch["vtype"], dtype=torch.bool)

    m = (torch.rand(batch["vtype"].shape, device=dev) >= t[:, None]) & ~keep
    s = {"cond": batch["cond"],
         "vtype": torch.where(m, torch.full_like(batch["vtype"], MASK_VT),
                              batch["vtype"]),
         "vcoarse": torch.zeros_like(batch["vcoarse"]), **dummy_edges(B, dev)}

    x1 = batch["vabs"] * 2.0 - 1.0
    x0 = torch.randn_like(x1)
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    live = batch["vtype"] != VT_ID["PAD"]
    if anchor:
        xt = torch.where(keep[..., None], x1, xt)
        live = live & ~keep

    out = model(s, t, xt, cfg_drop(B, p_drop, dev, model.training))
    loss = masked_ce(out["vtype"], batch["vtype"], m)
    vel = ((out["vel"] - (x1 - x0)) ** 2).mean(-1)
    return loss + (vel * live).sum() / live.sum().clamp(min=1)


def edges_loss(model: SlotNet, batch: dict, p_drop: float = 0.0):
    """Wire up a KNOWN point set: the stage that was geometry-blind before."""
    dev = batch["vtype"].device
    B = batch["vtype"].shape[0]
    t = torch.rand(B, device=dev)
    s = {"cond": batch["cond"], "vtype": batch["vtype"],
         "vcoarse": torch.zeros_like(batch["vcoarse"])}
    masks = {}
    for key in ("etype", "ecls", "erefs"):
        x = batch[key]
        m = torch.rand(x.shape, device=dev) >= t.view(B, *([1] * (x.dim() - 1)))
        s[key] = torch.where(m, torch.full_like(x, MASK_OF[key]), x)
        masks[key] = m
    xt = batch["vabs"] * 2.0 - 1.0                       # clean coordinates
    out = model(s, t, xt, cfg_drop(B, p_drop, dev, model.training))
    loss = (masked_ce(out["etype"], batch["etype"], masks["etype"])
            + masked_ce(out["ecls"], batch["ecls"], masks["ecls"]))
    for k in range(3):
        loss = loss + masked_ce(out["refs"][k], batch["erefs"][..., k],
                                masks["erefs"][..., k]) / 3.0
    return loss


def guided(model, s, t, xt, scale: float):
    """One forward, or two extrapolated when guidance is on."""
    if scale <= 1.0:
        return model(s, t, xt)
    B = t.shape[0]
    dev = t.device
    keep = torch.zeros(B, dtype=torch.bool, device=dev)
    cond = model(s, t, xt, keep)
    null = model(s, t, xt, ~keep)
    out = {}
    for k, v in cond.items():
        if isinstance(v, list):
            out[k] = [n + scale * (c - n) for c, n in zip(v, null[k])]
        else:
            out[k] = null[k] + scale * (v - null[k])
    return out


def draw_from(logits, gen, temp=1.0):
    p = F.softmax(logits / temp, dim=-1)
    return torch.multinomial(p.reshape(-1, p.shape[-1]), 1,
                             generator=gen).reshape(p.shape[:-1])


@torch.no_grad()
def sample_points(model, cond, anchors, steps=24, gen=None, scale=1.0):
    dev = cond.device
    B = cond.shape[0]
    s = {"cond": cond,
         "vtype": torch.full((B, K_V), MASK_VT, dtype=torch.long, device=dev),
         "vcoarse": torch.zeros((B, K_V, 3), dtype=torch.long, device=dev),
         **dummy_edges(B, dev)}
    x = torch.randn((B, K_V, 3), device=dev, generator=gen)
    if anchors is not None:
        s["vtype"][:, : anchors["n"]] = VT_ID["FIX"]
        x[:, : anchors["n"]] = anchors["x1"]
    dt = 1.0 / steps
    for k in range(steps):
        t = k * dt
        rate = dt / max(1.0 - t, 1e-6)
        tt = torch.full((B,), t, device=dev)
        out = guided(model, s, tt, x, scale)
        um = (s["vtype"] == MASK_VT) & (
            torch.rand(s["vtype"].shape, device=dev, generator=gen) < rate)
        s["vtype"] = torch.where(um, draw_from(out["vtype"], gen), s["vtype"])
        x = x + out["vel"] * dt
        if anchors is not None:
            x[:, : anchors["n"]] = anchors["x1"]
    tt = torch.ones(B, device=dev)
    out = guided(model, s, tt, x, scale)
    s["vtype"] = torch.where(s["vtype"] == MASK_VT, out["vtype"].argmax(-1),
                             s["vtype"])
    if anchors is not None:
        s["vtype"][:, : anchors["n"]] = VT_ID["FIX"]
    return s, x


def force_anchor_incidence(s: dict, logits: list, n_fix: int) -> dict:
    """R1, corrected. Freezing an anchor pins where the vertex IS; it does not
    make any edge USE it, and an unreferenced vertex never appears in the
    realized curve -- measured 0/2 anchors referenced in the first run. So for
    any anchor no surviving edge points at, re-point the single (edge, slot) the
    model itself scored highest. A grammar constraint, not a shape rule.

    Candidates are restricted to edges that will survive decode: re-pointing an
    edge whose other references are dead just moves the anchor onto an edge that
    is about to be dropped.
    """
    if n_fix <= 0:
        return s
    B = s["etype"].shape[0]
    dead = s["vtype"] == VT_ID["PAD"]                        # (B, K_V)
    for bi in range(B):
        arity = [ARITY.get(ET[int(v)], 0) for v in s["etype"][bi]]
        alive = []
        for ei, ar in enumerate(arity):
            if ar == 0:
                continue
            refs = s["erefs"][bi, ei, :ar].tolist()
            if all(r < K_V and not bool(dead[bi, r]) for r in refs):
                alive.append(ei)
        if not alive:
            continue
        claimed: set = set()
        for a in range(n_fix):
            if any(a in s["erefs"][bi, ei, :arity[ei]].tolist() for ei in alive):
                continue
            best = None
            for ei in alive:
                for j in range(arity[ei]):
                    if (ei, j) in claimed:      # never overwrite another anchor
                        continue
                    v = float(logits[j][bi, ei, a])
                    if best is None or v > best[0]:
                        best = (v, ei, j)
            if best is not None:
                s["erefs"][bi, best[1], best[2]] = a
                claimed.add((best[1], best[2]))
    return s


@torch.no_grad()
def sample_edges(model, points, xyz, steps=24, gen=None, scale=1.0, n_fix=0):
    dev = xyz.device
    B = xyz.shape[0]
    s = {"cond": points["cond"], "vtype": points["vtype"],
         "vcoarse": points["vcoarse"], **dummy_edges(B, dev)}
    dead = s["vtype"] == VT_ID["PAD"]
    dt = 1.0 / steps
    for k in range(steps + 1):
        t = min(k * dt, 1.0)
        rate = dt / max(1.0 - t, 1e-6) if k < steps else 1.0
        tt = torch.full((B,), t, device=dev)
        out = guided(model, s, tt, xyz, scale)
        for key in ("etype", "ecls"):
            r = torch.rand(s[key].shape, device=dev, generator=gen) < rate
            pick = (draw_from(out[key], gen) if k < steps
                    else out[key].argmax(-1))
            s[key] = torch.where((s[key] == MASK_OF[key]) & r, pick, s[key])
        for j in range(3):
            lg = out["refs"][j].clone()
            lg[:, :, :K_V] = lg[:, :, :K_V].masked_fill(
                dead[:, None, :], torch.finfo(lg.dtype).min)
            r = torch.rand(s["erefs"][..., j].shape, device=dev, generator=gen) < rate
            pick = draw_from(lg, gen) if k < steps else lg.argmax(-1)
            s["erefs"][..., j] = torch.where(
                (s["erefs"][..., j] == MASK_REF) & r, pick, s["erefs"][..., j])
    return force_anchor_incidence(s, out["refs"], n_fix)


# ---------------------------------------------------------------- decode

def decode(skel: dict, xs: torch.Tensor, part, coarse_in: bool, b: int = 0) -> dict:
    """Slots -> Q. References are shared indices, so endpoints are welded by
    construction: no distance-threshold welding step exists (I1)."""
    vtype = skel["vtype"][b].tolist()
    coarse = skel["vcoarse"][b].tolist()
    x = ((xs[b].clamp(-1, 1) + 1.0) / 2.0).cpu().numpy()
    fix = [v for v in part.vertices if v["T"] == "FIX"]
    vertices, slot_map, n_fix_seen = [], {}, 0
    for s, vt in enumerate(vtype):
        if VT[vt] == "PAD":
            continue
        if coarse_in:
            b_i = [int(np.clip(coarse[s][a] * N_FINE + int(x[s][a] * N_FINE),
                               0, N_BINS - 1)) for a in range(3)]
        else:
            b_i = [int(np.clip(x[s][a] * N_BINS, 0, N_BINS - 1)) for a in range(3)]
        nf = None
        if VT[vt] == "FIX" and n_fix_seen < len(fix):
            nf = fix[n_fix_seen]["nf"]
            n_fix_seen += 1
        slot_map[s] = len(vertices)
        vertices.append({"T": VT[vt], "bin": tuple(b_i), "nf": nf})

    edges, dropped = [], 0
    for s, et in enumerate(skel["etype"][b].tolist()):
        tau = ET[et]
        if tau == "PAD":
            continue
        refs = skel["erefs"][b, s].tolist()[: ARITY[tau]]
        if any(r not in slot_map for r in refs):
            dropped += 1
            continue
        mapped = [slot_map[r] for r in refs]
        if tau == "CIRCLE_C" and vertices[mapped[0]]["nf"] is None:
            dropped += 1                    # a CIRCLE_C centre must carry an axis
            continue
        edges.append({"tau": tau, "refs": mapped,
                      "cls": CLS[skel["ecls"][b, s].item()]})
    Q = realized_q(part, vertices, edges)
    Q["_dropped"] = dropped
    return Q


def relax_q(Q: dict, part, mu: float = 1e3, lam: float = 1e-3) -> dict:
    """Elastic relaxation: keep the generated edge vectors while pulling the FIX
    vertices onto their true positions, and let connectivity carry that pull
    through the wire. Linear least squares, so it is differentiable and can move
    into training later. Measured post-hoc: FIX error 6.11mm -> 0.00mm."""
    V = Q["vertices"]
    if not V or not Q["edges"]:
        return Q
    lo = np.asarray(Q["env_lo"], dtype=np.float64)
    span = np.maximum(np.asarray(Q["env_hi"], dtype=np.float64) - lo, 1e-9)
    x0 = np.stack([lo + (np.asarray(v["bin"], np.float64) + 0.5) / N_BINS * span
                   for v in V])
    gtq = realized_q(part, part.vertices, part.edges)
    from .codec import bin_center
    targets = [bin_center(gtq, v) for v in part.vertices if v["T"] == "FIX"]
    anchors = [i for i, v in enumerate(V) if v["T"] == "FIX"][: len(targets)]

    A, b = [], []
    for e in Q["edges"]:
        refs = e["refs"][: ARITY[e["tau"]]]
        for i, j in zip(refs[:-1], refs[1:]):
            row = np.zeros(len(V)); row[i], row[j] = 1.0, -1.0
            A.append(row); b.append(x0[i] - x0[j])
    for k, i in enumerate(anchors):
        row = np.zeros(len(V)); row[i] = np.sqrt(mu)
        A.append(row); b.append(np.sqrt(mu) * targets[k])
    for i in range(len(V)):
        row = np.zeros(len(V)); row[i] = np.sqrt(lam)
        A.append(row); b.append(np.sqrt(lam) * x0[i])

    x = np.linalg.lstsq(np.stack(A), np.stack(b), rcond=None)[0]
    out = [{"T": v["T"], "nf": v["nf"],
            "bin": tuple(int(t) for t in
                         np.clip((p - lo) / span * N_BINS, 0, N_BINS - 1))}
           for v, p in zip(V, x)]
    return dict(Q, vertices=out)


@torch.no_grad()
def generate_pe(points_net, edges_net, part, device, anchor: bool, steps: int,
                seed: int = 0, scale: float = 1.0, do_relax: bool = True) -> dict:
    """points -> edges -> decode -> elastic relaxation."""
    gen = torch.Generator(device=device).manual_seed(seed)
    item = build_item(part, part.vertices, np.random.default_rng(0), shuffle=False)
    cond = item["cond"][None].to(device)
    anc = anchors_of(part, device, False) if anchor else None
    pts, xyz = sample_points(points_net, cond, anc, steps, gen, scale)
    skel = sample_edges(edges_net, pts, xyz, steps, gen, scale,
                        n_fix=anc["n"] if anc else 0)
    Q = decode(skel, xyz, part, False)
    return relax_q(Q, part) if do_relax else Q


def anchors_of(part, device, coarse_in: bool) -> dict:
    fix = [v for v in part.vertices if v["T"] == "FIX"]
    b = np.asarray([v["bin"] for v in fix], dtype=np.int64)
    x = ((b & LO_MASK) + 0.5) / N_FINE if coarse_in else (b + 0.5) / N_BINS
    return {"n": len(fix),
            "coarse": torch.tensor(b >> HI_BITS, device=device)[None],
            "x1": torch.tensor(x * 2.0 - 1.0, dtype=torch.float32, device=device)[None]}


# ---------------------------------------------------------------- generation

@torch.no_grad()
def generate(topo: SlotNet, geo: SlotNet, part, device: str, anchor: bool,
             steps: int, seed: int = 0) -> dict:
    gen = torch.Generator(device=device).manual_seed(seed)
    item = build_item(part, part.vertices, np.random.default_rng(0), shuffle=False)
    cond = item["cond"][None].to(device)
    anc = anchors_of(part, device, topo.coarse_in) if anchor else None
    skel = sample_topo(topo, cond, anc, steps=steps, gen=gen)
    skel_geo = dict(skel)
    if not geo.coarse_in:
        skel_geo["vcoarse"] = torch.zeros_like(skel["vcoarse"])
    xs = sample_geo(geo, skel_geo, anc if anchor else None, steps=steps, gen=gen)
    return decode(skel, xs, part, geo.coarse_in)


def fix_error_mm(Q: dict, part) -> float:
    """Distance from every FIX point to the nearest generated curve point: the
    condition-satisfaction metric the frontier never reports (G1)."""
    pts = realize_points(Q)
    if len(pts) == 0:
        return float("nan")
    gtq = realized_q(part, part.vertices, part.edges)
    from .codec import bin_center
    fix = [bin_center(gtq, v) for v in part.vertices if v["T"] == "FIX"]
    return float(np.mean([np.min(np.linalg.norm(pts - f, axis=1)) for f in fix]))


# ---------------------------------------------------------------- stages

def build_splits(args):
    parts = load_curve_parts(pathlib.Path(args.dataset))
    val = set(json.loads(pathlib.Path(args.val_list).read_text(encoding="utf-8")))
    return ([p for p in parts if p.name not in val],
            [p for p in parts if p.name in val])


def to_device(b: dict, dev: str) -> dict:
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


def make_net(args, geo: bool = False, mode: str | None = None) -> SlotNet:
    return SlotNet(args.dim, args.layers, args.heads,
                   coarse_in=(args.arm == "gagf"), geo=geo, mode=mode,
                   rel_attn=(mode == "edges" and getattr(args, "rel_attn", 1)))


def stage_loss_fn(mode: str, args):
    anchor = args.arm != "bcore"
    p = getattr(args, "cfg_drop", 0.0)
    return {
        "topo": lambda m, b: topo_loss(m, b, anchor),
        "geo": lambda m, b: geo_loss(m, b, anchor),
        "points": lambda m, b: points_loss(m, b, anchor, p),
        "edges": lambda m, b: edges_loss(m, b, p),
    }[mode]


def run_stage(args, out: pathlib.Path, mode: str) -> None:
    train_parts, val_parts = build_splits(args)
    name = mode
    geo = mode == "geo"
    print(f"[{name}] arm={args.arm} parts: train {len(train_parts)} val {len(val_parts)}")
    tl = DataLoader(SlotDataset(train_parts, augment=True), batch_size=args.batch_size,
                    shuffle=True, drop_last=True)
    vl = DataLoader(SlotDataset(val_parts, augment=False, base_seed=555),
                    batch_size=args.batch_size)
    model = make_net(args, geo, mode).to(args.device)
    loss_fn = stage_loss_fn(mode, args)
    print(f"[{name}] params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, args.lr * 0.05)
    history, best, t_start = [], float("inf"), time.time()
    hist_path = out / f"{name}_history.json"
    start = 1
    ck_path = out / f"{name}_last.pt"
    if args.resume and ck_path.exists():
        # local training runs in short slices, so resume is the normal path
        ck = torch.load(ck_path, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start = ck["epoch"] + 1
        if hist_path.exists():
            history = [r for r in json.loads(hist_path.read_text()) if r["epoch"] < start]
            best = min((r["val"] for r in history if "val" in r), default=float("inf"))
        for _ in range(start - 1):
            sched.step()
        print(f"[{name}] resumed at epoch {start} (best val {best:.4f})", flush=True)
    for epoch in range(start, args.epochs + 1):
        tl.dataset.set_epoch(epoch)
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for b in tl:
            loss = loss_fn(model, to_device(b, args.device))
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
                vs = [float(loss_fn(model, to_device(b, args.device)))
                      for b in vl]
            row["val"] = float(np.mean(vs))
            ck = {"model": model.state_dict(), "opt": opt.state_dict(),
                  "args": vars(args), "epoch": epoch}
            if row["val"] < best:
                best = row["val"]
                safe_save(ck, out / f"{name}_best.pt")
            safe_save(ck, out / f"{name}_last.pt")
        history.append(row)
        hist_path.write_text(json.dumps(history), encoding="utf-8")
        msg = f"[{name}] epoch {epoch}: loss {row['train']:.4f} ({row['seconds']}s)"
        if "val" in row:
            msg += f"  val {row['val']:.4f}"
        print(msg, flush=True)
        if (time.time() - t_start) / 3600.0 > args.max_hours:
            print(f"[{name}] stopping cleanly at the {args.max_hours}h budget",
                  flush=True)
            break


def stage_eval(args, out: pathlib.Path) -> None:
    from .evaluate_curve2 import class_points, one_way
    _, val_parts = build_splits(args)
    modes = ("points", "edges") if (out / "points_best.pt").exists() else ("topo", "geo")
    nets = {}
    for mode in modes:
        ck = torch.load(out / f"{mode}_best.pt", map_location=args.device,
                        weights_only=False)
        a = argparse.Namespace(**ck["args"])
        net = make_net(a, mode == "geo", mode).to(args.device)
        net.load_state_dict(ck["model"])
        net.eval()
        nets[mode] = net
        args.arm = a.arm            # the checkpoints decide the arm, not the flag
    anchor = args.arm != "bcore"
    new_pipeline = modes[0] == "points"
    print(f"[eval] pipeline={'points/edges' if new_pipeline else 'topo/geo'} "
          f"arm={args.arm} guidance={args.cfg_scale} relax={not args.no_relax}")
    rows = []
    for p in val_parts[: args.eval_parts]:
        gt = realize_points(realized_q(p, p.vertices, p.edges))
        cands = []
        for s in range(args.best_of):
            Q = (generate_pe(nets["points"], nets["edges"], p, args.device,
                             anchor, args.steps, seed=1000 * s + 7,
                             scale=args.cfg_scale, do_relax=not args.no_relax)
                 if new_pipeline else
                 generate(nets["topo"], nets["geo"], p, args.device, anchor,
                          args.steps, seed=1000 * s + 7))
            gen = realize_points(Q)
            cd = chamfer_mm(gen[::3], gt[::3]) if len(gen) else float("nan")
            cands.append((cd, Q, gen))
        first = cands[0]
        best = min((c for c in cands if np.isfinite(c[0])), key=lambda c: c[0],
                   default=first)
        cd, Q, gen = first
        row = {"name": p.name, "chamfer_mm": cd, "best_of_n_mm": best[0],
               "n_edges": len(Q["edges"]), "gt_edges": len(p.edges),
               "dropped_edges": Q["_dropped"],
               "fix_err_mm": fix_error_mm(Q, p)}
        if len(gen):
            row["outline_mm"] = one_way(class_points(p, {"outer_boundary"}), gen)[0]
            row["bend_mm"] = one_way(class_points(p, {"bend_line"}), gen)[0]
            row["infill_mm"] = one_way(gen, gt)[0]
        rows.append(row)
        print(f"  {p.name}: {cd:.1f}mm (best-of-{args.best_of} {best[0]:.1f}) "
              f"E={len(Q['edges'])}/{len(p.edges)} fix={row['fix_err_mm']:.2f}mm",
              flush=True)

    def med(key):
        v = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        return float(np.median(v)) if v else float("nan")

    total_e = sum(r["n_edges"] + r["dropped_edges"] for r in rows)
    summary = {"arm": args.arm, "parts": len(rows), "steps": args.steps,
               "chamfer_median_mm": med("chamfer_mm"),
               "best_of_n_median_mm": med("best_of_n_mm"), "best_of": args.best_of,
               "outline_median_mm": med("outline_mm"), "bend_median_mm": med("bend_mm"),
               "infill_median_mm": med("infill_mm"),
               "fix_err_median_mm": med("fix_err_mm"),
               "edge_ratio": float(np.mean([r["n_edges"] / max(r["gt_edges"], 1)
                                            for r in rows])),
               "topo_valid_rate": 1.0 - sum(r["dropped_edges"] for r in rows) / max(total_e, 1)}
    print(json.dumps(summary, indent=1))
    (out / "eval.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")


def stage_smoke(args, out: pathlib.Path) -> None:
    """Self-check: the slot representation must be lossless, every stage must
    run and learn, generation must decode, and the elastic relaxation must put
    the wire exactly on the fastening points."""
    parts = load_curve_parts(pathlib.Path(args.dataset))[:24]

    def signature(vertices, edges):
        vs = sorted((v["T"], tuple(v["bin"])) for v in vertices)
        es = sorted((e["tau"], e["cls"],
                     tuple(tuple(vertices[r]["bin"]) for r in e["refs"]))
                    for e in edges)
        return vs, es

    for arm in ("bcore", "agf2"):
        a = argparse.Namespace(**{**vars(args), "arm": arm})
        coarse = arm == "gagf"
        for p in parts[:8]:
            item = build_item(p, p.vertices, np.random.default_rng(0), shuffle=True)
            skel = {k: v[None] for k, v in item.items() if torch.is_tensor(v)}
            Q = decode(skel, geo_target(skel, coarse), p, coarse)
            assert Q["_dropped"] == 0, f"{p.name}: dropped {Q['_dropped']} edges"
            assert signature(Q["vertices"], Q["edges"]) ==                 signature(p.vertices, p.edges), f"{p.name}: roundtrip changed the wire"
        print(f"[{arm}] roundtrip ok (exact on 8 parts)")

        ds = SlotDataset(parts, augment=True)
        loader = DataLoader(ds, batch_size=4, shuffle=True, drop_last=True)
        nets = {}
        for mode in ("points", "edges"):
            net = make_net(a, False, mode).to(args.device)
            nets[mode] = net
            fn = stage_loss_fn(mode, a)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
            losses = []
            for epoch in range(8):
                ds.set_epoch(epoch)
                for b in loader:
                    loss = fn(net, to_device(b, args.device))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    losses.append(float(loss))
            head, tail = np.mean(losses[:5]), np.mean(losses[-5:])
            n_par = sum(q.numel() for q in net.parameters()) / 1e6
            assert tail < head, f"{arm}/{mode} did not learn"
            print(f"[{arm}] {mode}: {n_par:.2f}M params, {head:.3f} -> {tail:.3f}")

        for relax in (False, True):
            Q = generate_pe(nets["points"], nets["edges"], parts[0], args.device,
                            anchor=(arm != "bcore"), steps=6, seed=1,
                            scale=args.cfg_scale, do_relax=relax)
            err = fix_error_mm(Q, parts[0])
            fix_idx = {i for i, v in enumerate(Q["vertices"]) if v["T"] == "FIX"}
            used = {r for e in Q["edges"] for r in e["refs"][: ARITY[e["tau"]]]}
            inc = len(fix_idx & used)
            print(f"[{arm}] relax={relax}: V={len(Q['vertices'])} E={len(Q['edges'])} "
                  f"dropped={Q['_dropped']} anchors_referenced={inc}/{len(fix_idx)} "
                  f"fix_err={err:.2f}mm")
            if relax and arm != "bcore" and inc:
                assert err < 1.0, f"{arm}: relaxation left {err:.2f}mm at the anchors"
    print("smoke ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("points", "edges", "topo", "geo", "eval", "smoke"))
    ap.add_argument("--arm", choices=("bcore", "gagf", "agf2"), default="agf2")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--val-list", default="")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=224)   # 2 stages ~= 12M total,
    ap.add_argument("--layers", type=int, default=7)  # matching C (10.5M)/AB (13.3M)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--eval-parts", type=int, default=40)
    ap.add_argument("--best-of", type=int, default=5)
    ap.add_argument("--resume", action="store_true",
                    help="continue from <stage>_last.pt (local runs go in slices)")
    ap.add_argument("--cfg-drop", type=float, default=0.1,
                    help="condition dropout during training (classifier-free guidance)")
    ap.add_argument("--cfg-scale", type=float, default=1.5,
                    help="guidance scale at sampling; 1.0 disables it")
    ap.add_argument("--rel-attn", type=int, default=1,
                    help="relative-geometry attention bias in the edges stage")
    ap.add_argument("--no-relax", action="store_true",
                    help="skip the elastic relaxation at decode")
    ap.add_argument("--max-hours", type=float, default=8.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    if args.stage == "smoke":
        stage_smoke(args, out)
    elif args.stage == "eval":
        stage_eval(args, out)
    else:
        run_stage(args, out, args.stage)


if __name__ == "__main__":
    main()

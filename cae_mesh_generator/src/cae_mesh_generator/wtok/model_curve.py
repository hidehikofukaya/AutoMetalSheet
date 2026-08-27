"""Curve-major AR model: decoder-only transformer with a joint softmax over
[static vocabulary ; dynamic pointer slots] (PolyGen-style pointers).

Constructive masks at sampling (revision §3): slot-type consistency, FIX-only
CIRCLE_C centers, intra-edge non-degeneracy, pointer availability. Duplicate
edges / coincident NEW vertices are normalized at decode (delta_curve).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import HI_BITS, E_ARITY
from .codec_curve import SLOT_TYPES
from .dataset_curve import (ADV, BOS, COORD0, NEW, PAD, PTR, STOP, TAU_BASE,
                            TAU_TOK, TAUS, VOCAB_C, VTYPE_ID, digits_of)
from .model_ar import CausalBlock


class CurveAR(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 8, layers: int = 8,
                 cond_dim: int = 8, max_len: int = 15000, dropout: float = 0.0):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_C, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.cond_in = nn.Linear(cond_dim, dim)
        self.cond_seg = nn.Parameter(torch.zeros(dim))
        self.vtype_emb = nn.Embedding(3, dim)
        self.digit_emb = nn.Embedding(1 << HI_BITS, dim)
        self.digit_axis = nn.Parameter(torch.randn(6, dim) * 0.02)
        self.vertex_mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(CausalBlock(dim, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, VOCAB_C)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.dim = dim

    def vertex_embed(self, mat_type, mat_digits):
        """(M,) type ids + (M,6) digits -> (M,dim) pointer key/value features."""
        if len(mat_type) == 0:
            return torch.zeros(0, self.dim, device=mat_type.device)
        d = (self.digit_emb(mat_digits) + self.digit_axis[None]).sum(dim=1)
        return self.vertex_mlp(self.vtype_emb(mat_type) + d)

    def forward(self, item) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch size 1. Returns (static_logits (S,VOCAB_C), ptr_logits (S,M))."""
        in_tok, in_ptr = item["in_tok"], item["in_ptr"]
        vemb = self.vertex_embed(item["mat_type"], item["mat_digits"])
        x = self.tok(in_tok)
        has_ptr = in_ptr >= 0
        if has_ptr.any():
            x = x.clone()
            x[has_ptr] = x[has_ptr] + vemb[in_ptr[has_ptr]]
        x = x + self.pos(torch.arange(len(in_tok), device=in_tok.device))
        cond = self.cond_in(item["cond"]) + self.cond_seg
        h = torch.cat([cond, x], dim=0)[None]
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h[0, len(cond):])
        static = self.head(h)
        q = self.q_proj(h)
        k = self.k_proj(vemb)
        ptr = (q @ k.T) / (self.dim ** 0.5) if len(k) else torch.zeros(len(h), 0,
                                                                      device=h.device)
        return static, ptr

    def loss(self, item) -> torch.Tensor:
        static, ptr = self.forward(item)
        S = static.shape[0]
        pos_idx = torch.arange(S, device=static.device)
        # pointer legality: materialized before this prediction + type match
        if ptr.shape[1]:
            avail = item["mat_pos"][None, :] <= pos_idx[:, None]
            st = item["slot_next"]  # 0 none / 1 END / 2 MID / 3 FIX
            mtype = item["mat_type"]  # 0 FIX / 1 END / 2 MID
            want = torch.full_like(st, -1)
            want[st == 1] = 1
            want[st == 2] = 2
            want[st == 3] = 0
            tmatch = (mtype[None, :] == want[:, None]) & (want[:, None] >= 0)
            ptr = ptr.masked_fill(~(avail & tmatch), -1e9)
        logits = torch.cat([static, ptr], dim=1)
        ce = F.cross_entropy(logits, item["target"], reduction="none")
        m = item["loss_mask"].float()
        return (ce * m).sum() / m.sum().clamp(min=1)


class CurveSampler:
    """Ancestral sampling with the constructive slot state machine."""

    def __init__(self, model: CurveAR, device: str):
        self.model = model
        self.device = device

    @torch.no_grad()
    def run(self, cond: torch.Tensor, fix_vertices: list[dict],
            observed_edges: list[dict] | None = None, max_edges: int = 800,
            temperature: float = 1.0, seed: int = 0,
            coord_temperature: float | None = None):
        # coord_temperature: separate (usually low) temperature for coordinate
        # digits -- teacher-forced coarse-digit error is 1-2 bins, so T=1
        # sampling over 128 bins injects spatial noise that compounds over the
        # rollout; structure tokens keep `temperature`.
        """observed_edges: sigma-style records to clamp (ADV consumption).
        Returns (vertices(list of {T,bin,nf}), edges(list of {tau,refs}))."""
        gen = torch.Generator(self.device).manual_seed(seed)
        obs = observed_edges or []  # geometry records: {tau, verts: [(T, bin)]}
        obs_i = 0
        dev = self.device
        vertices = [dict(v) for v in fix_vertices]
        mat_type = [VTYPE_ID[v["T"]] for v in vertices]
        mat_digits = [digits_of(v["bin"]) for v in vertices]
        bin_index = {(v["T"], tuple(v["bin"])): i for i, v in enumerate(vertices)}
        in_tok, in_ptr = [BOS], [-1]
        edges_out = []

        def item():
            return {
                "in_tok": torch.tensor(in_tok, dtype=torch.long, device=dev),
                "in_ptr": torch.tensor(in_ptr, dtype=torch.long, device=dev),
                "mat_type": torch.tensor(mat_type, dtype=torch.long, device=dev),
                "mat_digits": (torch.tensor(mat_digits, dtype=torch.long, device=dev)
                               if mat_digits else torch.zeros(0, 6, dtype=torch.long,
                                                              device=dev)),
                "cond": cond,
            }

        def next_logits():
            static, ptr = self.model.forward(item())
            return static[-1] / max(temperature, 1e-6), ptr[-1] / max(temperature, 1e-6)

        def sample_from(static_l, ptr_l, static_ok: torch.Tensor,
                        ptr_ok: torch.Tensor | None):
            sl = static_l.masked_fill(~static_ok, -1e9)
            if ptr_ok is not None and len(ptr_l):
                pl = ptr_l.masked_fill(~ptr_ok, -1e9)
                logits = torch.cat([sl, pl])
            else:
                logits = torch.cat([sl, torch.full_like(ptr_l, -1e9)]) if len(ptr_l) else sl
            probs = torch.softmax(logits, dim=-1)
            return int(torch.multinomial(probs, 1, generator=gen))

        def push(tok: int, ptr_id: int = -1):
            in_tok.append(tok)
            in_ptr.append(ptr_id)

        while len(edges_out) < max_edges:
            # ---- edge boundary decision ----
            static_ok = torch.zeros(VOCAB_C, dtype=torch.bool, device=dev)
            if obs_i < len(obs):
                static_ok[ADV] = True
            for t in TAUS:
                static_ok[TAU_TOK[t]] = True
            if obs_i >= len(obs):
                static_ok[STOP] = True
            # CIRCLE_C needs at least one FIX vertex
            if not any(t == 0 for t in mat_type):
                static_ok[TAU_TOK["CIRCLE_C"]] = False
            sl, pl = next_logits()
            choice = sample_from(sl, pl, static_ok, None)
            if choice == STOP:
                break
            if choice == ADV:
                push(ADV)
                e = obs[obs_i]
                obs_i += 1
                push(TAU_TOK[e["tau"]])
                refs = []
                # observed slots arrive as geometry; resolve to the sampler's
                # own namespace (pointer if already materialized, NEW otherwise)
                for (vt, vb) in e["verts"]:
                    key = (vt, tuple(vb))
                    if key in bin_index:
                        vid = bin_index[key]
                        push(PTR, vid)
                        refs.append(vid)
                    else:
                        push(NEW)
                        digits = digits_of(tuple(vb))
                        for c in digits:
                            push(COORD0 + c)
                        vertices.append({"T": vt, "bin": tuple(vb), "nf": None})
                        mat_type.append(VTYPE_ID[vt])
                        mat_digits.append(digits)
                        bin_index[key] = len(vertices) - 1
                        refs.append(len(vertices) - 1)
                edges_out.append({"tau": e["tau"], "refs": refs, "cls": "observed"})
                continue
            tau = TAUS[choice - TAU_BASE]
            push(choice)
            refs = []
            ok_edge = True
            for st, slot_t in enumerate(SLOT_TYPES[tau]):
                want = VTYPE_ID[slot_t] if slot_t != "FIX" else 0
                cand = torch.tensor(
                    [mt == want and i not in refs for i, mt in enumerate(mat_type)],
                    dtype=torch.bool, device=dev) if mat_type else None
                static_ok = torch.zeros(VOCAB_C, dtype=torch.bool, device=dev)
                if slot_t != "FIX":
                    static_ok[NEW] = True
                if cand is None or not cand.any():
                    cand = None
                    if slot_t == "FIX":
                        ok_edge = False  # no FIX available (masked earlier, safety)
                        break
                sl, pl = next_logits()
                choice2 = sample_from(sl, pl, static_ok, cand)
                if choice2 >= VOCAB_C:  # pointer
                    vid = choice2 - VOCAB_C
                    push(PTR, vid)
                    refs.append(vid)
                else:  # NEW + 6 digits
                    push(NEW)
                    digits = []
                    coord_ok = torch.zeros(VOCAB_C, dtype=torch.bool, device=dev)
                    coord_ok[COORD0:] = True
                    ct = coord_temperature if coord_temperature is not None else 1.0
                    for _ in range(6):
                        sl, pl = next_logits()
                        sl = sl * (max(temperature, 1e-6) / max(ct, 1e-6))
                        c = sample_from(sl, pl, coord_ok, None) - COORD0
                        push(COORD0 + c)
                        digits.append(c)
                    b = tuple((digits[2 * a] << HI_BITS) | digits[2 * a + 1]
                              for a in range(3))
                    vertices.append({"T": slot_t, "bin": b, "nf": None})
                    mat_type.append(VTYPE_ID[slot_t])
                    mat_digits.append(digits)
                    bin_index.setdefault((slot_t, b), len(vertices) - 1)
                    refs.append(len(vertices) - 1)
            if ok_edge:
                edges_out.append({"tau": tau, "refs": refs, "cls": "generated"})
        return vertices, edges_out

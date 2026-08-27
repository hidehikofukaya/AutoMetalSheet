"""Vertex-stage autoregressive Transformer with constructive monotonic masking
(theory §5.1: sortedness enforced by a key-interval state machine, §5.2:
clamped alternating insertion for infilling; proposition F2/F3).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import BITS, HI_BITS
from .dataset_ar import (ADV, BOS, COORD0, PAD, STOP, TYPE_END, TYPE_MID, VOCAB,
                         vertex_key)

LO_MASK = (1 << HI_BITS) - 1
MAXC = (1 << HI_BITS) - 1  # 127
KEY_MIN = (-1, -1, -1, -1)
KEY_MAX = (3, 1 << BITS, 1 << BITS, 1 << BITS)


class CausalBlock(nn.Module):
    """Pre-LN causal self-attention via SDPA (flash/mem-efficient path)."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.p = dropout
        self.n1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(dim * 4, dim))

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        shape = (B, L, self.heads, D // self.heads)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, L, D)))
        return x + self.drop(self.mlp(self.n2(x)))


class VertexAR(nn.Module):
    """Decoder-only AR over [condition prefix][token stream], causal throughout
    (condition rows see only earlier condition rows — acceptable, they are a
    set encoded before the stream begins)."""

    def __init__(self, dim: int = 256, heads: int = 8, layers: int = 8,
                 cond_dim: int = 8, max_len: int = 9000, dropout: float = 0.0):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.cond_in = nn.Linear(cond_dim, dim)
        self.cond_seg = nn.Parameter(torch.zeros(dim))
        self.blocks = nn.ModuleList(CausalBlock(dim, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, VOCAB)

    def forward(self, stream, cond, cond_mask=None):
        """Returns logits aligned so logits[:, i] predicts stream[:, i+1].
        Designed for batch size 1 (no padding pollution); collate pads only
        within same-size batches."""
        B, S = stream.shape
        C = cond.shape[1]
        x = torch.cat([
            self.cond_in(cond) + self.cond_seg,
            self.tok(stream) + self.pos(torch.arange(S, device=stream.device))[None],
        ], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:, C:]))

    def loss(self, batch) -> torch.Tensor:
        logits = self.forward(batch["stream"][:, :-1], batch["cond"], batch["cond_mask"])
        target = batch["stream"][:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1), reduction="none")
        mask = batch["loss_mask"].reshape(-1).float()
        return (ce * mask).sum() / mask.sum().clamp(min=1)


# ---------------------------------------------------------------- sampling

def _digits(key: tuple) -> list[int]:
    """(z,y,x) -> 6 coarse/fine digits."""
    out = []
    for b in key:
        out += [b >> HI_BITS, b & LO_MASK]
    return out


def _digit_mask(low: list[int] | None, high: list[int] | None,
                tight_l: bool, tight_h: bool, pos: int) -> tuple[int, int, bool, bool]:
    """Allowed digit range at position pos, given exclusive bounds low<K<high."""
    lo_d = low[pos] if (low is not None and tight_l) else 0
    hi_d = high[pos] if (high is not None and tight_h) else MAXC
    return lo_d, hi_d, tight_l, tight_h


class MonotonicSampler:
    """Ancestral sampling under the key-interval mask. Guarantees sortedness,
    vertex non-duplication and correct ADV/STOP structure (proposition F3)."""

    def __init__(self, model: VertexAR, device: str):
        self.model = model
        self.device = device

    @torch.no_grad()
    def run(self, cond: torch.Tensor, cond_mask: torch.Tensor,
            observations: list[dict], max_vertices: int = 1200,
            temperature: float = 1.0, seed: int = 0) -> list[dict]:
        gen = torch.Generator(self.device).manual_seed(seed)
        obs = sorted(observations, key=vertex_key)
        obs_i = 0
        stream = [BOS]
        result: list[dict] = []
        last_key = KEY_MIN

        def logits_next():
            s = torch.tensor([stream], dtype=torch.long, device=self.device)
            out = self.model.forward(s, cond, cond_mask)
            return out[0, -1] / max(temperature, 1e-6)

        def pick(logits, allowed: torch.Tensor) -> int:
            logits = logits.masked_fill(~allowed, -1e9)
            probs = torch.softmax(logits, dim=-1)
            return int(torch.multinomial(probs, 1, generator=gen))

        while len(result) < max_vertices:
            next_obs_key = vertex_key(obs[obs_i]) if obs_i < len(obs) else KEY_MAX
            # ---- decision token: ADV / STOP / TYPE ----
            allowed = torch.zeros(VOCAB, dtype=torch.bool, device=self.device)
            if obs_i < len(obs):
                allowed[ADV] = True
            else:
                allowed[STOP] = True
            def key_int(coords):
                return (coords[0] << (2 * BITS)) | (coords[1] << BITS) | coords[2]

            top = 1 << (3 * BITS)
            for trank, ttok in ((1, TYPE_END), (2, TYPE_MID)):
                if last_key[0] > trank or next_obs_key[0] < trank:
                    continue  # every key of this type lies outside the interval
                low_i = key_int(last_key[1:]) if last_key[0] == trank else -1
                high_i = key_int(next_obs_key[1:]) if next_obs_key[0] == trank else top
                if high_i - low_i >= 2:  # an integer key strictly between exists
                    allowed[ttok] = True
            tok = pick(logits_next(), allowed)
            stream.append(tok)
            if tok == STOP:
                break
            if tok == ADV:
                v = obs[obs_i]
                obs_i += 1
                last_key = vertex_key(v)
                from .dataset_ar import vertex_tokens
                stream.extend(vertex_tokens(v))
                result.append(dict(v, observed=True))
                continue
            # ---- coordinate digits under the open interval (last_key, next_obs_key)
            trank = 1 if tok == TYPE_END else 2
            low = _digits(last_key[1:]) if last_key[0] == trank else None
            high = _digits(next_obs_key[1:]) if next_obs_key[0] == trank else None
            tight_l = low is not None
            tight_h = high is not None
            digits = []
            vals = torch.arange(128, device=self.device)
            for pos in range(6):
                lo_d = low[pos] if tight_l else 0
                hi_d = high[pos] if tight_h else MAXC
                ok = torch.ones(128, dtype=torch.bool, device=self.device)
                if tight_l:
                    if pos < 5:
                        # v > lo_d always fine; v == lo_d only if the remaining
                        # low digits are not all at max (an escape must exist)
                        can_eq = not all(d == MAXC for d in low[pos + 1:])
                        ok &= (vals > lo_d) | ((vals == lo_d) & can_eq)
                    else:
                        ok &= vals > lo_d  # strict: open interval
                if tight_h:
                    if pos < 5:
                        can_eq = not all(d == 0 for d in high[pos + 1:])
                        ok &= (vals < hi_d) | ((vals == hi_d) & can_eq)
                    else:
                        ok &= vals < hi_d  # strict: open interval
                allowed = torch.zeros(VOCAB, dtype=torch.bool, device=self.device)
                allowed[COORD0:] = ok
                assert allowed.any(), "digit interval empty (mask machine bug)"
                d = pick(logits_next(), allowed) - COORD0
                stream.append(COORD0 + d)
                digits.append(d)
                tight_l = tight_l and d == lo_d
                tight_h = tight_h and d == hi_d
            b = tuple((digits[2 * a] << HI_BITS) | digits[2 * a + 1] for a in range(3))
            v = {"T": "END" if tok == TYPE_END else "MID", "bin": b, "nf": None,
                 "observed": False}
            key = (trank, *b)
            assert key > last_key and key < next_obs_key, "monotonic mask violated"
            last_key = key
            result.append(v)
        return result

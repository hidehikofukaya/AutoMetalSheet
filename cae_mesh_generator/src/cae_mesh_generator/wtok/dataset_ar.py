"""Phase 1 dataset: vertex-stage AR sequences with clamped alternating
insertion (theory §5.1-5.2, §7 curriculum mu_1).

Every epoch each part gets a fresh observation subset (rate ~ U(0,1)) and the
sequence is rebuilt: observed vertices cost one ADV decision (clamped tokens
follow as context, no loss), unobserved vertices are insertion targets.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import BITS, HI_BITS
from .constants import stable_seed

LO_MASK = (1 << HI_BITS) - 1
N_BINS = 1 << BITS

# vocabulary
PAD, BOS, STOP, ADV, TYPE_END, TYPE_MID = 0, 1, 2, 3, 4, 5
COORD0 = 6
VOCAB = COORD0 + (1 << HI_BITS)  # 134
TYPE_TOK = {"END": TYPE_END, "MID": TYPE_MID}
T_RANK = {"END": 1, "MID": 2}


def vertex_key(v) -> tuple:
    return (T_RANK[v["T"]], *v["bin"])


def vertex_tokens(v) -> list[int]:
    out = [TYPE_TOK[v["T"]]]
    for axis in range(3):
        b = v["bin"][axis]
        out += [COORD0 + (b >> HI_BITS), COORD0 + (b & LO_MASK)]
    return out


def mirror_q_vertices(vertices: list[dict], axis: int) -> list[dict]:
    out = []
    for v in vertices:
        b = list(v["bin"])
        b[axis] = (N_BINS - 1) - b[axis]
        nf = v.get("nf")
        if nf is not None:
            nf = list(nf)
            nf[axis] = -nf[axis]
        out.append({"T": v["T"], "bin": tuple(b), "nf": nf})
    return out


class Part:
    def __init__(self, path: pathlib.Path):
        d = json.loads(path.read_text(encoding="utf-8"))["Q"]
        self.name = path.stem
        self.env_lo = np.asarray(d["env_lo"], dtype=np.float64)
        self.env_hi = np.asarray(d["env_hi"], dtype=np.float64)
        self.vertices = [
            {"T": v["T"], "bin": tuple(v["bin"]), "nf": v.get("nf")} for v in d["vertices"]
        ]

    def variant(self, mirror: bool):
        return self.variant_axis(1 if mirror else None)

    def variant_axis(self, axis: int | None):
        vs = mirror_q_vertices(self.vertices, axis) if axis is not None else self.vertices
        fix = [v for v in vs if v["T"] == "FIX"]
        gen = sorted([v for v in vs if v["T"] != "FIX"], key=vertex_key)
        return fix, gen


def load_parts(dataset_dir: pathlib.Path) -> list[Part]:
    return [Part(f) for f in sorted((dataset_dir / "parts").glob("*.json"))]


def cond_features(fix: list[dict], env_lo, env_hi) -> np.ndarray:
    """Per-FIX condition features + one global envelope row (marked by flag)."""
    span = np.maximum(env_hi - env_lo, 1e-9)
    rows = []
    for v in fix:
        xyz = (np.asarray(v["bin"], dtype=np.float64) + 0.5) / N_BINS
        nf = np.asarray(v["nf"] if v["nf"] is not None else [0, 0, 1], dtype=np.float64)
        rows.append(np.concatenate([xyz, nf, [1.0, 0.0]]))
    g = np.concatenate([span / span.max(), [np.log(span.max()) / 10.0], [0.0, 0.0, 0.0, 1.0]])
    rows.append(g)
    return np.asarray(rows, dtype=np.float32)


def build_sequence(gen_vertices: list[dict], observed: np.ndarray
                   ) -> tuple[list[int], list[int]]:
    """Returns (stream tokens incl BOS/STOP, per-position loss mask for the
    *target* at that step). Alignment: logits[i] predict stream[i+1]."""
    stream = [BOS]
    loss = []  # loss flag for predicting stream[i+1] from position i
    for v, obs in zip(gen_vertices, observed):
        toks = vertex_tokens(v)
        if obs:
            stream.append(ADV)
            loss.append(1)          # the ADV decision is supervised
            stream += toks
            loss += [0] * len(toks)  # clamped context, not predicted
        else:
            stream += toks
            loss += [1] * len(toks)
    stream.append(STOP)
    loss.append(1)
    return stream, loss


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


class VertexARDataset(Dataset):
    """v2: multi-axis mirror, +-jitter-bin coordinate noise, and an observation
    rate curriculum — after `stage2_after` epochs the rate distribution shifts
    toward sparse/empty observations (theory §7: mu_1 -> mu_3 staging, the fix
    for early-STOP in pure generation)."""

    def __init__(self, parts: list[Part], mirror_axes: tuple[str, ...] = (),
                 base_seed: int = 0, obs_rate: float | None = None,
                 jitter_bins: int = 0, stage2_after: int | None = None):
        self.items = [(p, None) for p in parts]
        self.items += [(p, AXIS_INDEX[a]) for p in parts for a in mirror_axes]
        self.base_seed = base_seed
        self.epoch = 0
        self.obs_rate = obs_rate  # None -> curriculum
        self.jitter_bins = jitter_bins
        self.stage2_after = stage2_after

    def set_epoch(self, e: int) -> None:
        self.epoch = int(e)

    def _rate(self, rng: np.random.Generator) -> float:
        if self.obs_rate is not None:
            return self.obs_rate
        if self.stage2_after is not None and self.epoch > self.stage2_after:
            if rng.uniform() < 0.3:
                return 0.0                      # pure-generation samples
            return float(rng.beta(0.6, 2.2))    # sparse observations
        return float(rng.uniform())             # mu_1

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        part, mir_axis = self.items[i]
        rng = np.random.default_rng(
            stable_seed(self.base_seed + 999983 * self.epoch, f"{part.name}|{mir_axis}"))
        fix, gen = part.variant_axis(mir_axis)
        if self.jitter_bins > 0 and len(gen):
            bins = np.asarray([v["bin"] for v in gen], dtype=np.int64)
            bins += rng.integers(-self.jitter_bins, self.jitter_bins + 1, bins.shape)
            bins = np.clip(bins, 0, N_BINS - 1)
            seen = set()
            jittered = []
            for v, b in zip(gen, bins):
                key = (v["T"], tuple(int(x) for x in b))
                if key not in seen:
                    seen.add(key)
                    jittered.append({"T": v["T"], "bin": key[1], "nf": v["nf"]})
            gen = sorted(jittered, key=vertex_key)
        rate = self._rate(rng)
        observed = rng.uniform(size=len(gen)) < rate
        stream, loss = build_sequence(gen, observed)
        return {
            "stream": torch.tensor(stream, dtype=torch.long),
            "loss_mask": torch.tensor(loss, dtype=torch.bool),
            "cond": torch.from_numpy(cond_features(fix, part.env_lo, part.env_hi)),
            "name": f"{part.name}|m{mir_axis}",
        }


def collate(batch: list[dict]) -> dict:
    max_s = max(len(b["stream"]) for b in batch)
    max_c = max(len(b["cond"]) for b in batch)
    B = len(batch)
    stream = torch.full((B, max_s), PAD, dtype=torch.long)
    loss = torch.zeros((B, max_s - 1), dtype=torch.bool)
    cond = torch.zeros((B, max_c, batch[0]["cond"].shape[1]))
    cond_mask = torch.zeros((B, max_c), dtype=torch.bool)
    for i, b in enumerate(batch):
        s, l, c = b["stream"], b["loss_mask"], b["cond"]
        stream[i, : len(s)] = s
        loss[i, : len(l)] = l
        cond[i, : len(c)] = c
        cond_mask[i, : len(c)] = True
    return {"stream": stream, "loss_mask": loss, "cond": cond, "cond_mask": cond_mask,
            "names": [b["name"] for b in batch]}

"""Curve-major AR dataset (definition 6'): one training item = one part
variant serialized as wire commands with pointer/new slots.

Stream records:
  BOS, then per edge (canonical order): observed -> [ADV] + clamped edge
  tokens (loss on ADV only); unobserved -> edge tokens with loss.
  Edge tokens: [TAU][slot...], slot = PTR record (pointer to materialized
  vertex) or [NEW][6 coord tokens]. STOP ends the stream.

Augmentation: one variant per part per epoch (none/x/y/z mirror) + coordinate
jitter; observation-rate curriculum identical to the vertex stage.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import BITS, HI_BITS
from .codec_curve import SLOT_TYPES, sigma_curve
from .constants import stable_seed

LO_MASK = (1 << HI_BITS) - 1
N_BINS = 1 << BITS

# static vocabulary
PAD, BOS, STOP, ADV = 0, 1, 2, 3
TAU_BASE = 4
TAUS = ("LINE", "ARC", "CIRCLE", "CIRCLE_C")
TAU_TOK = {t: TAU_BASE + i for i, t in enumerate(TAUS)}
NEW, PTR = 8, 9
COORD0 = 10
VOCAB_C = COORD0 + (1 << HI_BITS)  # 138
VTYPE_ID = {"FIX": 0, "END": 1, "MID": 2}
SLOT_TYPE_ID = {"none": 0, "END": 1, "MID": 2, "FIX": 3}


def digits_of(b: tuple) -> list[int]:
    out = []
    for axis in range(3):
        out += [b[axis] >> HI_BITS, b[axis] & LO_MASK]
    return out


class CurvePart:
    def __init__(self, path: pathlib.Path):
        d = json.loads(path.read_text(encoding="utf-8"))["Q"]
        self.name = path.stem
        self.env_lo = np.asarray(d["env_lo"])
        self.env_hi = np.asarray(d["env_hi"])
        self.vertices = [{"T": v["T"], "bin": tuple(v["bin"]), "nf": v.get("nf")}
                         for v in d["vertices"]]
        from .codec_curve import traversal_order
        self.edges = traversal_order(
            [{"tau": e["tau"], "refs": list(e["refs"]), "cls": e["cls"]}
             for e in d["edges"]])


def load_curve_parts(dataset_dir: pathlib.Path) -> list[CurvePart]:
    return [CurvePart(f) for f in sorted((dataset_dir / "parts").glob("*.json"))]


def transform_vertices(vertices, mirror_axis, jitter_bins, rng):
    out = []
    for v in vertices:
        b = list(v["bin"])
        nf = list(v["nf"]) if v["nf"] is not None else None
        if mirror_axis is not None:
            b[mirror_axis] = (N_BINS - 1) - b[mirror_axis]
            if nf is not None:
                nf[mirror_axis] = -nf[mirror_axis]
        if jitter_bins and v["T"] != "FIX":
            b = [int(np.clip(x + rng.integers(-jitter_bins, jitter_bins + 1),
                             0, N_BINS - 1)) for x in b]
        out.append({"T": v["T"], "bin": tuple(b), "nf": nf})
    return out


def build_curve_item(vertices, edges, observed_mask, rng):
    """Returns tensors for one sequence + the materialized vertex table."""
    Q = {"vertices": vertices, "edges": edges}
    seq = sigma_curve(Q)
    # materialized vertex table: FIX block, then NEW in emission order
    fix_ids = [i for i, v in enumerate(vertices) if v["T"] == "FIX"]
    mat_type = [0] * len(fix_ids)
    mat_digits = [digits_of(vertices[i]["bin"]) for i in fix_ids]
    mat_pos: list[int] = [-1] * len(fix_ids)  # FIX available from the start

    in_tok, in_ptr, target, loss, slot_next = [BOS], [-1], [], [], []

    def emit(rec: int | tuple, lossy: bool, slot_t: str = "none"):
        """Append one record. `rec` is a static id or ('ptr', vid).
        slot_t is the type constraint OF THIS record when it is a pointer."""
        if isinstance(rec, tuple):
            target.append(VOCAB_C + rec[1])
            slot_next.append(SLOT_TYPE_ID[slot_t])
            in_tok.append(PTR)
            in_ptr.append(rec[1])
        else:
            target.append(rec)
            slot_next.append(0)
            in_tok.append(rec)
            in_ptr.append(-1)
        loss.append(1 if lossy else 0)

    for e, obs in zip(seq["edges"], observed_mask):
        if obs:
            emit(ADV, True)
        lossy = not obs
        emit(TAU_TOK[e["tau"]], lossy)
        for st, s in zip(SLOT_TYPES[e["tau"]], e["slots"]):
            if s["kind"] == "ptr":
                emit(("ptr", s["id"]), lossy, st)
            else:
                emit(NEW, lossy)
                for c in s["coords"]:
                    emit(COORD0 + c, lossy)
                mat_type.append(VTYPE_ID[st])
                mat_digits.append(list(s["coords"]))
                mat_pos.append(len(target))  # referable from the next record on
    emit(STOP, True)

    return {
        "in_tok": torch.tensor(in_tok[:-1], dtype=torch.long),
        "in_ptr": torch.tensor(in_ptr[:-1], dtype=torch.long),
        "target": torch.tensor(target, dtype=torch.long),
        "loss_mask": torch.tensor(loss, dtype=torch.bool),
        "slot_next": torch.tensor(slot_next, dtype=torch.long),
        "mat_type": torch.tensor(mat_type, dtype=torch.long),
        "mat_digits": torch.tensor(mat_digits, dtype=torch.long)
        if mat_digits else torch.zeros(0, 6, dtype=torch.long),
        "mat_pos": torch.tensor(mat_pos, dtype=torch.long),
        "n_fix": len(fix_ids),
    }


class CurveARDataset(Dataset):
    def __init__(self, parts, augment: bool, base_seed: int = 0,
                 obs_rate: float | None = None, jitter_bins: int = 1,
                 stage2_after: int | None = None):
        self.parts = parts
        self.augment = augment
        self.base_seed = base_seed
        self.epoch = 0
        self.obs_rate = obs_rate
        self.jitter_bins = jitter_bins if augment else 0
        self.stage2_after = stage2_after

    def set_epoch(self, e: int) -> None:
        self.epoch = int(e)

    def _rate(self, rng) -> float:
        if self.obs_rate is not None:
            return self.obs_rate
        if self.stage2_after is not None and self.epoch > self.stage2_after:
            if rng.uniform() < 0.3:
                return 0.0
            return float(rng.beta(0.6, 2.2))
        return float(rng.uniform())

    def __len__(self) -> int:
        return len(self.parts)

    def __getitem__(self, i: int) -> dict:
        p = self.parts[i]
        rng = np.random.default_rng(
            stable_seed(self.base_seed + 999983 * self.epoch, p.name))
        axis = rng.choice([None, 0, 1, 2]) if self.augment else None
        vs = transform_vertices(p.vertices, axis, self.jitter_bins, rng)
        rate = self._rate(rng)
        observed = rng.uniform(size=len(p.edges)) < rate
        item = build_curve_item(vs, p.edges, observed, rng)
        from .dataset_ar import cond_features
        fix = [v for v in vs if v["T"] == "FIX"]
        item["cond"] = torch.from_numpy(cond_features(fix, p.env_lo, p.env_hi))
        item["name"] = p.name
        item["part_index"] = i
        return item

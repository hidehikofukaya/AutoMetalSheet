"""Curve-major AR v2: the three rollout-gap fixes from the ep200 diagnosis.

1. Relative coarse coordinates: each NEW vertex's coarse digits are emitted as
   signed offsets from the previously materialized vertex (fine digits stay
   absolute). Offsets in [-127, 127] cover any coarse-to-coarse move exactly.
   Absolute positions remain known at every sampling step, so future NG-zone
   masks (theory F5) apply as reference-dependent absolute masks.
2. Metric-aware coordinate loss: soft-target CE (kernel over +-2 neighboring
   bins) for offset and fine digits -- near misses stop counting as total misses.
3. Self-predicted edge-count conditioning (SOFT): a count-bucket token is
   appended to the condition; at inference the model's own count head fills it.
   STOP is still learned and can override -- a hint, not a constraint, so
   richer future conditions cannot be shackled by it.

Usage:
  python -m cae_mesh_generator.wtok.curve2 --dataset ../runs/wtok_synth \
      --output-dir ../runs/wtok_curve2_v1 --epochs 150 \
      --val-list ../runs/wtok_curve_synth_v1/val_names.json --device cuda
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

from .codec import realize_points
from .codec_curve import SLOT_TYPES, sigma_curve
from .constants import HI_BITS
from .dataset_curve import (ADV, BOS, COORD0, NEW, PTR, STOP, TAU_BASE, TAU_TOK,
                            TAUS, VOCAB_C, VTYPE_ID, CurveARDataset, digits_of,
                            load_curve_parts, transform_vertices)
from .model_curve import CurveAR
from .train_curve import part_cond, realized_q, to_device
from .train_ar import chamfer_mm
from .constants import stable_seed
from .constants import safe_save

LO_MASK = (1 << HI_BITS) - 1
OFF0 = VOCAB_C                 # 255 offset tokens: value = OFF0 + (off + 127)
N_OFF = 255
VOCAB2 = OFF0 + N_OFF          # 393
N_BUCKETS = 32
COUNT_LOSS_W = 0.2


def edge_bucket(n_edges: int) -> int:
    return min(N_BUCKETS - 1, n_edges // 16)


def coarse_of(b: tuple) -> tuple:
    return (b[0] >> HI_BITS, b[1] >> HI_BITS, b[2] >> HI_BITS)


def build_curve_item2(vertices, edges, observed_mask) -> dict:
    Q = {"vertices": vertices, "edges": edges}
    seq = sigma_curve(Q)
    fix_ids = [i for i, v in enumerate(vertices) if v["T"] == "FIX"]
    mat_type = [0] * len(fix_ids)
    mat_digits = [digits_of(vertices[i]["bin"]) for i in fix_ids]
    mat_pos: list[int] = [-1] * len(fix_ids)
    ref = coarse_of(vertices[fix_ids[-1]]["bin"]) if fix_ids else (64, 64, 64)

    in_tok, in_ptr, target, loss, slot_next = [BOS], [-1], [], [], []

    def emit(rec, lossy, slot_t: int = 0):
        if isinstance(rec, tuple):  # pointer
            target.append(VOCAB2 + rec[1])
            slot_next.append(slot_t)
            in_tok.append(PTR)
            in_ptr.append(rec[1])
        else:
            target.append(rec)
            slot_next.append(0)
            in_tok.append(rec)
            in_ptr.append(-1)
        loss.append(1 if lossy else 0)

    from .dataset_curve import SLOT_TYPE_ID
    for e, obs in zip(seq["edges"], observed_mask):
        if obs:
            emit(ADV, True)
        lossy = not obs
        emit(TAU_TOK[e["tau"]], lossy)
        for st, s in zip(SLOT_TYPES[e["tau"]], e["slots"]):
            if s["kind"] == "ptr":
                emit(("ptr", s["id"]), lossy, SLOT_TYPE_ID[st])
            else:
                emit(NEW, lossy)
                c = s["coords"]  # [zc,zf,yc,yf,xc,xf] absolute
                new_coarse = (c[0], c[2], c[4])
                for a in range(3):
                    off = new_coarse[a] - ref[a]
                    emit(OFF0 + off + 127, lossy)
                    emit(COORD0 + c[2 * a + 1], lossy)
                ref = new_coarse
                mat_type.append(VTYPE_ID[st])
                mat_digits.append(list(c))
                mat_pos.append(len(target))
    emit(STOP, True)
    return {
        "in_tok": torch.tensor(in_tok[:-1], dtype=torch.long),
        "in_ptr": torch.tensor(in_ptr[:-1], dtype=torch.long),
        "target": torch.tensor(target, dtype=torch.long),
        "loss_mask": torch.tensor(loss, dtype=torch.bool),
        "slot_next": torch.tensor(slot_next, dtype=torch.long),
        "mat_type": torch.tensor(mat_type, dtype=torch.long),
        "mat_digits": (torch.tensor(mat_digits, dtype=torch.long)
                       if mat_digits else torch.zeros(0, 6, dtype=torch.long)),
        "mat_pos": torch.tensor(mat_pos, dtype=torch.long),
        "bucket": torch.tensor(edge_bucket(len(edges)), dtype=torch.long),
    }


class CurveARDataset2(CurveARDataset):
    def __getitem__(self, i: int) -> dict:
        p = self.parts[i]
        rng = np.random.default_rng(
            stable_seed(self.base_seed + 999983 * self.epoch, p.name))
        axis = rng.choice([None, 0, 1, 2]) if self.augment else None
        vs = transform_vertices(p.vertices, axis, self.jitter_bins, rng)
        rate = self._rate(rng)
        observed = rng.uniform(size=len(p.edges)) < rate
        item = build_curve_item2(vs, p.edges, observed)
        from .dataset_ar import cond_features
        fix = [v for v in vs if v["T"] == "FIX"]
        item["cond"] = torch.from_numpy(cond_features(fix, p.env_lo, p.env_hi))
        item["name"] = p.name
        return item


class CurveAR2(CurveAR):
    def __init__(self, dim=256, heads=8, layers=8, dropout=0.0):
        super().__init__(dim, heads, layers, dropout=dropout)
        self.tok = nn.Embedding(VOCAB2, dim)
        self.head = nn.Linear(dim, VOCAB2)
        self.bucket_emb = nn.Embedding(N_BUCKETS, dim)
        self.count_head = nn.Linear(dim, N_BUCKETS)

    def cond_rows(self, cond):
        return self.cond_in(cond) + self.cond_seg

    def predict_bucket_logits(self, cond):
        return self.count_head(self.cond_rows(cond).mean(dim=0))

    def forward(self, item):
        in_tok, in_ptr = item["in_tok"], item["in_ptr"]
        vemb = self.vertex_embed(item["mat_type"], item["mat_digits"])
        x = self.tok(in_tok)
        has_ptr = in_ptr >= 0
        if has_ptr.any():
            x = x.clone()
            x[has_ptr] = x[has_ptr] + vemb[in_ptr[has_ptr]]
        x = x + self.pos(torch.arange(len(in_tok), device=in_tok.device))
        cond = self.cond_rows(item["cond"])
        cond = torch.cat([cond, self.bucket_emb(item["bucket"])[None]], dim=0)
        h = torch.cat([cond, x], dim=0)[None]
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h[0, len(cond):])
        static = self.head(h)
        q = self.q_proj(h)
        k = self.k_proj(vemb)
        ptr = (q @ k.T) / (self.dim ** 0.5) if len(k) else torch.zeros(
            len(h), 0, device=h.device)
        return static, ptr

    def loss(self, item) -> torch.Tensor:
        static, ptr = self.forward(item)
        S = static.shape[0]
        pos_idx = torch.arange(S, device=static.device)
        if ptr.shape[1]:
            avail = item["mat_pos"][None, :] <= pos_idx[:, None]
            st = item["slot_next"]
            mtype = item["mat_type"]
            want = torch.full_like(st, -1)
            want[st == 1] = 1
            want[st == 2] = 2
            want[st == 3] = 0
            tmatch = (mtype[None, :] == want[:, None]) & (want[:, None] >= 0)
            ptr = ptr.masked_fill(~(avail & tmatch), -1e9)
        logits = torch.cat([static, ptr], dim=1)
        logp = F.log_softmax(logits, dim=1)
        tgt = item["target"]
        nll = -logp.gather(1, tgt[:, None]).squeeze(1)
        # soft-target CE on coordinate digits: kernel over +-2 bins in-block
        is_fine = (tgt >= COORD0) & (tgt < COORD0 + 128)
        is_off = (tgt >= OFF0) & (tgt < OFF0 + N_OFF)
        coordish = is_fine | is_off
        if coordish.any():
            lo = torch.where(is_fine, COORD0, OFF0)
            hi = torch.where(is_fine, COORD0 + 128, OFF0 + N_OFF)
            kernel = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1], device=logits.device)
            acc = torch.zeros_like(nll)
            wsum = torch.zeros_like(nll)
            for k_i, d in enumerate(range(-2, 3)):
                idx = (tgt + d).clamp(min=0, max=logits.shape[1] - 1)
                valid = coordish & (tgt + d >= lo) & (tgt + d < hi)
                w = kernel[k_i] * valid.float()
                acc = acc - w * logp.gather(1, idx[:, None]).squeeze(1)
                wsum = wsum + w
            soft = acc / wsum.clamp(min=1e-9)
            nll = torch.where(coordish, soft, nll)
        m = item["loss_mask"].float()
        main = (nll * m).sum() / m.sum().clamp(min=1)
        count = F.cross_entropy(self.predict_bucket_logits(item["cond"])[None],
                                item["bucket"][None])
        return main + COUNT_LOSS_W * count


class CurveSampler2:
    def __init__(self, model: CurveAR2, device: str):
        self.model = model
        self.device = device

    @torch.no_grad()
    def run(self, cond, fix_vertices, observed_edges=None, max_edges: int = 400,
            temperature: float = 1.0, seed: int = 0, bucket: int | None = None):
        gen = torch.Generator(self.device).manual_seed(seed)
        dev = self.device
        obs = observed_edges or []
        obs_i = 0
        if bucket is None:
            bucket = int(self.model.predict_bucket_logits(cond).argmax())
        bucket_t = torch.tensor(bucket, dtype=torch.long, device=dev)
        vertices = [dict(v) for v in fix_vertices]
        mat_type = [VTYPE_ID[v["T"]] for v in vertices]
        mat_digits = [digits_of(v["bin"]) for v in vertices]
        bin_index = {(v["T"], tuple(v["bin"])): i for i, v in enumerate(vertices)}
        ref = coarse_of(fix_vertices[-1]["bin"]) if fix_vertices else (64, 64, 64)
        in_tok, in_ptr = [BOS], [-1]
        edges_out = []

        def item():
            return {"in_tok": torch.tensor(in_tok, dtype=torch.long, device=dev),
                    "in_ptr": torch.tensor(in_ptr, dtype=torch.long, device=dev),
                    "mat_type": torch.tensor(mat_type, dtype=torch.long, device=dev),
                    "mat_digits": (torch.tensor(mat_digits, dtype=torch.long, device=dev)
                                   if mat_digits else torch.zeros(0, 6, dtype=torch.long,
                                                                  device=dev)),
                    "cond": cond, "bucket": bucket_t}

        def next_logits():
            static, ptr = self.model.forward(item())
            return static[-1] / max(temperature, 1e-6), ptr[-1] / max(temperature, 1e-6)

        def sample_from(sl, pl, static_ok, ptr_ok):
            sl = sl.masked_fill(~static_ok, -1e9)
            if ptr_ok is not None and len(pl):
                logits = torch.cat([sl, pl.masked_fill(~ptr_ok, -1e9)])
            else:
                logits = torch.cat([sl, torch.full_like(pl, -1e9)]) if len(pl) else sl
            return int(torch.multinomial(torch.softmax(logits, -1), 1, generator=gen))

        def push(tok, ptr_id=-1):
            in_tok.append(tok)
            in_ptr.append(ptr_id)

        def materialize(vt, b):
            nonlocal ref
            vertices.append({"T": vt, "bin": b, "nf": None})
            mat_type.append(VTYPE_ID[vt])
            mat_digits.append(digits_of(b))
            bin_index.setdefault((vt, b), len(vertices) - 1)
            ref = coarse_of(b)
            return len(vertices) - 1

        while len(edges_out) < max_edges:
            static_ok = torch.zeros(VOCAB2, dtype=torch.bool, device=dev)
            if obs_i < len(obs):
                static_ok[ADV] = True
            for t in TAUS:
                static_ok[TAU_TOK[t]] = True
            if obs_i >= len(obs):
                static_ok[STOP] = True
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
                for (vt, vb) in e["verts"]:
                    key = (vt, tuple(vb))
                    if key in bin_index:
                        push(PTR, bin_index[key])
                        refs.append(bin_index[key])
                    else:
                        push(NEW)
                        c = digits_of(tuple(vb))
                        nc = coarse_of(tuple(vb))
                        for a in range(3):
                            push(OFF0 + (nc[a] - ref[a]) + 127)
                            push(COORD0 + c[2 * a + 1])
                        refs.append(materialize(vt, tuple(vb)))
                edges_out.append({"tau": e["tau"], "refs": refs, "cls": "observed"})
                continue
            tau = TAUS[choice - TAU_BASE]
            push(choice)
            refs = []
            ok_edge = True
            for slot_t in SLOT_TYPES[tau]:
                want = VTYPE_ID[slot_t] if slot_t != "FIX" else 0
                cand = (torch.tensor([mt == want and i not in refs
                                      for i, mt in enumerate(mat_type)],
                                     dtype=torch.bool, device=dev)
                        if mat_type else None)
                static_ok = torch.zeros(VOCAB2, dtype=torch.bool, device=dev)
                if slot_t != "FIX":
                    static_ok[NEW] = True
                if cand is None or not cand.any():
                    cand = None
                    if slot_t == "FIX":
                        ok_edge = False
                        break
                sl, pl = next_logits()
                choice2 = sample_from(sl, pl, static_ok, cand)
                if choice2 >= VOCAB2:
                    vid = choice2 - VOCAB2
                    push(PTR, vid)
                    refs.append(vid)
                else:
                    push(NEW)
                    coarse, fine = [], []
                    for a in range(3):
                        off_ok = torch.zeros(VOCAB2, dtype=torch.bool, device=dev)
                        lo = max(0, ref[a] - 127) - ref[a]
                        hi = min(127, ref[a] + 127) - ref[a]
                        off_ok[OFF0 + lo + 127: OFF0 + hi + 128] = True
                        sl, pl = next_logits()
                        off = sample_from(sl, pl, off_ok, None) - OFF0 - 127
                        push(OFF0 + off + 127)
                        coarse.append(int(np.clip(ref[a] + off, 0, 127)))
                        fine_ok = torch.zeros(VOCAB2, dtype=torch.bool, device=dev)
                        fine_ok[COORD0: COORD0 + 128] = True
                        sl, pl = next_logits()
                        f = sample_from(sl, pl, fine_ok, None) - COORD0
                        push(COORD0 + f)
                        fine.append(f)
                    b = tuple((coarse[a] << HI_BITS) | fine[a] for a in range(3))
                    refs.append(materialize(slot_t, b))
            if ok_edge:
                edges_out.append({"tau": tau, "refs": refs, "cls": "generated"})
        return vertices, edges_out


@torch.no_grad()
def sample_eval2(model, parts, device, obs_rate, max_parts=8, seed=0):
    sampler = CurveSampler2(model, device)
    cds, n_edges = [], []
    for p in parts[:max_parts]:
        cond, fix = part_cond(p, device)
        rng = np.random.default_rng(seed + 11)
        observed = [{"tau": e["tau"],
                     "verts": [(p.vertices[r]["T"], p.vertices[r]["bin"])
                               for r in e["refs"]]}
                    for e in p.edges if rng.uniform() < obs_rate]
        v, e = sampler.run(cond, fix, observed, seed=seed)
        gt = realize_points(realized_q(p, p.vertices, p.edges))
        gen = realize_points(realized_q(p, v, e))
        # an untrained/degenerate model can emit nothing; keep the metric finite
        cds.append(chamfer_mm(gen[::3], gt[::3]) if len(gen) else float("nan"))
        n_edges.append(len(e))
    finite = [c for c in cds if np.isfinite(c)]
    return {"curve_chamfer_mm": float(np.mean(finite)) if finite else float("nan"),
            "n_edges_mean": float(np.mean(n_edges))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--jitter-bins", type=int, default=1)
    ap.add_argument("--stage2-after", type=int, default=30)
    ap.add_argument("--val-list", required=True)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--sample-every", type=int, default=25)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", default="")
    ap.add_argument("--resume-dir", default="",
                    help="read-only dir holding last.pt/history.json (Kaggle: a "
                         "previous session's Output mounted as an input dataset)")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="stop cleanly after this wall time so the platform saves "
                         "the output (Kaggle sessions are killed at 12h)")
    args = ap.parse_args()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    parts = load_curve_parts(pathlib.Path(args.dataset))
    val_names = set(json.loads(pathlib.Path(args.val_list).read_text(encoding="utf-8")))
    train_parts = [p for p in parts if p.name not in val_names]
    val_parts = [p for p in parts if p.name in val_names]
    print(f"parts: train {len(train_parts)} val {len(val_parts)}")
    (out / "split.json").write_text(json.dumps(
        {"val": [p.name for p in val_parts], "args": vars(args)}), encoding="utf-8")

    train_ds = CurveARDataset2(train_parts, augment=True, jitter_bins=args.jitter_bins,
                               stage2_after=args.stage2_after)
    val_gen = CurveARDataset2(val_parts, augment=False, obs_rate=0.0, base_seed=555)
    val_half = CurveARDataset2(val_parts, augment=False, obs_rate=0.5, base_seed=555)

    model = CurveAR2(args.dim, 8, args.layers, dropout=args.dropout).to(args.device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    history, best = [], float("inf")
    start_epoch = 1
    if args.resume_dir and not args.resume:
        rd = pathlib.Path(args.resume_dir)
        if (rd / "last.pt").exists():
            args.resume = str(rd / "last.pt")
            src_hist = rd / "history.json"
            dst_hist = out / "history.json"
            if src_hist.exists() and not dst_hist.exists():
                dst_hist.write_text(src_hist.read_text(encoding="utf-8"),
                                    encoding="utf-8")
            print(f"[resume-dir] picked up {args.resume}")
        else:
            print(f"[resume-dir] no last.pt in {rd} -- starting fresh")
    if args.resume:
        ck = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        hist_file = out / "history.json"
        if hist_file.exists():
            history = [r for r in json.loads(hist_file.read_text(encoding="utf-8"))
                       if r["epoch"] < start_epoch]
            best = min((r["val_nll_gen"] for r in history if "val_nll_gen" in r),
                       default=float("inf"))
        print(f"resumed at epoch {start_epoch}")

    t_start = time.time()
    budget = args.max_hours * 3600.0
    for epoch in range(start_epoch, args.epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()
        t0, tot = time.time(), 0.0
        idx = torch.randperm(len(train_ds)).tolist()
        opt.zero_grad()
        for step, i in enumerate(idx):
            item = to_device(train_ds[i], args.device)
            loss = model.loss(item) / args.accum
            loss.backward()
            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
            tot += float(loss.detach()) * args.accum
        row = {"epoch": epoch, "train_nll": tot / len(idx),
               "seconds": round(time.time() - t0, 1)}
        if epoch % args.val_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                vls = [float(np.mean([float(model.loss(to_device(ds[i], args.device)))
                                      for i in range(len(ds))]))
                       for ds in (val_gen, val_half)]
            row["val_nll_gen"], row["val_nll_half"] = vls
            ck = {"model": model.state_dict(), "opt": opt.state_dict(),
                  "args": vars(args), "epoch": epoch}
            if vls[0] < best:
                best = vls[0]
                safe_save(ck, out / "best.pt")
            safe_save(ck, out / "last.pt")
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            model.eval()
            row["gen"] = sample_eval2(model, val_parts, args.device, 0.0)
            row["infill50"] = sample_eval2(model, val_parts, args.device, 0.5)
        history.append(row)
        (out / "history.json").write_text(json.dumps(history), encoding="utf-8")
        elapsed = time.time() - t_start
        msg = f"epoch {epoch}: nll {row['train_nll']:.4f} ({row['seconds']}s)"
        if "val_nll_gen" in row:
            msg += f"  val gen {row['val_nll_gen']:.4f} half {row['val_nll_half']:.4f}"
        if "gen" in row:
            msg += (f"  | gen {row['gen']['curve_chamfer_mm']:.1f}mm "
                    f"(E={row['gen']['n_edges_mean']:.0f}) "
                    f"infill {row['infill50']['curve_chamfer_mm']:.1f}mm")
        print(msg, flush=True)
        # time budget: save and exit cleanly so the platform keeps the output
        if budget and elapsed + row["seconds"] * 1.5 > budget:
            ck = {"model": model.state_dict(), "opt": opt.state_dict(),
                  "args": vars(args), "epoch": epoch}
            safe_save(ck, out / "last.pt")
            print(f"[budget] stopping at epoch {epoch} after {elapsed/3600:.2f}h. "
                  f"Resume with: --resume-dir <this output dir> --epochs {args.epochs}",
                  flush=True)
            break


if __name__ == "__main__":
    main()

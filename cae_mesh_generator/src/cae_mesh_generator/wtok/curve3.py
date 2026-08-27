"""curve3 = v2.1, fully learned ("E2E") and fast.

Changes vs curve2 (each grounded in the v2/150ep diagnosis):
  1. ABSOLUTE coordinates again (the coarse-relative/fine-absolute hybrid broke
     fine digits: acc 0.042, bin err 33). Keeps the two validated v2 fixes:
     soft-target CE on coordinate digits and self-predicted edge-count
     conditioning (soft hint; STOP still learned).
  2. No hand-crafted shape heuristics anywhere: sampling masks are grammar-only
     (slot types, FIX-only CIRCLE_C centers) exactly as the frozen theory's
     constructive guarantees. Candidate selection, when used, is a LEARNED
     ranker trained on self-generated rollouts (separate module).
  3. Batched training + mixed precision: batch=1 left the GPU idle; padded
     batching is safe because trailing PADs are never attended by real tokens
     (causal attention) and their loss is masked. Condition rows have constant
     length for a fixed fastening-point count (asserted).

Usage:
  python -m cae_mesh_generator.wtok.curve3 --dataset ../runs/wtok_synth \
      --val-list .../val_names_100.json --output-dir ../runs/wtok_curve3 \
      --epochs 150 --batch-size 16 --device cuda
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
from .constants import HI_BITS, safe_save
from .curve2 import COUNT_LOSS_W, N_BUCKETS, edge_bucket
from .dataset_curve import (ADV, BOS, COORD0, NEW, PAD, PTR, STOP, TAU_BASE,
                            TAU_TOK, TAUS, VOCAB_C, VTYPE_ID, CurveARDataset,
                            digits_of, load_curve_parts)
from .model_ar import CausalBlock
from .train_curve import part_cond, realized_q, to_device
from .train_ar import chamfer_mm

D_COORD = 128


# ---------------------------------------------------------------- data

class CurveARDataset3(CurveARDataset):
    """v1 absolute-coordinate stream + the edge-count bucket."""

    def __getitem__(self, i: int) -> dict:
        item = super().__getitem__(i)
        item["bucket"] = torch.tensor(edge_bucket(len(self.parts[i].edges)),
                                      dtype=torch.long)
        return item


def collate3(batch: list[dict]) -> dict:
    B = len(batch)
    S = max(len(b["in_tok"]) for b in batch)
    M = max(len(b["mat_type"]) for b in batch)
    C = batch[0]["cond"].shape[0]
    assert all(b["cond"].shape[0] == C for b in batch), \
        "condition length must be constant within a batch (fixed FIX count)"
    out = {
        "in_tok": torch.full((B, S), PAD, dtype=torch.long),
        "in_ptr": torch.full((B, S), -1, dtype=torch.long),
        "target": torch.zeros(B, S, dtype=torch.long),
        "loss_mask": torch.zeros(B, S, dtype=torch.bool),
        "slot_next": torch.zeros(B, S, dtype=torch.long),
        "mat_type": torch.zeros(B, M, dtype=torch.long),
        "mat_digits": torch.zeros(B, M, 6, dtype=torch.long),
        "mat_pos": torch.full((B, M), 10**9, dtype=torch.long),
        "mat_valid": torch.zeros(B, M, dtype=torch.bool),
        "cond": torch.stack([b["cond"] for b in batch]),
        "bucket": torch.stack([b["bucket"] for b in batch]),
    }
    for i, b in enumerate(batch):
        s, m = len(b["in_tok"]), len(b["mat_type"])
        out["in_tok"][i, :s] = b["in_tok"]
        out["in_ptr"][i, :s] = b["in_ptr"]
        out["target"][i, :s] = b["target"]
        out["loss_mask"][i, :s] = b["loss_mask"]
        out["slot_next"][i, :s] = b["slot_next"]
        out["mat_type"][i, :m] = b["mat_type"]
        if m:
            out["mat_digits"][i, :m] = b["mat_digits"]
            out["mat_pos"][i, :m] = b["mat_pos"]
            out["mat_valid"][i, :m] = True
    return out


# ---------------------------------------------------------------- model

class CurveAR3(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 8, layers: int = 8,
                 cond_dim: int = 8, max_len: int = 15000, dropout: float = 0.0):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_C, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.cond_in = nn.Linear(cond_dim, dim)
        self.cond_seg = nn.Parameter(torch.zeros(dim))
        self.bucket_emb = nn.Embedding(N_BUCKETS, dim)
        self.count_head = nn.Linear(dim, N_BUCKETS)
        self.vtype_emb = nn.Embedding(3, dim)
        self.digit_emb = nn.Embedding(D_COORD, dim)
        self.digit_axis = nn.Parameter(torch.randn(6, dim) * 0.02)
        self.vertex_mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(),
                                        nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(CausalBlock(dim, heads, dropout)
                                    for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, VOCAB_C)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.dim = dim

    def vertex_embed(self, mat_type, mat_digits):
        d = (self.digit_emb(mat_digits) + self.digit_axis[None, None]).sum(dim=2)
        return self.vertex_mlp(self.vtype_emb(mat_type) + d)

    def cond_rows(self, cond, bucket):
        rows = self.cond_in(cond) + self.cond_seg           # (B,C,D)
        return torch.cat([rows, self.bucket_emb(bucket)[:, None]], dim=1)

    @staticmethod
    def _wrap_single(item):
        """Adapt an unbatched dataset item (1-D tensors) to a batch of one, so
        evaluate_curve2's per-item calls work unchanged against the batched model."""
        M = len(item["mat_type"])
        b = {k: v[None] if torch.is_tensor(v) and v.dim() >= 1 else v
             for k, v in item.items() if torch.is_tensor(v)}
        b["bucket"] = item.get("bucket", torch.tensor(0)).reshape(1)
        b["mat_valid"] = torch.ones(1, max(M, 1), dtype=torch.bool,
                                    device=item["in_tok"].device)
        if M == 0:
            b["mat_type"] = torch.zeros(1, 1, dtype=torch.long)
            b["mat_digits"] = torch.zeros(1, 1, 6, dtype=torch.long)
            b["mat_pos"] = torch.full((1, 1), 10**9, dtype=torch.long)
            b["mat_valid"][:] = False
        return b

    def forward(self, batch):
        """Returns (static (B,S,V), ptr (B,S,M)); logits[i] predict target[i].
        Accepts an unbatched item too (returns squeezed (S,V),(S,M))."""
        if batch["in_tok"].dim() == 1:
            st, pt = self.forward(self._wrap_single(batch))
            return st[0], pt[0]
        it, ip = batch["in_tok"], batch["in_ptr"]
        B, S = it.shape
        vemb = self.vertex_embed(batch["mat_type"], batch["mat_digits"])  # (B,M,D)
        x = self.tok(it)
        has = ip >= 0
        gathered = torch.gather(vemb, 1,
                                ip.clamp(min=0)[:, :, None].expand(-1, -1, self.dim))
        x = x + gathered * has[:, :, None]
        x = x + self.pos(torch.arange(S, device=it.device))[None]
        cond = self.cond_rows(batch["cond"], batch["bucket"])
        Cn = cond.shape[1]
        h = torch.cat([cond, x], dim=1)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h[:, Cn:])
        static = self.head(h)
        k = self.k_proj(vemb)
        ptr = torch.einsum("bsd,bmd->bsm", self.q_proj(h), k) / (self.dim ** 0.5)
        return static, ptr

    def _masked_ptr(self, ptr, batch):
        B, S, M = ptr.shape
        pos = torch.arange(S, device=ptr.device)
        avail = batch["mat_pos"][:, None, :] <= pos[None, :, None]
        st = batch["slot_next"]
        want = torch.full_like(st, -1)
        want[st == 1], want[st == 2], want[st == 3] = 1, 2, 0
        tmatch = (batch["mat_type"][:, None, :] == want[:, :, None]) & \
                 (want[:, :, None] >= 0)
        ok = avail & tmatch & batch["mat_valid"][:, None, :]
        return ptr.masked_fill(~ok, torch.finfo(ptr.dtype).min)

    def loss(self, batch) -> torch.Tensor:
        if batch["in_tok"].dim() == 1:
            return self.loss(self._wrap_single(batch))
        static, ptr = self.forward(batch)
        ptr = self._masked_ptr(ptr, batch)
        logits = torch.cat([static, ptr], dim=2)            # (B,S,V+M)
        B, S, V = logits.shape
        logp = F.log_softmax(logits.float(), dim=2).reshape(B * S, V)
        tgt = batch["target"].reshape(B * S)
        nll = -logp.gather(1, tgt[:, None]).squeeze(1)
        # soft-target CE on coordinate digits (validated v2 fix)
        coord = (tgt >= COORD0) & (tgt < COORD0 + D_COORD)
        if coord.any():
            kernel = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1], device=logits.device)
            acc = torch.zeros_like(nll)
            wsum = torch.zeros_like(nll)
            for k_i, d in enumerate(range(-2, 3)):
                idx = (tgt + d).clamp(min=0, max=V - 1)
                valid = coord & (tgt + d >= COORD0) & (tgt + d < COORD0 + D_COORD)
                w = kernel[k_i] * valid.float()
                acc = acc - w * logp.gather(1, idx[:, None]).squeeze(1)
                wsum = wsum + w
            nll = torch.where(coord, acc / wsum.clamp(min=1e-9), nll)
        m = batch["loss_mask"].reshape(B * S).float()
        main = (nll * m).sum() / m.sum().clamp(min=1)
        pooled = self.cond_rows(batch["cond"], batch["bucket"])[:, :-1].mean(dim=1)
        count = F.cross_entropy(self.count_head(pooled), batch["bucket"])
        return main + COUNT_LOSS_W * count

    def predict_bucket(self, cond) -> int:
        rows = self.cond_in(cond) + self.cond_seg
        return int(self.count_head(rows.mean(dim=0)).argmax())


# ---------------------------------------------------------------- sampler

class CurveSampler3:
    """Ancestral sampling. Masks are grammar-only (theory's constructive
    guarantees): slot types, FIX-only CIRCLE_C centers, intra-edge dedup.
    No shape heuristics -- selection quality belongs to the learned ranker."""

    def __init__(self, model: CurveAR3, device: str):
        self.model = model
        self.device = device

    def _mini_batch(self, in_tok, in_ptr, mat_type, mat_digits, cond, bucket):
        dev = self.device
        M = max(len(mat_type), 1)
        b = {
            "in_tok": torch.tensor([in_tok], dtype=torch.long, device=dev),
            "in_ptr": torch.tensor([in_ptr], dtype=torch.long, device=dev),
            "mat_type": torch.zeros(1, M, dtype=torch.long, device=dev),
            "mat_digits": torch.zeros(1, M, 6, dtype=torch.long, device=dev),
            "mat_valid": torch.zeros(1, M, dtype=torch.bool, device=dev),
            "cond": cond[None], "bucket": bucket[None],
        }
        if mat_type:
            b["mat_type"][0, :len(mat_type)] = torch.tensor(mat_type, device=dev)
            b["mat_digits"][0, :len(mat_type)] = torch.tensor(mat_digits, device=dev)
            b["mat_valid"][0, :len(mat_type)] = True
        return b

    @torch.no_grad()
    def run(self, cond, fix_vertices, observed_edges=None, max_edges: int = 400,
            temperature: float = 1.0, seed: int = 0, bucket: int | None = None):
        from .codec_curve import SLOT_TYPES
        gen = torch.Generator(self.device).manual_seed(seed)
        dev = self.device
        obs = observed_edges or []
        obs_i = 0
        if bucket is None:
            bucket = self.model.predict_bucket(cond)
        bucket_t = torch.tensor(bucket, dtype=torch.long, device=dev)
        vertices = [dict(v) for v in fix_vertices]
        mat_type = [VTYPE_ID[v["T"]] for v in vertices]
        mat_digits = [digits_of(v["bin"]) for v in vertices]
        bin_index = {(v["T"], tuple(v["bin"])): i for i, v in enumerate(vertices)}
        in_tok, in_ptr = [BOS], [-1]
        edges_out = []

        def next_logits():
            b = self._mini_batch(in_tok, in_ptr, mat_type, mat_digits, cond, bucket_t)
            static, ptr = self.model.forward(b)
            n_mat = len(mat_type)
            return (static[0, -1] / max(temperature, 1e-6),
                    ptr[0, -1, :n_mat] / max(temperature, 1e-6))

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
            vertices.append({"T": vt, "bin": b, "nf": None})
            mat_type.append(VTYPE_ID[vt])
            mat_digits.append(digits_of(b))
            bin_index.setdefault((vt, b), len(vertices) - 1)
            return len(vertices) - 1

        while len(edges_out) < max_edges:
            static_ok = torch.zeros(VOCAB_C, dtype=torch.bool, device=dev)
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
                        for c in digits_of(tuple(vb)):
                            push(COORD0 + c)
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
                static_ok = torch.zeros(VOCAB_C, dtype=torch.bool, device=dev)
                if slot_t != "FIX":
                    static_ok[NEW] = True
                if cand is None or not cand.any():
                    cand = None
                    if slot_t == "FIX":
                        ok_edge = False
                        break
                sl, pl = next_logits()
                choice2 = sample_from(sl, pl, static_ok, cand)
                if choice2 >= VOCAB_C:
                    push(PTR, choice2 - VOCAB_C)
                    refs.append(choice2 - VOCAB_C)
                else:
                    push(NEW)
                    digits = []
                    coord_ok = torch.zeros(VOCAB_C, dtype=torch.bool, device=dev)
                    coord_ok[COORD0: COORD0 + D_COORD] = True
                    for _ in range(6):
                        sl, pl = next_logits()
                        c = sample_from(sl, pl, coord_ok, None) - COORD0
                        push(COORD0 + c)
                        digits.append(c)
                    b = tuple((digits[2 * a] << HI_BITS) | digits[2 * a + 1]
                              for a in range(3))
                    refs.append(materialize(slot_t, b))
            if ok_edge:
                edges_out.append({"tau": tau, "refs": refs, "cls": "generated"})
        return vertices, edges_out


# ---------------------------------------------------------------- training

@torch.no_grad()
def sample_eval3(model, parts, device, obs_rate, max_parts=8, seed=0):
    sampler = CurveSampler3(model, device)
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
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--jitter-bins", type=int, default=1)
    ap.add_argument("--stage2-after", type=int, default=30)
    ap.add_argument("--val-list", required=True)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--sample-every", type=int, default=25)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", default="")
    ap.add_argument("--resume-dir", default="")
    ap.add_argument("--max-hours", type=float, default=0.0)
    args = ap.parse_args()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    use_amp = args.device == "cuda" and not args.no_amp

    parts = load_curve_parts(pathlib.Path(args.dataset))
    val_names = set(json.loads(pathlib.Path(args.val_list).read_text(encoding="utf-8")))
    train_parts = [p for p in parts if p.name not in val_names]
    val_parts = [p for p in parts if p.name in val_names]
    print(f"parts: train {len(train_parts)} val {len(val_parts)} "
          f"| batch {args.batch_size} amp {use_amp}")
    (out / "split.json").write_text(json.dumps(
        {"val": [p.name for p in val_parts], "args": vars(args)}), encoding="utf-8")

    train_ds = CurveARDataset3(train_parts, augment=True, jitter_bins=args.jitter_bins,
                               stage2_after=args.stage2_after)
    val_gen = CurveARDataset3(val_parts, augment=False, obs_rate=0.0, base_seed=555)
    val_half = CurveARDataset3(val_parts, augment=False, obs_rate=0.5, base_seed=555)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate3, drop_last=False)
    vg_loader = DataLoader(val_gen, batch_size=args.batch_size, collate_fn=collate3)
    vh_loader = DataLoader(val_half, batch_size=args.batch_size, collate_fn=collate3)

    model = CurveAR3(args.dim, 8, args.layers, dropout=args.dropout).to(args.device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best = [], float("inf")
    start_epoch = 1
    if args.resume_dir and not args.resume:
        rd = pathlib.Path(args.resume_dir)
        if (rd / "last.pt").exists():
            args.resume = str(rd / "last.pt")
            if (rd / "history.json").exists() and not (out / "history.json").exists():
                (out / "history.json").write_text(
                    (rd / "history.json").read_text(encoding="utf-8"), encoding="utf-8")
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

    def eval_nll(loader):
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for b in loader:
                b = to_device(b, args.device)
                bs = b["in_tok"].shape[0]
                tot += float(model.loss(b)) * bs
                n += bs
        return tot / max(n, 1)

    t_start = time.time()
    budget = args.max_hours * 3600.0
    for epoch in range(start_epoch, args.epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for b in train_loader:
            b = to_device(b, args.device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model.loss(b)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach())
            n += 1
        row = {"epoch": epoch, "train_nll": tot / max(n, 1),
               "seconds": round(time.time() - t0, 1)}
        if epoch % args.val_every == 0 or epoch == args.epochs:
            row["val_nll_gen"] = eval_nll(vg_loader)
            row["val_nll_half"] = eval_nll(vh_loader)
            ck = {"model": model.state_dict(), "opt": opt.state_dict(),
                  "args": vars(args), "epoch": epoch}
            if row["val_nll_gen"] < best:
                best = row["val_nll_gen"]
                safe_save(ck, out / "best.pt")
            safe_save(ck, out / "last.pt")
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            model.eval()
            row["gen"] = sample_eval3(model, val_parts, args.device, 0.0)
            row["infill50"] = sample_eval3(model, val_parts, args.device, 0.5)
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
        if budget and elapsed + row["seconds"] * 1.5 > budget:
            safe_save({"model": model.state_dict(), "opt": opt.state_dict(),
                       "args": vars(args), "epoch": epoch}, out / "last.pt")
            print(f"[budget] stopping at epoch {epoch} after {elapsed/3600:.2f}h",
                  flush=True)
            break


if __name__ == "__main__":
    main()

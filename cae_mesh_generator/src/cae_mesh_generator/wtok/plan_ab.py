"""Plan A+B hybrid: flow-matched latent plan + plan-conditioned AR detailer.

  Stage "ae"   : PlanEncoder (GT wireframe -> 8 latent plan tokens, noised)
                 + PlanCurveAR (curve3 decoder with plan rows in the prefix),
                 trained jointly with teacher forcing.
  Stage "prior": freeze the encoder, embed every training part, fit a FiLM-MLP
                 flow matching model p(z | condition) (twopoint-flow twin).
  Stage "eval" : z-on (GT plan) vs z-off (prior-sampled plan) generation with
                 the session's standard error decomposition.

Everything is learned; sampling masks stay grammar-only (curve3 policy).

Usage (local, small):
  python -m cae_mesh_generator.wtok.plan_ab --stage ae \
      --dataset ../runs/wtok_synth --val-list .../val_names_100.json \
      --output-dir ../runs/plan_ab_v1 --epochs 100 --device cuda
  python -m cae_mesh_generator.wtok.plan_ab --stage prior --output-dir ../runs/plan_ab_v1 ...
  python -m cae_mesh_generator.wtok.plan_ab --stage eval  --output-dir ../runs/plan_ab_v1 ...
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
from torch.utils.data import DataLoader

from .codec import realize_points
from .constants import safe_save
from .curve3 import (CurveAR3, CurveARDataset3, CurveSampler3, collate3,
                     sample_eval3)
from .dataset_curve import TAUS, load_curve_parts
from .train_curve import part_cond, realized_q, to_device
from .train_ar import chamfer_mm

N_PLAN = 8
N_BINS_F = float(1 << 14)
EDGE_FEAT = 4 + 9  # tau one-hot + up to 3 vertices x 3 normalized coords


# ---------------------------------------------------------------- data

def edge_features(part) -> np.ndarray:
    out = np.zeros((len(part.edges), EDGE_FEAT), dtype=np.float32)
    for i, e in enumerate(part.edges):
        out[i, TAUS.index(e["tau"])] = 1.0
        for k, r in enumerate(e["refs"][:3]):
            out[i, 4 + 3 * k: 7 + 3 * k] = np.asarray(
                part.vertices[r]["bin"], dtype=np.float32) / N_BINS_F
    return out


class PlanDataset(CurveARDataset3):
    def __getitem__(self, i: int) -> dict:
        item = super().__getitem__(i)
        item["edge_feats"] = torch.from_numpy(edge_features(self.parts[i]))
        return item


def collate_plan(batch: list[dict]) -> dict:
    feats = [b.pop("edge_feats") for b in batch]
    out = collate3(batch)
    E = max(len(f) for f in feats)
    ef = torch.zeros(len(batch), E, EDGE_FEAT)
    em = torch.zeros(len(batch), E, dtype=torch.bool)
    for i, f in enumerate(feats):
        ef[i, : len(f)] = f
        em[i, : len(f)] = True
    out["edge_feats"] = ef
    out["edge_mask"] = em
    return out


# ---------------------------------------------------------------- model

def heads_for(dim: int) -> int:
    for h in (8, 6, 4, 2):
        if dim % h == 0:
            return h
    return 1


class PlanEncoder(nn.Module):
    def __init__(self, dim: int = 192, heads: int | None = None, layers: int = 3):
        heads = heads or heads_for(dim)
        super().__init__()
        self.embed = nn.Linear(EDGE_FEAT, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.queries = nn.Parameter(torch.randn(N_PLAN, dim) * 0.02)
        self.pool = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, edge_feats, edge_mask):
        h = self.encoder(self.embed(edge_feats), src_key_padding_mask=~edge_mask)
        q = self.queries[None].expand(h.shape[0], -1, -1)
        z, _ = self.pool(q, h, h, key_padding_mask=~edge_mask, need_weights=False)
        return self.norm(z)                      # (B, N_PLAN, dim)


class PlanCurveAR(CurveAR3):
    """curve3 decoder whose condition prefix additionally carries plan rows."""

    def __init__(self, dim=192, heads=None, layers=6, dropout=0.0):
        super().__init__(dim, heads or heads_for(dim), layers, dropout=dropout)
        self.plan_proj = nn.Linear(dim, dim)
        self.plan_seg = nn.Parameter(torch.zeros(dim))

    def cond_rows(self, cond, bucket, plan=None):
        rows = super().cond_rows(cond, bucket)
        if plan is not None:
            rows = torch.cat([rows, self.plan_proj(plan) + self.plan_seg], dim=1)
        return rows

    def forward(self, batch):
        if batch["in_tok"].dim() == 1:
            st, pt = self.forward(self._wrap_single(batch))
            return st[0], pt[0]
        it, ip = batch["in_tok"], batch["in_ptr"]
        B, S = it.shape
        vemb = self.vertex_embed(batch["mat_type"], batch["mat_digits"])
        x = self.tok(it)
        has = ip >= 0
        gathered = torch.gather(vemb, 1,
                                ip.clamp(min=0)[:, :, None].expand(-1, -1, self.dim))
        x = x + gathered * has[:, :, None]
        x = x + self.pos(torch.arange(S, device=it.device))[None]
        cond = self.cond_rows(batch["cond"], batch["bucket"], batch.get("plan"))
        h = torch.cat([cond, x], dim=1)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h[:, cond.shape[1]:])
        static = self.head(h)
        k = self.k_proj(vemb)
        ptr = torch.einsum("bsd,bmd->bsm", self.q_proj(h), k) / (self.dim ** 0.5)
        return static, ptr


class JointAE(nn.Module):
    def __init__(self, dim=192, enc_layers=3, dec_layers=6, dropout=0.1,
                 plan_noise: float = 0.1):
        super().__init__()
        self.encoder = PlanEncoder(dim, None, enc_layers)
        self.decoder = PlanCurveAR(dim, None, dec_layers, dropout=dropout)
        self.plan_noise = plan_noise

    def loss(self, batch) -> torch.Tensor:
        z = self.encoder(batch["edge_feats"], batch["edge_mask"])
        if self.training and self.plan_noise > 0:
            z = z + torch.randn_like(z) * self.plan_noise
        batch = dict(batch, plan=z)
        return self.decoder.loss(batch)


class PlanFlow(nn.Module):
    """v(z_t, t | cond_vec): FiLM MLP over the flattened plan (twopoint twin)."""

    def __init__(self, dim=192, cond_dim=24, hidden=512, blocks=4):
        super().__init__()
        self.zdim = N_PLAN * dim
        self.cond_net = nn.Sequential(nn.Linear(cond_dim + 1, hidden), nn.GELU(),
                                      nn.Linear(hidden, hidden))
        self.inp = nn.Linear(self.zdim, hidden)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                          nn.Linear(hidden, hidden)) for _ in range(blocks))
        self.films = nn.ModuleList(nn.Linear(hidden, hidden * 2) for _ in range(blocks))
        self.out = nn.Linear(hidden, self.zdim)

    def forward(self, z_t, t, cond_vec):
        c = self.cond_net(torch.cat([cond_vec, t[:, None]], dim=1))
        h = self.inp(z_t)
        for blk, film in zip(self.blocks, self.films):
            scale, shift = film(c).chunk(2, dim=1)
            h = h + blk(h * (1 + scale) + shift)
        return self.out(h)

    @torch.no_grad()
    def sample(self, cond_vec, steps=40, generator=None):
        B = cond_vec.shape[0]
        z = torch.randn(B, self.zdim, device=cond_vec.device, generator=generator)
        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((B,), k * dt, device=cond_vec.device)
            z = z + self.forward(z, t, cond_vec) * dt
        return z


def cond_vector(cond: torch.Tensor) -> torch.Tensor:
    """(C,8) condition rows -> flat 24-dim (2 FIX rows + 1 global row)."""
    flat = cond.reshape(-1)
    out = torch.zeros(24, device=cond.device)
    out[: min(24, len(flat))] = flat[:24]
    return out


class PlanSampler(CurveSampler3):
    def __init__(self, model: PlanCurveAR, device: str, plan: torch.Tensor):
        super().__init__(model, device)
        self.plan = plan                        # (N_PLAN, dim)

    def _mini_batch(self, *a, **k):
        b = super()._mini_batch(*a, **k)
        b["plan"] = self.plan[None]
        return b


# ---------------------------------------------------------------- eval

@torch.no_grad()
def gen_eval(decoder, parts, device, plan_of, max_parts=8, seed=0, tag=""):
    cds, n_edges = [], []
    for p in parts[:max_parts]:
        cond, fix = part_cond(p, device)
        plan = plan_of(p, cond)
        sampler = PlanSampler(decoder, device, plan)
        v, e = sampler.run(cond, fix, [], seed=seed)
        gt = realize_points(realized_q(p, p.vertices, p.edges))
        gen = realize_points(realized_q(p, v, e))
        cds.append(chamfer_mm(gen[::3], gt[::3]) if len(gen) else float("nan"))
        n_edges.append(len(e))
    finite = [c for c in cds if np.isfinite(c)]
    # the median is the number that compares with evaluate_curve2 (curve3 = 52.2mm);
    # the mean is kept because the training loop has always printed it.
    return {f"{tag}chamfer_mm": float(np.mean(finite)) if finite else float("nan"),
            f"{tag}chamfer_median_mm": float(np.median(finite)) if finite else float("nan"),
            f"{tag}n_edges": float(np.mean(n_edges))}


# ---------------------------------------------------------------- stages

def build_splits(args):
    parts = load_curve_parts(pathlib.Path(args.dataset))
    val_names = set(json.loads(pathlib.Path(args.val_list).read_text(encoding="utf-8")))
    return ([p for p in parts if p.name not in val_names],
            [p for p in parts if p.name in val_names])


def stage_ae(args, out):
    train_parts, val_parts = build_splits(args)
    print(f"parts: train {len(train_parts)} val {len(val_parts)}")
    train_ds = PlanDataset(train_parts, augment=True, jitter_bins=1,
                           stage2_after=args.stage2_after)
    val_ds = PlanDataset(val_parts, augment=False, obs_rate=0.0, base_seed=555)
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=collate_plan)
    vl = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_plan)
    model = JointAE(args.dim, args.enc_layers, args.dec_layers, args.dropout).to(args.device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    history, best = [], float("inf")
    start_epoch = 1
    if args.resume:
        ck = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        hf = out / "history.json"
        if hf.exists():
            history = [r for r in json.loads(hf.read_text(encoding="utf-8"))
                       if r["epoch"] < start_epoch]
            best = min((r["val_nll"] for r in history if "val_nll" in r),
                       default=float("inf"))
        print(f"resumed at {start_epoch}")
    for epoch in range(start_epoch, args.epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for b in tl:
            b = to_device(b, args.device)
            loss = model.loss(b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            n += 1
        row = {"epoch": epoch, "train_nll": tot / max(n, 1),
               "seconds": round(time.time() - t0, 1)}
        if epoch % args.val_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                vs = [float(model.loss(to_device(b, args.device))) for b in vl]
            row["val_nll"] = float(np.mean(vs))
            ck = {"model": model.state_dict(), "opt": opt.state_dict(),
                  "args": vars(args), "epoch": epoch}
            if row["val_nll"] < best:
                best = row["val_nll"]
                safe_save(ck, out / "ae_best.pt")
            safe_save(ck, out / "ae_last.pt")
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            model.eval()
            enc, dec = model.encoder, model.decoder
            def plan_gt(p, cond):
                ef = torch.from_numpy(edge_features(p))[None].to(args.device)
                em = torch.ones(1, ef.shape[1], dtype=torch.bool, device=args.device)
                return enc(ef, em)[0]
            row["zon"] = gen_eval(dec, val_parts, args.device, plan_gt, tag="zon_")
        history.append(row)
        (out / "history.json").write_text(json.dumps(history), encoding="utf-8")
        msg = f"epoch {epoch}: nll {row['train_nll']:.4f} ({row['seconds']}s)"
        if "val_nll" in row:
            msg += f"  val {row['val_nll']:.4f}"
        if "zon" in row:
            msg += (f"  | z-on gen {row['zon']['zon_chamfer_mm']:.1f}mm "
                    f"(E={row['zon']['zon_n_edges']:.0f})")
        print(msg, flush=True)


def stage_prior(args, out):
    train_parts, val_parts = build_splits(args)
    ck = torch.load(out / "ae_best.pt", map_location=args.device, weights_only=False)
    a = ck["args"]
    model = JointAE(a["dim"], a["enc_layers"], a["dec_layers"], 0.0).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    Z, C = [], []
    with torch.no_grad():
        for p in train_parts:
            cond, _ = part_cond(p, args.device)
            ef = torch.from_numpy(edge_features(p))[None].to(args.device)
            em = torch.ones(1, ef.shape[1], dtype=torch.bool, device=args.device)
            Z.append(model.encoder(ef, em)[0].reshape(-1))
            C.append(cond_vector(cond))
    Z = torch.stack(Z)
    C = torch.stack(C)
    mu, sd = C.mean(0), C.std(0) + 1e-6
    Cz = (C - mu) / sd
    # standardize latents too: the flow trains in whitened space (the unwhitened
    # 1536-dim latent blew the FiLM blocks up to 1e11 loss on the first run)
    z_mu, z_sd = Z.mean(0), Z.std(0) + 1e-6
    Z = (Z - z_mu) / z_sd
    flow = PlanFlow(a["dim"]).to(args.device)
    print(f"prior params: {sum(p.numel() for p in flow.parameters())/1e6:.2f}M "
          f"| latents {tuple(Z.shape)}")
    opt = torch.optim.AdamW(flow.parameters(), lr=3e-4, weight_decay=1e-4)
    for epoch in range(args.prior_epochs):
        perm = torch.randperm(len(Z), device=args.device)
        for i in range(0, len(Z), 256):
            idx = perm[i:i + 256]
            z1 = Z[idx]
            z0 = torch.randn_like(z1)
            t = torch.rand(len(idx), device=args.device)
            zt = (1 - t[:, None]) * z0 + t[:, None] * z1
            cj = Cz[idx] + torch.randn_like(Cz[idx]) * 0.05
            v = flow(zt, t, cj)
            loss = F.mse_loss(v, z1 - z0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            opt.step()
        if (epoch + 1) % 200 == 0:
            print(f"prior epoch {epoch+1}: fm {float(loss):.4f}", flush=True)
    safe_save({"flow": flow.state_dict(), "cond_mu": mu.cpu(), "cond_sd": sd.cpu(),
               "z_mu": z_mu.cpu(), "z_sd": z_sd.cpu(), "args": a}, out / "prior.pt")
    print("prior saved")


def stage_eval(args, out):
    train_parts, val_parts = build_splits(args)
    ck = torch.load(out / "ae_best.pt", map_location=args.device, weights_only=False)
    a = ck["args"]
    model = JointAE(a["dim"], a["enc_layers"], a["dec_layers"], 0.0).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    pk = torch.load(out / "prior.pt", map_location=args.device, weights_only=False)
    flow = PlanFlow(a["dim"]).to(args.device)
    flow.load_state_dict(pk["flow"])
    flow.eval()
    mu, sd = pk["cond_mu"].to(args.device), pk["cond_sd"].to(args.device)
    z_mu, z_sd = pk["z_mu"].to(args.device), pk["z_sd"].to(args.device)

    def plan_gt(p, cond):
        ef = torch.from_numpy(edge_features(p))[None].to(args.device)
        em = torch.ones(1, ef.shape[1], dtype=torch.bool, device=args.device)
        return model.encoder(ef, em)[0]

    def plan_prior(p, cond):
        cz = ((cond_vector(cond) - mu) / sd)[None]
        zn = flow.sample(cz, steps=40)[0]
        return (zn * z_sd + z_mu).reshape(N_PLAN, a["dim"])

    r_on = gen_eval(model.decoder, val_parts, args.device, plan_gt,
                    max_parts=args.eval_parts, tag="zon_")
    r_off = gen_eval(model.decoder, val_parts, args.device, plan_prior,
                     max_parts=args.eval_parts, tag="zoff_")
    result = {**r_on, **r_off, "epoch": ck["epoch"]}
    print(json.dumps(result, indent=1))
    (out / "eval.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("ae", "prior", "eval"), required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--val-list", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--prior-epochs", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--enc-layers", type=int, default=3)
    ap.add_argument("--dec-layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--stage2-after", type=int, default=30)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--sample-every", type=int, default=25)
    ap.add_argument("--eval-parts", type=int, default=20)
    ap.add_argument("--resume", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    {"ae": stage_ae, "prior": stage_prior, "eval": stage_eval}[args.stage](args, out)


if __name__ == "__main__":
    main()

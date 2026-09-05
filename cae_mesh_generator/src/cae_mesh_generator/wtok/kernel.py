"""Feature detection by a local kernel: one shared operator over neighbourhoods.

Idea from the user. For each point, look only at a small neighbourhood and ask
what that neighbourhood says about the point: if the points around it form a
sheet and it sits at the edge of that sheet, it is on the outline; if the
orientation flips across it, it is on a bend.

Why this beats the global-attention version:

  it learns a rule, not a layout
      Attention over all 512 points can memorise "on THIS part the rim runs
      here", which is exactly what will not survive more complex parts. A
      neighbourhood operator has no way to express that.

  weight sharing multiplies the data
      2,200 parts is 2,200 examples of a part, but 2,200 x 512 = 1.1M examples
      of a neighbourhood. This is the mechanism a convolution gives an image
      model and the thing this project has been missing.

  the generator's error largely vanishes under normalisation
      Each patch is centred, divided by its own spacing and aligned to its
      normal. The generator's error is 2.81mm of rigid offset plus a warp still
      correlated 0.755 across 5mm -- locally almost a rigid motion, so it washes
      out. Measured on normalised patches, GT and generated clouds agree:
          out-of-plane/spacing  0.188 vs 0.160
          one-sidedness         0.178 vs 0.158
      which is why this trains without the degradation the global version needed.

  density stops mattering
      The hand-written detector used a fixed k and a fixed threshold, so its
      false-rim rate went 5.7% -> 22.8% as the cloud grew from 256 to 4096
      points. Here the neighbourhood itself is the input.
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
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

from .codec import realize_points
from .connect import densify, trace
from .constants import safe_save, stable_seed
from .dataset_curve import load_curve_parts
from .evaluate_curve2 import one_way
from .meshgen import fps_order
from .ridge import CLASSES, ON_CURVE, curve_points
from .train_curve import realized_q

K_NN = 32          # neighbours per patch
# Both thresholds are in units of the point's OWN neighbour spacing, so they
# mean the same thing on a small part and a large one, at 256 points and at
# 512. The previous absolute versions (BAND/ON_CURVE times a nominal 40mm part)
# made the on-curve label 0.30x the spacing at 512 points: far finer than the
# sampling can resolve, so only 5% of points were positive while tracing a
# curve needs the ~15% that form its nearest row. Training drove the kept
# fraction down towards that 5% and coverage degraded as it did (19%/3.11mm at
# epoch 15 -> 14%/8.30mm at epoch 30) while the loss kept falling.
# `spacing` is the mean distance to the K neighbours, a steady 2.80x the point
# pitch at both 256 and 512 points, so 0.5 spacings = 1.4 pitches = the row of
# points nearest the curve: 14.8% of points for the outline at 512 and 23.2% at
# 256, the 1/sqrt(N) fall that one row must show. The band keeps the old 4:1
# ratio to the on-curve threshold.
BAND_SP = 2.0
ON_CURVE_SP = 0.5


# per-neighbour class channels: [confirmed, is_outline, is_bend]
CONFIRM_CH = 1 + len(CLASSES)


def degrade(xyz, nrm, rng, kinds=("holes", "density", "noise", "fringe")):
    """Make a clean cloud look like one the generator produced.

    Labels are taken from the point's CLEAN position, so the task stays "recover
    the original shape's frame" -- the returned index says where each surviving
    point came from. Only reproducible defects are simulated: the rigid offset
    and smooth warp the generator also has are invisible to a local operator and
    training against them would be asking for the impossible.

    Measured on val (kernel512_sp ep25, AUC outline): clean 0.988, generated
    clouds 0.852, and these four together 0.924 -- about half the gap. The other
    half is the offset and warp, and no amount of this closes it.
    """
    idx = np.arange(len(xyz))
    pitch = float(np.median(cKDTree(xyz).query(xyz, k=2)[0][:, 1])) or 1e-9
    if "holes" in kinds:
        keep = np.ones(len(xyz), bool)
        for _ in range(int(rng.integers(2, 5))):
            c = xyz[rng.integers(len(xyz))]
            keep &= np.linalg.norm(xyz - c, axis=1) > rng.uniform(2.0, 5.0) * pitch
        n_gone = int((~keep).sum())
        if n_gone and keep.sum() >= 16:
            live = np.flatnonzero(keep)
            j = np.concatenate([live, rng.choice(live, n_gone, replace=True)])
            xyz, nrm, idx = xyz[j], nrm[j], idx[j]
    if "density" in kinds:
        d = np.linalg.norm(xyz - xyz[rng.integers(len(xyz))], axis=1)
        w = 1.0 / (1.0 + (d / (d.mean() + 1e-9)) ** 2)
        j = rng.choice(len(xyz), len(xyz), replace=True, p=w / w.sum())
        xyz, nrm, idx = xyz[j], nrm[j], idx[j]
    if "noise" in kinds:
        v = rng.normal(size=xyz.shape)
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        xyz = xyz + v * (rng.uniform(0.0, 0.8, size=(len(xyz), 1)) * pitch)
    if "fringe" in kinds:
        # the cloud runs 2-5% past where the sheet ends, so the row that IS the
        # boundary stops looking like one. A uniform scale would not do this --
        # the patch is normalised by its own spacing and cannot see it.
        c = xyz.mean(0)
        r = np.linalg.norm(xyz - c, axis=1)
        out = r > np.quantile(r, 0.80)
        u = (xyz - c) / np.maximum(r[:, None], 1e-12)
        xyz = xyz.copy()
        xyz[out] += u[out] * (rng.uniform(1.0, 3.0, size=(int(out.sum()), 1)) * pitch)
    return xyz, nrm, idx


def patches(xyz: np.ndarray, nrm: np.ndarray, k: int = K_NN, confirmed=None,
            return_idx: bool = False):
    """Neighbourhoods in their own frame: centred, scaled by their own spacing,
    with the normal as the third axis. Everything the generator gets wrong at
    part scale is gone by construction.

    `confirmed` is (N, CONFIRM_CH): for a point already committed by an earlier
    round, a 1 followed by its known classes; all zeros for one still in play.
    It is read for the NEIGHBOURS, never for the point being judged -- so a
    point can see that it sits among confirmed outline points without being
    told anything about itself. Without these channels a committed point would
    only ever act as a coordinate and its class would be thrown away."""
    k = min(k, len(xyz) - 1)
    d, idx = cKDTree(xyz).query(xyz, k=k + 1)
    nb = xyz[idx[:, 1:]] - xyz[:, None, :]
    spacing = np.maximum(d[:, 1:].mean(1), 1e-9)
    nb = nb / spacing[:, None, None]

    n = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    ref = np.tile(np.array([1.0, 0.0, 0.0]), (len(xyz), 1))
    bad = np.abs(np.einsum("ij,ij->i", ref, n)) > 0.9
    ref[bad] = np.array([0.0, 1.0, 0.0])
    e1 = ref - n * np.einsum("ij,ij->i", ref, n)[:, None]
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
    e2 = np.cross(n, e1)
    basis = np.stack([e1, e2, n], axis=1)                 # (N,3,3)
    local = np.einsum("nkj,nij->nki", nb, basis)

    nb_n = np.einsum("nkj,nij->nki", n[idx[:, 1:]], basis)  # neighbour normals
    ch = [local, nb_n]
    if confirmed is not None:
        ch.append(confirmed[idx[:, 1:]])
    feat = np.concatenate(ch, axis=-1).astype(np.float32)
    return (feat, spacing, idx[:, 1:]) if return_idx else (feat, spacing)


class PatchDataset(Dataset):
    def __init__(self, parts, mesh_dir, n_pts, augment, base_seed=0, even=True,
                 per_item=0, confirm_max=0.0, degrade_rate=0.0):
        self.parts, self.dir, self.n_pts = parts, pathlib.Path(mesh_dir), n_pts
        self.per_item = per_item        # patches supervised per part per epoch
        # Fraction of points already committed by an earlier round, drawn per
        # item up to this. 0 keeps the old cold-start behaviour, which stays in
        # the mix so the model still works on the first round when nothing is
        # confirmed yet.
        self.confirm_max = confirm_max
        # how often an item arrives looking like generator output rather than
        # like the clean cloud it is scored against
        self.degrade_rate = degrade_rate
        self.augment, self.base_seed, self.epoch, self.even = augment, base_seed, 0, even
        self.cache: dict = {}
        self.patch_cache: dict = {}

    def set_epoch(self, e):
        self.epoch = int(e)

    def __len__(self):
        return len(self.parts)

    def load(self, part):
        if part.name not in self.cache:
            d = np.load(self.dir / f"{part.name}.npz")
            span = np.maximum(d["env_hi"] - d["env_lo"], 1e-9)
            xyz = d["xyz"].astype(np.float64) * span + d["env_lo"]
            tgt = np.zeros((len(xyz), len(CLASSES), 3))
            for ci, cls in enumerate(CLASSES):
                cp = curve_points(part, cls)
                tgt[:, ci] = cKDTree(cp).data[cKDTree(cp).query(xyz)[1]] \
                    if len(cp) else xyz
            order = fps_order(xyz.astype(np.float32),
                              min(len(xyz), 4 * self.n_pts)) if self.even else None
            self.cache[part.name] = (xyz, d["normal"].astype(np.float64), tgt, order)
        return self.cache[part.name]

    def __getitem__(self, i):
        p = self.parts[i]
        rng = np.random.default_rng(
            stable_seed(self.base_seed + 999983 * self.epoch, p.name))
        xyz, nrm, tgt, order = self.load(p)
        n = min(self.n_pts, len(xyz))
        take = (order[:n] if order is not None and len(order) >= n
                else rng.choice(len(xyz), n, replace=False))
        x, nn_, tt = xyz[take], nrm[take], tgt[take]
        # With farthest-point order the selection is deterministic and nothing
        # perturbs the positions, so the patches are the same every epoch.
        # Building them costs a kd-tree query per part and was most of the epoch
        # time; caching is free accuracy-wise.
        # Whether this point is ON a curve is a property of the point, not of
        # where a degradation moved it: a boundary point shoved outwards is
        # still a boundary point. So the label is taken before degrading, and
        # only the displacement target follows the point to its new position.
        on_clean = None
        if self.degrade_rate > 0 and rng.random() < self.degrade_rate:
            xd, nnd, j = degrade(x, nn_, rng)
            sp_clean = np.maximum(cKDTree(x).query(x, k=min(K_NN, len(x) - 1) + 1)[0]
                                  [:, 1:].mean(1), 1e-9)
            on_clean = (np.linalg.norm(tt - x[:, None, :], axis=-1)
                        < ON_CURVE_SP * sp_clean[:, None])[j]
            x, nn_, tt = xd, nnd, tt[j]
            feat, spacing, nb_idx = patches(x, nn_, return_idx=True)
        else:
            cached = self.patch_cache.get(p.name)
            if cached is None:
                cached = patches(x, nn_, return_idx=True)
                self.patch_cache[p.name] = cached
            feat, spacing, nb_idx = cached
        disp = (tt - x[:, None, :]) / spacing[:, None, None]
        if on_clean is None:
            on_clean = np.linalg.norm(disp, axis=-1) < ON_CURVE_SP

        # Which points a previous round would already have committed, and what
        # they would be known to be. Judged points are the rest -- confirmed
        # ones are still in every neighbourhood, they are just not scored.
        supervise = np.ones(len(x), np.float32)
        if self.confirm_max > 0:
            on = on_clean.astype(np.float32)
            f = rng.random() * self.confirm_max if rng.random() > 0.25 else 0.0
            n_conf = int(f * len(x))
            conf = np.zeros((len(x), CONFIRM_CH), np.float32)
            if n_conf:
                sel = rng.permutation(len(x))[:n_conf]
                conf[sel, 0] = 1.0
                conf[sel, 1:] = on[sel]
                supervise[sel] = 0.0
            feat = np.concatenate([feat, conf[nb_idx]], axis=-1)

        # Train on a subset of the patches, but build every patch from the FULL
        # cloud first. The operator is per-point, so which points are supervised
        # this epoch does not change what is learned -- but computing the
        # neighbourhoods among a subset would make them sparser than the ones
        # seen at inference, and that WOULD change it. At inference all points
        # are classified, as before.
        if self.per_item and self.per_item < len(feat):
            # draw from the points actually being judged, not the committed ones
            pool = np.flatnonzero(supervise > 0)
            sel = rng.choice(pool, min(self.per_item, len(pool)), replace=False)
            feat, spacing, disp, x = feat[sel], spacing[sel], disp[sel], x[sel]
            supervise, on_clean = supervise[sel], on_clean[sel]
        return {"patch": torch.from_numpy(feat),
                "disp": torch.from_numpy(disp.astype(np.float32)),
                "on": torch.from_numpy(on_clean.astype(np.float32)),
                "scale": torch.from_numpy(spacing.astype(np.float32)),
                "supervise": torch.from_numpy(supervise),
                "xyz": torch.from_numpy(x.astype(np.float32))}


class PatchNet(nn.Module):
    """One small transformer, applied to every neighbourhood independently."""

    def __init__(self, dim=128, layers=4, heads=4, in_ch=6):
        super().__init__()
        self.in_ch = in_ch
        self.embed = nn.Linear(in_ch, dim)
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, len(CLASSES) * 4)

    def forward(self, patch):
        B, N, K, _ = patch.shape
        h = self.embed(patch).reshape(B * N, K, -1)
        q = self.query.expand(B * N, 1, -1)
        h = self.enc(torch.cat([q, h], dim=1))[:, 0]     # read out the query slot
        out = self.head(self.norm(h)).reshape(B, N, len(CLASSES), 4)
        return out[..., :3], out[..., 3]


def loss_fn(model, batch):
    pred, logit = model(batch["patch"])
    tgt = batch["disp"]
    # tgt is already the displacement divided by the point's own spacing, so a
    # threshold in spacing units is just a comparison -- no part scale needed.
    d = torch.linalg.norm(tgt, dim=-1)
    sup = batch.get("supervise")
    sup = torch.ones_like(d[..., 0]) if sup is None else sup
    sup = sup[..., None]                       # (B, N, 1), broadcast over class
    band = (d < BAND_SP).float() * sup
    # the label rides with the point; the displacement target does not
    on = batch.get("on")
    on = (d < ON_CURVE_SP).float() if on is None else on
    err = ((pred - tgt) ** 2).mean(-1)
    disp_loss = (err * band).sum() / band.sum().clamp(min=1)
    pos = ((on * sup).sum() / sup.expand_as(on).sum().clamp(min=1)).clamp(1e-3, 1 - 1e-3)
    w = torch.where(on > 0.5, 1.0 / pos, 1.0 / (1 - pos)) * sup
    bce = ((F.binary_cross_entropy_with_logits(logit, on, reduction="none")
            * w).sum() / w.sum().clamp(min=1))
    return disp_loss + 0.1 * bce


@torch.no_grad()
def curves_from(model, xyz, nrm, device, thr=0.5, confirmed=None):
    feat, spacing = patches(xyz, nrm, confirmed=confirmed)
    pred, logit = model(torch.from_numpy(feat)[None].to(device))
    pred = pred[0].cpu().numpy() * spacing[:, None, None]
    conf = torch.sigmoid(logit[0]).cpu().numpy()
    out, kept = [], []
    for ci in range(len(CLASSES)):
        keep = conf[:, ci] > thr
        kept.append(float(keep.mean()))
        if keep.sum() < 4:
            out.append(np.zeros((0, 3)))
            continue
        proj = xyz[keep] + pred[keep, ci]
        kk = min(13, len(proj))
        dd, nb = cKDTree(proj).query(proj, k=kk)
        pi = np.repeat(np.arange(len(proj)), kk - 1)
        pj = nb[:, 1:].reshape(-1)
        span = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
        out.append(densify(trace(proj, np.ones(len(proj), int),
                                 (dd[:, 1:].reshape(-1) < span * 0.04).astype(float),
                                 np.stack([pi, pj], 1))))
    return out, kept


@torch.no_grad()
def score(model, parts, md, n_pts, device, max_parts=12):
    from .evaluate_curve2 import class_points
    # scored cold-start (nothing confirmed) so the number stays comparable
    # across runs with and without the confirmed channels
    zero = None
    if getattr(model, "in_ch", 6) > 6:
        zero = "cold"
    ds = PatchDataset(parts[:max_parts], md, n_pts, False, 555)
    acc = {"outline": [], "bend": [], "spur": [], "kept": []}
    for i in range(len(ds)):
        it = ds[i]
        xyz = it["xyz"].numpy().astype(np.float64)
        nrm = it["patch"].numpy()[:, 0, 3:]     # placeholder, recompute below
        d = np.load(pathlib.Path(md) / f"{parts[i].name}.npz")
        span = np.maximum(d["env_hi"] - d["env_lo"], 1e-9)
        full = d["xyz"] * span + d["env_lo"]
        nrm = d["normal"][cKDTree(full).query(xyz)[1]]
        cf = (np.zeros((len(xyz), CONFIRM_CH), np.float32)
              if zero is not None else None)
        (op, bp), kept = curves_from(model, xyz, nrm, device, confirmed=cf)
        p = parts[i]
        gw = realize_points(realized_q(p, p.vertices, p.edges))
        acc["kept"].append(float(np.mean(kept)))
        if len(op):
            acc["outline"].append(one_way(class_points(p, {"outer_boundary"}), op)[0])
        if len(bp):
            acc["bend"].append(one_way(class_points(p, {"bend_line"}), bp)[0])
        w = np.concatenate([q for q in (op, bp) if len(q)]) \
            if len(op) or len(bp) else np.zeros((0, 3))
        if len(w):
            acc["spur"].append(one_way(w, gw)[0])
    med = lambda k: float(np.median(acc[k])) if acc[k] else float("nan")
    return {k: med(k) for k in acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--wtok", required=True)
    ap.add_argument("--val-list", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--points", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--degrade-rate", type=float, default=0.0,
                    help="fraction of items that arrive looking like generator "
                         "output (holes, density, noise, fringe)")
    ap.add_argument("--confirm-max", type=float, default=0.0,
                    help="up to this fraction of points arrive already "
                         "committed, with their classes, as in the loop")
    ap.add_argument("--per-item", type=int, default=128,
                    help="patches supervised per part per epoch; neighbourhoods "
                         "are always built from the full cloud, and inference "
                         "still classifies every point")
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--eval-parts", type=int, default=12)
    ap.add_argument("--max-hours", type=float, default=2.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    parts = load_curve_parts(pathlib.Path(args.wtok))
    have = {f.stem for f in (pathlib.Path(args.dataset) / "parts").glob("*.npz")}
    parts = [p for p in parts if p.name in have]
    vn = set(json.loads(pathlib.Path(args.val_list).read_text(encoding="utf-8")))
    train_parts = [p for p in parts if p.name not in vn]
    val_parts = [p for p in parts if p.name in vn]
    md = pathlib.Path(args.dataset) / "parts"
    print(f"parts: train {len(train_parts)} val {len(val_parts)}  N={args.points} "
          f"K={K_NN}")
    tl = DataLoader(PatchDataset(train_parts, md, args.points, True,
                                 per_item=args.per_item,
                                 confirm_max=args.confirm_max,
                                 degrade_rate=args.degrade_rate),
                    batch_size=args.batch_size, shuffle=True, drop_last=True)
    args.in_ch = 6 + (CONFIRM_CH if args.confirm_max > 0 else 0)
    model = PatchNet(args.dim, args.layers, args.heads, args.in_ch).to(args.device)
    print(f"params: {sum(q.numel() for q in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, args.lr * 0.05)
    hist, best, t0, start = [], float("inf"), time.time(), 1
    last = out / "last.pt"
    if args.resume and last.exists():
        ck = torch.load(last, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["epoch"] + 1
        hp = out / "history.json"
        if hp.exists():
            hist = [r for r in json.loads(hp.read_text()) if r["epoch"] < start]
        for _ in range(start - 1):
            sched.step()
        print(f"resumed at {start}", flush=True)
    for epoch in range(start, args.epochs + 1):
        tl.dataset.set_epoch(epoch)
        model.train()
        te, tot, n = time.time(), 0.0, 0
        for b in tl:
            loss = loss_fn(model, {k: v.to(args.device) for k, v in b.items()})
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            n += 1
        sched.step()
        row = {"epoch": epoch, "loss": tot / max(n, 1),
               "seconds": round(time.time() - te, 1)}
        msg = f"epoch {epoch}: loss {row['loss']:.5f} ({row['seconds']}s)"
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            row.update(score(model, val_parts, md, args.points, args.device,
                             args.eval_parts))
            msg += (f" | outline {row['outline']:.2f}mm bend {row['bend']:.2f}mm "
                    f"spur {row['spur']:.2f}mm kept {row['kept']*100:.0f}%")
            key = float(np.nanmean([row["outline"], row["bend"]]))
            if key < best:
                best = key
                safe_save({"model": model.state_dict(), "args": vars(args),
                           "epoch": epoch}, out / "best.pt")
                msg += " *best*"
            safe_save({"model": model.state_dict(), "opt": opt.state_dict(),
                       "args": vars(args), "epoch": epoch}, last)
        hist.append(row)
        (out / "history.json").write_text(json.dumps(hist), encoding="utf-8")
        print(msg, flush=True)
        if (time.time() - t0) / 3600.0 > args.max_hours:
            print("stopping at the budget", flush=True)
            break
    print(f"\nlocal kernel: mean(outline, bend) {best:.2f}mm")


if __name__ == "__main__":
    main()

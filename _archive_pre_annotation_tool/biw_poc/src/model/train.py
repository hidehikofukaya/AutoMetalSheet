"""
train.py -- Stage B training loop for the VecSet Autoencoder
(CLAUDE.md section 13, Round 6 PoC architecture).

Loss
----
UDF loss:  L1(pred_udf, gt_udf), weighted by query_category (R7-09: near /
           far / boundary-shadow, assigned at Stage A sampling time --
           replaces the old single near_dist_mm threshold). Each category
           is clamped to its own clip distance before the L1 (R7-03-style
           truncated distance loss): near/far use --clip-dist-mm (default
           5mm), boundary-shadow queries use the much larger
           --boundary-clip-dist-mm (default 40mm, matched to
           midsurface_sampler.py's --boundary-shadow-max-mm) so the network
           gets real gradient signal that UDF keeps GROWING past a cut edge,
           instead of every shadow-region query collapsing to one flat
           clamped target. Confirmed empirically (R7-03) that under-
           clamping starves near-surface signal and over-clamping (no
           boundary category) starves exactly the cut-locus region TD-16
           is about -- this is why boundary-shadow needs its own, much
           larger clip distance rather than reusing clip_dist_mm.
Grad loss: 1 - cosine_similarity(pred_grad, gt_grad), same category
           weighting (R7-01: grad is the UDF gradient direction, supervised
           via trimesh-derived nearest-point vectors, see midsurface_sampler.py).
R7-16 (experimental, --loss-eps-mm): optionally replace the per-query
           L1 residual |pred_c - gt_c| with a Charbonnier / pseudo-Huber
           smooth-abs form sqrt((pred_c-gt_c)^2 + eps^2) - eps. This is
           DIFFERENT from R7-09's Eikonal term: Eikonal reshapes what the
           network is asked to fit (pushes the predicted FIELD toward true
           unit-gradient distance shape, including near the cut locus).
           R7-16 instead reshapes the LOSS LANDSCAPE around zero residual
           (pred==target) without changing the target itself -- L1's
           gradient is a discontinuous +-1 sign flip exactly at zero
           residual, which is a plausible source of gradient jitter when
           the network must compromise between conflicting nearby targets
           (e.g. two cut-locus branches pulling pred_udf in different
           directions for nearby query points). eps=0 (default) recovers
           the exact original L1 loss bit-for-bit. This does NOT smooth the
           cut locus itself (see CLAUDE.md Round 8/Phase 2a discussion) --
           it is a single, cheap, isolated experiment, not a TD-16 fix.
R7-17追補3 (model/loss-side mitigation for softmin's residual negative-GT
           symptom, --gt-floor-tau-mm / --neg-gt-weight): after R7-17追補2's
           対策A+B (branch dedup + 5mm clip radius, see
           experiments/loss_smoothing/softmin_r717/), a residual ~57%% of
           near-surface (<0.5mm) query points still carry a physically
           invalid NEGATIVE softmin GT UDF (genuine multi-branch
           competition, not the dedup bug or far-field parallax those two
           fixes targeted). QueryDecoder's output head is already softplus-
           activated (R7-04) so the MODEL can never itself predict a
           negative value -- the mismatch is entirely between an
           occasionally-negative GT *target* and an always-positive
           prediction head, not a representable-output gap. Two
           complementary, opt-in (default = bit-identical to before),
           loss-side responses, deliberately NOT touching the persisted
           dataset (unlike 対策C/hard-clamp, which was considered and
           explicitly deferred -- see CLAUDE.md section 19):
             対策D (--gt-floor-tau-mm, default 0.0=off): a smooth softplus
               floor tau*softplus(gt/tau) applied to the GT UDF ONLY at
               loss-computation time, before the existing R7-03 upper-bound
               clip. Unlike max(gt,0), this is differentiable everywhere
               (no kink at zero) and uses the same functional family as the
               model's own output activation. near_mae (the overfit-sanity
               diagnostic) deliberately keeps reading the RAW (unfloored)
               gt_udf so it still reports the true, honest discrepancy.
             対策E (--neg-gt-weight, default 1.0=off): multiplies the
               per-query loss weight by this factor wherever the raw GT is
               negative, down-weighting (not altering) those samples'
               contribution to udf_loss and grad_loss -- a deterministic,
               precomputed weight derived from the GT data itself, NOT the
               learned confidence head (using the not-yet-converged
               confidence prediction to weight its own training loss in the
               same step would be circular/unstable).
           Both default to no-op so existing runs/checkpoints are
           unaffected; see CLAUDE.md section 19 for the full design
           rationale and the user's approved "D+E併用" direction.
Eikonal loss (R7-09, Option 2): true unsigned distance fields satisfy
           |grad d| = 1 almost everywhere except exactly at the cut locus
           (the non-differentiable kink the network is extrapolating across
           at TD-16's boundary). This is NOT enforced via QueryDecoder's
           `grad` output channel -- that channel is F.normalize()'d to unit
           length by construction (vecset_ae.py), so its "magnitude" is not
           a free/learnable quantity. Instead this loss autodiffs the
           network's scalar pred_udf output w.r.t. query_xyz directly
           (torch.autograd.grad with create_graph=True) and penalizes
           (|d(pred_udf)/d(query_xyz)| - 1)^2. Because dataset.py's R7-02
           per-part normalization is isotropic, the scale factor relating
           normalized and world-mm coordinates cancels exactly in this
           ratio, so the constraint target is exactly 1.0 in EITHER unit
           system -- the loss is computed directly on the raw (normalized-
           space) model output, no rescaling needed. Auxiliary regularizer
           (small --lambda-eikonal weight): it nudges the field toward a
           true-distance shape everywhere, including the under-supervised
           shadow region, without requiring any new ground truth.

Usage
-----
Single-part overfit sanity check (the PoC's own "H2 hypothesis" test --
can the model memorize one part's field at all):
    python train.py --data path/to/body002_filled_dataset.h5 --epochs 300 --overfit-check

Multi-part training:
    python train.py --data path/to/output_root --epochs 200 --batch-size 2
"""

import argparse
import os
import pathlib
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vecset_ae import VecSetAE
from dataset import MidSurfaceUDFDataset, collate_fixed_size


# R7-09: query_category values (must match midsurface_sampler.py)
CATEGORY_NEAR, CATEGORY_FAR, CATEGORY_BOUNDARY = 0, 1, 2


def weighted_losses(pred_udf, pred_grad, gt_udf, gt_grad, query_category,
                     near_weight: float, far_weight: float, boundary_weight: float,
                     clip_dist_mm: float, boundary_clip_dist_mm: float,
                     loss_eps_mm: float = 0.0,
                     gt_floor_tau_mm: float = 0.0,
                     neg_gt_weight: float = 1.0):
    weight_table = torch.tensor(
        [near_weight, far_weight, boundary_weight], device=gt_udf.device)
    clip_table = torch.tensor(
        [clip_dist_mm, clip_dist_mm, boundary_clip_dist_mm], device=gt_udf.device)
    w = weight_table[query_category]

    if neg_gt_weight != 1.0:
        # 対策E (R7-17追補3): down-weight (not alter) any query point whose
        # raw GT softmin UDF is negative -- a deterministic signal computed
        # directly from the GT data, not the learned confidence head (using
        # a co-evolving predicted confidence to weight its own training
        # signal in the same step would be circular/unstable). Composes
        # multiplicatively with the existing per-category weight.
        neg_mask = gt_udf < 0
        w = torch.where(neg_mask, w * neg_gt_weight, w)

    clip = clip_table[query_category]
    w_sum = w.sum().clamp_min(1e-8)

    if gt_floor_tau_mm > 0:
        # 対策D (R7-17追補3): smooth softplus floor on the GT target ONLY,
        # applied here at loss time -- the persisted dataset and the raw
        # gt_udf used below for near_mae are left untouched. Differentiable
        # everywhere (no kink at zero), unlike a hard max(gt_udf, 0) clamp.
        gt_for_loss = gt_floor_tau_mm * F.softplus(gt_udf / gt_floor_tau_mm)
    else:
        gt_for_loss = gt_udf

    # R7-03/R7-09: truncated distance loss -- clamp both sides before the L1
    # so a huge raw far-field target cannot dominate the gradient, with a
    # per-category clip distance so the boundary-shadow category (R7-09)
    # still gets real gradient signal far past the old 5mm near/far clip.
    pred_c = torch.minimum(pred_udf, clip)
    gt_c = torch.minimum(gt_for_loss, clip)
    if loss_eps_mm > 0:
        # R7-16 (experimental): Charbonnier / pseudo-Huber smooth-abs of the
        # residual -- removes L1's gradient discontinuity at zero residual
        # without changing the (clamped) target itself. eps=0 path above is
        # bit-for-bit identical to the pre-R7-16 L1 loss.
        residual = pred_c - gt_c
        udf_err = torch.sqrt(residual * residual + loss_eps_mm * loss_eps_mm) - loss_eps_mm
    else:
        udf_err = (pred_c - gt_c).abs()
    udf_loss = (udf_err * w).sum() / w_sum

    cos = F.cosine_similarity(pred_grad, gt_grad, dim=-1)
    grad_loss = ((1.0 - cos) * w).sum() / w_sum

    with torch.no_grad():
        is_near = query_category == CATEGORY_NEAR
        true_err = (pred_udf - gt_udf).abs()
        near_mae = true_err[is_near].mean() if is_near.any() else torch.tensor(0.0, device=gt_udf.device)
    return udf_loss, grad_loss, near_mae


def confidence_loss(pred_conf: torch.Tensor, gt_udf: torch.Tensor, horizon_mm: float) -> torch.Tensor:
    """
    R7-10: train a confidence/uncertainty head independently of the UDF
    regression, to suppress TD-16's boundary "ghost mesh" at Stage C.

    Target derives from the SAME gt_udf already computed for every query
    (no new Stage A sampling needed, and uniform across near/far/boundary-
    shadow categories): target=1 at the true surface, decaying linearly to
    0 by horizon_mm away. Points just past a true cut edge legitimately
    have small gt_udf (the boundary curve is part of the mesh), so this
    naturally stays confident right up to real geometry and only collapses
    deeper into the extrapolated shadow region -- the region Stage C should
    refuse to mesh.

    BCE (not L1) follows arxiv.org/html/2306.02099v4's uncertainty-
    augmented SDF/UDF design: a classification-style loss only needs the
    right side of a soft threshold near the surface's non-differentiable
    cut locus, rather than the exact extrapolated distance magnitude the
    main UDF regression struggles with (TD-16's documented root cause).
    """
    target = (1.0 - gt_udf / horizon_mm).clamp(0.0, 1.0)
    return F.binary_cross_entropy(pred_conf.clamp(1e-6, 1 - 1e-6), target)


def eikonal_loss(pred_udf_norm: torch.Tensor, query_xyz: torch.Tensor) -> torch.Tensor:
    """
    R7-09 Option 2: penalize |d(pred_udf_norm)/d(query_xyz)| deviating from 1.
    query_xyz must have requires_grad_(True) set before the forward pass.
    See the module docstring for why this uses the normalized-space udf
    output directly rather than the world-mm rescaled pred_udf.
    """
    grad = torch.autograd.grad(
        outputs=pred_udf_norm, inputs=query_xyz,
        grad_outputs=torch.ones_like(pred_udf_norm),
        create_graph=True, retain_graph=True,
    )[0]
    return ((grad.norm(dim=-1) - 1.0) ** 2).mean()


def save_ckpt(obj, path: pathlib.Path, retries: int = 10, delay_s: float = 1.0) -> None:
    """
    torch.save() via write-to-temp + atomic os.replace(), with retries
    around BOTH steps as a safety net. On this Windows dev box, repeated
    overwrites of the SAME checkpoint filename (best.pt gets rewritten every
    time near_mae improves, often dozens of times) intermittently hit
    "RuntimeError: File ... cannot be opened" from PyTorchFileWriter --
    confirmed via `Get-MpPreference` that Windows Defender real-time
    protection is active, which transiently locks a just-written file for
    scanning. Writing to a fresh, never-before-seen temp filename sidesteps
    the lock during the write itself (nothing has scanned it yet). However
    the rename step is NOT immune: os.replace() needs to open/overwrite the
    DESTINATION path, and if Defender is mid-scan of the previous best.pt
    at that instant, replace() itself raises PermissionError ([WinError 5]
    Access is denied) -- observed in practice (training reached epoch ~90
    before hitting this on the replace() call specifically, distinct from
    the earlier torch.save()-side RuntimeError). So both the save-to-temp
    and the temp-to-final rename are retried independently; the temp file
    is only written once and reused across replace() retries (no need to
    redo the expensive torch.save). Not a logic bug; this is the standard
    Windows+AV workaround for frequent-checkpoint-overwrite races.
    """
    path = pathlib.Path(path)
    tmp_path = path.with_name(f"{path.name}.tmp{os.getpid()}_{int(time.time()*1000)}")

    last_exc = None
    for attempt in range(retries):
        try:
            torch.save(obj, tmp_path)
            break
        except RuntimeError as exc:
            last_exc = exc
            time.sleep(delay_s)
    else:
        raise last_exc

    last_exc = None
    for attempt in range(retries):
        try:
            tmp_path.replace(path)
            return
        except OSError as exc:
            last_exc = exc
            time.sleep(delay_s)
    tmp_path.unlink(missing_ok=True)
    raise last_exc


def run_epoch(model, loader, device, optimizer, cfg, train: bool):
    model.train(train)
    tot_udf = tot_grad = tot_near_mae = tot_eik = tot_conf = 0.0
    n_batches = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            points = batch["points"].to(device)
            normals = batch["normals"].to(device)
            query_xyz = batch["query_xyz"].to(device)
            gt_udf = batch["query_udf"].to(device)
            gt_grad = batch["query_grad"].to(device)
            query_category = batch["query_category"].to(device)   # R7-09
            scale = batch["scale"].to(device)            # [B], R7-02

            if train and cfg.lambda_eikonal > 0:
                query_xyz.requires_grad_(True)

            pred_udf_norm, pred_grad, pred_conf, _ = model(points, normals, query_xyz)
            pred_udf = pred_udf_norm * scale.unsqueeze(-1)   # back to mm for the loss
            udf_loss, grad_loss, near_mae = weighted_losses(
                pred_udf, pred_grad, gt_udf, gt_grad, query_category,
                cfg.near_weight, cfg.far_weight, cfg.boundary_weight,
                cfg.clip_dist_mm, cfg.boundary_clip_dist_mm,
                loss_eps_mm=cfg.loss_eps_mm,
                gt_floor_tau_mm=cfg.gt_floor_tau_mm,
                neg_gt_weight=cfg.neg_gt_weight,
            )
            conf_loss = confidence_loss(pred_conf, gt_udf, cfg.conf_horizon_mm)   # R7-10
            loss = udf_loss + cfg.lambda_grad * grad_loss + cfg.lambda_conf * conf_loss

            eik = torch.tensor(0.0, device=device)
            if train and cfg.lambda_eikonal > 0:
                eik = eikonal_loss(pred_udf_norm, query_xyz)
                loss = loss + cfg.lambda_eikonal * eik

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            tot_udf += udf_loss.item()
            tot_grad += grad_loss.item()
            tot_near_mae += near_mae.item()
            tot_eik += eik.item()
            tot_conf += conf_loss.item()
            n_batches += 1

    return (tot_udf / n_batches, tot_grad / n_batches,
            tot_near_mae / n_batches, tot_eik / n_batches, tot_conf / n_batches)


def main():
    ap = argparse.ArgumentParser(description="Train the Stage B VecSet Autoencoder.")
    ap.add_argument("--data", required=True,
                     help="*_dataset.h5 file, OR a directory to glob recursively")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n-query-sample", type=int, default=4096)
    ap.add_argument("--n-points-sample", type=int, default=None)
    ap.add_argument("--near-weight", type=float, default=1.0)
    ap.add_argument("--far-weight", type=float, default=0.2)
    ap.add_argument("--boundary-weight", type=float, default=0.5,
                     help="R7-09: loss weight for boundary-shadow queries (between "
                          "near and far weight -- important for TD-16 but not as "
                          "critical as the exact zero level set).")
    ap.add_argument("--clip-dist-mm", type=float, default=5.0,
                     help="Truncated-distance clamp for near/far queries (R7-03).")
    ap.add_argument("--boundary-clip-dist-mm", type=float, default=40.0,
                     help="R7-09: truncated-distance clamp for boundary-shadow queries. "
                          "Should match midsurface_sampler.py's --boundary-shadow-max-mm.")
    ap.add_argument("--loss-eps-mm", type=float, default=0.0,
                     help="R7-16 (experimental): if >0, replace the UDF L1 residual with "
                          "a Charbonnier smooth-abs sqrt(residual^2+eps^2)-eps using this "
                          "eps (mm). 0 (default) is the exact original L1 loss.")
    ap.add_argument("--gt-floor-tau-mm", type=float, default=0.0,
                     help="対策D (R7-17追補3): if >0, apply a smooth softplus floor "
                          "tau*softplus(gt/tau) to the GT UDF target ONLY at loss "
                          "computation time (the persisted dataset and the near_mae "
                          "diagnostic both keep reading the raw, unfloored value). "
                          "Mitigates softmin's residual negative-GT symptom (see "
                          "CLAUDE.md R7-17追補2/3) without touching the dataset. "
                          "0 (default) disables this and is bit-identical to before.")
    ap.add_argument("--neg-gt-weight", type=float, default=1.0,
                     help="対策E (R7-17追補3): multiply the loss weight of any query "
                          "point whose raw GT softmin UDF is negative by this factor "
                          "(e.g. 0.2 down-weights such points to 20%% of their normal "
                          "weight in udf_loss/grad_loss). A deterministic, GT-derived "
                          "weight -- NOT the learned confidence head. 1.0 (default) is "
                          "a no-op, bit-identical to before.")
    ap.add_argument("--lambda-grad", type=float, default=0.5)
    ap.add_argument("--lambda-eikonal", type=float, default=0.05,
                     help="R7-09 Option 2: weight of the Eikonal |grad pred_udf|=1 "
                          "regularizer. 0 disables it.")
    ap.add_argument("--lambda-conf", type=float, default=0.3,
                     help="R7-10: weight of the confidence/uncertainty head's BCE loss.")
    ap.add_argument("--conf-horizon-mm", type=float, default=15.0,
                     help="R7-10: confidence target decays from 1 (at gt_udf=0) to 0 by "
                          "this distance (mm). Should be on the order of the boundary "
                          "ghost-mesh extent Stage C needs to suppress (TD-16).")
    ap.add_argument("--token-dim", type=int, default=384)
    ap.add_argument("--n-tokens", type=int, default=256)
    ap.add_argument("--k-neighbors", type=int, default=32)
    ap.add_argument("--n-latents", type=int, default=128)
    ap.add_argument("--enc-layers", type=int, default=6)
    ap.add_argument("--dec-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--n-freqs", type=int, default=8,
                     help="Fourier positional encoding bands in QueryDecoder (R7-05)")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overfit-check", action="store_true",
                     help="Single-part sanity mode: print a pass/fail verdict at the end "
                          "based on whether near-surface UDF MAE converges below 0.1mm.")
    cfg = ap.parse_args()

    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    dataset = MidSurfaceUDFDataset(
        cfg.data, n_query_sample=cfg.n_query_sample,
        n_points_sample=cfg.n_points_sample, seed=cfg.seed,
    )
    print(f"dataset: {len(dataset)} part(s): {[p.name for p in dataset.h5_paths]}")
    loader = DataLoader(dataset, batch_size=min(cfg.batch_size, len(dataset)),
                         shuffle=True, collate_fn=collate_fixed_size, drop_last=False)

    model = VecSetAE(
        token_dim=cfg.token_dim, n_tokens=cfg.n_tokens, k_neighbors=cfg.k_neighbors,
        n_latents=cfg.n_latents, enc_layers=cfg.enc_layers, dec_layers=cfg.dec_layers,
        n_heads=cfg.n_heads, n_freqs=cfg.n_freqs,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    ckpt_dir = pathlib.Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_near_mae = float("inf")
    history = []
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        udf_loss, grad_loss, near_mae, eik_loss, conf_loss = run_epoch(
            model, loader, device, optimizer, cfg, train=True)
        scheduler.step()
        history.append((epoch, udf_loss, grad_loss, near_mae, eik_loss, conf_loss))

        if epoch % cfg.log_every == 0 or epoch == 1 or epoch == cfg.epochs:
            print(f"  epoch {epoch:4d}/{cfg.epochs}  udf_l1={udf_loss:.4f}mm  "
                  f"grad_cos_loss={grad_loss:.4f}  near_mae={near_mae:.4f}mm  "
                  f"eikonal={eik_loss:.4f}  conf_bce={conf_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  ({time.time()-t0:.1f}s)")

        if near_mae < best_near_mae:
            best_near_mae = near_mae
            save_ckpt({"model": model.state_dict(), "cfg": vars(cfg), "epoch": epoch},
                      ckpt_dir / "best.pt")
        if epoch % cfg.ckpt_every == 0:
            save_ckpt({"model": model.state_dict(), "cfg": vars(cfg), "epoch": epoch},
                      ckpt_dir / f"epoch{epoch:04d}.pt")

    save_ckpt({"model": model.state_dict(), "cfg": vars(cfg), "epoch": cfg.epochs},
              ckpt_dir / "last.pt")
    print(f"done in {time.time()-t0:.1f}s. best near_mae={best_near_mae:.4f}mm "
          f"-> {ckpt_dir/'best.pt'}")

    if cfg.overfit_check:
        verdict = "PASS" if best_near_mae < 0.1 else "FAIL"
        print(f"\noverfit sanity check: {verdict} "
              f"(near-surface UDF MAE {best_near_mae:.4f}mm, threshold 0.1mm)")


if __name__ == "__main__":
    main()

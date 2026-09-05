"""The commit loop: generate, classify, freeze the confident points, repeat.

One pass of the generator has to place every point at once, and with only two
fastening points to go on there are many plausible parts. The L2-optimal answer
to "which one" is their average, which is no real part -- measured as a cloud
that runs 2-5% past the true boundary at every guidance scale. Committing a few
points collapses the distribution to parts consistent with them, and the next
round completes rather than averages.

Measured before building this (see docs/loop_architecture_proposal.md):
  - giving the generator 128 of 512 true points halves the error on the other
    384 (5.43 -> 2.50mm), and lifts top-5% classification from 16% to 72%
  - the gain survives ~5mm of error on the committed points, breaks by 15mm
  - the outline is read reliably with no context (AUC 0.983) while the bend
    lines need it (0.933 -> 0.980 once points are confirmed), which is why the
    schedule commits outline first and bends in the middle rounds
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree

from .kernel import CONFIRM_CH, K_NN, patches
from .meshgen import CH, FIELD_CAP, fastener_frame, fps_order, generate
from .ridge import CLASSES

# Newly committed points per round, and what fraction of them each class takes.
# Cumulatively this lands on 14/29/57%, which is what the parts actually are --
# a fixed schedule, but one that adds up.
# Two rounds, committing a fifth. Five rounds bought nothing: the loop's value
# is keeping the confident points and redrawing the rest, not accumulating
# information, so one selection and one redraw is the whole mechanism. Measured
# over 10 val parts (surface / span / AUC outline / AUC bend):
#
#   1 round            8.51mm  0.976  0.935  0.747
#   2 rounds, 20%      8.87mm  0.927  0.963  0.736   <- this
#   2 rounds, 70%      8.41mm  0.928  0.942  0.730
#   3 rounds           8.71mm  0.925  0.956  0.711
#   5 rounds           8.88mm  0.928  0.971  0.720
#
# Committing more sharpens the surface and blunts the outline; 20% takes the
# outline. Bend detection is best with no loop at all, because it is read from
# the generator's own field and committing dilutes it.
SCHEDULE = (
    (0.20, (0.15, 0.29, 0.56)),      # the proportions the parts actually have
    (1.00, (0.15, 0.29, 0.56)),
)
# Round 1 as a spread rather than as pure outline. Confirming 64 true outline
# points lifted bend AUC 0.735 -> 0.849, which is real, but 64 points spread at
# random did better on every measure (bend 0.939, surface 4.79mm vs 6.12mm,
# span 0.978 vs 1.023). Outline points all sit on the rim and leave the interior
# unconstrained; what the generator needs is coverage, not one class.
SCHEDULE_MIX = (
    (0.05, (0.40, 0.25, 0.35)),
    (0.15, (0.35, 0.35, 0.30)),
    (0.30, (0.15, 0.50, 0.35)),
    (0.55, (0.05, 0.35, 0.60)),
    (1.00, (0.00, 0.10, 0.90)),
)
OUTLIER_DROP = 0.5      # never commit a point the generator disowns
SPREAD_POOL = 3         # rank by confidence, then spread within this many x k

# How much of each class's score comes from the generator's own distance field
# rather than from the kernel reading the geometry. They are not redundant: the
# field is what the model intended, the kernel is what the cloud looks like, and
# the cloud's normals are 15.4 degrees off. Measured AUC on generated clouds:
#   outline  kernel 0.928  field 0.893  best blend 0.934 at 0.25
#   bend     kernel 0.736  field 0.798  best blend 0.804 at 0.75
# The asymmetry is consistent -- an outline is visible in the geometry, a shallow
# bend is not, once the normals are that noisy.
FIELD_W = (0.25, 0.75)


def _rank(v):
    """Values as ranks in [0, 1], so two differently-scaled scores can be mixed."""
    r = np.empty(len(v))
    r[np.argsort(v)] = np.arange(len(v))
    return r / max(len(v) - 1, 1)


def _confidence(kernel, xyz, nrm, confirmed, device, fld=None):
    """Per-point class score, judged with the confirmed points visible.

    Blends the kernel's reading of the geometry with the generator's own
    distance field (a small field distance means the point is on that curve).
    """
    feat, _ = patches(xyz, nrm, K_NN,
                      confirmed if getattr(kernel, "in_ch", 6) > 6 else None)
    with torch.no_grad():
        _, logit = kernel(torch.from_numpy(feat)[None].to(device))
    conf = torch.sigmoid(logit[0]).cpu().numpy()
    if fld is None:
        return conf
    out = np.empty_like(conf)
    for ci in range(len(CLASSES)):
        w = FIELD_W[ci]
        out[:, ci] = (1 - w) * _rank(conf[:, ci]) + w * _rank(-fld[:, ci])
    return out


def _spread(xyz, cand, k, score):
    """The k most confident candidates, spread out rather than clustered.

    Straight top-k puts every committed point wherever the classifier happens to
    be surest, which for the outline is the whole rim and nothing inside. Taking
    a larger confident pool and then farthest-point sampling it keeps the
    confidence gate but buys coverage.
    """
    pool = cand[np.argsort(-score[cand])][: max(k, SPREAD_POOL * k)]
    if len(pool) <= k:
        return pool
    return pool[fps_order(xyz[pool].astype(np.float32), k)[:k]]


def _pick(conf, avail, want, outlier, xyz=None, spread=True):
    """Choose which available points to commit, per class, by confidence.

    Confidence ranks class, not position, and the two are not the same: the
    most confident outline points were measured 8.21mm from the truth against a
    4.88mm cloud average, because the cloud's overrun is exactly what looks like
    an edge. The outlier flag is what separates them, so anything the generator
    disowns is dropped from the pool before ranking.
    """
    pool = avail & (outlier < OUTLIER_DROP)
    if not pool.any():
        pool = avail
    out = {}
    taken = np.zeros(len(conf), bool)
    for ci in range(len(CLASSES)):
        k = want[ci]
        if k <= 0:
            continue
        cand = np.flatnonzero(pool & ~taken)
        if not len(cand):
            continue
        order = (_spread(xyz, cand, k, conf[:, ci]) if spread and xyz is not None
                 else cand[np.argsort(-conf[cand, ci])][:k])
        taken[order] = True
        out[ci] = order
    # whatever is left of the quota goes to plain surface points: the ones the
    # classifier is most sure are NEITHER curve
    k = want[len(CLASSES)]
    if k > 0:
        cand = np.flatnonzero(pool & ~taken)
        if len(cand):
            surf = 1.0 - conf.max(1)
            out[len(CLASSES)] = (_spread(xyz, cand, k, surf)
                                 if spread and xyz is not None
                                 else cand[np.argsort(-surf[cand])][:k])
    return out


# 24 steps rather than 48: measured 8.51mm against 8.27mm on a single pass,
# a difference well inside the +-1-2mm a probe of this size swings, for exactly
# half the model calls. Halving guidance instead costs more (8.91mm at cfg 1.0).
# The loop runs the generator five times, so this is where its cost lives.
def relax_spacing(xyz, nrm, hold=None, iters=8):
    """Even out the spacing of the filler points, in the tangent plane only.

    The generator's clouds are 4.5x less regular than the true ones (spacing CV
    0.52 against 0.12) and 2.7% of points sit effectively on top of another,
    which a mesher cannot use. Nothing in flow matching penalises that: the
    points are exchangeable and each is pulled to a plausible position
    independently, so several landing together costs nothing.

    Movement is projected out of the normal direction, so points slide along
    the sheet rather than off it, and `hold` (the outline and bend points) is
    frozen -- those are the geometry downstream actually wants, and an outline
    point nudged inward stops being one.
    """
    pts = xyz.copy()
    n = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    free = np.ones(len(pts), bool) if hold is None else ~hold
    if free.sum() < 4:
        return pts
    for _ in range(iters):
        k = min(7, len(pts))
        d, nb = cKDTree(pts).query(pts, k=k)
        target = float(np.median(d[:, 1]))
        step = np.zeros_like(pts)
        for j in range(1, k):
            delta = pts - pts[nb[:, j]]
            dist = np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-9)
            step += delta / dist * np.clip((target - dist) / target, 0.0, 1.0)                 * target * 0.25
        step /= max(k - 1, 1)
        step -= n * np.einsum("ij,ij->i", step, n)[:, None]   # stay on the sheet
        pts[free] += step[free]
    return pts


def run(gen_model, kernel, part, device, n_pts, steps=24, cfg=2.0, seed=7,
        schedule=SCHEDULE, spread=True, verbose=False):
    """Returns (xyz, normals, fields, per-point class committed or -1)."""
    frame = fastener_frame(part)
    known = np.zeros((0, CH))          # mm: xyz, normal, fields, outlier flag
    klass = np.zeros(0, dtype=int)
    xyz = nrm = fld = None
    for rnd, (cum, mix) in enumerate(schedule):
        xyz, nrm, fld = generate(gen_model, part, device, steps, seed + rnd, cfg,
                                 n_pts, True, per_fix=1,
                                 known=known if len(known) else None)
        n_conf = len(known)
        if cum >= 1.0:
            break
        confirmed = np.zeros((len(xyz), CONFIRM_CH), np.float32)
        if n_conf:
            confirmed[:n_conf, 0] = 1.0
            for ci in range(len(CLASSES)):
                confirmed[:n_conf, 1 + ci] = (klass == ci).astype(np.float32)
        conf = _confidence(kernel, xyz, nrm, confirmed, device, fld)
        avail = np.ones(len(xyz), bool)
        avail[:n_conf] = False          # already committed, never re-judged
        target = int(cum * n_pts) - n_conf
        if target <= 0:
            continue
        want = [int(round(target * m)) for m in mix]
        picks = _pick(conf, avail, want, fld[:, 2], xyz, spread)
        rows, cls = [], []
        for ci, idx in picks.items():
            f = fld[idx].copy()
            f[:, 2] = 0.0               # committed points are not outliers
            if ci < len(CLASSES):       # pin the field that says what it is
                f[:, ci] = 0.0
            rows.append(np.concatenate([xyz[idx], nrm[idx], f], axis=1))
            cls.append(np.full(len(idx), ci if ci < len(CLASSES) else -1))
        if not rows:
            continue
        known = np.concatenate([known] + rows) if len(known) else np.concatenate(rows)
        klass = np.concatenate([klass] + cls) if len(klass) else np.concatenate(cls)
        if verbose:
            got = {CLASSES[c] if c < len(CLASSES) else "surface": len(i)
                   for c, i in picks.items()}
            print(f"  round {rnd + 1}: committed {len(known)}/{n_pts}  {got}",
                  flush=True)
    committed = -np.ones(len(xyz), int)
    committed[:len(klass)] = klass
    return xyz, nrm, fld, committed


def demo():
    """Self-check on the schedule arithmetic and on spreading -- no model."""
    for name, sched in (("SCHEDULE", SCHEDULE), ("SCHEDULE_MIX", SCHEDULE_MIX)):
        check(name, sched)
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(400, 3))
    cand = np.arange(400)
    score = np.exp(-np.linalg.norm(xyz - xyz[0], axis=1))   # confident in a blob
    top = cand[np.argsort(-score)][:20]
    spr = _spread(xyz, cand, 20, score)
    r_top = float(np.linalg.norm(xyz[top] - xyz[top].mean(0), axis=1).mean())
    r_spr = float(np.linalg.norm(xyz[spr] - xyz[spr].mean(0), axis=1).mean())
    print(f"spread: mean radius {r_top:.2f} (top-k) -> {r_spr:.2f} (spread)")
    assert r_spr > r_top, "spreading did not spread"


def check(name, schedule):
    cum = 0.0
    totals = np.zeros(3)
    for c, mix in schedule:
        assert c > cum, "the schedule must commit more each round"
        assert abs(sum(mix) - 1.0) < 1e-9, f"mix {mix} does not sum to 1"
        totals += np.array(mix) * (c - cum)
        cum = c
    assert abs(cum - 1.0) < 1e-9, "the schedule must finish the cloud"
    print(f"{name:14s} outline {totals[0]:.3f} bend {totals[1]:.3f} "
          f"surface {totals[2]:.3f}   (parts are 0.148 / 0.293 / 0.559)")


if __name__ == "__main__":
    demo()

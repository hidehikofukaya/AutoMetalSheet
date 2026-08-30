"""Generate the midsurface as an oriented point cloud, then extract the wire.

The wire is a sparse combinatorial object, and every failure measured so far was
combinatorial: a third of B-core's edges referenced vertices that did not exist.
A surface sample has no connectivity to get wrong -- only coordinates, which is
the one paradigm that has worked here (flow matching). The wire comes out
afterwards from generic geometry (mesh_extract), measured at ~4mm even with 3mm
of noise, against generators that are at 23-31mm.

  stage "train": conditional flow matching over N points x (xyz, normal)
  stage "eval" : sample -> extract -> the session's standard decomposition
  stage "smoke": self-check

Anchors: the fastening points lie ON the midsurface (measured 0.78mm to the
nearest of 4096 samples), so slots 0..n_fix-1 are pinned to them at zero noise
for every step -- R1, and here it needs no incidence rule because a point IS the
surface.
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
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

from .codec import bin_center, realize_points
from .constants import safe_save, stable_seed
from .dataset_ar import cond_features
from .dataset_curve import load_curve_parts
from .mesh_extract import extract
from .plan_g import COND_ROWS, Block, RelBias, timestep_embedding
from .train_ar import chamfer_mm
from .train_curve import realized_q

CH = 9      # xyz, normal, two learned fields, then an off-the-part flag
# Nothing in the training data is off the part -- every sample sits on the
# surface -- so the flag has no signal unless we make some. A fraction of the
# free points are pushed off the sheet and labelled 1; every real sample is 0.
# The generated cloud measurably overruns the true part (span ratio 1.02-1.05
# at every guidance scale, so it is not a CFG artifact), and the points it
# overruns with are exactly the ones the classifier most confidently calls
# outline: 8.21mm from the truth against a 4.88mm cloud average. The flag is
# what lets the model disown them instead of being forced to place every point
# on the sheet.
OUTLIER_RATE = 0.35     # of items, how often outliers are injected at all
OUTLIER_FRAC = 0.10     # of the free points in such an item
OUTLIER_PITCH = (1.0, 4.0)   # displacement, in units of the point spacing
FIELD_CAP = 0.5    # frame units; distances past this carry no useful signal

# Channels 6 and 7 hold the distance from the point to the outline and to the
# nearest bend line. Regressing a field instead of labelling points is EC-Net's
# trick and generating it jointly with the geometry is SeaLion's; together they
# fix three measured failures.
#   - the curves are measure-zero in the surface, so a uniform sample lands on
#     each one about once. A field is defined at every point.
#   - the geometric rim test is density-dependent and gets WORSE with more
#     points: on PERFECT clouds its false-rim rate went 5.7% at 256 points to
#     22.8% at 4096, and recall fell 73% -> 25%.
#   - the trim is a design decision, not a geometric feature. A bend turns the
#     normal; a cut just stops the sheet, with no curvature to find. No detector
#     can recover it, so the model has to say where it is -- and it knows,
#     because it drew the sheet.


# ---------------------------------------------------------------- data

def fix_points_mm(part) -> tuple[np.ndarray, np.ndarray]:
    """Fastening points and their axes in mm. The bins/envelope here are only the
    storage encoding of a position that is itself legitimate input."""
    from .plan_g import N_BINS
    lo = np.asarray(part.env_lo, dtype=np.float64)
    span = np.asarray(part.env_hi, dtype=np.float64) - lo
    fx = [v for v in part.vertices if v["T"] == "FIX"]
    P = np.stack([lo + (np.asarray(v["bin"], np.float64) + 0.5) / N_BINS * span
                  for v in fx])
    A = np.stack([np.asarray(v["nf"] if v["nf"] is not None else [0, 0, 1],
                             dtype=np.float64) for v in fx])
    return P, A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)


def fastener_frame(part):
    """Canonical frame from the fastening points ALONE -- no part geometry.

      origin : their centroid
      unit   : RMS distance from the centroid, scaled so two points reproduce
               exactly the separation the old two-point frame used
      axes   : principal directions of the point set, and where the points do
               not span three dimensions (they never do for two of them) the
               remaining axes come from the fastener AXES, which are input too

    Any number of fasteners from two upward, with no branch on the count: the
    principal directions of two points are one direction, of three points two,
    and the fastener axes fill whatever is left. At N=2 this is bit-for-bit the
    old frame, which the self-check below asserts.

    One point is not supported and cannot be: with a single fastener the part's
    size is not determined by the input at all, so a normalised frame has no
    scale to use. That is a property of the task, not of this function.
    """
    P, A = fix_points_mm(part)
    if len(P) < 2:
        return (P[0] if len(P) else np.zeros(3)), np.eye(3), 1.0
    o = P.mean(0)
    Q = P - o
    # unit: RMS spread, times 2 so that two points give their separation
    d = float(np.sqrt((Q ** 2).sum(1).mean())) * 2.0
    U, S, Vt = np.linalg.svd(Q, full_matrices=True)
    axes, rank = [], int((S > 1e-6 * max(S[0], 1e-12)).sum())
    for i in range(rank):
        v = Vt[i]
        # canonical sign: point the axis the way the fastener order runs
        if float(Q[-1] @ v) < float(Q[0] @ v):
            v = -v
        axes.append(v)
    # fill the remaining axes from the fastener axes, most orthogonal first
    for _ in range(3 - len(axes)):
        best, bv = None, None
        for a in A:
            r = a - sum(float(a @ e) * e for e in axes)
            n = float(np.linalg.norm(r))
            if n > 1e-6 and (best is None or n > best):
                best, bv = n, r / n
        if bv is None:                     # degenerate: take any orthogonal axis
            for alt in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                        np.array([0, 0, 1.0])):
                r = alt - sum(float(alt @ e) * e for e in axes)
                if np.linalg.norm(r) > 1e-6:
                    bv = r / np.linalg.norm(r)
                    break
        axes.append(bv)
    e1, e2 = axes[0], axes[1]
    return o, np.stack([e1, e2, np.cross(e1, e2)]), max(d, 1e-9)


def frame_demo():
    """At two fasteners the general frame must equal the old two-point one."""
    import pathlib as _pl

    from .dataset_curve import load_curve_parts

    R = _pl.Path(__file__).resolve().parents[4]
    parts = load_curve_parts(R / "runs" / "wtok_synth")[:40]
    du, da = [], []
    for p in parts:
        P, A = fix_points_mm(p)
        if len(P) != 2:
            continue
        d0 = float(np.linalg.norm(P[1] - P[0]))
        e1 = (P[1] - P[0]) / d0
        a = A[int(np.argmin(np.abs(A @ e1)))]
        e2 = a - (a @ e1) * e1
        e2 /= np.linalg.norm(e2)
        o1, R1, u1 = P.mean(0), np.stack([e1, e2, np.cross(e1, e2)]), d0
        o2, R2, u2 = fastener_frame(p)
        du.append(abs(u1 - u2))
        da.append(float(np.abs(R1 - R2).max()) + float(np.abs(o1 - o2).max()))
    print(f"{len(du)} two-fastener parts")
    print(f"  unit differs by at most   {max(du):.3e}")
    print(f"  frame differs by at most  {max(da):.3e}")
    assert max(du) < 1e-9 and max(da) < 1e-9, "the general frame moved N=2"
    # and it must stay defined for more fasteners
    import copy
    for n in (3, 4, 6):
        q = copy.deepcopy(parts[0])
        fx = [v for v in q.vertices if v["T"] == "FIX"]
        extra = []
        for i in range(n - len(fx)):
            v = dict(fx[i % len(fx)])
            v["bin"] = tuple(int(c * (0.6 + 0.1 * i)) for c in v["bin"])
            extra.append(v)
        q.vertices = fx + extra
        o, Rm, u = fastener_frame(q)
        det = float(np.linalg.det(Rm))
        assert abs(det - 1.0) < 1e-6 and u > 1e-6, (n, det, u)
        print(f"  {n} fasteners: unit {u:8.2f}  det(R) {det:.6f}  ok")
    print("ok")


def to_frame(xyz, normal, frame):
    o, R, d = frame
    return ((xyz - o) @ R.T) / d, normal @ R.T


def from_frame(xyz, normal, frame):
    o, R, d = frame
    return (xyz * d) @ R + o, normal @ R


def frame_cond_rows(part, rows: int = 0) -> np.ndarray:
    """Fastening points only: position and axis in their own frame, one row each,
    plus a final row carrying the log of the frame's scale.

    The row count follows the part instead of a constant. COND_ROWS was 4, so a
    part with four fasteners silently lost one; the model reads these rows with
    attention, which does not care how many there are.
    """
    P, A = fix_points_mm(part)
    frame = fastener_frame(part)
    Pf, Af = to_frame(P, A, frame)
    n = max(rows, len(Pf) + 1, COND_ROWS)
    out = np.zeros((n, 8), dtype=np.float32)
    for i in range(len(Pf)):
        out[i] = np.concatenate([Pf[i], Af[i], [1.0, 0.0]])
    out[n - 1] = np.concatenate([[np.log(frame[2]) / 10.0], np.zeros(6), [1.0]])
    return out


def part_cond_rows(part, vertices=None) -> np.ndarray:
    vs = vertices if vertices is not None else part.vertices
    rows = cond_features([v for v in vs if v["T"] == "FIX"], part.env_lo, part.env_hi)
    out = np.zeros((COND_ROWS, 8), dtype=np.float32)
    out[: min(COND_ROWS, len(rows))] = rows[:COND_ROWS]
    return out


def fix_xyz_norm(part, vertices=None) -> np.ndarray:
    """FIX positions in the envelope frame ([0,1]), the same frame as the cloud."""
    vs = vertices if vertices is not None else part.vertices
    from .plan_g import N_BINS
    return np.asarray([(np.asarray(v["bin"], np.float64) + 0.5) / N_BINS
                       for v in vs if v["T"] == "FIX"], dtype=np.float32)


def mirror(xyz, normal, fix, cond_rows, axis):
    """Mirror in the normalized envelope frame: position reflects, normal flips.
    The condition rows carry FIX xyz in columns 0..2 and the axis in 3..5."""
    xyz, normal, fix = xyz.copy(), normal.copy(), fix.copy()
    xyz[:, axis] = 1.0 - xyz[:, axis]
    normal[:, axis] = -normal[:, axis]
    fix[:, axis] = 1.0 - fix[:, axis]
    c = cond_rows.copy()
    is_fix = c[:, 6] > 0.5
    c[is_fix, axis] = 1.0 - c[is_fix, axis]
    c[is_fix, 3 + axis] = -c[is_fix, 3 + axis]
    return xyz, normal, fix, c


ANCHOR_R = 10.0        # mm; measured exact out to here (see fastener_disc)
ANCHOR_PER_FIX = 8


def fastener_disc(part, per_fix: int = ANCHOR_PER_FIX, r: float = ANCHOR_R,
                  rng=None, use_spec: bool = True):
    """Points the fastener itself determines, SYNTHESISED -- never read from the
    part.

    A bolt needs a flat seat normal to its axis, and the part is flat there
    exactly. Measured over 400 fastener sites:

        radius used                          tilt p95   off-plane p95
        fixed 10mm                             0.0 deg        0.002mm
        the part's own min_bearing_radius      0.0 deg        0.002mm
        1.2x that radius                      25.2 deg        0.541mm

    So the flat region ends AT the required bearing radius and not before. The
    fixed 10mm was chosen without knowing each part's requirement, and it is
    smaller than the requirement on 100% of parts (median 17.9mm, min 12.5) --
    a third of the available flat area by area.

    `min_bearing_radius_mm` is a design input, not part geometry, so using it
    keeps the disc synthetic: it is still the fastener drawn out, from the point,
    its axis, and the seat the bolt requires.
    """
    rng = rng or np.random.default_rng(0)
    P, A = fix_points_mm(part)
    if use_spec:
        from .sidecar import SPEC_KEYS, SPEC_SCALE, load_spec
        sp = load_spec(part)
        if sp is not None:
            i = SPEC_KEYS.index("min_bearing_radius_mm")
            rb = float(sp[i] / SPEC_SCALE[i])
            if rb > 0:
                r = rb
    pts, nrm = [], []
    for c, ax in zip(P, A):
        e1 = np.array([1.0, 0.0, 0.0])
        if abs(e1 @ ax) > 0.9:
            e1 = np.array([0.0, 1.0, 0.0])
        e1 = e1 - (e1 @ ax) * ax
        e1 /= max(np.linalg.norm(e1), 1e-12)
        e2 = np.cross(ax, e1)
        rad = r * np.sqrt(rng.random(per_fix))       # uniform over the area
        th = rng.random(per_fix) * 2 * np.pi
        rad[0], th[0] = 0.0, 0.0                     # keep the exact centre
        pts.append(c + rad[:, None] * (np.cos(th)[:, None] * e1
                                       + np.sin(th)[:, None] * e2))
        nrm.append(np.tile(ax, (per_fix, 1)))
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(pts), np.concatenate(nrm)


def gt_fields(part, npz) -> np.ndarray:
    """(N,2): mm distance from each surface sample to the outline and to the
    nearest bend line. These are the supervision for channels 6 and 7."""
    from scipy.spatial import cKDTree

    from .codec import realize_edge
    lo = np.asarray(npz["env_lo"], dtype=np.float64)
    span = np.maximum(np.asarray(npz["env_hi"], dtype=np.float64) - lo, 1e-9)
    xyz = npz["xyz"].astype(np.float64) * span + lo
    q = realized_q(part, part.vertices, part.edges)
    out = np.full((len(xyz), 2), 1e6, dtype=np.float64)
    for col, cls in enumerate(("outer_boundary", "bend_line")):
        polys = [poly for poly in
                 (realize_edge(q, e) for e in part.edges if e.get("cls") == cls)
                 if poly is not None and len(poly) >= 2]
        if polys:
            out[:, col] = cKDTree(np.concatenate(polys)).query(xyz)[0]
    return out.astype(np.float32)


def fps_order(xyz: np.ndarray, k: int) -> np.ndarray:
    """Farthest-point order: the first m entries are a good even cover for any
    m <= k, so one pass serves every point count.

    Evenly spread points are markedly easier to read curves off. Measured with
    the ridge model on GT clouds:
        256 random 3.03/3.72mm   256 even 2.22/3.36mm
        512 random 1.48/1.90mm   512 even 1.27/1.47mm
    and the target itself becomes deterministic -- a random subset differs every
    epoch, which is noise the model cannot fit and should not have to.
    """
    k = min(k, len(xyz))
    order = np.empty(k, dtype=np.int64)
    order[0] = 0
    dist = np.linalg.norm(xyz - xyz[0], axis=1)
    for i in range(1, k):
        j = int(dist.argmax())
        order[i] = j
        np.minimum(dist, np.linalg.norm(xyz - xyz[j], axis=1), out=dist)
    return order


class MeshDataset(Dataset):
    def __init__(self, parts, mesh_dir: pathlib.Path, n_pts: int,
                 augment: bool, base_seed: int = 0,
                 anchor_per_fix: int = ANCHOR_PER_FIX,
                 outlier_rate: float = 0.0, self_rate: float = 0.0,
                 mask_rate: float = 0.0, mask_max: float = 0.5,
                 even: bool = True):
        self.anchor_per_fix = anchor_per_fix
        self.outlier_rate = outlier_rate
        # how often the pinned rows come from the model's own output, and the
        # bank they come from: {part name -> (N, CH) array in frame units}.
        # Refreshed every few epochs; generating inside the loop would cost a
        # full 24-step sample per item.
        self.self_rate = self_rate
        self.self_bank: dict = {}
        self.even = even
        self.mask_rate = mask_rate
        self.mask_max = mask_max
        self.parts = parts
        self.dir = pathlib.Path(mesh_dir)
        self.n_pts = n_pts
        self.augment = augment
        self.base_seed = base_seed
        self.epoch = 0
        self.cache: dict = {}
        self.outside: dict = {}

    def set_epoch(self, e: int) -> None:
        self.epoch = int(e)

    def __len__(self) -> int:
        return len(self.parts)

    def load(self, part):
        if part.name not in self.cache:
            d = np.load(self.dir / f"{part.name}.npz")
            xyz = d["xyz"].astype(np.float32)
            order = (fps_order(xyz, min(len(xyz), 4 * self.n_pts))
                     if self.even else None)
            self.cache[part.name] = (xyz, d["normal"].astype(np.float32),
                                     gt_fields(part, d), order)
        return self.cache[part.name]

    def __getitem__(self, i: int) -> dict:
        p = self.parts[i]
        rng = np.random.default_rng(
            stable_seed(self.base_seed + 999983 * self.epoch, p.name))
        xyz_n, nrm, fields, order = self.load(p)
        lo = np.asarray(p.env_lo, dtype=np.float64)
        span = np.maximum(np.asarray(p.env_hi, dtype=np.float64) - lo, 1e-9)
        xyz_mm = xyz_n.astype(np.float64) * span + lo      # envelope: storage only
        frame = fastener_frame(p)
        xyz, nrm = to_frame(xyz_mm, nrm.astype(np.float64), frame)
        dp, dn = fastener_disc(p, self.anchor_per_fix, rng=rng)
        fix, fixn = to_frame(dp, dn, frame)
        cond = frame_cond_rows(p)
        if self.anchor_per_fix:
            # free slots come from outside the disc so the density near a
            # fastener matches the rest of the sheet. The disc is centred on the
            # fasteners, so this only needs the distance to those two points --
            # and it never changes, so cache it per part.
            keep_idx = self.outside.get(p.name)
            if keep_idx is None:
                centres, _ = to_frame(*fix_points_mm(p), frame)
                far = np.linalg.norm(
                    xyz[:, None, :] - centres[None, :, :], axis=-1).min(1)
                keep_idx = np.flatnonzero(far > ANCHOR_R / frame[2])
                self.outside[p.name] = keep_idx
            if len(keep_idx) > self.n_pts:
                xyz, nrm, fields = xyz[keep_idx], nrm[keep_idx], fields[keep_idx]
                if order is not None:
                    pos = -np.ones(len(xyz_n), dtype=np.int64)
                    pos[keep_idx] = np.arange(len(keep_idx))
                    order = pos[order]
                    order = order[order >= 0]
        if self.augment and rng.random() < 0.5:
            # the frame fixes e1 and e2, so only the e3 component is free to
            # mirror -- flipping the others would just redefine the frame
            xyz, nrm, fix = xyz.copy(), nrm.copy(), fix.copy()
            xyz[:, 2] *= -1.0
            nrm[:, 2] *= -1.0
            fix[:, 2] *= -1.0
            fixn = fixn.copy()
            fixn[:, 2] *= -1.0
            cond = cond.copy()
            cond[:COND_ROWS - 1, 2] *= -1.0
            cond[:COND_ROWS - 1, 5] *= -1.0
        n_fix = min(len(fix), self.n_pts)
        n_free = self.n_pts - n_fix
        take = (order[:n_free] if order is not None and len(order) >= n_free
                else rng.choice(len(xyz), n_free, replace=False))
        fld = np.clip(fields / frame[2], 0.0, FIELD_CAP)      # frame units
        anc_fld = np.clip(fields[cKD(xyz, fix[:n_fix])] / frame[2],
                          0.0, FIELD_CAP)     # anchors sit on the sheet too
        # anchors occupy the first slots; every other slot is a random surface
        # sample, so slot order carries no information the model could lean on
        pts = np.concatenate([fix[:n_fix], xyz[take]])
        nn_ = np.concatenate([fixn[:n_fix], nrm[take]])
        fl = np.concatenate([anc_fld, fld[take]])
        out = np.zeros((len(pts), 1))
        if self.outlier_rate > 0 and rng.random() < self.outlier_rate:
            # push points off the part in a random direction. The failure being
            # taught against is a cloud that overruns its own boundary, so the
            # displacement is isotropic rather than along the normal only.
            free = np.arange(n_fix, len(pts))
            k = max(1, int(len(free) * OUTLIER_FRAC))
            sel = rng.choice(free, k, replace=False)
            # NOT np.diff over consecutive rows: farthest-point order puts
            # consecutive samples as far apart as possible, which measured
            # 14-19x the real spacing and ~30% of the part, so injected
            # outliers were flung a whole part-width away and the surface
            # probe went from 9mm to 39mm.
            free_pts = pts[n_fix:]
            pitch = float(np.median(
                cKDTree(free_pts).query(free_pts, k=2)[0][:, 1])) or 1e-3
            v = rng.normal(size=(k, 3))
            v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
            pts = pts.copy()
            pts[sel] += v * (pitch * rng.uniform(*OUTLIER_PITCH, size=(k, 1)))
            out[sel] = 1.0
        # Masked training: pin a random extra set of surface points as "known"
        # and let the model infer the rest. Pinning is already how anchors work,
        # so a constraint point and a fastening point are the same mechanism --
        # this keeps the model usable with any set of known points at inference.
        n_anchor = n_fix        # rows 0..n_anchor-1 are the fastening discs
        if self.mask_rate > 0 and rng.random() < self.mask_rate:
            free = self.n_pts - n_fix
            k = int(rng.integers(1, max(2, int(free * self.mask_max))))
            sel = n_fix + rng.permutation(free)[:k]
            order = np.concatenate([np.arange(n_fix), sel,
                                    np.setdiff1d(np.arange(n_fix, self.n_pts), sel)])
            pts, nn_, fl, out = pts[order], nn_[order], fl[order], out[order]
            n_fix += k
            # Scheduled sampling: sometimes the pinned rows are replaced by the
            # model's OWN earlier output for this part. Only the CONDITIONING
            # moves -- the target stays the true cloud, because training on its
            # own errors is how a model drifts.
            #
            # Confirming its own points measured worth nothing (8.51 -> 8.68mm,
            # against 3.92mm for true points at the same count): the model was
            # trained to treat a pinned row as exact, so pinning a row that is
            # 5mm out forces a shape consistent with that error. This is what
            # teaches it to hold them loosely.
            bank = self.self_bank.get(p.name)
            if bank is not None and rng.random() < self.self_rate:
                q = cKDTree(bank[:, :3]).query(pts[n_anchor:n_fix])[1]
                pts[n_anchor:n_fix] = bank[q, :3]
                nn_[n_anchor:n_fix] = bank[q, 3:6]
                fl[n_anchor:n_fix] = bank[q, 6:6 + fl.shape[1]]
        return {"x": torch.from_numpy(
                    np.concatenate([pts, nn_, fl, out], axis=1).astype(np.float32)),
                "cond": torch.from_numpy(cond),
                "n_fix": n_fix,
                "n_anchor": n_anchor}


def cKD(xyz, q):
    if len(q) == 0:
        return np.zeros(0, dtype=np.int64)
    d = np.linalg.norm(xyz[None, :, :] - q[:, None, :], axis=-1)
    return d.argmin(axis=1)


# ---------------------------------------------------------------- model

class PointFlow(nn.Module):
    """v(x_t, t | condition) over a permutation-equivariant set of points."""

    def __init__(self, dim=256, layers=8, heads=8, rel_attn=True, ch: int = CH):
        super().__init__()
        # ch is stored so pre-field checkpoints (6 channels) still load
        self.dim, self.heads, self.ch = dim, heads, ch
        self.inp = nn.Linear(ch, dim)
        self.cond_proj = nn.Linear(8, dim)
        self.null_cond = nn.Parameter(torch.zeros(COND_ROWS, dim))
        self.seg = nn.Parameter(torch.zeros(2, dim))
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, ch)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.rel = RelBias(heads) if rel_attn else None

    def forward(self, x, t, cond, drop=None):
        B, N, _ = x.shape
        c = self.cond_proj(cond)
        if drop is not None:
            c = torch.where(drop[:, None, None], self.null_cond, c)
        c = c + self.seg[0]
        h = torch.cat([c, self.inp(x) + self.seg[1]], dim=1)
        g = self.t_mlp(timestep_embedding(t, self.dim)) + c.mean(1)
        bias = None
        if self.rel is not None:
            S = h.shape[1]
            m = torch.zeros(B, self.heads, S, S, device=x.device, dtype=h.dtype)
            m[:, :, COND_ROWS:, COND_ROWS:] = self.rel(x[..., :3]).to(h.dtype)
            bias = m.reshape(B * self.heads, S, S)
        for blk in self.blocks:
            h = blk(h, g, bias)
        return self.head(self.norm(h[:, COND_ROWS:]))


def anchor_keep(n_fix: torch.Tensor, N: int) -> torch.Tensor:
    return torch.arange(N, device=n_fix.device)[None] < n_fix[:, None]


def repulsion(pred_x1, true_x1, live, sigma_pitch: float = 2.0):
    """Match how evenly the generated points cover the sheet.

    The first version of this punished only pairs closer than the target
    spacing. It fixed duplicates (2.7% -> 1.9% of points) and left the spread
    exactly where it was (0.580 -> 0.582): a cloud with a dense patch beside a
    hole has no pair that is too close, because every point still has a
    neighbour at a reasonable distance. The wrong quantity was penalised.

    Measured where the unevenness lives, as the spread of the local point count
    at several radii (generated over true):

        0.5x pitch   duplicates, since fixed
        1.0x pitch   1.13x
        2.0x pitch   1.90x   <- here
        4.0x pitch   1.37x
        8.0x pitch   1.05x   the global layout is already fine

    So the term acts on the density within a couple of spacings, estimated with
    a Gaussian kernel because counting points is not differentiable, and matched
    to the true cloud of the same batch rather than to a constant.
    """
    N = pred_x1.shape[1]
    eye = torch.eye(N, device=pred_x1.device)[None] * 1e9
    pitch = (torch.cdist(true_x1, true_x1) + eye).min(-1).values.median(-1).values
    s = (sigma_pitch * pitch).clamp(min=1e-6)[:, None, None]

    def density(x):
        return torch.exp(-(torch.cdist(x, x) / s) ** 2).sum(-1) - 1.0

    def rel_spread(rho, m):
        mu = (rho * m).sum(-1) / m.sum(-1).clamp(min=1)
        var = ((rho - mu[:, None]) ** 2 * m).sum(-1) / m.sum(-1).clamp(min=1)
        return var.sqrt() / mu.clamp(min=1e-6)

    ones = torch.ones_like(live)
    return ((rel_spread(density(pred_x1), live)
             - rel_spread(density(true_x1), ones).detach()) ** 2).mean()


def repulsion_demo():
    """The density term, checked without needing a model or data."""
    n = 64
    g = torch.stack(torch.meshgrid(torch.arange(8.), torch.arange(8.),
                                   indexing="ij"), -1).reshape(-1, 2)
    even = (torch.cat([g, torch.zeros(n, 1)], 1)[None].repeat(2, 1, 1) * 0.1)
    live = torch.ones(2, n)
    clumped = even.clone()
    a = even[:, 0:1, :]
    clumped[:, :16] = a + (even[:, :16] - a) * 0.15      # a clump and a hole
    r_even, r_bad = float(repulsion(even, even, live)),         float(repulsion(clumped, even, live))
    assert r_even < 1e-6 < r_bad, (r_even, r_bad)
    x = clumped.clone().requires_grad_(True)
    repulsion(x, even, live).backward()
    c = clumped[0, :16].mean(0)
    dot = float(((-x.grad[0, :16]) * (clumped[0, :16] - c)).sum())
    assert dot > 0, "descent does not expand the clump"
    print(f"repulsion ok: even {r_even:.6f}, clumped {r_bad:.6f}, "
          f"gradient expands the clump (dot {dot:+.5f})")


def flow_loss(model, batch, anchor: bool, p_drop: float, w_normal: float = 0.2,
              w_field: float = 0.5, w_repel: float = 0.0):
    # w_normal is the knob 施策4 turns. Bend lines are read from a change of
    # orientation across a point, and the generated normals are 16.3 degrees out
    # -- while this weight is a fifth of the position weight. The quantity the
    # bend classifier depends on most is the one trained least.
    x1 = batch["x"]          # already in the fastener frame: coords ~ +-0.85
    x0 = torch.randn_like(x1)
    B, N, _ = x1.shape
    t = torch.rand(B, device=x1.device)
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    keep = anchor_keep(batch["n_fix"], N)
    # A fastening disc pins position and normal only -- whether it sits on an
    # outline is exactly what we are asking. A point committed by an earlier
    # round of the loop is different: its class is the reason it was committed,
    # so its fields are pinned too. Without this the loop can tell the
    # generator where a point is but not that it is an outline point.
    n_anc = batch.get("n_anchor")
    keep_cls = (keep & ~anchor_keep(n_anc, N)) if n_anc is not None         else torch.zeros_like(keep)
    if anchor:
        pinned = torch.where(keep[..., None], x1, xt)
        fld = torch.where(keep_cls[..., None], x1[..., 6:], xt[..., 6:])
        xt = torch.cat([pinned[..., :6], fld], dim=-1)
    drop = (torch.rand(B, device=x1.device) < p_drop) if (p_drop > 0 and model.training) else None
    v = model(xt, t, batch["cond"], drop)
    target = x1 - x0
    err = (v - target) ** 2
    w = torch.cat([torch.ones(3, device=x1.device),
                   torch.full((3,), w_normal, device=x1.device),
                   torch.full((CH - 6,), w_field, device=x1.device)])
    err = (err * w).mean(-1)
    if anchor:
        geo = ((v[..., :6] - target[..., :6]) ** 2 * w[:6]).mean(-1)
        fldl = ((v[..., 6:] - target[..., 6:]) ** 2 * w[6:]).mean(-1)
        live = (~keep).float()
        live_f = (~keep_cls).float()
        out = ((geo * live).sum() / live.sum().clamp(min=1)
               + (fldl * live_f).sum() / live_f.sum().clamp(min=1))
        if w_repel > 0:
            # Where the sample would land if the flow were followed from here.
            #
            # This term is only defined once the model already puts points
            # roughly on the sheet. It is a RELATIVE spread -- standard
            # deviation over mean -- of a Gaussian density whose width is a
            # couple of point spacings. From a random initialisation the
            # predicted points scatter past the part itself, every one of them
            # is isolated at that width, the mean density goes to zero and the
            # ratio explodes: measured 39.6 against 0.017 for a trained model,
            # a factor of 2300, and training diverged (1.15 -> 0.72 without the
            # term, 2.80 -> 3.59 with it). Gating by t does not rescue it, since
            # early in training every t is in that regime.
            #
            # So it is used only when fine-tuning from trained weights, which is
            # how every generator run here starts. stage_smoke, which trains
            # from scratch, switches it off.
            pred = xt[..., :3] + (1 - t[:, None, None]) * v[..., :3]
            out = out + w_repel * repulsion(pred, x1[..., :3], live)
        return out
    return err.mean()


@torch.no_grad()
def sample(model, cond, fix_xyz, steps=48, gen=None, scale=1.0, n_pts=256,
           fix_fld=None):
    dev = cond.device
    B = cond.shape[0]
    x = torch.randn(B, n_pts, model.ch, device=dev, generator=gen)
    n_fix = fix_xyz.shape[1] if fix_xyz is not None else 0
    def pin(z):
        z[:, :n_fix, :6] = fix_xyz
        if fix_fld is not None:            # NaN = class unknown, leave it free
            z[:, :n_fix, 6:] = torch.where(torch.isnan(fix_fld),
                                           z[:, :n_fix, 6:], fix_fld)
        return z
    if n_fix:
        x = pin(x)
    dt = 1.0 / steps
    for k in range(steps):
        t = torch.full((B,), k * dt, device=dev)
        if scale > 1.0:
            keep = torch.zeros(B, dtype=torch.bool, device=dev)
            v = model(x, t, cond, ~keep)
            v = v + scale * (model(x, t, cond, keep) - v)
        else:
            v = model(x, t, cond)
        x = x + v * dt
        if n_fix:
            x = pin(x)
    return x


@torch.no_grad()
def refresh_self_bank(model, dataset, parts, device, steps=48, cfg=2.0,
                      n_pts=512, limit=0, seed=0):
    """Fill the dataset's bank with the model's own clouds, in frame units.

    Generating inside the training loop would cost a full sample per item, so a
    bank is built every few epochs and reused. It goes stale as the model
    improves, which is the point of refreshing rather than doing it once.
    """
    was = model.training
    model.eval()
    # A rotating window rather than the same head of the list: every entry is
    # freshly generated, and over len(parts)/limit epochs the whole training set
    # gets its turn at being self-conditioned. Cleared first so nothing stale
    # survives.
    if limit and limit < len(parts):
        off = (seed * limit) % len(parts)
        sel = [parts[(off + i) % len(parts)] for i in range(limit)]
    else:
        sel = parts
    dataset.self_bank.clear()
    for i, p in enumerate(sel):
        xyz, nrm, fld = generate(model, p, device, steps, seed + i, cfg, n_pts,
                                 True, per_fix=1)
        f, n = to_frame(xyz, nrm, fastener_frame(p))
        fl = fld.copy()
        fl[:, :2] /= fastener_frame(p)[2]           # fields come back in mm
        dataset.self_bank[p.name] = np.concatenate(
            [f, n, fl], axis=1).astype(np.float64)
    model.train(was)
    return len(sel)


def to_world(x, part):
    """model output (fastener frame) -> (points mm, unit normals, fields mm).
    The envelope is never consulted: the frame comes from the fastening points
    alone."""
    frame = fastener_frame(part)
    xyz = x[..., :3].cpu().numpy().astype(np.float64)
    nrm = x[..., 3:6].cpu().numpy().astype(np.float64)
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-9)
    world, normal = from_frame(xyz, nrm, frame)
    if x.shape[-1] <= 6:                     # a pre-field model
        return world, normal, None
    raw = x[..., 6:].cpu().numpy().astype(np.float64)
    fields = np.concatenate(
        [np.clip(raw[..., :2], 0.0, FIELD_CAP) * frame[2], raw[..., 2:]],
        axis=-1)                      # last column stays the raw 0/1 flag
    return world, normal, fields


# ---------------------------------------------------------------- stages

def load_model(path, device):
    """Rebuild a PointFlow from a checkpoint, whatever channel count it used."""
    ck = torch.load(path, map_location=device, weights_only=False)
    a = argparse.Namespace(**ck["args"])
    ch = ck["model"]["inp.weight"].shape[1]
    model = PointFlow(a.dim, a.layers, a.heads, bool(a.rel_attn), ch=ch).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, a, ck["epoch"], ch


def build_splits(args):
    parts = load_curve_parts(pathlib.Path(args.wtok))
    have = {f.stem for f in (pathlib.Path(args.dataset) / "parts").glob("*.npz")}
    parts = [p for p in parts if p.name in have]
    val = set(json.loads(pathlib.Path(args.val_list).read_text(encoding="utf-8")))
    return ([p for p in parts if p.name not in val],
            [p for p in parts if p.name in val])


def to_device(b, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


def stage_train(args, out: pathlib.Path) -> None:
    train_parts, val_parts = build_splits(args)
    print(f"parts: train {len(train_parts)} val {len(val_parts)}  N={args.points}")
    md = pathlib.Path(args.dataset) / "parts"
    tl = DataLoader(MeshDataset(train_parts, md, args.points, True,
                                anchor_per_fix=args.anchor_per_fix,
                                outlier_rate=args.outlier_rate,
                                mask_rate=args.mask_rate, even=bool(args.even)),
                    batch_size=args.batch_size, shuffle=True, drop_last=True)
    vl = DataLoader(MeshDataset(val_parts, md, args.points, False, 555,
                                anchor_per_fix=args.anchor_per_fix,
                                even=bool(args.even)),
                    batch_size=args.batch_size)
    model = PointFlow(args.dim, args.layers, args.heads, bool(args.rel_attn)).to(args.device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, args.lr * 0.05)
    hist_path = out / "history.json"
    history, best, start = [], float("inf"), 1
    last = out / "last.pt"
    if args.resume and last.exists():
        ck = torch.load(last, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["epoch"] + 1
        if hist_path.exists():
            history = [r for r in json.loads(hist_path.read_text()) if r["epoch"] < start]
            best = min((r["surf_mm"] for r in history if "surf_mm" in r),
                       default=float("inf"))
        for _ in range(start - 1):
            sched.step()
        print(f"resumed at {start} (best {best:.5f})", flush=True)
    # the loss is a conditional variance and plateaus long before the geometry
    # does, so probe the actual surface error during training as well
    probe_gt = {}
    for p in val_parts[: args.probe_parts]:
        d = np.load(md / f"{p.name}.npz")
        probe_gt[p.name] = d["xyz"] * np.maximum(d["env_hi"] - d["env_lo"], 1e-9)             + d["env_lo"]
    t_start = time.time()
    for epoch in range(start, args.epochs + 1):
        # started before the bank refresh, which costs a full sample per part
        # and is the dominant cost when it runs -- timing only the training
        # portion reported 55s for epochs that actually took 27 minutes
        t0 = time.time()
        tl.dataset.set_epoch(epoch)
        if args.self_rate > 0 and epoch >= args.self_warmup:
            if (epoch - args.self_warmup) % args.self_every == 0:
                k = refresh_self_bank(model, tl.dataset, train_parts, args.device,
                                      steps=args.self_steps, n_pts=args.points,
                                      limit=args.self_bank, seed=epoch)
                print(f"  self-bank refreshed: {k} parts", flush=True)
            # ramp in, never to 1.0: the model still has to learn to exploit a
            # clean pin when it gets one
            span = max(args.epochs - args.self_warmup, 1)
            tl.dataset.self_rate = args.self_rate * min(
                1.0, (epoch - args.self_warmup) / (0.5 * span))
        model.train()
        tot, n = 0.0, 0
        for b in tl:
            loss = flow_loss(model, to_device(b, args.device), args.anchor,
                             args.cfg_drop, w_normal=args.w_normal,
                             w_repel=args.w_repel)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            n += 1
        sched.step()
        row = {"epoch": epoch, "train": tot / max(n, 1),
               "seconds": round(time.time() - t0, 1)}
        if epoch % args.val_every == 0 or epoch == args.epochs:
            model.eval()
            # flow_loss draws t and x0 at random, so an unseeded val score moves
            # by more than the training signal does and picks best.pt by luck
            with torch.no_grad():
                torch.manual_seed(1234)
                row["val"] = float(np.mean(
                    [float(flow_loss(model, to_device(b, args.device), args.anchor,
                                     0.0, w_normal=args.w_normal,
                                     w_repel=args.w_repel))
                     for b in vl]))
                torch.manual_seed(epoch)
            ck = {"model": model.state_dict(), "opt": opt.state_dict(),
                  "args": vars(args), "epoch": epoch}
            safe_save(ck, last)
        history.append(row)
        hist_path.write_text(json.dumps(history), encoding="utf-8")
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            model.eval()
            # select on geometry, never on the loss: the flow-matching loss is a
            # conditional variance whose minimum is noise, and picking best.pt by
            # it once froze a run at epoch 30 while the surface kept improving
            # through epoch 60
            row["surf_mm"] = float(np.median([
                chamfer_mm(generate(model, p, args.device, args.steps, 7,
                                    args.cfg_scale, args.points, args.anchor,
                                    per_fix=args.anchor_per_fix)[0],
                           probe_gt[p.name])
                for p in val_parts[: args.probe_parts]]))
        msg = f"epoch {epoch}: loss {row['train']:.5f} ({row['seconds']}s)"
        if "val" in row:
            msg += f"  val {row['val']:.5f}"
        if "surf_mm" in row:
            msg += f"  | surf {row['surf_mm']:.1f}mm"
            if row["surf_mm"] < best:
                best = row["surf_mm"]
                safe_save({"model": model.state_dict(), "opt": opt.state_dict(),
                           "args": vars(args), "epoch": epoch}, out / "best.pt")
                msg += " *best*"
        print(msg, flush=True)
        if (time.time() - t_start) / 3600.0 > args.max_hours:
            print(f"stopping cleanly at the {args.max_hours}h budget", flush=True)
            break


@torch.no_grad()
def generate(model, part, device, steps, seed, scale, n_pts, anchor=True,
             per_fix=ANCHOR_PER_FIX, known=None):
    gen = torch.Generator(device=device).manual_seed(seed)
    cond = torch.from_numpy(frame_cond_rows(part))[None].to(device)
    fx = None
    rows = []
    if anchor:
        dp, dn = fastener_disc(part, per_fix, rng=np.random.default_rng(seed))
        f, fn = to_frame(dp, dn, fastener_frame(part))
        rows.append(np.concatenate([f, fn], axis=1))
    nf = model.ch - 6                    # this model's field width, not CH
    flds = [np.full((len(rows[0]), nf), np.nan)] if rows else []
    if known is not None and len(known):
        # any point the caller already knows, in mm with its normal, and -- if
        # the caller committed it because of what it is -- its class fields
        frame = fastener_frame(part)
        kf, kn = to_frame(known[:, :3], known[:, 3:6], frame)
        rows.append(np.concatenate([kf, kn], axis=1))
        f = np.full((len(known), nf), np.nan)
        if known.shape[1] >= 6 + nf:
            f = np.concatenate(
                [np.clip(known[:, 6:8] / frame[2], 0.0, FIELD_CAP),
                 known[:, 8:6 + nf]], axis=1)
        flds.append(f)
    ff = None
    if rows:
        fx = torch.from_numpy(np.concatenate(rows).astype(np.float32))[None].to(device)
        ff = torch.from_numpy(np.concatenate(flds).astype(np.float32))[None].to(device)
    x = sample(model, cond, fx, steps, gen, scale, n_pts, fix_fld=ff)
    return to_world(x[0], part)


def stage_eval(args, out: pathlib.Path) -> None:
    from .evaluate_curve2 import class_points, one_way
    _, val_parts = build_splits(args)
    model, a, _, _ = load_model(out / "best.pt", args.device)
    md = pathlib.Path(args.dataset) / "parts"
    rows = []
    for p in val_parts[: args.eval_parts]:
        d = np.load(md / f"{p.name}.npz")
        lo, hi = d["env_lo"], d["env_hi"]
        gt_surf = d["xyz"] * np.maximum(hi - lo, 1e-9) + lo
        gt_wire = realize_points(realized_q(p, p.vertices, p.edges))
        fix_mm = [bin_center(realized_q(p, p.vertices, p.edges), v)
                  for v in p.vertices if v["T"] == "FIX"]
        xyz, nrm, fld = generate(model, p, args.device, args.steps, 7,
                                 args.cfg_scale, a.points, args.anchor,
                                 per_fix=getattr(a, "anchor_per_fix", ANCHOR_PER_FIX))
        if args.recon:
            # rebuild the sheet, then read the rim and the bends off it. Labelling
            # the points directly cannot work: the curves are measure-zero in the
            # surface, so a uniform sample lands on each one about once.
            from .surface import extract_features
            # the generated fields say where the curves are; geometry only
            # provides the surface they live on
            bp, cp, _ = extract_features(xyz, nrm, fld)
        else:
            f = extract(xyz, nrm)
            bp, cp = xyz[f["boundary"]], xyz[f["crease"]]
        wire = np.concatenate([x for x in (bp, cp) if len(x)]) \
            if len(bp) or len(cp) else np.zeros((0, 3))
        row = {"part": p.name,
               "surface_chamfer_mm": chamfer_mm(xyz, gt_surf),
               "n_boundary": len(bp), "n_crease": len(cp),
               "fix_err_mm": float(np.mean(
                   [np.min(np.linalg.norm(xyz - q, axis=1)) for q in fix_mm]))}
        if len(wire):
            row["wire_chamfer_mm"] = chamfer_mm(wire, gt_wire)
            row["outline_mm"] = one_way(class_points(p, {"outer_boundary"}), bp)[0] \
                if len(bp) else float("nan")
            row["bend_mm"] = one_way(class_points(p, {"bend_line"}), cp)[0] \
                if len(cp) else float("nan")
            row["spurious_mm"] = one_way(wire, gt_wire)[0]
        rows.append(row)
        print(f"  {p.name[-8:]}: surf {row['surface_chamfer_mm']:.1f}mm  "
              f"wire {row.get('wire_chamfer_mm', float('nan')):.1f}mm  "
              f"fix {row['fix_err_mm']:.2f}mm  b/c={len(bp)}/{len(cp)}", flush=True)

    def med(k):
        v = [r[k] for r in rows if k in r and np.isfinite(r[k])]
        return float(np.median(v)) if v else float("nan")

    summary = {"parts": len(rows), "points": a.points, "steps": args.steps,
               "cfg_scale": args.cfg_scale,
               **{k: round(med(k), 2) for k in
                  ("surface_chamfer_mm", "wire_chamfer_mm", "outline_mm",
                   "bend_mm", "spurious_mm", "fix_err_mm")}}
    print(json.dumps(summary, indent=1))
    (out / "eval.json").write_text(json.dumps({"summary": summary, "rows": rows},
                                              indent=1), encoding="utf-8")


def check_no_envelope_leak(parts) -> None:
    """Nothing but the fastening points may reach the model or the decode.

    The envelope is how a FIX position is *stored* (a quantised bin inside it),
    so the test re-quantises into a different box that keeps every fastening
    point at the same millimetre position. Condition, anchors and decode must
    then be bit-for-bit unmoved. This is the check that would have caught the
    ground-truth bounding box serving as the coordinate frame.
    """
    from .plan_g import N_BINS
    worst = 0.0
    for p in parts[:6]:
        before_cond = frame_cond_rows(p)
        before_fix, _ = to_frame(*fix_points_mm(p), fastener_frame(p))
        x = torch.randn(8, CH)
        before_world = to_world(x, p)[0]

        lo0, hi0 = np.array(p.env_lo), np.array(p.env_hi)
        mm = {id(v): lo0 + (np.asarray(v["bin"], float) + 0.5) / N_BINS * (hi0 - lo0)
              for v in p.vertices if v["T"] == "FIX"}
        pad = 0.37 * (hi0 - lo0) + 5.0
        lo1, hi1 = lo0 - pad, hi0 + pad
        old_bins = {id(v): v["bin"] for v in p.vertices if v["T"] == "FIX"}
        p.env_lo, p.env_hi = lo1, hi1
        for v in p.vertices:
            if v["T"] == "FIX":
                b = (mm[id(v)] - lo1) / (hi1 - lo1) * N_BINS - 0.5
                v["bin"] = tuple(int(round(c)) for c in b)
        try:
            assert np.allclose(before_cond, frame_cond_rows(p), atol=1e-3),                 f"{p.name}: condition moved with the envelope"
            after_fix, _ = to_frame(*fix_points_mm(p), fastener_frame(p))
            assert np.allclose(before_fix, after_fix, atol=1e-3),                 f"{p.name}: anchor coords moved with the envelope"
            # re-quantising into a 1.7x looser box moves each fastening point by
            # up to a bin, which propagates through the frame; a real leak moved
            # the decode by ~100mm, so 1mm separates the two cases cleanly
            drift = float(np.abs(before_world - to_world(x, p)[0]).max())
            worst = max(worst, drift)
            assert drift < 1.0,                 f"{p.name}: decode moved {drift:.2f}mm with the envelope"
        finally:
            p.env_lo, p.env_hi = lo0, hi0
            for v in p.vertices:
                if v["T"] == "FIX":
                    v["bin"] = old_bins[id(v)]
    # the disc must come from the fastener alone: scramble every non-FIX vertex
    # and it has to be unchanged
    for p in parts[:3]:
        before = fastener_disc(p, rng=np.random.default_rng(0))[0]
        saved = [(v, v["bin"]) for v in p.vertices if v["T"] != "FIX"]
        for v, _ in saved:
            v["bin"] = tuple(int(b) ^ 0x2AAA for b in v["bin"])
        try:
            after = fastener_disc(p, rng=np.random.default_rng(0))[0]
            assert np.allclose(before, after),                 f"{p.name}: the anchor disc depends on the part, not the fastener"
        finally:
            for v, b in saved:
                v["bin"] = b
    print(f"no-envelope-leak ok ({min(len(parts), 6)} parts, "
          f"worst decode drift {worst:.3f}mm = re-quantisation noise); "
          f"anchor disc independent of part geometry")


def stage_smoke(args, out: pathlib.Path) -> None:
    # This stage trains from a random initialisation, which is the one regime
    # the density term is not defined in (see flow_loss). Everything else is
    # still exercised; the term has its own check in repulsion_demo().
    if args.w_repel:
        print("  note: density term off for the smoke stage (trains from "
              "scratch; the term needs weights that already place points "
              "roughly on the sheet)")
        args.w_repel = 0.0
    train_parts, _ = build_splits(args)
    parts = train_parts[:24]
    check_no_envelope_leak(parts)
    md = pathlib.Path(args.dataset) / "parts"
    ds = MeshDataset(parts, md, args.points, True,
                     anchor_per_fix=args.anchor_per_fix)
    loader = DataLoader(ds, batch_size=4, shuffle=True, drop_last=True)
    model = PointFlow(args.dim, args.layers, args.heads, bool(args.rel_attn)).to(args.device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for epoch in range(8):
        ds.set_epoch(epoch)
        for b in loader:
            loss = flow_loss(model, to_device(b, args.device), args.anchor,
                             args.cfg_drop, w_normal=args.w_normal,
                             w_repel=args.w_repel)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
    head, tail = np.mean(losses[:5]), np.mean(losses[-5:])
    assert tail < head, "did not learn"
    print(f"loss {head:.4f} -> {tail:.4f}")

    # masked training must actually vary how much is pinned
    ds_m = MeshDataset(parts, md, args.points, True,
                       anchor_per_fix=args.anchor_per_fix, mask_rate=1.0)
    counts = set()
    for e in range(6):
        ds_m.set_epoch(e)
        counts.update(int(ds_m[i]["n_fix"]) for i in range(4))
    assert len(counts) > 1, f"masking pinned the same count every time: {counts}"
    print(f"masked training: pinned-point counts seen {sorted(counts)[:6]}...")

    p = parts[0]
    # a caller-supplied constraint point must be honoured exactly
    d = np.load(md / f"{p.name}.npz")
    span = np.maximum(d["env_hi"] - d["env_lo"], 1e-9)
    surf = d["xyz"] * span + d["env_lo"]
    pick = surf[[7, 900, 3000]]
    known = np.concatenate([pick, d["normal"][[7, 900, 3000]]], axis=1)
    kx, _, _ = generate(model, p, args.device, 8, 1, args.cfg_scale, args.points,
                        args.anchor, per_fix=args.anchor_per_fix, known=known)
    err = float(max(np.min(np.linalg.norm(kx - q, axis=1)) for q in pick))
    print(f"caller constraint points honoured to {err:.4f}mm")
    assert err < 1e-2, f"known points were not pinned ({err:.3f}mm)"

    xyz, nrm, fld = generate(model, p, args.device, 8, 1, args.cfg_scale,
                             args.points, args.anchor,
                             per_fix=args.anchor_per_fix)
    f = extract(xyz, nrm)
    assert fld.shape[1] == CH - 6, "fields missing from the generator output"
    fixq = [bin_center(realized_q(p, p.vertices, p.edges), v)
            for v in p.vertices if v["T"] == "FIX"]
    err = float(np.mean([np.min(np.linalg.norm(xyz - q, axis=1)) for q in fixq]))
    print(f"generated {len(xyz)} pts, boundary={len(f['boundary'])} "
          f"crease={len(f['crease'])}, fix_err={err:.3f}mm")
    if args.anchor:
        assert err < 1e-3, f"anchor pinning failed ({err:.4f}mm)"
    print("smoke ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("train", "eval", "smoke"), required=True)
    ap.add_argument("--dataset", required=True, help="runs/mesh_synth")
    ap.add_argument("--wtok", required=True, help="runs/wtok_synth (conditions + GT wire)")
    ap.add_argument("--val-list", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--points", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--rel-attn", type=int, default=1)
    ap.add_argument("--anchor", type=int, default=1)
    # 1 = pin just the fastening points. The 8-per-fastener disc converged
    # faster early but ended 0.76mm worse over 40 parts, so 1 is the default.
    ap.add_argument("--anchor-per-fix", type=int, default=1)
    ap.add_argument("--outlier-rate", type=float, default=0.0,
                    help="fraction of items that get off-the-part points "
                         "injected, so the outlier flag has something to learn")
    ap.add_argument("--self-rate", type=float, default=0.0,
                    help="final fraction of masked items whose pinned rows come "
                         "from the model's own output instead of the truth")
    ap.add_argument("--self-warmup", type=int, default=10,
                    help="epoch to start scheduled sampling")
    ap.add_argument("--self-every", type=int, default=10,
                    help="epochs between regenerating the bank")
    ap.add_argument("--self-steps", type=int, default=48,
                    help="sampling steps for the bank clouds; fewer is cheaper "
                         "but moves them away from what inference produces")
    ap.add_argument("--self-bank", type=int, default=600,
                    help="parts in the bank (0 = all); a sample each is costly")
    ap.add_argument("--mask-rate", type=float, default=0.5,
                    help="fraction of training items that also pin a random set "
                         "of known surface points (masked / inpainting training)")
    ap.add_argument("--w-normal", type=float, default=0.2,
                    help="loss weight on the normals (position is 1.0)")
    ap.add_argument("--w-repel", type=float, default=0.0,
                    help="weight on the term that keeps generated points from "
                         "landing on top of each other")
    ap.add_argument("--cfg-drop", type=float, default=0.1)
    # swept at epoch 65: wire 14.3 / 13.2 / 12.2 / 13.8 / 15.0mm at scale
    # 1.0 / 1.5 / 2.0 / 3.0 / 4.0, and span ratio 0.92 -> 1.06 across it.
    # Without guidance the model hedges toward the mean and the part comes
    # out too small, which is exactly where the outline metric was losing.
    ap.add_argument("--cfg-scale", type=float, default=2.0)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--sample-every", type=int, default=20)
    ap.add_argument("--probe-parts", type=int, default=6)
    ap.add_argument("--eval-parts", type=int, default=40)
    ap.add_argument("--even", type=int, default=1,
                    help="draw the target points by farthest-point order "
                         "instead of at random")
    ap.add_argument("--recon", type=int, default=1,
                    help="rebuild the sheet before reading off the curves; "
                         "0 labels the raw points instead")
    ap.add_argument("--max-hours", type=float, default=0.55)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    {"train": stage_train, "eval": stage_eval, "smoke": stage_smoke}[args.stage](args, out)


if __name__ == "__main__":
    main()

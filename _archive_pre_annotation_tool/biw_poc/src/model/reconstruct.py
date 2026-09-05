"""
reconstruct.py -- Stage C: deterministic geometry reconstruction from a
trained Stage B VecSet Autoencoder (CLAUDE.md section 13, Round 6).

Pipeline (per part)
--------------------
  trained model + input point cloud (noisy *_dataset.h5 "points"/"normals",
  or any point cloud sampled the same way)
    -> [C-1] encode -> Z[128, token_dim]                       (Stage B encoder)
    -> [C-2] evaluate (udf, grad) on a dense grid in normalized [-1,1]^3
    -> [C-3] gradient-projection: grid points with small UDF are pulled onto
             the zero level set via x_new = x + udf(x) * grad(x), iterated a
             few times (NOT marching cubes -- a learned UDF has no sign, so
             marching cubes would extract a doubled/offset shell instead of
             the true zero-level surface; this is the documented reason a
             UDF was chosen at all, see CLAUDE.md R6-01)
    -> [C-4] surface reconstruction from the projected point cloud via
             pyvista's reconstruct_surface() (VTK / Hoppe's algorithm --
             works on unorganized points, no explicit normals required)
             -> mid_surface_reconstructed.ply
    -> [C-5] denormalize back to world mm (R7-02: invert the dataset's
             per-part center/scale normalization)
    -> [C-6] +-thickness/2 offset along the reconstructed mesh's own vertex
             normals -> two open shell surfaces, merged (flipped inner
             normals) into a single mesh -> reconstructed_solid.ply

Scope note (R6-05)
--------------------
Output is mesh-only, NOT a parametric B-Rep/STEP feature tree. The Stage
C-6 "solid" is an OPEN double-shell approximation: outer and inner offset
surfaces are concatenated but their boundary loops are NOT capped/stitched
into a watertight solid (boundary stitching is a substantially harder
geometric problem, deliberately out of scope for this PoC -- see
CLAUDE.md's technical-debt convention). This is sufficient for visual /
dimensional comparison against the GT mid-surface but is not
print-/CAM-ready.

Usage
-----
    python reconstruct.py --ckpt checkpoints/best.pt --data body002_filled_dataset.h5 \\
        --thickness-mm 1.37 --grid-res 96 --out body002_reconstructed
"""

import argparse
import functools
import pathlib
import time

import h5py
import numpy as np
import pyvista as pv
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree

from vecset_ae import VecSetAE

print = functools.partial(print, flush=True)


class InsufficientEvidenceError(RuntimeError):
    """Raised when deterministic evidence gates cannot support reconstruction."""


def select_candidate_pool(gate_idx: np.ndarray, grid_count: int, minimum_required: int) -> np.ndarray:
    """Fail closed when evidence gates leave too few candidate points."""
    gate_idx = np.asarray(gate_idx, dtype=np.int64)
    if len(gate_idx) < minimum_required:
        raise InsufficientEvidenceError(
            f"Only {len(gate_idx)} grid points pass required evidence gates; "
            f"minimum is {minimum_required}. Ungated fallback is prohibited."
        )
    if np.any(gate_idx < 0) or np.any(gate_idx >= grid_count):
        raise ValueError("gate_idx contains an out-of-range grid index")
    return gate_idx


def load_model(ckpt_path: pathlib.Path, device: str) -> VecSetAE:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = VecSetAE(
        token_dim=cfg.get("token_dim", 384),
        n_tokens=cfg.get("n_tokens", 256),
        k_neighbors=cfg.get("k_neighbors", 32),
        n_latents=cfg.get("n_latents", 128),
        enc_layers=cfg.get("enc_layers", 6),
        dec_layers=cfg.get("dec_layers", 2),
        n_heads=cfg.get("n_heads", 6),
        n_freqs=cfg.get("n_freqs", 8),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded checkpoint: {ckpt_path} (epoch {ckpt.get('epoch')})")
    return model


def load_points_from_h5(h5_path: pathlib.Path):
    with h5py.File(h5_path, "r") as f:
        points = f["points"][:]
        normals = f["normals"][:]
        thickness_mm = float(f.attrs.get("thickness_mm", 0.0))
    return points, normals, thickness_mm


def normalize_points(points: np.ndarray):
    """Same convention as dataset.py R7-02: center + isotropic rescale."""
    center = points.mean(axis=0).astype(np.float32)
    extent = float(np.abs(points - center).max())
    scale = extent * 1.05 + 1e-6
    return (points - center) / scale, center, scale


@torch.no_grad()
def encode_points(model, points_norm, normals, device):
    points_t = torch.from_numpy(points_norm).float().unsqueeze(0).to(device)
    normals_t = torch.from_numpy(normals).float().unsqueeze(0).to(device)
    return model.encode(points_t, normals_t)            # [1, n_latents, dim]


@torch.no_grad()
def eval_grid(model, z, grid_res: int, device, points_norm: np.ndarray,
              chunk: int = 100_000, bbox_pad_frac: float = 0.15):
    """
    Evaluate (udf, grad) on a dense grid covering the ACTUAL input point
    cloud's bounding box (in normalized coordinates), padded by
    bbox_pad_frac of each axis's extent -- NOT the full isotropic
    [-1,1]^3 cube spanned by R7-02's per-part normalization.

    R7-07: R7-02's per-part normalization rescales by the part's LONGEST
    axis extent and applies that scale isotropically to all 3 axes, so an
    anisotropic part (e.g. a long, thin sheet) leaves large genuinely-empty
    regions of the normalized cube far outside the actual geometry. The
    network receives no meaningful supervision there, and was found to
    predict spurious near-zero "ghost" UDF minima scattered through that
    empty volume -- confirmed empirically: evaluating the full [-1,1]^3
    cube, 86% of the rank-selected near-surface candidates (R7-06) were
    >0.05 normalized units (~20mm+) from ANY real input point (median
    ~0.58 normalized units, ~220mm), producing a reconstructed mesh whose
    mean distance to the GT mid-surface was 328mm. Restricting the
    evaluation grid to the input cloud's own (padded) bbox sidesteps the
    extrapolation region entirely -- no retraining needed (user-approved
    fix direction over retraining with a softer R7-03 clamp).
    Returns grid_xyz [G,3], udf [G], grad [G,3].
    """
    lo = points_norm.min(axis=0)
    hi = points_norm.max(axis=0)
    pad = (hi - lo) * bbox_pad_frac
    lo = lo - pad
    hi = hi + pad

    axes = [np.linspace(lo[d], hi[d], grid_res, dtype=np.float32) for d in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    grid_xyz = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)   # [G,3]

    udf_out, grad_out, conf_out = decode_chunked(model, z, grid_xyz, device, chunk)
    return grid_xyz, udf_out, grad_out, conf_out


@torch.no_grad()
def decode_chunked(model, z, xyz: np.ndarray, device, chunk: int = 100_000):
    """Batched model.decode() over an arbitrary xyz array. Returns udf [N], grad [N,3], conf [N] (R7-10)."""
    udf_out = np.empty(len(xyz), dtype=np.float32)
    grad_out = np.empty((len(xyz), 3), dtype=np.float32)
    conf_out = np.empty(len(xyz), dtype=np.float32)
    for i in range(0, len(xyz), chunk):
        q = torch.from_numpy(xyz[i:i + chunk]).float().unsqueeze(0).to(device)
        udf, grad, conf = model.decode(q, z)
        udf_out[i:i + chunk] = udf[0].cpu().numpy()
        grad_out[i:i + chunk] = grad[0].cpu().numpy()
        conf_out[i:i + chunk] = conf[0].cpu().numpy()
    return udf_out, grad_out, conf_out


@torch.no_grad()
def gradient_project(model, z, xyz: np.ndarray, n_iters: int, device, chunk: int = 100_000):
    """Newton-style projection toward the zero level set: x <- x + udf(x)*grad(x)."""
    xyz = xyz.copy()
    udf_out = None
    for it in range(n_iters):
        udf_out, grad_out, _ = decode_chunked(model, z, xyz, device, chunk)
        xyz = xyz + udf_out[:, None] * grad_out
    return xyz, udf_out


def subdivide_and_project(model, z, pts: np.ndarray, tris: np.ndarray, device, n_proj_iters: int,
                           max_disp_frac: float = 0.5):
    """
    R7-08: densify a mesh ALONG the surface via 1-to-4 triangle subdivision,
    snapping each new edge-midpoint vertex onto the true zero level set.

    gradient_project() only moves a point toward the zero level set along
    the local UDF gradient -- it never fills the gaps BETWEEN neighboring
    points, so the post-projection point spacing (and therefore
    reconstruct_surface()'s achievable mesh resolution) stays pinned to
    whatever spacing the input candidates had, which for eval_grid()'s
    coarse grid is ~grid spacing (confirmed empirically: grid_res=96 over an
    ~900mm part gives ~10.2mm grid spacing, matching the observed ~6-7mm
    GT->Recon floor even after excluding boundary-region outliers).

    An earlier version of this fix densified by jittering+reprojecting
    offspring points and re-running reconstruct_surface() on the combined
    (much larger) point cloud -- but VTK's Hoppe algorithm scales far worse
    than linearly (confirmed empirically: 5x more points took 57x longer,
    138k points took 402s, and 580k points was killed after >8 minutes with
    no end in sight). This version instead reuses the COARSE mesh's
    triangle CONNECTIVITY (computed once, cheaply, by reconstruct_surface())
    and only adds new vertices at edge midpoints, snapped onto the surface
    via the already fast, linearly-scaling gradient_project() (~70k pts/sec
    on this GPU) -- avoiding the expensive re-triangulation step entirely.

    R7-14 (wrinkling fix): gradient_project() moves each point INDEPENDENTLY
    via per-point Newton iteration on the learned (imperfect) UDF/gradient
    field, with no awareness of neighboring points or the local mesh
    topology. Two adjacent edge-midpoints starting a few mm apart can
    converge to different local critical points of that field -- this is a
    literal self-folding of the surface, confirmed empirically: per-triangle
    normals of a refine_rounds=3 production mesh disagreed with the nearest
    GT triangle's normal (dot<0, i.e. inverted) over 49% of total mesh area,
    and total mesh surface area inflated monotonically and compoundingly
    with refine_rounds (0.97x/1.51x/1.98x/3.26x/6.96x of true GT area at
    refine_rounds=0/1/2/3/4) -- each round's wrinkled output becomes the next
    round's subdivision input, compounding the defect. Point-distance metrics
    (GT<->Recon nearest-neighbor, used throughout R7-08-R7-13) cannot detect
    this: a folded-over point can still be geometrically close to the true
    surface in isolation. Fix: clamp each midpoint's projection displacement
    to at most max_disp_frac * (its own original edge length) -- this is a
    LOCAL, scale-relative cap (not a fixed mm value), so it tightens
    automatically each refine_rounds iteration as edges shrink, structurally
    preventing the compounding runaway rather than just slowing it down.
    """
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    uniq_edges, inverse = np.unique(edges, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)

    midpoints = (pts[uniq_edges[:, 0]] + pts[uniq_edges[:, 1]]) / 2.0
    proj_mid, _ = gradient_project(model, z, midpoints, n_proj_iters, device)

    edge_len = np.linalg.norm(pts[uniq_edges[:, 0]] - pts[uniq_edges[:, 1]], axis=1)
    disp = proj_mid - midpoints
    disp_norm = np.linalg.norm(disp, axis=1)
    max_disp = max_disp_frac * edge_len
    clamp_scale = np.minimum(1.0, max_disp / np.maximum(disp_norm, 1e-12))
    n_clamped = int((clamp_scale < 1.0).sum())
    proj_mid = midpoints + disp * clamp_scale[:, None]
    print(f"  R7-14 displacement clamp: {n_clamped}/{len(midpoints)} midpoints clamped "
          f"(max_disp_frac={max_disp_frac})")

    n_orig = len(pts)
    new_pts = np.concatenate([pts, proj_mid], axis=0)

    n_tris = len(tris)
    e01 = inverse[0:n_tris] + n_orig
    e12 = inverse[n_tris:2 * n_tris] + n_orig
    e20 = inverse[2 * n_tris:3 * n_tris] + n_orig
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]

    new_tris = np.concatenate([
        np.stack([v0, e01, e20], axis=1),
        np.stack([v1, e12, e01], axis=1),
        np.stack([v2, e20, e12], axis=1),
        np.stack([e01, e12, e20], axis=1),
    ], axis=0)
    return new_pts, new_tris


def laplacian_smooth_project(model, z, pts: np.ndarray, tris: np.ndarray, device,
                              n_proj_iters: int, n_smooth_iters: int = 1,
                              lam: float = 0.5):
    """
    R7-15 (de-wrinkling): tangential Laplacian smoothing, re-snapped onto the
    zero level set after every smoothing pass.

    R7-14's displacement clamp on subdivide_and_project() turned out to be a
    one-dimensional trade-off, not a fix: sweeping max_disp_frac 0.5->0.02
    (tighter clamp, less per-point movement) DECREASED area inflation
    (2.15x->0.96x of true GT area) but INCREASED the true point-to-surface
    coverage gap (gap>2mm 16.4%->50.8%) -- the opposite of what a "wrinkling
    causes the gap" hypothesis predicts. Root cause: gradient_project()
    moves every point completely INDEPENDENTLY (a per-point Newton step
    with no knowledge of neighboring points), so reaching the true zero
    level set in some regions genuinely requires a large per-point step,
    and clamping that step trades "doesn't reach the surface" against
    "reaches the surface but neighbors fold over each other". Capping
    displacement magnitude cannot fix this because the defect is a lack of
    COHERENCE between neighboring points' projections, not the magnitude
    of any single point's projection.

    Tangential Laplacian smoothing adds that coherence directly: each
    vertex is pulled toward its 1-ring neighborhood average position, but
    ONLY the in-surface (tangential) component of that pull is kept -- the
    component along the local UDF gradient (the learned surface-normal
    direction) is discarded, so smoothing itself cannot push a vertex off
    the zero level set. A gradient_project() re-snap after each smoothing
    pass then corrects the small residual normal-direction drift that
    smoothing introduces anyway (the tangent plane used for the split is
    only exact at a single point, not across the whole averaging step).
    """
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=0)
    n = len(pts)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    adj = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)).tocsr()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    deg_safe = np.maximum(deg, 1.0)
    no_neighbor = deg < 1.0

    for _ in range(n_smooth_iters):
        neighbor_avg = np.asarray(adj.dot(pts)) / deg_safe[:, None]
        _, grad, _ = decode_chunked(model, z, pts, device)
        grad_norm = grad / np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-12)

        disp = lam * (neighbor_avg - pts)
        disp_normal = np.einsum('ij,ij->i', disp, grad_norm)[:, None] * grad_norm
        disp_tangential = disp - disp_normal
        disp_tangential[no_neighbor] = 0.0

        pts = pts + disp_tangential
        pts, _ = gradient_project(model, z, pts, n_proj_iters, device)

    return pts


def orient_components_consistently(pts: np.ndarray, tris: np.ndarray,
                                    ref_xyz: np.ndarray, ref_normals: np.ndarray):
    """
    R7-14 (wrinkling-fix companion, "20% holes look" investigation):
    pyvista's reconstruct_surface() produces a mesh fragmented into many
    DISCONNECTED components (confirmed empirically: 38 components on
    body002's coarse mesh, for what should be one connected sheet-metal
    mid-surface). VTK's compute_normals(consistent_normals=True) only
    propagates winding consistency WITHIN each connected component (via a
    per-component spanning tree) -- it cannot relate two components that
    share no edge, so different fragments can end up with opposite winding
    from each other. Confirmed empirically: applying compute_normals to a
    refine_rounds=0 mesh did NOT change its flipped-triangle fraction at
    all (50.3% before and after, vs the nearest GT triangle's normal,
    constant across refine_rounds 0-4) -- exactly consistent with a
    per-fragment (not per-triangle) orientation defect. This is the direct
    mechanism for the user-reported "GT visibly shows through scattered
    gaps in the recon layer": ~half the mesh AREA faces away from the
    camera/light from a typical viewing angle and renders as missing under
    backface culling, even though the vertex POSITIONS are geometrically
    close to GT (which is why point-distance metrics never caught it).

    Fixes this by orienting each connected component as a whole against
    ref_normals -- the REAL Stage-A input point cloud's per-point normals,
    which are the only orientation ground truth available at actual
    inference time (no GT mesh exists in production; ref_xyz/ref_normals
    are the same `points_norm`/`normals` already used to encode z).
    """
    n = len(pts)
    i0, i1, i2 = tris[:, 0], tris[:, 1], tris[:, 2]
    rows = np.concatenate([i0, i1, i2])
    cols = np.concatenate([i1, i2, i0])
    adj = sp.coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    adj = adj + adj.T
    n_comp, labels = sp.csgraph.connected_components(adj, directed=False)

    e1 = pts[i1] - pts[i0]
    e2 = pts[i2] - pts[i0]
    raw_normals = np.cross(e1, e2)
    face_areas = np.linalg.norm(raw_normals, axis=1)
    face_normals = raw_normals / np.maximum(face_areas[:, None], 1e-12)
    tri_centroids = (pts[i0] + pts[i1] + pts[i2]) / 3.0

    tree = cKDTree(ref_xyz)
    _, idx = tree.query(tri_centroids, k=1)
    dots = np.einsum('ij,ij->i', face_normals, ref_normals[idx])

    tri_comp = labels[i0]
    out_tris = tris.copy()
    n_flipped = 0
    for c in range(n_comp):
        mask = tri_comp == c
        if not mask.any():
            continue
        score = np.sum(dots[mask] * face_areas[mask])
        if score < 0:
            out_tris[mask] = out_tris[mask][:, ::-1]
            n_flipped += 1
    print(f"  R7-14 component orientation: {n_comp} connected components, "
          f"{n_flipped} flipped to match input-cloud normals")
    return out_tris


def prune_far_vertices(pts: np.ndarray, tris: np.ndarray, ref_xyz: np.ndarray,
                        threshold_norm: float):
    """
    R7-12: drop mesh vertices (and their incident triangles) farther than
    threshold_norm from the nearest point in ref_xyz, the candidate cloud
    that was actually fed into reconstruct_surface(). Returns
    (pruned_pts, pruned_tris, n_removed).
    """
    tree = cKDTree(ref_xyz)
    d, _ = tree.query(pts, k=1)
    bad_v = d > threshold_norm
    bad_tri_mask = bad_v[tris].any(axis=1)
    good_tris = tris[~bad_tri_mask]
    used_v = np.unique(good_tris)
    remap = -np.ones(len(pts), dtype=np.int64)
    remap[used_v] = np.arange(len(used_v))
    pruned_pts = pts[used_v]
    pruned_tris = remap[good_tris]
    return pruned_pts, pruned_tris, len(pts) - len(pruned_pts)


def reconstruct_midsurface(model, z, device, points_norm: np.ndarray, grid_res: int,
                            band_factor: float, n_proj_iters: int, nbr_sz: int,
                            refine_rounds: int = 0, return_stages: bool = False,
                            conf_threshold: float = 0.5,
                            input_dist_threshold_mm: float | None = None,
                            prune_dist_threshold_mm: float | None = None,
                            scale: float = 1.0,
                            points_normals: np.ndarray | None = None,
                            smooth_iters: int = 0,
                            smooth_lambda: float = 0.5):
    """
    [C-2]+[C-3]+[C-4]: dense grid -> select near-surface candidates by RANK ->
    gradient-project -> pyvista surface reconstruction -> [R7-08 subdivide +
    reproject refinement].
    Returns a pv.PolyData mid-surface mesh in NORMALIZED coordinates.

    return_stages=True additionally returns a list of (label, pv.PolyData)
    tuples, one per intermediate stage (the coarse pre-refinement mesh from
    reconstruct_surface(), then the mesh after each subdivide_and_project
    round) -- for viewer inspection of how R7-08 densification progresses.
    The last stage is identical to the single returned mid_mesh.

    R7-06: candidate selection is rank-based (smallest-UDF top_k), NOT an
    absolute distance threshold (voxel*band_factor). R7-03's truncated-
    distance training loss only requires the network to clear clip_dist_mm
    for far points -- it gives zero gradient signal to grow predictions any
    larger beyond that, so a trained model's predicted UDF can plateau at a
    small, roughly uniform ceiling across the ENTIRE [-1,1]^3 grid (confirmed
    empirically: max~=0.152 normalized units even at the cube corners, far
    below what an absolute band threshold assumes for "far" points). An
    absolute band test against that ceiling misclassifies almost the whole
    volume as "near surface" (110147/110592 = 99.6% at grid_res=48), which
    made pyvista's reconstruct_surface() take 253s on 110k points here, and
    would scale to hours at the production grid_res=96 (~800k points) given
    VTK's worse-than-linear scaling. A 2-manifold mid-surface intersects
    O(grid_res^2) cells of a grid_res^3 volume regardless of the model's
    absolute UDF calibration, so target_k scales with grid_res^2 -- this
    bounds reconstruct_surface()'s input size deterministically and sidesteps
    the calibration issue entirely, since only the RELATIVE ranking of
    predicted UDF (not its absolute magnitude) is needed to find the zero
    level set.

    R7-10: candidates are additionally gated by the model's learned
    confidence head BEFORE the UDF-rank selection above -- grid points
    with confidence below conf_threshold are excluded from the candidate
    pool entirely, regardless of how small their (possibly ghost) UDF
    prediction is. This targets TD-16 directly: the confidence head is
    trained with its own loss (train.py's confidence_loss), independent
    of the UDF regression that produces the ghost values, and is expected
    to collapse toward 0 in the extrapolated region past a true cut-edge
    boundary.

    R7-11: a SECOND, non-learned gate -- candidates farther than
    input_dist_threshold_mm from the nearest point in the actual input
    cloud are excluded, regardless of confidence or UDF rank. Diagnostic
    investigation (diagnose_outliers.py) of R7-10's worst-case residual
    outliers (the ones surviving even conf_threshold=0.99) found they sit
    in genuinely empty bbox-corner space far from EVERYTHING -- mean
    distance to nearest input point, nearest GT mid-surface point, and
    nearest GT boundary edge were all ~260mm and nearly equal, with
    learned confidence as high as 0.74 at one such point. This is a
    distribution-shift failure distinct from TD-16's boundary cut-locus
    problem: both the UDF and confidence heads are unsupervised (and
    therefore uncalibrated) far outside the near/far/boundary-shadow query
    sampling distribution (R7-09), so a single network's own self-reported
    confidence cannot be trusted to flag it (textbook epistemic- vs.
    aleatoric-uncertainty distinction; single-network heads estimate the
    latter well but not the former -- see e.g. Lakshminarayanan et al.
    2017 on deep ensembles for OOD detection). A deterministic kNN
    distance-to-input-cloud filter sidesteps the network's own
    (unreliable, in this regime) outputs entirely and directly targets
    the diagnosed failure (Sun et al. 2022, "Out-of-Distribution Detection
    with Deep Nearest Neighbors").

    R7-12: a THIRD, independent gate applied to pyvista's reconstruct_surface()
    OUTPUT (not its input). Even with R7-11 gating candidates to within 10mm
    of the input cloud, a ~170-200mm Recon->GT outlier persisted. Diagnostic
    investigation (diagnose_outliers.py) found this was NOT caused by the
    network (gradient_project() drift on gated candidates was only 3.7mm mean
    / 23mm max -- far too small to explain it) nor by subdivision bridging
    long edges (coarse-mesh max edge length was only 21.8mm). The actual
    cause: pyvista's reconstruct_surface() (VTK's Hoppe algorithm) itself
    synthesizes coarse-mesh VERTICES that extrapolate far beyond its own
    (well-behaved) input candidates in sparse regions of the point cloud --
    confirmed directly: candidates fed into reconstruct_surface() had max
    distance-to-GT of 43.7mm, but the resulting coarse-mesh vertices had max
    distance-to-GT of 197.64mm. Each coarse-mesh vertex's distance to its
    OWN candidate cloud correlates with its distance-to-GT at corrcoef=0.9917
    -- an almost perfect, directly actionable signal. prune_far_vertices()
    removes coarse-mesh (and, every refine_rounds iteration, newly
    subdivided) vertices farther than prune_dist_threshold_mm from the
    candidate cloud that was actually fed into reconstruct_surface().
    Re-applying the prune after every subdivide_and_project() round (not
    just once before refinement) is necessary because subdivide_and_project's
    own per-round gradient_project() call was empirically found to
    re-introduce similar (smaller-scale) extrapolation in sparse regions --
    a single one-shot prune before refinement is non-monotonic w.r.t.
    refine_rounds (e.g. prune=8mm: max error 44mm at refine_rounds=2 but
    75mm at refine_rounds=3). Empirical result (body002, conf_threshold=0.0,
    input_dist_threshold_mm=10, prune_dist_threshold_mm=12, refine_rounds=3):
    Recon->GT mean 53.83mm->9.98mm, max 244.77mm->74.37mm; GT->Recon mean
    1.58mm->1.62mm (no internal-accuracy regression).
    """
    t0 = time.time()
    grid_xyz, udf, _, conf = eval_grid(model, z, grid_res, device, points_norm)
    print(f"  [timing] eval_grid: {time.time()-t0:.1f}s")
    print(f"  udf stats (normalized units): min={udf.min():.4f} p5={np.percentile(udf,5):.4f} "
          f"median={np.median(udf):.4f} max={udf.max():.4f}")
    print(f"  conf stats: min={conf.min():.3f} p5={np.percentile(conf,5):.3f} "
          f"median={np.median(conf):.3f} max={conf.max():.3f}")

    target_k = min(len(udf), max(nbr_sz * 2, int(band_factor * grid_res ** 2)))

    gate_idx = np.where(conf >= conf_threshold)[0]
    if input_dist_threshold_mm is not None:
        tree_input = cKDTree(points_norm)
        d_input_norm, _ = tree_input.query(grid_xyz, k=1)
        d_input_mm = d_input_norm * scale
        ood_idx = np.where(d_input_mm <= input_dist_threshold_mm)[0]
        gate_idx = np.intersect1d(gate_idx, ood_idx)
        print(f"  R7-11 kNN-OOD gate: {len(ood_idx)}/{len(grid_xyz)} pts within "
              f"{input_dist_threshold_mm}mm of nearest input point")

    pool_idx = select_candidate_pool(gate_idx, len(udf), nbr_sz * 2)

    k = min(len(pool_idx), target_k)
    order = np.argsort(udf[pool_idx])
    cand = grid_xyz[pool_idx[order[:k]]]
    print(f"  grid {grid_res}^3 = {len(grid_xyz)} pts, confidence-gated pool="
          f"{len(pool_idx)}/{len(grid_xyz)} (conf_threshold={conf_threshold}), "
          f"selected {len(cand)} smallest-UDF candidates within pool (target_k={target_k})")

    t0 = time.time()
    proj_xyz, proj_udf = gradient_project(model, z, cand, n_proj_iters, device)
    print(f"  [timing] gradient_project: {time.time()-t0:.1f}s")
    print(f"  after {n_proj_iters} projection iters: residual udf "
          f"mean={proj_udf.mean():.5f} max={proj_udf.max():.5f} (normalized units)")

    t0 = time.time()
    cloud = pv.PolyData(proj_xyz)
    mid_mesh = cloud.reconstruct_surface(nbr_sz=nbr_sz)
    print(f"  [timing] reconstruct_surface ({len(proj_xyz)} pts): {time.time()-t0:.1f}s")

    if points_normals is not None:
        pts0 = np.asarray(mid_mesh.points)
        tris0 = mid_mesh.faces.reshape(-1, 4)[:, 1:4]
        tris0 = orient_components_consistently(pts0, tris0, points_norm, points_normals)
        faces_flat = np.column_stack([np.full(len(tris0), 3), tris0]).ravel()
        mid_mesh = pv.PolyData(pts0, faces_flat)

    prune_threshold_norm = None
    if prune_dist_threshold_mm is not None:
        prune_threshold_norm = prune_dist_threshold_mm / scale
        pts0 = np.asarray(mid_mesh.points)
        tris0 = mid_mesh.faces.reshape(-1, 4)[:, 1:4]
        pts0, tris0, n_removed = prune_far_vertices(pts0, tris0, proj_xyz, prune_threshold_norm)
        faces_flat = np.column_stack([np.full(len(tris0), 3), tris0]).ravel()
        mid_mesh = pv.PolyData(pts0, faces_flat)
        print(f"  R7-12 coarse-mesh prune: removed {n_removed} verts "
              f">{prune_dist_threshold_mm}mm from candidate cloud "
              f"({mid_mesh.n_points} verts, {mid_mesh.n_cells} tris remain)")

    if smooth_iters > 0:
        pts0 = np.asarray(mid_mesh.points)
        tris0 = mid_mesh.faces.reshape(-1, 4)[:, 1:4]
        pts0 = laplacian_smooth_project(model, z, pts0, tris0, device, n_proj_iters,
                                         smooth_iters, smooth_lambda)
        mid_mesh = pv.PolyData(pts0, mid_mesh.faces)
        print(f"  R7-15 coarse-mesh tangential Laplacian smooth: {smooth_iters} iters, "
              f"lambda={smooth_lambda}")

    stages = [("coarse", mid_mesh)]

    if refine_rounds > 0:
        pts = np.asarray(mid_mesh.points)
        tris = mid_mesh.faces.reshape(-1, 4)[:, 1:4]
        t0 = time.time()
        for r in range(refine_rounds):
            pts, tris = subdivide_and_project(model, z, pts, tris, device, n_proj_iters)
            if prune_threshold_norm is not None:
                pts, tris, n_removed = prune_far_vertices(pts, tris, proj_xyz, prune_threshold_norm)
                print(f"  R7-12 refine{r+1} prune: removed {n_removed} verts "
                      f">{prune_dist_threshold_mm}mm from candidate cloud")
            if smooth_iters > 0:
                pts = laplacian_smooth_project(model, z, pts, tris, device, n_proj_iters,
                                                smooth_iters, smooth_lambda)
            faces_flat = np.column_stack([np.full(len(tris), 3), tris]).ravel()
            stages.append((f"refine{r+1}", pv.PolyData(pts, faces_flat)))
        print(f"  [timing] subdivide_and_project x{refine_rounds} rounds: {time.time()-t0:.1f}s "
              f"({len(pts)} pts, {len(tris)} tris)")
        mid_mesh = stages[-1][1]

    if return_stages:
        return mid_mesh, stages
    return mid_mesh


def offset_solid(mid_mesh_world: pv.PolyData, thickness_mm: float) -> pv.PolyData:
    """
    [C-6]: +-thickness/2 offset along the mesh's own vertex normals.
    Returns an OPEN double-shell mesh (boundary loops not capped, see
    module docstring scope note).
    """
    mesh = mid_mesh_world.compute_normals(point_normals=True, cell_normals=False,
                                           consistent_normals=True, inplace=False)
    normals = np.asarray(mesh.point_normals)
    pts = np.asarray(mesh.points)
    half = thickness_mm / 2.0

    outer_pts = pts + normals * half
    inner_pts = pts - normals * half

    tris = mesh.faces.reshape(-1, 4)[:, 1:4]
    n_pts = len(pts)

    outer_faces = np.column_stack([np.full(len(tris), 3), tris]).ravel()
    inner_tris_flipped = tris[:, ::-1] + n_pts          # flip winding + offset indices
    inner_faces = np.column_stack([np.full(len(tris), 3), inner_tris_flipped]).ravel()

    all_pts = np.concatenate([outer_pts, inner_pts], axis=0)
    all_faces = np.concatenate([outer_faces, inner_faces])
    return pv.PolyData(all_pts, all_faces)


def process(ckpt_path: pathlib.Path, data_h5: pathlib.Path, out_stem: pathlib.Path,
            thickness_mm: float | None, grid_res: int, band_factor: float,
            n_proj_iters: int, nbr_sz: int, refine_rounds: int = 0,
            save_intermediate: bool = False, conf_threshold: float = 0.5,
            input_dist_threshold_mm: float | None = None,
            prune_dist_threshold_mm: float | None = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model = load_model(ckpt_path, device)
    points, normals, thickness_from_h5 = load_points_from_h5(data_h5)
    if thickness_mm is None:
        thickness_mm = thickness_from_h5
    print(f"thickness: {thickness_mm:.4f}mm (source: "
          f"{'h5 attrs' if thickness_mm == thickness_from_h5 else 'override'})")

    points_norm, center, scale = normalize_points(points)
    z = encode_points(model, points_norm, normals, device)

    result = reconstruct_midsurface(
        model, z, device, points_norm, grid_res=grid_res, band_factor=band_factor,
        n_proj_iters=n_proj_iters, nbr_sz=nbr_sz, refine_rounds=refine_rounds,
        return_stages=save_intermediate, conf_threshold=conf_threshold,
        input_dist_threshold_mm=input_dist_threshold_mm,
        prune_dist_threshold_mm=prune_dist_threshold_mm, scale=scale,
        points_normals=normals,
    )
    mid_mesh_norm, stages = result if save_intermediate else (result, [])
    print(f"  reconstructed mid-surface: {mid_mesh_norm.n_points} pts, "
          f"{mid_mesh_norm.n_cells} faces")

    mid_mesh_world = mid_mesh_norm.copy()
    mid_mesh_world.points = mid_mesh_norm.points * scale + center

    solid_mesh = offset_solid(mid_mesh_world, thickness_mm)

    out_mid = out_stem.parent / (out_stem.name + "_midsurface.ply")
    out_solid = out_stem.parent / (out_stem.name + "_solid.ply")
    mid_mesh_world.save(str(out_mid))
    solid_mesh.save(str(out_solid))
    print(f"  -> {out_mid}")
    print(f"  -> {out_solid}  ({solid_mesh.n_points} pts, {solid_mesh.n_cells} faces, "
          f"OPEN shell -- boundary not capped, see module docstring)")

    stage_plys = []
    if save_intermediate:
        for label, stage_mesh_norm in stages:
            stage_world = stage_mesh_norm.copy()
            stage_world.points = stage_mesh_norm.points * scale + center
            out_stage = out_stem.parent / (out_stem.name + f"_stage_{label}_midsurface.ply")
            stage_world.save(str(out_stage))
            stage_plys.append(str(out_stage))
            print(f"  -> {out_stage}  (intermediate, {stage_world.n_points} pts, "
                  f"{stage_world.n_cells} faces)")

    return {"midsurface_ply": str(out_mid), "solid_ply": str(out_solid),
            "thickness_mm": thickness_mm, "center": center.tolist(), "scale": scale,
            "stage_plys": stage_plys}


def main():
    ap = argparse.ArgumentParser(description="Stage C: reconstruct geometry from a trained "
                                              "VecSet Autoencoder.")
    ap.add_argument("--ckpt", required=True, help="Trained checkpoint .pt (from train.py)")
    ap.add_argument("--data", required=True, help="*_dataset.h5 to supply the input point cloud")
    ap.add_argument("--out", required=True, help="Output path stem (no extension)")
    ap.add_argument("--thickness-mm", type=float, default=None,
                     help="Override thickness (default: h5 attrs thickness_mm)")
    ap.add_argument("--grid-res", type=int, default=96)
    ap.add_argument("--band-factor", type=float, default=3.0,
                     help="Near-surface candidate count = band_factor * grid_res^2 (R7-06)")
    ap.add_argument("--n-proj-iters", type=int, default=5)
    ap.add_argument("--nbr-sz", type=int, default=20,
                     help="pyvista reconstruct_surface neighborhood size")
    ap.add_argument("--refine-rounds", type=int, default=4,
                     help="R7-08: mesh subdivision-and-reprojection rounds (0=off). Each round "
                          "performs 1-to-4 triangle subdivision on the reconstructed mesh, "
                          "snapping every new edge-midpoint vertex onto the true zero level set "
                          "via gradient_project(), halving the along-surface point spacing "
                          "without re-running the expensive reconstruct_surface() at higher "
                          "point counts. Default raised 3->4 by R7-13: refine_rounds is the "
                          "dominant lever for closing true GT->Recon coverage gaps ('holey "
                          "mesh'); rr=4 cuts true gap>5mm from 5.3%->1.1% vs rr=3.")
    ap.add_argument("--save-intermediate", action="store_true",
                     help="Also save the coarse pre-refinement mesh and the mesh after each "
                          "subdivide_and_project round as separate "
                          "*_stage_<label>_midsurface.ply files, for viewer inspection "
                          "(poc_result_viewer.py).")
    ap.add_argument("--conf-threshold", type=float, default=0.0,
                     help="R7-10: grid points with predicted confidence below this are "
                          "excluded from Stage C candidate selection, regardless of UDF "
                          "rank (TD-16 boundary ghost-mesh suppression). Default changed "
                          "0.5->0.0 by R7-13: every validated production combo (R7-11, "
                          "R7-12, R7-13) used conf_threshold=0.0 -- the learned confidence "
                          "head's calibration was found unreliable on its own (Round 8 Deep "
                          "Ensembles investigation), so OOD/ghost suppression is delegated "
                          "to the deterministic --input-dist-threshold-mm and "
                          "--prune-dist-threshold-mm gates instead; leaving this at 0.5 was "
                          "stale and untested against the actual production pipeline.")
    ap.add_argument("--input-dist-threshold-mm", type=float, default=10.0,
                     help="R7-11: grid points farther than this from the nearest input "
                          "cloud point are excluded from Stage C candidate selection, "
                          "regardless of confidence or UDF rank. Deterministic kNN-distance "
                          "OOD gate, complementary to the learned confidence head -- see "
                          "reconstruct_midsurface() docstring. Default changed None(disabled)"
                          "->10.0mm by R7-13: this is the value every R7-11/R7-12/R7-13 "
                          "validation run actually used; leaving the CLI default disabled "
                          "was an untested divergence from the validated production setting.")
    ap.add_argument("--prune-dist-threshold-mm", type=float, default=20.0,
                     help="R7-12: mesh vertices (coarse + each refine_rounds iteration) "
                          "farther than this from the candidate cloud actually fed into "
                          "reconstruct_surface() are removed. Targets VTK's own "
                          "reconstruct_surface() extrapolation artifacts in sparse regions, "
                          "independent of R7-10/R7-11 -- see reconstruct_midsurface() "
                          "docstring. Default raised 12->20mm by R7-13: loosening this gate "
                          "(paired with refine_rounds=4) trades a negligible overshoot cost "
                          "(mean 8.94->9.53mm, max 46.95->46.42mm) for far fewer true "
                          "coverage holes (gap>5mm 5.3%->1.1%, gap mean 1.43->0.89mm).")
    args = ap.parse_args()

    process(
        pathlib.Path(args.ckpt), pathlib.Path(args.data), pathlib.Path(args.out),
        thickness_mm=args.thickness_mm, grid_res=args.grid_res,
        band_factor=args.band_factor, n_proj_iters=args.n_proj_iters, nbr_sz=args.nbr_sz,
        refine_rounds=args.refine_rounds, save_intermediate=args.save_intermediate,
        conf_threshold=args.conf_threshold,
        input_dist_threshold_mm=args.input_dist_threshold_mm,
        prune_dist_threshold_mm=args.prune_dist_threshold_mm,
    )


if __name__ == "__main__":
    main()

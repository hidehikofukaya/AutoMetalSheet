"""
midsurface_sampler.py -- Stage A: deterministic mid-surface point cloud +
UDF training dataset generator for the Round 6 PoC architecture
(see CLAUDE.md section 13, decisions R6-01..R6-06).

Pipeline (per body)
--------------------
  body###_filled.stp   (hole-filled solid; run hole_filler.py first -- R6-03)
    -> [A-1] tessellate                  (reuse annotator.step_to_pyvista)
    -> [A-2] mid-surface projection of every vertex, wall/edge faces excluded
             when the ray misses within max_depth = thickness * max_depth_factor
             (reuse annotator.project_to_midsurface -- R6-02, R6-04)
    -> [A-3] mid-surface mesh: original triangle connectivity is kept,
             only vertex positions are replaced by the projected P_mid
             (no Poisson reconstruction -- R6-01)            -> *_midsurface.ply
    -> [A-4] noisy training input point cloud
             (area-weighted resample to N pts + Gaussian noise + random
             spherical occlusion mask, re-padded back to N pts)
    -> [A-5] GT UDF query sampling: near-surface band + far uniform +
             boundary-shadow queries (R7-09), unsigned distance computed via
             trimesh.proximity.closest_point (open3d is NOT installed in
             this environment -- substituted per the environment correction
             noted in CLAUDE.md section 13)
    -> *_dataset.h5 { points, normals, query_xyz, query_udf, query_grad,
                       query_category }

Scope note (R6-05, R6-06)
--------------------------
This tool produces training data for mesh/implicit-field reconstruction
only. No thickness field is stored in the dataset (R6-06: no thickness
decoder head). A single scalar thickness estimate per body is recorded in
the metadata JSON only, for bookkeeping / later solid-offset use (Stage C).

Usage
-----
Single body:
    python midsurface_sampler.py body003_filled.stp

Batch from hierarchy.json (all sheet_metal bodies; resolves *_filled.stp):
    python midsurface_sampler.py --hierarchy C:/.../output/A0072600002/hierarchy.json

Batch across entire output directory:
    python midsurface_sampler.py --batch-dir C:/.../Mesh_Generater/output

Output per processed body
--------------------------
bodies/body003_filled_midsurface.ply        -- mid-surface mesh (GT geometry)
bodies/body003_filled_dataset.h5            -- training dataset
bodies/body003_filled_midsurface_meta.json  -- processing metadata
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pyvista as pv
import h5py
import trimesh

from annotator import step_to_pyvista, project_to_midsurface

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

MAX_DEPTH_FACTOR_DEFAULT  = 3.0   # R6-02: wall-face exclusion = thickness * this
N_POINTS_DEFAULT          = 8192
N_QUERY_NEAR_DEFAULT      = 8192
N_QUERY_FAR_DEFAULT       = 2048
N_QUERY_BOUNDARY_DEFAULT  = 4096   # R7-09: boundary-shadow oversampling
BOUNDARY_SHADOW_MAX_MM_DEFAULT = 40.0   # R7-09: max outward offset from a cut edge
NOISE_STD_MM_DEFAULT      = 0.2
BAND_FACTOR_DEFAULT       = 5.0   # near-query band = thickness * this

# query_category values stored in the h5 (R7-09)
QUERY_CATEGORY_NEAR     = 0
QUERY_CATEGORY_FAR      = 1
QUERY_CATEGORY_BOUNDARY = 2

# ─────────────────────────────────────────────────────────────────────────────
# Thickness auto-estimation (used as the wall-face exclusion threshold, R6-02)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_thickness_mm(mesh: pv.PolyData, n_probe: int = 300,
                           max_depth_mm: float = 50.0,
                           rng: np.random.Generator | None = None) -> float:
    """Robust thickness estimate via a random subsample of ray-cast probes."""
    rng = rng or np.random.default_rng(0)
    pts     = np.asarray(mesh.points)
    normals = np.asarray(mesh.point_normals)
    n       = len(pts)
    idx     = rng.choice(n, size=min(n_probe, n), replace=False)

    vals = []
    for i in idx:
        _, t = project_to_midsurface(mesh, pts[i], normals[i], max_depth_mm=max_depth_mm)
        if t is not None and t > 1e-6:
            vals.append(t)

    if len(vals) < 10:
        raise RuntimeError(
            "Could not auto-estimate thickness (too few valid ray hits); "
            "pass --thickness-hint explicitly"
        )
    return float(np.median(vals))

# ─────────────────────────────────────────────────────────────────────────────
# [A-2] + [A-3] Mid-surface mesh construction
# ─────────────────────────────────────────────────────────────────────────────

TOPOLOGY_PATCH_MIN_VALID_NEIGHBOR_FRAC_DEFAULT = 0.8
TOPOLOGY_PATCH_MAX_ITERS_DEFAULT = 3


def _build_vertex_adjacency(n: int, tris: np.ndarray) -> list:
    """1-ring vertex adjacency (by mesh connectivity) from triangle indices."""
    adj = [set() for _ in range(n)]
    for a, b, c in tris:
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))
    return adj


def patch_isolated_invalid_vertices(tris: np.ndarray,
                                     valid: np.ndarray,
                                     mid_pts: np.ndarray,
                                     min_valid_neighbor_frac: float = TOPOLOGY_PATCH_MIN_VALID_NEIGHBOR_FRAC_DEFAULT,
                                     max_iters: int = TOPOLOGY_PATCH_MAX_ITERS_DEFAULT
                                     ) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Boundary-topology fix (root cause of the "moth-eaten mesh" defect,
    fact-checked 2026-06-17): project_to_midsurface() decides wall/edge
    exclusion per vertex via a single independent ray-cast with no spatial
    coherence. Near true cut edges and in curved/bent regions, grazing-angle
    ray misses flip the valid/invalid label noisily vertex-by-vertex,
    shattering what should be one continuous boundary loop (the true outer
    perimeter, or a real >15mm design hole -- hole_filler.py has already
    removed every real hole <=15mm per R6-03) into hundreds of spurious
    micro-loops. Empirically confirmed on body002: GT had 211 boundary loops,
    none tracing the part's true ~900mm outer silhouette (largest loop bbox
    diagonal was only 202.7mm); 62% of loops were <=6-vertex artifacts.

    Since hole_filler.py guarantees no real feature is smaller than 15mm, an
    isolated invalid vertex whose 1-ring neighbours are overwhelmingly valid
    cannot be a genuine design feature -- it is ray-cast noise. Patch it by
    interpolating P_mid from its valid neighbours and flipping it valid.
    Iterated so each pass can also recover the next ring inward. A vertex
    along a genuinely large open boundary never qualifies, because most of
    its neighbours are also invalid there.
    """
    n = len(valid)
    adj = _build_vertex_adjacency(n, tris)
    valid = valid.copy()
    mid_pts = mid_pts.copy()
    n_patched_total = 0

    for _ in range(max_iters):
        invalid_idx = np.where(~valid)[0]
        if len(invalid_idx) == 0:
            break
        to_patch = []
        for v in invalid_idx:
            nbrs = adj[v]
            if not nbrs:
                continue
            valid_nbrs = [u for u in nbrs if valid[u]]
            if len(valid_nbrs) / len(nbrs) >= min_valid_neighbor_frac:
                to_patch.append((v, valid_nbrs))
        if not to_patch:
            break
        for v, valid_nbrs in to_patch:
            mid_pts[v] = mid_pts[valid_nbrs].mean(axis=0)
            valid[v] = True
        n_patched_total += len(to_patch)

    return valid, mid_pts, n_patched_total


def build_midsurface_mesh(mesh: pv.PolyData,
                           thickness_hint_mm: float | None = None,
                           max_depth_factor: float = MAX_DEPTH_FACTOR_DEFAULT,
                           rng: np.random.Generator | None = None,
                           topology_patch: bool = True,
                           patch_min_valid_neighbor_frac: float = TOPOLOGY_PATCH_MIN_VALID_NEIGHBOR_FRAC_DEFAULT,
                           patch_max_iters: int = TOPOLOGY_PATCH_MAX_ITERS_DEFAULT,
                           ) -> tuple[pv.PolyData, float]:
    """
    R6-01/R6-02: project every vertex to the mid-surface; vertices whose ray
    misses within (thickness * max_depth_factor) are wall/edge faces and are
    dropped. Triangle connectivity from the original mesh is preserved --
    a face survives only if all 3 of its vertices projected successfully.

    If topology_patch (default True), spurious isolated invalid vertices are
    recovered first via patch_isolated_invalid_vertices() (see its docstring
    for the root-cause justification).

    Returns (mid_surface_mesh, thickness_hint_used_mm).
    """
    rng = rng or np.random.default_rng(0)
    if thickness_hint_mm is None:
        thickness_hint_mm = estimate_thickness_mm(mesh, rng=rng)

    max_depth = thickness_hint_mm * max_depth_factor

    pts     = np.asarray(mesh.points)
    normals = np.asarray(mesh.point_normals)
    n       = len(pts)

    mid_pts = np.zeros((n, 3), dtype=np.float64)
    valid   = np.zeros(n, dtype=bool)

    t0 = time.time()
    for i in range(n):
        P_mid, t = project_to_midsurface(mesh, pts[i], normals[i], max_depth_mm=max_depth)
        if P_mid is not None:
            mid_pts[i] = P_mid
            valid[i]   = True
        if n >= 2000 and i % 2000 == 0 and i > 0:
            print(f"      projected {i}/{n} ({time.time()-t0:.1f}s)", flush=True)

    tris = mesh.faces.reshape(-1, 4)[:, 1:4]

    if topology_patch:
        valid, mid_pts, n_patched = patch_isolated_invalid_vertices(
            tris, valid, mid_pts,
            min_valid_neighbor_frac=patch_min_valid_neighbor_frac,
            max_iters=patch_max_iters,
        )
        if n_patched:
            print(f"      topology patch: recovered {n_patched} spurious-invalid "
                  f"vertices (ray-cast noise, not real <=15mm features)", flush=True)

    keep_face = valid[tris[:, 0]] & valid[tris[:, 1]] & valid[tris[:, 2]]
    kept_tris = tris[keep_face]

    if len(kept_tris) == 0:
        empty = pv.PolyData(np.zeros((0, 3)))
        return empty, thickness_hint_mm

    # Compact: drop unreferenced vertices so the output mesh has no orphans.
    used_idx = np.unique(kept_tris)
    remap = -np.ones(n, dtype=np.int64)
    remap[used_idx] = np.arange(len(used_idx))
    compact_pts  = mid_pts[used_idx]
    compact_tris = remap[kept_tris]

    faces_flat = np.column_stack(
        [np.full(len(compact_tris), 3), compact_tris]
    ).ravel()
    mid_mesh = pv.PolyData(compact_pts, faces_flat)
    mid_mesh.compute_normals(point_normals=True, cell_normals=True,
                              consistent_normals=True, inplace=True)
    return mid_mesh, thickness_hint_mm

# ─────────────────────────────────────────────────────────────────────────────
# pyvista -> trimesh bridge
# ─────────────────────────────────────────────────────────────────────────────

def _pv_to_trimesh(mid_mesh: pv.PolyData) -> trimesh.Trimesh:
    verts = np.asarray(mid_mesh.points)
    tris  = mid_mesh.faces.reshape(-1, 4)[:, 1:4]
    return trimesh.Trimesh(vertices=verts, faces=tris, process=False)

# ─────────────────────────────────────────────────────────────────────────────
# [A-4] Noisy training input point cloud
# ─────────────────────────────────────────────────────────────────────────────

def sample_training_cloud(tri_mesh: trimesh.Trimesh,
                           n_points: int = N_POINTS_DEFAULT,
                           noise_std_mm: float = NOISE_STD_MM_DEFAULT,
                           occlusion_prob: float = 0.5,
                           mask_ratio_range: tuple[float, float] = (0.05, 0.3),
                           rng: np.random.Generator | None = None
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted surface sample + occlusion mask + Gaussian noise."""
    rng = rng or np.random.default_rng()

    oversample = int(n_points * 1.5)
    pts, face_idx = trimesh.sample.sample_surface(tri_mesh, oversample)
    pts = np.asarray(pts)
    normals = tri_mesh.face_normals[face_idx]

    if rng.random() < occlusion_prob:
        ratio = rng.uniform(*mask_ratio_range)
        center = pts[rng.integers(len(pts))]
        d = np.linalg.norm(pts - center, axis=1)
        radius = np.quantile(d, ratio)
        keep = d > radius
        if keep.sum() >= 50:   # don't occlude away almost everything
            pts, normals = pts[keep], normals[keep]

    if len(pts) < n_points:
        pad_idx = rng.integers(0, len(pts), size=n_points - len(pts))
        pts     = np.concatenate([pts, pts[pad_idx]])
        normals = np.concatenate([normals, normals[pad_idx]])
    elif len(pts) > n_points:
        sel = rng.choice(len(pts), size=n_points, replace=False)
        pts, normals = pts[sel], normals[sel]

    pts = pts + rng.normal(0.0, noise_std_mm, size=pts.shape)
    return pts.astype(np.float32), normals.astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# [A-5] GT UDF query sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_udf_queries(tri_mesh: trimesh.Trimesh,
                        n_near: int = N_QUERY_NEAR_DEFAULT,
                        n_far: int = N_QUERY_FAR_DEFAULT,
                        band_mm: float = 1.0,
                        rng: np.random.Generator | None = None
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Near-surface band queries (Gaussian offset, sigma = band_mm/3) +
    far uniform-in-bbox queries. GT UDF via trimesh.proximity.closest_point
    (open3d substitute, see module docstring / CLAUDE.md section 13).

    Also returns query_grad: the unit vector at each query point pointing
    TOWARD its nearest mid-surface point, i.e. the (negative) UDF gradient
    direction -- grad = (closest_point - query) / |closest_point - query|.
    This is the correct GT supervision target for the decoder's "normal"
    head (R7-01, CLAUDE.md section 13): a learned UDF has no sign, so what
    the decoder predicts is the gradient direction, not a surface normal in
    the usual sense. Stage C's gradient-projection mesh extraction needs
    exactly this field (x_new = x + UDF(x) * grad). At distance ~0 this
    direction converges to the true surface normal at the closest point.
    """
    rng = rng or np.random.default_rng()

    near_surf, _ = trimesh.sample.sample_surface(tri_mesh, n_near)
    near_surf = np.asarray(near_surf)
    offset_dir = rng.normal(size=(n_near, 3))
    offset_dir /= (np.linalg.norm(offset_dir, axis=1, keepdims=True) + 1e-12)
    offset_mag = np.abs(rng.normal(0.0, band_mm / 3.0, size=n_near))
    near_q = near_surf + offset_dir * offset_mag[:, None]

    bmin, bmax = tri_mesh.bounds
    pad = 0.1 * (bmax - bmin)
    far_q = rng.uniform(bmin - pad, bmax + pad, size=(n_far, 3))

    query_xyz = np.concatenate([near_q, far_q]).astype(np.float64)
    closest, dist, _ = trimesh.proximity.closest_point(tri_mesh, query_xyz)

    grad = closest - query_xyz
    grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
    query_grad = np.divide(grad, grad_norm, out=np.zeros_like(grad),
                            where=grad_norm > 1e-9)

    return (query_xyz.astype(np.float32), dist.astype(np.float32),
            query_grad.astype(np.float32))


def sample_boundary_shadow_queries(tri_mesh: trimesh.Trimesh,
                                    n_boundary: int = N_QUERY_BOUNDARY_DEFAULT,
                                    shadow_max_mm: float = BOUNDARY_SHADOW_MAX_MM_DEFAULT,
                                    rng: np.random.Generator | None = None
                                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    R7-09: boundary-shadow query oversampling (CLAUDE.md TD-16 fix, Option 1).

    The cut edge of an open mid-surface mesh is a true UDF cut locus
    (non-differentiable kink -- see CLAUDE.md TD-16 / R7-09 literature
    notes). sample_udf_queries()'s near-band queries only offset a few mm
    (thickness-scaled) from ANY surface point, and its far-uniform queries
    are spread too thinly across the entire padded bbox to land near a cut
    edge in any numbers -- so the thin "shadow" region just beyond a
    boundary loop is almost entirely unsupervised. This function explicitly
    samples there: pick a random point on a GT boundary loop, then offset
    outward (away from the body, roughly in the local tangent plane) by a
    distance biased toward the edge (sqrt(uniform) in [0, shadow_max_mm]).
    True UDF/gradient are computed exactly like the other categories via
    trimesh.proximity.closest_point -- no synthetic/approximated targets.

    Outward direction: project (boundary_point - mesh_centroid) onto the
    local tangent plane (perpendicular to the nearest mesh vertex normal),
    then normalize. This is a cheap heuristic (no loop-tangent / ordered-
    traversal bookkeeping needed) that is correct for any locally star-
    convex boundary region, which sheet-metal cut edges typically are.
    """
    rng = rng or np.random.default_rng()
    empty = np.zeros((0, 3), dtype=np.float32)

    boundary = tri_mesh.outline()
    if boundary is None or len(boundary.entities) == 0:
        return empty, empty.reshape(0).astype(np.float32), empty

    loops = [np.asarray(loop) for loop in boundary.discrete if len(loop) >= 2]
    if not loops:
        return empty, empty.reshape(0).astype(np.float32), empty
    loop_pts = np.concatenate(loops, axis=0)                  # [Nb,3]

    nearest_vidx = tri_mesh.kdtree.query(loop_pts, k=1)[1]
    local_normal = tri_mesh.vertex_normals[nearest_vidx]
    centroid = tri_mesh.vertices.mean(axis=0)

    radial = loop_pts - centroid
    radial -= np.sum(radial * local_normal, axis=1, keepdims=True) * local_normal
    radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
    outward = np.divide(radial, radial_norm, out=np.zeros_like(radial),
                         where=radial_norm > 1e-9)

    pick = rng.integers(0, len(loop_pts), size=n_boundary)
    t = np.sqrt(rng.uniform(0.0, 1.0, size=n_boundary))        # bias toward the edge
    offset_mm = t * shadow_max_mm
    lateral_jitter = rng.normal(0.0, shadow_max_mm * 0.05, size=(n_boundary, 3))
    query_xyz = loop_pts[pick] + outward[pick] * offset_mm[:, None] + lateral_jitter

    closest, dist, _ = trimesh.proximity.closest_point(tri_mesh, query_xyz)
    grad = closest - query_xyz
    grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
    query_grad = np.divide(grad, grad_norm, out=np.zeros_like(grad),
                            where=grad_norm > 1e-9)

    return (query_xyz.astype(np.float32), dist.astype(np.float32),
            query_grad.astype(np.float32))

# ─────────────────────────────────────────────────────────────────────────────
# Per-body processing
# ─────────────────────────────────────────────────────────────────────────────

def process_body(stp_path: pathlib.Path,
                  thickness_hint_mm: float | None = None,
                  n_points: int = N_POINTS_DEFAULT,
                  n_query_near: int = N_QUERY_NEAR_DEFAULT,
                  n_query_far: int = N_QUERY_FAR_DEFAULT,
                  n_query_boundary: int = N_QUERY_BOUNDARY_DEFAULT,
                  boundary_shadow_max_mm: float = BOUNDARY_SHADOW_MAX_MM_DEFAULT,
                  noise_std_mm: float = NOISE_STD_MM_DEFAULT,
                  band_factor: float = BAND_FACTOR_DEFAULT,
                  max_depth_factor: float = MAX_DEPTH_FACTOR_DEFAULT,
                  deflection: float = 0.15,
                  seed: int = 0) -> dict:
    """
    Process a single body###_filled.stp.
    Outputs *_midsurface.ply, *_dataset.h5, *_midsurface_meta.json alongside it.
    """
    t0  = time.time()
    rng = np.random.default_rng(seed)

    out_ply  = stp_path.parent / (stp_path.stem + "_midsurface.ply")
    out_h5   = stp_path.parent / (stp_path.stem + "_dataset.h5")
    out_json = stp_path.parent / (stp_path.stem + "_midsurface_meta.json")

    print(f"  [{stp_path.stem}] tessellating ...", end=" ", flush=True)
    try:
        mesh = step_to_pyvista(stp_path, deflection)
    except Exception as e:
        print(f"ERROR: {e}")
        return {"source_stp": str(stp_path), "status": "read_error", "error": str(e)}
    print(f"{mesh.n_points} pts / {mesh.n_cells} faces", flush=True)

    print(f"  [{stp_path.stem}] mid-surface projection ...", flush=True)
    try:
        mid_mesh, thickness_used = build_midsurface_mesh(
            mesh, thickness_hint_mm, max_depth_factor=max_depth_factor, rng=rng
        )
    except Exception as e:
        print(f"    ERROR: {e}")
        return {"source_stp": str(stp_path), "status": "midsurface_error", "error": str(e)}

    n_valid = mid_mesh.n_points
    print(f"    -> {n_valid}/{mesh.n_points} vertices valid, "
          f"{mid_mesh.n_cells} faces, thickness~{thickness_used:.3f}mm")

    if mid_mesh.n_points < 50 or mid_mesh.n_cells == 0:
        meta = {
            "source_stp": str(stp_path),
            "status": "midsurface_too_small",
            "thickness_used_mm": round(thickness_used, 4),
            "n_vertices_valid": int(n_valid),
        }
        out_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return meta

    mid_mesh.save(str(out_ply))
    tri_mesh = _pv_to_trimesh(mid_mesh)

    points, normals = sample_training_cloud(
        tri_mesh, n_points=n_points, noise_std_mm=noise_std_mm, rng=rng
    )
    query_xyz, query_udf, query_grad = sample_udf_queries(
        tri_mesh, n_near=n_query_near, n_far=n_query_far,
        band_mm=thickness_used * band_factor, rng=rng
    )
    query_category = np.concatenate([
        np.full(n_query_near, QUERY_CATEGORY_NEAR, dtype=np.int64),
        np.full(n_query_far, QUERY_CATEGORY_FAR, dtype=np.int64),
    ])

    b_xyz, b_udf, b_grad = sample_boundary_shadow_queries(
        tri_mesh, n_boundary=n_query_boundary,
        shadow_max_mm=boundary_shadow_max_mm, rng=rng,
    )
    print(f"    -> {len(b_xyz)} boundary-shadow queries (R7-09, max {boundary_shadow_max_mm}mm)")
    query_xyz = np.concatenate([query_xyz, b_xyz])
    query_udf = np.concatenate([query_udf, b_udf])
    query_grad = np.concatenate([query_grad, b_grad])
    query_category = np.concatenate([
        query_category, np.full(len(b_xyz), QUERY_CATEGORY_BOUNDARY, dtype=np.int64),
    ])

    with h5py.File(out_h5, "w") as f:
        f.create_dataset("points",         data=points)
        f.create_dataset("normals",        data=normals)
        f.create_dataset("query_xyz",      data=query_xyz)
        f.create_dataset("query_udf",      data=query_udf)
        f.create_dataset("query_grad",     data=query_grad)
        f.create_dataset("query_category", data=query_category)
        f.attrs["source_stp"]    = str(stp_path)
        f.attrs["thickness_mm"]  = thickness_used

    meta = {
        "source_stp":         str(stp_path),
        "midsurface_ply":     str(out_ply),
        "dataset_h5":         str(out_h5),
        "thickness_used_mm":  round(thickness_used, 4),
        "n_vertices_total":   int(mesh.n_points),
        "n_vertices_valid":   int(n_valid),
        "n_faces_midsurface": int(mid_mesh.n_cells),
        "n_points":           int(n_points),
        "n_query":            int(len(query_xyz)),
        "n_query_boundary":   int(len(b_xyz)),
        "status":             "ok",
        "elapsed_s":          round(time.time() - t0, 2),
    }
    out_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"    -> {out_h5.name}  ({meta['elapsed_s']}s)")
    return meta

# ─────────────────────────────────────────────────────────────────────────────
# Batch helpers
# ─────────────────────────────────────────────────────────────────────────────

def batch_from_hierarchy(hierarchy_json: pathlib.Path,
                          thickness_hint_mm: float | None = None,
                          **kwargs) -> list:
    """
    R6-03: only runs on hole-filled bodies. Resolves body###.stp from
    hierarchy.json's 'tag'/'stp_file'/'body_idx' fields (same schema as
    hole_filler.batch_from_hierarchy), then looks for the sibling
    body###_filled.stp produced by hole_filler.py.
    """
    data   = json.loads(hierarchy_json.read_text("utf-8"))
    root   = hierarchy_json.parent
    bodies = data.get("bodies", [])

    sm_bodies = [b for b in bodies
                 if b.get("tag", b.get("classification", "")) == "sheet_metal"]

    print(f"\n{root.name}: {len(sm_bodies)} sheet_metal / {len(bodies)} total bodies")

    results = []
    for b in sm_bodies:
        rel = b.get("stp_file")
        if rel:
            base_stp = root / rel
        else:
            idx = b.get("body_idx", b.get("index"))
            if idx is None:
                print(f"  SKIP: no stp_file/body_idx in {b}")
                continue
            base_stp = root / "bodies" / f"body{int(idx):03d}.stp"

        filled_stp = base_stp.parent / (base_stp.stem + "_filled.stp")
        if not filled_stp.exists():
            print(f"  SKIP (no _filled.stp; run hole_filler.py first): {filled_stp.name}")
            continue

        hint = thickness_hint_mm if thickness_hint_mm is not None else b.get("thickness_est_mm")
        try:
            meta = process_body(filled_stp, thickness_hint_mm=hint, **kwargs)
        except Exception as e:
            print(f"  ERROR: {filled_stp.name}: {e}")
            meta = {"source_stp": str(filled_stp), "status": "error", "error": str(e)}
        results.append(meta)

    print(f"  Done ({len(results)} processed)")
    return results


def batch_from_dir(root: pathlib.Path,
                    thickness_hint_mm: float | None = None,
                    **kwargs) -> list:
    hfiles = sorted(root.rglob("hierarchy.json"))
    print(f"Found {len(hfiles)} hierarchy.json files under {root}")
    all_results = []
    for hf in hfiles:
        all_results.extend(
            batch_from_hierarchy(hf, thickness_hint_mm=thickness_hint_mm, **kwargs)
        )
    return all_results

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate mid-surface point cloud + UDF training dataset "
                     "from hole-filled sheet metal STEP solids.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("stp", nargs="?",
                    help="Single body###_filled.stp to process")
    ap.add_argument("--hierarchy", metavar="PATH",
                    help="hierarchy.json -- process all sheet_metal bodies "
                         "(resolves the *_filled.stp sibling)")
    ap.add_argument("--batch-dir", metavar="PATH",
                    help="Root output dir -- process every hierarchy.json found")
    ap.add_argument("--thickness-hint", type=float, default=None,
                    help="Override thickness mm (default: hierarchy's "
                         "thickness_est_mm, or auto-estimated for single-file mode)")
    ap.add_argument("--n-points",      type=int,   default=N_POINTS_DEFAULT)
    ap.add_argument("--n-query-near",  type=int,   default=N_QUERY_NEAR_DEFAULT)
    ap.add_argument("--n-query-far",   type=int,   default=N_QUERY_FAR_DEFAULT)
    ap.add_argument("--n-query-boundary", type=int, default=N_QUERY_BOUNDARY_DEFAULT,
                    help="R7-09: boundary-shadow query count (TD-16 fix, Option 1)")
    ap.add_argument("--boundary-shadow-max-mm", type=float, default=BOUNDARY_SHADOW_MAX_MM_DEFAULT,
                    help="R7-09: max outward offset from a cut edge for boundary-shadow queries")
    ap.add_argument("--noise-std",     type=float, default=NOISE_STD_MM_DEFAULT,
                    help="Gaussian noise sigma mm added to input points")
    ap.add_argument("--band-factor",   type=float, default=BAND_FACTOR_DEFAULT,
                    help="Near-query band = thickness * this factor")
    ap.add_argument("--max-depth-factor", type=float, default=MAX_DEPTH_FACTOR_DEFAULT,
                    help="Wall-face exclusion depth = thickness * this factor (R6-02)")
    ap.add_argument("--deflection",    type=float, default=0.15)
    ap.add_argument("--seed",          type=int,   default=0)
    args = ap.parse_args()

    kwargs = dict(
        n_points=args.n_points,
        n_query_near=args.n_query_near,
        n_query_far=args.n_query_far,
        n_query_boundary=args.n_query_boundary,
        boundary_shadow_max_mm=args.boundary_shadow_max_mm,
        noise_std_mm=args.noise_std,
        band_factor=args.band_factor,
        max_depth_factor=args.max_depth_factor,
        deflection=args.deflection,
        seed=args.seed,
    )

    if args.stp:
        process_body(pathlib.Path(args.stp), thickness_hint_mm=args.thickness_hint, **kwargs)
    elif args.hierarchy:
        batch_from_hierarchy(pathlib.Path(args.hierarchy),
                              thickness_hint_mm=args.thickness_hint, **kwargs)
    elif args.batch_dir:
        batch_from_dir(pathlib.Path(args.batch_dir),
                        thickness_hint_mm=args.thickness_hint, **kwargs)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

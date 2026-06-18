"""
diagnose_outliers.py -- ad-hoc diagnostic (NOT part of the Stage A/B/C
pipeline): locate WHERE in space R7-10's worst Recon->GT outlier points
are, and what the model's own signals (pred_udf, pred_conf) look like
there, vs. their distance to (a) the nearest actual input point and
(b) the nearest GT mid-surface boundary edge.

Usage:
    python diagnose_outliers.py --ckpt checkpoints_r710/best.pt --conf-threshold 0.5
"""
import argparse
import pathlib

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

import reconstruct as R

DATA_H5 = pathlib.Path(
    r"C:\Users\hide2\IdeaBox\Mesh_Generater\output\A0072600002\bodies\body002_filled_dataset.h5")
GT_PLY = pathlib.Path(
    r"C:\Users\hide2\IdeaBox\Mesh_Generater\output\A0072600002\bodies\body002_filled_midsurface.ply")


def boundary_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    """Vertices touching an edge used by exactly one face (open boundary)."""
    edges_sorted = mesh.edges_sorted
    groups = trimesh.grouping.group_rows(edges_sorted, require_count=1)
    boundary_edges = mesh.edges[groups]
    idx = np.unique(boundary_edges)
    return mesh.vertices[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=pathlib.Path, default=pathlib.Path("checkpoints_r710/best.pt"))
    ap.add_argument("--conf-threshold", type=float, default=0.5)
    ap.add_argument("--input-dist-threshold-mm", type=float, default=None)
    ap.add_argument("--prune-dist-threshold-mm", type=float, default=None)
    ap.add_argument("--grid-res", type=int, default=96)
    ap.add_argument("--band-factor", type=float, default=3.0)
    ap.add_argument("--n-proj-iters", type=int, default=5)
    ap.add_argument("--nbr-sz", type=int, default=20)
    ap.add_argument("--refine-rounds", type=int, default=2)
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = R.load_model(args.ckpt, device)
    points, normals, _ = R.load_points_from_h5(DATA_H5)
    points_norm, center, scale = R.normalize_points(points)
    z = R.encode_points(model, points_norm, normals, device)

    gt_mesh = trimesh.load(str(GT_PLY), process=False)
    bnd_pts = boundary_vertices(gt_mesh)
    print(f"GT mid-surface: {len(gt_mesh.vertices)} verts, {len(bnd_pts)} boundary verts")

    if args.input_dist_threshold_mm is not None:
        # Does gradient_project() DRIFT a gated (near-input) candidate far
        # away via a bad learned gradient field, even though its STARTING
        # grid position passed the kNN-distance gate?
        grid_xyz, udf, _, conf = R.eval_grid(model, z, args.grid_res, device, points_norm)
        tree_input_norm = cKDTree(points_norm)
        d_in_norm, _ = tree_input_norm.query(grid_xyz, k=1)
        d_in_mm = d_in_norm * scale
        gate = np.where((conf >= args.conf_threshold) & (d_in_mm <= args.input_dist_threshold_mm))[0]
        target_k = min(len(udf), max(args.nbr_sz * 2, int(args.band_factor * args.grid_res ** 2)))
        k = min(len(gate), target_k)
        order = np.argsort(udf[gate])
        cand0 = grid_xyz[gate[order[:k]]]
        cand0_d_in = d_in_mm[gate[order[:k]]]
        proj_xyz, _ = R.gradient_project(model, z, cand0, args.n_proj_iters, device)
        proj_world = proj_xyz * scale + center
        d_in_after, _ = cKDTree(points).query(proj_world, k=1)
        drift_mm = np.linalg.norm((proj_xyz - cand0) * scale, axis=1)
        print(f"\n[pre/post-projection drift check] {len(cand0)} gated candidates "
              f"(conf>={args.conf_threshold}, d_input<={args.input_dist_threshold_mm}mm at start)")
        print(f"  d_input BEFORE projection: mean={cand0_d_in.mean():.2f}mm max={cand0_d_in.max():.2f}mm")
        print(f"  d_input AFTER  projection: mean={d_in_after.mean():.2f}mm max={d_in_after.max():.2f}mm")
        print(f"  projection displacement:   mean={drift_mm.mean():.2f}mm max={drift_mm.max():.2f}mm")
        worst_drift = np.argsort(-drift_mm)[:10]
        print(f"  top-10 largest displacements (mm): {drift_mm[worst_drift].round(1)}")
        print(f"  their d_input after projection (mm): {d_in_after[worst_drift].round(1)}")

    mid_mesh_norm = R.reconstruct_midsurface(
        model, z, device, points_norm, grid_res=args.grid_res, band_factor=args.band_factor,
        n_proj_iters=args.n_proj_iters, nbr_sz=args.nbr_sz, refine_rounds=args.refine_rounds,
        conf_threshold=args.conf_threshold,
        input_dist_threshold_mm=args.input_dist_threshold_mm,
        prune_dist_threshold_mm=args.prune_dist_threshold_mm, scale=scale,
        points_normals=normals,
    )
    pts_world = np.asarray(mid_mesh_norm.points) * scale + center

    tree_gt = cKDTree(gt_mesh.vertices)
    d_gt, _ = tree_gt.query(pts_world, k=1)

    tree_input = cKDTree(points)
    d_input, _ = tree_input.query(pts_world, k=1)

    tree_bnd = cKDTree(bnd_pts)
    d_bnd, _ = tree_bnd.query(pts_world, k=1)

    order = np.argsort(-d_gt)
    worst = order[:args.top_n]
    worst_world = pts_world[worst]
    worst_norm = ((worst_world - center) / scale).astype(np.float32)
    udf_w, grad_w, conf_w = R.decode_chunked(model, z, worst_norm, device)

    print(f"\ntop-{args.top_n} worst Recon->GT vertices "
          f"(all distances mm except pred_udf_norm which is normalized units):")
    print(f"{'rank':>4} {'d_GT':>8} {'d_input':>8} {'d_bnd_edge':>10} "
          f"{'pred_conf':>9} {'pred_udf_mm':>11}  world_xyz")
    for i, wi in enumerate(worst):
        pred_udf_mm = udf_w[i] * scale
        print(f"{i:4d} {d_gt[wi]:8.1f} {d_input[wi]:8.1f} {d_bnd[wi]:10.1f} "
              f"{conf_w[i]:9.4f} {pred_udf_mm:11.2f}  "
              f"({pts_world[wi,0]:.1f}, {pts_world[wi,1]:.1f}, {pts_world[wi,2]:.1f})")

    print(f"\nsummary over top-{args.top_n}:")
    print(f"  d_input  (dist to nearest actual input point):   "
          f"mean={d_input[worst].mean():.1f}mm  min={d_input[worst].min():.1f}mm")
    print(f"  d_bnd    (dist to nearest GT boundary edge):      "
          f"mean={d_bnd[worst].mean():.1f}mm  min={d_bnd[worst].min():.1f}mm")
    print(f"  pred_conf (model's own confidence at these pts):  "
          f"mean={conf_w.mean():.4f}  min={conf_w.min():.4f}  max={conf_w.max():.4f}")


if __name__ == "__main__":
    main()

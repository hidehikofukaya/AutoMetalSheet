"""
diagnose_holes.py -- ad-hoc diagnostic (NOT part of the Stage A/B/C
pipeline): locate WHERE in space the reconstructed mid-surface mesh has
HOLES (missing coverage), i.e. real GT mid-surface vertices that have no
nearby reconstructed point. This is the GT->Recon direction, the opposite
of diagnose_outliers.py's Recon->GT ghost-overshoot investigation.

For each of the worst (most under-covered) GT vertices, reports:
  - d_recon:  distance to nearest point in the FINAL reconstructed mesh
  - d_input:  distance to nearest point in the actual (noisy) input cloud
              fed to the model -- large values here mean the input point
              cloud itself was locally sparse there (candidate selection
              upstream of any gate has nothing to work with).
  - d_cand:   distance to nearest point in the CANDIDATE cloud actually
              fed into reconstruct_surface() (i.e. after the R7-10/R7-11
              gates + rank selection) -- large values here (while d_input
              is small) mean a gate excluded legitimate near-surface grid
              points in this region.
  - d_bnd:    distance to nearest GT boundary (cut) edge -- to separate
              "hole at a real open edge" (expected, not a defect) from
              "hole in the genuine interior" (a defect).
  - pred_conf / pred_udf at this GT location, decoded directly from the
    trained model, to see what the network itself believes is here.

Usage:
    python diagnose_holes.py --ckpt checkpoints_r710/best.pt \\
        --conf-threshold 0.0 --input-dist-threshold-mm 10 \\
        --prune-dist-threshold-mm 12 --refine-rounds 3
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
    edges_sorted = mesh.edges_sorted
    groups = trimesh.grouping.group_rows(edges_sorted, require_count=1)
    boundary_edges = mesh.edges[groups]
    idx = np.unique(boundary_edges)
    return mesh.vertices[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=pathlib.Path, default=pathlib.Path("checkpoints_r710/best.pt"))
    ap.add_argument("--conf-threshold", type=float, default=0.0)
    ap.add_argument("--input-dist-threshold-mm", type=float, default=10.0)
    ap.add_argument("--prune-dist-threshold-mm", type=float, default=12.0)
    ap.add_argument("--grid-res", type=int, default=96)
    ap.add_argument("--band-factor", type=float, default=3.0)
    ap.add_argument("--n-proj-iters", type=int, default=5)
    ap.add_argument("--nbr-sz", type=int, default=20)
    ap.add_argument("--refine-rounds", type=int, default=3)
    ap.add_argument("--top-n", type=int, default=25)
    ap.add_argument("--hole-threshold-mm", type=float, default=5.0,
                     help="report what fraction of GT vertices exceed this d_recon")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = R.load_model(args.ckpt, device)
    points, normals, _ = R.load_points_from_h5(DATA_H5)
    points_norm, center, scale = R.normalize_points(points)
    z = R.encode_points(model, points_norm, normals, device)

    gt_mesh = trimesh.load(str(GT_PLY), process=False)
    bnd_pts = boundary_vertices(gt_mesh)
    print(f"GT mid-surface: {len(gt_mesh.vertices)} verts, {len(bnd_pts)} boundary verts")

    # Re-derive the CANDIDATE cloud (post-gate, pre-reconstruct_surface) the
    # same way reconstruct_midsurface() does internally, so we can measure
    # d_cand separately from d_input/d_recon.
    grid_xyz, udf, _, conf = R.eval_grid(model, z, args.grid_res, device, points_norm)
    target_k = min(len(udf), max(args.nbr_sz * 2, int(args.band_factor * args.grid_res ** 2)))
    gate_idx = np.where(conf >= args.conf_threshold)[0]
    if args.input_dist_threshold_mm is not None:
        tree_input_n = cKDTree(points_norm)
        d_in_n, _ = tree_input_n.query(grid_xyz, k=1)
        ood_idx = np.where(d_in_n * scale <= args.input_dist_threshold_mm)[0]
        gate_idx = np.intersect1d(gate_idx, ood_idx)
    k = min(len(gate_idx), target_k)
    order = np.argsort(udf[gate_idx])
    cand_norm = grid_xyz[gate_idx[order[:k]]]
    cand_world = cand_norm * scale + center
    print(f"candidate cloud (post-gate, pre-projection): {len(cand_world)} pts")

    mid_mesh_norm = R.reconstruct_midsurface(
        model, z, device, points_norm, grid_res=args.grid_res, band_factor=args.band_factor,
        n_proj_iters=args.n_proj_iters, nbr_sz=args.nbr_sz, refine_rounds=args.refine_rounds,
        conf_threshold=args.conf_threshold,
        input_dist_threshold_mm=args.input_dist_threshold_mm,
        prune_dist_threshold_mm=args.prune_dist_threshold_mm, scale=scale,
        points_normals=normals,
    )
    recon_world = np.asarray(mid_mesh_norm.points) * scale + center
    print(f"final reconstructed mesh: {len(recon_world)} verts")

    tree_recon = cKDTree(recon_world)
    d_recon, _ = tree_recon.query(gt_mesh.vertices, k=1)

    tree_input = cKDTree(points)
    d_input, _ = tree_input.query(gt_mesh.vertices, k=1)

    tree_cand = cKDTree(cand_world)
    d_cand, _ = tree_cand.query(gt_mesh.vertices, k=1)

    tree_bnd = cKDTree(bnd_pts)
    d_bnd, _ = tree_bnd.query(gt_mesh.vertices, k=1)

    frac_hole = float((d_recon > args.hole_threshold_mm).mean())
    print(f"\nfraction of GT vertices with d_recon > {args.hole_threshold_mm}mm: "
          f"{frac_hole*100:.1f}% ({int(frac_hole*len(gt_mesh.vertices))}/{len(gt_mesh.vertices)})")
    print(f"d_recon over ALL GT verts: mean={d_recon.mean():.2f} median={np.median(d_recon):.2f} "
          f"p90={np.percentile(d_recon,90):.2f} max={d_recon.max():.2f}")

    order = np.argsort(-d_recon)
    worst = order[:args.top_n]
    worst_xyz = gt_mesh.vertices[worst]
    udf_w, grad_w, conf_w = R.decode_chunked(
        model, z, ((worst_xyz - center) / scale).astype(np.float32), device)

    print(f"\ntop-{args.top_n} worst GT vertices (largest d_recon = biggest hole):")
    print(f"{'rank':>4} {'d_recon':>8} {'d_input':>8} {'d_cand':>8} {'d_bnd':>8} "
          f"{'pred_conf':>9} {'pred_udf_mm':>11}  world_xyz")
    for i, wi in enumerate(worst):
        pred_udf_mm = udf_w[i] * scale
        print(f"{i:4d} {d_recon[wi]:8.1f} {d_input[wi]:8.1f} {d_cand[wi]:8.1f} {d_bnd[wi]:8.1f} "
              f"{conf_w[i]:9.4f} {pred_udf_mm:11.2f}  "
              f"({worst_xyz[i,0]:.1f}, {worst_xyz[i,1]:.1f}, {worst_xyz[i,2]:.1f})")

    print(f"\nsummary over top-{args.top_n} worst holes:")
    print(f"  d_input (nearest actual input pt):     mean={d_input[worst].mean():.1f}mm  "
          f"max={d_input[worst].max():.1f}mm")
    print(f"  d_cand  (nearest post-gate candidate):  mean={d_cand[worst].mean():.1f}mm  "
          f"max={d_cand[worst].max():.1f}mm")
    print(f"  d_bnd   (nearest GT boundary edge):     mean={d_bnd[worst].mean():.1f}mm  "
          f"min={d_bnd[worst].min():.1f}mm")
    print(f"  pred_conf at hole locations:            mean={conf_w.mean():.4f}  "
          f"min={conf_w.min():.4f}  max={conf_w.max():.4f}")

    # Diagnostic interpretation hint, printed (not asserted) for the user to read.
    interior_holes = worst[d_bnd[worst] > 15.0]
    print(f"\n{len(interior_holes)}/{args.top_n} worst holes are >15mm from any GT boundary "
          f"edge (i.e. NOT explainable as 'expected gap at an open cut edge').")


if __name__ == "__main__":
    main()

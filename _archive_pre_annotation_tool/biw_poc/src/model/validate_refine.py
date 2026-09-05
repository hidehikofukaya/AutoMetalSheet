"""
R7-08 validation: sweep --refine-rounds (subdivision-based density refinement)
and report timing + GT<->Recon surface distance at each round.

Usage (run from biw_poc/src/model, matching reconstruct.py's own convention):
    python validate_refine.py --rounds 0 1 2 3
"""
import argparse
import functools
import pathlib
import time

import numpy as np
import trimesh
import torch
from scipy.spatial import cKDTree

import reconstruct as R

print = functools.partial(print, flush=True)

DEFAULT_CKPT = pathlib.Path(r"C:\Users\hide2\IdeaBox\AutoMetalSheet\biw_poc\src\model\checkpoints_sanity\best.pt")
DATA_H5 = pathlib.Path(
    r"C:\Users\hide2\IdeaBox\Mesh_Generater\output\A0072600002\bodies\body002_filled_dataset.h5")
GT_PLY = pathlib.Path(
    r"C:\Users\hide2\IdeaBox\Mesh_Generater\output\A0072600002\bodies\body002_filled_midsurface.ply")
N_SAMPLE = 50_000


def surface_distances(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, n_sample=N_SAMPLE):
    """mean/median/p90/max of dist(sample(a) -> nearest point on b's vertex set)."""
    pa, _ = trimesh.sample.sample_surface(mesh_a, n_sample)
    tree_b = cKDTree(mesh_b.vertices)
    d, _ = tree_b.query(pa, k=1)
    return dict(mean=float(d.mean()), median=float(np.median(d)),
                p90=float(np.percentile(d, 90)), max=float(d.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--grid-res", type=int, default=96)
    ap.add_argument("--band-factor", type=float, default=3.0)
    ap.add_argument("--n-proj-iters", type=int, default=5)
    ap.add_argument("--nbr-sz", type=int, default=20)
    ap.add_argument("--ckpt", type=pathlib.Path, default=DEFAULT_CKPT)
    ap.add_argument("--conf-threshold", type=float, default=0.5,
                     help="R7-10: confidence gate passed through to reconstruct_midsurface().")
    ap.add_argument("--input-dist-threshold-mm", type=float, default=None,
                     help="R7-11: kNN-distance-to-input-cloud gate passed through to "
                          "reconstruct_midsurface() (default: disabled).")
    ap.add_argument("--prune-dist-threshold-mm", type=float, default=None,
                     help="R7-12: coarse/refined mesh vertex prune-by-distance-to-candidate-"
                          "cloud gate passed through to reconstruct_midsurface() "
                          "(default: disabled).")
    ap.add_argument("--data", type=pathlib.Path, default=DATA_H5,
                     help="override dataset h5 path (for cross-condition comparisons, "
                          "e.g. pre-topology-patch backup files).")
    ap.add_argument("--gt", type=pathlib.Path, default=GT_PLY,
                     help="override GT mid-surface ply path (must match --data's condition).")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"ckpt: {args.ckpt}")
    print(f"data: {args.data}")
    print(f"gt: {args.gt}")
    model = R.load_model(args.ckpt, device)
    points, normals, _ = R.load_points_from_h5(args.data)
    points_norm, center, scale = R.normalize_points(points)
    z = R.encode_points(model, points_norm, normals, device)

    gt_mesh = trimesh.load(str(args.gt), process=False)
    print(f"GT mid-surface: {gt_mesh.vertices.shape} verts, {gt_mesh.faces.shape} faces")

    for rr in args.rounds:
        t0 = time.time()
        mid_mesh_norm = R.reconstruct_midsurface(
            model, z, device, points_norm, grid_res=args.grid_res,
            band_factor=args.band_factor, n_proj_iters=args.n_proj_iters,
            nbr_sz=args.nbr_sz, refine_rounds=rr, conf_threshold=args.conf_threshold,
            input_dist_threshold_mm=args.input_dist_threshold_mm,
            prune_dist_threshold_mm=args.prune_dist_threshold_mm, scale=scale,
            points_normals=normals,
        )
        total_time = time.time() - t0

        pts_world = np.asarray(mid_mesh_norm.points) * scale + center
        tris = mid_mesh_norm.faces.reshape(-1, 4)[:, 1:4]
        recon_mesh = trimesh.Trimesh(vertices=pts_world, faces=tris, process=False)

        g2r = surface_distances(gt_mesh, recon_mesh)
        r2g = surface_distances(recon_mesh, gt_mesh)

        print(f"=== refine_rounds={rr} total_time={total_time:.1f}s "
              f"verts={len(pts_world)} faces={len(tris)} ===")
        print(f"  Recon->GT mean={r2g['mean']:.2f} median={r2g['median']:.2f} "
              f"p90={r2g['p90']:.2f} max={r2g['max']:.2f}")
        print(f"  GT->Recon mean={g2r['mean']:.2f} median={g2r['median']:.2f} "
              f"p90={g2r['p90']:.2f} max={g2r['max']:.2f}")


if __name__ == "__main__":
    main()

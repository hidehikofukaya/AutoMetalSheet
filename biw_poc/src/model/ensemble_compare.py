"""
ensemble_compare.py -- ad-hoc comparison script (NOT part of the Stage A/B/C
production pipeline): Deep Ensembles epistemic-uncertainty gate vs. R7-11's
deterministic kNN-distance-to-input-cloud gate, as requested by the user
("両方実装して比較する" -- implement both and compare).

Loads 5 independently-trained models (checkpoints_r710 = seed0, plus
checkpoints_r710_seed1..4, identical hyperparameters, only --seed differs)
and evaluates the SAME query grid with each. Per Lakshminarayanan et al.
2017 ("Simple and Scalable Predictive Uncertainty Estimation using Deep
Ensembles"), disagreement (std) across independently-initialized/trained
ensemble members is a standard epistemic-uncertainty / OOD signal,
complementary to a single network's own (aleatoric-only) confidence head.

Two comparisons:
  [1] Does ensemble_std correlate with actual reconstruction error at the
      worst-outlier vertices from the R7-10-only baseline run (the same
      vertices diagnose_outliers.py already characterized)?
  [2] Used ALONE as a candidate gate (no R7-11/R7-12), how does ensemble_std
      gating perform vs R7-10's confidence gate alone and R7-11's distance
      gate alone, on the same Recon->GT / GT->Recon metrics?

Usage:
    python ensemble_compare.py
"""
import pathlib
import time

import numpy as np
import pyvista as pv
import torch
import trimesh
from scipy.spatial import cKDTree

import reconstruct as R

DATA_H5 = pathlib.Path(
    r"C:\Users\hide2\IdeaBox\Mesh_Generater\output\A0072600002\bodies\body002_filled_dataset.h5")
GT_PLY = pathlib.Path(
    r"C:\Users\hide2\IdeaBox\Mesh_Generater\output\A0072600002\bodies\body002_filled_midsurface.ply")
CKPTS = [
    pathlib.Path("checkpoints_r710/best.pt"),
    pathlib.Path("checkpoints_r710_seed1/best.pt"),
    pathlib.Path("checkpoints_r710_seed2/best.pt"),
    pathlib.Path("checkpoints_r710_seed3/best.pt"),
    pathlib.Path("checkpoints_r710_seed4/best.pt"),
]
GRID_RES = 96
N_PROJ_ITERS = 5
NBR_SZ = 20
BAND_FACTOR = 3.0


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    points, normals, _ = R.load_points_from_h5(DATA_H5)
    points_norm, center, scale = R.normalize_points(points)
    gt_mesh = trimesh.load(str(GT_PLY), process=False)
    tree_gt = cKDTree(gt_mesh.vertices)

    print(f"loading {len(CKPTS)} ensemble members and evaluating grid_res={GRID_RES}^3 each...")
    models, zs = [], []
    udfs, confs = [], []
    grid_xyz = None
    for ckpt in CKPTS:
        t0 = time.time()
        model = R.load_model(ckpt, device)
        z = R.encode_points(model, points_norm, normals, device)
        gx, udf, _, conf = R.eval_grid(model, z, GRID_RES, device, points_norm)
        if grid_xyz is None:
            grid_xyz = gx
        models.append(model)
        zs.append(z)
        udfs.append(udf)
        confs.append(conf)
        print(f"  {ckpt}: eval done in {time.time()-t0:.1f}s "
              f"(udf mean={udf.mean():.4f} conf mean={conf.mean():.4f})")

    udfs = np.stack(udfs, axis=0)        # [5, G]
    confs = np.stack(confs, axis=0)      # [5, G]
    ens_mean_udf = udfs.mean(axis=0)
    ens_std_udf = udfs.std(axis=0)
    ens_std_udf_mm = ens_std_udf * scale
    print(f"\nensemble udf disagreement (mm): mean={ens_std_udf_mm.mean():.3f} "
          f"p50={np.percentile(ens_std_udf_mm,50):.3f} p90={np.percentile(ens_std_udf_mm,90):.3f} "
          f"p99={np.percentile(ens_std_udf_mm,99):.3f} max={ens_std_udf_mm.max():.3f}")

    # ---- [1] correlation check at R7-10-only baseline's worst outliers ----
    print("\n[1] reproducing R7-10-only baseline (conf_threshold=0.7, no R7-11/R7-12) "
          "with seed0 model to locate the known bad vertices...")
    model0, z0 = models[0], zs[0]
    mid_mesh0 = R.reconstruct_midsurface(
        model0, z0, device, points_norm, grid_res=GRID_RES, band_factor=BAND_FACTOR,
        n_proj_iters=N_PROJ_ITERS, nbr_sz=NBR_SZ, refine_rounds=2, conf_threshold=0.7,
        input_dist_threshold_mm=None, prune_dist_threshold_mm=None, scale=scale,
    )
    pts_world0 = np.asarray(mid_mesh0.points) * scale + center
    d_gt0, _ = tree_gt.query(pts_world0, k=1)
    worst = np.argsort(-d_gt0)[:30]
    worst_norm = ((pts_world0[worst] - center) / scale).astype(np.float32)

    # query ensemble disagreement AT these specific worst-vertex locations
    member_udfs_at_worst = []
    for model, z in zip(models, zs):
        u, _, _ = R.decode_chunked(model, z, worst_norm, device)
        member_udfs_at_worst.append(u)
    member_udfs_at_worst = np.stack(member_udfs_at_worst, axis=0)   # [5, 30]
    std_at_worst_mm = member_udfs_at_worst.std(axis=0) * scale

    print(f"  worst-30 Recon->GT vertices (R7-10-only baseline, mean d_gt={d_gt0[worst].mean():.1f}mm):")
    print(f"    ensemble_std at these points (mm): mean={std_at_worst_mm.mean():.3f} "
          f"min={std_at_worst_mm.min():.3f} max={std_at_worst_mm.max():.3f}")
    print(f"    (compare to whole-grid ensemble_std distribution above -- "
          f"is this elevated relative to the p90/p99 of the full grid?)")

    corr = np.corrcoef(std_at_worst_mm, d_gt0[worst])[0, 1]
    print(f"    corrcoef(ensemble_std, d_gt) over these 30 points = {corr:.4f}")

    # ---- [2] ensemble_std used ALONE as a gate, full reconstruction ----
    print("\n[2] ensemble_std used ALONE as a candidate gate (no R7-11/R7-12), "
          "full reconstruction with seed0 model...")
    for std_threshold_mm in [1.0, 0.5, 0.2]:
        gate_idx = np.where(ens_std_udf_mm <= std_threshold_mm)[0]
        if len(gate_idx) < NBR_SZ * 2:
            print(f"  std_threshold={std_threshold_mm}mm: only {len(gate_idx)} pts pass, skipping")
            continue
        target_k = min(len(ens_mean_udf), max(NBR_SZ * 2, int(BAND_FACTOR * GRID_RES ** 2)))
        k = min(len(gate_idx), target_k)
        order = np.argsort(ens_mean_udf[gate_idx])
        cand = grid_xyz[gate_idx[order[:k]]]
        proj_xyz, _ = R.gradient_project(model0, z0, cand, N_PROJ_ITERS, device)
        cloud = pv.PolyData(proj_xyz)
        coarse = cloud.reconstruct_surface(nbr_sz=NBR_SZ)
        pts_r = np.asarray(coarse.points)
        tris_r = coarse.faces.reshape(-1, 4)[:, 1:4]
        for _ in range(2):
            pts_r, tris_r = R.subdivide_and_project(model0, z0, pts_r, tris_r, device, N_PROJ_ITERS)
        final_world = pts_r * scale + center
        recon_mesh = trimesh.Trimesh(vertices=final_world, faces=tris_r, process=False)
        pa, _ = trimesh.sample.sample_surface(recon_mesh, 50000)
        d_r2g, _ = tree_gt.query(pa, k=1)
        pb, _ = trimesh.sample.sample_surface(gt_mesh, 50000)
        tree_recon = cKDTree(final_world)
        d_g2r, _ = tree_recon.query(pb, k=1)
        print(f"  std_threshold={std_threshold_mm}mm: gate_pool={len(gate_idx)}/{len(grid_xyz)} "
              f"selected={len(cand)} | Recon->GT mean={d_r2g.mean():.2f} "
              f"p90={np.percentile(d_r2g,90):.2f} max={d_r2g.max():.2f} | "
              f"GT->Recon mean={d_g2r.mean():.2f} max={d_g2r.max():.2f}")

    print("\n--- for reference, prior results on the same part ---")
    print("  R7-10 conf-gate ALONE (conf_threshold=0.7, refine=2): "
          "Recon->GT mean=53.83 max=244.77 | GT->Recon mean=1.58")
    print("  R7-11 distance-gate ALONE (input_dist=10mm, refine=2): "
          "Recon->GT mean=31.57 max=182.83 | GT->Recon mean~1.6")
    print("  R7-11+R7-12 combined (input_dist=10mm, prune=12mm, refine=3): "
          "Recon->GT mean=8.57 max=47.27 | GT->Recon mean=1.62")


if __name__ == "__main__":
    main()

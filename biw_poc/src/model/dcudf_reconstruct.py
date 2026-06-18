"""
dcudf_reconstruct.py -- prototype driver: run the ported DCUDF pipeline
(dcudf_extract.py) against this project's trained VecSetAE model on the
established body002 single-part overfit sanity-check (same convention as
validate_refine.py / diagnose_holes.py / diagnose_outliers.py), and report
both the historical vertex-NN surface_distances() metric (for direct
comparability with prior R7-08/R7-12/R7-13 numbers in CLAUDE.md) and the
true point-to-surface distance (trimesh.proximity.closest_point against a
densely/evenly resampled GT surface -- the methodology R7-13's
diagnose_holes.py investigation found more trustworthy; naive vertex-NN
distance conflates real coverage gaps with ordinary triangulation-density
interpolation error).

Usage:
    python dcudf_reconstruct.py --resolution 48 --max-iter 30 --smoke-test
    python dcudf_reconstruct.py --resolution 96 --max-iter 400
"""
import argparse
import functools
import pathlib
import time

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

import reconstruct as R
from dcudf_extract import DCUDFExtractor

print = functools.partial(print, flush=True)

DEFAULT_CKPT = pathlib.Path(r"C:\Users\hide2\IdeaBox\AutoMetalSheet\biw_poc\src\model\checkpoints_r710\best.pt")
DATA_H5 = pathlib.Path(
    r"C:\Users\hide2\IdeaBox\Mesh_Generater\output\A0072600002\bodies\body002_filled_dataset.h5")
GT_PLY = pathlib.Path(
    r"C:\Users\hide2\IdeaBox\Mesh_Generater\output\A0072600002\bodies\body002_filled_midsurface.ply")
N_SAMPLE = 50_000


def surface_distances_vertex_nn(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, n_sample=N_SAMPLE):
    """validate_refine.py's metric: sample(a)'s surface -> nearest VERTEX of b."""
    pa, _ = trimesh.sample.sample_surface(mesh_a, n_sample)
    tree_b = cKDTree(mesh_b.vertices)
    d, _ = tree_b.query(pa, k=1)
    return dict(mean=float(d.mean()), median=float(np.median(d)),
                p90=float(np.percentile(d, 90)), max=float(d.max()))


def surface_distances_true(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, n_sample=N_SAMPLE,
                            chunk=2000):
    """True point-to-SURFACE distance: sample(a) -> closest point on b's
    triangles (trimesh.proximity.closest_point), not just b's vertices.
    Chunked: trimesh's candidate-triangle broadcast can spike memory by
    orders of magnitude on some point batches (observed: 50k points against
    a 64829-face mesh tried to allocate a (116M,3) float64 array)."""
    pa, _ = trimesh.sample.sample_surface(mesh_a, n_sample)
    pa = np.asarray(pa)
    d_parts = []
    for i in range(0, len(pa), chunk):
        _, d_i, _ = trimesh.proximity.closest_point(mesh_b, pa[i:i + chunk])
        d_parts.append(d_i)
    d = np.concatenate(d_parts)
    return dict(mean=float(d.mean()), median=float(np.median(d)),
                p90=float(np.percentile(d, 90)), max=float(d.max()))


def build_query_func(model, z, device):
    """Differentiable pts[N,3] -> udf[N] adapter for DCUDFExtractor.
    Model parameters are frozen (requires_grad_(False)) so backward() only
    accumulates gradient into the query points, not the network weights --
    this is NOT wrapped in @torch.no_grad(), unlike every other decode call
    in reconstruct.py, because DCUDF's optimize() needs autograd through
    query_func back to the candidate vertex positions."""
    for p in model.parameters():
        p.requires_grad_(False)

    def query_func(pts: torch.Tensor) -> torch.Tensor:
        udf, _, _ = model.decode(pts.unsqueeze(0), z)
        return udf[0]

    return query_func


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=pathlib.Path, default=DEFAULT_CKPT)
    ap.add_argument("--resolution", type=int, default=96,
                     help="DCUDF marching-cubes grid resolution (production reconstruct.py "
                          "uses grid_res=96 for its rank-based candidate grid).")
    ap.add_argument("--threshold", type=float, default=None,
                     help="MC iso-value. Default: DCUDF README's resolution=256/threshold=0.005 "
                          "reference point, linearly rescaled to --resolution.")
    ap.add_argument("--laplacian-weight", type=float, default=None,
                     help="Default: DCUDF README's resolution=256/weight=2000 reference point, "
                          "linearly rescaled to --resolution (README also shows "
                          "256->64 paired with 2000->500, i.e. ~linear in resolution).")
    ap.add_argument("--max-iter", type=int, default=400)
    ap.add_argument("--normal-step", type=int, default=300)
    ap.add_argument("--region-rate", type=float, default=20)
    ap.add_argument("--is-cut", action="store_true", default=True)
    ap.add_argument("--no-cut", dest="is_cut", action="store_false")
    ap.add_argument("--bbox-pad-frac", type=float, default=0.15,
                     help="R7-07 bbox padding fraction, reused here for DCUDF's bound_min/max "
                          "(restricts the dense grid to near the real input cloud, not the "
                          "full anisotropic-normalization [-1,1]^3 cube).")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("body002_dcudf.ply"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.threshold is None:
        args.threshold = 0.005 * (256 / args.resolution)
    if args.laplacian_weight is None:
        args.laplacian_weight = 2000.0 * (args.resolution / 256)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} ckpt={args.ckpt}")
    print(f"resolution={args.resolution} threshold={args.threshold:.5f} "
          f"laplacian_weight={args.laplacian_weight:.1f} max_iter={args.max_iter} "
          f"normal_step={args.normal_step} is_cut={args.is_cut}")

    model = R.load_model(args.ckpt, device)
    points, normals, _ = R.load_points_from_h5(DATA_H5)
    points_norm, center, scale = R.normalize_points(points)
    z = R.encode_points(model, points_norm, normals, device)

    lo = points_norm.min(axis=0)
    hi = points_norm.max(axis=0)
    pad = (hi - lo) * args.bbox_pad_frac
    bound_min = (lo - pad).astype(np.float32)
    bound_max = (hi + pad).astype(np.float32)
    print(f"bound_min={bound_min} bound_max={bound_max} (R7-07 input-cloud bbox + "
          f"{args.bbox_pad_frac*100:.0f}% pad, NOT the full normalization cube)")

    query_func = build_query_func(model, z, device)

    extractor = DCUDFExtractor(
        query_func, args.resolution, args.threshold,
        max_iter=args.max_iter, normal_step=args.normal_step,
        laplacian_weight=args.laplacian_weight,
        bound_min=bound_min, bound_max=bound_max,
        is_cut=args.is_cut, region_rate=args.region_rate,
        max_batch=100_000, device=device,
    )

    t0 = time.time()
    recon_mesh_norm = extractor.optimize()
    total_time = time.time() - t0
    print(f"[timing] DCUDF optimize() total: {total_time:.1f}s "
          f"(extract_fields={extractor.extract_time:.1f}s, "
          f"cut={extractor.cut_time if extractor.cut_time is not None else 'N/A'})")
    print(f"recon mesh: {len(recon_mesh_norm.vertices)} verts, {len(recon_mesh_norm.faces)} faces")

    pts_world = recon_mesh_norm.vertices * scale + center
    recon_mesh_world = trimesh.Trimesh(vertices=pts_world, faces=recon_mesh_norm.faces, process=False)
    recon_mesh_world.export(str(args.out))
    print(f"saved: {args.out}")

    gt_mesh = trimesh.load(str(GT_PLY), process=False)
    print(f"GT mid-surface: {gt_mesh.vertices.shape} verts, {gt_mesh.faces.shape} faces")

    g2r_nn = surface_distances_vertex_nn(gt_mesh, recon_mesh_world)
    r2g_nn = surface_distances_vertex_nn(recon_mesh_world, gt_mesh)
    g2r_true = surface_distances_true(gt_mesh, recon_mesh_world)
    r2g_true = surface_distances_true(recon_mesh_world, gt_mesh)

    print("\n=== vertex-NN methodology (validate_refine.py-compatible) ===")
    print(f"  GT->Recon mean={g2r_nn['mean']:.2f} median={g2r_nn['median']:.2f} "
          f"p90={g2r_nn['p90']:.2f} max={g2r_nn['max']:.2f}")
    print(f"  Recon->GT mean={r2g_nn['mean']:.2f} median={r2g_nn['median']:.2f} "
          f"p90={r2g_nn['p90']:.2f} max={r2g_nn['max']:.2f}")

    print("\n=== true point-to-surface methodology (diagnose_holes.py-style) ===")
    print(f"  GT->Recon mean={g2r_true['mean']:.2f} median={g2r_true['median']:.2f} "
          f"p90={g2r_true['p90']:.2f} max={g2r_true['max']:.2f}")
    print(f"  Recon->GT mean={r2g_true['mean']:.2f} median={r2g_true['median']:.2f} "
          f"p90={r2g_true['p90']:.2f} max={r2g_true['max']:.2f}")

    gt_area = gt_mesh.area
    recon_area = recon_mesh_world.area
    print(f"\narea_ratio (recon/GT): {recon_area/gt_area:.3f} (GT area={gt_area:.0f}mm^2, "
          f"recon area={recon_area:.0f}mm^2)")


if __name__ == "__main__":
    main()

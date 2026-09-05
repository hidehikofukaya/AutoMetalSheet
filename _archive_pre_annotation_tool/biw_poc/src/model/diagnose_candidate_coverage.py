"""
diagnose_candidate_coverage.py -- Tier 0 (CLAUDE.md Round 15 / R20-01) diagnostic
for the "normal orientation anomaly" (HANDOFF_codex.md SS0).

Read-only / non-destructive: imports reconstruct_softmin_guidance.py and runs the
SAME candidate-selection + bounded-projection pipeline it uses internally, but
stops BEFORE the `pv.PolyData(...).reconstruct_surface()` call so the candidate
cloud actually fed into VTK can be inspected directly. Nothing in
reconstruct_softmin_guidance.py or diagnose_softmin_guidance.py is modified.

Two independent measurements, both in WORLD mm:

  1. Candidate-cloud local density: for each projected candidate point, the
     distance to its k-th nearest neighbour WITHIN the candidate cloud itself
     (k = --nbr-sz, matching the same parameter PyVista's Hoppe-type
     reconstruct_surface() uses for local tangent-plane fitting). A wide
     density spread here means the implicit local-plane fit VTK performs is
     well-conditioned in some places and starved of neighbours in others.

  2. GT coverage gap: the candidate cloud's TRUE coverage of the GT
     mid-surface, measured by densely resampling the GT mesh surface (NOT
     using GT vertices, which are non-uniformly distributed) and measuring
     each sample's distance to the nearest candidate point. Reports the
     fraction beyond --input-dist-threshold-mm (the same support-gate radius
     reconstruct_coarse_surface() already enforces against the INPUT cloud,
     applied here against the post-projection CANDIDATE cloud instead).

If --recon-ply is given, additionally correlates each reconstructed vertex's
local candidate density (distance to its nearest candidate's k-th neighbour)
against its along-input-normal deviation (the same decomposition
diagnose_softmin_guidance.py's recon_orientation_report computes), to test
the study/12 hypothesis that sparse/uneven candidate density -- not a wrong
direction-head prediction -- drives the normal misorientation.

Usage:
    python diagnose_candidate_coverage.py \
        --checkpoint checkpoints_softmin_r717_p1only/best.pt \
        --data ../../experiments/loss_smoothing/softmin_r717/body002_filled_dataset.h5 \
        --gt-ply ../../experiments/loss_smoothing/softmin_r717/body002_filled_midsurface.ply \
        --recon-ply ../../experiments/loss_smoothing/softmin_r717/diag_tier0/body002_p1only_fresh_recon.ply
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import trimesh
from scipy.spatial import cKDTree

import reconstruct_softmin_guidance as RSG


def _print_stats(name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    print(f"  {name:42s} mean={arr.mean():9.3f} median={np.median(arr):9.3f} "
          f"p10={np.percentile(arr, 10):9.3f} p90={np.percentile(arr, 90):9.3f} "
          f"p95={np.percentile(arr, 95):9.3f} max={arr.max():9.3f}  (n={len(arr)})")


def build_candidate_cloud(
    model, cloud, device, *,
    grid_res: int, candidate_factor: float, projection_iterations: int,
    nbr_sz: int, max_input_distance_mm: float | None, max_step_mm: float,
    ambiguity_damping: float, bbox_pad_fraction: float,
    ranking_mode: str, ambiguity_penalty_mm: float, chunk_size: int,
) -> np.ndarray:
    """Re-run reconstruct_coarse_surface() up to (not including) the VTK call.

    Mirrors reconstruct_softmin_guidance.reconstruct_coarse_surface() lines
    585-638 exactly (same function calls, same order, same gate re-check),
    returning the candidate cloud in WORLD mm instead of handing it to VTK.
    """
    latent = RSG.encode_input(model, cloud, device)
    guidance = RSG.evaluate_grid(
        model, latent, cloud.points_norm, grid_res, device,
        bbox_pad_fraction=bbox_pad_fraction, chunk_size=chunk_size,
    )

    minimum_required = max(3, nbr_sz * 2)
    candidate_count = max(minimum_required, int(candidate_factor * grid_res * grid_res))
    candidate_indices = RSG.select_ranked_candidates(
        guidance.xyz, guidance.soft_potential,
        input_points_norm=cloud.points_norm, scale_mm=cloud.scale_mm,
        max_input_distance_mm=max_input_distance_mm,
        candidate_count=candidate_count, minimum_required=minimum_required,
        step_distance=guidance.step_distance, branch_ambiguity=guidance.branch_ambiguity,
        ranking_mode=ranking_mode, ambiguity_penalty_mm=ambiguity_penalty_mm,
    )
    projected = RSG.project_candidates(
        model, latent, guidance.xyz[candidate_indices], device,
        scale_mm=cloud.scale_mm, max_step_mm=max_step_mm,
        iterations=projection_iterations, ambiguity_damping=ambiguity_damping,
        chunk_size=chunk_size,
    )
    projected_gate = RSG._input_gate_indices(
        projected, cloud.points_norm, cloud.scale_mm, max_input_distance_mm,
    )
    if len(projected_gate) < minimum_required:
        raise RSG.InsufficientEvidenceError(
            f"Only {len(projected_gate)} projected candidates remain within "
            f"the input-cloud gate; minimum is {minimum_required}."
        )
    projected = projected[projected_gate]
    return (projected * cloud.scale_mm + cloud.center).astype(np.float32, copy=False)


def candidate_density_report(candidates_world: np.ndarray, nbr_sz: int) -> np.ndarray:
    print(f"\n=== candidate-cloud local density (n={len(candidates_world)}, "
          f"k={nbr_sz}-NN within candidate cloud, WORLD mm) ===")
    tree = cKDTree(candidates_world)
    k = min(nbr_sz + 1, len(candidates_world))
    dist, _ = tree.query(candidates_world, k=k)
    knn_radius = dist[:, -1]  # distance to k-th neighbour (excludes self at col 0)
    _print_stats(f"dist to {k - 1}-th nearest candidate", knn_radius)
    spread = knn_radius.max() / max(knn_radius.min(), 1e-9)
    print(f"  max/min ratio = {spread:.1f}x "
          f"({'highly uneven density' if spread > 10 else 'roughly uniform'})")
    return knn_radius


def gt_coverage_gap_report(
    candidates_world: np.ndarray, gt_mesh: trimesh.Trimesh,
    threshold_mm: float, n_samples: int,
) -> np.ndarray:
    print(f"\n=== candidate-cloud -> GT coverage gap "
          f"(GT resampled densely to n={n_samples}, threshold={threshold_mm:.1f}mm) ===")
    samples, _ = trimesh.sample.sample_surface(gt_mesh, n_samples)
    tree = cKDTree(candidates_world)
    dist, _ = tree.query(samples, k=1)
    _print_stats("dist from GT sample to nearest candidate", dist)
    frac_beyond = float((dist > threshold_mm).mean())
    print(f"  fraction of GT surface beyond {threshold_mm:.1f}mm from any candidate: "
          f"{100 * frac_beyond:.1f}%")
    return dist


def correlate_with_recon(
    recon_mesh: trimesh.Trimesh, cloud, candidates_world: np.ndarray, nbr_sz: int,
) -> None:
    print(f"\n=== local candidate density vs recon vertex along-normal deviation ===")
    input_points_world = cloud.points_norm * cloud.scale_mm + cloud.center
    input_normals = cloud.normals
    tree_input = cKDTree(input_points_world)
    tree_cand = cKDTree(candidates_world)

    _, idx_in = tree_input.query(recon_mesh.vertices, k=1)
    nearest_normal = input_normals[idx_in]
    disp = recon_mesh.vertices - input_points_world[idx_in]
    along_normal = np.abs(np.einsum("ij,ij->i", disp, nearest_normal))

    k = min(nbr_sz + 1, len(candidates_world))
    dist_to_cand, _ = tree_cand.query(recon_mesh.vertices, k=k)
    local_density = dist_to_cand[:, -1]

    corr = float(np.corrcoef(local_density, along_normal)[0, 1])
    print(f"  corrcoef(local candidate sparsity, |along-normal| deviation) = {corr:.4f}")
    print(f"  (close to 1 => sparse/uneven candidate density drives the normal "
          f"misorientation, supporting the study/12 VTK-extrapolation hypothesis;\n"
          f"   close to 0 => deviation is unrelated to local density, pointing back "
          f"at the model's direction_toward_surface prediction instead)")
    _print_stats("local candidate density (k-th NN dist, mm)", local_density)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", "--ckpt", dest="checkpoint", type=pathlib.Path, required=True)
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--gt-ply", type=pathlib.Path, required=True)
    ap.add_argument("--recon-ply", type=pathlib.Path, default=None,
                     help="Optional: correlate local candidate density with this "
                          "already-reconstructed mesh's normal deviation")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--grid-res", type=int, default=48)
    ap.add_argument("--candidate-factor", type=float, default=3.0)
    ap.add_argument("--projection-iterations", type=int, default=3)
    ap.add_argument("--nbr-sz", type=int, default=20)
    ap.add_argument("--input-dist-threshold-mm", type=float, default=10.0)
    ap.add_argument("--no-input-gate", action="store_true")
    ap.add_argument("--max-step-mm", type=float, default=5.0)
    ap.add_argument("--ambiguity-damping", type=float, default=1.0)
    ap.add_argument("--bbox-pad-fraction", type=float, default=0.15)
    ap.add_argument("--ranking-mode", choices=RSG.RANKING_MODES, default="abs_potential_ambiguity")
    ap.add_argument("--ambiguity-penalty-mm", type=float, default=1.0)
    ap.add_argument("--chunk-size", type=int, default=100_000)
    ap.add_argument("--gt-sample-count", type=int, default=50_000,
                     help="Number of dense uniform samples drawn from the GT surface")
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    model = RSG.load_model(args.checkpoint, device)
    cloud = RSG.load_input_h5(args.data)
    gt_mesh = trimesh.load(str(args.gt_ply), process=False)

    max_input_distance_mm = None if args.no_input_gate else args.input_dist_threshold_mm
    candidates_world = build_candidate_cloud(
        model, cloud, device,
        grid_res=args.grid_res, candidate_factor=args.candidate_factor,
        projection_iterations=args.projection_iterations, nbr_sz=args.nbr_sz,
        max_input_distance_mm=max_input_distance_mm, max_step_mm=args.max_step_mm,
        ambiguity_damping=args.ambiguity_damping, bbox_pad_fraction=args.bbox_pad_fraction,
        ranking_mode=args.ranking_mode, ambiguity_penalty_mm=args.ambiguity_penalty_mm,
        chunk_size=args.chunk_size,
    )
    print(f"candidate cloud (post-projection, post-gate, pre-VTK): {len(candidates_world)} points")

    candidate_density_report(candidates_world, args.nbr_sz)
    gt_coverage_gap_report(candidates_world, gt_mesh, args.input_dist_threshold_mm, args.gt_sample_count)

    if args.recon_ply:
        recon_mesh = trimesh.load(str(args.recon_ply), process=False)
        correlate_with_recon(recon_mesh, cloud, candidates_world, args.nbr_sz)


if __name__ == "__main__":
    main()

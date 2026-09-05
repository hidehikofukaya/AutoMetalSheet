"""
diagnose_softmin_guidance.py -- ad-hoc diagnostic (NOT part of the Stage A/B/C
pipeline) for the SoftminGuidanceModel pipeline (softmin_guidance.py /
reconstruct_softmin_guidance.py). Two independent reports, selected by flags:

  --cross-section   Report the guidance-field (soft_potential / step_distance /
                     branch_ambiguity) distribution within a thin slab around an
                     axis-aligned plane through the part, in WORLD mm coordinates.
                     If --gt-ply is given, also reports the TRUE distance to the
                     GT mid-surface at every slice point and flags "ghost" grid
                     points where the model claims to be near the surface
                     (|soft_potential| small) while actually being far from it.

  --recon-ply PATH  Report mesh orientation/connectivity diagnostics for an
                     already-reconstructed world-coordinate PLY: connected
                     component count/sizes, per-component PCA shape
                     (panel-like vs spike/needle-like), and per-vertex/per-face
                     deviation decomposed into an along-input-normal component
                     vs a lateral (tangent-plane) component relative to the
                     nearest Stage-A INPUT point's normal. This tells apart two
                     different failure modes: deviation concentrated ALONG the
                     local surface normal points at a guidance-field/projection
                     bug, while deviation that is mostly LATERAL points at a
                     candidate-selection/VTK-stitching bug.

Ground-truth distance comparisons always use trimesh.proximity.closest_point()
(NOT trimesh.proximity.ProximityQuery(...).on_surface(), which was found to
return grossly incorrect results -- e.g. ~196mm mean self-distance for a mesh
queried against its own vertices -- on this project's open/non-watertight
mid-surface meshes).

Usage:
    python diagnose_softmin_guidance.py --checkpoint ckpt/best.pt --data body002_filled_dataset.h5 \
        --gt-ply body002_filled_midsurface.ply --cross-section --axis z --slice-frac 0.5

    python diagnose_softmin_guidance.py --data body002_filled_dataset.h5 \
        --recon-ply body002_guidance_codex_midsurface.ply --gt-ply body002_filled_midsurface.ply
"""
import argparse
import pathlib

import numpy as np
import trimesh
from scipy.spatial import cKDTree

import reconstruct_softmin_guidance as RSG


def _print_stats(name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    print(f"  {name:34s} mean={arr.mean():9.3f} median={np.median(arr):9.3f} "
          f"p10={np.percentile(arr, 10):9.3f} p90={np.percentile(arr, 90):9.3f} "
          f"min={arr.min():9.3f} max={arr.max():9.3f}  (n={len(arr)})")


def boundary_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    """Vertices touching an edge used by exactly one face (open boundary)."""
    edges_sorted = mesh.edges_sorted
    groups = trimesh.grouping.group_rows(edges_sorted, require_count=1)
    boundary_edges = mesh.edges[groups]
    idx = np.unique(boundary_edges)
    return mesh.vertices[idx]


def _classify_shape(eigvals_desc: np.ndarray) -> str:
    e0, e1, e2 = np.maximum(eigvals_desc, 1e-9)
    if e0 / e1 > 5.0 and e1 / e2 < 5.0:
        return "spike/needle"
    if e1 / e2 > 5.0:
        return "panel/flat"
    return "blob/irregular"


def cross_section_report(
    model, cloud, device, axis: str, slice_mm: float | None,
    slice_thickness_mm: float, grid_res: int, bbox_pad_fraction: float,
    gt_mesh: trimesh.Trimesh | None, top_n: int,
) -> dict:
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]

    latent = RSG.encode_input(model, cloud, device)
    grid = RSG.evaluate_grid(
        model, latent, cloud.points_norm, grid_res, device,
        bbox_pad_fraction=bbox_pad_fraction,
    )
    world_xyz = grid.xyz * cloud.scale_mm + cloud.center

    if slice_mm is None:
        slice_mm = float(np.median(world_xyz[:, axis_idx]))

    sel = np.abs(world_xyz[:, axis_idx] - slice_mm) <= slice_thickness_mm / 2.0
    if sel.sum() == 0:
        raise ValueError(
            f"no evaluated grid points within {slice_thickness_mm}mm of "
            f"{axis}={slice_mm:.2f}; widen --slice-thickness-mm or change --slice-mm"
        )

    xyz_sel = world_xyz[sel]
    pot_mm = grid.soft_potential[sel] * cloud.scale_mm
    step_mm = grid.step_distance[sel] * cloud.scale_mm
    amb = grid.branch_ambiguity[sel]

    d_input_norm, _ = cKDTree(cloud.points_norm).query(grid.xyz[sel], k=1)
    d_input_mm = d_input_norm * cloud.scale_mm

    print(f"\n=== cross-section: axis={axis} value={slice_mm:.2f}mm "
          f"thickness={slice_thickness_mm:.2f}mm -> {sel.sum()} grid points "
          f"(grid_res={grid_res}) ===")
    _print_stats("soft_potential (mm, signed)", pot_mm)
    _print_stats("step_distance (mm)", step_mm)
    _print_stats("branch_ambiguity [0,1]", amb)
    _print_stats("dist to input cloud (mm)", d_input_mm)

    out = dict(
        axis=axis, slice_mm=slice_mm, world_xyz=xyz_sel,
        soft_potential_mm=pot_mm, step_distance_mm=step_mm,
        branch_ambiguity=amb, d_input_mm=d_input_mm,
    )

    if gt_mesh is not None:
        _, d_gt, _ = trimesh.proximity.closest_point(gt_mesh, xyz_sel)
        _print_stats("TRUE dist to GT (mm)", d_gt)
        corr_pot = float(np.corrcoef(pot_mm, d_gt)[0, 1])
        corr_step = float(np.corrcoef(step_mm, d_gt)[0, 1])
        print(f"  corrcoef(soft_potential, true_dist_to_GT) = {corr_pot:.4f}")
        print(f"  corrcoef(step_distance,  true_dist_to_GT) = {corr_step:.4f}")

        ghost = (np.abs(pot_mm) < 2.0) & (d_gt > 10.0)
        print(f"  'ghost' points (|soft_potential|<2mm but true dist>10mm): "
              f"{ghost.sum()}/{sel.sum()} ({100 * ghost.mean():.1f}%)")
        if ghost.sum() > 0:
            idxs = np.flatnonzero(ghost)
            order = idxs[np.argsort(-d_gt[idxs])][:top_n]
            print(f"  worst ghost points (true_d_gt, pred_potential, pred_step, world_xyz):")
            for gi in order:
                print(f"    d_gt={d_gt[gi]:7.2f}mm pot={pot_mm[gi]:7.2f}mm "
                      f"step={step_mm[gi]:6.2f}mm xyz=({xyz_sel[gi,0]:.1f},"
                      f"{xyz_sel[gi,1]:.1f},{xyz_sel[gi,2]:.1f})")

        out["d_gt_mm"] = d_gt

    return out


def recon_orientation_report(
    recon_mesh: trimesh.Trimesh, cloud, gt_mesh: trimesh.Trimesh | None, top_n: int,
) -> None:
    print(f"\n=== recon orientation: {len(recon_mesh.vertices)} verts, "
          f"{len(recon_mesh.faces)} faces ===")

    input_points_world = cloud.points_norm * cloud.scale_mm + cloud.center
    input_normals = cloud.normals
    tree_input = cKDTree(input_points_world)

    # whole-mesh along-normal vs lateral decomposition (vertex-level)
    d_in, idx_in = tree_input.query(recon_mesh.vertices, k=1)
    nearest_normal = input_normals[idx_in]
    disp = recon_mesh.vertices - input_points_world[idx_in]
    along_normal = np.einsum("ij,ij->i", disp, nearest_normal)
    lateral = np.linalg.norm(disp - along_normal[:, None] * nearest_normal, axis=1)
    print("whole-mesh deviation, decomposed relative to nearest input point's normal:")
    _print_stats("|along-normal| component (mm)", np.abs(along_normal))
    _print_stats("lateral (tangent-plane) component (mm)", lateral)
    frac_normal_dominant = float((np.abs(along_normal) > lateral).mean())
    print(f"  fraction of vertices where |along-normal| > lateral: "
          f"{frac_normal_dominant:.3f}  "
          f"({'normal/projection-dominated' if frac_normal_dominant > 0.5 else 'lateral/stitching-dominated'})")

    # whole-mesh face-normal misorientation
    face_centers = recon_mesh.triangles_center
    face_normals = recon_mesh.face_normals
    _, idx_in_f = tree_input.query(face_centers, k=1)
    align = np.einsum("ij,ij->i", face_normals, input_normals[idx_in_f])
    misoriented = np.abs(align) < 0.5
    print(f"face-normal alignment vs nearest input normal: mean(cos)={align.mean():.3f} "
          f"frac(|cos|<0.5, i.e. >60deg off)={misoriented.mean():.3f}")

    if gt_mesh is not None:
        _, d_gt_all, _ = trimesh.proximity.closest_point(gt_mesh, recon_mesh.vertices)
        _print_stats("Recon -> GT true distance (mm)", d_gt_all)
        bnd_pts = boundary_vertices(gt_mesh)
        d_bnd, _ = cKDTree(bnd_pts).query(recon_mesh.vertices, k=1)
        corr_bnd = float(np.corrcoef(d_gt_all, d_bnd)[0, 1])
        print(f"  corrcoef(Recon->GT distance, dist to nearest GT boundary edge) = {corr_bnd:.4f} "
              f"(close to 1 => errors concentrated at cut-locus boundary, per TD-16)")

    components = recon_mesh.split(only_watertight=False)
    sizes = sorted((len(c.vertices) for c in components), reverse=True)
    print(f"\nconnected components: {len(components)}  sizes(desc, top 15)={sizes[:15]}")

    rows = []
    for ci, comp in enumerate(components):
        verts = comp.vertices
        if len(verts) >= 3:
            centered = verts - verts.mean(axis=0)
            eigvals = np.sort(np.linalg.eigvalsh(np.cov(centered.T)))[::-1]
            shape = _classify_shape(eigvals)
            eig10 = float(eigvals[0] / max(eigvals[1], 1e-9))
            eig21 = float(eigvals[1] / max(eigvals[2], 1e-9))
        else:
            shape, eig10, eig21 = "n/a", float("nan"), float("nan")

        d_in_c, idx_in_c = tree_input.query(verts, k=1)
        nn_c = input_normals[idx_in_c]
        disp_c = verts - input_points_world[idx_in_c]
        along_c = np.abs(np.einsum("ij,ij->i", disp_c, nn_c))
        lat_c = np.linalg.norm(disp_c - np.einsum("ij,ij->i", disp_c, nn_c)[:, None] * nn_c, axis=1)

        d_gt_mean = d_gt_max = None
        if gt_mesh is not None:
            _, d_gt_c, _ = trimesh.proximity.closest_point(gt_mesh, verts)
            d_gt_mean, d_gt_max = float(d_gt_c.mean()), float(d_gt_c.max())

        rows.append(dict(
            idx=ci, n=len(verts), shape=shape, eig10=eig10, eig21=eig21,
            along_mean=float(along_c.mean()), along_max=float(along_c.max()),
            lateral_mean=float(lat_c.mean()), lateral_max=float(lat_c.max()),
            d_gt_mean=d_gt_mean, d_gt_max=d_gt_max,
        ))

    rows.sort(key=lambda r: -r["n"])
    print(f"\nper-component report (top {top_n} by vertex count):")
    for r in rows[:top_n]:
        line = (f"  comp#{r['idx']:3d} n={r['n']:5d} shape={r['shape']:13s} "
                f"eig[0/1]={r['eig10']:7.2f} eig[1/2]={r['eig21']:7.2f}  "
                f"along: mean={r['along_mean']:6.2f} max={r['along_max']:6.2f}  "
                f"lateral: mean={r['lateral_mean']:6.2f} max={r['lateral_max']:6.2f}")
        if r["d_gt_mean"] is not None:
            line += f"  d_GT: mean={r['d_gt_mean']:6.2f} max={r['d_gt_max']:6.2f}"
        print(line)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", "--ckpt", dest="checkpoint", type=pathlib.Path, default=None,
                     help="Required for --cross-section")
    ap.add_argument("--data", type=pathlib.Path, required=True,
                     help="Stage-A H5 (points/normals/thickness_mm)")
    ap.add_argument("--gt-ply", type=pathlib.Path, default=None,
                     help="GT mid-surface mesh for true-distance comparisons")
    ap.add_argument("--recon-ply", type=pathlib.Path, default=None,
                     help="Reconstructed mesh (world coords) for orientation diagnostics")
    ap.add_argument("--cross-section", action="store_true",
                     help="Run the UDF/guidance-field cross-section report")
    ap.add_argument("--axis", choices=["x", "y", "z"], default="z")
    ap.add_argument("--slice-mm", type=float, default=None,
                     help="Plane position in world mm along --axis (default: median of grid)")
    ap.add_argument("--slice-frac", type=float, default=None,
                     help="Alternative to --slice-mm: 0..1 fraction across the input cloud's bbox")
    ap.add_argument("--slice-thickness-mm", type=float, default=3.0)
    ap.add_argument("--grid-res", type=int, default=64)
    ap.add_argument("--bbox-pad-fraction", type=float, default=0.15)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--save-npz", type=pathlib.Path, default=None,
                     help="Optional path to dump raw cross-section arrays for external plotting")
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    if not args.cross_section and args.recon_ply is None:
        _build_parser().error("specify --cross-section and/or --recon-ply")

    cloud = RSG.load_input_h5(args.data)
    gt_mesh = trimesh.load(str(args.gt_ply), process=False) if args.gt_ply else None

    if args.cross_section:
        if args.checkpoint is None:
            _build_parser().error("--checkpoint is required for --cross-section")
        import torch
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        model = RSG.load_model(args.checkpoint, device)

        slice_mm = args.slice_mm
        if slice_mm is None and args.slice_frac is not None:
            axis_idx = {"x": 0, "y": 1, "z": 2}[args.axis]
            world_pts = cloud.points_norm * cloud.scale_mm + cloud.center
            lo, hi = world_pts[:, axis_idx].min(), world_pts[:, axis_idx].max()
            slice_mm = float(lo + args.slice_frac * (hi - lo))

        result = cross_section_report(
            model, cloud, device, args.axis, slice_mm, args.slice_thickness_mm,
            args.grid_res, args.bbox_pad_fraction, gt_mesh, args.top_n,
        )
        if args.save_npz:
            np.savez(args.save_npz, **result)
            print(f"\nsaved slice arrays -> {args.save_npz}")

    if args.recon_ply:
        recon_mesh = trimesh.load(str(args.recon_ply), process=False)
        recon_orientation_report(recon_mesh, cloud, gt_mesh, args.top_n)


if __name__ == "__main__":
    main()

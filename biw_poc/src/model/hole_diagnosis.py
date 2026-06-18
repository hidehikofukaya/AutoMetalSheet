"""
hole_diagnosis.py -- ad-hoc diagnostic (NOT part of the Stage A/B/C pipeline):
fact-check the "moth-eaten mesh" defect visible in poc_result_viewer.py for
body002_r711r712_stage_refine3_midsurface.ply.

Checks:
  [1] GT mid-surface boundary-loop topology: how many distinct open-boundary
      loops does the GT mesh have (outer perimeter + every hole), and their
      sizes? This tests whether the part has many internal cut loci (holes),
      not just one outer boundary -- a topology fact the R7-09/R7-11/R7-12
      design implicitly assumed was "the boundary" (singular).
  [2] Recon mesh (final, refine3) boundary-loop count, for direct comparison.
      If Recon has far more loops than GT, that is direct quantitative
      evidence of spurious (model/pipeline-introduced) holes vs. genuine
      part features (real holes that already exist in GT).
  [3] Sparse-input-cloud coverage check: for every GT mesh vertex, distance
      to the nearest point in the h5 *training input* cloud (8192 pts,
      world mm). Reports what fraction of true surface area is farther than
      R7-11's input_dist_threshold_mm (10mm) / R7-12's prune_dist_threshold_mm
      (12mm) from the nearest input sample -- i.e. how much of the GT surface
      these gates would exclude/prune purely due to input-cloud sparsity,
      independent of model accuracy.
  [4] Vertex-distance vs face-distance metric comparison on the SAME pair
      (GT, Recon refine3): validate_refine.py/diagnose_outliers.py use
      nearest-VERTEX distance (cKDTree on mesh.vertices). This recomputes
      using nearest-FACE (surface) distance via trimesh.proximity, which is
      sensitive to small holes that nearest-vertex distance can mask (a GT
      sample point sitting exactly inside a small Recon hole can still be
      "close" to a vertex on the hole's rim even though no actual surface
      exists there).

Usage:
    python hole_diagnosis.py
"""
import pathlib

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import reconstruct as R

DATA_H5 = pathlib.Path(
    r"C:\Users\hide2\IdeaBox\Mesh_Generater\output\A0072600002\bodies\body002_filled_dataset.h5")
GT_PLY = pathlib.Path(
    r"C:\Users\hide2\IdeaBox\Mesh_Generater\output\A0072600002\bodies\body002_filled_midsurface.ply")
RECON_PLY = pathlib.Path("body002_r711r712_stage_refine3_midsurface.ply")

INPUT_DIST_THRESHOLD_MM = 10.0
PRUNE_DIST_THRESHOLD_MM = 12.0


def boundary_loops(mesh: trimesh.Trimesh):
    """Return list of boundary loops, each as an array of vertex indices,
    via connected-components over the open-boundary-edge graph."""
    edges_sorted = mesh.edges_sorted
    groups = trimesh.grouping.group_rows(edges_sorted, require_count=1)
    if len(groups) == 0:
        return []
    b_edges = mesh.edges[groups]
    verts = np.unique(b_edges)
    remap = {v: i for i, v in enumerate(verts)}
    rows = [remap[e[0]] for e in b_edges] + [remap[e[1]] for e in b_edges]
    cols = [remap[e[1]] for e in b_edges] + [remap[e[0]] for e in b_edges]
    n = len(verts)
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    n_comp, labels = connected_components(adj, directed=False)
    loops = []
    for c in range(n_comp):
        idx = verts[labels == c]
        loops.append(idx)
    return loops


def summarize_loops(name: str, mesh: trimesh.Trimesh):
    loops = boundary_loops(mesh)
    sizes = sorted([len(l) for l in loops], reverse=True)
    print(f"\n[{name}] {mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces, "
          f"surface_area={mesh.area:.1f}mm^2")
    print(f"  boundary loops: {len(loops)} total")
    if loops:
        print(f"    largest 5 loop sizes (n_verts): {sizes[:5]}")
        print(f"    smallest 5 loop sizes (n_verts): {sizes[-5:]}")
        n_tiny = sum(1 for s in sizes if s <= 6)
        print(f"    loops with <=6 verts (likely spurious moth-eaten micro-holes): "
              f"{n_tiny} ({100*n_tiny/len(loops):.1f}%)")
    return loops, sizes


def main():
    gt_mesh = trimesh.load(str(GT_PLY), process=False)
    recon_mesh = trimesh.load(str(RECON_PLY), process=False)

    print("=" * 78)
    print("[1]+[2] boundary-loop topology comparison")
    print("=" * 78)
    gt_loops, gt_sizes = summarize_loops("GT", gt_mesh)
    recon_loops, recon_sizes = summarize_loops("Recon (refine3)", recon_mesh)
    print(f"\n  GT has {len(gt_loops)} boundary loops (1 outer perimeter + "
          f"{len(gt_loops)-1} real part features e.g. bolt/rivet holes, "
          f"assuming the single largest loop is the outer perimeter).")
    print(f"  Recon has {len(recon_loops)} boundary loops.")
    print(f"  => excess loops in Recon vs GT: {len(recon_loops) - len(gt_loops)} "
          f"(spurious-hole signal if >> 0)")

    print("\n" + "=" * 78)
    print("[3] sparse-input-cloud coverage vs R7-11/R7-12 thresholds")
    print("=" * 78)
    points, normals, thickness = R.load_points_from_h5(DATA_H5)
    print(f"  training input cloud: {len(points)} pts (world mm)")
    tree_input = cKDTree(points)
    d_in, _ = tree_input.query(gt_mesh.vertices, k=1)
    print(f"  GT vertex -> nearest INPUT point distance (mm): "
          f"mean={d_in.mean():.2f} median={np.median(d_in):.2f} "
          f"p90={np.percentile(d_in,90):.2f} max={d_in.max():.2f}")
    for thr, label in [(INPUT_DIST_THRESHOLD_MM, "R7-11 input_dist_threshold_mm"),
                        (PRUNE_DIST_THRESHOLD_MM, "R7-12 prune_dist_threshold_mm")]:
        frac = (d_in > thr).mean()
        print(f"  fraction of GT vertices farther than {thr}mm ({label}): "
              f"{100*frac:.2f}%  ({int(frac*len(d_in))}/{len(d_in)} verts)")

    print("\n" + "=" * 78)
    print("[4] nearest-VERTEX vs nearest-FACE distance, GT<->Recon (refine3)")
    print("=" * 78)
    n_sample = 50_000
    gt_samples, _ = trimesh.sample.sample_surface(gt_mesh, n_sample)

    tree_recon_verts = cKDTree(recon_mesh.vertices)
    d_vert, _ = tree_recon_verts.query(gt_samples, k=1)

    closest, d_face, _ = trimesh.proximity.closest_point(recon_mesh, gt_samples)

    print(f"  GT->Recon nearest-VERTEX distance (mm): mean={d_vert.mean():.2f} "
          f"median={np.median(d_vert):.2f} p90={np.percentile(d_vert,90):.2f} "
          f"max={d_vert.max():.2f}")
    print(f"  GT->Recon nearest-FACE   distance (mm): mean={d_face.mean():.2f} "
          f"median={np.median(d_face):.2f} p90={np.percentile(d_face,90):.2f} "
          f"max={d_face.max():.2f}")
    gap = d_face - d_vert
    print(f"  face-vs-vertex gap (mm): mean={gap.mean():.2f} max={gap.max():.2f} "
          f"(large positive gap = sample sits near a vertex but far from any "
          f"actual face -> hiding a coverage hole)")
    for thr in [2.0, 5.0, 10.0]:
        frac = (d_face > thr).mean()
        frac_v = (d_vert > thr).mean()
        print(f"  fraction of GT samples with nearest-FACE dist > {thr}mm: "
              f"{100*frac:.2f}%   (nearest-VERTEX dist > {thr}mm: {100*frac_v:.2f}%)")


if __name__ == "__main__":
    main()

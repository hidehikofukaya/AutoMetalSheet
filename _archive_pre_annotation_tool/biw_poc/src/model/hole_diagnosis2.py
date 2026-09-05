"""
hole_diagnosis2.py -- follow-up to hole_diagnosis.py: characterize the giant
(9589-vertex) boundary loop found in the Recon refine3 mesh, and quantify
how much actual surface AREA (not just vertex count) is corrupted, at the
coarse (post R7-12 prune, pre-refine) stage vs after 3 refine rounds --
to test whether per-round subdivide+reproject+prune is amplifying a
fractal/jagged boundary rather than converging to a clean one.
"""
import pathlib

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

GT_PLY = pathlib.Path(
    r"C:\Users\hide2\IdeaBox\Mesh_Generater\output\A0072600002\bodies\body002_filled_midsurface.ply")
STAGE_FILES = {
    "coarse": pathlib.Path("body002_r711r712_stage_coarse_midsurface.ply"),
    "refine1": pathlib.Path("body002_r711r712_stage_refine1_midsurface.ply"),
    "refine2": pathlib.Path("body002_r711r712_stage_refine2_midsurface.ply"),
    "refine3": pathlib.Path("body002_r711r712_stage_refine3_midsurface.ply"),
}


def boundary_loops(mesh: trimesh.Trimesh):
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
    return [verts[labels == c] for c in range(n_comp)]


def main():
    gt_mesh = trimesh.load(str(GT_PLY), process=False)
    gt_loops = boundary_loops(gt_mesh)
    gt_loops_sorted = sorted(gt_loops, key=len, reverse=True)
    gt_outer = gt_loops_sorted[0]
    gt_outer_pts = gt_mesh.vertices[gt_outer]
    print(f"GT outer-perimeter loop: {len(gt_outer)} verts, "
          f"bbox={gt_outer_pts.min(0).round(1)} .. {gt_outer_pts.max(0).round(1)}")
    tree_gt = cKDTree(gt_mesh.vertices)
    tree_gt_outer = cKDTree(gt_outer_pts)

    print("\n" + "=" * 78)
    print("Per-stage: area-weighted corruption fraction + giant-loop identity")
    print("=" * 78)
    for label, f in STAGE_FILES.items():
        mesh = trimesh.load(str(f), process=False)
        loops = boundary_loops(mesh)
        loops_sorted = sorted(loops, key=len, reverse=True)
        sizes = [len(l) for l in loops_sorted]

        # vertex -> nearest GT vertex distance
        d_gt, _ = tree_gt.query(mesh.vertices, k=1)

        # per-face max-vertex distance -> face "bad" if any vertex far from GT
        face_d = d_gt[mesh.faces].max(axis=1)
        face_area = mesh.area_faces
        total_area = face_area.sum()
        for thr in [5.0, 10.0, 20.0]:
            bad_area = face_area[face_d > thr].sum()
            print(f"  [{label}] area with max-vertex-dist-to-GT > {thr}mm: "
                  f"{bad_area:.0f}/{total_area:.0f} mm^2 ({100*bad_area/total_area:.1f}%)")

        if sizes:
            biggest = loops_sorted[0]
            biggest_pts = mesh.vertices[biggest]
            # is the giant loop's bbox close to GT's outer-perimeter bbox?
            d_to_gt_outer, _ = tree_gt_outer.query(biggest_pts, k=1)
            print(f"  [{label}] n_loops={len(loops)} biggest_loop={len(biggest)} verts, "
                  f"bbox={biggest_pts.min(0).round(1)} .. {biggest_pts.max(0).round(1)}")
            print(f"  [{label}] biggest loop -> nearest GT-outer-perimeter-vertex dist (mm): "
                  f"mean={d_to_gt_outer.mean():.2f} median={np.median(d_to_gt_outer):.2f} "
                  f"p90={np.percentile(d_to_gt_outer,90):.2f} max={d_to_gt_outer.max():.2f}")
        print()


if __name__ == "__main__":
    main()

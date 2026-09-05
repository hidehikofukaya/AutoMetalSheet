"""
verify_topology_patch.py -- Stage 1 verification (ad-hoc, NOT part of the
production pipeline): regenerate body002's mid-surface ONLY (no dataset, no
retraining) with the new topology-patch fix in build_midsurface_mesh()
(midsurface_sampler.py), and compare boundary-loop topology before/after
against the original (unpatched) GT mesh already on disk.

Usage:
    python verify_topology_patch.py
"""
import pathlib
import sys
import time

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "preprocess"))

from annotator import step_to_pyvista
import midsurface_sampler as MS

STP_PATH = pathlib.Path(
    r"C:\Users\hide2\IdeaBox\Mesh_Generater\output\A0072600002\bodies\body002_filled.stp")
OLD_GT_PLY = pathlib.Path(
    r"C:\Users\hide2\IdeaBox\Mesh_Generater\output\A0072600002\bodies\body002_filled_midsurface.ply")
NEW_GT_PLY = pathlib.Path(__file__).resolve().parent / "body002_filled_midsurface_patched.ply"


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


def summarize(name, mesh):
    full_diag = float(np.linalg.norm(mesh.vertices.max(0) - mesh.vertices.min(0)))
    loops = boundary_loops(mesh)
    sizes = sorted((len(l) for l in loops), reverse=True)
    diags = []
    for l in loops:
        p = mesh.vertices[l]
        diags.append(float(np.linalg.norm(p.max(0) - p.min(0))))
    diags_sorted_idx = np.argsort(diags)[::-1]
    print(f"\n[{name}] {mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces, "
          f"area={mesh.area:.1f}mm^2, full-mesh bbox diag={full_diag:.1f}mm")
    print(f"  boundary loops: {len(loops)} total")
    if loops:
        n_tiny = sum(1 for s in sizes if s <= 6)
        print(f"    largest 5 loop sizes (n_verts): {sizes[:5]}")
        print(f"    loops with <=6 verts (likely spurious artifacts): "
              f"{n_tiny} ({100*n_tiny/len(loops):.1f}%)")
        top5_diag = [round(diags[i], 1) for i in diags_sorted_idx[:5]]
        print(f"    largest 5 loop bbox-diagonals (mm): {top5_diag}  "
              f"(part full-mesh diag={full_diag:.1f}mm)")
    return loops, sizes, diags, full_diag


def main():
    print(f"loading + tessellating {STP_PATH.name} ...")
    t0 = time.time()
    mesh = step_to_pyvista(STP_PATH, linear_deflection=0.15)
    print(f"  tessellated: {mesh.n_points} pts, {mesh.n_cells} cells ({time.time()-t0:.1f}s)")

    print("\nbuilding mid-surface WITHOUT topology patch (reproduces original GT)...")
    t0 = time.time()
    mesh_unpatched, thickness = MS.build_midsurface_mesh(mesh, topology_patch=False)
    print(f"  done in {time.time()-t0:.1f}s, thickness_hint={thickness:.3f}mm, "
          f"{mesh_unpatched.n_points} pts, {mesh_unpatched.n_cells} cells")

    print("\nbuilding mid-surface WITH topology patch (Stage 1 fix)...")
    t0 = time.time()
    mesh_patched, thickness2 = MS.build_midsurface_mesh(mesh, topology_patch=True)
    print(f"  done in {time.time()-t0:.1f}s, thickness_hint={thickness2:.3f}mm, "
          f"{mesh_patched.n_points} pts, {mesh_patched.n_cells} cells")

    mesh_patched.save(str(NEW_GT_PLY))
    print(f"\nsaved patched mid-surface to {NEW_GT_PLY}")

    tri_unpatched = trimesh.Trimesh(
        vertices=np.asarray(mesh_unpatched.points),
        faces=mesh_unpatched.faces.reshape(-1, 4)[:, 1:4], process=False)
    tri_patched = trimesh.Trimesh(
        vertices=np.asarray(mesh_patched.points),
        faces=mesh_patched.faces.reshape(-1, 4)[:, 1:4], process=False)
    tri_old_disk = trimesh.load(str(OLD_GT_PLY), process=False)

    print("\n" + "=" * 78)
    print("Boundary-loop topology comparison")
    print("=" * 78)
    summarize("old GT (on disk, unpatched, pre-existing)", tri_old_disk)
    summarize("re-run WITHOUT patch (sanity: should match old GT closely)", tri_unpatched)
    summarize("re-run WITH topology patch (Stage 1 fix)", tri_patched)


if __name__ == "__main__":
    main()

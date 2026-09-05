"""
dcudf_extract.py -- ported implementation of DCUDF (Hou et al., SIGGRAPH Asia
2023, "Robust Zero Level-Set Extraction from Unsigned Distance Fields Based
on Double Covering", https://arxiv.org/abs/2310.03431), adapted to this
project's VecSetAE model and Python 3.13 / open3d-free environment.

This replaces the Stage C R7-06..R7-15 rank-based-candidate + pyvista
reconstruct_surface() + ad-hoc gap/ghost gating pipeline (reconstruct.py)
with DCUDF's jointly-optimized double-cover-then-cut approach, adopted after
the R7-15 Laplacian-smoothing-then-reproject patch was found NOT to fix mesh
incoherence (the post-smooth gradient_project() re-snap is itself an
independent per-point Newton step that partially undoes the smoothing).

Porting notes vs. the original https://github.com/jjjkkyz/DCUDF :
  - `dcudf.mesh_extraction`'s `dcudf` class hardcoded `torch.device('cuda')`
    and called `.cuda()` throughout; here `device` is a constructor
    parameter threaded through instead, so this also runs on CPU.
  - `laplacian_calculation()` used `torch.sparse.FloatTensor(...)`, deprecated
    since torch 1.x; replaced with `torch.sparse_coo_tensor(...)` (same
    semantics, this project's torch is 2.6.0).
  - `detect_manifold()` in the original `mesh_cut.py` depended on
    `open3d.geometry.TriangleMesh.get_non_manifold_vertices()`. open3d has no
    Python 3.13 wheel (verified: `pip install open3d` fails in this conda
    env, "no matching distribution"). Reimplemented from scratch below as
    `detect_non_manifold_vertices()` using only trimesh/numpy: a vertex is
    non-manifold if its incident faces do not form a single connected fan
    (the standard "bowtie vertex" definition; see module docstring there).
  - Otherwise the algorithm (threshold-level marching cubes for the double
    cover -> joint distance+Laplacian-loss VectorAdam optimization -> face-
    angle-weighted min-cut split) is unchanged from the original.
"""
import math
import time

import maxflow
import numpy as np
import torch
import trimesh
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from skimage import measure


# ─────────────────────────────────────────────────────────────────────────────
# VectorAdam (DCUDF's custom optimizer: per-vertex 3-vector-aware Adam, so a
# vertex's xyz step direction follows its gradient direction rather than
# being distorted by per-coordinate second-moment normalization)
# ─────────────────────────────────────────────────────────────────────────────

class VectorAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, betas=(0.9, 0.999), eps=1e-8, axis=-1):
        defaults = dict(lr=lr, betas=betas, eps=eps, axis=axis)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            axis = group["axis"]
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["g1"] = torch.zeros_like(p.data)
                    state["g2"] = torch.zeros_like(p.data)

                g1 = state["g1"]
                g2 = state["g2"]
                state["step"] += 1
                grad = p.grad.data

                g1.mul_(b1).add_(grad, alpha=1 - b1)
                if axis is not None:
                    dim = grad.shape[axis]
                    grad_norm = torch.norm(grad, dim=axis).unsqueeze(axis).repeat_interleave(dim, dim=axis)
                    grad_sq = grad_norm * grad_norm
                    g2.mul_(b2).add_(grad_sq, alpha=1 - b2)
                else:
                    g2.mul_(b2).add_(grad.square(), alpha=1 - b2)

                m1 = g1 / (1 - (b1 ** state["step"]))
                m2 = g2 / (1 - (b2 ** state["step"]))
                gr = m1 / (eps + m2.sqrt())
                p.data.sub_(gr, alpha=lr)


# ─────────────────────────────────────────────────────────────────────────────
# Double-cover extraction (threshold-level marching cubes) + Laplacian terms
# ─────────────────────────────────────────────────────────────────────────────

def threshold_MC(ndf, threshold, resolution, bound_min=None, bound_max=None):
    """Marching cubes at a small positive threshold (not zero -- UDF never
    crosses zero) extracts a closed double-cover shell wrapping both sides
    of the true (possibly open) surface."""
    try:
        vertices, triangles, _, _ = measure.marching_cubes(
            ndf, threshold, spacing=(2 / resolution, 2 / resolution, 2 / resolution))
        vertices -= 1
        mesh = trimesh.Trimesh(vertices, triangles, process=False)
    except ValueError:
        print("threshold too high")
        mesh = None

    if bound_min is not None:
        bound_min = bound_min.cpu().numpy()
        bound_max = bound_max.cpu().numpy()
        mesh.apply_scale((bound_max - bound_min) / 2)
        mesh.apply_translation((bound_min + bound_max) / 2)
    mesh.apply_translation([1 / resolution, 1 / resolution, 1 / resolution])
    mesh.apply_scale(resolution / (resolution - 1))
    return mesh


def laplacian_calculation(mesh, equal_weight=True):
    """trimesh vertex-neighbor uniform/umbrella Laplacian, as a torch sparse
    matrix (ported from trimesh.smoothing's internal helper)."""
    neighbors = mesh.vertex_neighbors
    vertices = mesh.vertices.view(np.ndarray)

    col = np.concatenate(neighbors)
    row = np.concatenate([[i] * len(n) for i, n in enumerate(neighbors)])

    if equal_weight:
        data = np.concatenate([[1.0 / len(n)] * len(n) for n in neighbors])
    else:
        ones = np.ones(3)
        norms = [1.0 / np.sqrt(np.dot((vertices[i] - vertices[n]) ** 2, ones))
                 for i, n in enumerate(neighbors)]
        data = np.concatenate([i / i.sum() for i in norms])

    matrix = coo_matrix((data, (row, col)), shape=[len(vertices)] * 2)
    values = matrix.data
    indices = np.vstack((matrix.row, matrix.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse_coo_tensor(i, v, torch.Size(matrix.shape))


def laplacian_step(laplacian_op, samples):
    return torch.sparse.mm(laplacian_op, samples[:, 0:3]) - samples[:, 0:3]


def get_abc(vertices, faces):
    fvs = vertices[faces]
    sub_a = torch.linalg.norm(fvs[:, 0, :] - fvs[:, 1, :], dim=1)
    sub_b = torch.linalg.norm(fvs[:, 1, :] - fvs[:, 2, :], dim=1)
    sub_c = torch.linalg.norm(fvs[:, 0, :] - fvs[:, 2, :], dim=1)
    return sub_a, sub_b, sub_c


def calculate_s(vertices, faces):
    """Heron's-formula face area (sqrt(p(p-a)(p-b)(p-c))); used to inverse-
    area-weight the per-vertex Laplacian loss (small/sliver triangles get
    relatively more smoothing pressure)."""
    sub_a, sub_b, sub_c = get_abc(vertices, faces)
    p = (sub_a + sub_b + sub_c) / 2
    s = p * (p - sub_a) * (p - sub_b) * (p - sub_c)
    s[s < 1e-30] = 1e-30
    return torch.sqrt(s)


def get_mid(vertices, faces):
    return torch.mean(vertices[faces], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Main DCUDF extractor
# ─────────────────────────────────────────────────────────────────────────────

class DCUDFExtractor:
    """Ported from dcudf.mesh_extraction.dcudf (see module docstring for
    porting deltas: device is parameterized, not hardcoded to 'cuda').

    Args:
        query_func:  differentiable fn, pts[N,3] (torch, on `device`) -> udf[N]
        resolution:  marching-cubes grid resolution
        threshold:   MC iso-value (small positive; NOT zero, see threshold_MC)
        max_iter:    total optimization iterations
        normal_step: iterations using the Laplacian-smoothness loss; the
                     remainder switch to a normal-direction-preserving loss
        laplacian_weight: weight of the Laplacian term vs. the distance term
        bound_min/bound_max: [3] grid bounds (torch tensor / list / ndarray)
        is_cut:      run mesh_cut() to split the double cover into 2 single
                     sheets, keeping the larger (set False for closed/solid
                     models where no cut is needed)
        region_rate: mesh_cut() seed/sink region size denominator (smaller
                     denominator -> larger region)
        max_batch:   query_func batch chunk size
        device:      torch device string/object (NOT hardcoded, see above)
    """

    def __init__(self, query_func, resolution, threshold,
                 max_iter=400, normal_step=300, laplacian_weight=2000.0,
                 bound_min=None, bound_max=None,
                 is_cut=True, region_rate=20,
                 max_batch=100_000, learning_rate=0.0005, warm_up_end=25,
                 report_freq=1, device=None):
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.max_iter = max_iter
        self.max_batch = max_batch
        self.report_freq = report_freq
        self.normal_step = normal_step
        self.laplacian_weight = laplacian_weight
        self.warm_up_end = warm_up_end
        self.learning_rate = learning_rate
        self.resolution = resolution
        self.threshold = threshold

        if bound_min is None:
            bound_min = torch.tensor([-1 + threshold] * 3, dtype=torch.float32)
        if bound_max is None:
            bound_max = torch.tensor([1 - threshold] * 3, dtype=torch.float32)
        bound_min = torch.as_tensor(bound_min, dtype=torch.float32)
        bound_max = torch.as_tensor(bound_max, dtype=torch.float32)
        self.bound_min = bound_min - threshold
        self.bound_max = bound_max + threshold

        self.is_cut = is_cut
        self.region_rate = region_rate

        self.u = None
        self.mesh = None
        self.optimizer = None
        self.cut_time = None
        self.extract_time = None
        self.query_func = query_func

    def extract_fields(self):
        N = 32
        X = torch.linspace(self.bound_min[0], self.bound_max[0], self.resolution).split(N)
        Y = torch.linspace(self.bound_min[1], self.bound_max[1], self.resolution).split(N)
        Z = torch.linspace(self.bound_min[2], self.bound_max[2], self.resolution).split(N)

        u = np.zeros([self.resolution, self.resolution, self.resolution], dtype=np.float32)
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                                     dim=-1).to(self.device)
                    val = self.query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
        self.u = u
        return u

    def update_learning_rate(self, iter_step):
        warm_up = self.warm_up_end
        max_iter = self.max_iter
        init_lr = self.learning_rate
        if iter_step < warm_up:
            lr = iter_step / warm_up
        else:
            lr = 0.5 * (math.cos((iter_step - warm_up) / (max_iter - warm_up) * math.pi) + 1)
        lr = lr * init_lr
        if iter_step >= 200:
            lr *= 0.1
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def optimize(self):
        t_extract0 = time.time()
        query_func = self.query_func
        u = self.extract_fields()
        self.extract_time = time.time() - t_extract0

        self.mesh = threshold_MC(u, self.threshold, self.resolution,
                                  bound_min=self.bound_min, bound_max=self.bound_max)
        if self.mesh is None:
            raise RuntimeError("threshold_MC produced no surface at this threshold/resolution")

        xyz = torch.from_numpy(self.mesh.vertices.astype(np.float32)).to(self.device)
        xyz.requires_grad_(True)
        self.optimizer = VectorAdam([xyz])
        laplacian_op = laplacian_calculation(self.mesh).to(self.device)

        vertex_faces = np.asarray(self.mesh.vertex_faces)
        face_mask = np.ones_like(vertex_faces).astype(bool)
        face_mask[vertex_faces == -1] = False

        normals = None
        origin_points = None

        for it in range(self.max_iter):
            if it == self.normal_step:
                points = xyz.detach().cpu().numpy()
                normal_mesh = trimesh.Trimesh(vertices=points, faces=self.mesh.faces, process=False)
                normals = torch.tensor(normal_mesh.face_normals, dtype=torch.float32, device=self.device)
                origin_points = get_mid(xyz, self.mesh.faces).detach().clone()

            self.update_learning_rate(it)

            epoch_loss = 0.0
            self.optimizer.zero_grad()
            num_samples = xyz.shape[0]
            head = 0
            while head < num_samples:
                sample_subset = xyz[head: min(head + self.max_batch, num_samples)]
                df = query_func(sample_subset)
                loss = df.mean()

                if it <= self.normal_step:
                    s_value = calculate_s(xyz, self.mesh.faces)
                    face_weight = s_value[vertex_faces[head: min(head + self.max_batch, num_samples)]]
                    face_weight[~face_mask[head: min(head + self.max_batch, num_samples)]] = 0
                    face_weight = torch.sum(face_weight, dim=1)
                    face_weight = torch.sqrt(face_weight.detach())
                    face_weight = face_weight.max() / face_weight

                    lap_v = laplacian_step(laplacian_op, xyz)
                    lap_v = torch.mul(lap_v, lap_v)
                    lap_v = lap_v[head: min(head + self.max_batch, num_samples)]
                    laplacian_loss = face_weight * torch.sum(lap_v, dim=1)
                    laplacian_loss = self.laplacian_weight * laplacian_loss.mean()
                    loss = loss + laplacian_loss

                epoch_loss += float(loss.data)
                loss.backward()
                head += self.max_batch

            mid_num_samples = len(self.mesh.faces)
            mid_head = 0
            while mid_head < mid_num_samples:
                mid_points = get_mid(xyz, self.mesh.faces)
                sub_mid_points = mid_points[mid_head: min(mid_head + self.max_batch, mid_points.shape[0])]
                loss = query_func(sub_mid_points).mean()
                if it > self.normal_step:
                    sl = slice(mid_head, min(mid_head + self.max_batch, mid_points.shape[0]))
                    offset = mid_points[sl] - origin_points[sl]
                    normal_loss = torch.norm(torch.cross(offset, normals[sl], dim=-1), dim=-1)
                    loss = loss + 0.5 * normal_loss.mean()
                epoch_loss += float(loss.data)
                loss.backward()
                mid_head += self.max_batch

            self.optimizer.step()
            if (it + 1) % self.report_freq == 0:
                print(f"  [DCUDF] iter {it} loss={epoch_loss:.6g}")

        final_mesh = trimesh.Trimesh(vertices=xyz.detach().cpu().numpy(), faces=self.mesh.faces, process=False)

        if self.is_cut:
            from dcudf_extract import mesh_cut
            t0 = time.time()
            cut = mesh_cut(final_mesh, region_rate=self.region_rate)
            self.cut_time = time.time() - t0
            if cut is not None:
                m1, m2 = cut
                final_mesh = m1 if len(m1.vertices) > len(m2.vertices) else m2
            else:
                print("  [DCUDF] WARNING: mesh_cut failed (model may be too complex / non-manifold); "
                      "returning UNCUT double-cover mesh")

        return final_mesh


# ─────────────────────────────────────────────────────────────────────────────
# Double-cover -> single-sheet split: region-growing seed/sink + global
# min-cut (PyMaxflow), weighted by face dihedral angle.
# Ported from dcudf.mesh_cut; the only change is detect_manifold() (which
# required open3d, unavailable on this project's Python 3.13 environment) ->
# detect_non_manifold_vertices() below, a from-scratch trimesh/numpy-only
# equivalent.
# ─────────────────────────────────────────────────────────────────────────────

def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            return None
        return trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces, process=False)
                  for g in scene_or_mesh.geometry.values()))
    assert isinstance(scene_or_mesh, trimesh.Trimesh)
    return scene_or_mesh


def compute_neighbor(face_neighbor, idx, point_num, rate):
    """Breadth-first region growth from face `idx` until it covers
    `point_num/rate` faces (or stalls for 20 consecutive non-growing steps,
    in which case None is returned and the caller retries with a new seed).
    """
    size = int(point_num / rate)
    mask_wait = np.zeros(point_num, dtype=bool)
    mask_wait[idx] = True
    mask_wait[face_neighbor[idx]] = True
    wait_list = list(face_neighbor[idx])
    neighbor_list = list(wait_list)
    flag = True
    count = 0
    while len(neighbor_list) < size:
        if len(wait_list) == 0:
            return None
        k_point = wait_list.pop()
        for k in face_neighbor[k_point]:
            if not mask_wait[k]:
                wait_list.append(k)
                mask_wait[k] = True
                flag = False
                neighbor_list.append(k)
        if flag:
            count += 1
        else:
            count = 0
        if count == 20:
            return None
    return set(neighbor_list)


def detect_non_manifold_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    """trimesh/numpy-only replacement for
    open3d.geometry.TriangleMesh.get_non_manifold_vertices() (DCUDF's
    original mesh_cut.detect_manifold(), which required open3d -- no Python
    3.13 wheel exists for it, confirmed via `pip install open3d` failure in
    this project's conda env).

    Definition (matches open3d's): a vertex is a "non-manifold" / "bowtie"
    vertex if the faces incident to it do NOT form a single connected fan,
    i.e. if you walk the faces touching the vertex via shared edges (edges
    that also touch the vertex), they split into more than one connected
    component. This happens e.g. where two locally-disjoint surface patches
    are joined only at a single point.

    Implementation: for vertex v with incident faces F(v), build a graph on
    F(v) where two faces are adjacent iff they share an edge incident to v
    (i.e. they share v and exactly one other vertex). A vertex is
    non-manifold iff this graph has more than one connected component
    (trivial case: 0 or 1 incident face is always manifold).
    """
    vertex_faces = mesh.vertex_faces  # [V, max_deg], -1 padded
    faces = mesh.faces
    non_manifold = []

    for v in range(len(mesh.vertices)):
        finc = vertex_faces[v]
        finc = finc[finc >= 0]
        n = len(finc)
        if n <= 1:
            continue

        # group incident faces by the "other" vertex they share an edge
        # with v through -- two faces sharing such an edge are fan-adjacent.
        other_to_faces = {}
        for local_i, f in enumerate(finc):
            tri = faces[f]
            for o in tri:
                if o == v:
                    continue
                other_to_faces.setdefault(o, []).append(local_i)

        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for flist in other_to_faces.values():
            for a, b in zip(flist[:-1], flist[1:]):
                union(a, b)

        if len({find(i) for i in range(n)}) > 1:
            non_manifold.append(v)

    return np.array(non_manifold, dtype=np.int64)


def mesh_cut(mesh, region_rate=20, exp_weight=200):
    """Split a closed double-cover mesh into its two single-sheet halves via
    randomized seed/sink region growing + a global min-cut (graph weighted
    by 1/face-dihedral-angle, so the cut prefers to run along sharp/flat
    creases -- the double cover's "rim" where the two sheets meet). Returns
    (mesh_a, mesh_b) or None if no valid cut was found after retrying.
    """
    cc = trimesh.graph.connected_components(mesh.face_adjacency, min_len=len(mesh.vertices) // 5)
    mask = np.zeros(len(mesh.faces), dtype=bool)
    mask[np.concatenate(cc)] = True
    mesh.update_faces(mask)
    mesh.remove_unreferenced_vertices()

    points = np.sum(np.array(mesh.triangles), axis=1)
    ptree = cKDTree(points)
    face_neighbor = [[] for _ in range(len(points))]
    for d in mesh.face_adjacency:
        face_neighbor[d[0]].append(d[1])
        face_neighbor[d[1]].append(d[0])

    try_time = 0
    ite_num = 0
    while True:
        if ite_num > 5:
            if try_time > 20:
                print("  [DCUDF mesh_cut] cut failed")
                return None
            ite_num = 0
            region_rate *= 1.5
            print(f"  [DCUDF mesh_cut] decrease region {region_rate}")
            try_time += 1

        seed = np.random.choice(points.shape[0], 1, replace=False)[0]
        _, near_idx = ptree.query(points[seed], 50)
        near_idx = near_idx - 1

        seed_neighbor = compute_neighbor(face_neighbor, seed, len(points), region_rate)
        if seed_neighbor is None:
            continue
        seed_neighbor.add(seed)

        idx_flag = False
        for idx in near_idx:
            if idx in seed_neighbor:
                continue
            idx_flag = True
            sink_neighbor = compute_neighbor(face_neighbor, idx, len(points), region_rate)
            if sink_neighbor is None:
                continue
            sink_neighbor.add(idx)
            if len(sink_neighbor.intersection(seed_neighbor)) > 0:
                break

            weight = mesh.face_adjacency_angles[:, np.newaxis].copy()
            weight = weight.max() - weight
            weight = np.exp(exp_weight * weight)

            edges = np.concatenate((mesh.face_adjacency, weight), axis=1)
            new_idx = []
            for i in range(edges.shape[0]):
                if edges[i][0] in seed_neighbor and edges[i][1] in seed_neighbor:
                    continue
                elif edges[i][0] in sink_neighbor and edges[i][1] in sink_neighbor:
                    continue
                elif edges[i][0] in seed_neighbor:
                    edges[i][0] = seed
                elif edges[i][0] in sink_neighbor:
                    edges[i][0] = idx
                elif edges[i][1] in seed_neighbor:
                    edges[i][1] = seed
                elif edges[i][1] in sink_neighbor:
                    edges[i][1] = idx
                new_idx.append(i)
            edges = edges[new_idx]
            edges[:, 2] = edges[:, 2] + 1

            count = 0
            ori_g = [-1] * len(points)
            g_ori = [-1] * len(points)
            for e in edges:
                if e[0] == seed or e[0] == idx:
                    if ori_g[int(e[1])] == -1:
                        ori_g[int(e[1])] = count
                        g_ori[count] = e[1]
                        count += 1
                elif e[1] == seed or e[1] == idx:
                    if ori_g[int(e[0])] == -1:
                        ori_g[int(e[0])] = count
                        g_ori[count] = e[0]
                        count += 1
                else:
                    if ori_g[int(e[0])] == -1:
                        ori_g[int(e[0])] = count
                        g_ori[count] = e[0]
                        count += 1
                    if ori_g[int(e[1])] == -1:
                        ori_g[int(e[1])] = count
                        g_ori[count] = e[1]
                        count += 1
            g_ori.remove(-1.0)

            g = maxflow.Graph[float]()
            nodes = g.add_nodes(len(g_ori))
            for e in edges:
                if e[0] == seed:
                    g.add_tedge(nodes[ori_g[int(e[1])]], e[2], 0)
                elif e[0] == idx:
                    g.add_tedge(nodes[ori_g[int(e[1])]], 0, e[2])
                elif e[1] == seed:
                    g.add_tedge(nodes[ori_g[int(e[0])]], e[2], 0)
                elif e[1] == idx:
                    g.add_tedge(nodes[ori_g[int(e[0])]], 0, e[2])
                else:
                    g.add_edge(nodes[ori_g[int(e[0])]], nodes[ori_g[int(e[1])]], e[2], e[2])

            g.maxflow()
            seg = g.get_grid_segments(nodes)
            partition = (set(), set())
            for gid in range(seg.shape[0]):
                if g_ori[gid] < 0:
                    continue
                partition[1 if seg[gid] else 0].add(g_ori[gid])

            rate = abs(len(partition[0]) - len(partition[1])) / (len(partition[0]) + len(partition[1]))
            if rate > 0.15:
                ite_num += 1
                print(f"  [DCUDF mesh_cut] not an OK cut, rate is {rate}")
                break

            seed_list = set(np.array(list(partition[0].union(seed_neighbor)), dtype=int).tolist())
            sink_list = set(np.array(list(partition[1].union(sink_neighbor)), dtype=int).tolist())

            out_mesh_1 = trimesh.base.Trimesh(vertices=mesh.vertices, faces=mesh.faces[list(seed_list)], process=False)
            out_mesh_2 = trimesh.base.Trimesh(vertices=mesh.vertices, faces=mesh.faces[list(sink_list)], process=False)
            out_mesh = out_mesh_1 if len(seed_list) > len(sink_list) else out_mesh_2

            while True:
                # a cut can leave non-manifold (bowtie) vertices where the
                # split runs through a noisy/thin region; iteratively fold
                # any such face back into the larger side until clean.
                re = detect_non_manifold_vertices(out_mesh)
                if re.shape[0] == 0:
                    break
                for vidx in re:
                    add_face = mesh.vertex_faces[vidx].tolist()
                    if len(seed_list) > len(sink_list):
                        seed_list = seed_list.union(set(add_face))
                        sink_list = sink_list.difference(set(add_face))
                        seed_list.discard(-1)
                    else:
                        sink_list = sink_list.union(set(add_face))
                        seed_list = seed_list.difference(set(add_face))
                        sink_list.discard(-1)

                out_mesh_1 = trimesh.base.Trimesh(vertices=mesh.vertices, faces=mesh.faces[list(seed_list)], process=False)
                out_mesh_2 = trimesh.base.Trimesh(vertices=mesh.vertices, faces=mesh.faces[list(sink_list)], process=False)
                out_mesh = out_mesh_1 if len(seed_list) > len(sink_list) else out_mesh_2

            out_mesh_1.remove_unreferenced_vertices()
            out_mesh_2.remove_unreferenced_vertices()
            return out_mesh_1, out_mesh_2

        if not idx_flag:
            ite_num += 1
            print("  [DCUDF mesh_cut] near region may be too small")
    return None

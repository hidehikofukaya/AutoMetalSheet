"""Oriented point cloud -> midsurface mesh -> feature curves.

The generator emits points; a wireframe needs curves. Going straight from points
to curves failed every way it was tried, because the curves are measure-zero in
the surface: a uniform surface sample lands on them about once per curve, which
is not enough to trace. Rebuilding the surface first fixes that -- a continuous
sheet has ridges and a rim whether or not a sample happened to land on them.

Measured ceilings on perfect clouds:
    points -> surface (here)        2.2mm
    surface -> wireframe            1.26mm
against a generator that is at ~10mm, so the post-process is not the bottleneck.

The implicit is MLS -- the signed distance to a locally fitted plane -- not an
unsigned distance field. A midsurface is an open sheet of zero thickness, and
the level set of an unsigned field wraps both of its faces: that produced a
closed "pillow" with 14-22x the triangles of the real sheet and ~13mm of error
no matter how many points were thrown at it.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

VOX = 2.0          # marching-cubes voxel (mm)
SUPPORT = 6.0      # MLS support radius (mm); 6 beat 10 and 14
K_FIT = 12

# The two feature classes want opposite settings, so extract_features runs the
# reconstruction twice. Measured on perfect clouds (256 points, 8 parts):
#   support 6mm / 25 deg : outline 2.13mm  bend 5.54mm   <- the bend is rounded off
#   support 6mm / 12 deg : outline 2.10mm  bend 3.11mm
#   support 4mm / 12 deg : outline 2.93mm  bend 2.20mm   <- sharper, noisier sheet
# A wide support smooths the sheet (good for the rim) but rounds the bend until
# the dihedral test stops seeing it; a narrow one keeps the bend and roughens
# the rest.
OUTLINE_SUPPORT, OUTLINE_DEG = 6.0, 12.0
BEND_SUPPORT, BEND_DEG = 4.0, 12.0


def mls_surface(xyz: np.ndarray, normal: np.ndarray, vox: float = VOX,
                support: float = SUPPORT, k: int = K_FIT):
    """Returns (vertices, triangles) of the sheet the oriented points sample."""
    from skimage import measure

    xyz = np.asarray(xyz, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    k = min(k, len(xyz))

    lo, hi = xyz.min(0) - support, xyz.max(0) + support
    axes = [np.arange(a, b, vox) for a, b in zip(lo, hi)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    flat = grid.reshape(-1, 3)

    tree = cKDTree(xyz)
    d, idx = tree.query(flat, k=k)
    if k == 1:
        d, idx = d[:, None], idx[:, None]
    w = np.exp(-(d / support) ** 2)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-30)
    p_bar = (xyz[idx] * w[..., None]).sum(1)
    n_bar = (n[idx] * w[..., None]).sum(1)
    n_bar /= np.maximum(np.linalg.norm(n_bar, axis=1, keepdims=True), 1e-12)

    field = np.einsum("ij,ij->i", flat - p_bar, n_bar).reshape(grid.shape[:3])
    # away from any point the plane fit means nothing; push the field out so the
    # level set closes there instead of wandering
    far = (d[:, 0] > support).reshape(grid.shape[:3])
    field[far] = np.sign(field[far]) * support * 3.0

    V, T, _, _ = measure.marching_cubes(field, level=0.0, spacing=(vox,) * 3)
    V = V + lo
    keep = tree.query(V)[0] < support * 0.9          # trim the padding
    remap = -np.ones(len(V), dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    good = keep[T].all(1)
    return V[keep], remap[T[good]]


def mesh_features(V: np.ndarray, T: np.ndarray, dihedral_deg: float = 25.0):
    """(boundary edges, crease edges) as index pairs -- the outline and the
    bend lines of a sheet, read off the surface rather than off the samples."""
    if len(T) == 0:
        return [], []
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    owners: dict = {}
    for ti, tri in enumerate(T):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            owners.setdefault((min(a, b), max(a, b)), []).append(ti)
    thresh = np.cos(np.deg2rad(dihedral_deg))
    boundary, crease = [], []
    for (a, b), ts in owners.items():
        if len(ts) == 1:
            boundary.append((a, b))
        elif len(ts) == 2 and float(fn[ts[0]] @ fn[ts[1]]) < thresh:
            crease.append((a, b))
    return boundary, crease


def edge_points(V: np.ndarray, edges, step: float = 2.0) -> np.ndarray:
    out = []
    for a, b in edges:
        p, q = V[a], V[b]
        dist = float(np.linalg.norm(q - p))
        n = max(2, int(dist / step) + 1)
        out.append(p + (q - p) * np.linspace(0.0, 1.0, n)[:, None])
    return np.concatenate(out) if out else np.zeros((0, 3))


def sample_mesh(V: np.ndarray, T: np.ndarray, n: int, rng) -> np.ndarray:
    """Area-weighted points on a mesh. Comparing a mesh's *vertices* to an area
    sample reads the sampling difference as error -- 11-20mm on surfaces that
    are identical -- so every comparison has to go through this."""
    if len(T) == 0:
        return np.zeros((0, 3))
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    keep = area > 1e-12
    if not keep.any():
        return np.zeros((0, 3))
    a, b, c, area = a[keep], b[keep], c[keep], area[keep]
    i = rng.choice(len(area), size=n, p=area / area.sum())
    u, v = rng.random((n, 1)), rng.random((n, 1))
    over = (u + v) > 1.0
    u[over], v[over] = 1.0 - u[over], 1.0 - v[over]
    return a[i] + (b[i] - a[i]) * u + (c[i] - a[i]) * v


def relax_uniform(xyz: np.ndarray, V: np.ndarray, T: np.ndarray, rng,
                  iters: int = 6) -> np.ndarray:
    """Even out the point spacing by pushing points apart along the surface and
    snapping them back to it. Measured worth ~7%: going from a random sample
    (spacing CV 0.54) to a perfectly even one (0.11) moved reconstruction from
    2.46mm to 2.28mm, so this is a polish step, not a fix."""
    if len(T) == 0 or len(xyz) < 4:
        return xyz
    pts = xyz.copy()
    dense = sample_mesh(V, T, max(4096, 8 * len(xyz)), rng)
    tree = cKDTree(dense)
    for _ in range(iters):
        d, nb = cKDTree(pts).query(pts, k=min(7, len(pts)))
        step = np.zeros_like(pts)
        target = np.median(d[:, 1])
        for j in range(1, nb.shape[1]):
            delta = pts - pts[nb[:, j]]
            dist = np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-9)
            push = np.clip((target - dist) / target, 0.0, 1.0)
            step += delta / dist * push * target * 0.25
        pts = pts + step / max(nb.shape[1] - 1, 1)
        pts = dense[tree.query(pts)[1]]              # back onto the surface
    return pts


def interpolate(xyz: np.ndarray, values: np.ndarray, at: np.ndarray,
                k: int = 8, support: float = SUPPORT) -> np.ndarray:
    """Carry a per-point field onto arbitrary positions (the mesh vertices)."""
    k = min(k, len(xyz))
    d, idx = cKDTree(xyz).query(at, k=k)
    if k == 1:
        d, idx = d[:, None], idx[:, None]
    w = np.exp(-(d / support) ** 2)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-30)
    return (values[idx] * w[..., None]).sum(1)


def level_set(V: np.ndarray, T: np.ndarray, field: np.ndarray,
              level: float = 0.0) -> np.ndarray:
    """Points where a scalar field crosses `level` on a triangle mesh.

    This is what makes a generated field usable: the curve is the zero set of a
    field defined everywhere, so no generated point has to land on it. Labelling
    points cannot do this -- a curve is measure-zero in a surface.
    """
    if len(T) == 0:
        return np.zeros((0, 3))
    f = field - level
    out = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        ia, ib = T[:, a], T[:, b]
        fa, fb = f[ia], f[ib]
        cross = (fa > 0) != (fb > 0)
        if not cross.any():
            continue
        t = fa[cross] / (fa[cross] - fb[cross] + 1e-12)
        out.append(V[ia[cross]] + (V[ib[cross]] - V[ia[cross]]) * t[:, None])
    return np.concatenate(out) if out else np.zeros((0, 3))


def extract_features(xyz: np.ndarray, normal: np.ndarray,
                     fields: np.ndarray | None = None, level: float = 1.5):
    """Outline and bend curves.

    With `fields` (distance to outline, distance to bend) the curves are the
    level sets of those fields on the reconstructed sheet -- the model says
    where they are. Without them it falls back to geometry: boundary edges and
    dihedral creases, which cannot find a trim at all, since a cut leaves no
    curvature behind.
    """
    V_o, T_o = mls_surface(xyz, normal, VOX, OUTLINE_SUPPORT)
    if fields is not None:
        f = interpolate(xyz, np.asarray(fields, dtype=np.float64), V_o)
        return (level_set(V_o, T_o, f[:, 0], level),
                level_set(V_o, T_o, f[:, 1], level), (V_o, T_o))
    boundary, _ = mesh_features(V_o, T_o, OUTLINE_DEG)
    V_b, T_b = mls_surface(xyz, normal, VOX, BEND_SUPPORT)
    _, crease = mesh_features(V_b, T_b, BEND_DEG)
    return edge_points(V_o, boundary), edge_points(V_b, crease), (V_o, T_o)


def demo() -> None:
    """Self-check on a plate with a right-angle flange -- what sheet metal
    actually looks like. A shallow bend is a harder case than reality, and the
    first version of this test used one, which made it fail for the wrong
    reason."""
    rng = np.random.default_rng(0)
    n_pts = 6000
    u = rng.random(n_pts) * 100.0          # 0..50 flat, 50..100 folded up
    v = rng.random(n_pts) * 60.0
    flat = u <= 50.0
    pts = np.where(flat[:, None],
                   np.stack([u, v, np.zeros_like(u)], 1),
                   np.stack([np.full_like(u, 50.0), v, u - 50.0], 1))
    nrm = np.where(flat[:, None],
                   np.tile([0.0, 0.0, 1.0], (n_pts, 1)),
                   np.tile([1.0, 0.0, 0.0], (n_pts, 1)))
    take = rng.choice(n_pts, 600, replace=False)

    V, T = mls_surface(pts[take], nrm[take])
    assert len(T) > 100, f"reconstruction collapsed ({len(T)} triangles)"
    rec = sample_mesh(V, T, 4000, rng)
    err = float(np.median(cKDTree(pts).query(rec)[0]))
    assert err < 4.0, f"reconstruction is {err:.2f}mm off the sheet"

    # a GENERATED field must localise the rim where geometry cannot: build the
    # true distance-to-rim field and check the level set lands on the edge
    rim = np.minimum.reduce([pts[take][:, 0], 100.0 - pts[take][:, 0],
                             pts[take][:, 1], 60.0 - pts[take][:, 1]])
    fld = np.stack([np.clip(rim, 0, 30.0), np.full(len(take), 30.0)], 1)
    from_field, _, _ = extract_features(pts[take], nrm[take], fld, level=1.5)
    assert len(from_field) > 0, "the field level set produced nothing"
    off = float(np.median(np.minimum.reduce([
        np.abs(from_field[:, 0]), np.abs(from_field[:, 0] - 100.0),
        np.abs(from_field[:, 1]), np.abs(from_field[:, 1] - 60.0)])))
    assert off < 6.0, f"field level set sits {off:.1f}mm from the true rim"
    print(f"field level set lands {off:.1f}mm from the rim "
          f"({len(from_field)} points)")

    outline, bend, _ = extract_features(pts[take], nrm[take])
    assert len(outline) > 0, "no boundary found on an open sheet"
    assert len(bend) > 0, "the bend was smoothed away entirely"
    # the fold runs along x=50, z=0; MLS rounds it, so allow the support radius
    along = float(np.median(np.hypot(bend[:, 0] - 50.0, bend[:, 2])))
    assert along < 15.0, f"crease sits {along:.1f}mm from the fold"
    print(f"surface demo ok: {len(T)} triangles, {err:.2f}mm to the sheet, "
          f"crease {along:.1f}mm from the fold, "
          f"{len(outline)} outline / {len(bend)} bend points")


if __name__ == "__main__":
    demo()

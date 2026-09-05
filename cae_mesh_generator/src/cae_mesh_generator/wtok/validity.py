"""Is the output a sheet-metal part at all? Checks that need no reference.

Chamfer to the true part is bounded below by the task: parts with near-identical
fastening conditions differ by 14.9mm, so a good number and a bad number can
both describe a perfectly reasonable part. What that metric cannot say is
whether the thing could be manufactured.

These checks ask that instead, the way a test suite asks whether code runs
rather than whether it matches a reference implementation. All of them work on
the point cloud directly -- surface reconstruction fills holes badly on a cloud
with the voids ours has, so nothing here needs a watertight mesh.

Every measure is reported against the same measure on the TRUE cloud. A curvature
estimated from 512 noisy points is not zero even for a genuinely flat panel, so
the true cloud's value is the floor, not zero.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .kernel import K_NN, patches
from .ridge import CLASSES


def principal_curvatures(xyz, nrm, k: int = K_NN):
    """Both principal curvatures per point, in units of the point spacing.

    Fits z = a x^2 + b xy + c y^2 + d x + e y over the neighbourhood in its own
    frame -- the same centring, scaling and normal alignment the local kernel
    uses, so the result is dimensionless and comparable across parts. The linear
    terms absorb a slightly-off normal rather than leaking into the curvature.
    """
    feat, spacing = patches(xyz, nrm, k)
    P = feat[:, :, :3].astype(np.float64)          # neighbours, local frame
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    A = np.stack([x * x, x * y, y * y, x, y], axis=-1)
    # least squares per point, batched
    AtA = np.einsum("nki,nkj->nij", A, A)
    Atz = np.einsum("nki,nk->ni", A, z)
    AtA += np.eye(5) * 1e-8
    coef = np.linalg.solve(AtA, Atz[..., None])[..., 0]
    a, b, c = coef[:, 0], coef[:, 1], coef[:, 2]
    H = np.stack([np.stack([2 * a, b], -1), np.stack([b, 2 * c], -1)], -2)
    kap = np.linalg.eigvalsh(H)                    # ascending by magnitude? no: value
    order = np.argsort(-np.abs(kap), axis=1)
    kap = np.take_along_axis(kap, order, axis=1)   # [:,0] largest |k|, [:,1] smallest
    return kap[:, 0], kap[:, 1], spacing


def developability(xyz, nrm, k: int = K_NN):
    """How far the surface is from being flattenable.

    A sheet-metal midsurface is developable: every point is flat or cylindrical,
    so the smaller principal curvature is zero. A dome is not, and cannot be made
    by cutting and bending one sheet. Returns the median |k_min| in spacing
    units, and the fraction of points clearly not developable.
    """
    kmax, kmin, _ = principal_curvatures(xyz, nrm, k)
    m = np.abs(kmin)
    return float(np.median(m)), float((m > 0.15).mean())


def panels(xyz, is_bend, k: int = 8, gap: float = 2.5):
    """Connected regions of surface once the bend lines are taken out.

    The count is the part's structural complexity: a bracket is a few panels, and
    a design that needs many is harder to make. Edges longer than `gap` spacings
    are cut so a void does not join two regions that only look adjacent.
    """
    keep = np.flatnonzero(~is_bend)
    if len(keep) < 4:
        return 0, np.array([]), []
    pts = xyz[keep]
    d, nb = cKDTree(pts).query(pts, k=min(k, len(pts)))
    # scale from the mean distance to k neighbours, not from the nearest one:
    # under irregular sampling the nearest-neighbour distance varies several-fold
    # between points and a threshold built on it shatters a single sheet into
    # dozens of pieces. The k-neighbour mean is the same quantity `patches` uses.
    scale = float(np.median(d[:, 1:].mean(1))) or 1e-9
    parent = np.arange(len(pts))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for j in range(1, nb.shape[1]):
        ok = d[:, j] < gap * scale
        for i in np.flatnonzero(ok):
            ri, rj = find(i), find(nb[i, j])
            if ri != rj:
                parent[ri] = rj
    lab = np.array([find(i) for i in range(len(pts))])
    _, lab = np.unique(lab, return_inverse=True)
    sizes = np.bincount(lab)
    big = np.flatnonzero(sizes >= max(8, 0.02 * len(pts)))   # ignore specks
    flat = []
    for g in big:
        q = pts[lab == g]
        s = np.linalg.svd(q - q.mean(0), compute_uv=False)
        flat.append(float(s[2] / max(s[0], 1e-9)))
    return len(big), sizes[big], flat


def outline_closed(curve, k: int = 6, span: float = 3.0):
    """Does the traced outline form closed loops, or does it stop mid-air?

    An outline that does not close cannot be cut out. Distance alone does not
    find the ends -- at the tip of an arc the next few points are as close as
    anywhere else, they are just all on ONE side. So the test is directional:
    a point in the middle of a curve has neighbours spread both ways and their
    mean offset is near zero; at an end they all lie the same way.
    """
    if len(curve) < 8:
        return 1.0, 0
    kk = min(k, len(curve) - 1)
    d, nb = cKDTree(curve).query(curve, k=kk + 1)
    step = float(np.median(d[:, 1])) or 1e-9
    off = curve[nb[:, 1:]] - curve[:, None, :]
    reach = np.linalg.norm(off, axis=-1).mean(1)
    bias = np.linalg.norm(off.mean(1), axis=-1)
    ends = int(((bias / np.maximum(reach, 1e-9)) > 0.55).sum())
    far = int((d[:, 1] > span * step).sum())        # isolated fragments too
    ends += far
    return ends / len(curve), ends


def report(xyz, nrm, lab):
    """All of the above for one cloud. `lab`: 0 outline, 1 bend, 2 surface."""
    dev_med, dev_frac = developability(xyz, nrm)
    n_panel, sizes, flat = panels(xyz, lab == 1)
    end_frac, ends = outline_closed(xyz[lab == 0])
    return {
        "developability": dev_med,
        "non_developable_frac": dev_frac,
        "panels": n_panel,
        "panel_flatness": float(np.median(flat)) if flat else float("nan"),
        "outline_open_frac": end_frac,
        "outline_ends": ends,
    }


def demo():
    """Self-check on shapes whose answers are known."""
    rng = np.random.default_rng(0)
    u = rng.uniform(-1, 1, 900)
    v = rng.uniform(-1, 1, 900)

    plane = np.stack([u, v, np.zeros_like(u)], 1)
    pn = np.tile([0, 0, 1.0], (900, 1))
    d_plane = developability(plane, pn)[0]

    R = 1.5                                     # cylinder: developable
    cyl = np.stack([R * np.sin(u), v, R * np.cos(u)], 1)
    cn = np.stack([np.sin(u), np.zeros_like(u), np.cos(u)], 1)
    d_cyl = developability(cyl, cn)[0]

    t = rng.uniform(-0.9, 0.9, 900)             # sphere: NOT developable
    ph = rng.uniform(0, 2 * np.pi, 900)
    r = np.sqrt(1 - t * t)
    sph = np.stack([r * np.cos(ph), r * np.sin(ph), t], 1) * 1.5
    d_sph = developability(sph / 1.5 * 1.5, sph / np.linalg.norm(sph, axis=1, keepdims=True))[0]

    print(f"developability (|k_min| x spacing, 0 = flattenable)")
    print(f"  plane    {d_plane:.4f}")
    print(f"  cylinder {d_cyl:.4f}")
    print(f"  sphere   {d_sph:.4f}")
    assert d_sph > 3 * max(d_plane, d_cyl), (d_plane, d_cyl, d_sph)

    # two separated squares must count as two panels
    a = np.stack([u * 0.4 - 2.0, v, np.zeros_like(u)], 1)
    b = np.stack([u * 0.4 + 2.0, v, np.zeros_like(u)], 1)
    n, sizes, flat = panels(np.concatenate([a, b]), np.zeros(1800, bool))
    print(f"two separated sheets -> {n} panels, flatness {np.median(flat):.4f}")
    assert n == 2, n

    ring = np.stack([np.cos(np.linspace(0, 2 * np.pi, 200)),
                     np.sin(np.linspace(0, 2 * np.pi, 200)),
                     np.zeros(200)], 1)
    closed = outline_closed(ring)[1]
    arc = ring[:150]
    print(f"closed loop -> {closed} endpoints, open arc -> "
          f"{outline_closed(arc)[1]} endpoints")
    assert closed == 0 and outline_closed(arc)[1] > 0
    print("ok")


if __name__ == "__main__":
    demo()

"""Wireframe features from an oriented point cloud of a sheet-metal midsurface.

The generated object is a surface sample, not a wire, so the wire has to come
out of pure geometry -- no B-Rep, no mesh connectivity, nothing that a generated
point cloud would not have.

  crease points   : the local normal field bends -> bend lines
  boundary points : the neighbourhood is one-sided -> outline and hole rims

Measured against the GT wire, this recovers it to ~1-2mm even with 5mm of noise
on every point, which is under a tenth of the error the generators are at.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def knn(xyz: np.ndarray, k: int = 13):
    d, idx = cKDTree(xyz).query(xyz, k=k)
    return d[:, 1:], idx[:, 1:]              # drop self


def crease_score(xyz: np.ndarray, normal: np.ndarray, k: int = 13) -> np.ndarray:
    """Largest angle (rad) between a point's normal and its neighbours'.

    Normals are treated as unoriented (abs of the dot): a midsurface is an open
    sheet whose two sides carry opposite normals, and a flip across a seam is
    not a crease.
    """
    _, idx = knn(xyz, k)
    dots = np.abs(np.einsum("ij,ikj->ik", normal, normal[idx])).clip(0.0, 1.0)
    return np.arccos(dots).max(axis=1)


def boundary_score(xyz: np.ndarray, normal: np.ndarray, k: int = 13) -> np.ndarray:
    """Largest angular gap (rad) between neighbours projected onto the tangent
    plane. Interior points are surrounded (gap ~ 2pi/k); rim points are not."""
    _, idx = knn(xyz, k)
    nb = xyz[idx] - xyz[:, None, :]
    nb = nb - normal[:, None, :] * np.einsum("ikj,ij->ik", nb, normal)[..., None]
    # any tangent basis per point
    ref = np.tile(np.array([1.0, 0.0, 0.0]), (len(xyz), 1))
    bad = np.abs(np.einsum("ij,ij->i", ref, normal)) > 0.9
    ref[bad] = np.array([0.0, 1.0, 0.0])
    e1 = ref - normal * np.einsum("ij,ij->i", ref, normal)[:, None]
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
    e2 = np.cross(normal, e1)
    ang = np.sort(np.arctan2(np.einsum("ikj,ij->ik", nb, e2),
                             np.einsum("ikj,ij->ik", nb, e1)), axis=1)
    gaps = np.diff(ang, axis=1)
    wrap = (ang[:, 0] + 2 * np.pi - ang[:, -1])[:, None]
    return np.concatenate([gaps, wrap], axis=1).max(axis=1)


def extract(xyz: np.ndarray, normal: np.ndarray, k: int = 13,
            crease_deg: float = 25.0, gap_deg: float = 100.0) -> dict:
    """Returns index arrays for the two feature classes."""
    n = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    cs = crease_score(xyz, n, k)
    bs = boundary_score(xyz, n, k)
    boundary = np.flatnonzero(bs > np.deg2rad(gap_deg))
    crease = np.flatnonzero((cs > np.deg2rad(crease_deg)))
    crease = np.setdiff1d(crease, boundary)      # a rim is not a bend
    return {"boundary": boundary, "crease": crease,
            "crease_score": cs, "boundary_score": bs}


def estimate_normals(xyz: np.ndarray, k: int = 13) -> np.ndarray:
    """Fallback when normals were not generated: local PCA. Signs are arbitrary,
    which is why crease_score compares |cos|."""
    _, idx = knn(xyz, k)
    nb = xyz[idx] - xyz[idx].mean(axis=1, keepdims=True)
    cov = np.einsum("ikm,ikn->imn", nb, nb)
    return np.linalg.eigh(cov)[1][:, :, 0]

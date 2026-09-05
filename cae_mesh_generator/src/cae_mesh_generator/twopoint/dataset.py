"""Two-point synthetic parts: condition canonicalization + parameter vectors.

Data: PartMaker/synthetic_parts (1,300 parts, params JSON is the ground-truth
DOF vector). Condition C = fastening points (pos+normal), thickness, hole
diameter, bearing radius. Free DOF theta (12-dim, class-masked) + discrete c.

Canonical frame (condition-derived, theory definition 4 style):
  origin = p1, ex = (p2-p1)/L, ey = orthonormalized n1, ez = ex x ey.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

DEFAULT_BASE = pathlib.Path("C:/Users/hide2/IdeaBox/PartMaker/synthetic_parts")

# unified continuous DOF layout (class-irrelevant dims are masked)
THETA_NAMES = (
    "half_width_mm", "bend_radius_mm", "fold1_slack_mm", "fold2_slack_mm",
    "fold1_tilt_perturbation_rad",
    "flange_height_mm", "flange_root_radius_mm",
    "bead_depth_mm", "bead_top_width_mm", "bead_wall_angle_deg",
    "bead_ridge_radius_mm", "bead_corner_radius_mm",
)
COMMON_DIMS = tuple(range(5))
FLANGE_DIMS = COMMON_DIMS + (5, 6)
BEAD_DIMS = COMMON_DIMS + (7, 8, 9, 10, 11)
CLASSES = ("flange", "bead")


def canonical_frame(p1, n1, p2):
    ex = p2 - p1
    L = float(np.linalg.norm(ex))
    ex = ex / max(L, 1e-9)
    ey = n1 - (n1 @ ex) * ex
    ny = np.linalg.norm(ey)
    if ny < 1e-6:  # n1 parallel to the axis: pick any stable perpendicular
        ey = np.cross(ex, [0.0, 0.0, 1.0])
        if np.linalg.norm(ey) < 1e-6:
            ey = np.cross(ex, [0.0, 1.0, 0.0])
        ny = np.linalg.norm(ey)
    ey = ey / ny
    ez = np.cross(ex, ey)
    return np.stack([ex, ey, ez]), L


def load_parts(base: pathlib.Path = DEFAULT_BASE) -> list[dict]:
    dirs = [base / "batch02/params"] + [base / f"prod01/chunk_0{i}/params"
                                        for i in range(1, 7)]
    out = []
    for d in dirs:
        for f in sorted(d.glob("*.json")):
            p = json.loads(f.read_text(encoding="utf-8"))
            s = p["spec"]
            p1 = np.asarray(s["point1"]["position_xyz"])
            n1 = np.asarray(s["point1"]["normal_xyz"])
            p2 = np.asarray(s["point2"]["position_xyz"])
            n2 = np.asarray(s["point2"]["normal_xyz"])
            R, L = canonical_frame(p1, n1, p2)
            n1c, n2c = R @ n1, R @ n2
            cond = np.array([
                L / 100.0, *n1c, *n2c, s["thickness_mm"], s["hole_diameter_mm"] / 10.0,
                s["min_bearing_radius_mm"] / 10.0,
            ], dtype=np.float32)  # 10 dims

            theta = np.zeros(len(THETA_NAMES), dtype=np.float32)
            theta[0] = s["half_width_mm"]
            theta[1] = s["bend_radius_mm"]
            theta[2] = s["fold1_slack_mm"]
            theta[3] = s["fold2_slack_mm"]
            theta[4] = s["fold1_tilt_perturbation_rad"]
            if p.get("flange") is not None:
                cls = 0
                fl = p["flange"]
                theta[5] = fl["height_mm"]
                theta[6] = fl["root_radius_mm"]
                discrete = (int(fl["side"] > 0), int(fl["direction"] > 0))
            else:
                cls = 1
                bd = p["bead"]
                theta[7] = bd["depth_mm"]
                theta[8] = bd["top_width_mm"]
                theta[9] = bd["wall_angle_deg"]
                theta[10] = bd["ridge_radius_mm"]
                theta[11] = bd["corner_radius_mm"]
                discrete = (0, 0)
            out.append({
                "part_id": p["part_id"], "path": str(f), "cond": cond,
                "theta": theta, "cls": cls, "flange_bits": discrete,
                "normal_dot": float(n1 @ n2),
                "frame_R": R, "p1": p1, "L": L,
                "fold_tilts": p.get("fold_tilts_deg"),
            })
    return out


def theta_mask(cls: int) -> np.ndarray:
    m = np.zeros(len(THETA_NAMES), dtype=bool)
    m[list(FLANGE_DIMS if cls == 0 else BEAD_DIMS)] = True
    return m


def stratified_split(parts: list[dict], seed: int = 13, val_fraction: float = 0.1):
    """Stratify by (class, normal-dot bin) so val covers the configuration space."""
    rng = np.random.default_rng(seed)
    buckets: dict[tuple, list[int]] = {}
    for i, p in enumerate(parts):
        dot_bin = int(np.clip((p["normal_dot"] + 1.0) / 0.5, 0, 3))
        buckets.setdefault((p["cls"], dot_bin), []).append(i)
    val_idx = set()
    for ids in buckets.values():
        ids = list(ids)
        rng.shuffle(ids)
        n = max(1, int(round(len(ids) * val_fraction)))
        val_idx.update(ids[:n])
    train = [p for i, p in enumerate(parts) if i not in val_idx]
    val = [p for i, p in enumerate(parts) if i in val_idx]
    return train, val


class ThetaNormalizer:
    """Per-dim robust affine map to roughly [-1, 1] using train quantiles."""

    def __init__(self, train_parts: list[dict]):
        thetas = np.stack([p["theta"] for p in train_parts])
        masks = np.stack([theta_mask(p["cls"]) for p in train_parts])
        self.lo = np.zeros(thetas.shape[1], dtype=np.float32)
        self.hi = np.ones(thetas.shape[1], dtype=np.float32)
        for k in range(thetas.shape[1]):
            vals = thetas[masks[:, k], k]
            if len(vals):
                self.lo[k] = np.quantile(vals, 0.01)
                self.hi[k] = np.quantile(vals, 0.99)
                if self.hi[k] - self.lo[k] < 1e-6:
                    self.hi[k] = self.lo[k] + 1.0

    def encode(self, theta: np.ndarray) -> np.ndarray:
        return (2.0 * (theta - self.lo) / (self.hi - self.lo) - 1.0).astype(np.float32)

    def decode(self, z: np.ndarray) -> np.ndarray:
        return ((z + 1.0) / 2.0 * (self.hi - self.lo) + self.lo).astype(np.float32)

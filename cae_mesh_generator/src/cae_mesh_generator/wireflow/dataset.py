"""Wireframe flow dataset: typed-wireframe teacher + joint-constraint conditioning.

Teacher: wireframe v4.4 JSON (fill_volume/fill_mid_surf/<asm>/wireframes/) with
manual overrides merged. Points are resampled along polylines every epoch,
stratified by edge-type group. Constraints come from annotations/joints.json.

Assemblies: A0072600002 (43 parts) + A0072601285 (32 parts). A0072600529 is
excluded (no joint annotations, user decision 2026-07-08).
"""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_BASE = Path("C:/Users/hide2/IdeaBox/fill_volume/fill_mid_surf")
ASSEMBLIES = ("A0072600002_AllCATPart", "A0072601285_AllCATPart")

TYPE_NAMES = ("outer_boundary", "hole_boundary", "bend_line", "surface_frame")
TYPE_REMAP = {"crease": "bend_line", "feature_rim": "bend_line"}  # tiny classes
TYPE_FRACTIONS = {"outer_boundary": 0.30, "hole_boundary": 0.15,
                  "bend_line": 0.30, "surface_frame": 0.25}
JOINT_TYPES = ("weld", "bolt", "mounting_hole")  # anything else -> 'other' slot
JOINT_FEAT_DIM = 3 + 3 + len(JOINT_TYPES) + 1 + 1  # xyz, dir, onehot, other, size
MAX_JOINTS = 64
DEFAULT_PAIRS = Path("C:/Users/hide2/IdeaBox/AutoMetalSheet/runs/cga_dataset_mesh/parts")
RELATION_TAU = 0.1           # softmax temperature on normalized distances
HEURISTIC_NORMAL_BETA = 1.0  # euclid penalty for incompatible normals


def stable_seed(base: int, key: str) -> int:
    return int((int(base) + zlib.crc32(key.encode("utf-8"))) % (2**32 - 1))


def part_id_from_stem(stem: str) -> str:
    return stem.split("_")[0] if "_" in stem else stem


@dataclass
class PartWireframe:
    assembly: str
    stem: str
    part_id: str
    center: np.ndarray
    scale: float
    # per type group: (seg_start (S,3), seg_vec (S,3), cum_len (S,)) in world mm
    segments: dict = field(default_factory=dict)
    joints: np.ndarray = None  # (K, JOINT_FEAT_DIM) normalized, K<=MAX_JOINTS
    d_geo: np.ndarray = None   # (K, K) geodesic distance / scale, or None
    d_euc: np.ndarray = None   # (K, K) euclidean distance / scale, or None

    @property
    def name(self) -> str:
        return f"{self.assembly}:{self.stem}"


def load_wireframe_segments(path: Path) -> tuple[dict, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    overrides_path = path.with_name(path.stem + ".overrides.json")
    overrides = (
        json.loads(overrides_path.read_text(encoding="utf-8")) if overrides_path.exists() else {}
    )
    groups: dict[str, list] = {t: [] for t in TYPE_NAMES}
    all_pts = []
    for e in data["edges"]:
        etype = overrides.get(e.get("fingerprint"), e["type"])
        etype = TYPE_REMAP.get(etype, etype)
        poly = np.asarray(e["polyline"], dtype=np.float64)
        if len(poly) >= 2:
            all_pts.append(poly)
        if etype not in groups or len(poly) < 2:
            continue
        groups[etype].append(poly)
    segments = {}
    for t, polys in groups.items():
        if not polys:
            continue
        starts, vecs = [], []
        for poly in polys:
            starts.append(poly[:-1])
            vecs.append(poly[1:] - poly[:-1])
        starts = np.concatenate(starts)
        vecs = np.concatenate(vecs)
        lens = np.linalg.norm(vecs, axis=1)
        keep = lens > 1e-9
        if not keep.any():
            continue
        segments[t] = (starts[keep], vecs[keep], np.cumsum(lens[keep]))
    pts = np.concatenate(all_pts) if all_pts else np.zeros((1, 3))
    return segments, pts


def load_joints(base: Path, assembly: str) -> dict[str, list[dict]]:
    path = base / assembly / "annotations" / "joints.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    per_part: dict[str, list[dict]] = {}
    for j in data.get("joints", []):
        axis = j.get("axis") or {}
        direction = axis.get("direction_xyz") or [0.0, 0.0, 1.0]
        for pp in j.get("per_part", []):
            xyz = pp.get("hole_center_xyz") or pp.get("contact_xyz")
            if xyz is None:
                continue
            size = pp.get("hole_diameter_mm") or axis.get("length_mm") or 0.0
            per_part.setdefault(str(pp["part_id"]), []).append(
                {"xyz": xyz, "dir": direction, "type": j.get("type", "other"), "size": size}
            )
    return per_part


def joint_features(joints: list[dict], center: np.ndarray, scale: float) -> np.ndarray:
    out = np.zeros((len(joints), JOINT_FEAT_DIM), dtype=np.float32)
    for i, j in enumerate(joints):
        out[i, 0:3] = (np.asarray(j["xyz"]) - center) / scale
        out[i, 3:6] = np.asarray(j["dir"])
        if j["type"] in JOINT_TYPES:
            out[i, 6 + JOINT_TYPES.index(j["type"])] = 1.0
        else:
            out[i, 6 + len(JOINT_TYPES)] = 1.0
        out[i, -1] = j["size"] / scale
    return out


def discover_parts(base: Path = DEFAULT_BASE,
                   assemblies: tuple[str, ...] = ASSEMBLIES,
                   pairs_dir: Path | None = None) -> list[PartWireframe]:
    parts = []
    for asm in assemblies:
        joints_map = load_joints(base, asm)
        wf_dir = base / asm / "wireframes"
        for f in sorted(wf_dir.glob("*.json")):
            if f.name.endswith(".overrides.json"):
                continue
            segments, pts = load_wireframe_segments(f)
            if not segments:
                continue
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            center = (lo + hi) * 0.5
            scale = float(max(hi - lo)) or 1.0
            pid = part_id_from_stem(f.stem)
            joints_list = joints_map.get(pid, [])
            # subsample indices are shared with the relation matrices below
            if len(joints_list) > MAX_JOINTS:
                rng = np.random.default_rng(stable_seed(0, f.stem))
                sel = np.sort(rng.choice(len(joints_list), MAX_JOINTS, replace=False))
            else:
                sel = np.arange(len(joints_list))
            jf = joint_features([joints_list[i] for i in sel], center, scale)
            d_geo = d_euc = None
            if pairs_dir is not None:
                npz = Path(pairs_dir) / f"{asm}__{f.stem}.npz"
                if npz.exists():
                    d = np.load(npz)
                    d_geo = np.nan_to_num(d["d_geo"], posinf=1e6)[np.ix_(sel, sel)] / scale
                    d_euc = d["d_euclid"][np.ix_(sel, sel)] / scale
            parts.append(PartWireframe(asm, f.stem, pid, center, scale, segments, jf,
                                       d_geo, d_euc))
    return parts


def split_parts(parts: list[PartWireframe], seed: int, val_fraction: float
                ) -> tuple[list[PartWireframe], list[PartWireframe]]:
    order = np.random.default_rng(seed).permutation(len(parts))
    n_val = max(1, int(round(len(parts) * val_fraction)))
    val_idx = set(order[:n_val].tolist())
    train = [p for i, p in enumerate(parts) if i not in val_idx]
    val = [p for i, p in enumerate(parts) if i in val_idx]
    return train, val


def allocate_counts(segments: dict, n_points: int) -> dict[str, int]:
    present = {t: TYPE_FRACTIONS[t] for t in segments}
    total = sum(present.values())
    counts = {t: int(round(n_points * f / total)) for t, f in present.items()}
    # fix rounding drift on the largest group
    drift = n_points - sum(counts.values())
    if drift and counts:
        counts[max(counts, key=counts.get)] += drift
    return counts


def sample_wireframe(part: PartWireframe, n_points: int, rng: np.random.Generator
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns normalized (points (N,3), tangents (N,3), type_ids (N,))."""
    counts = allocate_counts(part.segments, n_points)
    pts, tans, tids = [], [], []
    for t, n in counts.items():
        if n <= 0:
            continue
        starts, vecs, cum = part.segments[t]
        r = rng.uniform(0.0, cum[-1], n)
        idx = np.searchsorted(cum, r)
        prev = np.where(idx > 0, cum[idx - 1], 0.0)
        seg_len = cum[idx] - prev
        frac = np.where(seg_len > 0, (r - prev) / np.maximum(seg_len, 1e-12), 0.0)
        p = starts[idx] + vecs[idx] * frac[:, None]
        tan = vecs[idx] / np.maximum(np.linalg.norm(vecs[idx], axis=1, keepdims=True), 1e-12)
        pts.append(p)
        tans.append(tan)
        tids.append(np.full(n, TYPE_NAMES.index(t), dtype=np.int64))
    points = (np.concatenate(pts) - part.center) / part.scale
    return (points.astype(np.float32), np.concatenate(tans).astype(np.float32),
            np.concatenate(tids))


class WireflowDataset(Dataset):
    """One item = one part variant; points resampled every epoch."""

    def __init__(self, parts: list[PartWireframe], n_points: int = 1024,
                 mirror_axes: tuple[str, ...] = (), base_seed: int = 0):
        self.parts = parts
        self.n_points = n_points
        self.base_seed = base_seed
        self.epoch = 0
        self.variants: list[tuple[int, str]] = [(i, "") for i in range(len(parts))]
        axis_index = {"x": 0, "y": 1, "z": 2}
        self.axis_index = axis_index
        for ax in mirror_axes:
            self.variants += [(i, ax) for i in range(len(parts))]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.variants)

    def __getitem__(self, index: int) -> dict:
        part_idx, axis = self.variants[index]
        part = self.parts[part_idx]
        rng = np.random.default_rng(
            stable_seed(self.base_seed + 1000003 * self.epoch, f"{part.name}|{axis}")
        )
        points, tangents, type_ids = sample_wireframe(part, self.n_points, rng)
        joints = part.joints.copy()
        if axis:
            k = self.axis_index[axis]
            points[:, k] *= -1
            tangents[:, k] *= -1
            if len(joints):
                joints[:, k] *= -1        # xyz
                joints[:, 3 + k] *= -1    # direction
        joints_pad = np.zeros((MAX_JOINTS, JOINT_FEAT_DIM), dtype=np.float32)
        joints_mask = np.zeros(MAX_JOINTS, dtype=bool)
        if len(joints):
            joints_pad[: len(joints)] = joints
            joints_mask[: len(joints)] = True
        # relation attention biases (mirror-invariant: distances and |n.n| survive flips)
        rel_oracle = np.zeros((MAX_JOINTS, MAX_JOINTS), dtype=np.float32)
        rel_heur = np.zeros((MAX_JOINTS, MAX_JOINTS), dtype=np.float32)
        if part.d_geo is not None:
            K = len(part.d_geo)
            rel_oracle[:K, :K] = -part.d_geo / RELATION_TAU
            dirs = joints[:, 3:6]
            ndot = np.abs(dirs @ dirs.T)
            d_eff = part.d_euc * (1.0 + HEURISTIC_NORMAL_BETA * (1.0 - ndot))
            rel_heur[:K, :K] = -d_eff / RELATION_TAU
        lo, hi = points.min(axis=0), points.max(axis=0)
        bbox = np.concatenate([(hi - lo) * 0.5, [np.log(part.scale)]]).astype(np.float32)
        return {
            "points": torch.from_numpy(points),
            "tangents": torch.from_numpy(tangents),
            "type_ids": torch.from_numpy(type_ids),
            "joints": torch.from_numpy(joints_pad),
            "joints_mask": torch.from_numpy(joints_mask),
            "relation_oracle": torch.from_numpy(rel_oracle),
            "relation_heuristic": torch.from_numpy(rel_heur),
            "bbox": torch.from_numpy(bbox),
            "scale": torch.tensor(part.scale, dtype=torch.float32),
            "part_index": torch.tensor(part_idx, dtype=torch.long),
            "mirrored": axis or "",
        }

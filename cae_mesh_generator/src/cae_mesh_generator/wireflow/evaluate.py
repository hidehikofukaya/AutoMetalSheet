"""Evaluate the wireframe flow AE: calibrated Chamfer, constraint satisfaction,
retrieval baseline, and viewer-compatible JSON export.

Usage:
  python -m cae_mesh_generator.wireflow.evaluate --checkpoint ../runs/wireflow_v1/best_zon.pt \
      --output-dir ../runs/wireflow_v1_eval --device cuda
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import pathlib

import numpy as np
import torch

from .dataset import (DEFAULT_BASE, DEFAULT_PAIRS, TYPE_NAMES, WireflowDataset,
                      discover_parts, sample_wireframe, split_parts, stable_seed)
from .model import WireFlowModel

EVAL_GT_POINTS = 4096


def chamfer_mm(a: np.ndarray, b: np.ndarray, scale: float) -> dict:
    """a: generated (M,3) normalized, b: GT (N,3) normalized."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    a2b = d.min(axis=1) * scale
    b2a = d.min(axis=0) * scale
    return {"gen_to_gt_mean": float(a2b.mean()), "gt_to_gen_mean": float(b2a.mean()),
            "chamfer": float(a2b.mean() + b2a.mean()),
            "gen_p95": float(np.percentile(a2b, 95)), "gt_p95": float(np.percentile(b2a, 95))}


def sampling_floor(part, n_gen: int, rng) -> float:
    p1, _, _ = sample_wireframe(part, n_gen, rng)
    p2, _, _ = sample_wireframe(part, EVAL_GT_POINTS, rng)
    d = np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=-1)
    return float((d.min(axis=1).mean() + d.min(axis=0).mean()) * part.scale)


def joint_set_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Chamfer between two normalized joint xyz sets (retrieval key)."""
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        return 10.0
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(d.min(axis=1).mean() + d.min(axis=0).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--base-dir", default=str(DEFAULT_BASE))
    ap.add_argument("--n-points", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    targs = ckpt["args"]
    model = WireFlowModel(targs["dim"], 8, targs["enc_layers"], targs["dec_layers"],
                          targs["n_latent"]).to(args.device)
    # strict=False: checkpoints from before the relation encoder lack its keys;
    # for those, relation_mode is None and the encoder is skipped entirely.
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    relation_mode = targs.get("relation_bias")  # None for pre-relation checkpoints
    parts = discover_parts(pathlib.Path(args.base_dir), pairs_dir=DEFAULT_PAIRS)
    train_parts, val_parts = split_parts(parts, targs["split_seed"], targs["val_fraction"])
    print(f"relation bias mode from checkpoint: {relation_mode}")
    out = pathlib.Path(args.output_dir)
    (out / "generated").mkdir(parents=True, exist_ok=True)

    results = []
    for split_name, split in (("train", train_parts), ("val", val_parts)):
        ds = WireflowDataset(split, args.n_points, (), base_seed=777)
        for i, part in enumerate(split):
            item = {k: (v.unsqueeze(0).to(args.device) if torch.is_tensor(v) else v)
                    for k, v in ds[i].items()}
            rng = np.random.default_rng(stable_seed(999, part.name))
            gt, _, gt_types = sample_wireframe(part, EVAL_GT_POINTS, rng)
            floor = sampling_floor(part, args.n_points, rng)
            row = {"part": part.name, "split": split_name, "scale_mm": part.scale,
                   "n_joints": int(item["joints_mask"].sum()), "floor_mm": floor}

            rel_bias = (model.relation_bias_from_batch(item, relation_mode)
                        if relation_mode else None)
            for mode in ("zon", "zoff"):
                z = (model.encoder(item["points"], item["type_ids"], item["tangents"])
                     if mode == "zon" else None)
                gen, gen_types = model.sample(
                    item["joints"], item["joints_mask"], item["bbox"], args.n_points,
                    steps=args.steps, z=z, guidance=args.guidance,
                    generator=torch.Generator(args.device).manual_seed(1),
                    relation_bias=rel_bias)
                gen_np = gen[0].cpu().numpy()
                types_np = gen_types[0].cpu().numpy()
                cm = chamfer_mm(gen_np, gt, part.scale)
                row[f"{mode}_chamfer_mm"] = cm["chamfer"]
                row[f"{mode}_gen_p95_mm"] = cm["gen_p95"]
                row[f"{mode}_gt_p95_mm"] = cm["gt_p95"]
                row[f"{mode}_chamfer_to_floor"] = cm["chamfer"] / max(floor, 1e-9)
                # per-type gt coverage (how well each wire class is reproduced)
                for ti, tname in enumerate(TYPE_NAMES):
                    sel = gt_types == ti
                    if sel.sum() > 4:
                        d = np.linalg.norm(gt[sel][:, None, :] - gen_np[None, :, :], axis=-1)
                        row[f"{mode}_{tname}_gt_p95_mm"] = float(
                            np.percentile(d.min(axis=1), 95) * part.scale)
                # constraint satisfaction: joints -> nearest generated point
                jm = item["joints_mask"][0].cpu().numpy()
                if jm.any():
                    jxyz = item["joints"][0].cpu().numpy()[jm][:, :3]
                    d = np.linalg.norm(jxyz[:, None, :] - gen_np[None, :, :], axis=-1)
                    nearest = d.min(axis=1) * part.scale
                    row[f"{mode}_constraint_mean_mm"] = float(nearest.mean())
                    row[f"{mode}_constraint_max_mm"] = float(nearest.max())
                # viewer export: points as short typed dashes (0.8mm along +z)
                edges = [{"id": k, "type": TYPE_NAMES[int(types_np[k])], "length_mm": 0.8,
                          "closed": False, "face_types": [], "fingerprint": f"g{k}",
                          "polyline": [
                              (gen_np[k] * part.scale + part.center).round(3).tolist(),
                              (gen_np[k] * part.scale + part.center + [0, 0, 0.8]).round(3).tolist(),
                          ]} for k in range(len(gen_np))]
                counts: dict[str, int] = {}
                for e in edges:
                    counts[e["type"]] = counts.get(e["type"], 0) + 1
                (out / "generated" / f"{part.assembly}__{part.stem}__{mode}.json").write_text(
                    json.dumps({"schema": "wireframe v4.4", "source_stp": "generated",
                                "n_faces": 0, "face_types": [], "clusters": [],
                                "edge_counts": counts, "edges": edges}), encoding="utf-8")

            # retrieval baseline (val only): nearest train part by joint configuration
            if split_name == "val":
                jxyz_val = part.joints[:, :3] if len(part.joints) else np.zeros((0, 3))
                dists = [joint_set_distance(jxyz_val,
                                            tp.joints[:, :3] if len(tp.joints) else np.zeros((0, 3)))
                         for tp in train_parts]
                nn_part = train_parts[int(np.argmin(dists))]
                nn_pts, _, _ = sample_wireframe(nn_part, args.n_points,
                                                np.random.default_rng(1))
                cm = chamfer_mm(nn_pts, gt, part.scale)
                row["retrieval_part"] = nn_part.name
                row["retrieval_chamfer_mm"] = cm["chamfer"]
            results.append(row)
            print(f"[{split_name}] {part.name}: "
                  f"zon {row['zon_chamfer_mm']:.2f}mm zoff {row['zoff_chamfer_mm']:.2f}mm "
                  f"floor {floor:.2f}mm", flush=True)

    (out / "metrics.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    for split_name in ("train", "val"):
        rows = [r for r in results if r["split"] == split_name]
        agg = {"split": split_name, "count": len(rows)}
        for key in ("zon_chamfer_mm", "zoff_chamfer_mm", "zon_chamfer_to_floor",
                    "zoff_chamfer_to_floor", "zon_constraint_mean_mm",
                    "zoff_constraint_mean_mm", "retrieval_chamfer_mm", "floor_mm"):
            vals = [r[key] for r in rows if key in r]
            if vals:
                agg[key] = float(np.mean(vals))
        print(json.dumps(agg, indent=1))
        results.append(agg)
    (out / "aggregate.json").write_text(
        json.dumps([r for r in results if "count" in r], indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

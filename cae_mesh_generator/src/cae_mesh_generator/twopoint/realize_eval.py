"""P2 gate: realizer floor = chamfer(realize(GT theta), GT mid STP) on val parts.

Usage:
  python -m cae_mesh_generator.twopoint.realize_eval --output-dir ../runs/twopoint_p2 \
      [--max-parts 40]
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import pathlib

import numpy as np

from ..data.step_tessellate import TessellationConfig, load_or_tessellate, sample_surface_points
from .dataset import load_parts, stratified_split
from .realize import realize_points
from ..wtok.train_ar import chamfer_mm  # noqa: F401  (chunked/subsampled chamfer)


def gt_points(params_path: pathlib.Path, cache: pathlib.Path, n: int = 4096) -> np.ndarray:
    stp = params_path.parent.parent / "mid" / f"{params_path.stem}_mid.stp"
    mesh = load_or_tessellate(stp, cache, TessellationConfig())
    pts, _ = sample_surface_points(mesh, n, seed=0)
    return pts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--split-seed", type=int, default=13)
    ap.add_argument("--max-parts", type=int, default=40)
    args = ap.parse_args()
    out = pathlib.Path(args.output_dir)
    (out / "tess_cache").mkdir(parents=True, exist_ok=True)

    parts = load_parts()
    _, val = stratified_split(parts, args.split_seed)
    rows = []
    for p in val[: args.max_parts]:
        path = pathlib.Path(p["path"])
        spec = json.loads(path.read_text(encoding="utf-8"))["spec"]
        gt = gt_points(path, out / "tess_cache")
        best = None
        try:
            for rise in ((1, -1) if p["cls"] == 1 else (1,)):
                gen = realize_points(spec, p["theta"], p["cls"],
                                     p["flange_bits"], bead_rise_sign=rise)
                cd = chamfer_mm(gen, gt)
                if best is None or cd < best[0]:
                    best = (cd, rise, len(gen))
        except ValueError as exc:
            rows.append({"part": p["part_id"], "cls": p["cls"], "error": str(exc)[:100]})
            print(f"[INFEASIBLE?] {p['part_id']}: {str(exc)[:90]}", flush=True)
            continue
        rows.append({"part": p["part_id"], "cls": p["cls"], "chamfer_mm": best[0],
                     "rise": best[1], "n_points": best[2]})
        print(f"{p['part_id']:36s} {'flange' if p['cls']==0 else 'bead':6s} "
              f"chamfer {best[0]:6.2f}mm (rise {best[1]:+d})", flush=True)

    ok = [r for r in rows if "chamfer_mm" in r]
    agg = {"n": len(rows), "n_ok": len(ok),
           "floor_mean_mm": float(np.mean([r["chamfer_mm"] for r in ok])) if ok else None,
           "floor_worst_mm": float(max(r["chamfer_mm"] for r in ok)) if ok else None}
    for c, name in ((0, "flange"), (1, "bead")):
        sub = [r["chamfer_mm"] for r in ok if r["cls"] == c]
        if sub:
            agg[f"floor_{name}_mean_mm"] = float(np.mean(sub))
            agg[f"floor_{name}_worst_mm"] = float(max(sub))
    (out / "floor.json").write_text(json.dumps({"aggregate": agg, "parts": rows},
                                               indent=1), encoding="utf-8")
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()

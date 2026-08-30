"""Kaggle launcher for the outline-frame sweep.

The question it is here to answer: the model does NOT overfit -- measured train
error 9.03mm against val 7.44mm, and the best checkpoint on a 300-epoch run
lands at epoch 260 still improving. Both point at capacity and duration, which
is the one lever this project dismissed without testing. The local GPU is busy
with the relational-loss runs, so the sweep goes here.

Every run is the same recipe with one axis moved, and every run reports the
MICRO indicators, because Chamfer moves 0.03mm when every wrong arc is corrected
by hand.
"""
from __future__ import annotations

import glob
import json
import os
import pathlib
import shutil
import subprocess
import sys


def _find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    return hits[0] if hits else None


def main():
    code = _find("kaggle_run.py")
    root = pathlib.Path(code).resolve().parents[2] if code else None
    if root is None:
        raise SystemExit("wtok-code dataset not attached")
    sys.path.insert(0, str(root))

    parts = _find("parts")
    data = pathlib.Path(parts).parent if parts else None
    if data is None:
        raise SystemExit("wtok-synth dataset not attached")

    work = pathlib.Path("/kaggle/working")
    # sidecar and frame_target read from a repo-shaped tree
    tree = work / "repo"
    (tree / "runs" / "wtok_synth").mkdir(parents=True, exist_ok=True)
    for name in ("parts", "spec_vectors.json"):
        src = data / name
        dst = tree / "runs" / "wtok_synth" / name
        if src.exists() and not dst.exists():
            (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)
    vl = data / "val_names_100.json"
    (tree / "runs" / "wtok_curve_synth_v1").mkdir(parents=True, exist_ok=True)
    if vl.exists():
        shutil.copy2(vl, tree / "runs" / "wtok_curve_synth_v1" / "val_names_100.json")
    # the package expects to sit at <root>/cae_mesh_generator/src/cae_mesh_generator
    pkg = tree / "cae_mesh_generator" / "src"
    pkg.mkdir(parents=True, exist_ok=True)
    if not (pkg / "cae_mesh_generator").exists():
        shutil.copytree(root / "cae_mesh_generator", pkg / "cae_mesh_generator")
    sys.path.insert(0, str(pkg))

    sweep = json.loads(os.environ.get("WTOK_SWEEP", "[]")) or [
        {"tag": "base", "dim": 256, "layers": 8, "epochs": 300},
        {"tag": "wide", "dim": 384, "layers": 8, "epochs": 300},
        {"tag": "deep", "dim": 256, "layers": 14, "epochs": 300},
        {"tag": "long", "dim": 256, "layers": 8, "epochs": 900},
    ]
    hours = float(os.environ.get("WTOK_MAX_HOURS", "8.5"))
    each = hours / max(len(sweep), 1)
    for cfg in sweep:
        out = work / f"frame_{cfg['tag']}"
        cmd = [sys.executable, "-u", "-m", "cae_mesh_generator.wtok.staged",
               "--stage", "outline_frame", "--use-spec",
               "--dataset", str(tree / "runs" / "mesh_synth"),
               "--wtok", str(tree / "runs" / "wtok_synth"),
               "--val-list", str(tree / "runs" / "wtok_curve_synth_v1"
                                 / "val_names_100.json"),
               "--output-dir", str(out),
               "--train-parts", str(cfg.get("parts", 1000)), "--val-parts", "50",
               "--epochs", str(cfg["epochs"]), "--batch-size", "16",
               "--dim", str(cfg["dim"]), "--layers", str(cfg["layers"]),
               "--heads", "8", "--cfg-scale", "1.0",
               "--probe-every", "20", "--probe-parts", "16",
               "--max-hours", f"{each:.2f}"]
        for k, flag in (("w_d1", "--w-d1"), ("w_d2", "--w-d2")):
            if cfg.get(k):
                cmd += [flag, str(cfg[k])]
        if cfg.get("rel_attn"):
            cmd += ["--rel-attn"]
        print("\n" + "=" * 70 + f"\n{cfg['tag']}: {' '.join(cmd[4:])}\n", flush=True)
        env = dict(os.environ, PYTHONPATH=str(pkg))
        subprocess.run(cmd, env=env, cwd=str(tree))


if __name__ == "__main__":
    main()

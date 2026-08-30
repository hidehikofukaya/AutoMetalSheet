"""Kaggle launcher for the outline-frame sweep.

The question it is here to answer: the model does NOT overfit -- measured train
error 9.03mm against val 7.44mm, and the best checkpoint on a 300-epoch run
lands at epoch 260 still improving. Both point at capacity and duration, which
is the one lever this project dismissed without testing.

The first version copied the code and data into /kaggle/working and ran from
there; the kernel completed with the copy as its only output and no training
log. This one touches nothing: the package is imported from the read-only input
dataset, the data is read where it is mounted, and only the run directories are
written. Every run reports the rationality tiers, not Chamfer.
"""
from __future__ import annotations

import glob
import json
import os
import pathlib
import subprocess
import sys


def _find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    return hits[0] if hits else None


def main():
    code = _find("kaggle_frame.py")
    if code is None:
        raise SystemExit("wtok-code dataset not attached")
    # .../cae_mesh_generator/wtok/kaggle_frame.py -> the dir holding the package
    root = pathlib.Path(code).resolve().parents[2]

    parts = _find("parts")
    if parts is None:
        raise SystemExit("wtok-synth dataset not attached")
    data = pathlib.Path(parts).parent
    val = data / "val_names_100.json"
    work = pathlib.Path("/kaggle/working")

    env = dict(os.environ, PYTHONPATH=str(root),
               WTOK_SPEC_FILE=str(data / "spec_vectors.json"))
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
               "--stage", "outline_frame", "--use-spec", "--rel-attn",
               "--dataset", str(data), "--wtok", str(data),
               "--val-list", str(val), "--output-dir", str(out),
               "--train-parts", str(cfg.get("parts", 1000)), "--val-parts", "50",
               "--epochs", str(cfg["epochs"]), "--batch-size", "16",
               "--dim", str(cfg["dim"]), "--layers", str(cfg["layers"]),
               "--heads", "8", "--cfg-scale", "1.0",
               "--probe-every", "20", "--probe-parts", "16",
               "--max-hours", f"{each:.2f}"]
        print("\n" + "=" * 70 + f"\n{cfg['tag']}: {' '.join(cmd[4:])}\n", flush=True)
        rc = subprocess.run(cmd, env=env, cwd=str(work)).returncode
        print(f"{cfg['tag']}: exit {rc}", flush=True)


if __name__ == "__main__":
    main()

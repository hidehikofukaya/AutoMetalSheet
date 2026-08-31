"""Kaggle launcher for the KB 20 chain stages (2a chain_set / 2b chain_edges).

Runs from the read-only inputs (the kaggle_frame.py lesson), reads the
precomputed chain targets via WTOK_CHAINS (Kaggle cannot see the 244MB
wireframes), and gives every arm ITS OWN SEED -- the g1a==g1b sweep trained
twice bit-identically because the launcher had none (KB 18.6).
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
    code = _find("kaggle_chains.py")
    if code is None:
        raise SystemExit("wtok-code dataset not attached")
    root = pathlib.Path(code).resolve().parents[2]
    parts = _find("parts")
    if parts is None:
        raise SystemExit("wtok-synth dataset not attached")
    data = pathlib.Path(parts).parent
    chains = data / "chain_targets"
    if not chains.exists():
        raise SystemExit("chain_targets missing from the data dataset")
    val = data / "val_names_100.json"
    work = pathlib.Path("/kaggle/working")

    env = dict(os.environ, PYTHONPATH=str(root),
               WTOK_SPEC_FILE=str(data / "spec_vectors.json"),
               WTOK_CHAINS=str(chains))
    sweep = json.loads(os.environ.get("WTOK_SWEEP") or "[]") or [
        {"tag": "2a_s0", "stage": "chain_set", "epochs": 900, "seed": 0,
         "batch": 16},
        {"tag": "2a_s1", "stage": "chain_set", "epochs": 900, "seed": 1,
         "batch": 16},
        {"tag": "2b_s0", "stage": "chain_edges", "epochs": 120, "seed": 0,
         "batch": 64, "rel": 1},
        {"tag": "2b_s1", "stage": "chain_edges", "epochs": 120, "seed": 1,
         "batch": 64, "rel": 1},
    ]
    hours = float(os.environ.get("WTOK_MAX_HOURS", "8.5"))
    each = hours / max(len(sweep), 1)
    for cfg in sweep:
        out = work / f"chain_{cfg['tag']}"
        cmd = [sys.executable, "-u", "-m", "cae_mesh_generator.wtok.staged",
               "--stage", cfg["stage"], "--use-spec",
               "--dataset", str(data), "--wtok", str(data),
               "--val-list", str(val), "--output-dir", str(out),
               "--train-parts", str(cfg.get("parts", 1000)), "--val-parts", "50",
               "--epochs", str(cfg["epochs"]),
               "--batch-size", str(cfg.get("batch", 16)),
               "--dim", str(cfg.get("dim", 256)),
               "--layers", str(cfg.get("layers", 8)),
               "--heads", "8", "--cfg-scale", "1.0",
               "--seed", str(cfg.get("seed", 0)),
               "--probe-every", "10", "--probe-parts", "16",
               "--max-hours", f"{each:.2f}"]
        if cfg.get("rel"):
            cmd += ["--rel-attn"]
        print("\n" + "=" * 70 + f"\n{cfg['tag']}: {' '.join(cmd[4:])}\n", flush=True)
        rc = subprocess.run(cmd, env=env, cwd=str(work)).returncode
        print(f"{cfg['tag']}: exit {rc}", flush=True)


if __name__ == "__main__":
    main()

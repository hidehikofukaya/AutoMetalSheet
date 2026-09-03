"""vast.ai launcher: run several training arms CONCURRENTLY on one GPU.

The model is small (11.8M params, ~2GB VRAM per arm) and the step is
data-loader bound, so a 12GB card runs 4-5 arms side by side; that is how
habit B1 (compare several independent runs) gets satisfied cheaply.

  python tools/vast/run_arms.py sweep.json --data /workspace/data --out /workspace/runs \
      --workers 8 --parallel 4

sweep.json is a list of arms in the kaggle_faces.py format:
  [{"tag": "2b_sib_s0", "stage": "face_ring", "epochs": 140, "seed": 0,
    "parts": 2677, "batch": 64, "rel": 1, "extra": ["--sib-ctx"], "hours": 10}]
Each arm writes <out>/<tag>/ (best.pt, last.pt, history.json) and <out>/<tag>.log.
Re-running with the same tags resumes (--resume) whatever has a last.pt.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time


def build_cmd(cfg, data, out, workers, val_list):
    od = out / cfg["tag"]
    cmd = [sys.executable, "-u", "-m", "cae_mesh_generator.wtok.staged",
           "--stage", cfg["stage"], "--use-spec",
           "--dataset", str(data), "--wtok", str(data),
           "--val-list", str(val_list), "--output-dir", str(od),
           "--train-parts", str(cfg.get("parts", 1000)), "--val-parts", "50",
           "--epochs", str(cfg["epochs"]),
           "--batch-size", str(cfg.get("batch", 16)),
           "--dim", str(cfg.get("dim", 256)), "--layers", str(cfg.get("layers", 8)),
           "--heads", "8", "--cfg-scale", str(cfg.get("cfg", 1.0)),
           "--seed", str(cfg.get("seed", 0)),
           "--probe-every", str(cfg.get("probe_every", 10)), "--probe-parts", "16",
           "--max-hours", str(cfg.get("hours", 8.0)),
           "--workers", str(workers)]
    if cfg.get("rel", 1):
        cmd.append("--rel-attn")
    if cfg.get("desc_noise"):
        cmd += ["--desc-noise", str(cfg["desc_noise"])]
    if (od / "last.pt").exists():
        cmd.append("--resume")
    cmd += list(cfg.get("extra", []))
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--data", required=True, help="dir with parts/, face_targets/, spec_vectors.json, val list")
    ap.add_argument("--out", required=True)
    ap.add_argument("--code", default="", help="repo root holding cae_mesh_generator/ (default: cwd)")
    ap.add_argument("--val-list", default="val_names_130.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--parallel", type=int, default=4, help="arms running at once")
    a = ap.parse_args()

    data, out = pathlib.Path(a.data), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    root = pathlib.Path(a.code or os.getcwd()).resolve()
    val_list = data / a.val_list
    env = dict(os.environ, PYTHONPATH=str(root),
               WTOK_SPEC_FILE=str(data / "spec_vectors.json"),
               WTOK_FACES=str(data / "face_targets"),
               WTOK_FACES_MODE=os.environ.get("WTOK_FACES_MODE", "all"),
               OMP_NUM_THREADS="2")
    sweep = json.load(open(a.sweep))
    pending = list(sweep)
    running = []
    while pending or running:
        while pending and len(running) < a.parallel:
            cfg = pending.pop(0)
            cmd = build_cmd(cfg, data, out, a.workers, val_list)
            log = open(out / f"{cfg['tag']}.log", "a")
            print(f"[{time.strftime('%H:%M')}] start {cfg['tag']}: {' '.join(cmd[4:])}", flush=True)
            running.append((cfg["tag"], subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        time.sleep(20)
        still = []
        for tag, proc, log in running:
            rc = proc.poll()
            if rc is None:
                still.append((tag, proc, log))
            else:
                log.close()
                print(f"[{time.strftime('%H:%M')}] done  {tag}: exit {rc}", flush=True)
        running = still
    print("ALL_ARMS_DONE", flush=True)


if __name__ == "__main__":
    main()

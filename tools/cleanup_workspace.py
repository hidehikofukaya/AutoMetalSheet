"""Reclaim disk space in AutoMetalSheet. Dry-run by default.

Tiers, ordered by how safe they are. Nothing here touches source code, JSON
metrics, histories, renders, or any dataset still in use.

  G1  stale git temp pack from an interrupted `git gc` (pure garbage)
  T1  smoke-test run directories (2-3 epoch throwaways)
  T2  checkpoints (*.pt) of abandoned experiment lines -- the wireflow
      point-cloud flow and the vertex-stage AR. history.json / metrics.json /
      renders are KEPT, so the experimental record survives intact.
  T3  non-best checkpoints (last.pt, best_zoff.pt ...) of retained models;
      each line keeps its best.pt
  T4  regenerable binaries (.ply/.pt/.h5) inside the biw_poc archive; its
      source and configs stay (and are also in git history)
  G2  `git gc` -- 66% of the object store is unreachable garbage. Reachable
      history (12 commits, both branches, incl. unpushed work) is untouched.

Usage:
  python tools/cleanup_workspace.py                 # report only
  python tools/cleanup_workspace.py --apply g1,t1   # delete selected tiers
  python tools/cleanup_workspace.py --apply all
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess

REPO = pathlib.Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
ARCHIVE = REPO / "_archive_pre_annotation_tool" / "biw_poc"

ABANDONED_PREFIXES = ("wireflow_", "wtok_ar_")
KEEP_BEST_ONLY = ("wtok_curve_v1", "wtok_curve_v2_traversal", "wtok_curve_synth_v1")


def mb(paths) -> float:
    return sum(p.stat().st_size for p in paths if p.is_file()) / 1e6


def collect() -> dict[str, tuple[list[pathlib.Path], str]]:
    plan: dict[str, tuple[list[pathlib.Path], str]] = {}

    tmp_packs = list((REPO / ".git" / "objects" / "pack").glob("tmp_pack_*"))
    plan["g1"] = (tmp_packs, "interrupted-gc temp pack (git regenerates as needed)")

    smoke = [d for d in RUNS.iterdir() if d.is_dir() and "smoke" in d.name] if RUNS.is_dir() else []
    plan["t1"] = (smoke, "smoke-test run directories (whole dirs)")

    aband_pt = []
    if RUNS.is_dir():
        for d in RUNS.iterdir():
            if (d.is_dir() and d.name.startswith(ABANDONED_PREFIXES)
                    and "smoke" not in d.name):
                aband_pt += list(d.rglob("*.pt"))
    plan["t2"] = (aband_pt, "checkpoints of abandoned lines (JSON/PNG kept)")

    non_best = []
    for name in KEEP_BEST_ONLY:
        d = RUNS / name
        if d.is_dir():
            non_best += [f for f in d.glob("*.pt") if f.name != "best.pt"]
    plan["t3"] = (non_best, "non-best checkpoints of retained models")

    arch = ([f for f in ARCHIVE.rglob("*")
             if f.is_file() and f.suffix in (".ply", ".pt", ".h5")]
            if ARCHIVE.is_dir() else [])
    plan["t4"] = (arch, "regenerable binaries in the biw_poc archive")
    return plan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", default="",
                    help="comma list of tiers (g1,t1,t2,t3,t4,g2) or 'all'")
    args = ap.parse_args()
    apply = ([t.strip() for t in args.apply.split(",") if t.strip()]
             if args.apply else [])
    if "all" in apply:
        apply = ["g1", "t1", "t2", "t3", "t4", "g2"]

    plan = collect()
    total = 0.0
    print(f"{'tier':5s} {'MB':>9s} {'items':>6s}  description")
    for tier, (paths, desc) in plan.items():
        size = mb([p for path in paths for p in
                   ([path] if path.is_file() else path.rglob("*"))])
        total += size
        print(f"{tier:5s} {size:9.0f} {len(paths):6d}  {desc}")
    print(f"{'':5s} {total:9.0f}         (git gc adds ~48 GB more, tier g2)")

    if not apply:
        print("\nDry run. Re-run with --apply g1,t1,... to delete.")
        return

    for tier in [t for t in apply if t != "g2"]:
        paths, desc = plan[tier]
        freed = mb([p for path in paths for p in
                    ([path] if path.is_file() else path.rglob("*"))])
        for path in paths:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                try:
                    path.unlink()
                except OSError as exc:      # read-only temp packs on Windows
                    path.chmod(0o600)
                    try:
                        path.unlink()
                    except OSError:
                        print(f"  [warn] could not delete {path.name}: {exc}")
        print(f"[{tier}] freed ~{freed:.0f} MB ({desc})")

    if "g2" in apply:
        print("[g2] running git gc (keeps all reachable history)...", flush=True)
        subprocess.run(["git", "-C", str(REPO), "gc", "--prune=now"], check=False)
        out = subprocess.run(["git", "-C", str(REPO), "count-objects", "-vH"],
                             capture_output=True, text=True).stdout
        print(out.strip())


if __name__ == "__main__":
    main()

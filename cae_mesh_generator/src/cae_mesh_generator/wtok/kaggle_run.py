"""Kaggle launcher: discovers the mounted datasets, trains, then evaluates.

Lives inside the code dataset so the notebook cell stays three lines -- code
changes ship by re-uploading the code zip, never by re-pasting the cell.

Settings come from environment variables so the notebook can override them:
  WTOK_EPOCHS      (default 150)
  WTOK_MAX_HOURS   (default 8.5)  clean stop before Kaggle's 12h kill
  WTOK_RESUME_DIR  (default "")   a previous session's Output, mounted as input
  WTOK_WORK        (default /kaggle/working/run_curve2)
  WTOK_EVAL_PARTS  (default 40)
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

INPUT = pathlib.Path(os.environ.get("WTOK_INPUT", "/kaggle/input"))


def find(pred, what: str) -> pathlib.Path:
    for d in sorted(INPUT.rglob("*")):
        if d.is_dir() and pred(d):
            return d
    print(f"!! could not locate {what}.\n   Contents of {INPUT}:")
    for q in sorted(INPUT.rglob("*"))[:60]:
        print("   ", q)
    raise SystemExit(1)


def check_gpu() -> None:
    """Fail fast (and legibly) if the assigned GPU predates this torch build.

    Kaggle's API can only ask for `enable_gpu`, not a GPU type, and the legacy
    allocation is a Tesla P100 (sm_60) which current Kaggle torch builds no
    longer support (sm_70+). The interactive UI can pick T4 x2, which works.
    """
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        print("!! no CUDA device visible", flush=True)
        return
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    supported = torch.cuda.get_arch_list()
    print(f"GPU: {name} (sm_{major}{minor}) | torch {torch.__version__} "
          f"supports {' '.join(supported)}", flush=True)
    if f"sm_{major}{minor}" not in supported:
        raise SystemExit(
            f"!! {name} (sm_{major}{minor}) is not supported by this torch build. "
            "Set the notebook accelerator to 'GPU T4 x2' in the Kaggle UI "
            "(Settings > Accelerator) and re-run; the API cannot choose the GPU type.")


def run_plan_g(module, data, val_list, work, epochs, max_hours, batch,
               eval_parts, env) -> None:
    """Set-based arms: each is topo -> geo -> eval. Both arms run in one session
    so the comparison shares a GPU, a dataset and a wall-clock budget."""
    arms = [a for a in os.environ.get("WTOK_ARMS", "bcore,gagf").split(",") if a]
    budget = float(max_hours) / (2 * len(arms))    # topo and geo per arm
    results = {}
    for arm in arms:
        out = f"{work}_{arm}"
        for stage in ("topo", "geo"):
            cmd = [sys.executable, "-m", module, "--stage", stage, "--arm", arm,
                   "--dataset", str(data), "--val-list", str(val_list),
                   "--output-dir", out, "--epochs", str(epochs),
                   "--batch-size", str(batch), "--max-hours", f"{budget:.3f}",
                   "--device", "cuda"]
            print("\n" + " ".join(cmd), flush=True)
            subprocess.run(cmd, env=env, check=True)
        cmd = [sys.executable, "-m", module, "--stage", "eval", "--arm", arm,
               "--dataset", str(data), "--val-list", str(val_list),
               "--output-dir", out, "--eval-parts", str(eval_parts),
               "--device", "cuda"]
        print("\n" + " ".join(cmd), flush=True)
        subprocess.run(cmd, env=env, check=True)
        ev = pathlib.Path(out) / "eval.json"
        if ev.exists():
            results[arm] = json.loads(ev.read_text())["summary"]
    print("\n==== arm comparison ====")
    print(json.dumps(results, indent=1), flush=True)
    pathlib.Path(f"{work}_comparison.json").write_text(json.dumps(results, indent=1))


def main() -> None:
    check_gpu()
    code = find(lambda d: (d / "cae_mesh_generator" / "wtok" / "curve2.py").exists(),
                "the code dataset (expects cae_mesh_generator/wtok/curve2.py)")
    data = find(lambda d: (d / "parts").is_dir() and any((d / "parts").glob("*.json")),
                "the data dataset (expects parts/*.json)")
    val_list = next(iter(data.rglob("val_names_100.json")), None)
    if val_list is None:
        raise SystemExit("val_names_100.json not found in the data dataset")

    module = os.environ.get("WTOK_MODULE", "cae_mesh_generator.wtok.curve3")
    arch = os.environ.get("WTOK_ARCH", "v3" if module.endswith("curve3") else "v2")
    batch = os.environ.get("WTOK_BATCH", "16")
    work = os.environ.get("WTOK_WORK", "/kaggle/working/run_curve2")
    epochs = os.environ.get("WTOK_EPOCHS", "150")
    max_hours = os.environ.get("WTOK_MAX_HOURS", "8.5")
    resume_dir = os.environ.get("WTOK_RESUME_DIR", "").strip()
    eval_parts = os.environ.get("WTOK_EVAL_PARTS", "40")

    n_parts = len(list((data / "parts").glob("*.json")))
    print(f"CODE     = {code}")
    print(f"DATA     = {data}  ({n_parts} parts)")
    print(f"val list = {val_list}")
    print(f"work     = {work} | epochs={epochs} max_hours={max_hours} "
          f"resume={resume_dir or 'none'}", flush=True)

    pathlib.Path(work).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(code))

    if module.endswith("plan_g"):
        run_plan_g(module, data, val_list, work, epochs, max_hours, batch,
                   eval_parts, env)
        return

    train = [sys.executable, "-m", module,
             "--dataset", str(data), "--val-list", str(val_list),
             "--output-dir", work, "--epochs", epochs, "--stage2-after", "30",
             "--val-every", "5", "--sample-every", "25",
             "--max-hours", max_hours, "--device", "cuda"]
    if module.endswith("curve3"):
        train += ["--batch-size", batch]
    if resume_dir:
        train += ["--resume-dir", resume_dir]
    print(" ".join(train), flush=True)
    subprocess.run(train, env=env, check=True)

    evaluate = [sys.executable, "-m", "cae_mesh_generator.wtok.evaluate_curve2",
                "--dataset", str(data), "--checkpoint", f"{work}/best.pt",
                "--val-list", str(val_list), "--output-dir", f"{work}_eval",
                "--arch", arch, "--max-parts", eval_parts, "--device", "cuda"]
    print(" ".join(evaluate), flush=True)
    subprocess.run(evaluate, env=env, check=True)
    print("done -- press Save Version > Save & Run All (Commit) to keep the Output")


if __name__ == "__main__":
    main()

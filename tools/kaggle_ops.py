"""Drive Kaggle from the command line: update the code dataset, push the
training kernel, poll it, and pull the output.

  python tools/kaggle_ops.py update-code        # new version of wtok-code
  python tools/kaggle_ops.py update-data        # new version of wtok-synth
  python tools/kaggle_ops.py push --epochs 150  # fresh run (restarts training)
  python tools/kaggle_ops.py push --epochs 650 --resume-from <output-dataset>
  python tools/kaggle_ops.py status
  python tools/kaggle_ops.py pull               # download the finished output

The kernel runs as a batch job, so its output is committed automatically -- no
"Save Version" click, and no dataset-version pinning (a pushed kernel always
resolves dataset_sources to the newest version).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import zipfile

REPO = pathlib.Path(__file__).resolve().parent.parent
BUNDLE = REPO / "kaggle_bundle"
USER = "hidehikofukaya"
KERNEL = f"{USER}/autometalsheet-v0-1"
CODE_DS = f"{USER}/wtok-code"
DATA_DS = f"{USER}/wtok-synth"
STAGE = BUNDLE / "_kernel"          # working dir for kernel push
DS_STAGE = BUNDLE / "_dataset"      # working dir for dataset versions


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run([sys.executable, "-m", "kaggle"] + cmd, **kw)


def unzip_to(zip_path: pathlib.Path, dest: pathlib.Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)


def update_dataset(slug: str, zip_name: str, notes: str) -> None:
    """Push a new version of a dataset from the bundle zip.

    Folder handling is fiddly: the CLI skips sub-folders unless --dir-mode zip,
    and Kaggle then auto-extracts <folder>.zip so the folder's *contents* land
    at the dataset root. So we stage the payload inside a throwaway wrapper
    folder -- after extraction the tree we actually want sits at the root.
    """
    if DS_STAGE.exists():
        shutil.rmtree(DS_STAGE)
    payload = DS_STAGE / "payload"
    payload.mkdir(parents=True)
    with zipfile.ZipFile(BUNDLE / zip_name) as z:
        z.extractall(payload)
    meta = {"title": slug.split("/")[1], "id": slug,
            "licenses": [{"name": "CC0-1.0"}]}
    (DS_STAGE / "dataset-metadata.json").write_text(json.dumps(meta, indent=1),
                                                    encoding="utf-8")
    run(["datasets", "version", "-p", str(DS_STAGE), "-m", notes,
         "--dir-mode", "zip"], check=True)


KERNEL_SCRIPT = '''"""Pushed by tools/kaggle_ops.py -- edits belong in the repo, not here."""
import glob, os, subprocess, sys

os.environ["WTOK_EPOCHS"] = "{epochs}"
os.environ["WTOK_MAX_HOURS"] = "{max_hours}"
os.environ["WTOK_RESUME_DIR"] = "{resume_dir}"
os.environ["WTOK_EVAL_PARTS"] = "{eval_parts}"
os.environ["WTOK_MODULE"] = "{module}"
os.environ["WTOK_BATCH"] = "{batch}"
os.environ["WTOK_ARMS"] = "{arms}"

hits = glob.glob("/kaggle/input/**/kaggle_run.py", recursive=True)
if not hits:
    for p in sorted(glob.glob("/kaggle/input/*")):
        print(p)
        for q in sorted(glob.glob(p + "/*"))[:8]:
            print("   ", q)
    raise SystemExit("kaggle_run.py not found in the attached datasets")
print("launcher:", hits[0], flush=True)
subprocess.run([sys.executable, hits[0]], check=True)
'''


def push(epochs: int, max_hours: float, resume_from: str, eval_parts: int,
         module: str = "cae_mesh_generator.wtok.curve3", batch: int = 16,
         arms: str = "bcore,gagf") -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    sources = [CODE_DS, DATA_DS]
    resume_dir = ""
    if resume_from:
        sources.append(resume_from)
        # the mount path Kaggle uses for a dataset source
        resume_dir = f"/kaggle/input/{resume_from.split('/')[1]}"
    (STAGE / "script.py").write_text(
        KERNEL_SCRIPT.format(epochs=epochs, max_hours=max_hours,
                             resume_dir=resume_dir, eval_parts=eval_parts,
                             module=module, batch=batch, arms=arms),
        encoding="utf-8")
    meta = {
        "id": KERNEL,
        "title": "AutoMetalSheet_v0.1",
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        # GPU type MUST be pinned: without it Kaggle assigns a Tesla P100
        # (sm_60), which current Kaggle torch builds no longer support, and the
        # push also overrides whatever the web UI had selected.
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": False,
        "dataset_sources": sources,
        "competition_sources": [],
        "kernel_sources": [],
    }
    (STAGE / "kernel-metadata.json").write_text(json.dumps(meta, indent=1),
                                                encoding="utf-8")
    print(f"epochs={epochs} max_hours={max_hours} "
          f"resume={resume_dir or 'none'} sources={sources}")
    run(["kernels", "push", "-p", str(STAGE)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("update-code")
    sub.add_parser("update-data")
    p = sub.add_parser("push")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--max-hours", type=float, default=8.5)
    p.add_argument("--resume-from", default="", help="dataset slug of a prior output")
    p.add_argument("--eval-parts", type=int, default=40)
    p.add_argument("--module", default="cae_mesh_generator.wtok.curve3")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--arms", default="bcore,gagf", help="plan_g arms to run")
    sub.add_parser("status")
    q = sub.add_parser("pull")
    q.add_argument("--out", default=str(REPO / "runs" / "kaggle_output"))
    args = ap.parse_args()

    if args.cmd == "update-code":
        update_dataset(CODE_DS, "wtok_code.zip", "code update")
    elif args.cmd == "update-data":
        update_dataset(DATA_DS, "wtok_synth_data.zip", "data update")
    elif args.cmd == "push":
        push(args.epochs, args.max_hours, args.resume_from, args.eval_parts,
             args.module, args.batch, args.arms)
    elif args.cmd == "status":
        run(["kernels", "status", KERNEL], check=False)
    elif args.cmd == "pull":
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        run(["kernels", "output", KERNEL, "-p", str(out)], check=False)
        print("output ->", out)


if __name__ == "__main__":
    main()

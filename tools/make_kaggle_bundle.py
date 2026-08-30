"""Build the two Kaggle upload zips (code / data) + the notebook script.

  code zip : the OCC-free subset of cae_mesh_generator needed for training and
             evaluation. Small, so re-upload it whenever the code changes.
  data zip : runs/wtok_synth/parts/*.json + the val list. Re-upload only when
             new synthetic chunks are converted locally.

Usage (from the repo root):
  python tools/make_kaggle_bundle.py
  python tools/make_kaggle_bundle.py --code-only        # after a code edit
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import zipfile

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "cae_mesh_generator" / "src" / "cae_mesh_generator"
OUT = REPO / "kaggle_bundle"

# OCC-free modules only. convert/build_dataset/build_synthetic need pythonocc
# and are local-only (STEP -> wtok conversion happens on the workstation).
CODE_FILES = [
    "__init__.py",
    "wtok/__init__.py",
    "wtok/constants.py",
    "wtok/codec.py",
    "wtok/codec_curve.py",
    "wtok/dataset_ar.py",
    "wtok/dataset_curve.py",
    "wtok/model_ar.py",
    "wtok/model_curve.py",
    "wtok/train_ar.py",
    "wtok/train_curve.py",
    "wtok/curve2.py",
    "wtok/curve3.py",
    "wtok/plan_g.py",
    "wtok/evaluate_curve.py",
    "wtok/evaluate_curve2.py",
    "wtok/kaggle_run.py",
    "wtok/kaggle_frame.py",
    # the current line of work: parametric frames, the staged trainer and the
    # readers they need. meshgen brings the fastener frame and the guard disc.
    "wtok/frame.py",
    "wtok/staged.py",
    "wtok/meshgen.py",
    "wtok/mesh_extract.py",
    "wtok/connect.py",
    "wtok/surface.py",
    "wtok/sidecar.py",
    "wtok/bendlines.py",
    "wtok/judge.py",
    "wtok/rational.py",
    "wtok/rational_eval.py",
    "wtok/tokens.py",
    "wtok/deltatok.py",
    "wtok/feature.py",
    "wtok/kernel.py",
    "wtok/ridge.py",
    "wtok/validity.py",
]

NOTEBOOK = '''# Kaggle cell -- clear the cell completely (Ctrl+A, Delete) before pasting.
import glob, os, subprocess, sys
os.environ["WTOK_EPOCHS"] = "150"
os.environ["WTOK_MAX_HOURS"] = "8.5"
os.environ["WTOK_RESUME_DIR"] = ""   # set to a previous Output dataset to continue
LAUNCH = os.environ.get("WTOK_LAUNCHER", "kaggle_frame.py")
hits = glob.glob(f"/kaggle/input/**/{LAUNCH}", recursive=True)
if not hits:
    print("kaggle_run.py NOT FOUND. /kaggle/input currently holds:")
    for p in sorted(glob.glob("/kaggle/input/*")):
        print("  ", p)
        for q in sorted(glob.glob(p + "/*"))[:6]:
            print("      ", q)
    raise SystemExit("Attach the wtok-code dataset, or refresh it to the newest "
                     "version (Kaggle pins the version an input was added at).")
print("launcher:", hits[0], flush=True)
subprocess.run([sys.executable, hits[0]], check=True)
'''


def zip_dir(zpath: pathlib.Path, files: list[tuple[pathlib.Path, str]]) -> None:
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in files:
            z.write(src, arc)
    print(f"[ok] {zpath.name}  ({zpath.stat().st_size/1e6:.1f} MB, {len(files)} files)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO / "runs" / "wtok_synth"))
    ap.add_argument("--val-list", default=str(REPO / "runs" / "wtok_curve_synth_v1"
                                              / "val_names_100.json"))
    ap.add_argument("--code-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    missing = [f for f in CODE_FILES if not (SRC / f).exists()]
    if missing:
        raise SystemExit(f"missing source files: {missing}")
    zip_dir(OUT / "wtok_code.zip",
            [(SRC / f, f"cae_mesh_generator/{f}") for f in CODE_FILES])
    (OUT / "kaggle_notebook.py").write_text(NOTEBOOK, encoding="utf-8")
    print(f"[ok] kaggle_notebook.py")

    if args.code_only:
        return
    data = pathlib.Path(args.data_dir)
    parts = sorted((data / "parts").glob("*.json"))
    if not parts:
        raise SystemExit(f"no parts under {data}/parts")
    files = [(p, f"parts/{p.name}") for p in parts]
    vl = pathlib.Path(args.val_list)
    if vl.exists():
        files.append((vl, "val_names_100.json"))
    # the design spec lives in the PartMaker tree, which Kaggle cannot see, so
    # it travels as one exported table (sidecar.load_spec falls back to it)
    sv = data / "spec_vectors.json"
    if sv.exists():
        files.append((sv, "spec_vectors.json"))
    zip_dir(OUT / "wtok_synth_data.zip", files)
    print(f"\nUpload {OUT}:\n"
          f"  wtok_code.zip       -> Kaggle Dataset 'wtok-code'\n"
          f"  wtok_synth_data.zip -> Kaggle Dataset 'wtok-synth'\n"
          f"  kaggle_notebook.py  -> paste into a GPU notebook cell")


if __name__ == "__main__":
    main()

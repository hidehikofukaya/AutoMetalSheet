"""Retire the CATIA-derived corpus (batch02 / prod01 c0* / prod02 p2c* / flange01 f0*) from the
active dataset into runs/catia_archive/ with the same layout, so load_curve_parts() no longer
sees it. Nothing is deleted; `--restore` moves everything back. Dry-run by default.

Why (2026-09-06, user decision): the CATIA teacher carries sub-thickness edges (8.9%) and
0.008 mm garbage edges, its PartMaker sources are gone (spec only survives as a table), and
training already excludes it (--train-filter ^o1). PartMaker's OCCT families grow by 600/batch.

  python tools/retire_catia.py            # report
  python tools/retire_catia.py --apply    # move
  python tools/retire_catia.py --restore  # move back
"""
import argparse, pathlib, re, shutil

REPO = pathlib.Path(__file__).resolve().parent.parent
TAG = re.compile(r"^(batch02|c0\d|p2c\d\d|f0\d)__")
SETS = [("runs/wtok_synth_g1/parts", "*.json"), ("runs/wtok_synth_g1/face_targets", "*.npz"),
        ("runs/wtok_synth_g1/face_targets_curved", "*.npz"), ("runs/wtok_synth_g1/chain_targets", "*.npz"),
        ("runs/wtok_synth_v45/wireframes", "*.json"), ("runs/mesh_synth/parts", "*.npz")]
ARCH = REPO / "runs" / "catia_archive"


def main():
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group()
    g.add_argument("--apply", action="store_true"); g.add_argument("--restore", action="store_true"); a = ap.parse_args()
    total = 0
    for rel, pat in SETS:
        src, dst = (REPO / rel, ARCH / rel) if not a.restore else (ARCH / rel, REPO / rel)
        if not src.exists():
            continue
        files = [f for f in src.glob(pat) if TAG.match(f.name)]
        total += len(files)
        print(f"{rel}: {len(files)} CATIA files")
        if a.apply or a.restore:
            dst.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.move(str(f), str(dst / f.name))
    print(("moved" if (a.apply or a.restore) else "would move"), total, "files", "->" if not a.restore else "<-", ARCH)


if __name__ == "__main__":
    main()

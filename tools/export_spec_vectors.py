"""Export the design-spec vectors of every converted part to one JSON table.

Cloud machines (Kaggle, vast.ai) cannot see the PartMaker tree, so sidecar.load_spec
falls back to this table (WTOK_SPEC_FILE). Re-run after every new chunk is converted.

  python tools/export_spec_vectors.py --wtok runs/wtok_synth_g1
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "cae_mesh_generator" / "src"))
# read PartMaker directly, never an older table
os.environ["WTOK_SPEC_FILE"] = str(REPO / "runs" / "_no_such_spec_table.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wtok", default="runs/wtok_synth_g1")
    a = ap.parse_args()
    from cae_mesh_generator.wtok.dataset_curve import load_curve_parts
    from cae_mesh_generator.wtok.sidecar import SPEC_KEYS, load_spec

    parts = load_curve_parts(pathlib.Path(a.wtok))
    table, missing = {}, []
    for p in parts:
        s = load_spec(p)
        if s is None:
            missing.append(p.name)
        else:
            table[p.name] = [float(v) for v in s]
    out = pathlib.Path(a.wtok) / "spec_vectors.json"
    # MERGE with the existing table: the CATIA families' PartMaker sources were deleted
    # on 2026-09-04, so their spec can only come from the table already on disk
    if out.exists():
        old = json.loads(out.read_text()).get("spec", {})
        kept = {k: v for k, v in old.items() if k not in table}
        table = {**kept, **table}
        print(f"kept {len(kept)} entries from the existing table")
    out.write_text(json.dumps({"keys": list(SPEC_KEYS), "spec": table}))
    print(f"{out}: {len(table)} parts with spec, {len(missing)} without"
          + (f" (e.g. {missing[:3]})" if missing else ""))


if __name__ == "__main__":
    main()

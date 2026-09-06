"""Convert PartMaker synthetic parts (1,300 mid STPs) into the wtok sparse
representation for E2E curve-major AR training.

Pipeline per part: v4.4 typed-wireframe extraction (B-Rep, wireframe_app)
-> convert_part (chains -> LINE/ARC/CIRCLE/CIRCLE_C with joint matching)
-> quantize -> roundtrip gates. FIX vertices come from the synthetic
joints.json (2 fastening points per part).

Usage:
  python -m cae_mesh_generator.wtok.build_synthetic --output-dir ../runs/wtok_synth
Resumable: parts with an existing output JSON are skipped.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import pathlib
import sys

import numpy as np

WF_APP = "C:/Users/hide2/IdeaBox/fill_volume/wireframe_app"
if WF_APP not in sys.path:
    sys.path.insert(0, WF_APP)
import extract as wf_extract  # noqa: E402  (wireframe_app v4.4)

from .convert import convert_part  # noqa: E402
from .codec import quantize_W, realize_points, roundtrip_ok  # noqa: E402
from .codec_curve import n_tokens_curve, roundtrip_curve_ok, sigma_curve  # noqa: E402
from .build_dataset import chamfer, source_scope_points  # noqa: E402

SYNTH_BASE = pathlib.Path("C:/Users/hide2/IdeaBox/PartMaker/synthetic_parts")
# (group_dir, tag). prod02/chunk_03 is still being generated -- do not add
# it until the user confirms it is complete.
# CATIA-era groups (batch02, prod01 c0*, prod02 p2c*, flange01 f0*) were retired and deleted on
# 2026-09-06 (user decision): sub-thickness edges, garbage edges, sources gone. OCCT families only.
GROUPS = ([("occt11/chunk_01", "o11c01"), ("occt12/chunk_01", "o12c01"),
           ("occt13/chunk_01", "o13c01"), ("occt15/chunk_01", "o15c01"), ("occt16/chunk_01", "o16c01"),
           ("occt17/chunk_01", "o17c01"), ("occt18/chunk_01", "o18c01"), ("occt19/chunk_01", "o19c01"),
           ("occt20/chunk_01", "o20c01")])


def load_synth_joints(group_dir: pathlib.Path) -> dict[str, list[dict]]:
    path = group_dir / "annotations" / "joints.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    per_part: dict[str, list[dict]] = {}
    for j in data.get("joints", []):
        axis = j.get("axis") or {}
        direction = axis.get("direction_xyz") or [0.0, 0.0, 1.0]
        for pp in j.get("per_part", []):
            xyz = pp.get("hole_center_xyz") or pp.get("contact_xyz")
            if xyz is None:
                continue
            size = pp.get("hole_diameter_mm") or axis.get("length_mm") or 0.0
            per_part.setdefault(str(pp["part_id"]), []).append(
                {"xyz": xyz, "dir": direction, "type": j.get("type", "other"),
                 "size": size})
    return per_part


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--fidelity-every", type=int, default=25,
                    help="run realize-vs-source chamfer on every Nth part")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = pathlib.Path(args.output_dir)
    (out / "wireframes").mkdir(parents=True, exist_ok=True)
    (out / "parts").mkdir(exist_ok=True)

    report = []
    done = 0
    for group, tag in GROUPS:
        gdir = SYNTH_BASE / group
        joints_map = load_synth_joints(gdir)
        for stp in sorted((gdir / "mid").glob("*_mid.stp")):
            part_id = stp.stem[: -len("_mid")]
            uid = f"{tag}__{part_id}"
            out_part = out / "parts" / f"{uid}.json"
            if out_part.exists():
                done += 1
                continue
            if args.limit and done >= args.limit:
                break
            try:
                wf_json = out / "wireframes" / f"{uid}.json"
                if not wf_json.exists():
                    data = wf_extract.extract_wireframe(stp)
                    wf_json.write_text(json.dumps(data), encoding="utf-8")
                joints = joints_map.get(part_id, [])
                W, cstats = convert_part(wf_json, joints)
                Q = quantize_W(W)
                rt = roundtrip_ok(Q) and roundtrip_curve_ok(Q)
                row = {"part": uid, "roundtrip_ok": bool(rt),
                       "n_vertices": len(Q["vertices"]),
                       "n_fix": sum(1 for v in Q["vertices"] if v["T"] == "FIX"),
                       "n_edges": len(Q["edges"]),
                       "n_tokens_curve": n_tokens_curve(sigma_curve(Q)), **cstats}
                if done % args.fidelity_every == 0:
                    cd, p95 = chamfer(realize_points(Q), source_scope_points(wf_json))
                    row["fidelity_chamfer_mm"] = cd
                    row["fidelity_p95_mm"] = p95
                out_part.write_text(json.dumps({"Q": Q}, default=str), encoding="utf-8")
                report.append(row)
                done += 1
                if done % 50 == 0:
                    fid = [r.get("fidelity_chamfer_mm") for r in report
                           if r.get("fidelity_chamfer_mm") is not None]
                    print(f"[{done}] rt_fail={sum(not r['roundtrip_ok'] for r in report)} "
                          f"fid_mean={np.mean(fid):.2f}mm" if fid else f"[{done}]",
                          flush=True)
            except Exception as exc:
                report.append({"part": uid, "error": str(exc)[:150]})
                print(f"[FAIL] {uid}: {str(exc)[:120]}", flush=True)
        if args.limit and done >= args.limit:
            break

    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    good = [r for r in report if "error" not in r]
    print(f"--- converted {len(good)}, failed {len(report) - len(good)}")
    if good:
        print(f"roundtrip: {sum(r['roundtrip_ok'] for r in good)}/{len(good)}")
        for key in ("n_vertices", "n_edges", "n_tokens_curve", "n_fix"):
            vals = [r[key] for r in good]
            print(f"{key}: median {int(np.median(vals))} p90 {int(np.percentile(vals, 90))} "
                  f"max {max(vals)}")
        fid = [r["fidelity_chamfer_mm"] for r in good if "fidelity_chamfer_mm" in r]
        if fid:
            print(f"fidelity ({len(fid)} sampled): mean {np.mean(fid):.2f}mm "
                  f"worst {max(fid):.2f}mm")
        print("circle_c:", sum(r.get("n_circle_c", 0) for r in good),
              "circle:", sum(r.get("n_circle", 0) for r in good))


if __name__ == "__main__":
    main()

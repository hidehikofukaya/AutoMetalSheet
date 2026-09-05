"""Convert PartMaker design variants (<chunk>/variants/<id>__side±1_mid.stp) into wtok parts.

Writes runs/<wtok>/variants/<tag>__<id>__side±1.json ({"Q": ...}) through the SAME pipeline as
build_synthetic (v4.5 extraction -> convert_part with the base part's joints -> quantize), so a
variant is a valid alternative TARGET for the base part <tag>__<id> (KB roadmap 6.4).
Resumable.  python tools/ingest_variants.py --wtok runs/wtok_synth_g1
"""
from __future__ import annotations
import argparse, json, os, pathlib, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "cae_mesh_generator" / "src"))
from cae_mesh_generator.wtok.build_synthetic import GROUPS, SYNTH_BASE, load_synth_joints, wf_extract  # noqa: E402
from cae_mesh_generator.wtok.convert import convert_part  # noqa: E402
from cae_mesh_generator.wtok.codec import quantize_W, roundtrip_ok  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--wtok", default="runs/wtok_synth_g1"); a = ap.parse_args()
    out = pathlib.Path(a.wtok) / "variants"; out.mkdir(exist_ok=True)
    wfd = pathlib.Path(a.wtok) / "variants_wireframes"; wfd.mkdir(exist_ok=True)
    done = skipped = failed = 0
    for group, tag in GROUPS:
        vdir = SYNTH_BASE / group / "variants"
        if not vdir.exists():
            continue
        joints = load_synth_joints(SYNTH_BASE / group)
        for stp in sorted(vdir.glob("*_mid.stp")):
            vid = stp.stem[: -len("_mid")]             # SYN_..._0001__side-1
            base_id = vid.split("__")[0]
            uid = f"{tag}__{vid}"
            outp = out / f"{uid}.json"
            if outp.exists():
                skipped += 1
                continue
            try:
                wf = wfd / f"{uid}.json"
                if not wf.exists():
                    wf.write_text(json.dumps(wf_extract.extract_wireframe(stp)), encoding="utf-8")
                W, _ = convert_part(wf, joints.get(base_id, []))
                Q = quantize_W(W)
                if not roundtrip_ok(Q):
                    raise ValueError("roundtrip failed")
                outp.write_text(json.dumps({"Q": Q}, default=str), encoding="utf-8")
                done += 1
            except Exception as exc:
                failed += 1
                print(f"[FAIL] {uid}: {str(exc)[:100]}", flush=True)
    print(f"variants converted {done}, already {skipped}, failed {failed} -> {out}")


if __name__ == "__main__":
    main()

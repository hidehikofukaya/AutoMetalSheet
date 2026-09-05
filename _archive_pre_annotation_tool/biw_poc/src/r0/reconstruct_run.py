"""Run the profile-driven Stage C reconstruction and immediately create an R0 audit bundle."""

from __future__ import annotations

import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
MODEL_DIR = HERE.parent / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from audit_run import audit, load_profile
import reconstruct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--data", type=pathlib.Path, required=True)
    parser.add_argument("--reference", type=pathlib.Path, required=True)
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--thickness-mm", type=float)
    parser.add_argument("--part-id")
    parser.add_argument("--part-family")
    args = parser.parse_args()

    profile_path = args.profile.resolve()
    profile = load_profile(profile_path)
    reconstruction = profile["reconstruction"]
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    out_stem = artifacts_dir / "candidate"

    try:
        outputs = reconstruct.process(
            args.checkpoint.resolve(),
            args.data.resolve(),
            out_stem,
            thickness_mm=args.thickness_mm,
            grid_res=int(reconstruction["grid_res"]),
            band_factor=float(reconstruction["band_factor"]),
            n_proj_iters=int(reconstruction["n_proj_iters"]),
            nbr_sz=int(reconstruction["nbr_sz"]),
            refine_rounds=int(reconstruction["refine_rounds"]),
            save_intermediate=True,
            conf_threshold=float(reconstruction["conf_threshold"]),
            input_dist_threshold_mm=float(
                reconstruction["input_dist_threshold_mm"]
            ),
            prune_dist_threshold_mm=float(
                reconstruction["prune_dist_threshold_mm"]
            ),
        )
    except reconstruct.InsufficientEvidenceError as exc:
        failure = run_dir / "INSUFFICIENT_EVIDENCE.txt"
        failure.write_text(str(exc) + "\n", encoding="utf-8")
        raise

    stages: list[tuple[str, pathlib.Path]] = []
    for stage_path_raw in outputs["stage_plys"]:
        stage_path = pathlib.Path(stage_path_raw)
        marker = "_stage_"
        label = stage_path.stem.split(marker, 1)[1].rsplit("_midsurface", 1)[0]
        stages.append((label, stage_path))
    if not stages:
        stages.append(("candidate", pathlib.Path(outputs["midsurface_ply"])))

    manifest = audit(
        profile_path,
        args.reference.resolve(),
        stages,
        run_dir / "audit",
        part_id=args.part_id,
        part_family=args.part_family,
    )
    print(manifest)


if __name__ == "__main__":
    main()

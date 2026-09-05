"""One command for a decision point: score a generator run and say pass/hold/fail.

Every judgement in the overnight plan goes through this so the numbers are
always produced the same way -- 20 parts, guidance 2.0, last.pt, area-weighted
sampling on both sides of every comparison.

  python tools/judge_run.py --run meshgen_even512 --tag "T+1.8 slice 1"
  python tools/judge_run.py --run meshgen_even512 --ridge ridge_even512
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "cae_mesh_generator" / "src"))

from cae_mesh_generator.wtok.codec import bin_center, realize_points   # noqa: E402
from cae_mesh_generator.wtok.connect import densify, trace             # noqa: E402
from cae_mesh_generator.wtok.dataset_curve import load_curve_parts     # noqa: E402
from cae_mesh_generator.wtok.evaluate_curve2 import class_points, one_way  # noqa: E402
from cae_mesh_generator.wtok.meshgen import generate, load_model       # noqa: E402
from cae_mesh_generator.wtok.surface import extract_features           # noqa: E402
from cae_mesh_generator.wtok.train_ar import chamfer_mm                # noqa: E402
from cae_mesh_generator.wtok.train_curve import realized_q             # noqa: E402

# the 256-point random model, measured the same way
BASE = {"wire": 12.50, "outline": 5.73, "bend": 6.62}
HOLD = 14.0


def ridge_curves(rm, xyz, nrm, device):
    """Returns (curves per class, fraction of points each class kept).

    The kept fraction has to be reported: selecting on |displacement| alone once
    kept 100% of points for both classes and projected the whole cloud twice,
    which coverage rewards -- carpeting a part covers every curve on it.
    """
    from cae_mesh_generator.wtok.ridge import CLASSES
    c = xyz.mean(0)
    s = float(np.linalg.norm(xyz - c, axis=1).max())
    x = torch.from_numpy(np.concatenate([(xyz - c) / s, nrm], 1).astype(np.float32))
    with torch.no_grad():
        pr, lg = rm(x[None].to(device))
    pr = pr[0].cpu().numpy()
    conf = torch.sigmoid(lg[0]).cpu().numpy()
    out, kept = [], []
    for ci in range(len(CLASSES)):
        disp = pr[:, ci] * s
        keep = conf[:, ci] > 0.5
        kept.append(float(keep.mean()))
        if keep.sum() < 4:
            out.append(np.zeros((0, 3)))
            continue
        proj = xyz[keep] + disp[keep]
        kk = min(13, len(proj))
        dd, nb = cKDTree(proj).query(proj, k=kk)
        pi = np.repeat(np.arange(len(proj)), kk - 1)
        pj = nb[:, 1:].reshape(-1)
        out.append(densify(trace(proj, np.ones(len(proj), int),
                                 (dd[:, 1:].reshape(-1) < s * 0.06).astype(float),
                                 np.stack([pi, pj], 1))))
    return out, kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="last.pt")
    ap.add_argument("--ridge", default="", help="also score with a ridge model")
    ap.add_argument("--kernel", default="", help="also score with a local-kernel model")
    ap.add_argument("--parts", type=int, default=20)
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    mesh = REPO / "runs" / "mesh_synth" / "parts"
    parts = load_curve_parts(REPO / "runs" / "wtok_synth")
    val_names = set(json.loads((REPO / "runs" / "wtok_curve_synth_v1" /
                                "val_names_100.json").read_text(encoding="utf-8")))
    val = [p for p in parts if p.name in val_names][: args.parts]

    model, a, epoch, ch = load_model(REPO / "runs" / args.run / args.ckpt, args.device)
    rm = None
    if args.ridge:
        from cae_mesh_generator.wtok.ridge import RidgeNet
        rk = torch.load(REPO / "runs" / args.ridge / "best.pt",
                        map_location=args.device, weights_only=False)
        ra = argparse.Namespace(**rk["args"])
        rm = RidgeNet(ra.dim, ra.layers, ra.heads).to(args.device)
        rm.load_state_dict(rk["model"])
        rm.eval()

    km = None
    if args.kernel:
        from cae_mesh_generator.wtok.kernel import PatchNet
        kk = torch.load(REPO / "runs" / args.kernel / "best.pt",
                        map_location=args.device, weights_only=False)
        ka = argparse.Namespace(**kk["args"])
        km = PatchNet(ka.dim, ka.layers, ka.heads).to(args.device)
        km.load_state_dict(kk["model"])
        km.eval()

    modes = (["surface"] + (["ridge"] if rm is not None else [])
             + (["kernel"] if km is not None else []))
    results = {}
    for mode in modes:
        acc = {k: [] for k in ("surf", "wire", "outline", "bend", "spur",
                               "fix", "self", "kept")}
        for p in val:
            d = np.load(mesh / f"{p.name}.npz")
            span = np.maximum(d["env_hi"] - d["env_lo"], 1e-9)
            gs = d["xyz"] * span + d["env_lo"]
            gtq = realized_q(p, p.vertices, p.edges)
            gw = realize_points(gtq)
            fixq = [bin_center(gtq, v) for v in p.vertices if v["T"] == "FIX"]
            xyz, nrm, _ = generate(model, p, args.device, 48, 7, 2.0, a.points,
                                   True, per_fix=getattr(a, "anchor_per_fix", 1))
            assert len(xyz) == a.points, "generator returned the wrong count"
            acc["surf"].append(chamfer_mm(xyz, gs))
            acc["fix"].append(float(np.mean(
                [np.min(np.linalg.norm(xyz - q, axis=1)) for q in fixq])))
            if mode == "ridge":
                (op, bp), kept = ridge_curves(rm, xyz, nrm, args.device)
                acc["kept"].append(float(np.mean(kept)))
            elif mode == "kernel":
                from cae_mesh_generator.wtok.kernel import curves_from
                with torch.no_grad():
                    (op, bp), kept = curves_from(km, xyz, nrm, args.device)
                acc["kept"].append(float(np.mean(kept)))
            else:
                op, bp = extract_features(xyz, nrm)[:2]
            w = np.concatenate([x for x in (op, bp) if len(x)]) \
                if len(op) or len(bp) else np.zeros((0, 3))
            if not len(w):
                continue
            acc["wire"].append(chamfer_mm(w, gw))
            acc["spur"].append(one_way(w, gw)[0])
            if len(op):
                acc["outline"].append(one_way(class_points(p, {"outer_boundary"}), op)[0])
            if len(bp):
                acc["bend"].append(one_way(class_points(p, {"bend_line"}), bp)[0])
            # self-consistency: does the wire sit on the cloud it came from?
            acc["self"].append(float(np.median(cKDTree(xyz).query(w)[0])))
        med = lambda k: float(np.nanmedian(acc[k])) if acc[k] else float("nan")
        results[mode] = {k: round(med(k), 2) for k in acc}

    best = min(r["wire"] for r in results.values())
    verdict = ("PASS" if best < BASE["wire"]
               else "HOLD" if best < HOLD else "FAIL")
    out = {"tag": args.tag, "run": args.run, "ckpt": args.ckpt, "epoch": epoch,
           "points": a.points, "parts": len(val), "baseline": BASE,
           "results": results, "verdict": verdict}
    print(json.dumps(out, indent=1))
    log = REPO / "runs" / args.run / "judgements.jsonl"
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")
    print(f"\n{verdict}: best wire {best:.2f}mm vs baseline {BASE['wire']}mm "
          f"(hold band up to {HOLD}mm)")


if __name__ == "__main__":
    main()

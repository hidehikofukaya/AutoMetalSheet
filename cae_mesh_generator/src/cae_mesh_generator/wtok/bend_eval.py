"""Does the outline condition actually matter to the bend stage?

The staged design rests on one claim: stage 2 receives something stage 1
produced that it could not have worked out for itself. If bends generated with
the outline are no better than bends generated with the outline blanked out,
the series is decorative and the design has to go back.

So this measures the same model twice on the same parts, changing only whether
the context is the real outline or zeros, and reports the gap. It also reports
the flag error (does the model use the right number of strands) and the strand
straightness, because a bend line that wanders is not a bend line.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
from scipy.spatial import cKDTree

from .dataset_curve import load_curve_parts
from .meshgen import fastener_frame
from .staged import (BEND_PER_STRAND, BEND_STRANDS, StageDataset, StageFlow,
                     sample)


def straightness(strand):
    """Residual of a straight-line fit, over the strand's own length.

    0 = perfectly straight. A bend line is straight or a gentle arc; a value
    near 0.3 means the points are not following any single line.
    """
    q = strand - strand.mean(0)
    s = np.linalg.svd(q, compute_uv=False)
    span = float(np.linalg.norm(strand[-1] - strand[0])) or 1e-9
    return float(s[1] / span)


def score(model, ds, device, n_parts, steps, scale, blank: bool):
    """Median coverage / flag error / straightness over the parts.

    Coverage is converted to mm. Frame coordinates are divided by the part's own
    fastener separation, so a raw 0.039 is not 0.039mm -- on a 100mm separation
    it is 3.9mm, a hundredfold difference in what the number means.
    """
    cov, flag, strt = [], [], []
    for i in range(min(n_parts, len(ds))):
        it = ds[i]
        ctx = it["ctx"][None].to(device)
        if blank:
            ctx = torch.zeros_like(ctx)
        x = sample(model, it["cond"][None].to(device), it["fix"][None].to(device),
                   ctx, ds.n_pts, steps, scale)[0].cpu().numpy().astype(np.float64)
        want = it["x"].numpy()
        live_t = want[:, 6] < 0.5
        if live_t.sum() < BEND_PER_STRAND:
            continue
        live_g = x[:, 6] < 0.5
        flag.append(float(np.mean(live_t != live_g)))
        mm = fastener_frame(ds.parts[i])[2]        # frame unit, in mm
        if live_g.sum() >= 3:
            # coverage: how far each TRUE bend point is from the nearest
            # generated one. Distance to the truth is not the pass condition
            # (the task's own floor is 14.9mm) but the WITH-vs-WITHOUT gap is.
            cov.append(mm * float(
                cKDTree(x[live_g, :3]).query(want[live_t, :3])[0].mean()))
        g = x[:, :3].reshape(BEND_STRANDS, BEND_PER_STRAND, 3)
        on = x[:, 6].reshape(BEND_STRANDS, BEND_PER_STRAND)[:, 0] < 0.5
        for s_ in g[on]:
            strt.append(straightness(s_))
    return (float(np.median(cov)) if cov else float("nan"),
            float(np.median(flag)) if flag else float("nan"),
            float(np.median(strt)) if strt else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/stage2_bend/best.pt")
    ap.add_argument("--dataset", default="runs/mesh_synth")
    ap.add_argument("--wtok", default="runs/wtok_synth")
    ap.add_argument("--val-list", default="runs/wtok_curve_synth_v1/val_names_100.json")
    ap.add_argument("--parts", type=int, default=40)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    ta = ck["args"]
    model = StageFlow(ta["dim"], ta["layers"], ta["heads"],
                      cross=True, ordered="strand").to(a.device)
    model.load_state_dict(ck["model"])
    model.eval()

    names = set(json.load(open(a.val_list)))
    md = pathlib.Path(a.dataset) / "parts"
    have = {f.stem for f in md.glob("*.npz")}
    parts = [p for p in load_curve_parts(pathlib.Path(a.wtok))
             if p.name in names and p.name in have]
    ds = StageDataset(parts, md, "bend", base_seed=11)
    print(f"checkpoint {a.ckpt} (epoch {ck.get('epoch', '?')}), {len(parts)} val parts\n")

    rows = []
    with torch.no_grad():
        for blank in (False, True):
            rows.append(score(model, ds, a.device, a.parts, a.steps,
                              a.cfg_scale, blank))
    print(f"{'context':<12}{'coverage/mm':>12}{'flag err':>10}{'straight':>10}")
    for lbl, r in zip(("outline", "blanked"), rows):
        print(f"{lbl:<12}{r[0]:>12.3f}{r[1]:>10.3f}{r[2]:>10.3f}")
    print()
    # The gate is RELATIVE on all three measures. An absolute millimetre
    # threshold cannot be set here: parts differ by 14.9mm under identical
    # fastening conditions, so what counts as a meaningful distance is a
    # property of the part, not of the model.
    ok = True
    for j, name in enumerate(("coverage", "flag err", "straight")):
        lift = rows[1][j] / max(rows[0][j], 1e-9)
        ok &= lift > 1.2
        print(f"  {name:<10} blanking makes it {lift:.2f}x worse")
    print("\nPASS: the outline carries information" if ok else
          "\nFAIL: the outline is not being used -- the series is decorative")


if __name__ == "__main__":
    main()

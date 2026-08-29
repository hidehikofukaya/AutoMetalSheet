"""Are the generated folds attached to the part, or floating?

There is no "floating" flag in the data, so the test is distributional: a fold
in the true data sits a bounded distance from the outline (median 11.4mm, p95
19.5mm over 200 parts) and inside the part's silhouette. A wire that drifts
away shows up as a tail those numbers do not have.

Reported against the true folds of the SAME parts, never against an absolute
millimetre threshold -- parts with identical fastening differ by 14.9mm.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
from scipy.spatial import cKDTree

from .dataset_curve import load_curve_parts
from .frame import BEND_SLOTS, realize_bend, realize_frame
from .meshgen import fastener_frame
from .staged import StageDataset, StageFlow, sample


def attach(folds, outline):
    """Distance from every fold endpoint to the outline, in frame units."""
    if not folds or len(outline) < 4:
        return np.array([])
    t = cKDTree(outline)
    return np.array([t.query(f[i])[0] for f in folds for i in (0, -1)])


def inside_frac(folds, outline):
    """Share of fold points inside the outline's silhouette.

    Projected onto the outline's best-fit plane and tested by ray crossing. A
    fold outside the boundary is not a fold in the part.
    """
    if not folds or len(outline) < 8:
        return float("nan")
    c = outline.mean(0)
    V = np.linalg.svd(outline - c)[2]
    e1, e2 = V[0], V[1]
    poly = np.stack([(outline - c) @ e1, (outline - c) @ e2], 1)
    pts = np.concatenate(folds)
    q = np.stack([(pts - c) @ e1, (pts - c) @ e2], 1)
    n = len(poly)
    hits = np.zeros(len(q), int)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        cond = (a[1] > q[:, 1]) != (b[1] > q[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            xin = (b[0] - a[0]) * (q[:, 1] - a[1]) / (b[1] - a[1]) + a[0]
        hits += (cond & (q[:, 0] < xin)).astype(int)
    return float((hits % 2 == 1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/bendframe1/best.pt")
    ap.add_argument("--dataset", default="runs/mesh_synth")
    ap.add_argument("--wtok", default="runs/wtok_synth")
    ap.add_argument("--val-list", default="runs/wtok_curve_synth_v1/val_names_100.json")
    ap.add_argument("--cloud-bank", default="")
    ap.add_argument("--parts", type=int, default=40)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--blank-ctx", action="store_true",
                    help="zero the context, to show whether it is being used")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    ta = ck["args"]
    m = StageFlow(ta["dim"], ta["layers"], ta["heads"], cross=True,
                  ordered="", ch=11).to(a.device)
    m.load_state_dict(ck["model"])
    m.eval()

    names = set(json.load(open(a.val_list)))
    md = pathlib.Path(a.dataset) / "parts"
    have = {f.stem for f in md.glob("*.npz")}
    parts = [p for p in load_curve_parts(pathlib.Path(a.wtok))
             if p.name in names and p.name in have]
    bank = a.cloud_bank or ta.get("cloud_bank") or None
    ds = StageDataset(parts, md, "bend_frame", base_seed=11, cloud_bank=bank)
    ods = StageDataset(parts, md, "outline_frame", base_seed=11)

    rows = {"true": [], "generated": []}
    with torch.no_grad():
        for k in range(min(a.parts, len(ds))):
            it, ot = ds[k], ods[k]
            unit = fastener_frame(ds.parts[k])[2]
            outline = realize_frame(ot["x"].numpy().astype(np.float64), 60)
            ctx = it["ctx"][None].to(a.device)
            if a.blank_ctx:
                ctx = torch.zeros_like(ctx)
            g = sample(m, it["cond"][None].to(a.device), it["fix"][None].to(a.device),
                       ctx, BEND_SLOTS, a.steps, a.cfg_scale
                       )[0].cpu().numpy().astype(np.float64)
            w = it["x"].numpy().astype(np.float64)
            for tag, z in (("true", w), ("generated", g)):
                folds = realize_bend(z)
                d = attach(folds, outline)
                if not len(d):
                    continue
                rows[tag].append((
                    len(folds),
                    unit * float(np.median(d)),
                    unit * float(np.percentile(d, 95)),
                    unit * float(d.max()),
                    inside_frac(folds, outline),
                ))
    print(f"{a.ckpt} (epoch {ck.get('epoch','?')})"
          f"{'  [context blanked]' if a.blank_ctx else ''}, {len(rows['true'])} parts\n")
    print(f"{'':<12}{'folds':>7}{'end->outline':>14}{'p95':>8}{'worst':>9}{'inside':>9}")
    for tag, v in rows.items():
        A = np.array(v, float)
        print(f"{tag:<12}{np.median(A[:,0]):>7.0f}{np.median(A[:,1]):>12.1f}mm"
              f"{np.median(A[:,2]):>7.1f}{np.median(A[:,3]):>9.1f}"
              f"{100*np.median(A[:,4]):>8.0f}%")
    T, G = np.array(rows["true"], float), np.array(rows["generated"], float)
    # a floating wire is one whose end is further out than the true data ever goes
    lim = np.percentile(T[:, 3], 95)
    print(f"\nfloating test: worst endpoint beyond {lim:.1f}mm "
          f"(the true worst case) in {100*np.mean(G[:,3] > lim):.0f}% of parts")
    print(f"inside the silhouette: true {100*np.median(T[:,4]):.0f}%  "
          f"generated {100*np.median(G[:,4]):.0f}%")


if __name__ == "__main__":
    main()

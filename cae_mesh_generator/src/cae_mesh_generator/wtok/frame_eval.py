"""The parametric frame against the sampled-point outline it replaces.

Both produce an outline for the same part from the same two fastening points,
so they can be held to the same measure: how far the produced curve sits from
the true one, and whether it is a curve at all. The sampled version has to earn
closure -- its endpoint fraction was the gate it was trained against. The frame
gets closure for free and is measured on what it can still get wrong: the
number of edges, the share that are arcs, and corners piling up in one place
instead of spreading around the loop.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
from scipy.spatial import cKDTree

from .dataset_curve import load_curve_parts
from .frame import EDGE_SLOTS, realize_frame
from .meshgen import fastener_frame
from .staged import N_OUTLINE, StageDataset, StageFlow, sample
from .validity import outline_closed


def corner_spread(x):
    """Smallest gap between consecutive corners, over the mean gap.

    Near 1 the corners are spread evenly around the loop. Near 0 two of them sit
    on top of each other -- the model hedging by spending edges in one place
    rather than committing to a corner position.
    """
    p = x[x[:, 7] < 0.5, 0:3]
    if len(p) < 3:
        return np.nan
    d = np.linalg.norm(np.roll(p, -1, 0) - p, axis=1)
    return float(d.min() / max(d.mean(), 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-ckpt", default="runs/frame1/best.pt")
    ap.add_argument("--point-ckpt", default="runs/stage1_1k/best.pt")
    ap.add_argument("--dataset", default="runs/mesh_synth")
    ap.add_argument("--wtok", default="runs/wtok_synth")
    ap.add_argument("--val-list", default="runs/wtok_curve_synth_v1/val_names_100.json")
    ap.add_argument("--parts", type=int, default=50)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    def load(path, **kw):
        ck = torch.load(path, map_location=a.device, weights_only=False)
        ta = ck["args"]
        m = StageFlow(ta["dim"], ta["layers"], ta["heads"], **kw).to(a.device)
        m.load_state_dict(ck["model"])
        m.eval()
        return m, ck["epoch"]

    mf, ef = load(a.frame_ckpt, cross=False, ordered="loop", ch=8)
    mp, ep = load(a.point_ckpt, cross=False, ordered="loop")

    names = set(json.load(open(a.val_list)))
    md = pathlib.Path(a.dataset) / "parts"
    have = {f.stem for f in md.glob("*.npz")}
    parts = [p for p in load_curve_parts(pathlib.Path(a.wtok))
             if p.name in names and p.name in have]
    df = StageDataset(parts, md, "outline_frame", base_seed=11)
    dp = StageDataset(parts, md, "outline", base_seed=11)

    off_f, off_p, ends_f, ends_p = [], [], [], []
    de, da, spread, spread_t = [], [], [], []
    with torch.no_grad():
        for i in range(min(a.parts, len(df))):
            itf, itp = df[i], dp[i]
            mm = fastener_frame(df.parts[i])[2]
            g = sample(mf, itf["cond"][None].to(a.device), itf["fix"][None].to(a.device),
                       None, EDGE_SLOTS, a.steps, a.cfg_scale)[0].cpu().numpy().astype(np.float64)
            q = sample(mp, itp["cond"][None].to(a.device), itp["fix"][None].to(a.device),
                       None, N_OUTLINE, a.steps, a.cfg_scale)[0, :, :3].cpu().numpy().astype(np.float64)
            w = itf["x"].numpy().astype(np.float64)
            ref = realize_frame(w, per_edge=60)
            got = realize_frame(g, per_edge=60)
            if not len(got) or not len(ref):
                continue
            off_f.append(mm * float(cKDTree(ref).query(got)[0].mean()))
            off_p.append(mm * float(cKDTree(ref).query(q)[0].mean()))
            ends_f.append(outline_closed(got)[0])
            ends_p.append(outline_closed(q)[0])
            nl_g, nl_w = int((g[:, 7] < .5).sum()), int((w[:, 7] < .5).sum())
            de.append(abs(nl_g - nl_w) / max(nl_w, 1))
            live_g, live_w = g[g[:, 7] < .5], w[w[:, 7] < .5]
            da.append(float((live_g[:, 6] > .5).mean()) - float((live_w[:, 6] > .5).mean()))
            spread.append(corner_spread(g))
            spread_t.append(corner_spread(w))

    m_ = lambda v: float(np.nanmedian(v))
    print(f"frame ep {ef} vs sampled points ep {ep},  {len(off_f)} val parts\n")
    print(f"{'':<26}{'frame (32 edges)':>18}{'points (300)':>16}")
    print(f"{'offset from true / mm':<26}{m_(off_f):>18.2f}{m_(off_p):>16.2f}")
    print(f"{'endpoint fraction':<26}{m_(ends_f):>18.3f}{m_(ends_p):>16.3f}")
    print(f"\nframe only:")
    print(f"  edge-count error      {100*m_(de):.1f}%")
    print(f"  arc share vs true     {100*m_(da):+.1f} points")
    print(f"  corner spread          {m_(spread):.3f}   (true {m_(spread_t):.3f}"
          f"; 1.0 = corners evenly placed, 0 = piled up)")


if __name__ == "__main__":
    main()

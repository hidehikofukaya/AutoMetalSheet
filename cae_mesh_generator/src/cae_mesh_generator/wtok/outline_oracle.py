"""D2 decomposition for the outline stage: single / ranked / best-of-K (oracle by near mm).

If best-of-K is far better than ranked, the model *can* draw the part and selection or
conditioning is the lever.  If best-of-K is also bad, the model cannot draw it -- the lever
is representation or input information.

  python -m cae_mesh_generator.wtok.outline_oracle --ckpt runs/frame_2677/best.pt \
      --wtok runs/wtok_synth_g1 --val-list runs/wtok_synth_g1/val_flange_30.json
"""
import argparse, json, pathlib
import numpy as np
import torch
from scipy.spatial import cKDTree

from .dataset_curve import load_curve_parts
from .frame import EDGE_SLOTS, frame_target, realize_frame
from .rational import despike, rank, score, seat_edge_project, seat_project, seats
from .rational_eval import K_DRAWS, load_model, part_inputs
from .staged import sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wtok", default="runs/wtok_synth_g1")
    ap.add_argument("--val-list", required=True)
    ap.add_argument("--parts", type=int, default=30)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cfg-scale", type=float, default=0.0,
                    help="override the checkpoint's guidance scale (0 = keep)")
    a = ap.parse_args()

    model, ep, targs, ch = load_model(a.ckpt, a.device)
    if a.cfg_scale > 0:
        targs = dict(targs, cfg_scale=a.cfg_scale)
    wtok = pathlib.Path(a.wtok)
    names = set(json.load(open(a.val_list)))
    parts = [p for p in load_curve_parts(wtok) if p.name in names][: a.parts]
    from .sidecar import load_spec

    rows = []
    for p in parts:
        fr, cond, fix = part_inputs(p, targs.get("use_spec", False), a.device)
        xt = frame_target(p, fr)
        if xt is None:
            continue
        P, A, r, t = seats(p)

        def constrain(x1):
            y = x1.clone()
            for b in range(y.shape[0]):
                y[b] = torch.from_numpy(seat_project(
                    y[b].cpu().numpy().astype(np.float64), fr, P, A, r).astype(np.float32)).to(y.device)
            return y

        draws = []
        for k in range(K_DRAWS):
            g = torch.Generator(a.device).manual_seed(1000 + k)
            x = sample(model, cond, fix, None, EDGE_SLOTS, a.steps, targs.get("cfg_scale", 1.0), gen=g, constrain=constrain)
            draws.append(seat_edge_project(despike(x[0].cpu().numpy().astype(np.float64), t / fr[2]), fr, P, A, r))
        ref = realize_frame(xt, 60)
        near = []
        for x in draws:
            got = realize_frame(x, 60)
            near.append(fr[2] * float(cKDTree(ref).query(got)[0].mean()))
        order, _ = rank(draws, fr, P, r, t)
        ex = [score(x, fr, P, r, t) for x in draws]
        ex = [s["excess_turn"] if s else float("nan") for s in ex]
        st = score(xt, fr, P, r, t)
        rows.append({"part": p.name, "spec": load_spec(p) is not None,
                     "near": near, "ranked_idx": int(order[0]),
                     "excess": ex, "teacher_excess": st["excess_turn"] if st else None})

    near = np.array([r["near"] for r in rows])
    ranked = np.array([r["near"][r["ranked_idx"]] for r in rows])
    ex = np.array([r["excess"] for r in rows])
    ex_ranked = np.array([r["excess"][r["ranked_idx"]] for r in rows])
    ex_t = np.array([r["teacher_excess"] for r in rows], float)
    print(f"ckpt {a.ckpt} ep {ep}  {len(rows)} parts  K={K_DRAWS}  spec rows present: {sum(r['spec'] for r in rows)}/{len(rows)}")
    print(f"{'':<22}{'near mm (median over parts)':>30}{'excess deg':>12}")
    print(f"{'single (mean of K)':<22}{np.median(near.mean(1)):>30.2f}{np.nanmedian(np.nanmean(ex, 1)):>12.0f}")
    print(f"{'ranked':<22}{np.median(ranked):>30.2f}{np.nanmedian(ex_ranked):>12.0f}")
    print(f"{'best-of-K (oracle)':<22}{np.median(near.min(1)):>30.2f}{np.nanmedian(np.nanmin(ex, 1)):>12.0f}")
    print(f"{'worst-of-K':<22}{np.median(near.max(1)):>30.2f}")
    print(f"{'teacher excess':<22}{'':>30}{np.nanmedian(ex_t):>12.0f}")
    for thr in (2.0, 3.0):
        frac = (near < thr).mean(1)
        print(f"draws with near<{thr}mm: median share per part {np.median(frac):.2f}; parts with >=1 such draw {np.mean(frac > 0):.2f}; ranked picked one {np.mean(ranked < thr):.2f}")


if __name__ == "__main__":
    main()

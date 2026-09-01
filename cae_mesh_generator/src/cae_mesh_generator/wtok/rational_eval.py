"""The standard outline evaluation: K draws, seat-constrained sampling,
rationality rerank, three-tier report -- the harness KB 17.5's numbers came
from, rebuilt as a module after the scratchpad copy was cleaned away.

Adds KB 18's junction agreement when both the checkpoint and the teacher carry
the G1 channel, and draws the teacher/median/ranked overlays the habit sheet
demands (a number alone has lied nine times on this project).

  python -m cae_mesh_generator.wtok.rational_eval --ckpt runs/frame_rel/best.pt \
      --wtok runs/wtok_synth --out runs/frame_rel/rational_eval
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
from scipy.spatial import cKDTree

from .dataset_curve import load_curve_parts
from .frame import EDGE_SLOTS, G1_CH, frame_target, medoid, realize_frame
from .meshgen import fastener_disc, fastener_frame, frame_cond_rows, to_frame
from .rational import despike, junction_match, rank, score, seat_project, seats
from .staged import GUARD_PER_FIX, StageFlow, sample

K_DRAWS = 9


def load_model(path, device):
    """Rebuild a StageFlow from its checkpoint alone: channel width from the
    weights, cross-attention from whether the context encoder was saved, and
    the slot structure from the stage it was trained for. The old hardcoded
    cross=False/'loop' silently mis-built every non-outline model."""
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    ch = ck["model"]["inp.weight"].shape[1]     # the layout the ckpt was trained on
    cross = any(k.startswith("enc.") for k in ck["model"])
    from .staged import ORDERED_PE
    ordered = ORDERED_PE.get(a.get("stage", "outline_frame"), "loop")
    m = StageFlow(a["dim"], a["layers"], a["heads"], cross=cross,
                  ordered=ordered, ch=ch, rel=a.get("rel_attn", False)).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck.get("epoch", -1), a, ch


def part_inputs(p, use_spec, device):
    fr = fastener_frame(p)
    rng = np.random.default_rng(0)
    fx, fn = fastener_disc(p, GUARD_PER_FIX, rng=rng)
    f, fd = to_frame(fx, fn, fr)
    cond = frame_cond_rows(p)
    if use_spec:
        from .sidecar import load_spec
        sp = load_spec(p)
        if sp is not None:
            row = np.zeros((1, cond.shape[1]), np.float32)
            row[0, :min(len(sp), cond.shape[1])] = sp[:cond.shape[1]]
            cond = np.concatenate([cond, row])
    return (fr,
            torch.from_numpy(cond)[None].to(device),
            torch.from_numpy(np.concatenate([f, fd], 1).astype(np.float32)
                             )[None].to(device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wtok", default="runs/wtok_synth")
    ap.add_argument("--val-list", default="")
    ap.add_argument("--parts", type=int, default=30)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    out = pathlib.Path(a.out or (pathlib.Path(a.ckpt).parent / "rational_eval"))
    out.mkdir(parents=True, exist_ok=True)

    model, ep, targs, ch = load_model(a.ckpt, a.device)
    wtok = pathlib.Path(a.wtok)
    vl = pathlib.Path(a.val_list or (wtok / "val_names_100.json"))
    if not vl.exists():
        vl = pathlib.Path("runs/wtok_curve_synth_v1/val_names_100.json")
    names = set(json.load(open(vl)))
    mesh = pathlib.Path("runs/mesh_synth/parts")
    if mesh.exists():        # the subset every previous eval ran on
        names &= {f.stem for f in mesh.glob("*.npz")}
    parts = [p for p in load_curve_parts(wtok) if p.name in names][: a.parts]

    rows, panels = [], []
    for p in parts:
        fr, cond, fix = part_inputs(p, targs.get("use_spec", False), a.device)
        xt = frame_target(p, fr)
        if xt is None:
            continue
        P, A, r, t = seats(p)
        sf, af = to_frame(P, A, fr)

        def constrain(x1):
            y = x1.clone()
            for b in range(y.shape[0]):
                y[b] = torch.from_numpy(seat_project(
                    y[b].cpu().numpy().astype(np.float64), fr, P, A, r
                ).astype(np.float32)).to(y.device)
            return y

        def k_draws(constr):
            out = []
            for k in range(K_DRAWS):
                g = torch.Generator(a.device).manual_seed(1000 + k)
                x = sample(model, cond, fix, None, EDGE_SLOTS, a.steps,
                           targs.get("cfg_scale", 1.0), gen=g, constrain=constr)
                out.append(x[0].cpu().numpy().astype(np.float64))
            return out

        draws = [despike(x, t / fr[2]) for x in k_draws(constrain)]
        order, _ = rank(draws, fr, P, r, t)
        best = draws[order[0]]
        # the baseline column stays what KB 17.5 measured: unconstrained
        # sampling picked by the medoid
        med = medoid(k_draws(None))[1]

        st = score(xt, fr, P, r, t)
        row = {"part": p.name, "teacher": st}
        for tag, x in (("ranked", best), ("medoid", med)):
            s = score(x, fr, P, r, t)
            if s is None:
                continue
            ref = realize_frame(xt, 60)
            got = realize_frame(x, 60)
            # offset from true, the direction KB 17.5 reported
            s["near_mm"] = fr[2] * float(cKDTree(ref).query(got)[0].mean())
            jm = junction_match(x, xt, tol=2.0 * t / fr[2]) if ch > 11 else None
            if jm:
                s.update(jm)
            row[tag] = s
        rows.append(row)
        if len(panels) < 12:
            panels.append((p.name, realize_frame(xt, 60), realize_frame(best, 60)))

    (out / "scores.json").write_text(json.dumps(rows, indent=1))

    def med_of(tag, key):
        v = [r[tag][key] for r in rows if tag in r and r[tag] and key in r[tag]]
        return float(np.median(v)) if v else float("nan")

    print(f"ckpt {a.ckpt} ep {ep}  ch {ch}  {len(rows)} val parts, K={K_DRAWS}")
    print(f"{'':<14}{'seat':>7}{'seat>=0.95':>11}{'excess':>8}{'sliver':>8}"
          f"{'spikes':>8}{'cross':>7}{'edges':>7}{'near mm':>9}")
    for tag in ("teacher", "medoid", "ranked"):
        ok = [r[tag]["seat"] >= 0.95 for r in rows if tag in r and r[tag]]
        sp = [r[tag].get("spikes", 0) > 0 for r in rows if tag in r and r[tag]]
        cx = [r[tag].get("crossings", 0) > 0 for r in rows if tag in r and r[tag]]
        print(f"{tag:<14}{med_of(tag,'seat'):>7.3f}{100*np.mean(ok):>10.0f}%"
              f"{med_of(tag,'excess_turn'):>8.0f}{med_of(tag,'sliver'):>8.2f}"
              f"{100*np.mean(sp):>7.0f}%{100*np.mean(cx):>6.0f}%"
              f"{med_of(tag,'edges'):>7.0f}{med_of(tag,'near_mm'):>9.2f}")
    if ch > 11:
        print(f"junction (ranked): g0_missed {med_of('ranked','g0_missed'):.2f}  "
              f"g1_as_g0 {med_of('ranked','g1_as_g0'):.2f}  "
              f"g1 share gen {med_of('ranked','g1_share_gen'):.2f} "
              f"/ true {med_of('ranked','g1_share_true'):.2f}")

    # the picture: teacher and ranked pick, same linewidth, oblique view,
    # per-part tight box (the drawing rules in CLAUDE.md section C)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(panels)
    if n:
        fig = plt.figure(figsize=(3.2 * 4, 3.0 * ((n + 3) // 4)))
        for i, (name, ref, got) in enumerate(panels):
            ax = fig.add_subplot((n + 3) // 4, 4, i + 1, projection="3d")
            for c, curve in (("0.3", ref), ("tab:red", got)):
                q = np.concatenate([curve, curve[:1]])
                ax.plot(q[:, 0], q[:, 1], q[:, 2], color=c, lw=1.0)
            ax.view_init(elev=28, azim=-55)
            ax.set_title(name[-18:], fontsize=7)
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out / "overlay_3d.png", dpi=150)
        print(f"picture -> {out / 'overlay_3d.png'}")


if __name__ == "__main__":
    main()

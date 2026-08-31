"""E2E bend evaluation for the two-level chain pipeline (KB 20).

fasteners -> outline (constrained + ranked, the KB 18 standard) -> chain set
(2a) -> per-chain edges (2b, batched over chains, open ends snapped to the
generated outline) -> realized chains -> the bend rationality tiers, teacher
as the floor, same 30 val parts as every earlier bend measurement.

  python -m cae_mesh_generator.wtok.chain_eval \
      --ckpt2a runs/chainset1/best.pt --ckpt2b runs/chainedges1/best.pt \
      --outline runs/kaggle_output/frame_g1long/best.pt
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
from scipy.spatial import cKDTree

from .chains import (CE_G1, CEDGE_SLOTS, CHAIN_CH, CHAIN_SLOTS, chain_targets,
                     realize_chain)
from .dataset_curve import load_curve_parts
from .frame import EDGE_SLOTS, frame_target, realize_frame
from .meshgen import fastener_frame
from .rational import bend_score, despike, rank, seat_project, seats
from .rational_eval import load_model, part_inputs
from .staged import sample

K_DRAWS = 9


def snap_ends(x, outline, t_u, reach: float = 4.0):
    """Open-chain terminal corners onto the outline when within reach*t.
    Input-derived: 100% of true open ends lie on the outline (KB 20.1)."""
    x = x.copy()
    live = np.flatnonzero(x[:, 7] < 0.5)
    if len(live) < 2:
        return x
    tree = cKDTree(outline)
    for s in (live[0], live[-1]):
        d, j = tree.query(x[s, 0:3])
        if d < reach * t_u:
            x[s, 0:3] = outline[j]
    return x


def dens(c, step):
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    w = np.linspace(0, s[-1], max(int(s[-1] / step) + 2, 2))
    return np.stack([np.interp(w, s, c[:, d]) for d in range(3)], 1)


def gen_part(p, m_out, ta_out, m2a, ta2a, m2b, ta2b, device, steps=24):
    """One part end to end. Returns dict of realized curve sets and scores."""
    fr, cond, fix = part_inputs(p, ta_out.get("use_spec", False), device)
    u = fr[2]
    P, A, r, t = seats(p)
    t_u = t / u

    def con_seat(x1):
        y = x1.clone()
        for b in range(y.shape[0]):
            y[b] = torch.from_numpy(seat_project(
                y[b].cpu().numpy().astype(np.float64), fr, P, A, r
            ).astype(np.float32)).to(y.device)
        return y

    # stage 1: outline, the KB 18.7 standard recipe
    draws = []
    for k in range(K_DRAWS):
        g = torch.Generator(device).manual_seed(1000 + k)
        draws.append(despike(sample(m_out, cond, fix, None, EDGE_SLOTS, steps,
                     ta_out.get("cfg_scale", 1.0), gen=g, constrain=con_seat
                     )[0].cpu().numpy().astype(np.float64), t_u))
    order, _ = rank(draws, fr, P, r, t)
    out_x = draws[order[0]]
    outline = realize_frame(out_x, 60)

    # outline frame as context rows, the same layout StageDataset builds
    def octx(ch):
        liv = out_x[:, 7] < 0.5
        c = np.zeros((int(liv.sum()), ch), np.float32)
        c[:, :3] = out_x[liv, 0:3]
        d = np.roll(out_x[liv, 0:3], -1, 0) - out_x[liv, 0:3]
        c[:, 3:6] = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        extra = out_x[liv, 6:]
        n_extra = min(extra.shape[1], ch - 6)
        c[:, 6:6 + n_extra] = extra[:, :n_extra]
        if len(c) < EDGE_SLOTS:
            c = np.concatenate([c, c[np.arange(EDGE_SLOTS - len(c)) % len(c)]])
        return torch.from_numpy(c).float().to(device)[None]

    def chains_once(seed):
        # stage 2a
        g = torch.Generator(device).manual_seed(seed)
        x2a = sample(m2a, cond, fix, octx(CHAIN_CH), CHAIN_SLOTS, steps,
                     ta2a.get("cfg_scale", 1.0), gen=g)[0].cpu().numpy().astype(np.float64)
        rows = x2a[x2a[:, CHAIN_CH - 1] < 0.5]
        if not len(rows):
            return [], x2a
        # stage 2b: one draw per chain, batched
        cond2 = []
        for d9 in rows:
            row = np.zeros((1, cond.shape[2]), np.float32)
            row[0, 0:3] = d9[0:3]
            v = d9[3:6]
            row[0, 3:6] = v / max(np.linalg.norm(v), 1e-9)
            row[0, 6] = d9[6]
            row[0, 7] = 1.0 if d9[7] > 0.5 else 0.0
            cond2.append(np.concatenate([cond[0].cpu().numpy(), row]))
        cond2 = torch.from_numpy(np.stack(cond2)).float().to(device)
        B = len(rows)
        fix2 = fix.repeat(B, 1, 1)
        ctx2 = octx(x2b_ch).repeat(B, 1, 1)
        g = torch.Generator(device).manual_seed(seed + 7)
        xb = sample(m2b, cond2, fix2, ctx2, CEDGE_SLOTS, steps,
                    ta2b.get("cfg_scale", 1.0), gen=g).cpu().numpy().astype(np.float64)
        curves = []
        for d9, x in zip(rows, xb):
            closed = d9[7] > 0.5
            if not closed:
                x = snap_ends(x, outline, t_u)
            c = realize_chain(x, closed=closed)
            if c is not None and len(c) >= 2:
                curves.append(c)
        return curves, x2a

    x2b_ch = 10
    sets = [chains_once(31337 * s + 1) for s in range(K_DRAWS)]
    scored = [(bend_score(cs, outline, u, t), cs, xa) for cs, xa in sets]
    order2 = sorted(range(len(scored)), key=lambda i: (
        (1, 0, 0) if scored[i][0] is None
        else (0, scored[i][0]["dangling"], scored[i][0]["excess"])))
    s, cs, xa = scored[order2[0]]
    return {"fr": fr, "u": u, "t": t, "outline": outline,
            "curves": cs, "score": s, "single": scored[0][0], "x2a": xa}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt2a", required=True)
    ap.add_argument("--ckpt2b", required=True)
    ap.add_argument("--outline", default="runs/kaggle_output/frame_g1long/best.pt")
    ap.add_argument("--wtok", default="runs/wtok_synth_g1")
    ap.add_argument("--parts", type=int, default=30)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    out = pathlib.Path(a.out or (pathlib.Path(a.ckpt2b).parent / "chain_eval"))
    out.mkdir(parents=True, exist_ok=True)

    m_out, _, ta_out, _ = load_model(a.outline, a.device)
    m2a, _, ta2a, _ = load_model(a.ckpt2a, a.device)
    m2b, _, ta2b, _ = load_model(a.ckpt2b, a.device)
    wtok = pathlib.Path(a.wtok)
    names = set(json.load(open(wtok / "val_names_100.json")))
    mesh = pathlib.Path("runs/mesh_synth/parts")
    if mesh.exists():
        names &= {f.stem for f in mesh.glob("*.npz")}
    parts = [p for p in load_curve_parts(wtok) if p.name in names][: a.parts]

    rows, panels = [], []
    for p in parts:
        fr = fastener_frame(p)
        tt = chain_targets(p, fr)
        if tt is None:
            continue
        x2a_t, c2b_t = tt
        teach = [realize_chain(x, closed=cl) for x, cl in c2b_t]
        teach = [c for c in teach if c is not None]
        ft = frame_target(p, fr)
        r = gen_part(p, m_out, ta_out, m2a, ta2a, m2b, ta2b, a.device)
        u, t = r["u"], r["t"]
        st = bend_score(teach, realize_frame(ft, 60), u, t) if ft is not None else None
        row = {"part": p.name, "teacher": st, "single": r["single"],
               "ranked": r["score"],
               "n_true": len(teach), "n_gen": len(r["curves"])}
        if r["curves"] and teach:
            A = np.concatenate([dens(c, 1.0 / u) for c in r["curves"]])
            Bv = np.concatenate([dens(c, 1.0 / u) for c in teach])
            row["chamfer"] = u * 0.5 * (cKDTree(Bv).query(A)[0].mean()
                                        + cKDTree(A).query(Bv)[0].mean())
        rows.append(row)
        if len(panels) < 8:
            panels.append((p.name, teach, r["curves"], r["outline"],
                           realize_frame(ft, 60) if ft is not None else None))

    (out / "scores.json").write_text(json.dumps(
        rows, indent=1, default=lambda o: float(o)))

    def med(tag, key):
        v = [r[tag][key] for r in rows if r.get(tag) and key in r[tag]]
        return float(np.median(v)) if v else float("nan")

    print(f"{len(rows)} val parts, K={K_DRAWS} full-pipeline draws")
    print(f"{'':<10}{'excess turn':>12}{'dangling':>10}{'curves':>8}{'chamfer':>9}")
    for tag in ("teacher", "single", "ranked"):
        n = np.median([r["n_gen" if tag != "teacher" else "n_true"] for r in rows])
        print(f"{tag:<10}{med(tag, 'excess'):>9.0f}deg"
              f"{100 * med(tag, 'dangling'):>9.0f}%{n:>8.0f}", end="")
        if tag == "ranked":
            ch = [r["chamfer"] for r in rows if "chamfer" in r]
            print(f"{np.median(ch):>8.2f}mm" if ch else "", end="")
        print()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(panels)
    if n:
        fig = plt.figure(figsize=(7.2 * 2, 3.4 * n))
        for i, (name, teach, gen, ol, ol_t) in enumerate(panels):
            for col, (tag, curves, outline) in enumerate(
                    (("teacher", teach, ol_t), ("generated", gen, ol))):
                ax = fig.add_subplot(n, 2, i * 2 + col + 1, projection="3d")
                if outline is not None:
                    q = np.concatenate([outline, outline[:1]])
                    ax.plot(*q.T, color="0.75", lw=1.0)
                for c in curves:
                    ax.plot(*np.asarray(c).T, lw=0.9,
                            color="k" if col == 0 else "tab:red")
                ax.view_init(elev=28, azim=-55)
                ax.set_title(f"{name[-16:]} {tag}", fontsize=7)
                ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out / "overlay_3d.png", dpi=140)
        print(f"picture -> {out / 'overlay_3d.png'}")


if __name__ == "__main__":
    main()

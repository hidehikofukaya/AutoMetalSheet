"""E2E evaluation of the face-loop pipeline (KB 21).

fasteners -> outline (standard recipe) -> face_set (2a) -> face_ring (2b,
batched over faces) -> realized rings. The network property gives a
TEACHER-FREE self-consistency criterion for ranking: every point of a true
face boundary lies either on another face's boundary or on the outline, so
the fraction of generated ring length that touches neither ("unmatched") is
a defect measure that needs no ground truth. Reported against the teacher:
face count, ring chamfer, high-curvature coverage (the KB 20.8 measure, floor
0.2%), ring excess turning.

  python -m cae_mesh_generator.wtok.face_eval \
      --ckpt2a runs/faceset1/best.pt --ckpt2b runs/facering1/best.pt
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
from scipy.spatial import cKDTree

from .dataset_curve import load_curve_parts
from .faces import (FACE_CH, FACE_SLOTS, FRING_SLOTS, face_targets,
                    realize_face_ring)
from .frame import EDGE_SLOTS, frame_target, realize_frame
from .meshgen import fastener_frame
from .rational import seat_edge_project, despike, rank, seat_project, seats
from .rational_eval import load_model, part_inputs
from .staged import curvature_field, sample

K_DRAWS = 9


def dens(c, step):
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    w = np.linspace(0, s[-1], max(int(s[-1] / step) + 2, 2))
    return np.stack([np.interp(w, s, c[:, d]) for d in range(3)], 1)


def ring_stats(rings, outline, t_u, corners=None):
    """(unmatched share, total excess turn) -- both teacher-free.

    `corners`: per-ring corner arrays; when given, excess turning is taken on
    the corner polygon, not the realized curve. A realized arc turns by its
    sweep legitimately, so on true arcs the curve-based number is dominated
    by arc sweep (teacher floor 12,568deg) and no longer measures wiggle."""
    if not rings:
        return None
    pts = [dens(r, max(t_u, 1e-6)) for r in rings]
    trees = [cKDTree(q) for q in pts]
    otree = cKDTree(outline)
    unmatched = 0
    total = 0
    for i, q in enumerate(pts):
        best = np.full(len(q), np.inf)
        for j, tr in enumerate(trees):
            if j == i:
                continue
            best = np.minimum(best, tr.query(q)[0])
        best = np.minimum(best, otree.query(q)[0])
        unmatched += int((best > 2 * t_u).sum())
        total += len(q)
    excess = 0.0
    for r in (corners if corners is not None else rings):
        d = np.diff(np.concatenate([r, r[:1]]), axis=0)
        L = np.linalg.norm(d, axis=1)
        T = d[L > 1e-12] / L[L > 1e-12, None]
        turn = np.degrees(np.arccos(np.clip(
            np.einsum("ij,ij->i", T, np.roll(T, -1, 0)), -1, 1)))
        excess += max(float(turn.sum()) - 360.0, 0.0)
    return {"unmatched": unmatched / max(total, 1), "excess": excess,
            "faces": len(rings)}


def _ring_key(poly):
    d = np.diff(poly, axis=0)
    L = np.linalg.norm(d, axis=1)
    T = d[L > 1e-12] / L[L > 1e-12, None]
    turn = np.degrees(np.arccos(np.clip(
        np.einsum("ij,ij->i", T, np.roll(T, -1, 0)), -1, 1)))
    return (int((turn > 150).sum()), max(float(turn.sum()) - 360.0, 0.0))


def outline_pin(x, outline, reach):
    """Ring corners within `reach` (frame units) of the outline polyline are
    moved onto it. A face that touches the outline shares those edges with
    stage 1's result, so this is the same kind of input-derived constraint
    as the fastener seat: nothing about shape is decided here."""
    x = x.copy()
    live = np.flatnonzero(x[:, 7] < 0.5)
    if not len(live):
        return x
    d, j = cKDTree(outline).query(x[live, 0:3])
    hit = d < reach
    x[live[hit], 0:3] = outline[j[hit]]
    return x


def gen_faces(p, models, device, steps=24, ring_k=1, ring_despike=False,
              outline_pin_t=0.0):
    """One part end to end; K_DRAWS full-pipeline draws, self-consistency rank."""
    (m_out, ta_out), (m2a, ta2a), (m2b, ta2b) = models
    fr, cond, fix = part_inputs(p, ta_out.get("use_spec", False), device)
    u = fr[2]
    P, A, r, t = seats(p)
    t_u = t / u

    def con(x1):
        y = x1.clone()
        for b in range(y.shape[0]):
            y[b] = torch.from_numpy(seat_project(
                y[b].cpu().numpy().astype(np.float64), fr, P, A, r
            ).astype(np.float32)).to(y.device)
        return y

    draws = []
    for k in range(K_DRAWS):
        g = torch.Generator(device).manual_seed(1000 + k)
        draws.append(seat_edge_project(despike(sample(m_out, cond, fix, None, EDGE_SLOTS, steps,
                     ta_out.get("cfg_scale", 1.0), gen=g, constrain=con
                     )[0].cpu().numpy().astype(np.float64), t_u), fr, P, A, r))   # KB 21.24
    order, _ = rank(draws, fr, P, r, t)
    out_x = draws[order[0]]
    outline = realize_frame(out_x, 60)

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

    def once(seed):
        g = torch.Generator(device).manual_seed(seed)
        xa = sample(m2a, cond, fix, octx(FACE_CH), FACE_SLOTS, steps,
                    ta2a.get("cfg_scale", 1.0), gen=g
                    )[0].cpu().numpy().astype(np.float64)
        rows = xa[xa[:, FACE_CH - 1] < 0.5]
        if not len(rows):
            return [], []
        cond2 = []
        for d8 in rows:
            row = np.zeros((1, cond.shape[2]), np.float32)
            row[0, 0:3] = d8[0:3]
            v = d8[3:6]
            row[0, 3:6] = v / max(np.linalg.norm(v), 1e-9)
            row[0, 6] = d8[6]
            cond2.append(np.concatenate([cond[0].cpu().numpy(), row]))
        cond2 = torch.from_numpy(np.stack(cond2)).float().to(device)
        B = len(rows)
        ctx2 = octx(xb_ch)
        if ta2b.get("sib_ctx", False):
            # KB 21.19: the generated 2a rows of this part as sibling context,
            # laid out exactly as StageDataset does (marker in the last column)
            sib = np.zeros((FACE_SLOTS, xb_ch), np.float32)
            m = len(rows)
            sib[:m, 0:3] = rows[:, 0:3]
            v = rows[:, 3:6]
            sib[:m, 3:6] = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
            sib[:m, 6] = rows[:, 6]
            sib[:m, 7] = 1.0                 # marker column, as in StageDataset
            if m < FACE_SLOTS:
                sib[m:] = sib[np.arange(FACE_SLOTS - m) % m]
            ctx2 = torch.cat([ctx2, torch.from_numpy(sib).float().to(device)[None]], 1)
        # ring_k draws per face; rings are independent, so the pick is
        # PER RING by (spikes, excess) -- the 2b-intrinsic wiggle is the one
        # defect a per-ring choice can reach (KB 21.6.1)
        con2 = None
        if outline_pin_t > 0:
            reach = outline_pin_t * t_u

            def con2(x1):
                y = x1.clone()
                for b in range(y.shape[0]):
                    y[b] = torch.from_numpy(outline_pin(
                        y[b].cpu().numpy().astype(np.float64), outline, reach
                    ).astype(np.float32)).to(y.device)
                return y
        cands = []
        for k in range(ring_k):
            g = torch.Generator(device).manual_seed(seed + 7 + 101 * k)
            xb = sample(m2b, cond2, fix.repeat(B, 1, 1),
                        ctx2.repeat(B, 1, 1), FRING_SLOTS, steps,
                        ta2b.get("cfg_scale", 1.0), gen=g, constrain=con2
                        ).cpu().numpy().astype(np.float64)
            cands.append(xb)
        rings, corners = [], []
        for i in range(B):
            best = None
            for k in range(ring_k):
                x = cands[k][i]
                if ring_despike:
                    x = despike(x, t_u)
                c = realize_face_ring(x)
                if c is None or len(c) < 4:
                    continue
                key = _ring_key(c)
                if best is None or key < best[0]:
                    best = (key, c, x[x[:, 7] < 0.5, 0:3])
            if best is not None:
                rings.append(best[1])
                corners.append(best[2])
        return rings, corners

    xb_ch = 10
    with torch.no_grad():
        sets = [once(31337 * s + 1) for s in range(K_DRAWS)]
    scored = [(ring_stats(rs, outline, t_u, cs), rs) for rs, cs in sets]
    order2 = sorted(range(len(scored)), key=lambda i: (
        (9, 0) if scored[i][0] is None
        else (scored[i][0]["unmatched"], scored[i][0]["excess"])))
    s, rs = scored[order2[0]]
    return {"fr": fr, "u": u, "t": t, "outline": outline, "rings": rs,
            "score": s, "single": scored[0][0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt2a", required=True)
    ap.add_argument("--ckpt2b", required=True)
    ap.add_argument("--outline", default="runs/frame_g1sm2/best.pt")
    ap.add_argument("--wtok", default="runs/wtok_synth_g1")
    ap.add_argument("--parts", type=int, default=30)
    ap.add_argument("--val-list", default="val_occt_60.json",
                    help="name list under --wtok (val_flange_30.json = flange holdout)")
    ap.add_argument("--ring-k", type=int, default=1)
    ap.add_argument("--ring-despike", action="store_true")
    ap.add_argument("--outline-pin", type=float, default=0.0,
                    help="pin ring corners within this many thicknesses of "
                         "the generated outline onto it during 2b sampling")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    out = pathlib.Path(a.out or (pathlib.Path(a.ckpt2b).parent / "face_eval"))
    out.mkdir(parents=True, exist_ok=True)

    models = [load_model(c, a.device)[0::2] for c in (a.outline, a.ckpt2a, a.ckpt2b)]
    models = [(m, t) for m, t in models]
    wtok = pathlib.Path(a.wtok)
    names = set(json.load(open(wtok / a.val_list)))
    mesh = pathlib.Path("runs/mesh_synth/parts")
    if mesh.exists():
        names &= {f.stem for f in mesh.glob("*.npz")}
    parts = [p for p in load_curve_parts(wtok) if p.name in names][: a.parts]

    rows, panels = [], []
    for p in parts:
        fr = fastener_frame(p)
        tt = face_targets(p, fr)
        ft = frame_target(p, fr)
        if tt is None or ft is None:
            continue
        x2a_t, r2b_t, _ = tt
        teach = [c for c in (realize_face_ring(x) for x in r2b_t) if c is not None]
        teach_c = [x[x[:, 7] < 0.5, 0:3] for x in r2b_t]
        r = gen_faces(p, models, a.device, ring_k=a.ring_k,
                      ring_despike=a.ring_despike, outline_pin_t=a.outline_pin)
        u, t = r["u"], r["t"]
        t_u = t / u
        st = ring_stats(teach, realize_frame(ft, 60), t_u, teach_c)
        row = {"part": p.name, "teacher": st, "single": r["single"],
               "ranked": r["score"]}
        if r["rings"] and teach:
            A = np.concatenate([dens(c, 1.0 / u) for c in r["rings"]])
            Bv = np.concatenate([dens(c, 1.0 / u) for c in teach])
            row["chamfer"] = u * 0.5 * (cKDTree(Bv).query(A)[0].mean()
                                        + cKDTree(A).query(Bv)[0].mean())
        # high-curvature coverage (KB 20.8 measure; teacher floor 0.2%)
        f = mesh / f"{p.name}.npz"
        if f.exists() and r["rings"]:
            d0 = np.load(f)
            sp = np.maximum(d0["env_hi"] - d0["env_lo"], 1e-9)
            Pm = d0["xyz"].astype(np.float64) * sp + d0["env_lo"]
            g = curvature_field(Pm, d0["normal"].astype(np.float64))
            hot = Pm[g > 15]
            if len(hot):
                from .meshgen import to_frame
                hf, _ = to_frame(hot, np.zeros_like(hot), fr)
                tr = cKDTree(np.concatenate([dens(c, 1.0 / u) for c in r["rings"]]))
                row["uncovered"] = float(np.mean(u * tr.query(hf)[0] > 3.0))
        rows.append(row)
        if len(panels) < 8:
            panels.append((p.name, teach, r["rings"], realize_frame(ft, 60),
                           r["outline"]))

    (out / "scores.json").write_text(json.dumps(rows, indent=1,
                                                default=lambda o: float(o)))

    def med(tag, key):
        v = [r[tag][key] for r in rows if r.get(tag) and key in r[tag]]
        return float(np.median(v)) if v else float("nan")

    print(f"{len(rows)} val parts, K={K_DRAWS} full-pipeline draws")
    print(f"{'':<10}{'unmatched':>10}{'excess':>9}{'faces':>7}{'chamfer':>9}{'uncovered':>11}")
    for tag in ("teacher", "single", "ranked"):
        print(f"{tag:<10}{100*med(tag,'unmatched'):>9.1f}%{med(tag,'excess'):>8.0f}"
              f"{med(tag,'faces'):>7.0f}", end="")
        if tag == "ranked":
            ch = [r.get("chamfer") for r in rows if "chamfer" in r]
            uc = [r.get("uncovered") for r in rows if "uncovered" in r]
            print(f"{np.median(ch):>8.2f}mm{100*np.median(uc):>10.1f}%", end="")
        print()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(panels)
    if n:
        fig = plt.figure(figsize=(7.2 * 2, 3.4 * n))
        for i, (name, teach, gen, ol_t, ol_g) in enumerate(panels):
            for col, (tag, curves, outline) in enumerate(
                    (("teacher", teach, ol_t), ("generated", gen, ol_g))):
                ax = fig.add_subplot(n, 2, i * 2 + col + 1, projection="3d")
                q = np.concatenate([outline, outline[:1]])
                ax.plot(*q.T, color="0.75", lw=1.0)
                for c in curves:
                    ax.plot(*np.asarray(c).T, lw=0.7,
                            color="k" if col == 0 else "tab:red")
                ax.view_init(elev=28, azim=-55)
                ax.set_title(f"{name[-16:]} {tag}", fontsize=7)
                ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out / "overlay_3d.png", dpi=140)
        print(f"picture -> {out / 'overlay_3d.png'}")


if __name__ == "__main__":
    main()

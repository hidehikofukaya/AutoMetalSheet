"""Symmetry-aware outline evaluation (2026-09-05, user's point: a flange on the other
side is a design choice the fastening points do not decide; a metric that counts it
as error is measuring the wrong thing).

Three distances per part, all in mm, generated (ranked pick) vs teacher:
  near        plain: mean nearest distance to the teacher outline (what rational_eval reports)
  near_sym    minimum over the INPUT SYMMETRY GROUP G: every isometry that maps the
              fastening configuration (positions as a set, axes as unsigned lines,
              seat radii) onto itself. Candidates are the sign flips of the fastener
              frame axes (8 maps incl. identity); a map is kept only if it preserves
              the input within 1 mm / 5 deg. Under such a map the teacher is exactly
              as valid an answer as before, so the generation may match either.
  near_shape  after rigid ICP alignment of the generation onto the best-symmetry
              teacher: the error of the SHAPE alone, with placement removed. The
              alignment magnitude (translation mm, rotation deg) is reported too.

  python -m cae_mesh_generator.wtok.sym_eval --ckpt runs/frame_occt_all_ema2400_s0/best.pt \
      --wtok runs/wtok_synth_g1 --val-list val_flange_30.json
"""
import argparse, itertools, json, pathlib
import numpy as np
import torch
from scipy.spatial import cKDTree

from .dataset_curve import load_curve_parts
from .face_eval import dens
from .frame import EDGE_SLOTS, frame_target, realize_frame
from .meshgen import fastener_frame, to_frame
from .rational import despike, rank, seat_edge_project, seat_project, seats
from .rational_eval import K_DRAWS, load_model, part_inputs
from .staged import sample


def input_symmetries(Pf, Af, tol_mm_u, tol_deg=5.0):
    """Sign-flip maps of the frame axes that preserve the fastening configuration."""
    keep = []
    cos_tol = np.cos(np.radians(tol_deg))
    for signs in itertools.product((1.0, -1.0), repeat=3):
        g = np.diag(signs)
        Q = Pf @ g.T
        B = Af @ g.T
        ok = True
        used = set()
        for i in range(len(Pf)):
            d = np.linalg.norm(Pf - Q[i], axis=1)
            j = int(np.argmin(d))
            if d[j] > tol_mm_u or j in used or abs(float(Af[j] @ B[i])) < cos_tol:
                ok = False
                break
            used.add(j)
        if ok:
            keep.append(g)
    return keep


def variant_outlines(part, fr, u):
    """PartMaker variants of this part (flange side flipped): outer_boundary polylines, frame units, 1 mm."""
    import sys
    from .sidecar import _WF
    wf = _WF / f"{part.name}.json"
    if not wf.exists():
        return []
    src = pathlib.Path(json.loads(wf.read_text())["source_stp"])
    vdir = src.parent.parent / "variants"
    stem = src.stem.replace("_mid", "")
    out = []
    if not vdir.exists():
        return out
    WF_APP = "C:/Users/hide2/IdeaBox/fill_volume/wireframe_app"
    if WF_APP not in sys.path:
        sys.path.insert(0, WF_APP)
    import extract
    for stp in sorted(vdir.glob(f"{stem}__*_mid.stp")):
        try:
            d = extract.extract_wireframe(stp)
        except Exception:
            continue
        pts = [np.asarray(e["polyline"], float) for e in d["edges"] if e.get("type") == "outer_boundary"]
        if not pts:
            continue
        P = np.concatenate(pts)
        Pf, _ = to_frame(P, np.zeros_like(P), fr)
        # the extractor's polylines are per edge; resample each edge at 1 mm then pool
        res = [dens(to_frame(q, np.zeros_like(q), fr)[0], 1.0 / u) for q in pts if len(q) >= 2]
        out.append((stp.stem, np.concatenate(res)))
    return out


def near(gen, ref_tree):
    return float(ref_tree.query(gen)[0].mean())


def icp_rigid(src, dst_tree, dst, iters=12):
    """Kabsch ICP without scale (the fasteners fix the scale). Returns aligned src, |t|, angle."""
    R_tot, t_tot = np.eye(3), np.zeros(3)
    cur = src.copy()
    for _ in range(iters):
        nn = dst[dst_tree.query(cur)[1]]
        mu_s, mu_d = cur.mean(0), nn.mean(0)
        H = (cur - mu_s).T @ (nn - mu_d)
        U, _, Vt = np.linalg.svd(H)
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
        t = mu_d - R @ mu_s
        cur = cur @ R.T + t
        R_tot, t_tot = R @ R_tot, R @ t_tot + t
    ang = np.degrees(np.arccos(np.clip((np.trace(R_tot) - 1) / 2, -1, 1)))
    return cur, float(np.linalg.norm(t_tot)), float(ang)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wtok", default="runs/wtok_synth_g1")
    ap.add_argument("--val-list", required=True)
    ap.add_argument("--parts", type=int, default=30)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="")
    ap.add_argument("--variants", action="store_true",
                    help="also accept PartMaker design-equivalent variants (<chunk>/variants/<id>__*.stp, "
                         "flange side flipped) as references: min over {teacher, variants} x symmetry")
    a = ap.parse_args()
    model, ep, targs, ch = load_model(a.ckpt, a.device)
    wtok = pathlib.Path(a.wtok)
    names = set(json.load(open(wtok / a.val_list)))
    parts = [p for p in load_curve_parts(wtok) if p.name in names][: a.parts]
    rows = []
    for p in parts:
        fr, cond, fix = part_inputs(p, targs.get("use_spec", False), a.device)
        xt = frame_target(p, fr)
        if xt is None:
            continue
        P, A, r, t = seats(p)
        u = fr[2]

        def con(x1):
            y = x1.clone()
            for b in range(y.shape[0]):
                y[b] = torch.from_numpy(seat_project(y[b].cpu().numpy().astype(np.float64), fr, P, A, r).astype(np.float32)).to(y.device)
            return y
        draws = []
        for k in range(K_DRAWS):
            g = torch.Generator(a.device).manual_seed(1000 + k)
            x = sample(model, cond, fix, None, EDGE_SLOTS, a.steps, targs.get("cfg_scale", 1.0), gen=g, constrain=con)
            draws.append(seat_edge_project(despike(x[0].cpu().numpy().astype(np.float64), t / u), fr, P, A, r))
        order, _ = rank(draws, fr, P, r, t)
        gen = dens(realize_frame(draws[order[0]], 60), 1.0 / u)      # 1 mm spacing (habit A1)
        ref = dens(realize_frame(xt, 60), 1.0 / u)
        refs = [("teacher", ref)]
        if a.variants:
            refs += variant_outlines(p, fr, u)
        Pf, Af = to_frame(P, A, fr)
        G = input_symmetries(Pf, Af, 1.0 / u)
        best = None
        best_ref = "teacher"
        for label, rf in refs:
            for g in G:
                refg = rf @ g.T
                d = u * near(gen, cKDTree(refg))
                if best is None or d < best[0]:
                    best = (d, g, refg)
                    best_ref = label
        d_plain = u * near(gen, cKDTree(ref))
        tree = cKDTree(best[2])
        aligned, tmag, ang = icp_rigid(gen, tree, best[2])
        d_shape = u * near(aligned, tree)
        rows.append({"part": p.name, "near": d_plain, "near_sym": best[0], "near_shape": d_shape,
                     "n_sym": len(G), "sym_used": bool(not np.allclose(best[1], np.eye(3))),
                     "n_refs": len(refs), "ref_used": best_ref,
                     "align_t_mm": u * tmag, "align_deg": ang})
    med = lambda k: float(np.median([r[k] for r in rows]))
    print(f"ckpt {a.ckpt} ep {ep}  {len(rows)} parts  K={K_DRAWS}")
    print(f"{'near (plain)':<26}{med('near'):>8.2f} mm")
    print(f"{'near_sym (input symmetry)':<26}{med('near_sym'):>8.2f} mm   |G| median {med('n_sym'):.0f}; non-identity map won on {100*np.mean([r['sym_used'] for r in rows]):.0f}% of parts")
    if a.variants:
        print(f"{'design variants':<26}  refs/part median {med('n_refs'):.0f}; a variant won on {100*np.mean([r['ref_used'] != 'teacher' for r in rows]):.0f}% of parts")
    print(f"{'near_shape (rigid-aligned)':<26}{med('near_shape'):>8.2f} mm   alignment median |t| {med('align_t_mm'):.1f} mm, rot {med('align_deg'):.1f} deg")
    for thr in (2.0, 3.0):
        print(f"parts with near_sym < {thr}mm: {100*np.mean([r['near_sym'] < thr for r in rows]):.0f}%   (plain: {100*np.mean([r['near'] < thr for r in rows]):.0f}%, shape-only: {100*np.mean([r['near_shape'] < thr for r in rows]):.0f}%)")
    if a.out:
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

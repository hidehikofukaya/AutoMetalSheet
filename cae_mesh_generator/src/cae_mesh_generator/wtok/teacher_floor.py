"""Teacher floors per family (habit D6: the teacher's value is the floor, never zero).

For every val list: outline teacher (seat, excess turn, sliver, edges) and face
teacher (faces, self-consistency `unmatched`, corner-polygon excess). Teacher-only,
no model, CPU -- the review gate for a new PartMaker family.

  python -m cae_mesh_generator.wtok.teacher_floor --wtok runs/wtok_synth_g1 \
      --val-list val_names_100.json val_flange_30.json val_occt11_30.json val_occt12_30.json
"""
import argparse, json, pathlib
import numpy as np

from .dataset_curve import load_curve_parts
from .face_eval import ring_stats
from .faces import face_targets, realize_face_ring
from .frame import frame_target, realize_frame
from .meshgen import fastener_frame
from .rational import score, seats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wtok", default="runs/wtok_synth_g1")
    ap.add_argument("--val-list", nargs="+", required=True)
    ap.add_argument("--parts", type=int, default=30)
    a = ap.parse_args()
    wtok = pathlib.Path(a.wtok)
    allparts = {p.name: p for p in load_curve_parts(wtok)}
    print(f"{'family':<20}{'n':>4}{'seat':>7}{'seat>=.95':>10}{'excess':>8}{'sliver':>8}{'edges':>7}"
          f"{'faces':>7}{'ring pts':>9}{'unmatched':>11}{'face excess':>12}{'skipped':>8}")
    for vl in a.val_list:
        names = json.load(open(wtok / vl))[: a.parts]
        rows, skipped = [], 0
        for n in names:
            p = allparts.get(n)
            if p is None:
                skipped += 1
                continue
            fr = fastener_frame(p)
            xt = frame_target(p, fr)
            tt = face_targets(p, fr)
            if xt is None or tt is None:
                skipped += 1
                continue
            P, A, r, t = seats(p)
            st = score(xt, fr, P, r, t) or {}
            x2a_t, r2b_t, _ = tt
            teach = [c for c in (realize_face_ring(x) for x in r2b_t) if c is not None]
            teach_c = [x[x[:, 7] < 0.5, 0:3] for x in r2b_t]
            fs = ring_stats(teach, realize_frame(xt, 60), t / fr[2], teach_c) or {}
            fs = (fs.get("unmatched", np.nan), fs.get("excess", np.nan))
            rows.append({"seat": st.get("seat", np.nan), "excess": st.get("excess_turn", np.nan),
                         "sliver": st.get("sliver", np.nan), "edges": int((xt[:, 7] < 0.5).sum()),
                         "faces": len(teach), "ringpts": float(np.mean([len(c) for c in teach_c])) if teach_c else np.nan,
                         "unmatched": fs[0], "fexcess": fs[1]})
        if not rows:
            print(f"{vl:<20}{0:>4}  (nothing loadable, skipped {skipped})")
            continue
        med = lambda k: float(np.nanmedian([r[k] for r in rows]))
        ok = float(np.mean([r["seat"] >= 0.95 for r in rows]))
        print(f"{vl.replace('.json',''):<20}{len(rows):>4}{med('seat'):>7.3f}{100*ok:>9.0f}%{med('excess'):>8.0f}"
              f"{med('sliver'):>8.2f}{med('edges'):>7.0f}{med('faces'):>7.0f}{med('ringpts'):>9.1f}"
              f"{100*med('unmatched'):>10.1f}%{med('fexcess'):>12.0f}{skipped:>8}")


if __name__ == "__main__":
    main()

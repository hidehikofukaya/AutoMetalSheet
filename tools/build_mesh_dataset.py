"""Tessellate the midsurface STEP of every wtok part into an oriented point cloud.

Output keys match runs/wtok_synth/parts/*.json exactly, so val_names_100.json and
every existing comparison carry over unchanged.

  runs/mesh_synth/parts/<same name>.npz
      xyz     (N,3) float32, normalised into the part's envelope -> [0,1]
      normal  (N,3) float32, unit, oriented by the B-Rep face
      env_lo / env_hi (3,) float64

Local-only: needs pythonocc. Run once, then ship the npz set anywhere.

  python tools/build_mesh_dataset.py
  python tools/build_mesh_dataset.py --limit 20      # smoke
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import topods

REPO = pathlib.Path(__file__).resolve().parent.parent
SYN = pathlib.Path(r"C:\Users\hide2\IdeaBox\PartMaker\synthetic_parts")


def source_dir(tag: str) -> pathlib.Path | None:
    """Part ids repeat across chunks -- the group tag, not the stem, picks the
    directory. Globbing the bare stem silently returns a different part."""
    if tag == "batch02":
        return SYN / "batch02"
    if tag.startswith("p2c"):
        return SYN / "prod02" / f"chunk_{int(tag[3:]):02d}"
    if tag.startswith("c") and tag[1:].isdigit():
        return SYN / "prod01" / f"chunk_{int(tag[1:]):02d}"
    return None


def find_mid(name: str) -> pathlib.Path | None:
    tag, stem = name.split("__", 1)
    d = source_dir(tag)
    if d is None:
        return None
    f = d / "mid" / f"{stem}_mid.stp"
    return f if f.exists() else None


def tessellate(path: pathlib.Path, deflection: float):
    """Triangles with face-oriented normals."""
    reader = STEPControl_Reader()
    reader.ReadFile(str(path))
    reader.TransferRoots()
    shape = reader.OneShape()
    BRepMesh_IncrementalMesh(shape, deflection, False, 0.5, True)
    V: list = []
    T: list = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is not None:
            off, tr = len(V), loc.Transformation()
            flip = face.Orientation() == TopAbs_REVERSED
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(tr)
                V.append([p.X(), p.Y(), p.Z()])
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                if flip:
                    b, c = c, b
                T.append([off + a - 1, off + b - 1, off + c - 1])
        exp.Next()
    return np.asarray(V, dtype=np.float64), np.asarray(T, dtype=np.int64)


def sample_surface(V, T, n: int, rng):
    """Area-weighted points + the triangle normal at each one. Mesh vertices are
    clustered where the tessellator needed detail, so sampling by area instead
    keeps the density uniform over the sheet."""
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    cross = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    keep = area > 1e-12
    a, b, c, cross, area = a[keep], b[keep], c[keep], cross[keep], area[keep]
    if len(area) == 0:
        return None, None
    idx = rng.choice(len(area), size=n, p=area / area.sum())
    u = rng.random((n, 1))
    v = rng.random((n, 1))
    over = (u + v) > 1.0
    u[over], v[over] = 1.0 - u[over], 1.0 - v[over]
    pts = a[idx] + (b[idx] - a[idx]) * u + (c[idx] - a[idx]) * v
    nrm = cross[idx] / np.maximum(np.linalg.norm(cross[idx], axis=1, keepdims=True), 1e-12)
    return pts, nrm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wtok", default=str(REPO / "runs" / "wtok_synth"))
    ap.add_argument("--out", default=str(REPO / "runs" / "mesh_synth"))
    ap.add_argument("--points", type=int, default=4096)
    ap.add_argument("--deflection", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src = sorted((pathlib.Path(args.wtok) / "parts").glob("*.json"))
    if args.limit:
        src = src[: args.limit]
    out = pathlib.Path(args.out) / "parts"
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    done, missing, failed, t0 = 0, [], [], time.time()
    for i, f in enumerate(src):
        name = f.stem
        if (out / f"{name}.npz").exists():          # resumable: new batches only
            done += 1
            continue
        mid = find_mid(name)
        if mid is None:
            missing.append(name)
            continue
        q = json.loads(f.read_text(encoding="utf-8"))["Q"]
        lo = np.asarray(q["env_lo"], dtype=np.float64)
        span = np.maximum(np.asarray(q["env_hi"], dtype=np.float64) - lo, 1e-9)
        try:
            V, T = tessellate(mid, args.deflection)
            pts, nrm = sample_surface(V, T, args.points, rng)
        except Exception as exc:                       # noqa: BLE001
            failed.append((name, str(exc)[:80]))
            continue
        if pts is None:
            failed.append((name, "no area"))
            continue
        np.savez_compressed(out / f"{name}.npz",
                            xyz=((pts - lo) / span).astype(np.float32),
                            normal=nrm.astype(np.float32),
                            env_lo=lo, env_hi=lo + span)
        done += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(src)}  ok={done} missing={len(missing)} "
                  f"failed={len(failed)}  {time.time()-t0:.0f}s", flush=True)

    print(f"done: {done} written, {len(missing)} missing, {len(failed)} failed "
          f"in {time.time()-t0:.0f}s -> {out}")
    for n, e in failed[:5]:
        print("  failed:", n, e)
    if missing[:5]:
        print("  missing:", missing[:5])


if __name__ == "__main__":
    main()

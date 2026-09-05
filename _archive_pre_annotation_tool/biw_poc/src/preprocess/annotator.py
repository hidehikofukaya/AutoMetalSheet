"""
annotator.py -- Interactive constraint-point annotator for BIW sheet metal solids.

For each constraint point placed by the user on the solid surface, the tool:
  1. Projects it to the mid-surface via ray casting through the solid
  2. Optionally annotates the partner part at the corresponding location

Coordinate frame
----------------
  P_outer            : picked surface point (on the outer face of the solid)
  N_outer            : surface normal at P_outer (outward-pointing)
  P_inner            : opposite surface point (ray cast: P_outer + depth*(-N_outer))
  P_mid              : mid-surface point = (P_outer + P_inner) / 2
  local_thickness_mm : |P_inner - P_outer|

Viewer controls
---------------
  Left-click         : add constraint point (cycles through types W/B/R/G)
  T                  : cycle constraint type (weld/bolt/rivet/glue)
  D                  : delete last point
  U                  : undo all points (clear)
  S                  : save annotations.json and quit
  Q / close window   : quit WITHOUT saving

Usage
-----
Single body (no partner):
    python annotator.py body003_filled.stp

With partner body:
    python annotator.py body003_filled.stp --partner body007_filled.stp

Load existing annotations to continue:
    python annotator.py body003_filled.stp --load body003_filled_annotations.json

Custom output:
    python annotator.py body003_filled.stp --out my_annotations.json

Output JSON schema
------------------
{
  "source_stp":  "...",
  "partner_stp": "..." | null,
  "constraints": [
    {
      "id":                   0,
      "type":                 "weld_spot",          // weld_spot | bolt | rivet | glue
      "pos_outer_xyz":        [x, y, z],            // picked point on surface
      "normal_xyz":           [nx, ny, nz],         // outward normal
      "pos_midsurface_xyz":   [x, y, z],            // projected to mid-surface
      "local_thickness_mm":   t,
      "partner_idx":          null | int,           // index in partner solid
      "partner_outer_xyz":    null | [x, y, z],
      "partner_midsurface_xyz": null | [x, y, z],
      "partner_thickness_mm": null | float,
      "note":                 ""
    },
    ...
  ]
}
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

# ── pythonocc for tessellation ────────────────────────────────────────────────
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect    import IFSelect_RetDone
from OCC.Core.BRepMesh    import BRepMesh_IncrementalMesh
from OCC.Core.BRep        import BRep_Tool
from OCC.Core.TopLoc      import TopLoc_Location
from OCC.Core.TopExp      import TopExp_Explorer
from OCC.Core.TopAbs      import TopAbs_FACE, TopAbs_FORWARD

# ─────────────────────────────────────────────────────────────────────────────
# Constraint type registry
# ─────────────────────────────────────────────────────────────────────────────

CONSTRAINT_TYPES = ["weld_spot", "bolt", "rivet", "glue"]
CONSTRAINT_COLORS = {
    "weld_spot": "orangered",
    "bolt":      "royalblue",
    "rivet":     "gold",
    "glue":      "mediumorchid",
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP -> pyvista mesh
# ─────────────────────────────────────────────────────────────────────────────

def _load_shape(stp: pathlib.Path):
    reader = STEPControl_Reader()
    if reader.ReadFile(str(stp)) != IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed: {stp}")
    reader.TransferRoots()
    return reader.OneShape()


def _tessellate(shape, linear_deflection: float = 0.15) -> pv.PolyData:
    """Tessellate an OCC shape and return a pyvista PolyData with point normals."""
    BRepMesh_IncrementalMesh(shape, linear_deflection, False, 0.5, True).Perform()

    all_pts  = []
    all_tris = []
    offset   = 0

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = exp.Current()
        loc  = TopLoc_Location()
        tri  = BRep_Tool.Triangulation(face, loc)
        if tri is not None:
            if not loc.IsIdentity():
                # Rare for CATIA STEP exports; warn and continue
                print("  WARNING: non-identity location on face -- world coords may be off")
            for i in range(1, tri.NbNodes() + 1):
                n = tri.Node(i)
                all_pts.append([n.X(), n.Y(), n.Z()])
            fwd = (face.Orientation() == TopAbs_FORWARD)
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                if fwd:
                    all_tris.append([offset + a - 1, offset + b - 1, offset + c - 1])
                else:
                    all_tris.append([offset + c - 1, offset + b - 1, offset + a - 1])
            offset += tri.NbNodes()
        exp.Next()

    verts = np.array(all_pts,  dtype=np.float64)
    tris  = np.array(all_tris, dtype=np.int64)
    faces = np.column_stack([np.full(len(tris), 3), tris]).ravel()

    mesh = pv.PolyData(verts, faces)
    mesh = mesh.clean(tolerance=0.01)
    mesh.compute_normals(cell_normals=True, point_normals=True,
                         consistent_normals=True, inplace=True)
    return mesh


def step_to_pyvista(stp: pathlib.Path, linear_deflection: float = 0.15) -> pv.PolyData:
    shape = _load_shape(stp)
    return _tessellate(shape, linear_deflection)

# ─────────────────────────────────────────────────────────────────────────────
# Mid-surface projection
# ─────────────────────────────────────────────────────────────────────────────

def project_to_midsurface(mesh: pv.PolyData,
                           P_outer: np.ndarray,
                           N_outer: np.ndarray,
                           max_depth_mm: float = 20.0
                           ) -> tuple[np.ndarray | None, float | None]:
    """
    Shoot a ray from P_outer inward (direction = -N_outer) to find the opposite
    surface of the solid, then return (P_mid, thickness_mm).

    Returns (None, None) if the ray misses the mesh interior.
    """
    eps   = 0.02          # nudge past the surface to avoid self-intersection
    start = P_outer + eps * (-N_outer)
    end   = P_outer + max_depth_mm * (-N_outer)

    pts, _ = mesh.ray_trace(start, end, first_point=False)
    if len(pts) == 0:
        return None, None

    P_inner   = pts[0]
    thickness = float(np.linalg.norm(P_inner - P_outer))
    P_mid     = (P_outer + P_inner) / 2.0
    return P_mid, thickness


def find_partner_annotation(partner_mesh: pv.PolyData,
                             partner_kdtree: KDTree,
                             partner_pts: np.ndarray,
                             partner_normals: np.ndarray,
                             P_mid: np.ndarray,
                             max_depth_mm: float = 10.0
                             ) -> dict:
    """
    Find the nearest partner surface point to P_mid, project it to the
    partner mid-surface, and return annotation metadata.
    """
    _, idx = partner_kdtree.query(P_mid)
    P_partner_outer  = partner_pts[idx]
    N_partner        = partner_normals[idx]

    # Make sure normal points outward (away from P_mid)
    if np.dot(N_partner, P_partner_outer - P_mid) < 0:
        N_partner = -N_partner

    P_partner_mid, t_partner = project_to_midsurface(
        partner_mesh, P_partner_outer, N_partner, max_depth_mm=max_depth_mm
    )

    return {
        "partner_outer_xyz":      P_partner_outer.tolist(),
        "partner_midsurface_xyz": P_partner_mid.tolist() if P_partner_mid is not None else None,
        "partner_thickness_mm":   round(t_partner, 3) if t_partner is not None else None,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Interactive annotator
# ─────────────────────────────────────────────────────────────────────────────

class Annotator:
    def __init__(self,
                 stp_path: pathlib.Path,
                 partner_path: pathlib.Path | None = None,
                 initial_annotations: list | None = None,
                 linear_deflection: float = 0.15):

        self.stp_path    = stp_path
        self.partner_path = partner_path
        self.annotations  = list(initial_annotations or [])
        self._next_id     = max((a["id"] for a in self.annotations), default=-1) + 1
        self._ctype_idx   = 0

        print(f"Tessellating {stp_path.name} ...", flush=True)
        self.mesh    = step_to_pyvista(stp_path, linear_deflection)
        self._pts    = np.asarray(self.mesh.points)
        self._normals = np.asarray(self.mesh.point_normals)
        self._kdtree  = KDTree(self._pts)

        self.partner_mesh    = None
        self._partner_pts    = None
        self._partner_normals = None
        self._partner_kdtree  = None
        if partner_path is not None:
            print(f"Tessellating partner {partner_path.name} ...", flush=True)
            self.partner_mesh      = step_to_pyvista(partner_path, linear_deflection)
            self._partner_pts      = np.asarray(self.partner_mesh.points)
            self._partner_normals  = np.asarray(self.partner_mesh.point_normals)
            self._partner_kdtree   = KDTree(self._partner_pts)

        self.plotter = None

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def current_type(self) -> str:
        return CONSTRAINT_TYPES[self._ctype_idx % len(CONSTRAINT_TYPES)]

    # ── viewer construction ───────────────────────────────────────────────────

    def _build_plotter(self):
        pl = pv.Plotter(title=f"Annotator  |  {self.stp_path.name}")

        # Main solid (semi-transparent)
        pl.add_mesh(self.mesh, color="lightsteelblue", opacity=0.7,
                    show_edges=False, name="main_mesh")

        # Partner solid (if given)
        if self.partner_mesh is not None:
            pl.add_mesh(self.partner_mesh, color="lightgreen", opacity=0.4,
                        show_edges=False, name="partner_mesh")

        # Instructions overlay
        instr = (
            "Left-click : add point   T : cycle type   D : delete last\n"
            "U : clear all   S : save & quit   Q : quit without saving\n"
            f"Current type: [{self.current_type}]"
        )
        pl.add_text(instr, position="lower_left", font_size=9,
                    color="white", name="instructions")

        # Draw any pre-loaded annotations
        for ann in self.annotations:
            self._draw_annotation(pl, ann)

        # Enable surface point picking
        pl.enable_surface_point_picking(
            callback=self._on_pick,
            show_message=False,
            show_point=False,          # we draw our own spheres
            picker="cell",
        )

        # Key bindings
        pl.add_key_event("t", self._cycle_type)
        pl.add_key_event("T", self._cycle_type)
        pl.add_key_event("d", self._delete_last)
        pl.add_key_event("D", self._delete_last)
        pl.add_key_event("u", self._clear_all)
        pl.add_key_event("U", self._clear_all)
        pl.add_key_event("s", self._save_and_quit)
        pl.add_key_event("S", self._save_and_quit)

        self.plotter = pl

    def _draw_annotation(self, pl, ann: dict):
        ctype   = ann.get("type", "weld_spot")
        color   = CONSTRAINT_COLORS.get(ctype, "red")
        p_outer = np.array(ann["pos_outer_xyz"])
        p_mid   = ann.get("pos_midsurface_xyz")
        aid     = ann["id"]

        # Sphere at surface pick point
        pl.add_mesh(
            pv.Sphere(center=p_outer, radius=1.5),
            color=color, name=f"ann_{aid}_outer"
        )
        # Sphere at mid-surface
        if p_mid is not None:
            pl.add_mesh(
                pv.Sphere(center=p_mid, radius=1.0),
                color="white", name=f"ann_{aid}_mid"
            )
            # Line outer -> mid
            line = pv.Line(p_outer, np.array(p_mid))
            pl.add_mesh(line, color=color, line_width=2, name=f"ann_{aid}_line")

        # Label
        pl.add_point_labels(
            [p_outer],
            [f"{aid}:{ctype[:1].upper()}"],
            font_size=9,
            shape_opacity=0.5,
            name=f"ann_{aid}_label",
        )

        # Partner surface point
        if ann.get("partner_outer_xyz") is not None:
            pp = np.array(ann["partner_outer_xyz"])
            pl.add_mesh(
                pv.Sphere(center=pp, radius=1.5),
                color="yellow", name=f"ann_{aid}_partner"
            )

    def _refresh_instructions(self):
        if self.plotter is None:
            return
        instr = (
            "Left-click : add point   T : cycle type   D : delete last\n"
            "U : clear all   S : save & quit   Q : quit without saving\n"
            f"Current type: [{self.current_type}]   Points: {len(self.annotations)}"
        )
        self.plotter.add_text(instr, position="lower_left", font_size=9,
                              color="white", name="instructions")

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _on_pick(self, picked_pt):
        """Called by pyvista for every left-click on the surface."""
        if picked_pt is None:
            return

        P_outer = np.asarray(picked_pt, dtype=np.float64)

        # Nearest mesh point -> outward normal
        _, idx = self._kdtree.query(P_outer)
        N_outer = self._normals[idx].copy()
        N_outer /= (np.linalg.norm(N_outer) + 1e-12)

        # Mid-surface projection
        P_mid, thickness = project_to_midsurface(self.mesh, P_outer, N_outer)

        ann = {
            "id":                   self._next_id,
            "type":                 self.current_type,
            "pos_outer_xyz":        P_outer.tolist(),
            "normal_xyz":           N_outer.tolist(),
            "pos_midsurface_xyz":   P_mid.tolist() if P_mid is not None else None,
            "local_thickness_mm":   round(thickness, 3) if thickness is not None else None,
            "partner_outer_xyz":    None,
            "partner_midsurface_xyz": None,
            "partner_thickness_mm": None,
            "note":                 "",
        }

        # Partner auto-annotation
        if self.partner_mesh is not None and P_mid is not None:
            partner_ann = find_partner_annotation(
                self.partner_mesh,
                self._partner_kdtree,
                self._partner_pts,
                self._partner_normals,
                P_mid,
            )
            ann.update(partner_ann)

        self.annotations.append(ann)
        self._next_id += 1

        t_str = f"{thickness:.2f}mm" if thickness is not None else "?"
        print(f"  [{ann['id']}] {self.current_type}  outer={[round(v,1) for v in P_outer]}  "
              f"t={t_str}", flush=True)

        self._draw_annotation(self.plotter, ann)
        self._refresh_instructions()

    def _cycle_type(self):
        self._ctype_idx += 1
        print(f"  Constraint type -> {self.current_type}")
        self._refresh_instructions()

    def _delete_last(self):
        if not self.annotations:
            print("  Nothing to delete.")
            return
        ann = self.annotations.pop()
        aid = ann["id"]
        for suffix in ("_outer", "_mid", "_line", "_label", "_partner"):
            self.plotter.remove_actor(f"ann_{aid}{suffix}", render=False)
        self.plotter.render()
        print(f"  Deleted annotation {aid}")
        self._refresh_instructions()

    def _clear_all(self):
        for ann in list(self.annotations):
            aid = ann["id"]
            for suffix in ("_outer", "_mid", "_line", "_label", "_partner"):
                self.plotter.remove_actor(f"ann_{aid}{suffix}", render=False)
        self.annotations.clear()
        self.plotter.render()
        print("  All annotations cleared.")
        self._refresh_instructions()

    def _save_and_quit(self):
        self.plotter.close()

    # ── public API ────────────────────────────────────────────────────────────

    def run(self) -> list:
        """Open the interactive viewer. Returns the annotations list on close."""
        self._build_plotter()
        self.plotter.show()
        return self.annotations

    def export(self, out_path: pathlib.Path):
        """Write annotations to JSON."""
        payload = {
            "source_stp":  str(self.stp_path),
            "partner_stp": str(self.partner_path) if self.partner_path else None,
            "constraints": self.annotations,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nSaved {len(self.annotations)} annotation(s) -> {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Interactive constraint-point annotator for BIW sheet metal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("stp",           help="body###_filled.stp to annotate")
    ap.add_argument("--partner",     metavar="STP",
                    help="Partner body###_filled.stp for auto-annotation")
    ap.add_argument("--load",        metavar="JSON",
                    help="Existing annotations JSON to continue editing")
    ap.add_argument("--out",         metavar="JSON",
                    help="Output path (default: <stp_stem>_annotations.json "
                         "alongside the STP file)")
    ap.add_argument("--deflection",  type=float, default=0.15,
                    help="Tessellation linear deflection mm (default 0.15)")
    args = ap.parse_args()

    stp_path = pathlib.Path(args.stp)
    if not stp_path.exists():
        print(f"ERROR: not found: {stp_path}")
        sys.exit(1)

    partner_path = pathlib.Path(args.partner) if args.partner else None
    if partner_path and not partner_path.exists():
        print(f"ERROR: partner not found: {partner_path}")
        sys.exit(1)

    # Default output path
    out_path = (
        pathlib.Path(args.out)
        if args.out
        else stp_path.parent / (stp_path.stem + "_annotations.json")
    )

    # Load existing annotations
    initial = []
    if args.load:
        load_path = pathlib.Path(args.load)
        if load_path.exists():
            data    = json.loads(load_path.read_text("utf-8"))
            initial = data.get("constraints", [])
            print(f"Loaded {len(initial)} existing annotation(s) from {load_path.name}")
        else:
            print(f"WARNING: --load file not found: {load_path}")

    annotator = Annotator(
        stp_path,
        partner_path=partner_path,
        initial_annotations=initial,
        linear_deflection=args.deflection,
    )

    annotations = annotator.run()

    if annotations:
        annotator.export(out_path)
    else:
        print("No annotations to save.")


if __name__ == "__main__":
    main()

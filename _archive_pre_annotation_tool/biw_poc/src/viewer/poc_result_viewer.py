"""
poc_result_viewer.py -- Stage A/B/C PoC result viewer (independently toggleable layers)

Extends the original GT-vs-one-recon viewer (CLAUDE.md section 13/14) to show
ALL pipeline intermediate outputs side by side, each independently toggleable:
  - GT mid-surface mesh
  - every Stage C reconstruction output discovered in --recon-dir, including
    the R7-08 intermediate stages (coarse pre-refinement mesh, and the mesh
    after each subdivide_and_project refinement round -- see reconstruct.py's
    --save-intermediate flag)
  - Stage A's noisy training input point cloud (the h5 "points" dataset)
  - Stage A's UDF query points, split by query_category (R7-09: near / far /
    boundary-shadow) so the boundary-shadow oversampling region used for the
    TD-16 fix is directly visible

Any two MESH layers (GT or a recon/stage variant) can be picked as the
Reference/Target pair for the distance heatmap + stats, generalizing the
original GT<->recon comparison to also compare e.g. a coarse vs. a refined
stage of the same reconstruction.

Layout
------
 ┌────────────────────┬──────────────────────────────────────────┐
 │ Layers:             │                                           │
 │ ☑ GT: body002…      │                                           │
 │ ☐ recon_midsurface  │               3D Viewport                │
 │ ☐ recon_stage_coarse│   Target layer: per-vertex distance       │
 │ ☐ recon_stage_refine1│  heatmap to the Reference layer          │
 │ ☐ Input points (Stage A)                                       │
 │ ☐ Query: near                                                   │
 │ ☐ Query: far                                                    │
 │ ☐ Query: boundary-shadow (R7-09)                                │
 │ ──────────────────                                             │
 │ Distance compare:                                               │
 │ Reference [GT v]                                                │
 │ Target    [recon v]                                             │
 │ Color range [...]                                               │
 │ ──────────────────                                             │
 │ mean   ...mm                                                    │
 │ median ...mm                                                    │
 │ p90    ...mm                                                    │
 │ max    ...mm                                                    │
 └────────────────────┴──────────────────────────────────────────┘

Distance methodology unchanged: nearest-vertex distance via scipy cKDTree
(matches validate_refine.py / CLAUDE.md R7-07/R7-08 diagnostics).

Usage
-----
    python poc_result_viewer.py
    python poc_result_viewer.py --gt path/to/..._midsurface.ply --recon-dir path/to/dir --h5 path/to/..._dataset.h5
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import h5py
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QFrame, QGroupBox, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMainWindow, QSplitter, QStatusBar,
    QVBoxLayout, QWidget,
)
from scipy.spatial import cKDTree

# ─────────────────────────────────────────────────────────────────────────────
# Defaults (CLAUDE.md section 13/14: body002 is the standing single-part
# overfit sanity benchmark used throughout Stage A/B/C development)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_GT = pathlib.Path(
    r"C:\Users\hide2\IdeaBox\Mesh_Generater\output\A0072600002\bodies\body002_filled_midsurface.ply")
DEFAULT_RECON_DIR = pathlib.Path(__file__).parent.parent / "model"
DEFAULT_H5 = pathlib.Path(
    r"C:\Users\hide2\IdeaBox\Mesh_Generater\output\A0072600002\bodies\body002_filled_dataset.h5")

COLOR_RANGE_PRESETS = {
    "0-2mm (interior, R7-08 target)":  (0.0, 2.0),
    "0-5mm (interior detail)":         (0.0, 5.0),
    "0-20mm":                          (0.0, 20.0),
    "0-max (full range, boundary)":    None,   # resolved per-field at draw time
}

GT_COLOR = (0.85, 0.85, 0.85)
RECON_COLOR = (0.25, 0.75, 1.0)
STAGE_COLOR = (1.0, 0.85, 0.2)
INPUT_POINTS_COLOR = (0.15, 1.0, 0.45)
# R7-09: query_category values must match midsurface_sampler.py
QUERY_CATEGORY_COLOR = {0: (0.25, 0.55, 1.0), 1: (0.55, 0.55, 0.55), 2: (1.0, 0.45, 0.05)}
QUERY_CATEGORY_LABEL = {0: "Query: near", 1: "Query: far", 2: "Query: boundary-shadow (R7-09)"}


# ─────────────────────────────────────────────────────────────────────────────
# Distance computation (matches validate_refine.py / CLAUDE.md R7-07/R7-08
# root-cause diagnostics: nearest-vertex distance via cKDTree)
# ─────────────────────────────────────────────────────────────────────────────

def per_vertex_distance(src: pv.PolyData, dst: pv.PolyData) -> np.ndarray:
    """For each vertex of *src*, distance to the nearest vertex of *dst*."""
    tree = cKDTree(dst.points)
    d, _ = tree.query(src.points, k=1)
    return d.astype(np.float32)


def distance_stats(d: np.ndarray) -> dict:
    return dict(mean=float(d.mean()), median=float(np.median(d)),
                p90=float(np.percentile(d, 90)), max=float(d.max()))


# ─────────────────────────────────────────────────────────────────────────────
# Layers
# ─────────────────────────────────────────────────────────────────────────────

class Layer:
    """One independently-toggleable viewer layer. `loader` is called lazily
    (once, then cached) and must return a pv.PolyData -- a face-having mesh
    for kind='mesh', or a faceless point-cloud PolyData for kind='points'."""

    def __init__(self, key: str, label: str, kind: str, color: tuple, loader):
        self.key = key
        self.label = label
        self.kind = kind          # 'mesh' or 'points'
        self.color = color
        self.loader = loader
        self._cache: pv.PolyData | None = None
        self._load_error: str | None = None

    def get(self) -> pv.PolyData | None:
        if self._cache is None and self._load_error is None:
            try:
                self._cache = self.loader()
            except Exception as exc:  # surfaced in the status bar by the caller
                self._load_error = str(exc)
        return self._cache


def discover_recon_files(recon_dir: pathlib.Path, gt_path: pathlib.Path) -> list[pathlib.Path]:
    files = sorted(recon_dir.glob("*_midsurface.ply"))
    return [f for f in files if f.resolve() != gt_path.resolve()]


def build_layers(gt_path: pathlib.Path, recon_files: list[pathlib.Path],
                  h5_path: pathlib.Path | None) -> list[Layer]:
    layers = []

    if gt_path.exists():
        layers.append(Layer("gt", f"GT: {gt_path.name}", "mesh", GT_COLOR,
                             lambda p=gt_path: pv.read(str(p))))

    for f in recon_files:
        is_stage = "_stage_" in f.name
        color = STAGE_COLOR if is_stage else RECON_COLOR
        layers.append(Layer(str(f), f.name, "mesh", color,
                             lambda p=f: pv.read(str(p))))

    if h5_path is not None and h5_path.exists():
        def load_points(p=h5_path):
            with h5py.File(str(p), "r") as hf:
                pts = hf["points"][:]
            return pv.PolyData(pts)
        layers.append(Layer("h5_points", "Input points (Stage A)", "points",
                             INPUT_POINTS_COLOR, load_points))

        for cat, label in QUERY_CATEGORY_LABEL.items():
            def load_query(p=h5_path, cat=cat):
                with h5py.File(str(p), "r") as hf:
                    xyz = hf["query_xyz"][:]
                    category = hf["query_category"][:]
                return pv.PolyData(xyz[category == cat])
            layers.append(Layer(f"h5_query_{cat}", label, "points",
                                 QUERY_CATEGORY_COLOR[cat], load_query))

    return layers


# ─────────────────────────────────────────────────────────────────────────────
# Left panel
# ─────────────────────────────────────────────────────────────────────────────

class ControlPanel(QWidget):
    def __init__(self, layers: list[Layer], on_change):
        super().__init__()
        self._layers = layers
        self._on_change = on_change
        self._build()

    def _build(self):
        mesh_layers = [l for l in self._layers if l.kind == "mesh"]
        non_gt_meshes = [l for l in mesh_layers if l.key != "gt"]
        final_candidates = [l for l in non_gt_meshes if "_stage_" not in l.label]
        default_target = (final_candidates[-1] if final_candidates
                           else (non_gt_meshes[-1] if non_gt_meshes else None))
        default_checked = {l.key for l in (mesh_layers[:1] if mesh_layers else [])}
        if default_target is not None:
            default_checked.add(default_target.key)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        lay.addWidget(QLabel("<b>Layers</b> (independently toggleable):"))
        self._list = QListWidget()
        for layer in self._layers:
            item = QListWidgetItem(layer.label)
            item.setData(Qt.UserRole, layer.key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if layer.key in default_checked else Qt.Unchecked)
            self._list.addItem(item)
        self._list.itemChanged.connect(lambda *_: self._on_change())
        lay.addWidget(self._list, stretch=1)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine); lay.addWidget(sep)

        dist_box = QGroupBox("Distance compare (mesh layers only)")
        dist_lay = QVBoxLayout(dist_box)

        dist_lay.addWidget(QLabel("Reference:"))
        self._reference = QComboBox()
        for l in mesh_layers:
            self._reference.addItem(l.label, l.key)
        dist_lay.addWidget(self._reference)

        dist_lay.addWidget(QLabel("Target (colored by distance-to-reference):"))
        self._target = QComboBox()
        for l in mesh_layers:
            self._target.addItem(l.label, l.key)
        if default_target is not None:
            idx = next(i for i in range(self._target.count())
                       if self._target.itemData(i) == default_target.key)
            self._target.setCurrentIndex(idx)
        dist_lay.addWidget(self._target)

        self._reference.currentIndexChanged.connect(lambda *_: self._on_change())
        self._target.currentIndexChanged.connect(lambda *_: self._on_change())

        dist_lay.addWidget(QLabel("Color range:"))
        self._color_range = QComboBox()
        self._color_range.addItems(list(COLOR_RANGE_PRESETS.keys()))
        self._color_range.currentIndexChanged.connect(lambda *_: self._on_change())
        dist_lay.addWidget(self._color_range)

        lay.addWidget(dist_box)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine); lay.addWidget(sep2)
        self._stats_label = QLabel("(no data)")
        self._stats_label.setWordWrap(True)
        self._stats_label.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 11px; padding: 4px;")
        lay.addWidget(self._stats_label)

        lay.addStretch(1)

    # ── accessors ─────────────────────────────────────────────────────────────

    def checked_keys(self) -> set[str]:
        keys = set()
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.checkState() == Qt.Checked:
                keys.add(item.data(Qt.UserRole))
        return keys

    def reference_key(self) -> str | None:
        return self._reference.currentData()

    def target_key(self) -> str | None:
        return self._target.currentData()

    def color_clim(self, field: np.ndarray) -> tuple[float, float]:
        preset = COLOR_RANGE_PRESETS[self._color_range.currentText()]
        if preset is not None:
            return preset
        return (0.0, float(field.max()) if len(field) else 1.0)

    def set_stats(self, stats: dict | None, label: str):
        if stats is None:
            self._stats_label.setText(f"<b>{label}</b>")
            return
        self._stats_label.setText(
            f"<b>{label}</b><br>"
            f"mean:   {stats['mean']:7.2f} mm<br>"
            f"median: {stats['median']:7.2f} mm<br>"
            f"p90:    {stats['p90']:7.2f} mm<br>"
            f"max:    {stats['max']:7.2f} mm"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, layers: list[Layer]):
        super().__init__()
        self._layers = {l.key: l for l in layers}

        self.setWindowTitle("PoC Result Viewer -- Stage A/B/C layers (GT / recon / intermediate)")
        self.resize(1500, 850)
        self._build_ui(layers)
        self._status("Ready." if layers else "WARNING: no layers found.")
        self._refresh()

    def _build_ui(self, layers):
        central = QWidget()
        self.setCentralWidget(central)
        main_lay = QHBoxLayout(central)
        main_lay.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        main_lay.addWidget(splitter)

        left = QWidget()
        left.setMaximumWidth(360)
        left.setMinimumWidth(280)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        self._panel = ControlPanel(layers, on_change=self._refresh)
        left_lay.addWidget(self._panel)
        splitter.addWidget(left)

        self._plotter = QtInteractor(self)
        self._plotter.set_background("black")
        splitter.addWidget(self._plotter)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)

    def _status(self, msg: str):
        self._statusbar.showMessage(msg)

    def _refresh(self):
        checked = self._panel.checked_keys()
        if not checked:
            self._plotter.clear()
            self._panel.set_stats(None, "(no layers shown)")
            self._status("Nothing checked.")
            return

        self._status("Loading…")
        QApplication.processEvents()

        ref_key = self._panel.reference_key()
        tgt_key = self._panel.target_key()
        ref_layer = self._layers.get(ref_key)
        tgt_layer = self._layers.get(tgt_key)

        field = None
        clim = None
        if ref_layer is not None and tgt_layer is not None and ref_key != tgt_key:
            ref_mesh, tgt_mesh = ref_layer.get(), tgt_layer.get()
            if ref_mesh is not None and tgt_mesh is not None:
                field = per_vertex_distance(tgt_mesh, ref_mesh)
                clim = self._panel.color_clim(field)
                stats = distance_stats(field)
                self._panel.set_stats(stats, f"{tgt_layer.label}  ->  {ref_layer.label}")
            else:
                self._panel.set_stats(None, "(layer failed to load, see status bar)")
        else:
            self._panel.set_stats(None, "(pick two distinct mesh layers to compare)")

        self._plotter.clear()
        n_shown, errors = 0, []
        for key in checked:
            layer = self._layers.get(key)
            if layer is None:
                continue
            mesh = layer.get()
            if mesh is None:
                if layer._load_error:
                    errors.append(f"{layer.label}: {layer._load_error}")
                continue
            if mesh.n_points == 0:
                continue
            n_shown += 1
            if layer.kind == "points":
                self._plotter.add_mesh(
                    mesh, color=layer.color, point_size=4, render_points_as_spheres=True,
                    name=f"layer_{key}",
                )
            elif key == tgt_key and field is not None:
                colored = mesh.copy()
                colored["dist_mm"] = field
                self._plotter.add_mesh(
                    colored, scalars="dist_mm", cmap="turbo", clim=clim,
                    scalar_bar_args={"title": "distance (mm)"},
                    smooth_shading=False, show_edges=False, name=f"layer_{key}",
                )
            else:
                self._plotter.add_mesh(
                    mesh, color=layer.color, opacity=0.3 if key == ref_key else 1.0,
                    smooth_shading=False, show_edges=False, name=f"layer_{key}",
                )

        self._plotter.reset_camera()
        msg = f"{n_shown} layer(s) shown."
        if errors:
            msg += "  ERRORS: " + "; ".join(errors)
        self._status(msg)

    def closeEvent(self, event):
        self._plotter.close()
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", default=str(DEFAULT_GT), help="GT mid-surface .ply")
    ap.add_argument("--recon-dir", default=str(DEFAULT_RECON_DIR),
                     help="Directory to scan for *_midsurface.ply outputs (final + "
                          "--save-intermediate stage files from reconstruct.py)")
    ap.add_argument("--h5", default=str(DEFAULT_H5),
                     help="Stage A *_dataset.h5 file, to show the input point cloud and "
                          "UDF query points by category (R7-09). Pass an empty string "
                          "to disable these layers.")
    args = ap.parse_args()

    gt_path = pathlib.Path(args.gt)
    recon_dir = pathlib.Path(args.recon_dir)
    h5_path = pathlib.Path(args.h5) if args.h5 else None
    recon_files = discover_recon_files(recon_dir, gt_path)

    if not gt_path.exists():
        print(f"WARNING: GT file not found: {gt_path}")
    if not recon_files:
        print(f"WARNING: no *_midsurface.ply recon files found in {recon_dir}")
    if h5_path is not None and not h5_path.exists():
        print(f"WARNING: h5 file not found, point-cloud layers disabled: {h5_path}")
        h5_path = None

    layers = build_layers(gt_path, recon_files, h5_path)

    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow(layers)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

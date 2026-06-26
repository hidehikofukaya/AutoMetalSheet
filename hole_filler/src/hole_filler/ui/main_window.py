"""Hole Filler main window.

Layout
------
Left  : part tree (reused from annotation_tool) + hole list + action buttons
Right : PyVista 3D viewer + view-toggle toolbar

Sphere colour key — 元の形状 (original) view
--------------------------------------------
Gold  (★) : matches a bolt/mounting_hole joint in joints.json
Green (●) : selected for fill (default)
Red   (○) : deselected — will NOT be filled

Sphere colour key — 穴埋め後 (after-fill) view
-----------------------------------------------
Cyan  (◎) : confirmed filled in output mesh
Red   (○) : will remain open in output mesh
"""
from __future__ import annotations

import pathlib

import numpy as np
import pyvista as pv
import vtk
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
from scipy.spatial import cKDTree

from annotation_tool.assembly import AssemblyStore
from annotation_tool.hierarchy import HierarchyBody
from annotation_tool.mesh_render import detect_holes
from annotation_tool.schema import AnnotationDocument
from annotation_tool.ui.part_tree import PartTreePanel

from hole_filler.filler import create_patched_output, save_result
from hole_filler.ui.hole_list_panel import HoleListPanel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PART_COLOR = (0.62, 0.65, 0.70)
PART_EDGE_COLOR = (0.18, 0.18, 0.18)

SPHERE_SELECTED_COLOR = (0.15, 0.80, 0.25)   # green  — original view, will be filled
SPHERE_ANNOTATED_COLOR = (1.00, 0.75, 0.05)  # gold   — original view, annotated + fill
SPHERE_DESELECTED_COLOR = (0.85, 0.15, 0.15) # red    — both views, will NOT be filled
SPHERE_FILLED_COLOR = (0.10, 0.65, 0.95)     # cyan   — after-fill view, confirmed filled

# Radius of the hole indicator sphere in the 3-D view (mm)
_SPHERE_BASE_R = 3.0
_SPHERE_SCALE = 0.35


# ---------------------------------------------------------------------------
# HoleFillerWindow
# ---------------------------------------------------------------------------

class HoleFillerWindow(QMainWindow):
    """Single-part hole filler with interactive hole selection."""

    def __init__(self, store: AssemblyStore, assembly_dir: pathlib.Path) -> None:
        super().__init__()
        self._store = store
        self._assembly_dir = assembly_dir

        # Current state
        self._body: HierarchyBody | None = None
        self._mesh: pv.PolyData | None = None    # original closed VTP
        self._filled: pv.PolyData | None = None  # output mesh after user selection
        self._holes: list[dict] = []
        self._selected: set[int] = set()
        self._annotated: set[int] = set()
        self._show_filled = False
        self._camera_needs_reset = False         # reset only on new part load

        # 3-D actor tracking: hole_index → vtkActor
        self._hole_actors: dict[int, vtk.vtkActor] = {}
        self._active_actors: set[str] = set()

        # VTK click picking state
        self._vtk_picker = vtk.vtkPropPicker()
        self._click_press_pos: tuple[int, int] | None = None

        self.setWindowTitle(f"Hole Filler — {assembly_dir.name}")
        self.resize(1400, 820)
        self._build_ui()
        self._setup_picking()
        self._status(f"{len(store.visible_bodies)} 部品を読み込みました。左のツリーから部品を選択してください。")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ── Left panel ──────────────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(240)
        left.setMaximumWidth(400)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)

        self._tree = PartTreePanel(
            self._store,
            on_select=self._on_part_selected,
        )
        ll.addWidget(self._tree, 1)

        # Hole list group
        hole_box = QGroupBox("検出穴")
        hl = QVBoxLayout(hole_box)
        hl.setSpacing(4)

        self._hole_panel = HoleListPanel(on_toggle=self._on_hole_toggled)
        hl.addWidget(self._hole_panel)

        sel_row = QHBoxLayout()
        sel_all = QPushButton("全選択")
        sel_all.clicked.connect(self._hole_panel.select_all)
        desel_all = QPushButton("全解除")
        desel_all.clicked.connect(self._hole_panel.deselect_all)
        sel_row.addWidget(sel_all)
        sel_row.addWidget(desel_all)
        hl.addLayout(sel_row)
        ll.addWidget(hole_box)

        # Action buttons
        act_box = QGroupBox("操作")
        al = QVBoxLayout(act_box)
        al.setSpacing(6)

        self._fill_btn = QPushButton("▶  穴埋め実行")
        self._fill_btn.setEnabled(False)
        self._fill_btn.setMinimumHeight(36)
        self._fill_btn.clicked.connect(self._on_fill_clicked)
        self._fill_btn.setStyleSheet(
            "QPushButton { background:#2a7a3c; color:white; font-weight:bold; border-radius:4px; }"
            "QPushButton:disabled { background:#555; color:#888; }"
        )
        al.addWidget(self._fill_btn)

        self._save_btn = QPushButton("💾  保存 (filled/ ディレクトリ)")
        self._save_btn.setEnabled(False)
        self._save_btn.setMinimumHeight(32)
        self._save_btn.clicked.connect(self._on_save_clicked)
        al.addWidget(self._save_btn)

        ll.addWidget(act_box)

        # ── Right panel (viewer + toggle) ────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        self._plotter = QtInteractor(right)
        self._plotter.set_background("white")
        rl.addWidget(self._plotter.interactor, 1)

        # View toggle bar
        toggle_bar = QWidget()
        tl = QHBoxLayout(toggle_bar)
        tl.setContentsMargins(6, 4, 6, 4)
        tl.setSpacing(8)

        tl.addWidget(QLabel("表示:"))
        self._orig_btn = QPushButton("元の形状")
        self._orig_btn.setCheckable(True)
        self._orig_btn.setChecked(True)
        self._orig_btn.clicked.connect(lambda: self._set_view(False))
        tl.addWidget(self._orig_btn)

        self._after_btn = QPushButton("穴埋め後")
        self._after_btn.setCheckable(True)
        self._after_btn.setEnabled(False)
        self._after_btn.clicked.connect(lambda: self._set_view(True))
        tl.addWidget(self._after_btn)

        tl.addStretch()
        self._view_label = QLabel("")
        tl.addWidget(self._view_label)

        rl.addWidget(toggle_bar)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([300, 1100])

        status = QStatusBar()
        self.setStatusBar(status)
        self._status_label = QLabel()
        status.addWidget(self._status_label)

    # ------------------------------------------------------------------
    # 3-D picking setup
    # ------------------------------------------------------------------

    def _setup_picking(self) -> None:
        iren = self._plotter.iren.interactor
        iren.AddObserver("LeftButtonPressEvent", self._on_vtk_press, 1.0)
        iren.AddObserver("LeftButtonReleaseEvent", self._on_vtk_release, 1.0)

    def _on_vtk_press(self, caller, _event) -> None:
        self._click_press_pos = caller.GetEventPosition()

    def _on_vtk_release(self, caller, _event) -> None:
        if self._click_press_pos is None:
            return
        pos = caller.GetEventPosition()
        dx = abs(pos[0] - self._click_press_pos[0])
        dy = abs(pos[1] - self._click_press_pos[1])
        self._click_press_pos = None
        if dx > 5 or dy > 5:
            return  # drag → camera move, ignore
        self._vtk_picker.Pick(pos[0], pos[1], 0, self._plotter.renderer)
        actor = self._vtk_picker.GetActor()
        if actor is None:
            return
        for idx, ha in self._hole_actors.items():
            if actor is ha:
                self._toggle_hole(idx)
                return

    # ------------------------------------------------------------------
    # Part selection
    # ------------------------------------------------------------------

    def _on_part_selected(self, body: HierarchyBody) -> None:
        self._body = body
        self._filled = None
        self._show_filled = False
        self._after_btn.setEnabled(False)
        self._orig_btn.setChecked(True)
        self._after_btn.setChecked(False)
        self._save_btn.setEnabled(False)
        self._view_label.setText("")

        self._status(f"{body.part_id} を読み込み中…")
        self._mesh = self._store.load_mesh(body)

        self._status(f"{body.part_id} の穴を検出中…")
        self._holes = detect_holes(self._mesh)
        self._annotated = self._find_annotated(body.part_id)
        self._selected = set(range(len(self._holes)))

        self._hole_panel.set_holes(self._holes, self._selected, self._annotated)
        self._fill_btn.setEnabled(bool(self._holes))
        self._camera_needs_reset = True
        self._render()

        n = len(self._holes)
        ann = len(self._annotated)
        self._status(
            f"{body.part_id}: {n} 穴検出"
            + (f"（うち ★{ann} 個がアノテ済み）" if ann else "")
            + "。穴埋めしたくない穴のチェックを外してから「穴埋め実行」を押してください。"
        )

    # ------------------------------------------------------------------
    # Annotation matching
    # ------------------------------------------------------------------

    def _find_annotated(self, part_id: str) -> set[int]:
        """Return indices of holes that match bolt/mounting_hole joints."""
        if not self._holes:
            return set()
        joints_path = self._assembly_dir / "annotations" / "joints.json"
        if not joints_path.exists():
            return set()
        try:
            doc = AnnotationDocument.load(self._assembly_dir)
        except Exception:
            return set()

        known: list[np.ndarray] = []
        for jt in doc.joints.values():
            if jt.type not in ("bolt", "mounting_hole"):
                continue
            for pr in jt.per_part:
                if pr.part_id == part_id and pr.hole_center_xyz is not None:
                    known.append(np.array(pr.hole_center_xyz, dtype=float))

        if not known:
            return set()

        ann_tree = cKDTree(np.array(known, dtype=float))
        hole_ctrs = np.array([h["center"] for h in self._holes], dtype=float)

        matched: set[int] = set()
        for i, c in enumerate(hole_ctrs):
            d, _ = ann_tree.query(c)
            tol = self._holes[i]["radius_mm"] * 0.5 + 3.0
            if d < tol:
                matched.add(i)
        return matched

    # ------------------------------------------------------------------
    # Hole toggle
    # ------------------------------------------------------------------

    def _on_hole_toggled(self, idx: int, is_selected: bool) -> None:
        if is_selected:
            self._selected.add(idx)
        else:
            self._selected.discard(idx)
        self._update_sphere(idx)

    def _toggle_hole(self, idx: int) -> None:
        """Toggle hole selection from 3-D click."""
        is_sel = idx not in self._selected
        if is_sel:
            self._selected.add(idx)
        else:
            self._selected.discard(idx)
        self._update_sphere(idx)
        self._hole_panel.update_selection(idx, is_sel)

    # ------------------------------------------------------------------
    # Fill & save
    # ------------------------------------------------------------------

    def _on_fill_clicked(self) -> None:
        if self._mesh is None:
            return
        n_sel = len(self._selected)
        n_skip = len(self._holes) - n_sel
        self._status(f"穴埋め処理中… ({n_sel} 穴をパッチ適用)")
        self._filled = create_patched_output(
            self._mesh, self._holes, self._selected
        )
        self._show_filled = True
        self._after_btn.setEnabled(True)
        self._orig_btn.setChecked(False)
        self._after_btn.setChecked(True)
        self._save_btn.setEnabled(True)
        self._render()
        self._status(
            f"穴埋め完了 — パッチ適用:{n_sel} 個  スキップ:{n_skip} 個。"
            "「保存」で filled/ ディレクトリに出力します。"
        )

    def _on_save_clicked(self) -> None:
        if self._filled is None or self._body is None:
            return
        try:
            vtp, js = save_result(
                self._assembly_dir,
                self._body.part_id,
                self._filled,
                self._holes,
                self._selected,
            )
        except Exception as e:
            QMessageBox.critical(self, "保存エラー", str(e))
            return
        self._status(f"保存完了 → {vtp.name}  /  {js.name}")

    # ------------------------------------------------------------------
    # View toggle
    # ------------------------------------------------------------------

    def _set_view(self, show_filled: bool) -> None:
        if show_filled and self._filled is None:
            return
        self._show_filled = show_filled
        self._orig_btn.setChecked(not show_filled)
        self._after_btn.setChecked(show_filled)
        self._render()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        """Re-draw mesh and hole spheres from scratch (named-actor swap).

        Both views show the same closed source mesh; the distinction is sphere colour:
        - 元の形状 (original): green/gold = will be filled, red = will be left open
        - 穴埋め後 (after fill): cyan = confirmed filled, red = remains open

        Camera is reset only on the first render of each newly loaded part.
        """
        # Remove old actors
        for name in list(self._active_actors):
            self._plotter.remove_actor(name, render=False)
        self._active_actors.clear()
        self._hole_actors.clear()

        mesh_to_show = self._filled if self._show_filled else self._mesh
        if mesh_to_show is None:
            self._plotter.render()
            return

        # Mesh — smooth shading for uniform lighting across fill caps and panel
        show_edges = mesh_to_show.n_cells < 30_000
        self._plotter.add_mesh(
            mesh_to_show,
            color=PART_COLOR,
            opacity=1.0,
            show_edges=show_edges,
            edge_color=PART_EDGE_COLOR,
            smooth_shading=True,
            name="part_mesh",
            render=False,
        )
        self._active_actors.add("part_mesh")

        # Hole spheres
        for i, h in enumerate(self._holes):
            name = f"hole_{i}"
            r = min(max(h["radius_mm"] * _SPHERE_SCALE, _SPHERE_BASE_R), 10.0)
            sphere = pv.Sphere(radius=r, center=h["center"].tolist())

            if self._show_filled:
                # After-fill view: cyan = confirmed filled, red = will remain open
                color = SPHERE_FILLED_COLOR if i in self._selected else SPHERE_DESELECTED_COLOR
                actor = self._plotter.add_mesh(
                    sphere, color=color, name=name, pickable=False, render=False,
                )
            else:
                # Original view: gold/green = will be filled, red = skipped
                color = self._sphere_color(i)
                actor = self._plotter.add_mesh(
                    sphere, color=color, name=name, pickable=True, render=False,
                )
            self._hole_actors[i] = actor
            self._active_actors.add(name)

        if self._show_filled:
            n_filled = sum(1 for i in self._selected if 0 <= i < len(self._holes))
            label = f"穴埋め後（水色={n_filled}個 確定済み）"
        else:
            label = "元の形状"
        self._view_label.setText(f"[{label}]")

        # Reset camera only when a new part is first loaded (not on view toggle)
        if self._camera_needs_reset:
            self._plotter.reset_camera()
            self._camera_needs_reset = False

        self._plotter.render()

    def _update_sphere(self, idx: int) -> None:
        """Redraw a single hole sphere with updated colour (no full re-render)."""
        if idx not in self._hole_actors and idx in self._selected:
            return  # sphere not visible yet
        name = f"hole_{idx}"
        h = self._holes[idx]
        color = self._sphere_color(idx)
        r = min(max(h["radius_mm"] * _SPHERE_SCALE, _SPHERE_BASE_R), 10.0)
        sphere = pv.Sphere(radius=r, center=h["center"].tolist())
        actor = self._plotter.add_mesh(
            sphere, color=color, name=name, pickable=True, render=True,
        )
        self._hole_actors[idx] = actor
        self._active_actors.add(name)

    def _sphere_color(self, idx: int) -> tuple[float, float, float]:
        if idx not in self._selected:
            return SPHERE_DESELECTED_COLOR
        if idx in self._annotated:
            return SPHERE_ANNOTATED_COLOR
        return SPHERE_SELECTED_COLOR

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _status(self, msg: str) -> None:
        self._status_label.setText(msg)

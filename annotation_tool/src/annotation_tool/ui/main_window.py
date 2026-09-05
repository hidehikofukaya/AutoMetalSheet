"""Main window: part tree (left) + 3D viewport (right).

Rendering
---------
Four display tiers; named-actor strategy — no plotter.clear() so the scene
never flashes.  Only stale actors (no longer in the "wanted" set) are removed;
existing named actors are replaced in-place by add_mesh.

Annotation picking — two phases
--------------------------------
Phase 1  Select joint location
  weld mode       left-click anywhere on the focus mesh surface
  bolt / mh mode  auto-detected holes are shown as orange spheres; click one

Phase 2  Select partner part
  Adjacent parts are highlighted; click one to set as partner.
  Click "Skip Partner" button (or click empty space / focus mesh) to proceed
  without a partner.
  Press [Escape] to abort the whole annotation and return to normal mode.

After Phase 2 a confirmation dialog shows the gathered info (type, point,
partner) — no re-selection of any kind.

Hover detection
---------------
A Qt timer fires every 80 ms, casts a ray at the current cursor position via
vtkPropPicker, and updates the permanent hover label in the status bar with
the part-id under the cursor.
"""

from __future__ import annotations

import datetime

import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from annotation_tool.adjacency import DEFAULT_GAP_THRESHOLD_MM, classify_adjacency
from annotation_tool.assembly import AssemblyStore
from annotation_tool.hierarchy import HierarchyBody
from annotation_tool.mesh_render import contact_zone_labels, detect_holes, feature_edges
from annotation_tool.schema import AnnotationDocument, Axis, Joint, PartRef
from annotation_tool.ui.joint_dialog import JointCreationDialog
from annotation_tool.ui.joint_panel import JointPanel
from annotation_tool.ui.part_tree import PartTreePanel

# ---------------------------------------------------------------------------
# Tier colours
# ---------------------------------------------------------------------------

FOCUS_COLOR = (0.20, 0.70, 0.30)
FOCUS_EDGE_COLOR = (0.05, 0.05, 0.05)
FOCUS_EDGE_WIDTH = 2.0

ADJACENT_PALETTE = [
    (0.121, 0.466, 0.705),
    (1.000, 0.498, 0.055),
    (0.839, 0.153, 0.157),
    (0.580, 0.404, 0.741),
    (0.549, 0.337, 0.294),
    (0.890, 0.467, 0.761),
    (0.737, 0.741, 0.133),
    (0.090, 0.745, 0.811),
    (0.498, 0.498, 0.498),
]
ADJACENT_OPACITY = 0.35
ADJACENT_EDGE_COLOR = (0.85, 0.85, 0.85)
ADJACENT_EDGE_WIDTH = 1.0

CONTEXT_COLOR = (0.60, 0.62, 0.65)
CONTEXT_OPACITY = 0.07

DIRECT_CONTACT_THRESHOLD_MM = 1.0

# ---------------------------------------------------------------------------
# Joint / hole gizmos
# ---------------------------------------------------------------------------

JOINT_GIZMO_COLOR = (1.0, 0.90, 0.10)
JOINT_GIZMO_LENGTH_MM = 20.0
JOINT_GIZMO_SPHERE_RADIUS_MM = 2.5
HOLE_SPHERE_COLOR = (1.0, 0.50, 0.00)

# Annotation modes
_ANNOTATION_MODES = [
    ("溶接  /  Weld", "weld"),
    ("ボルト  /  Bolt", "bolt"),
    ("取付穴  /  Mounting Hole", "mounting_hole"),
]

# Picking states
_ST_IDLE = ""
_ST_SURFACE = "surface"   # Phase 1: picking joint location on mesh surface
_ST_HOLE = "hole"          # Phase 1: picking a detected hole sphere
_ST_PARTNER = "partner"    # Phase 2: picking partner part by clicking


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _vec3(arr) -> tuple[float, float, float]:
    x, y, z = arr
    return (float(x), float(y), float(z))


def _make_part_ref(body: HierarchyBody, joint_type: str, xyz: tuple) -> PartRef:
    if joint_type == "weld":
        return PartRef(
            part_id=body.part_id,
            detection_method="manual",
            contact_xyz=xyz,
            local_thickness_mm=body.thickness_est_mm,
        )
    return PartRef(
        part_id=body.part_id,
        detection_method="manual",
        hole_center_xyz=xyz,
    )


def _set_pickable(actors: dict, pickable: bool) -> None:
    for actor in actors.values():
        try:
            actor.SetPickable(pickable)
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(
        self,
        store: AssemblyStore,
        doc: AnnotationDocument,
        gap_threshold_mm: float = DEFAULT_GAP_THRESHOLD_MM,
    ):
        super().__init__()
        self._store = store
        self._doc = doc
        self._gap_threshold_mm = gap_threshold_mm

        self._focus_body: HierarchyBody | None = None
        self._adjacent_ids: set[str] = set()
        self._adjacent_gaps: dict[str, float] = {}
        self._unchecked_parts: set[str] = set()

        # Picking state machine
        self._pick_state: str = _ST_IDLE
        self._pending_contact: np.ndarray | None = None
        self._pending_normal: np.ndarray | None = None
        self._pending_hole: dict | None = None
        self._detected_holes: list[dict] = []

        # Named-actor tracking
        self._active_actor_names: set[str] = set()

        # Hover + adjacent-id tracker (used by Phase-2 click)
        self._hover_picker = None
        self._hover_timer: QTimer | None = None
        self._hovered_adjacent_id: str | None = None

        self.setWindowTitle(f"Joint Annotation Tool — {store.assembly_dir.name}")
        self.resize(1400, 800)
        self._build_ui()
        self._setup_hover_detection()
        self._status(
            f"Loaded {len(store.visible_bodies)} parts"
            + (f", {len(doc.joints)} joints" if doc.joints else "")
            + ".  Select a part from the left panel."
        )

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

        # === Left panel ===
        left = QWidget()
        left.setMinimumWidth(220)
        left.setMaximumWidth(380)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)

        self._tree_panel = PartTreePanel(
            self._store,
            on_select=self._on_part_selected,
            on_visibility_change=self._on_visibility_changed,
        )
        ll.addWidget(self._tree_panel, 1)

        # -- View controls --
        view_box = QGroupBox("View")
        vl = QVBoxLayout(view_box)
        vl.setSpacing(4)
        self._adjacent_only_cb = QCheckBox("Direct contact only  (<1 mm)")
        self._adjacent_only_cb.stateChanged.connect(self._on_filter_changed)
        vl.addWidget(self._adjacent_only_cb)
        self._adjacent_edges_cb = QCheckBox("Show edges on adjacent parts")
        self._adjacent_edges_cb.stateChanged.connect(self._on_filter_changed)
        vl.addWidget(self._adjacent_edges_cb)
        show_all_btn = QPushButton("Show all parts")
        show_all_btn.clicked.connect(self._on_show_all)
        vl.addWidget(show_all_btn)
        ll.addWidget(view_box)

        # -- Annotation controls --
        ann_box = QGroupBox("Annotation")
        al = QVBoxLayout(ann_box)
        al.setSpacing(4)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        for label, value in _ANNOTATION_MODES:
            self._mode_combo.addItem(label, value)
        mode_row.addWidget(self._mode_combo)
        al.addLayout(mode_row)
        self._add_joint_btn = QPushButton("Add Joint")
        self._add_joint_btn.setEnabled(False)
        self._add_joint_btn.clicked.connect(self._on_add_joint_clicked)
        al.addWidget(self._add_joint_btn)
        ll.addWidget(ann_box)

        # -- Joint list --
        self._joint_panel = JointPanel(
            doc=self._doc,
            on_delete=self._on_joint_deleted,
        )
        ll.addWidget(self._joint_panel)

        splitter.addWidget(left)

        # === Viewport ===
        self._plotter = QtInteractor(self)
        self._plotter.set_background("black")
        splitter.addWidget(self._plotter)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Status bar: left = messages, right = hover label
        self._statusbar = QStatusBar()
        self._hover_label = QLabel("")
        self._hover_label.setMinimumWidth(220)
        self._statusbar.addPermanentWidget(self._hover_label)
        self.setStatusBar(self._statusbar)

    def _status(self, msg: str) -> None:
        self._statusbar.showMessage(msg)

    # ------------------------------------------------------------------
    # Hover detection (Qt timer + vtkPropPicker)
    # ------------------------------------------------------------------

    def _setup_hover_detection(self) -> None:
        try:
            import vtk  # noqa: PLC0415
            self._hover_picker = vtk.vtkPropPicker()
            self._hover_timer = QTimer(self)
            self._hover_timer.setInterval(80)
            self._hover_timer.timeout.connect(self._do_hover_check)
            self._hover_timer.start()
            # Qt event filter on the plotter widget intercepts left-clicks
            # before VTK converts them — avoids OpenGL-reentrant picking issues
            self._plotter.installEventFilter(self)
        except Exception:
            pass  # vtk unavailable or pyvistaqt too old

    def _do_hover_check(self) -> None:
        if self._hover_picker is None or self._pick_state in (_ST_SURFACE, _ST_HOLE):
            return
        try:
            gpos = QCursor.pos()
            lpos = self._plotter.mapFromGlobal(gpos)
            x = lpos.x()
            y = self._plotter.height() - lpos.y() - 1
            if not (0 <= x < self._plotter.width() and 0 <= y < self._plotter.height()):
                self._hover_label.setText("")
                self._hovered_adjacent_id = None
                return
            self._hover_picker.Pick(x, y, 0, self._plotter.renderer)
            actor = self._hover_picker.GetActor()
            label = ""
            self._hovered_adjacent_id = None
            if actor is not None:
                for name, a in self._plotter.actors.items():
                    if a is actor:
                        label = self._actor_name_to_display(name)
                        if (
                            name.startswith("adjacent_")
                            and not name.startswith("adjacent_edges_")
                        ):
                            self._hovered_adjacent_id = name[len("adjacent_"):]
                        break
            self._hover_label.setText(label)
        except Exception:
            pass

    def _actor_name_to_display(self, name: str) -> str:
        if name in ("focus", "focus_edges"):
            return f"[focus] {self._focus_body.part_id}" if self._focus_body else ""
        if name.startswith("adjacent_") and not name.startswith("adjacent_edges_"):
            return f"[adj] {name[len('adjacent_'):]}"
        if name.startswith("context_"):
            return f"[ctx] {name[len('context_'):]}"
        if name.startswith("hole_"):
            try:
                h = self._detected_holes[int(name[len("hole_"):])]
                return f"[hole] r={h['radius_mm']:.1f} mm"
            except (IndexError, ValueError):
                return "[hole]"
        return ""

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event) -> bool:
        """Intercept left-click on the plotter for Phase-2 partner picking.

        Fires at the Qt level — before rwi.py converts the Qt mouse event to a
        VTK LeftButtonPressEvent.  At this point the OpenGL context is in the
        same stable state as the hover timer, so vtkPropPicker works reliably.
        """
        from PyQt5.QtCore import QEvent  # noqa: PLC0415
        if (
            obj is self._plotter
            and event.type() == QEvent.MouseButtonPress
            and event.button() == Qt.LeftButton
            and self._pick_state == _ST_PARTNER
        ):
            partner_id = self._hovered_adjacent_id
            # partner_id may be None (click on empty space → Skip Partner)
            self._finish_partner_picking(partner_id)
            return True  # consume event → VTK never sees it → no camera rotation
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape and self._pick_state:
            self._cancel_picking()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._hover_timer is not None:
            self._hover_timer.stop()
        self._plotter.close()
        event.accept()

    # ------------------------------------------------------------------
    # Part-tree event handlers
    # ------------------------------------------------------------------

    def _on_part_selected(self, body: HierarchyBody) -> None:
        if self._pick_state:
            self._cancel_picking()
        self._focus_body = body
        results = classify_adjacency(self._store, body, self._gap_threshold_mm)
        self._adjacent_ids = {r.part_id for r in results if r.tier == "adjacent"}
        self._adjacent_gaps = {
            r.part_id: r.gap_distance_mm
            for r in results
            if r.tier == "adjacent" and r.gap_distance_mm is not None
        }
        self._add_joint_btn.setEnabled(True)
        self._joint_panel.refresh(body.part_id)
        self._refresh_viewport(reset_camera=True)

    def _on_visibility_changed(self, body: HierarchyBody, visible: bool) -> None:
        if visible:
            self._unchecked_parts.discard(body.part_id)
        else:
            self._unchecked_parts.add(body.part_id)
        self._refresh_viewport()

    def _on_filter_changed(self, _state: int) -> None:
        self._refresh_viewport()

    def _on_show_all(self) -> None:
        self._unchecked_parts.clear()
        self._tree_panel.check_all()
        self._refresh_viewport()

    # ------------------------------------------------------------------
    # Annotation button
    # ------------------------------------------------------------------

    def _on_add_joint_clicked(self) -> None:
        if self._pick_state in (_ST_SURFACE, _ST_HOLE):
            self._cancel_picking()
        elif self._pick_state == _ST_PARTNER:
            self._finish_partner_picking(None)   # skip partner → dialog
        else:
            if self._focus_body is None:
                return
            mode = self._mode_combo.currentData()
            if mode in ("bolt", "mounting_hole"):
                self._enter_hole_picking()
            else:
                self._enter_surface_picking()

    def _on_joint_deleted(self, joint_id: str) -> None:
        self._doc.remove_joint(joint_id)
        self._doc.save()
        if self._focus_body:
            self._joint_panel.refresh(self._focus_body.part_id)
        self._refresh_viewport()
        self._status(f"Deleted {joint_id}  |  saved → {self._doc.joints_json_path}")

    # ------------------------------------------------------------------
    # Phase 1a: surface picking (weld)
    # ------------------------------------------------------------------

    def _enter_surface_picking(self) -> None:
        self._pick_state = _ST_SURFACE
        self._add_joint_btn.setText("Cancel")
        _set_pickable(self._plotter.actors, False)
        if "focus" in self._plotter.actors:
            self._plotter.actors["focus"].SetPickable(True)
        self._plotter.enable_surface_point_picking(
            callback=self._on_surface_picked,
            left_clicking=True,
            show_message=False,
            show_point=False,      # no yellow dot artifact
            tolerance=0.025,
        )
        self._status("フォーカス部品をクリックしてジョイント位置を指定  |  [Escape] キャンセル")

    def _on_surface_picked(self, point) -> None:
        if self._pick_state != _ST_SURFACE or point is None:
            return
        try:
            self._plotter.disable_picking()
        except AttributeError:
            pass
        _set_pickable(self._plotter.actors, True)

        # Compute face normal at the picked point
        face_normal = np.array([0.0, 0.0, 1.0])
        try:
            fm = self._store.load_mesh(self._focus_body)
            nm = fm.compute_normals(cell_normals=True, point_normals=False, inplace=False)
            cid = nm.find_closest_cell(np.asarray(point, dtype=float))
            if cid >= 0:
                face_normal = np.array(nm.cell_data["Normals"][cid])
        except Exception:
            pass

        self._pending_contact = np.asarray(point, dtype=float)
        self._pending_normal = face_normal
        self._pending_hole = None
        self._enter_partner_picking()

    # ------------------------------------------------------------------
    # Phase 1b: hole picking (bolt / mounting_hole)
    # ------------------------------------------------------------------

    def _enter_hole_picking(self) -> None:
        try:
            focus_mesh = self._store.load_mesh(self._focus_body)
        except FileNotFoundError:
            self._status("フォーカス部品のメッシュが見つかりません。")
            return

        self._detected_holes = detect_holes(focus_mesh)
        if not self._detected_holes:
            self._status("穴が検出されませんでした。溶接モードで位置を手動指定してください。")
            return

        self._pick_state = _ST_HOLE
        self._add_joint_btn.setText("Cancel")
        _set_pickable(self._plotter.actors, False)

        for i, hole in enumerate(self._detected_holes):
            name = f"hole_{i}"
            sphere = pv.Sphere(
                radius=max(hole["radius_mm"], JOINT_GIZMO_SPHERE_RADIUS_MM),
                center=hole["center"].tolist(),
            )
            self._plotter.add_mesh(
                sphere, color=HOLE_SPHERE_COLOR, opacity=0.75,
                name=name, reset_camera=False,
            )
            self._plotter.actors[name].SetPickable(True)
            self._active_actor_names.add(name)

        self._plotter.enable_surface_point_picking(
            callback=self._on_hole_sphere_picked,
            left_clicking=True,
            show_message=False,
            show_point=False,
            tolerance=0.025,
        )
        self._status(
            f"穴 {len(self._detected_holes)} 個を検出 (橙色)。クリックして選択  |  [Escape] キャンセル"
        )

    def _on_hole_sphere_picked(self, point) -> None:
        if self._pick_state != _ST_HOLE or point is None or not self._detected_holes:
            return
        try:
            self._plotter.disable_picking()
        except AttributeError:
            pass
        _set_pickable(self._plotter.actors, True)

        pt = np.asarray(point, dtype=float)
        closest = min(
            self._detected_holes,
            key=lambda h: float(np.linalg.norm(h["center"] - pt)),
        )

        self._pending_contact = closest["center"]
        self._pending_normal = closest["normal"]
        self._pending_hole = closest
        self._enter_partner_picking()

    # ------------------------------------------------------------------
    # Phase 2: partner picking
    # ------------------------------------------------------------------

    def _enter_partner_picking(self) -> None:
        self._pick_state = _ST_PARTNER
        self._add_joint_btn.setText("Skip Partner")
        self._hovered_adjacent_id = None   # reset so stale hover doesn't auto-fire
        self._status(
            "隣接部品をクリックしてパートナー選択  |  「Skip Partner」= パートナーなし  |  [Escape] 全キャンセル"
        )

    def _finish_partner_picking(self, partner_id: str | None) -> None:
        _set_pickable(self._plotter.actors, True)

        # Snapshot pending data (will be cleared below)
        contact = self._pending_contact
        normal = self._pending_normal
        hole = self._pending_hole
        joint_type = self._mode_combo.currentData()

        # Reset state BEFORE opening the dialog
        self._pick_state = _ST_IDLE
        self._pending_contact = None
        self._pending_normal = None
        self._pending_hole = None
        self._add_joint_btn.setText("Add Joint")

        # Confirmation dialog (no re-selection of anything)
        dlg = JointCreationDialog(
            joint_type=joint_type,
            contact_point=_vec3(contact),
            partner_id=partner_id,
            hole_info=(
                {"radius_mm": hole["radius_mm"], "circularity": hole["circularity"]}
                if hole else None
            ),
            parent=self,
        )
        if dlg.exec_() == JointCreationDialog.Accepted:
            if hole is not None:
                self._create_hole_joint(hole, joint_type, partner_id)
            else:
                self._create_surface_joint(contact, normal, joint_type, partner_id)
        else:
            # Cancelled: rebuild to remove any hole-sphere actors
            self._refresh_viewport()

    # ------------------------------------------------------------------
    # Full cancel from any phase
    # ------------------------------------------------------------------

    def _cancel_picking(self) -> None:
        try:
            self._plotter.disable_picking()
        except AttributeError:
            pass
        self._pick_state = _ST_IDLE
        self._pending_contact = None
        self._pending_normal = None
        self._pending_hole = None
        self._add_joint_btn.setText("Add Joint")
        _set_pickable(self._plotter.actors, True)
        self._refresh_viewport()

    # ------------------------------------------------------------------
    # Joint creation
    # ------------------------------------------------------------------

    def _create_surface_joint(
        self,
        contact: np.ndarray,
        normal: np.ndarray,
        joint_type: str,
        partner_id: str | None,
    ) -> None:
        xyz = _vec3(contact)
        nrm = _vec3(normal)
        thickness = self._focus_body.thickness_est_mm or 1.0
        axis = Axis(start_xyz=xyz, direction_xyz=nrm, length_mm=thickness)
        parts = [self._focus_body.part_id]
        per_part = [_make_part_ref(self._focus_body, joint_type, xyz)]
        if partner_id:
            pb = next(
                (b for b in self._store.visible_bodies if b.part_id == partner_id), None
            )
            if pb:
                parts.append(partner_id)
                per_part.append(_make_part_ref(pb, joint_type, xyz))
        self._commit_joint(joint_type, parts, per_part, axis, "manual", partner_id)

    def _create_hole_joint(
        self,
        hole: dict,
        joint_type: str,
        partner_id: str | None,
    ) -> None:
        xyz = _vec3(hole["center"])
        nrm = _vec3(hole["normal"])
        thickness = self._focus_body.thickness_est_mm or 1.0
        diameter = float(hole["radius_mm"]) * 2
        axis = Axis(start_xyz=xyz, direction_xyz=nrm, length_mm=thickness)
        parts = [self._focus_body.part_id]
        per_part = [
            PartRef(
                part_id=self._focus_body.part_id,
                detection_method="auto_circle",
                hole_center_xyz=xyz,
                hole_diameter_mm=diameter,
            )
        ]
        if partner_id:
            pb = next(
                (b for b in self._store.visible_bodies if b.part_id == partner_id), None
            )
            if pb:
                parts.append(partner_id)
                per_part.append(
                    PartRef(
                        part_id=partner_id,
                        detection_method="auto_circle",
                        hole_center_xyz=xyz,
                        hole_diameter_mm=diameter,
                    )
                )
        self._commit_joint(joint_type, parts, per_part, axis, "user_confirmed", partner_id)

    def _commit_joint(
        self,
        joint_type: str,
        parts: list[str],
        per_part: list[PartRef],
        axis: Axis,
        confidence: str,
        partner_id: str | None,
    ) -> None:
        joint = Joint(
            joint_id=self._doc.next_joint_id(),
            type=joint_type,
            parts=parts,
            axis=axis,
            per_part=per_part,
            confidence=confidence,
            created_at=datetime.datetime.now().isoformat(),
        )
        self._doc.add_joint(joint)
        self._doc.save()
        self._joint_panel.refresh(self._focus_body.part_id)
        self._refresh_viewport()
        self._status(
            f"Added {joint.joint_id} [{joint_type}]"
            + (f" → {partner_id}" if partner_id else "")
            + f"  |  saved → {self._doc.joints_json_path}"
        )

    # ------------------------------------------------------------------
    # Viewport (named-actor, no clear())
    # ------------------------------------------------------------------

    def _refresh_viewport(self, reset_camera: bool = False) -> None:
        """Differential actor update — no plotter.clear()."""
        if self._focus_body is None:
            for name in list(self._active_actor_names):
                try:
                    self._plotter.remove_actor(name)
                except Exception:
                    pass
            self._active_actor_names.clear()
            return

        focus_mesh = None
        if self._focus_body.part_id not in self._unchecked_parts:
            try:
                focus_mesh = self._store.load_mesh(self._focus_body)
            except FileNotFoundError as exc:
                self._status(str(exc))
                return

        show_only_adj = self._adjacent_only_cb.isChecked()
        show_adj_edges = self._adjacent_edges_cb.isChecked()
        wanted: set[str] = set()

        # Collect adjacent in stable order
        adjacent_entries: list[tuple[HierarchyBody, object, tuple]] = []
        for body in self._store.visible_bodies:
            if body.part_id not in self._adjacent_ids:
                continue
            if body.part_id in self._unchecked_parts:
                continue
            if show_only_adj:
                if self._adjacent_gaps.get(body.part_id, float("inf")) >= DIRECT_CONTACT_THRESHOLD_MM:
                    continue
            try:
                mesh = self._store.load_mesh(body)
            except FileNotFoundError:
                continue
            color = ADJACENT_PALETTE[len(adjacent_entries) % len(ADJACENT_PALETTE)]
            adjacent_entries.append((body, mesh, color))

        # Context tier
        if not show_only_adj:
            for body in self._store.visible_bodies:
                if body == self._focus_body:
                    continue
                if body.part_id in self._adjacent_ids:
                    continue
                if body.part_id in self._unchecked_parts:
                    continue
                try:
                    mesh = self._store.load_mesh(body)
                except FileNotFoundError:
                    continue
                name = f"context_{body.part_id}"
                self._plotter.add_mesh(
                    mesh, color=CONTEXT_COLOR, opacity=CONTEXT_OPACITY,
                    smooth_shading=False, lighting=False,
                    name=name, reset_camera=False,
                )
                wanted.add(name)

        # Adjacent tier
        for body, mesh, color in adjacent_entries:
            name = f"adjacent_{body.part_id}"
            self._plotter.add_mesh(
                mesh, color=color, opacity=ADJACENT_OPACITY,
                smooth_shading=True, name=name, reset_camera=False,
            )
            wanted.add(name)
            edge_name = f"adjacent_edges_{body.part_id}"
            if show_adj_edges:
                self._plotter.add_mesh(
                    feature_edges(mesh), color=ADJACENT_EDGE_COLOR,
                    line_width=ADJACENT_EDGE_WIDTH, name=edge_name, reset_camera=False,
                )
                wanted.add(edge_name)

        # Focus tier with contact-zone coloring
        if focus_mesh is not None:
            wanted.add("focus")
            wanted.add("focus_edges")
            if adjacent_entries:
                adj_pairs = [(mesh, i + 1) for i, (_, mesh, _) in enumerate(adjacent_entries)]
                labels = contact_zone_labels(focus_mesh, adj_pairs)
                palette = [FOCUS_COLOR] + [c for _, _, c in adjacent_entries]
                cmap = ListedColormap(palette)
                self._plotter.add_mesh(
                    focus_mesh, scalars=labels, cmap=cmap,
                    clim=[0, len(palette) - 1], show_scalar_bar=False,
                    smooth_shading=True, specular=0.5, specular_power=20,
                    name="focus", reset_camera=False,
                )
            else:
                self._plotter.add_mesh(
                    focus_mesh, color=FOCUS_COLOR, opacity=1.0,
                    smooth_shading=True, specular=0.5, specular_power=20,
                    name="focus", reset_camera=False,
                )
            self._plotter.add_mesh(
                feature_edges(focus_mesh), color=FOCUS_EDGE_COLOR,
                line_width=FOCUS_EDGE_WIDTH, name="focus_edges", reset_camera=False,
            )

        # Joint gizmos — build partner→color map from the entries rendered this frame
        adj_color_map = {body.part_id: color for body, _, color in adjacent_entries}
        wanted.update(self._refresh_joint_gizmos(adj_color_map))

        # Remove stale actors
        stale = self._active_actor_names - wanted
        for name in stale:
            try:
                self._plotter.remove_actor(name)
            except Exception:
                pass
        self._active_actor_names = wanted

        if reset_camera:
            self._plotter.reset_camera()
            if focus_mesh is not None:
                # Translate camera + focal point by the same vector so the
                # view direction and zoom are preserved but the orbit center
                # moves to the focused mesh center (not the centroid of all actors).
                cam = self._plotter.camera
                delta = np.asarray(focus_mesh.center) - np.asarray(cam.focal_point)
                cam.focal_point = (np.asarray(cam.focal_point) + delta).tolist()
                cam.position = (np.asarray(cam.position) + delta).tolist()

        self._status(
            f"{self._focus_body.part_id}  [{self._focus_body.tag}]"
            f"  t={self._focus_body.thickness_est_mm:.2f} mm"
            f"  faces={self._focus_body.face_count}"
            f"  |  adjacent={len(adjacent_entries)}"
            f"  |  joints={len(self._doc.joints_for_part(self._focus_body.part_id))}"
        )

    def _refresh_joint_gizmos(self, adj_color_map: dict) -> set[str]:
        """Add arrow+sphere gizmos for each joint of the focused part.

        adj_color_map  -- {part_id: (r, g, b)} built from adjacent_entries
                          this frame; used to match the gizmo color to the
                          partner part's rendered color.

        A joint is omitted entirely when any of its non-focus partner parts
        is hidden by the user (in _unchecked_parts).
        """
        names: set[str] = set()
        if self._focus_body is None:
            return names
        focus_id = self._focus_body.part_id
        for joint in self._doc.joints_for_part(focus_id):
            partners = [p for p in joint.parts if p != focus_id]

            # Hide gizmo when any partner is unchecked (user-hidden)
            if any(p in self._unchecked_parts for p in partners):
                continue

            # Color: use the partner's adjacent-tier color; fall back to yellow
            color = JOINT_GIZMO_COLOR
            for p in partners:
                if p in adj_color_map:
                    color = adj_color_map[p]
                    break

            start = np.array(joint.axis.start_xyz)
            direction = np.array(joint.axis.direction_xyz)
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm
            an = f"gizmo_{joint.joint_id}"
            pn = f"gizmo_pt_{joint.joint_id}"
            arrow = pv.Arrow(
                start=start.tolist(),
                direction=direction.tolist(),
                tip_length=0.25,
                tip_radius=0.06,
                shaft_radius=0.025,
                scale=JOINT_GIZMO_LENGTH_MM,
            )
            sphere = pv.Sphere(radius=JOINT_GIZMO_SPHERE_RADIUS_MM, center=start.tolist())
            self._plotter.add_mesh(arrow, color=color, name=an, reset_camera=False)
            self._plotter.add_mesh(sphere, color=color, name=pn, reset_camera=False)
            names.add(an)
            names.add(pn)
        return names

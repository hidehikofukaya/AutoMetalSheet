"""
stp_viewer.py - BIW STP Viewer with integrated Hole Editor

Layout
------
 ┌──────────────┬──────────────────────────────────────────────────┐
 │  Part List   │                  3D Viewport                     │
 │  ──────────  │                                                   │
 │  [Filter  ]  │   Selected body: opaque                          │
 │  ▼ A0072600  │   Sibling bodies (same subassembly): 60% opacity  │
 │    ▼ Sub.A   │   Cousin bodies (parent assembly): 20% opacity    │
 │      body002 │                                                   │
 │      body003 │                                                   │
 │  ──────────  │                                                   │
 │  Selected:   │                                                   │
 │  body002     │                                                   │
 │  t=1.36mm    │                                                   │
 │  [Edit Holes]│                                                   │
 │  ──────────  │                                                   │
 │  Holes (1):  │                                                   │
 │  ☑ bolt φ12  │                                                   │
 │  [Fill Sel.] │                                                   │
 │  [Fill All ] │                                                   │
 │  [ Save  ]   │                                                   │
 └──────────────┴──────────────────────────────────────────────────┘

Usage
-----
    python stp_viewer.py path/to/hierarchy.json
    python stp_viewer.py --dir C:/.../Mesh_Generater/output
    python stp_viewer.py --dir C:/.../Mesh_Generater/output --filter sheet_metal
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import time
from collections import defaultdict

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPushButton, QSizePolicy, QSplitter, QStatusBar,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

# ── insert src/preprocess into path so we can import hole_filler / annotator ──
_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "preprocess"))
from hole_filler import detect_holes, fill_holes, _read_step, _write_step
from annotator import step_to_pyvista  # STEP -> pyvista PolyData

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MESH_CACHE_SUFFIX   = ".vtp"          # tessellation cache alongside STP files
TESSELLATION_MM     = 0.30            # coarser for interactive display
TAG_OPTIONS = ["sheet_metal", "thick_part", "small_hardware", "adhesive"]
TAG_COLORS = {
    "sheet_metal":   (0.55, 0.75, 0.95),  # light blue
    "thick_part":    (0.75, 0.75, 0.75),  # gray
    "small_hardware":(0.80, 0.65, 0.40),  # tan
    "adhesive":      (0.70, 0.95, 0.65),  # light green
}
SELECTED_COLOR      = (0.20, 0.70, 0.30)  # green
SIBLING_COLOR       = (0.65, 0.80, 0.95)  # pale blue
COUSIN_COLOR        = (0.75, 0.75, 0.85)  # pale lavender
HOLE_COLOR_NORMAL   = (1.00, 0.80, 0.00)  # yellow
HOLE_COLOR_SELECTED = (1.00, 0.25, 0.10)  # red

# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

class PartEntry:
    """One body extracted from hierarchy.json."""

    def __init__(self, root: pathlib.Path, d: dict):
        self.root         = root
        self.body_idx: int = d.get("body_idx", 0)
        self.catia_name: str = d.get("catia_name", "")
        self.tag: str      = d.get("tag", "")
        self.stp_file: str | None = d.get("stp_file")
        self.thickness_mm: float | None = d.get("thickness_est_mm")
        self.volume_mm3: float          = d.get("volume_mm3", 0.0)
        self.face_count: int            = d.get("face_count", 0)

    # ── path helpers ─────────────────────────────────────────────────────────

    @property
    def stp_path(self) -> pathlib.Path | None:
        if not self.stp_file:
            return None
        base = self.root / self.stp_file
        filled = base.parent / (base.stem + "_filled.stp")
        return filled if filled.exists() else (base if base.exists() else None)

    @property
    def orig_stp_path(self) -> pathlib.Path | None:
        """The unfilled original (for backup targets)."""
        if not self.stp_file:
            return None
        p = self.root / self.stp_file
        return p if p.exists() else None

    # ── catia_name path decomposition ─────────────────────────────────────────

    @property
    def _path_parts(self) -> list[str]:
        return self.catia_name.split("\\") if self.catia_name else []

    @property
    def direct_parent(self) -> str:
        """Second-to-last component of catia_name (one level above the geometry)."""
        p = self._path_parts
        return p[-2] if len(p) >= 2 else ""

    @property
    def subassembly(self) -> str:
        """Third-to-last component (the subassembly grouping level)."""
        p = self._path_parts
        return p[-3] if len(p) >= 3 else (p[-2] if len(p) >= 2 else "")

    @property
    def tree_group_key(self) -> str:
        """Key used to group bodies in the left-panel tree (second component)."""
        p = self._path_parts
        # Use second element if available (first-level subassembly under root)
        return p[1] if len(p) >= 2 else (p[0] if p else "")

    # ── display helpers ───────────────────────────────────────────────────────

    @property
    def label(self) -> str:
        t = f"  t={self.thickness_mm:.2f}mm" if self.thickness_mm else ""
        return f"body{self.body_idx:03d}  [{self.tag}]{t}"

    @property
    def part_root_name(self) -> str:
        return self.root.name

    def __repr__(self):
        return f"<PartEntry {self.part_root_name}/body{self.body_idx:03d} {self.tag}>"


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchy data store
# ─────────────────────────────────────────────────────────────────────────────

class HierarchyStore:
    def __init__(self):
        self.parts: list[PartEntry] = []

    def load_file(self, hierarchy_json: pathlib.Path, tag_filter: str | None = None):
        data  = json.loads(hierarchy_json.read_text("utf-8"))
        root  = hierarchy_json.parent
        for b in data.get("bodies", []):
            tag = b.get("tag", "")
            # Skip only truly-empty bodies (no geometry at all)
            if tag == "empty" and b.get("shapes_count", 0) == 0:
                continue
            if tag_filter and tag != tag_filter:
                continue
            self.parts.append(PartEntry(root, b))

    def load_dir(self, output_dir: pathlib.Path, tag_filter: str | None = None):
        for hf in sorted(output_dir.rglob("hierarchy.json")):
            self.load_file(hf, tag_filter=tag_filter)

    def siblings(self, part: PartEntry) -> list[PartEntry]:
        """Bodies in the same direct_parent group (same physical sub-part)."""
        return [p for p in self.parts
                if p.root == part.root
                and p.direct_parent == part.direct_parent
                and p.body_idx != part.body_idx
                and p.stp_path is not None]

    def cousins(self, part: PartEntry) -> list[PartEntry]:
        """Bodies in the same subassembly but DIFFERENT direct_parent."""
        return [p for p in self.parts
                if p.root == part.root
                and p.subassembly == part.subassembly
                and p.direct_parent != part.direct_parent
                and p.stp_path is not None]


# ─────────────────────────────────────────────────────────────────────────────
# Mesh cache
# ─────────────────────────────────────────────────────────────────────────────

def load_mesh(stp: pathlib.Path, deflection: float = TESSELLATION_MM) -> pv.PolyData:
    """Load mesh from STP, using a .vtp cache when available."""
    cache = stp.with_suffix(MESH_CACHE_SUFFIX)
    if cache.exists() and cache.stat().st_mtime >= stp.stat().st_mtime:
        try:
            return pv.read(str(cache))
        except Exception:
            pass
    mesh = step_to_pyvista(stp, linear_deflection=deflection)
    try:
        mesh.save(str(cache))
    except Exception:
        pass
    return mesh


# ─────────────────────────────────────────────────────────────────────────────
# Left panel: Part Tree
# ─────────────────────────────────────────────────────────────────────────────

class PartTreePanel(QWidget):
    def __init__(self, store: HierarchyStore, on_select):
        super().__init__()
        self._store    = store
        self._callback = on_select   # fn(PartEntry | None)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        # Filter box
        self._filter = QLineEdit(placeholderText="Filter by name / tag…")
        self._filter.textChanged.connect(self._apply_filter)
        lay.addWidget(self._filter)

        # Tree
        self._tree = QTreeWidget()
        self._tree.setHeaderLabel("Parts")
        self._tree.setColumnCount(1)
        self._tree.itemClicked.connect(self._on_item_clicked)
        lay.addWidget(self._tree)

        self.refresh()

    def refresh(self):
        self._tree.clear()
        # Group by (root_name, tree_group_key)
        groups: dict[tuple, list[PartEntry]] = defaultdict(list)
        for p in self._store.parts:
            groups[(p.part_root_name, p.tree_group_key)].append(p)

        # Build top-level items per root
        roots: dict[str, QTreeWidgetItem] = {}
        for (root_name, group_key), parts in sorted(groups.items()):
            if root_name not in roots:
                root_item = QTreeWidgetItem(self._tree, [root_name])
                root_item.setFlags(root_item.flags() & ~Qt.ItemIsSelectable)
                f = root_item.font(0)
                f.setBold(True)
                root_item.setFont(0, f)
                roots[root_name] = root_item
            parent_item = roots[root_name]

            # Sub-group item
            if group_key:
                grp_item = QTreeWidgetItem(parent_item, [f"▸ {group_key}"])
                grp_item.setFlags(grp_item.flags() & ~Qt.ItemIsSelectable)
                grp_item.setForeground(0, QColor(130, 130, 130))
            else:
                grp_item = parent_item

            for p in sorted(parts, key=lambda x: x.body_idx):
                item = QTreeWidgetItem(grp_item, [p.label])
                item.setData(0, Qt.UserRole, p)
                # Color by tag
                col = TAG_COLORS.get(p.tag, (0.85, 0.85, 0.85))
                item.setForeground(0, QColor(int(col[0]*200), int(col[1]*200), int(col[2]*255)))

        self._tree.expandAll()

    def _on_item_clicked(self, item: QTreeWidgetItem, _col):
        part = item.data(0, Qt.UserRole)
        if isinstance(part, PartEntry):
            self._callback(part)

    def _apply_filter(self, text: str):
        text = text.lower()
        def _walk(item: QTreeWidgetItem):
            any_visible = False
            for i in range(item.childCount()):
                child = item.child(i)
                part = child.data(0, Qt.UserRole)
                if part:
                    visible = (text in child.text(0).lower()
                               or text in part.catia_name.lower())
                    child.setHidden(not visible)
                    if visible:
                        any_visible = True
                else:
                    sub = _walk(child)
                    child.setHidden(not sub)
                    if sub:
                        any_visible = True
            return any_visible
        for i in range(self._tree.topLevelItemCount()):
            top = self._tree.topLevelItem(i)
            _walk(top)
            top.setHidden(False)

    def highlight_item(self, part: PartEntry):
        """Scroll to and select the tree item for *part*."""
        def _search(item: QTreeWidgetItem):
            p = item.data(0, Qt.UserRole)
            if p and p.body_idx == part.body_idx and p.root == part.root:
                self._tree.setCurrentItem(item)
                self._tree.scrollToItem(item)
                return True
            for i in range(item.childCount()):
                if _search(item.child(i)):
                    return True
            return False
        for i in range(self._tree.topLevelItemCount()):
            _search(self._tree.topLevelItem(i))

    def refresh_preserving_expansion(self):
        """Rebuild the tree while keeping currently expanded nodes open."""
        # Collect expanded item texts
        expanded = set()
        def _collect(item: QTreeWidgetItem):
            if item.isExpanded():
                expanded.add(item.text(0))
            for i in range(item.childCount()):
                _collect(item.child(i))
        for i in range(self._tree.topLevelItemCount()):
            _collect(self._tree.topLevelItem(i))

        self.refresh()

        def _restore(item: QTreeWidgetItem):
            if item.text(0) in expanded:
                item.setExpanded(True)
            for i in range(item.childCount()):
                _restore(item.child(i))
        for i in range(self._tree.topLevelItemCount()):
            _restore(self._tree.topLevelItem(i))


# ─────────────────────────────────────────────────────────────────────────────
# Left panel: Hole Editor Panel
# ─────────────────────────────────────────────────────────────────────────────

class HoleEditorPanel(QWidget):
    """Hole list + action buttons, shown below the part tree."""

    def __init__(self, on_fill_selected, on_fill_all, on_save, on_cancel):
        super().__init__()
        self._on_fill_selected = on_fill_selected
        self._on_fill_all      = on_fill_all
        self._on_save          = on_save
        self._on_cancel        = on_cancel
        self._holes: list[dict] = []
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine); lay.addWidget(sep)

        self._title = QLabel("Holes:")
        f = self._title.font(); f.setBold(True); self._title.setFont(f)
        lay.addWidget(self._title)

        self._list = QListWidget()
        self._list.setMaximumHeight(140)
        self._list.itemChanged.connect(self._on_item_changed)
        lay.addWidget(self._list)

        btn_row = QHBoxLayout()
        self._btn_sel = QPushButton("Fill Selected")
        self._btn_all = QPushButton("Fill All")
        self._btn_sel.clicked.connect(self._on_fill_selected)
        self._btn_all.clicked.connect(self._on_fill_all)
        btn_row.addWidget(self._btn_sel)
        btn_row.addWidget(self._btn_all)
        lay.addLayout(btn_row)

        self._btn_save = QPushButton("💾 Save (backup orig)")
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._on_save)
        lay.addWidget(self._btn_save)

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.clicked.connect(self._on_cancel)
        lay.addWidget(self._btn_cancel)

        self._note = QLabel("")
        self._note.setWordWrap(True)
        self._note.setStyleSheet("color: #888; font-size: 10px;")
        lay.addWidget(self._note)

    def _on_item_changed(self, item: QListWidgetItem):
        any_checked = any(
            self._list.item(i).checkState() == Qt.Checked
            for i in range(self._list.count())
        )
        self._btn_save.setEnabled(any_checked)

    def load_holes(self, holes: list[dict]):
        self._holes = holes
        self._list.blockSignals(True)
        self._list.clear()
        for h in holes:
            cat = h.get("category", "?")
            d   = h.get("diameter_mm", 0)
            text = f"{cat}  φ{d:.1f}mm  @ {[round(v,1) for v in h['center_xyz']]}"
            item = QListWidgetItem(text)
            item.setCheckState(Qt.Unchecked)
            self._list.addItem(item)
        self._list.blockSignals(False)
        self._title.setText(f"Holes ({len(holes)}):")
        self._btn_save.setEnabled(False)
        self._note.setText("")

    def get_selected_indices(self) -> list[int]:
        return [i for i in range(self._list.count())
                if self._list.item(i).checkState() == Qt.Checked]

    def select_all(self):
        self._list.blockSignals(True)
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(Qt.Checked)
        self._list.blockSignals(False)
        self._btn_save.setEnabled(bool(self._holes))

    def set_note(self, text: str):
        self._note.setText(text)

    def mark_saved(self):
        self._btn_save.setEnabled(False)
        self._note.setText("Saved. (Original backed up as .stp.bak)")


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, store: HierarchyStore):
        super().__init__()
        self._store         = store
        self._selected_part: PartEntry | None = None
        self._loaded_actors: dict[str, any] = {}   # actor_name -> pyvista actor
        self._hole_shapes:   list[dict]     = []   # detected holes (from hole_filler)
        self._pending_holes: list[dict]     = []   # holes marked for filling
        self._edit_mode     = False

        self.setWindowTitle("BIW STP Viewer")
        self.resize(1400, 800)
        self._build_ui()
        self._status("Ready. Select a body from the left panel.")

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_lay = QHBoxLayout(central)
        main_lay.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        main_lay.addWidget(splitter)

        # ── Left panel ────────────────────────────────────────────────────────
        left = QWidget()
        left.setMaximumWidth(320)
        left.setMinimumWidth(220)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(4, 4, 4, 4)
        left_lay.setSpacing(0)

        self._tree_panel = PartTreePanel(self._store, self._on_part_selected)
        left_lay.addWidget(self._tree_panel, stretch=1)

        # Info label
        self._info_label = QLabel("(nothing selected)")
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("padding: 4px; color: #ccc; font-size: 11px;")
        left_lay.addWidget(self._info_label)

        # ── Tag editor ────────────────────────────────────────────────────────
        tag_sep = QFrame(); tag_sep.setFrameShape(QFrame.HLine)
        left_lay.addWidget(tag_sep)

        tag_row = QHBoxLayout()
        tag_row.setContentsMargins(4, 0, 4, 0)
        tag_label = QLabel("Tag:")
        tag_label.setFixedWidth(28)
        tag_row.addWidget(tag_label)

        self._tag_combo = QComboBox()
        self._tag_combo.addItems(TAG_OPTIONS)
        self._tag_combo.setEnabled(False)
        self._tag_combo.currentTextChanged.connect(self._on_tag_changed)
        tag_row.addWidget(self._tag_combo, stretch=1)
        left_lay.addLayout(tag_row)

        self._tag_note = QLabel("")
        self._tag_note.setStyleSheet("color: #4a9; font-size: 10px; padding: 0 4px;")
        left_lay.addWidget(self._tag_note)

        tag_sep2 = QFrame(); tag_sep2.setFrameShape(QFrame.HLine)
        left_lay.addWidget(tag_sep2)
        # ─────────────────────────────────────────────────────────────────────

        self._btn_edit_holes = QPushButton("✏  Edit Holes")
        self._btn_edit_holes.setEnabled(False)
        self._btn_edit_holes.clicked.connect(self._enter_hole_edit_mode)
        left_lay.addWidget(self._btn_edit_holes)

        # Hole editor (hidden until activated)
        self._hole_panel = HoleEditorPanel(
            on_fill_selected=self._fill_selected_holes,
            on_fill_all=self._fill_all_holes,
            on_save=self._save_holes,
            on_cancel=self._exit_hole_edit_mode,
        )
        self._hole_panel.hide()
        left_lay.addWidget(self._hole_panel)

        splitter.addWidget(left)

        # ── 3D Viewport ───────────────────────────────────────────────────────
        self._plotter = QtInteractor(self)
        self._plotter.set_background("black")
        splitter.addWidget(self._plotter)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Status bar
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)

    # ── status helper ─────────────────────────────────────────────────────────

    def _status(self, msg: str):
        self._statusbar.showMessage(msg)

    # ── part selection ────────────────────────────────────────────────────────

    def _on_part_selected(self, part: PartEntry):
        self._selected_part = part
        self._exit_hole_edit_mode()
        self._update_info_label(part)
        self._btn_edit_holes.setEnabled(part.stp_path is not None)
        self._refresh_3d_scene(part)

    def _update_info_label(self, part: PartEntry):
        lines = [
            f"<b>{part.part_root_name} / body{part.body_idx:03d}</b>",
        ]
        if part.thickness_mm:
            lines.append(f"Thickness: {part.thickness_mm:.3f} mm")
        if part.volume_mm3:
            lines.append(f"Volume: {part.volume_mm3:,.0f} mm³")
        lines.append(f"Faces: {part.face_count}")
        lines.append(f"File: {part.stp_path.name if part.stp_path else 'N/A'}")
        self._info_label.setText("<br>".join(lines))

        # Update tag combo without triggering _on_tag_changed
        self._tag_combo.blockSignals(True)
        idx = TAG_OPTIONS.index(part.tag) if part.tag in TAG_OPTIONS else -1
        if idx >= 0:
            self._tag_combo.setCurrentIndex(idx)
        else:
            # Unknown tag: add it temporarily
            if self._tag_combo.findText(part.tag) == -1:
                self._tag_combo.addItem(part.tag)
            self._tag_combo.setCurrentText(part.tag)
        self._tag_combo.blockSignals(False)
        self._tag_combo.setEnabled(True)
        self._tag_note.setText("")

    # ── tag editing ───────────────────────────────────────────────────────────

    def _on_tag_changed(self, new_tag: str):
        """Called when the user selects a different tag from the combo."""
        part = self._selected_part
        if part is None or new_tag == part.tag:
            return
        old_tag  = part.tag
        part.tag = new_tag          # update in-memory model

        ok = self._save_tag_to_json(part)
        if ok:
            self._tag_note.setText(f"Saved: {old_tag} → {new_tag}")
            self._tree_panel.refresh_preserving_expansion()
            self._tree_panel.highlight_item(part)
        else:
            part.tag = old_tag      # rollback
            self._tag_combo.blockSignals(True)
            self._tag_combo.setCurrentText(old_tag)
            self._tag_combo.blockSignals(False)
            self._tag_note.setText("Save failed — check hierarchy.json")

    def _save_tag_to_json(self, part: PartEntry) -> bool:
        """Write the updated tag back to the part's hierarchy.json."""
        hfile = part.root / "hierarchy.json"
        try:
            data = json.loads(hfile.read_text("utf-8"))
            for b in data.get("bodies", []):
                if b.get("body_idx") == part.body_idx:
                    b["tag"] = part.tag
                    break
            hfile.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return True
        except Exception as e:
            self._status(f"Failed to save tag: {e}")
            return False

    # ── 3D scene management ───────────────────────────────────────────────────

    def _refresh_3d_scene(self, part: PartEntry):
        self._status(f"Loading {part}…")
        QApplication.processEvents()

        self._plotter.clear()
        self._loaded_actors.clear()

        t0 = time.time()

        # Selected body (opaque)
        if part.stp_path:
            try:
                mesh = load_mesh(part.stp_path)
                self._add_body(f"sel_{part.body_idx}", mesh,
                               SELECTED_COLOR, opacity=1.0)
            except Exception as e:
                self._status(f"Error loading {part.stp_path.name}: {e}")
                return

        # Siblings (same direct_parent, 60% opacity)
        siblings = self._store.siblings(part)[:12]
        for s in siblings:
            if s.stp_path:
                try:
                    mesh = load_mesh(s.stp_path)
                    self._add_body(f"sib_{s.body_idx}", mesh,
                                   SIBLING_COLOR, opacity=0.60)
                except Exception:
                    pass

        # Cousins (same subassembly, 20% opacity, limited to 8)
        cousins = self._store.cousins(part)[:8]
        for c in cousins:
            if c.stp_path:
                try:
                    mesh = load_mesh(c.stp_path)
                    self._add_body(f"cou_{c.body_idx}", mesh,
                                   COUSIN_COLOR, opacity=0.20)
                except Exception:
                    pass

        self._plotter.reset_camera()
        elapsed = time.time() - t0
        self._status(
            f"body{part.body_idx:03d}  [{part.tag}]  t={part.thickness_mm:.2f}mm  "
            f"| siblings={len(siblings)}  cousins={len(cousins)}  "
            f"({elapsed:.1f}s)"
        )

    def _add_body(self, name: str, mesh: pv.PolyData,
                  color: tuple, opacity: float):
        actor = self._plotter.add_mesh(
            mesh, color=color, opacity=opacity,
            smooth_shading=True, show_edges=False,
            name=name, reset_camera=False,
        )
        self._loaded_actors[name] = actor

    # ── hole editing ──────────────────────────────────────────────────────────

    def _enter_hole_edit_mode(self):
        if not self._selected_part or not self._selected_part.stp_path:
            return

        self._status("Detecting holes…")
        QApplication.processEvents()

        try:
            shape = _read_step(self._selected_part.stp_path)
            holes = detect_holes(shape)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read STP:\n{e}")
            return

        self._hole_shapes = holes
        self._pending_holes = []
        self._edit_mode = True

        self._hole_panel.load_holes(holes)
        self._hole_panel.show()
        self._btn_edit_holes.setEnabled(False)

        # Show hole spheres in viewport
        self._draw_hole_spheres(selected_indices=[])

        n = len(holes)
        self._status(f"Hole edit mode: {n} holes detected in body{self._selected_part.body_idx:03d}")

    def _exit_hole_edit_mode(self):
        if not self._edit_mode:
            return
        self._edit_mode = False
        self._hole_panel.hide()
        self._btn_edit_holes.setEnabled(self._selected_part is not None
                                        and self._selected_part.stp_path is not None)
        # Remove hole actors
        for name in list(self._loaded_actors.keys()):
            if name.startswith("hole_"):
                self._plotter.remove_actor(name)
                del self._loaded_actors[name]
        self._hole_shapes = []
        self._pending_holes = []
        self._status("Hole edit cancelled.")

    def _draw_hole_spheres(self, selected_indices: list[int]):
        """Update hole sphere actors based on current selection."""
        for i, hole in enumerate(self._hole_shapes):
            name  = f"hole_{i}"
            color = HOLE_COLOR_SELECTED if i in selected_indices else HOLE_COLOR_NORMAL
            # Remove old actor
            if name in self._loaded_actors:
                self._plotter.remove_actor(name)
            # Draw sphere at hole center, radius = hole_radius * 0.6 for visual
            r = hole["radius_mm"] * 0.6
            sphere = pv.Sphere(center=hole["center_xyz"], radius=max(r, 2.0))
            # Encode hole index as a scalar so picking can identify it
            sphere["hole_idx"] = np.full(sphere.n_points, i, dtype=np.int32)
            actor = self._plotter.add_mesh(
                sphere, color=color, opacity=0.85,
                name=name, reset_camera=False,
            )
            self._loaded_actors[name] = actor

    def _get_checked_hole_indices(self) -> list[int]:
        return self._hole_panel.get_selected_indices()

    def _fill_selected_holes(self):
        idxs = self._get_checked_hole_indices()
        if not idxs:
            QMessageBox.information(self, "No holes selected",
                                    "Check at least one hole in the list first.")
            return
        self._pending_holes = [self._hole_shapes[i] for i in idxs]
        self._draw_hole_spheres(selected_indices=idxs)
        self._hole_panel.set_note(f"{len(idxs)} hole(s) marked for filling. Press Save.")

    def _fill_all_holes(self):
        self._hole_panel.select_all()
        self._pending_holes = list(self._hole_shapes)
        self._draw_hole_spheres(selected_indices=list(range(len(self._hole_shapes))))
        self._hole_panel.set_note(f"All {len(self._hole_shapes)} hole(s) marked. Press Save.")

    def _save_holes(self):
        if not self._pending_holes:
            QMessageBox.information(self, "Nothing to fill",
                                    "Mark some holes first (Fill Selected or Fill All).")
            return
        part  = self._selected_part
        stp   = part.stp_path

        # Backup
        bak = stp.parent / (stp.stem + ".stp.bak")
        if not bak.exists():
            shutil.copy2(stp, bak)

        self._status("Running defeaturing (hole fill)…")
        QApplication.processEvents()

        try:
            shape = _read_step(stp)
            result, fill_results = fill_holes(shape, self._pending_holes)
            failed = [r for r in fill_results if r["fill_status"] != "ok"]
            if failed:
                lines = "\n".join(
                    f"  - {r['diameter_mm']:.1f}mm {r['category']} hole: {r['fill_status']}"
                    for r in failed
                )
                QMessageBox.warning(
                    self, "Partial success",
                    f"{len(failed)}/{len(fill_results)} hole(s) could not be filled "
                    f"and were left unchanged:\n{lines}\n\n"
                    f"All other selected holes were filled successfully.")
            _write_step(result, stp)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
            return

        # Invalidate mesh cache so next load re-tessellates
        cache = stp.with_suffix(MESH_CACHE_SUFFIX)
        if cache.exists():
            cache.unlink()

        self._hole_panel.mark_saved()
        self._status(f"Saved: {stp.name}  (backup: {bak.name})")

        # Refresh 3D scene with updated solid
        self._exit_hole_edit_mode()
        if part:
            QTimer.singleShot(100, lambda: self._refresh_3d_scene(part))

    # ── close ─────────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._plotter.close()
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="BIW STP Viewer with Hole Editor")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("hierarchy",     nargs="?",  help="hierarchy.json file")
    src.add_argument("--dir",         metavar="PATH",
                     help="Output root dir (scans for all hierarchy.json)")
    ap.add_argument("--filter",       default=None,
                    choices=["sheet_metal", "thick_part", "small_hardware"],
                    help="Show only parts with this tag")
    args = ap.parse_args()

    store = HierarchyStore()

    if args.hierarchy:
        p = pathlib.Path(args.hierarchy)
        if not p.exists():
            print(f"ERROR: not found: {p}"); sys.exit(1)
        store.load_file(p, tag_filter=args.filter)
    else:
        d = pathlib.Path(args.dir)
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}"); sys.exit(1)
        store.load_dir(d, tag_filter=args.filter)

    print(f"Loaded {len(store.parts)} parts.")

    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow(store)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

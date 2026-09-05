"""Part tree panel: hierarchy.json-driven, grouped by tag, with a text filter.

Follows the legacy stp_viewer.py UX pattern (design doc SS6.1) but groups by
``tag`` rather than CATIA-hierarchy path, since this tool's adjacency model
(SS6.2) is geometric, not hierarchical -- the tag grouping is purely a
browsing aid, not a relationship the rest of the tool relies on.
"""

from __future__ import annotations

from typing import Callable

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QLineEdit, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from annotation_tool.assembly import AssemblyStore
from annotation_tool.hierarchy import HierarchyBody

TAG_COLORS = {
    "sheet_metal": QColor(140, 190, 240),
    "thick_part": QColor(190, 190, 190),
    "small_hardware": QColor(200, 165, 100),
}
DEFAULT_TAG_COLOR = QColor(215, 215, 215)


def body_label(body: HierarchyBody) -> str:
    thickness = f"  t={body.thickness_est_mm:.2f}mm" if body.thickness_est_mm else ""
    return f"{body.part_id}{thickness}"


class PartTreePanel(QWidget):
    def __init__(
        self,
        store: AssemblyStore,
        on_select: Callable[[HierarchyBody], None],
        on_visibility_change: Callable[[HierarchyBody, bool], None] | None = None,
    ):
        super().__init__()
        self._store = store
        self._on_select = on_select
        self._on_visibility_change = on_visibility_change or (lambda b, v: None)
        # Flag set by _on_item_changed (checkbox toggle) so _on_item_clicked can
        # distinguish a checkbox click from a text-area click and skip focus selection.
        self._check_state_just_changed = False
        self._tree = QTreeWidget()
        self._filter = QLineEdit()
        self._build()
        self.refresh()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._filter.setPlaceholderText("Filter by part id / tag...")
        self._filter.textChanged.connect(self._apply_filter)
        layout.addWidget(self._filter)

        self._tree.setHeaderLabel("Parts")
        self._tree.setColumnCount(1)
        self._tree.itemClicked.connect(self._on_item_clicked)
        self._tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._tree)

    def refresh(self) -> None:
        self._tree.blockSignals(True)
        self._tree.clear()
        for tag, bodies in self._store.bodies_by_tag().items():
            tag_item = QTreeWidgetItem(self._tree, [f"{tag} ({len(bodies)})"])
            tag_item.setFlags(tag_item.flags() & ~Qt.ItemIsSelectable)
            font = tag_item.font(0)
            font.setBold(True)
            tag_item.setFont(0, font)

            for body in bodies:
                item = QTreeWidgetItem(tag_item, [body_label(body)])
                item.setData(0, Qt.UserRole, body)
                item.setForeground(0, TAG_COLORS.get(tag, DEFAULT_TAG_COLOR))
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(0, Qt.Checked)

        self._tree.expandAll()
        self._tree.blockSignals(False)

    def set_checked_parts(self, part_ids: set[str]) -> None:
        """Check only parts in part_ids; uncheck all others. Does not emit visibility signals."""
        self._tree.blockSignals(True)
        for i in range(self._tree.topLevelItemCount()):
            tag_item = self._tree.topLevelItem(i)
            for j in range(tag_item.childCount()):
                child = tag_item.child(j)
                body = child.data(0, Qt.UserRole)
                if isinstance(body, HierarchyBody):
                    state = Qt.Checked if body.part_id in part_ids else Qt.Unchecked
                    child.setCheckState(0, state)
        self._tree.blockSignals(False)

    def check_all(self) -> None:
        """Restore all parts to checked state. Does not emit visibility signals."""
        self._tree.blockSignals(True)
        for i in range(self._tree.topLevelItemCount()):
            tag_item = self._tree.topLevelItem(i)
            for j in range(tag_item.childCount()):
                child = tag_item.child(j)
                child.setCheckState(0, Qt.Checked)
        self._tree.blockSignals(False)

    def _on_item_changed(self, item: QTreeWidgetItem, _column: int) -> None:
        body = item.data(0, Qt.UserRole)
        if not isinstance(body, HierarchyBody):
            return
        # Mark that a checkbox state change just happened so the immediately
        # following itemClicked (which Qt also fires for checkbox clicks) can
        # skip triggering focus selection.
        self._check_state_just_changed = True
        visible = item.checkState(0) == Qt.Checked
        self._on_visibility_change(body, visible)

    def _on_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        if self._check_state_just_changed:
            self._check_state_just_changed = False
            return
        body = item.data(0, Qt.UserRole)
        if isinstance(body, HierarchyBody):
            self._on_select(body)

    def _apply_filter(self, text: str) -> None:
        text = text.lower()
        for i in range(self._tree.topLevelItemCount()):
            tag_item = self._tree.topLevelItem(i)
            any_visible = False
            for j in range(tag_item.childCount()):
                child = tag_item.child(j)
                body: HierarchyBody = child.data(0, Qt.UserRole)
                visible = text in body.part_id.lower() or text in body.tag.lower()
                child.setHidden(not visible)
                any_visible = any_visible or visible
            tag_item.setHidden(not any_visible)

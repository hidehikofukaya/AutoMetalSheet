"""Hole list panel — left-side panel showing detected holes with checkboxes."""
from __future__ import annotations

from typing import Callable

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Colour shown for holes already confirmed by joints.json annotation
_ANNOTATED_COLOR = QColor(200, 155, 0)
# Colour shown for holes already closed by Mesh_Generater (feature caps)
_CLOSED_COLOR = QColor(140, 180, 240)


class HoleListPanel(QWidget):
    """Scrollable list of detected holes with per-hole checkboxes.

    Parameters
    ----------
    on_toggle:
        Called with ``(hole_index, is_selected)`` whenever a checkbox changes.
    """

    def __init__(
        self,
        on_toggle: Callable[[int, bool], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_toggle = on_toggle
        self._holes: list[dict] = []
        self._selected: set[int] = set()
        self._annotated: set[int] = set()
        self._build()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_holes(
        self,
        holes: list[dict],
        selected: set[int],
        annotated: set[int],
    ) -> None:
        """Populate the list from a fresh detect_holes() result."""
        self._holes = holes
        self._selected = set(selected)
        self._annotated = set(annotated)
        self._rebuild()

    def update_selection(self, idx: int, is_selected: bool) -> None:
        """Sync checkbox state when toggled from the 3-D view."""
        self._list.blockSignals(True)
        item = self._list.item(idx)
        if item is not None:
            item.setCheckState(Qt.Checked if is_selected else Qt.Unchecked)
        if is_selected:
            self._selected.add(idx)
        else:
            self._selected.discard(idx)
        self._update_label()
        self._list.blockSignals(False)

    def select_all(self) -> None:
        self._list.blockSignals(True)
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(Qt.Checked)
        self._selected = set(range(len(self._holes)))
        self._update_label()
        self._list.blockSignals(False)
        for i in range(len(self._holes)):
            self._on_toggle(i, True)

    def deselect_all(self) -> None:
        self._list.blockSignals(True)
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(Qt.Unchecked)
        self._selected = set()
        self._update_label()
        self._list.blockSignals(False)
        for i in range(len(self._holes)):
            self._on_toggle(i, False)

    @property
    def selected_indices(self) -> set[int]:
        return set(self._selected)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._summary = QLabel("穴なし")
        layout.addWidget(self._summary)

        legend = QHBoxLayout()
        legend.setSpacing(6)
        for color, text in [
            ("#c8b400", "★ アノテ済"),
            ("#8cb4f0", "(済) 充填済"),
        ]:
            lbl = QLabel(f'<span style="color:{color}">{text}</span>')
            legend.addWidget(lbl)
        legend.addStretch()
        layout.addLayout(legend)

        self._list = QListWidget()
        self._list.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._list)

    def _rebuild(self) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        for i, h in enumerate(self._holes):
            ann = "★ " if i in self._annotated else "   "
            closed = "(済)" if h.get("_source") == "feature" else "    "
            label = (
                f"{ann}{closed}  "
                f"r={h['radius_mm']:.1f}mm  "
                f"{h['shape']}  "
                f"{h['circularity']:.0%}"
            )
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if i in self._selected else Qt.Unchecked)
            item.setData(Qt.UserRole, i)
            if i in self._annotated:
                item.setForeground(_ANNOTATED_COLOR)
            elif h.get("_source") == "feature":
                item.setForeground(_CLOSED_COLOR)
            self._list.addItem(item)
        self._list.blockSignals(False)
        self._update_label()

    def _update_label(self) -> None:
        n = len(self._holes)
        sel = len(self._selected)
        ann = len(self._annotated)
        parts = [f"検出: {n}個", f"選択: {sel}個"]
        if ann:
            parts.append(f"★{ann}個")
        self._summary.setText("  ".join(parts))

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        idx: int = item.data(Qt.UserRole)
        is_sel = item.checkState() == Qt.Checked
        if is_sel:
            self._selected.add(idx)
        else:
            self._selected.discard(idx)
        self._update_label()
        self._on_toggle(idx, is_sel)

"""Joint list panel: shows joints referencing the currently focused part."""

from __future__ import annotations

from typing import Callable

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget

from annotation_tool.schema import AnnotationDocument, Joint


class JointPanel(QWidget):
    def __init__(
        self,
        doc: AnnotationDocument,
        on_delete: Callable[[str], None],
        on_select: Callable[[Joint], None] | None = None,
    ):
        super().__init__()
        self._doc = doc
        self._on_delete = on_delete
        self._on_select = on_select or (lambda j: None)
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._header = QLabel("Joints")
        font = self._header.font()
        font.setBold(True)
        self._header.setFont(font)
        layout.addWidget(self._header)

        self._list = QListWidget()
        self._list.setMaximumHeight(160)
        self._list.currentItemChanged.connect(self._on_item_changed)
        layout.addWidget(self._list)

        self._delete_btn = QPushButton("Delete selected joint")
        self._delete_btn.setEnabled(False)
        self._delete_btn.clicked.connect(self._on_delete_clicked)
        layout.addWidget(self._delete_btn)

    def refresh(self, part_id: str | None) -> None:
        self._list.clear()
        self._delete_btn.setEnabled(False)
        if part_id is None:
            self._header.setText("Joints (no part selected)")
            return
        joints = self._doc.joints_for_part(part_id)
        self._header.setText(f"Joints for {part_id} ({len(joints)})")
        for joint in joints:
            partners = [p for p in joint.parts if p != part_id]
            partner_str = ", ".join(partners) if partners else "—"
            item = QListWidgetItem(f"{joint.joint_id}  [{joint.type}]  ↔ {partner_str}")
            item.setData(Qt.UserRole, joint.joint_id)
            self._list.addItem(item)

    def _on_item_changed(self, current: QListWidgetItem | None, _prev) -> None:
        has_sel = current is not None
        self._delete_btn.setEnabled(has_sel)
        if has_sel:
            jid = current.data(Qt.UserRole)
            if jid in self._doc.joints:
                self._on_select(self._doc.joints[jid])

    def _on_delete_clicked(self) -> None:
        item = self._list.currentItem()
        if item:
            self._on_delete(item.data(Qt.UserRole))

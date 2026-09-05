"""Confirmation dialog for a new joint.

Type and partner are determined BEFORE this dialog opens (by the picking flow
in MainWindow).  This dialog only shows the gathered info and asks for final
confirmation — no re-selection of mode or partner.
"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)


class JointCreationDialog(QDialog):
    def __init__(
        self,
        joint_type: str,
        contact_point: tuple[float, float, float],
        partner_id: str | None = None,
        hole_info: dict | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._joint_type = joint_type
        self._contact_point = contact_point
        self._partner_id = partner_id
        self._hole_info = hole_info
        self.setWindowTitle("Confirm Joint")
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        form = QFormLayout()
        x, y, z = self._contact_point
        form.addRow("Joint type:",    QLabel(self._joint_type))
        form.addRow("Contact point:", QLabel(f"({x:.2f}, {y:.2f}, {z:.2f}) mm"))
        form.addRow("Partner part:",  QLabel(self._partner_id or "— (none)"))
        if self._hole_info:
            form.addRow(
                "Hole diameter:",
                QLabel(f"{self._hole_info['radius_mm'] * 2:.2f} mm"),
            )
            form.addRow(
                "Circularity:",
                QLabel(f"{self._hole_info['circularity']:.1%}"),
            )
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def joint_type(self) -> str:
        return self._joint_type

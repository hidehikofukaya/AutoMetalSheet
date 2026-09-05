"""GUI for cohort-level R0 family/part Go results with drill-down to run viewer."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from r0_result_viewer import R0Viewer


PASS_COLOR = QColor("#1f8f4e")
FAIL_COLOR = QColor("#b3261e")
WARN_COLOR = QColor("#9a6700")


class CohortViewer(QMainWindow):
    def __init__(self, cohort_path: pathlib.Path | None = None):
        super().__init__()
        self.cohort_path: pathlib.Path | None = None
        self.cohort: dict = {}
        self.child_windows: list[R0Viewer] = []
        self.setWindowTitle("AutoMetalSheet R0 Cohort Go/No-Go Viewer")
        self.resize(1350, 820)
        self._build()
        if cohort_path:
            self.open_cohort(cohort_path)

    def _build(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        toolbar = QHBoxLayout()
        open_button = QPushButton("Open cohort_manifest.json")
        open_button.clicked.connect(self.choose_cohort)
        toolbar.addWidget(open_button)
        self.path_label = QLabel("No cohort loaded")
        toolbar.addWidget(self.path_label, 1)
        layout.addLayout(toolbar)

        self.status_label = QLabel("Phase gate: -")
        self.status_label.setStyleSheet("font-size: 20px; font-weight: bold; padding: 8px")
        layout.addWidget(self.status_label)
        self.summary_label = QLabel("-")
        layout.addWidget(self.summary_label)

        splitter = QSplitter(Qt.Horizontal)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Family / Part", "Status", "Improvement", "Failures"])
        self.tree.itemDoubleClicked.connect(self.open_selected_run)
        splitter.addWidget(self.tree)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.family_table = QTableWidget(0, 4)
        self.family_table.setHorizontalHeaderLabels(
            ["Family", "Parts", "Mean improvement", "Hard failures"]
        )
        self.family_table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self.family_table)
        self.reason_text = QTextEdit()
        self.reason_text.setReadOnly(True)
        self.reason_text.setMaximumHeight(130)
        right_layout.addWidget(self.reason_text)
        drill_button = QPushButton("Open selected part in 3D run viewer")
        drill_button.clicked.connect(self.open_selected_run)
        right_layout.addWidget(drill_button)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)
        self.setCentralWidget(root)

    def choose_cohort(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open R0 cohort manifest", "", "JSON (*.json)"
        )
        if filename:
            self.open_cohort(pathlib.Path(filename))

    def open_cohort(self, path: pathlib.Path):
        self.cohort_path = path.resolve()
        self.cohort = json.loads(self.cohort_path.read_text(encoding="utf-8"))
        if self.cohort.get("run_kind") != "R0_COHORT_AGGREGATE":
            raise ValueError("Not an R0 cohort manifest")
        self.path_label.setText(str(self.cohort_path))
        status = self.cohort["phase_gate_status"]
        color = (
            PASS_COLOR
            if status == "TECHNICAL_GO"
            else FAIL_COLOR if status == "NO_GO" else WARN_COLOR
        )
        self.status_label.setText(f"Phase gate: {status}")
        self.status_label.setStyleSheet(
            f"font-size: 20px; font-weight: bold; padding: 8px; color: {color.name()}"
        )
        boot = self.cohort["cluster_bootstrap"]
        self.summary_label.setText(
            f"Parts: {self.cohort['part_count']}  Families: {self.cohort['family_count']}  "
            f"Improvement: {boot['estimate']:.4g}  "
            f"95% CI [{boot['ci95_low']:.4g}, {boot['ci95_high']:.4g}]"
        )
        self.reason_text.setPlainText(
            "\n".join(self.cohort.get("phase_gate_reasons", [])) or "No blocking reason"
        )

        by_family: dict[str, list[dict]] = {}
        for part in self.cohort["parts"]:
            by_family.setdefault(part["part_family"], []).append(part)
        self.tree.clear()
        for family in sorted(by_family):
            family_item = QTreeWidgetItem([family, "", "", ""])
            self.tree.addTopLevelItem(family_item)
            for part in sorted(by_family[family], key=lambda item: item["part_id"]):
                failed = part["candidate_failed_gate_ids"]
                status_text = "PASS" if not failed else "HARD_GATE_FAILED"
                item = QTreeWidgetItem(
                    [
                        part["part_id"],
                        status_text,
                        f"{part['primary_improvement']:.4g}",
                        str(len(failed)),
                    ]
                )
                item.setData(0, Qt.UserRole, part["manifest_path"])
                item.setForeground(1, PASS_COLOR if not failed else FAIL_COLOR)
                family_item.addChild(item)
            family_item.setExpanded(True)
        self.tree.resizeColumnToContents(0)

        self.family_table.setRowCount(len(self.cohort["families"]))
        for row, family in enumerate(self.cohort["families"]):
            values = [
                family["part_family"],
                family["part_count"],
                family["mean_primary_improvement"],
                family["hard_failure_count"],
            ]
            for column, value in enumerate(values):
                self.family_table.setItem(row, column, QTableWidgetItem(str(value)))

    def open_selected_run(self, *_):
        item = self.tree.currentItem()
        if item is None:
            return
        manifest_path = item.data(0, Qt.UserRole)
        if not manifest_path:
            return
        window = R0Viewer(pathlib.Path(manifest_path))
        window.show()
        self.child_windows.append(window)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=pathlib.Path)
    parser.add_argument("--screenshot", type=pathlib.Path)
    args = parser.parse_args()
    app = QApplication.instance() or QApplication(sys.argv)
    viewer = CohortViewer(args.cohort)
    viewer.show()
    if args.screenshot:
        def capture():
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            viewer.grab().save(str(args.screenshot))
            viewer.close()
            app.quit()
        QTimer.singleShot(1500, capture)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

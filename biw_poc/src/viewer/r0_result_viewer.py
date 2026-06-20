"""GUI for inspecting R0 run bundles, stage artifacts, metrics and Go status."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


PASS_COLOR = QColor("#1f8f4e")
FAIL_COLOR = QColor("#b3261e")
WARN_COLOR = QColor("#9a6700")


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def verify_bundle(manifest_path: pathlib.Path, manifest: dict) -> list[str]:
    errors: list[str] = []
    run_dir = manifest_path.parent

    def check(path: pathlib.Path, expected: str, label: str):
        if not path.exists():
            errors.append(f"{label} missing: {path}")
        elif sha256_file(path) != expected:
            errors.append(f"{label} SHA-256 mismatch: {path}")

    check(
        pathlib.Path(manifest["profile_source"]["path"]),
        manifest["profile_source"]["sha256"],
        "profile",
    )
    check(
        pathlib.Path(manifest["reference"]["path"]),
        manifest["reference"]["sha256"],
        "reference",
    )
    for stage in manifest.get("stages", []):
        check(
            pathlib.Path(stage["artifact"]["path"]),
            stage["artifact"]["sha256"],
            f"stage {stage['label']}",
        )
        check(
            run_dir / stage["metrics_path"],
            stage["metrics_sha256"],
            f"metrics {stage['label']}",
        )
        metrics_path = run_dir / stage["metrics_path"]
        if metrics_path.exists():
            metrics = load_json(metrics_path)
            for direction in ("stage_to_reference", "reference_to_stage"):
                artifact = metrics.get(direction, {}).get("sample_artifact")
                if artifact:
                    check(
                        run_dir / artifact["path"],
                        artifact["sha256"],
                        f"samples {stage['label']} {direction}",
                    )

    audit = manifest.get("audit", {})
    audit_path = run_dir / audit.get("log_path", "")
    if audit.get("log_sha256"):
        check(audit_path, audit["log_sha256"], "audit log")
    if audit_path.exists():
        events = [
            json.loads(line)
            for line in audit_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        previous = None
        for expected_sequence, stored in enumerate(events, start=1):
            item = dict(stored)
            event_hash = item.pop("event_hash", None)
            if item.get("sequence_no") != expected_sequence:
                errors.append("AuditEvent sequence mismatch")
                break
            if item.get("previous_event_hash") != previous:
                errors.append("AuditEvent previous hash mismatch")
                break
            calculated = hashlib.sha256(canonical_json_bytes(item)).hexdigest()
            if calculated != event_hash:
                errors.append("AuditEvent content hash mismatch")
                break
            previous = event_hash
        if previous != audit.get("final_event_hash"):
            errors.append("AuditEvent final hash mismatch")
    return errors


def display_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


class R0Viewer(QMainWindow):
    def __init__(self, manifest_path: pathlib.Path | None = None):
        super().__init__()
        self.manifest_path: pathlib.Path | None = None
        self.manifest: dict = {}
        self.metrics: dict[str, dict] = {}
        self.mesh_cache: dict[str, pv.PolyData] = {}
        self.setWindowTitle("AutoMetalSheet R0 Go/No-Go Viewer")
        self.resize(1680, 960)
        self._build()
        if manifest_path:
            self.open_manifest(manifest_path)

    def _build(self):
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(4, 4, 4, 4)

        toolbar = QHBoxLayout()
        open_button = QPushButton("Open run_manifest.json")
        open_button.clicked.connect(self.choose_manifest)
        toolbar.addWidget(open_button)
        self.path_label = QLabel("No run loaded")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        toolbar.addWidget(self.path_label, 1)
        root_layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter, 1)

        left = QWidget()
        left.setMinimumWidth(390)
        left_layout = QVBoxLayout(left)
        self.status_label = QLabel("Run status: -")
        self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 8px;")
        left_layout.addWidget(self.status_label)

        stage_box = QGroupBox("Stage outputs")
        stage_layout = QVBoxLayout(stage_box)
        self.stage_tree = QTreeWidget()
        self.stage_tree.setHeaderLabels(["Stage", "Status", "Vertices", "Faces"])
        self.stage_tree.itemChanged.connect(self.refresh_scene)
        self.stage_tree.currentItemChanged.connect(self.refresh_details)
        stage_layout.addWidget(self.stage_tree)
        left_layout.addWidget(stage_box, 1)

        self.reason_text = QTextEdit()
        self.reason_text.setReadOnly(True)
        self.reason_text.setMaximumHeight(120)
        left_layout.addWidget(QLabel("Phase decision reasons"))
        left_layout.addWidget(self.reason_text)
        splitter.addWidget(left)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        self.plotter = QtInteractor(center)
        self.plotter.set_background("#16181c")
        center_layout.addWidget(self.plotter, 1)
        splitter.addWidget(center)

        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(460)
        self.metric_table = QTableWidget(0, 2)
        self.metric_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metric_table.horizontalHeader().setStretchLastSection(True)
        self.tabs.addTab(self.metric_table, "Metrics")
        self.gate_table = QTableWidget(0, 5)
        self.gate_table.setHorizontalHeaderLabels(
            ["Gate", "Actual", "Rule", "Threshold", "Result"]
        )
        self.gate_table.horizontalHeader().setStretchLastSection(True)
        self.tabs.addTab(self.gate_table, "Hard gates")
        self.manifest_text = QTextEdit()
        self.manifest_text.setReadOnly(True)
        self.manifest_text.setFontFamily("Consolas")
        self.tabs.addTab(self.manifest_text, "Manifest")
        splitter.addWidget(self.tabs)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        self.setCentralWidget(root)

    def choose_manifest(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open R0 run manifest", "", "JSON (*.json)"
        )
        if filename:
            self.open_manifest(pathlib.Path(filename))

    def open_manifest(self, path: pathlib.Path):
        self.manifest_path = path.resolve()
        self.manifest = load_json(self.manifest_path)
        integrity_errors = verify_bundle(self.manifest_path, self.manifest)
        self.metrics.clear()
        self.mesh_cache.clear()
        self.path_label.setText(str(self.manifest_path))
        self.manifest_text.setPlainText(
            json.dumps(self.manifest, ensure_ascii=False, indent=2)
        )

        phase_status = self.manifest.get("phase_gate_status", "UNKNOWN")
        effective_status = "BUNDLE_INVALID" if integrity_errors else phase_status
        self.status_label.setText(f"Phase gate: {effective_status}")
        if effective_status == "TECHNICAL_GO":
            color = PASS_COLOR.name()
        elif effective_status in {"NO_GO", "BUNDLE_INVALID"}:
            color = FAIL_COLOR.name()
        else:
            color = WARN_COLOR.name()
        self.status_label.setStyleSheet(
            f"font-size: 18px; font-weight: bold; padding: 8px; color: {color};"
        )
        reasons = integrity_errors + self.manifest.get("phase_gate_reasons", [])
        self.reason_text.setPlainText("\n".join(reasons) or "No blocking reason")

        self.stage_tree.blockSignals(True)
        self.stage_tree.clear()
        reference_item = QTreeWidgetItem(["REFERENCE", "-", "-", "-"])
        reference_item.setData(0, Qt.UserRole, {"kind": "reference"})
        reference_item.setFlags(reference_item.flags() | Qt.ItemIsUserCheckable)
        reference_item.setCheckState(0, Qt.Checked)
        self.stage_tree.addTopLevelItem(reference_item)

        run_dir = self.manifest_path.parent
        stages = self.manifest.get("stages", [])
        for stage_index, stage in enumerate(stages):
            metrics = load_json(run_dir / stage["metrics_path"])
            self.metrics[stage["label"]] = metrics
            item = QTreeWidgetItem(
                [
                    stage["label"],
                    stage["status"],
                    str(metrics.get("vertex_count", "-")),
                    str(metrics.get("face_count", "-")),
                ]
            )
            item.setData(0, Qt.UserRole, {"kind": "stage", "stage": stage})
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(
                0, Qt.Checked if stage_index == len(stages) - 1 else Qt.Unchecked
            )
            item.setForeground(
                1, PASS_COLOR if stage["status"] == "PASS" else FAIL_COLOR
            )
            self.stage_tree.addTopLevelItem(item)
        self.stage_tree.blockSignals(False)
        self.stage_tree.resizeColumnToContents(0)
        self.stage_tree.setCurrentItem(
            self.stage_tree.topLevelItem(self.stage_tree.topLevelItemCount() - 1)
            if self.stage_tree.topLevelItemCount() > 1
            else reference_item
        )
        self.refresh_scene()

    def _mesh_for(self, key: str, path: str) -> pv.PolyData:
        if key not in self.mesh_cache:
            self.mesh_cache[key] = pv.read(path)
        return self.mesh_cache[key]

    def refresh_scene(self, *_):
        if not self.manifest:
            return
        self.plotter.clear()
        reference = self._mesh_for("reference", self.manifest["reference"]["path"])
        stage_colors = ["#4cc9f0", "#f9c74f", "#f9844a", "#90be6d", "#b5179e"]
        shown = 0
        for index in range(self.stage_tree.topLevelItemCount()):
            item = self.stage_tree.topLevelItem(index)
            if item.checkState(0) != Qt.Checked:
                continue
            data = item.data(0, Qt.UserRole)
            if data["kind"] == "reference":
                self.plotter.add_mesh(
                    reference,
                    color="#d9d9d9",
                    opacity=0.22,
                    show_edges=False,
                    name="reference",
                )
            else:
                stage = data["stage"]
                mesh = self._mesh_for(stage["label"], stage["artifact"]["path"]).copy()
                # VTK computes signed point-to-triangle-surface distance; display magnitude.
                mesh.compute_implicit_distance(reference, inplace=True)
                mesh["distance_to_reference_mm"] = np.abs(
                    np.asarray(mesh["implicit_distance"])
                )
                self.plotter.add_mesh(
                    mesh,
                    scalars="distance_to_reference_mm",
                    cmap="turbo",
                    clim=(0.0, max(5.0, float(np.percentile(mesh["distance_to_reference_mm"], 95)))),
                    show_edges=False,
                    name=f"stage_{stage['label']}",
                    scalar_bar_args={"title": "|distance to reference| mm"},
                )
            shown += 1
        if shown:
            self.plotter.reset_camera()
        self.plotter.render()

    def refresh_details(self, current: QTreeWidgetItem | None, _previous=None):
        self.metric_table.setRowCount(0)
        self.gate_table.setRowCount(0)
        if current is None:
            return
        data = current.data(0, Qt.UserRole)
        if not data or data["kind"] != "stage":
            return
        stage = data["stage"]
        metrics = self.metrics[stage["label"]]
        flattened = []
        for key, value in metrics.items():
            if isinstance(value, dict):
                for child_key, child_value in value.items():
                    flattened.append((f"{key}.{child_key}", child_value))
            else:
                flattened.append((key, value))
        self.metric_table.setRowCount(len(flattened))
        for row, (key, value) in enumerate(flattened):
            self.metric_table.setItem(row, 0, QTableWidgetItem(key))
            self.metric_table.setItem(row, 1, QTableWidgetItem(display_value(value)))

        gates = stage.get("gates", [])
        self.gate_table.setRowCount(len(gates))
        for row, gate in enumerate(gates):
            values = [
                gate["gate_id"],
                display_value(gate["actual"]),
                gate["operator"],
                display_value(gate["threshold"]),
                "PASS" if gate["passed"] else "FAIL",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 4:
                    item.setForeground(PASS_COLOR if gate["passed"] else FAIL_COLOR)
                self.gate_table.setItem(row, col, item)

    def closeEvent(self, event):
        self.plotter.close()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument("--screenshot", type=pathlib.Path)
    args = parser.parse_args()
    app = QApplication.instance() or QApplication(sys.argv)
    viewer = R0Viewer(args.manifest)
    viewer.show()
    QTimer.singleShot(250, viewer.refresh_scene)
    if args.screenshot:
        def capture():
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            viewer.plotter.screenshot(str(args.screenshot))
            viewer.close()
            app.quit()
        QTimer.singleShot(5000, capture)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

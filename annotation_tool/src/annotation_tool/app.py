"""Entry point: ``python -m annotation_tool.app <assembly_dir>``.

``assembly_dir`` is a ``Mesh_Generater/output/<assembly_id>/`` directory
containing ``hierarchy.json`` (design doc SS2/SS3).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from PyQt5.QtWidgets import QApplication

from annotation_tool.assembly import AssemblyStore
from annotation_tool.schema import AnnotationDocument
from annotation_tool.ui.main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Joint annotation tool")
    parser.add_argument("assembly_dir", help="Path to <assembly_id> dir containing hierarchy.json")
    args = parser.parse_args(argv)

    assembly_dir = pathlib.Path(args.assembly_dir)
    if not (assembly_dir / "hierarchy.json").exists():
        print(f"ERROR: hierarchy.json not found in {assembly_dir}")
        return 1

    store = AssemblyStore.load(assembly_dir)
    print(f"Loaded {len(store.visible_bodies)} parts from {assembly_dir}")

    joints_json = assembly_dir / "annotations" / "joints.json"
    if joints_json.exists():
        doc = AnnotationDocument.load(assembly_dir)
        print(f"Loaded {len(doc.joints)} existing joints from {joints_json}")
    else:
        doc = AnnotationDocument.new_for_assembly(assembly_dir, store.hierarchy)
        print("No existing annotations — starting fresh document")

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(store, doc)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

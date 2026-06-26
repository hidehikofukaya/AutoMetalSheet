"""Entry point: ``python run_hole_filler.py <assembly_dir>``."""
from __future__ import annotations

import argparse
import pathlib
import sys

from PyQt5.QtWidgets import QApplication

from annotation_tool.assembly import AssemblyStore
from hole_filler.ui.main_window import HoleFillerWindow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hole filler tool")
    parser.add_argument(
        "assembly_dir",
        help="Path to <assembly_id> directory containing hierarchy.json",
    )
    args = parser.parse_args(argv)

    assembly_dir = pathlib.Path(args.assembly_dir)
    if not (assembly_dir / "hierarchy.json").exists():
        print(f"ERROR: hierarchy.json not found in {assembly_dir}")
        return 1

    store = AssemblyStore.load(assembly_dir)
    print(f"Loaded {len(store.visible_bodies)} parts from {assembly_dir}")

    app = QApplication.instance() or QApplication(sys.argv)
    window = HoleFillerWindow(store, assembly_dir)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

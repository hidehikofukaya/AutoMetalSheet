"""Launcher: run from the ``annotation_tool/`` project root.

    python run_annotation_tool.py <assembly_dir>

``annotation_tool.app`` uses absolute imports (``from annotation_tool...``),
so the package's parent dir (``src/``) must be on ``sys.path`` -- this script
adds it before delegating to ``annotation_tool.app.main()``.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from annotation_tool.app import main  # noqa: E402 (path setup must run first)

if __name__ == "__main__":
    sys.exit(main())

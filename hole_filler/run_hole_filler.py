"""Launch the hole-filler tool.

Usage
-----
    python run_hole_filler.py <assembly_dir>

Example
-------
    python run_hole_filler.py "C:/Users/hide2/IdeaBox/Mesh_Generater/output/A0072601285"
"""
import pathlib
import sys

# Ensure both packages are importable when run without pip install
_root = pathlib.Path(__file__).parent
for _p in [
    _root / "src",
    _root.parent / "annotation_tool" / "src",
]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hole_filler.app import main  # noqa: E402

sys.exit(main())

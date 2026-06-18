"""Resume the hole-fill batch across the output tree for the assemblies that
were not yet processed (the first 9 of 24 already have correctly-formatted
_filled.stp/_filled_holes.json from the validated algorithm)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from hole_filler import batch_from_hierarchy

ROOT = pathlib.Path(r"C:\Users\hide2\IdeaBox\Mesh_Generater\output")

REMAINING = [
    "A0072600529", "A0072600592", "A0072600635", "A0072600643", "A0072600710",
    "A0072600731", "A0072600750", "A0072600766", "A0072600781", "A0072600800",
    "A0072601002", "A0072601275", "A0072601282", "A0072601285", "A0072601294",
]

for name in REMAINING:
    hj = ROOT / name / "hierarchy.json"
    if not hj.exists():
        print(f"SKIP (no hierarchy.json): {name}")
        continue
    print(f"\n=== {name} ===", flush=True)
    batch_from_hierarchy(hj, dry_run=False)

print("\nALL REMAINING ASSEMBLIES DONE.")

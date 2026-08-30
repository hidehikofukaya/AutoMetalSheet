"""Read the PartMaker feature sidecar for a part.

Joined by `source_stp`, never by part id: 2300 sidecar files carry only 250
distinct stems, repeated across batch02 and the prod chunks, so a dict keyed by
id silently returns some other part's features. That bug reported 90.6% of our
bend lines as unexplainable; with the right key it is 1.0%.
"""
from __future__ import annotations

import json
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[4]
_WF = _ROOT / "runs" / "wtok_synth" / "wireframes"
_cache: dict = {}


def sidecar_path(part_name: str):
    w = _WF / f"{part_name}.json"
    if not w.exists():
        return None
    stp = pathlib.Path(json.loads(w.read_text())["source_stp"])
    f = stp.parent.parent / "features" / (stp.stem.replace("_mid", "") + ".json")
    return f if f.exists() else None


def load_features(part):
    name = getattr(part, "name", str(part))
    if name not in _cache:
        f = sidecar_path(name)
        # utf-8 explicitly: the machine default is cp932, which cannot read these
        _cache[name] = json.loads(f.read_text(encoding="utf-8")) if f else None
    return _cache[name]

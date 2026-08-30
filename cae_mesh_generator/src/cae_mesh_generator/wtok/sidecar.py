"""Read the PartMaker feature sidecar for a part.

Joined by `source_stp`, never by part id: 2300 sidecar files carry only 250
distinct stems, repeated across batch02 and the prod chunks, so a dict keyed by
id silently returns some other part's features. That bug reported 90.6% of our
bend lines as unexplainable; with the right key it is 1.0%.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

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


def spec_path(part_name: str):
    """The generator's own input for this part, beside the feature sidecar."""
    w = _WF / f"{part_name}.json"
    if not w.exists():
        return None
    stp = pathlib.Path(json.loads(w.read_text())["source_stp"])
    f = stp.parent.parent / "params" / (stp.stem.replace("_mid", "") + ".json")
    return f if f.exists() else None


SPEC_KEYS = ("thickness_mm", "half_width_mm", "bend_radius_mm",
             "fold1_slack_mm", "fold2_slack_mm", "min_bearing_radius_mm",
             "hole_diameter_mm")
# scales chosen so each lands near 1; the model sees them beside the fastener
# rows, which are already order 1 in frame units
SPEC_SCALE = np.array([2.0, 0.02, 0.05, 0.03, 0.03, 0.03, 0.1])


def load_spec(part):
    """The design spec as a fixed-length vector, or None where there is no file.

    This is the only information found so far that KB 12 does not rule out: it
    is the generator's INPUT, so it cannot be derived from the fastening points.

    What it does and does not do, measured over 900 parts. It cannot place a
    part -- a nearest-neighbour search on the spec alone is no better than random
    (25.99mm against a random control's 25.91mm), because thickness and bend
    radius say nothing about where the part is. What it does is fix scalars the
    fastening points leave open, and those are exactly today's weak spots:

        cross-validated R^2      fasteners   spec   both
        outline perimeter            0.205  0.090  0.356
        part width                   0.281  0.194  0.419
        bend-line spread             0.578 -0.009  0.694
        part extent                  0.924  0.085  0.965   (already determined)
    """
    name = getattr(part, "name", str(part))
    key = ("spec", name)
    if key not in _cache:
        f = spec_path(name)
        s = json.loads(f.read_text(encoding="utf-8")).get("spec") if f else None
        _cache[key] = (np.array([s.get(k, 0.0) for k in SPEC_KEYS], np.float32)
                       * SPEC_SCALE).astype(np.float32) if s else None
    return _cache[key]


def load_features(part):
    name = getattr(part, "name", str(part))
    if name not in _cache:
        f = sidecar_path(name)
        # utf-8 explicitly: the machine default is cp932, which cannot read these
        _cache[name] = json.loads(f.read_text(encoding="utf-8")) if f else None
    return _cache[name]

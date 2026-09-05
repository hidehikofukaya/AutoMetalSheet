#!/usr/bin/env python3
"""Web-based part viewer server for AutoMetalSheet.

Usage:
    python server.py [--port 8765]
"""
from __future__ import annotations

import argparse
import base64
import json
import pathlib
import threading
import time
import webbrowser
from datetime import datetime

import numpy as np
import pyvista as pv
from flask import Flask, jsonify, request, send_from_directory

ASSEMBLY_DIR = pathlib.Path(r"C:\Users\hide2\IdeaBox\Mesh_Generater\output\A0072601285")
REF_DIR = pathlib.Path(r"C:\Users\hide2\IdeaBox\AutoMetalSheet\ref")
VIEWER_DIR = pathlib.Path(__file__).parent
DEFAULT_PORT = 8765

app = Flask(__name__, static_folder=str(VIEWER_DIR))

# --- Mesh cache (key: (path_str, mtime) -> dict) ---
_mesh_cache: dict[tuple, dict] = {}


def _vtp_to_json(path: pathlib.Path) -> dict:
    """Convert VTP to Three.js-compatible dict (vertices flat, indices flat)."""
    mtime = path.stat().st_mtime
    key = (str(path), mtime)
    if key in _mesh_cache:
        return _mesh_cache[key]

    mesh = pv.read(str(path))
    mesh = mesh.triangulate()

    points = np.asarray(mesh.points, dtype=np.float32)  # (N, 3)
    faces = np.asarray(mesh.faces, dtype=np.int32)       # VTK flat: [3,i,j,k, ...]
    n_cells = mesh.n_cells

    # Extract triangle indices
    try:
        tris = faces.reshape(n_cells, 4)[:, 1:].astype(np.int32)
    except ValueError:
        tris_list: list[list[int]] = []
        ptr = 0
        while ptr < len(faces):
            n = int(faces[ptr])
            if n == 3:
                tris_list.append(faces[ptr + 1:ptr + 4].tolist())
            ptr += n + 1
        tris = np.array(tris_list, dtype=np.int32)

    result = {
        "vertices": points.flatten().tolist(),
        "indices": tris.flatten().tolist(),
        "n_vertices": int(len(points)),
        "n_faces": int(len(tris)),
        "bounds": {
            "min": points.min(axis=0).tolist(),
            "max": points.max(axis=0).tolist(),
        },
    }
    _mesh_cache[key] = result
    return result


def _load_hierarchy() -> dict[str, dict]:
    """Load hierarchy.json, keyed by part_id (e.g. 'body002')."""
    path = ASSEMBLY_DIR / "hierarchy.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict] = {}
    for entry in data.get("bodies", []):
        idx = entry.get("body_idx")
        if idx is not None:
            part_id = f"body{int(idx):03d}"
            result[part_id] = entry
    return result


# Loaded once at startup
_hierarchy: dict[str, dict] = {}


# ===== Routes =====

@app.route("/")
def index():
    return send_from_directory(str(VIEWER_DIR), "index.html")


@app.route("/api/parts")
def api_parts():
    bodies_dir = ASSEMBLY_DIR / "bodies"
    filled_dir = ASSEMBLY_DIR / "filled"

    parts = []
    for vtp in sorted(bodies_dir.glob("body*.vtp")):
        part_id = vtp.stem
        filled_vtp = filled_dir / f"{part_id}.vtp"
        holes_json = filled_dir / f"{part_id}_holes.json"

        n_holes = 0
        n_filled = 0
        n_open = 0
        if holes_json.exists():
            holes = json.loads(holes_json.read_text(encoding="utf-8")).get("holes", [])
            n_holes = len(holes)
            n_filled = sum(1 for h in holes if h.get("status") == "filled")
            n_open = n_holes - n_filled

        hier = _hierarchy.get(part_id, {})
        parts.append({
            "part_id": part_id,
            "has_original": vtp.exists(),
            "has_filled": filled_vtp.exists(),
            "has_holes": holes_json.exists(),
            "n_holes": n_holes,
            "n_filled": n_filled,
            "n_open": n_open,
            "tag": hier.get("tag", ""),
            "thickness_mm": hier.get("thickness_est_mm"),
            "catia_name": hier.get("catia_name", ""),
            "volume_mm3": hier.get("volume_mm3"),
        })

    return jsonify(parts)


@app.route("/api/mesh")
def api_mesh():
    part_id = request.args.get("part", "")
    mesh_type = request.args.get("type", "filled")

    if mesh_type == "original":
        path = ASSEMBLY_DIR / "bodies" / f"{part_id}.vtp"
    else:
        path = ASSEMBLY_DIR / "filled" / f"{part_id}.vtp"

    if not path.exists():
        return jsonify({"error": f"Not found: {path}"}), 404

    try:
        data = _vtp_to_json(path)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/holes")
def api_holes():
    part_id = request.args.get("part", "")
    path = ASSEMBLY_DIR / "filled" / f"{part_id}_holes.json"
    if not path.exists():
        return jsonify({})
    return jsonify(json.loads(path.read_text(encoding="utf-8")))


@app.route("/api/screenshot", methods=["POST"])
def api_screenshot():
    data = request.json or {}
    REF_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    part_id = data.get("part_id", "unknown")

    # Save PNG
    img_b64 = data.get("image", "").split(",")[-1]
    png_path = REF_DIR / f"{part_id}_{ts}.png"
    png_path.write_bytes(base64.b64decode(img_b64))

    # Save metadata log
    log = {
        "timestamp": ts,
        "part_id": part_id,
        "camera": data.get("camera"),
        "display_state": data.get("display_state"),
        "mesh_info": data.get("mesh_info"),
        "holes": data.get("holes_data"),
    }
    log_path = REF_DIR / f"{part_id}_{ts}.json"
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    return jsonify({"png": str(png_path), "log": str(log_path)})


def main():
    global _hierarchy
    parser = argparse.ArgumentParser(description="Part viewer server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    _hierarchy = _load_hierarchy()
    url = f"http://localhost:{args.port}"
    print(f"Part Viewer running at {url}")
    print("Press Ctrl+C to stop.")

    if not args.no_browser:
        def _open():
            time.sleep(0.8)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    app.run(host="localhost", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()

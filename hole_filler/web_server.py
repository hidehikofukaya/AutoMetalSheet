#!/usr/bin/env python3
"""Hole Filler Web Server — viewer + STP-based hole fill operations.

Usage:
    python hole_filler/web_server.py [--port 8766]
"""
from __future__ import annotations

import argparse
import base64
import json
import pathlib
import sys
import threading
import time
import uuid
import webbrowser
from datetime import datetime

import numpy as np
import pyvista as pv
from flask import Flask, jsonify, request, send_from_directory

# Add hole_filler src to path
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from hole_filler.stp_filler import (  # noqa: E402
    fill_stp_file,
    load_stp,
    detect_holes_from_stp,
    refine_holes_from_stp,
    build_holes_json,
    mesh_shape_to_vtp,
    diagnose_holes_geometry,
)

ASSEMBLY_DIR = pathlib.Path(r"C:\Users\hide2\IdeaBox\Mesh_Generater\output\A0072601285")
REF_DIR = pathlib.Path(r"C:\Users\hide2\IdeaBox\AutoMetalSheet\ref")
STATIC_DIR = pathlib.Path(__file__).parent
DEFAULT_PORT = 8766

app = Flask(__name__, static_folder=str(STATIC_DIR))

# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------

_mesh_cache: dict[tuple, dict] = {}
_hierarchy: dict[str, dict] = {}

# Job queue for background fill operations
_fill_jobs: dict[str, dict] = {}  # job_id → {status, part_id, log, result, error}
_jobs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vtp_to_json(path: pathlib.Path) -> dict:
    mtime = path.stat().st_mtime
    key = (str(path), mtime)
    if key in _mesh_cache:
        return _mesh_cache[key]

    mesh = pv.read(str(path))
    mesh = mesh.triangulate()
    points = np.asarray(mesh.points, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    n_cells = mesh.n_cells

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
    path = ASSEMBLY_DIR / "hierarchy.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict] = {}
    for entry in data.get("bodies", []):
        idx = entry.get("body_idx")
        if idx is not None:
            result[f"body{int(idx):03d}"] = entry
    return result


# ---------------------------------------------------------------------------
# Background fill job
# ---------------------------------------------------------------------------

def _run_fill_job(job_id: str, part_id: str, hole_ids: list[str] | None = None) -> None:
    stp_path = ASSEMBLY_DIR / "bodies" / f"{part_id}.stp"
    holes_path = ASSEMBLY_DIR / "filled" / f"{part_id}_holes.json"
    out_dir = ASSEMBLY_DIR / "filled"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stp = out_dir / f"{part_id}_filled.stp"
    out_vtp = out_dir / f"{part_id}.vtp"

    log_lines: list[str] = []

    def _cb(msg: str) -> None:
        log_lines.append(msg)
        with _jobs_lock:
            _fill_jobs[job_id]["log"] = list(log_lines)

    with _jobs_lock:
        _fill_jobs[job_id]["status"] = "running"

    try:
        # Step 1: Detect holes if holes.json is missing
        if not holes_path.exists():
            _cb(f"Detecting holes in {part_id}.stp...")
            shape = load_stp(stp_path)
            thickness_mm = _hierarchy.get(part_id, {}).get("thickness_est_mm")
            holes = detect_holes_from_stp(shape, panel_thickness_mm=thickness_mm)
            _cb(f"  Found {len(holes)} hole(s)")
            record = build_holes_json(part_id, ASSEMBLY_DIR.name, holes)
            holes_path.write_text(
                __import__("json").dumps(record, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        # Step 2: Fill STP holes and export VTP
        thickness_mm = _hierarchy.get(part_id, {}).get("thickness_est_mm")
        stats = fill_stp_file(
            stp_path=stp_path,
            holes_json_path=holes_path,
            output_stp_path=out_stp,
            output_vtp_path=out_vtp,
            progress_cb=_cb,
            hole_ids=hole_ids,
            panel_thickness_mm=thickness_mm,
        )

        # Invalidate mesh cache for this part
        for k in list(_mesh_cache.keys()):
            if part_id in k[0]:
                del _mesh_cache[k]

        with _jobs_lock:
            _fill_jobs[job_id].update({
                "status": "done",
                "result": stats,
                "log": list(log_lines),
            })
    except Exception as exc:
        import traceback
        with _jobs_lock:
            _fill_jobs[job_id].update({
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "log": list(log_lines),
            })


# ---------------------------------------------------------------------------
# Routes — static
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "web_index.html")


# ---------------------------------------------------------------------------
# Routes — viewer API (mirrors viewer/server.py)
# ---------------------------------------------------------------------------

@app.route("/api/parts")
def api_parts():
    bodies_dir = ASSEMBLY_DIR / "bodies"
    filled_dir = ASSEMBLY_DIR / "filled"
    parts = []
    for vtp in sorted(bodies_dir.glob("body*.vtp")):
        part_id = vtp.stem
        filled_vtp = filled_dir / f"{part_id}.vtp"
        holes_json = filled_dir / f"{part_id}_holes.json"
        stp_path = bodies_dir / f"{part_id}.stp"

        n_holes = n_filled = n_open = 0
        if holes_json.exists():
            holes = json.loads(holes_json.read_text(encoding="utf-8")).get("holes", [])
            n_holes = len(holes)
            n_filled = sum(1 for h in holes if h.get("status") == "filled")
            n_open = n_holes - n_filled

        hier = _hierarchy.get(part_id, {})
        parts.append({
            "part_id": part_id,
            "has_original": vtp.exists(),
            "has_filled": filled_vtp.exists() or vtp.exists(),
            "has_filled_stp": filled_vtp.exists(),
            "has_holes": holes_json.exists(),
            "has_stp": stp_path.exists(),
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
        # Prefer filled VTP; fall back to original bodies VTP
        path = ASSEMBLY_DIR / "filled" / f"{part_id}.vtp"
        if not path.exists():
            path = ASSEMBLY_DIR / "bodies" / f"{part_id}.vtp"

    if not path.exists():
        return jsonify({"error": f"Not found: {path}"}), 404
    try:
        return jsonify(_vtp_to_json(path))
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

    img_b64 = data.get("image", "").split(",")[-1]
    png_path = REF_DIR / f"{part_id}_{ts}.png"
    png_path.write_bytes(base64.b64decode(img_b64))

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


# ---------------------------------------------------------------------------
# Routes — fill API
# ---------------------------------------------------------------------------

@app.route("/api/fill", methods=["POST"])
def api_fill():
    data = request.json or {}
    part_id = data.get("part_id", "").strip()
    if not part_id:
        return jsonify({"error": "part_id required"}), 400

    stp_path = ASSEMBLY_DIR / "bodies" / f"{part_id}.stp"

    if not stp_path.exists():
        return jsonify({"error": f"STP not found: {stp_path.name}"}), 404

    hole_ids = data.get("hole_ids")  # list of strings or None

    job_id = str(uuid.uuid4())[:8]
    with _jobs_lock:
        _fill_jobs[job_id] = {
            "status": "pending",
            "part_id": part_id,
            "log": [],
        }

    t = threading.Thread(target=_run_fill_job, args=(job_id, part_id, hole_ids), daemon=True)
    t.start()
    return jsonify({"job_id": job_id, "part_id": part_id})


@app.route("/api/fill_status/<job_id>")
def api_fill_status(job_id: str):
    with _jobs_lock:
        job = _fill_jobs.get(job_id)
    if job is None:
        return jsonify({"error": "job not found"}), 404
    return jsonify(job)


@app.route("/api/fill_jobs")
def api_fill_jobs():
    with _jobs_lock:
        return jsonify(list(_fill_jobs.values()))


@app.route("/api/holes/diagnostic")
def api_holes_diagnostic():
    """Return diagnostic geometry (wall faces + boundary edges + inner wires) for each hole."""
    part_id = request.args.get("part", "").strip()
    if not part_id:
        return jsonify({"error": "part required"}), 400

    stp_path = ASSEMBLY_DIR / "bodies" / f"{part_id}.stp"
    holes_path = ASSEMBLY_DIR / "filled" / f"{part_id}_holes.json"

    if not stp_path.exists():
        return jsonify({"error": f"STP not found: {stp_path.name}"}), 404
    if not holes_path.exists():
        return jsonify({"error": f"holes.json not found — detect holes first"}), 404

    try:
        holes_record = __import__("json").loads(holes_path.read_text(encoding="utf-8"))
        holes_data = holes_record.get("holes", [])
        shape = load_stp(stp_path)
        # Refine centers/normals from STP before diagnosing
        holes_data = refine_holes_from_stp(shape, holes_data)
        diag = diagnose_holes_geometry(shape, holes_data)
        return jsonify({"part_id": part_id, "holes": diag})
    except Exception as exc:
        import traceback
        return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 500


@app.route("/api/detect_holes", methods=["POST"])
def api_detect_holes():
    """Detect holes from STP and save holes.json (without filling)."""
    data = request.json or {}
    part_id = data.get("part_id", "").strip()
    if not part_id:
        return jsonify({"error": "part_id required"}), 400

    stp_path = ASSEMBLY_DIR / "bodies" / f"{part_id}.stp"
    if not stp_path.exists():
        return jsonify({"error": f"STP not found: {stp_path.name}"}), 404

    try:
        shape = load_stp(stp_path)
        thickness_mm = _hierarchy.get(part_id, {}).get("thickness_est_mm")
        holes = detect_holes_from_stp(shape, panel_thickness_mm=thickness_mm)
        out_dir = ASSEMBLY_DIR / "filled"
        out_dir.mkdir(parents=True, exist_ok=True)
        holes_path = out_dir / f"{part_id}_holes.json"
        record = build_holes_json(part_id, ASSEMBLY_DIR.name, holes)
        holes_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return jsonify({"part_id": part_id, "n_holes": len(holes), "holes": holes})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _hierarchy
    parser = argparse.ArgumentParser(description="Hole Filler Web Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    _hierarchy = _load_hierarchy()
    url = f"http://localhost:{args.port}"
    print(f"Hole Filler running at {url}")
    print("Press Ctrl+C to stop.")

    if not args.no_browser:
        def _open():
            time.sleep(0.9)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    app.run(host="localhost", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()

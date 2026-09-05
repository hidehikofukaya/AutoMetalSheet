"""STEP open-shell tessellation helpers for filled midsurfaces."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.GCPnts import GCPnts_AbscissaPoint, GCPnts_UniformAbscissa
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCC.Core.TopoDS import topods

DEFAULT_WELD_TOLERANCE = 1.0e-4
DEFAULT_BOUNDARY_SEGMENT_LENGTH = 8.0
DEFAULT_BOUNDARY_MAX_SEGMENTS_PER_EDGE = 64
DEFAULT_CREASE_ANGLE_DEGREES = 30.0
DEFAULT_CORNER_ANGLE_DEGREES = 35.0
BOUNDARY_EXTRACTION_VERSION = "topo_open_edge_crease_v3"


@dataclass(frozen=True)
class TessellationConfig:
    linear_deflection: float = 2.0
    angular_deflection: float = 0.5

    def cache_key(self) -> str:
        return f"lin{self.linear_deflection:g}_ang{self.angular_deflection:g}"


@dataclass
class TessellatedMesh:
    vertices: np.ndarray
    faces: np.ndarray
    face_normals: np.ndarray
    boundary_edges: np.ndarray
    boundary_edge_normals: np.ndarray
    crease_edges: np.ndarray
    crease_edge_normals: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    source_path: str
    config: TessellationConfig

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def n_boundary_edges(self) -> int:
        return int(self.boundary_edges.shape[0])

    @property
    def n_crease_edges(self) -> int:
        return int(self.crease_edges.shape[0])


def _source_hash(path: Path, config: TessellationConfig) -> str:
    stat = path.stat()
    boundary_key = (
        f"{BOUNDARY_EXTRACTION_VERSION}|weld{DEFAULT_WELD_TOLERANCE:g}|"
        f"seg{DEFAULT_BOUNDARY_SEGMENT_LENGTH:g}|maxseg{DEFAULT_BOUNDARY_MAX_SEGMENTS_PER_EDGE:g}|"
        f"crease{DEFAULT_CREASE_ANGLE_DEGREES:g}|corner{DEFAULT_CORNER_ANGLE_DEGREES:g}"
    )
    raw = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{config.cache_key()}|{boundary_key}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def tessellate_step(path: str | Path, config: TessellationConfig | None = None) -> TessellatedMesh:
    """Read and tessellate a STEP surface with OpenCASCADE."""

    path = Path(path)
    config = config or TessellationConfig()
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != 1:
        raise RuntimeError(f"STEP read failed: {path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    BRepMesh_IncrementalMesh(
        shape,
        config.linear_deflection,
        False,
        config.angular_deflection,
        True,
    )

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    base = 0

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            reversed_face = face.Orientation() == TopAbs_REVERSED
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(trsf)
                vertices.append((p.X(), p.Y(), p.Z()))
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                if reversed_face:
                    b, c = c, b
                faces.append((base + a - 1, base + b - 1, base + c - 1))
            base += tri.NbNodes()
        exp.Next()

    verts = np.asarray(vertices, dtype=np.float32)
    fidx = np.asarray(faces, dtype=np.int64)
    if len(verts) == 0 or len(fidx) == 0:
        raise RuntimeError(f"STEP tessellated to empty mesh: {path}")

    normals = face_normals(verts, fidx)
    boundary_edges, boundary_normals = boundary_edges_from_shape(shape, verts, fidx, normals, config)
    if len(boundary_edges) == 0:
        boundary_edges, boundary_normals = boundary_edges_from_faces(verts, fidx, normals)
    crease_edges, crease_normals = crease_edges_from_faces(verts, fidx, normals)
    return TessellatedMesh(
        vertices=verts,
        faces=fidx,
        face_normals=normals,
        boundary_edges=boundary_edges,
        boundary_edge_normals=boundary_normals,
        crease_edges=crease_edges,
        crease_edge_normals=crease_normals,
        bounds_min=verts.min(axis=0),
        bounds_max=verts.max(axis=0),
        source_path=str(path),
        config=config,
    )


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    raw = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(raw, axis=1, keepdims=True)
    norm = np.maximum(norm, 1.0e-12)
    return (raw / norm).astype(np.float32)


def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    raw = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return (0.5 * np.linalg.norm(raw, axis=1)).astype(np.float64)


def boundary_edges_from_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    weld_tolerance: float = DEFAULT_WELD_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract open boundary edges after coordinate welding.

    OCC triangulations often duplicate vertices per CAD face. Counting raw triangle
    indices would therefore misclassify face seams as boundaries. We first weld
    vertices by coordinate and then keep only undirected triangle edges used once.
    """

    if len(vertices) == 0 or len(faces) == 0:
        return np.zeros((0, 2, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    welded_ids, welded_vertices = weld_vertices(vertices, tolerance=weld_tolerance)
    edge_counts: dict[tuple[int, int], int] = {}
    edge_face: dict[tuple[int, int], int] = {}
    for face_idx, tri in enumerate(faces):
        ids = [int(welded_ids[int(i)]) for i in tri]
        for a, b in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            edge_counts[key] = edge_counts.get(key, 0) + 1
            edge_face[key] = face_idx
    boundary_keys = [key for key, count in edge_counts.items() if count == 1]
    if not boundary_keys:
        return np.zeros((0, 2, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    edges = np.asarray([[welded_vertices[a], welded_vertices[b]] for a, b in boundary_keys], dtype=np.float32)
    edge_normals = np.asarray([normals[edge_face[key]] for key in boundary_keys], dtype=np.float32)
    return edges, edge_normals


def crease_edges_from_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    angle_degrees: float = DEFAULT_CREASE_ANGLE_DEGREES,
    weld_tolerance: float = DEFAULT_WELD_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract internal edges with a large adjacent-face normal change."""

    if len(vertices) == 0 or len(faces) == 0:
        return np.zeros((0, 2, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    welded_ids, welded_vertices = weld_vertices(vertices, tolerance=weld_tolerance)
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_idx, tri in enumerate(faces):
        ids = [int(welded_ids[int(i)]) for i in tri]
        for a, b in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            edge_faces.setdefault(key, []).append(face_idx)

    cos_threshold = float(np.cos(np.deg2rad(angle_degrees)))
    crease_keys: list[tuple[int, int]] = []
    crease_normals: list[np.ndarray] = []
    for key, face_ids in edge_faces.items():
        if len(face_ids) != 2:
            continue
        n0 = normals[face_ids[0]]
        n1 = normals[face_ids[1]]
        dot = float(np.clip(np.dot(n0, n1), -1.0, 1.0))
        if dot <= cos_threshold:
            crease_keys.append(key)
            avg = n0 + n1
            norm = float(np.linalg.norm(avg))
            if norm <= 1.0e-12:
                avg = n0
                norm = float(np.linalg.norm(avg))
            crease_normals.append((avg / max(norm, 1.0e-12)).astype(np.float32))

    if not crease_keys:
        return np.zeros((0, 2, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    edges = np.asarray([[welded_vertices[a], welded_vertices[b]] for a, b in crease_keys], dtype=np.float32)
    edge_normals = np.asarray(crease_normals, dtype=np.float32)
    return edges, edge_normals


def corner_points_from_edges(
    edges: np.ndarray,
    normals: np.ndarray | None = None,
    angle_degrees: float = DEFAULT_CORNER_ANGLE_DEGREES,
    weld_tolerance: float = DEFAULT_WELD_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Find endpoints where edge chains branch or turn sharply."""

    edges = np.asarray(edges, dtype=np.float32)
    if len(edges) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    flat = edges.reshape(-1, 3)
    point_ids, welded = weld_vertices(flat, tolerance=weld_tolerance)
    edge_ids = point_ids.reshape(-1, 2)
    incident: dict[int, list[tuple[int, np.ndarray]]] = {}
    normal_accum = np.zeros((len(welded), 3), dtype=np.float64)
    normal_counts = np.zeros((len(welded),), dtype=np.float64)
    if normals is None or len(normals) != len(edges):
        normals = np.zeros((len(edges), 3), dtype=np.float32)
    for edge_idx, (a, b) in enumerate(edge_ids):
        if int(a) == int(b):
            continue
        pa = welded[int(a)]
        pb = welded[int(b)]
        ab = pb - pa
        length = float(np.linalg.norm(ab))
        if length <= weld_tolerance:
            continue
        unit = ab / length
        incident.setdefault(int(a), []).append((int(b), unit))
        incident.setdefault(int(b), []).append((int(a), -unit))
        normal_accum[int(a)] += normals[edge_idx]
        normal_accum[int(b)] += normals[edge_idx]
        normal_counts[int(a)] += 1.0
        normal_counts[int(b)] += 1.0

    straight_dot_limit = -float(np.cos(np.deg2rad(angle_degrees)))
    corner_ids: list[int] = []
    for point_id, neighbors in incident.items():
        unique_neighbors = {neighbor for neighbor, _ in neighbors}
        if len(unique_neighbors) != 2:
            corner_ids.append(point_id)
            continue
        dirs = [direction for _, direction in neighbors[:2]]
        dot = float(np.clip(np.dot(dirs[0], dirs[1]), -1.0, 1.0))
        if dot > straight_dot_limit:
            corner_ids.append(point_id)

    if not corner_ids:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    pts = welded[corner_ids].astype(np.float32)
    nrm = normal_accum[corner_ids]
    norms = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, np.maximum(norms, 1.0e-12), out=np.zeros_like(nrm), where=norms > 0)
    return pts, nrm.astype(np.float32)


def boundary_edges_from_shape(
    shape,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    config: TessellationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract open shell boundary from B-Rep edge-to-face adjacency.

    Topological open edges avoid false boundary labels when neighboring CAD faces
    tessellate the same seam with different triangle subdivisions.
    """

    edge_to_faces = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndUniqueAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_to_faces)

    segments: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    for index in range(1, edge_to_faces.Size() + 1):
        ancestors = edge_to_faces.FindFromIndex(index)
        if ancestors.Size() != 1:
            continue
        edge = topods.Edge(edge_to_faces.FindKey(index))
        curve = BRepAdaptor_Curve(edge)
        params = boundary_curve_parameters(curve, config.linear_deflection)
        if len(params) < 2:
            continue
        points = [curve_point(curve, param) for param in params]
        for start, end in zip(points[:-1], points[1:]):
            if np.linalg.norm(np.asarray(end) - np.asarray(start)) > DEFAULT_WELD_TOLERANCE:
                segments.append((start, end))

    if not segments:
        return np.zeros((0, 2, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    edges = np.asarray(segments, dtype=np.float32)
    edge_normals = nearest_face_normals_for_edges(edges, vertices, faces, normals)
    return edges, edge_normals


def boundary_curve_parameters(curve: BRepAdaptor_Curve, linear_deflection: float) -> list[float]:
    first = float(curve.FirstParameter())
    last = float(curve.LastParameter())
    if not np.isfinite(first) or not np.isfinite(last) or first == last:
        return []
    if last < first:
        first, last = last, first
    try:
        length = float(GCPnts_AbscissaPoint.Length(curve, first, last, max(DEFAULT_WELD_TOLERANCE, 1.0e-7)))
    except Exception:
        start = np.asarray(curve_point(curve, first))
        end = np.asarray(curve_point(curve, last))
        length = float(np.linalg.norm(end - start))
    if not np.isfinite(length) or length <= DEFAULT_WELD_TOLERANCE:
        return [first, last]

    max_segment_length = max(float(linear_deflection) * 4.0, DEFAULT_BOUNDARY_SEGMENT_LENGTH)
    n_segments = int(np.ceil(length / max_segment_length))
    n_segments = max(1, min(n_segments, DEFAULT_BOUNDARY_MAX_SEGMENTS_PER_EDGE))
    n_points = n_segments + 1
    try:
        sampler = GCPnts_UniformAbscissa(curve, n_points, first, last)
        if sampler.IsDone() and sampler.NbPoints() >= 2:
            return [float(sampler.Parameter(i)) for i in range(1, sampler.NbPoints() + 1)]
    except Exception:
        pass
    return np.linspace(first, last, n_points).astype(np.float64).tolist()


def curve_point(curve: BRepAdaptor_Curve, parameter: float) -> tuple[float, float, float]:
    point = curve.Value(float(parameter))
    return (float(point.X()), float(point.Y()), float(point.Z()))


def nearest_face_normals_for_edges(
    edges: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    chunk_size: int = 512,
) -> np.ndarray:
    if len(edges) == 0 or len(faces) == 0:
        return np.zeros((len(edges), 3), dtype=np.float32)
    centroids = vertices[faces].mean(axis=1).astype(np.float32)
    midpoints = edges.mean(axis=1).astype(np.float32)
    chosen: list[np.ndarray] = []
    for start in range(0, len(midpoints), chunk_size):
        chunk = midpoints[start : start + chunk_size]
        dist2 = ((chunk[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        chosen.append(normals[np.argmin(dist2, axis=1)])
    return np.concatenate(chosen, axis=0).astype(np.float32)


def weld_vertices(vertices: np.ndarray, tolerance: float = DEFAULT_WELD_TOLERANCE) -> tuple[np.ndarray, np.ndarray]:
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    keys = np.round(np.asarray(vertices, dtype=np.float64) / tolerance).astype(np.int64)
    id_by_key: dict[tuple[int, int, int], int] = {}
    inverse = np.empty((len(vertices),), dtype=np.int64)
    sums: list[np.ndarray] = []
    counts: list[int] = []
    for idx, key_arr in enumerate(keys):
        key = (int(key_arr[0]), int(key_arr[1]), int(key_arr[2]))
        welded_id = id_by_key.get(key)
        if welded_id is None:
            welded_id = len(sums)
            id_by_key[key] = welded_id
            sums.append(np.asarray(vertices[idx], dtype=np.float64).copy())
            counts.append(1)
        else:
            sums[welded_id] += vertices[idx]
            counts[welded_id] += 1
        inverse[idx] = welded_id
    welded = np.asarray([s / c for s, c in zip(sums, counts)], dtype=np.float32)
    return inverse, welded


def load_or_tessellate(
    path: str | Path,
    cache_dir: str | Path | None = None,
    config: TessellationConfig | None = None,
) -> TessellatedMesh:
    """Load a cached tessellation or tessellate and save it."""

    path = Path(path)
    config = config or TessellationConfig()
    if cache_dir is None:
        return tessellate_step(path, config)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _source_hash(path, config)
    npz_path = cache_dir / f"{path.stem}_{key}.npz"
    json_path = cache_dir / f"{path.stem}_{key}.json"
    if npz_path.exists():
        data = np.load(npz_path)
        meta = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
        vertices = data["vertices"].astype(np.float32)
        faces = data["faces"].astype(np.int64)
        normals = data["face_normals"].astype(np.float32)
        if "boundary_edges" in data and "boundary_edge_normals" in data:
            boundary_edges = data["boundary_edges"].astype(np.float32)
            boundary_edge_normals = data["boundary_edge_normals"].astype(np.float32)
        else:
            boundary_edges, boundary_edge_normals = boundary_edges_from_faces(vertices, faces, normals)
        if "crease_edges" in data and "crease_edge_normals" in data:
            crease_edges = data["crease_edges"].astype(np.float32)
            crease_edge_normals = data["crease_edge_normals"].astype(np.float32)
        else:
            crease_edges, crease_edge_normals = crease_edges_from_faces(vertices, faces, normals)
        return TessellatedMesh(
            vertices=vertices,
            faces=faces,
            face_normals=normals,
            boundary_edges=boundary_edges,
            boundary_edge_normals=boundary_edge_normals,
            crease_edges=crease_edges,
            crease_edge_normals=crease_edge_normals,
            bounds_min=data["bounds_min"].astype(np.float32),
            bounds_max=data["bounds_max"].astype(np.float32),
            source_path=meta.get("source_path", str(path)),
            config=config,
        )

    mesh = tessellate_step(path, config)
    np.savez_compressed(
        npz_path,
        vertices=mesh.vertices,
        faces=mesh.faces,
        face_normals=mesh.face_normals,
        boundary_edges=mesh.boundary_edges,
        boundary_edge_normals=mesh.boundary_edge_normals,
        crease_edges=mesh.crease_edges,
        crease_edge_normals=mesh.crease_edge_normals,
        bounds_min=mesh.bounds_min,
        bounds_max=mesh.bounds_max,
    )
    json_path.write_text(
        json.dumps(
            {
                "source_path": str(path),
                "n_vertices": mesh.n_vertices,
                "n_faces": mesh.n_faces,
                "n_boundary_edges": mesh.n_boundary_edges,
                "n_crease_edges": mesh.n_crease_edges,
                "boundary_extraction_version": BOUNDARY_EXTRACTION_VERSION,
                "boundary_weld_tolerance": DEFAULT_WELD_TOLERANCE,
                "boundary_segment_length": DEFAULT_BOUNDARY_SEGMENT_LENGTH,
                "crease_angle_degrees": DEFAULT_CREASE_ANGLE_DEGREES,
                "corner_angle_degrees": DEFAULT_CORNER_ANGLE_DEGREES,
                "config": {
                    "linear_deflection": config.linear_deflection,
                    "angular_deflection": config.angular_deflection,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return mesh


def sample_surface_points(
    mesh: TessellatedMesh,
    n_points: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted sample points and face normals from a triangle mesh."""

    rng = np.random.default_rng(seed)
    areas = triangle_areas(mesh.vertices, mesh.faces)
    total = areas.sum()
    if total <= 0:
        raise RuntimeError(f"non-positive mesh area: {mesh.source_path}")
    probs = areas / total
    face_ids = rng.choice(len(mesh.faces), size=n_points, replace=True, p=probs)
    tri = mesh.vertices[mesh.faces[face_ids]]

    u = rng.random(n_points, dtype=np.float32)
    v = rng.random(n_points, dtype=np.float32)
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    pts = tri[:, 0] + u[:, None] * (tri[:, 1] - tri[:, 0]) + v[:, None] * (tri[:, 2] - tri[:, 0])
    normals = mesh.face_normals[face_ids]
    return pts.astype(np.float32), normals.astype(np.float32)


def sample_boundary_points(
    mesh: TessellatedMesh,
    n_points: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Length-weighted sample points from open boundary edges."""

    return sample_edge_points(mesh.boundary_edges, mesh.boundary_edge_normals, n_points, seed=seed)


def sample_crease_points(
    mesh: TessellatedMesh,
    n_points: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Length-weighted sample points from high-normal-change internal edges."""

    return sample_edge_points(mesh.crease_edges, mesh.crease_edge_normals, n_points, seed=seed)


def sample_edge_points(
    edges: np.ndarray,
    edge_normals: np.ndarray,
    n_points: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Length-weighted sample points from line segments."""

    if n_points <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    if len(edges) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)
    vec = edges[:, 1] - edges[:, 0]
    lengths = np.linalg.norm(vec, axis=1).astype(np.float64)
    total = lengths.sum()
    if total <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    edge_ids = rng.choice(len(edges), size=n_points, replace=True, p=lengths / total)
    t = rng.random(n_points, dtype=np.float32)[:, None]
    selected_edges = edges[edge_ids]
    pts = selected_edges[:, 0] + t * (selected_edges[:, 1] - selected_edges[:, 0])
    normals = edge_normals[edge_ids] if len(edge_normals) == len(edges) else np.zeros_like(pts)
    return pts.astype(np.float32), normals.astype(np.float32)


def sample_corner_points(
    mesh: TessellatedMesh,
    n_points: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample branch and sharp-turn endpoints from boundary and crease chains."""

    if n_points <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    edges = []
    normals = []
    if mesh.n_boundary_edges:
        edges.append(mesh.boundary_edges)
        normals.append(mesh.boundary_edge_normals)
    if mesh.n_crease_edges:
        edges.append(mesh.crease_edges)
        normals.append(mesh.crease_edge_normals)
    if not edges:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    corner_pts, corner_normals = corner_points_from_edges(
        np.concatenate(edges, axis=0),
        np.concatenate(normals, axis=0),
    )
    if len(corner_pts) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(corner_pts), size=n_points, replace=len(corner_pts) < n_points)
    return corner_pts[ids].astype(np.float32), corner_normals[ids].astype(np.float32)

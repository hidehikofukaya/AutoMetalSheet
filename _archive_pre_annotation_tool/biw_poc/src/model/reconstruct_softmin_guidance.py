"""Coarse Stage-C reconstruction for :mod:`softmin_guidance`.

The signed softmin potential is used only to rank grid candidates.  Projection
is deliberately driven by the independent non-negative ``step_distance`` head
and the unit ``direction_toward_surface`` head.  In particular, a negative
potential can never reverse a projection step.
"""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree

try:
    from .softmin_guidance import CHECKPOINT_SCHEMA_VERSION, SoftminGuidanceModel
except ImportError:  # Support direct execution from src/model.
    from softmin_guidance import (  # type: ignore[no-redef]
        CHECKPOINT_SCHEMA_VERSION,
        SoftminGuidanceModel,
    )


class InsufficientEvidenceError(RuntimeError):
    """Raised when a mandatory geometric evidence gate leaves too few points."""


@dataclass(frozen=True)
class InputCloud:
    """Validated model input plus its reversible world-coordinate transform."""

    points_norm: np.ndarray
    normals: np.ndarray
    center: np.ndarray
    scale_mm: float
    thickness_mm: float


@dataclass(frozen=True)
class GuidanceGrid:
    """Model guidance evaluated over a deterministic Cartesian grid."""

    xyz: np.ndarray
    soft_potential: np.ndarray
    step_distance: np.ndarray
    toward_direction: np.ndarray
    branch_ambiguity: np.ndarray


_ARCHITECTURE_KEYS = (
    "token_dim",
    "n_tokens",
    "k_neighbors",
    "n_latents",
    "enc_layers",
    "dec_layers",
    "n_heads",
    "use_normals_in_encoder",
    "use_thickness_in_encoder",
    "dropout",
    "n_freqs",
)
RANKING_MODES = (
    "raw_signed",
    "abs_potential",
    "step_distance",
    "abs_potential_ambiguity",
    "spatial_balanced",
)


def _checkpoint_metadata(checkpoint: dict[str, Any]) -> dict[str, Any]:
    for key in ("metadata", "checkpoint_metadata", "model_metadata"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return {}


def load_model(
    checkpoint_path: str | pathlib.Path, device: str | torch.device
) -> SoftminGuidanceModel:
    """Restore a SoftminGuidanceModel from a metadata-bearing checkpoint."""

    checkpoint_path = pathlib.Path(checkpoint_path)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:  # pragma: no cover - compatibility with older PyTorch.
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"{checkpoint_path}: checkpoint must be a mapping")

    metadata = _checkpoint_metadata(checkpoint)
    schema_version = metadata.get("schema_version")
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"{checkpoint_path}: unsupported schema_version {schema_version!r}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION!r}"
        )
    if metadata.get("model_type") != "SoftminGuidanceModel":
        raise ValueError(
            f"{checkpoint_path}: model_type must be 'SoftminGuidanceModel'"
        )
    if metadata.get("stage_a_schema_version") != "stage_a.softmin_guidance.v1":
        raise ValueError(
            f"{checkpoint_path}: incompatible Stage A schema contract"
        )

    architecture = metadata.get("architecture")
    if not isinstance(architecture, dict):
        architecture = checkpoint.get("architecture")
    if not isinstance(architecture, dict):
        cfg = checkpoint.get("cfg")
        if isinstance(cfg, dict):
            architecture = {key: cfg[key] for key in _ARCHITECTURE_KEYS if key in cfg}
    if not isinstance(architecture, dict) or not architecture:
        raise ValueError(
            f"{checkpoint_path}: model architecture metadata is required"
        )

    unknown = set(architecture) - set(_ARCHITECTURE_KEYS)
    if unknown:
        raise ValueError(
            f"{checkpoint_path}: unknown architecture fields: {sorted(unknown)}"
        )

    state_dict = checkpoint.get("model")
    if state_dict is None:
        state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"{checkpoint_path}: model state_dict is missing")

    model = SoftminGuidanceModel(**architecture).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _validated_xyz(array: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(array, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N,3], got {result.shape}")
    if len(result) == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def load_input_h5(path: str | pathlib.Path) -> InputCloud:
    """Load and normalize H5 ``points``/``normals`` using the training contract."""

    path = pathlib.Path(path)
    with h5py.File(path, "r") as h5_file:
        if "points" not in h5_file or "normals" not in h5_file:
            raise KeyError(f"{path}: both 'points' and 'normals' are required")
        points = _validated_xyz(h5_file["points"][:], "points")
        normals = _validated_xyz(h5_file["normals"][:], "normals")
        thickness_mm = float(h5_file.attrs.get("thickness_mm", 0.0))

    if points.shape != normals.shape:
        raise ValueError(
            f"{path}: normals shape {normals.shape} must match points {points.shape}"
        )
    if not np.isfinite(thickness_mm) or thickness_mm < 0.0:
        raise ValueError(f"{path}: thickness_mm must be finite and non-negative")

    normal_length = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(normal_length <= 1.0e-8):
        raise ValueError(f"{path}: zero-length input normal is not permitted")
    normals = (normals / normal_length).astype(np.float32, copy=False)

    center = points.mean(axis=0).astype(np.float32)
    extent = float(np.abs(points - center).max())
    scale_mm = extent * 1.05 + 1.0e-6
    points_norm = ((points - center) / scale_mm).astype(np.float32, copy=False)
    return InputCloud(
        points_norm=points_norm,
        normals=normals,
        center=center,
        scale_mm=scale_mm,
        thickness_mm=thickness_mm,
    )


@torch.no_grad()
def encode_input(
    model: SoftminGuidanceModel,
    cloud: InputCloud,
    device: str | torch.device,
) -> torch.Tensor:
    points = torch.from_numpy(cloud.points_norm).unsqueeze(0).to(device)
    normals = torch.from_numpy(cloud.normals).unsqueeze(0).to(device)
    thickness_norm = torch.tensor(
        [cloud.thickness_mm / cloud.scale_mm], dtype=torch.float32, device=device
    )
    return model.encode(points, normals, thickness_norm)


@torch.no_grad()
def decode_guidance_chunked(
    model: SoftminGuidanceModel,
    latent: torch.Tensor,
    xyz: np.ndarray,
    device: str | torch.device,
    chunk_size: int = 100_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate all four guidance heads without changing point order."""

    xyz = _validated_xyz(xyz, "xyz")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    potential = np.empty(len(xyz), dtype=np.float32)
    step = np.empty(len(xyz), dtype=np.float32)
    direction = np.empty((len(xyz), 3), dtype=np.float32)
    ambiguity = np.empty(len(xyz), dtype=np.float32)
    for start in range(0, len(xyz), chunk_size):
        stop = min(start + chunk_size, len(xyz))
        query = torch.from_numpy(xyz[start:stop]).unsqueeze(0).to(device)
        prediction = model.decode(query, latent)
        potential[start:stop] = prediction.soft_potential[0].cpu().numpy()
        step[start:stop] = prediction.step_distance[0].cpu().numpy()
        direction[start:stop] = (
            prediction.direction_toward_surface[0].cpu().numpy()
        )
        ambiguity[start:stop] = prediction.branch_ambiguity[0].cpu().numpy()

    arrays = (potential, step, direction, ambiguity)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("model guidance contains non-finite values")
    return arrays


def evaluate_grid(
    model: SoftminGuidanceModel,
    latent: torch.Tensor,
    points_norm: np.ndarray,
    grid_res: int,
    device: str | torch.device,
    *,
    bbox_pad_fraction: float = 0.15,
    chunk_size: int = 100_000,
) -> GuidanceGrid:
    """Evaluate a deterministic grid covering the padded input-cloud bbox."""

    points_norm = _validated_xyz(points_norm, "points_norm")
    if grid_res < 2:
        raise ValueError("grid_res must be at least 2")
    if bbox_pad_fraction < 0.0:
        raise ValueError("bbox_pad_fraction must be non-negative")

    lower = points_norm.min(axis=0)
    upper = points_norm.max(axis=0)
    padding = (upper - lower) * bbox_pad_fraction
    axes = [
        np.linspace(
            lower[axis] - padding[axis],
            upper[axis] + padding[axis],
            grid_res,
            dtype=np.float32,
        )
        for axis in range(3)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    xyz = np.stack([axis.ravel() for axis in mesh], axis=-1)
    potential, step, direction, ambiguity = decode_guidance_chunked(
        model, latent, xyz, device, chunk_size
    )
    return GuidanceGrid(xyz, potential, step, direction, ambiguity)


def _input_gate_indices(
    xyz: np.ndarray,
    input_points_norm: np.ndarray,
    scale_mm: float,
    max_input_distance_mm: float | None,
) -> np.ndarray:
    xyz = _validated_xyz(xyz, "xyz")
    input_points_norm = _validated_xyz(input_points_norm, "input_points_norm")
    if not np.isfinite(scale_mm) or scale_mm <= 0.0:
        raise ValueError("scale_mm must be finite and positive")
    if max_input_distance_mm is None:
        return np.arange(len(xyz), dtype=np.int64)
    if (
        not np.isfinite(max_input_distance_mm)
        or max_input_distance_mm < 0.0
    ):
        raise ValueError("max_input_distance_mm must be finite and non-negative")
    distance_norm, _ = cKDTree(input_points_norm).query(xyz, k=1)
    return np.flatnonzero(distance_norm * scale_mm <= max_input_distance_mm)


def candidate_ranking_score(
    soft_potential: np.ndarray,
    *,
    scale_mm: float,
    ranking_mode: str,
    step_distance: np.ndarray | None = None,
    branch_ambiguity: np.ndarray | None = None,
    ambiguity_penalty_mm: float = 1.0,
) -> np.ndarray:
    """Return a lower-is-better world-mm score for candidate selection."""
    potential = np.asarray(soft_potential, dtype=np.float32)
    if potential.ndim != 1 or not np.all(np.isfinite(potential)):
        raise ValueError("soft_potential must be a finite [N] array")
    if ranking_mode not in RANKING_MODES:
        raise ValueError(f"ranking_mode must be one of {RANKING_MODES}")
    if not np.isfinite(scale_mm) or scale_mm <= 0:
        raise ValueError("scale_mm must be finite and positive")
    if not np.isfinite(ambiguity_penalty_mm) or ambiguity_penalty_mm < 0:
        raise ValueError("ambiguity_penalty_mm must be finite and non-negative")

    if ranking_mode == "raw_signed":
        return potential * np.float32(scale_mm)
    if ranking_mode == "step_distance":
        if step_distance is None:
            raise ValueError("step_distance is required for step_distance ranking")
        step = np.asarray(step_distance, dtype=np.float32)
        if step.shape != potential.shape or not np.all(np.isfinite(step)):
            raise ValueError("step_distance must be finite and match soft_potential")
        return np.maximum(step, 0.0) * np.float32(scale_mm)

    score = np.abs(potential) * np.float32(scale_mm)
    if ranking_mode in ("abs_potential_ambiguity", "spatial_balanced"):
        if branch_ambiguity is None:
            raise ValueError(
                "branch_ambiguity is required for ambiguity-aware ranking"
            )
        ambiguity = np.asarray(branch_ambiguity, dtype=np.float32)
        if ambiguity.shape != potential.shape or not np.all(np.isfinite(ambiguity)):
            raise ValueError(
                "branch_ambiguity must be finite and match soft_potential"
            )
        score = score + np.clip(ambiguity, 0.0, 1.0) * np.float32(
            ambiguity_penalty_mm
        )
    return score


def _spatially_balanced_order(
    xyz: np.ndarray, score: np.ndarray, candidate_count: int
) -> np.ndarray:
    """Round-robin score order across deterministic Cartesian cells."""
    if len(xyz) == 0:
        return np.empty(0, dtype=np.int64)
    bins_per_axis = max(1, int(np.ceil(candidate_count ** (1.0 / 3.0))))
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    span = np.maximum(hi - lo, 1.0e-12)
    cell_xyz = np.minimum(
        ((xyz - lo) / span * bins_per_axis).astype(np.int64),
        bins_per_axis - 1,
    )
    cell_id = (
        cell_xyz[:, 0] * bins_per_axis * bins_per_axis
        + cell_xyz[:, 1] * bins_per_axis
        + cell_xyz[:, 2]
    )
    stable = np.argsort(score, kind="stable")
    queues: dict[int, list[int]] = {}
    for index in stable:
        queues.setdefault(int(cell_id[index]), []).append(int(index))
    cells = sorted(queues)
    output: list[int] = []
    depth = 0
    while len(output) < min(candidate_count, len(xyz)):
        added = False
        for cell in cells:
            queue = queues[cell]
            if depth < len(queue):
                output.append(queue[depth])
                added = True
                if len(output) >= candidate_count:
                    break
        if not added:
            break
        depth += 1
    return np.asarray(output, dtype=np.int64)


def select_ranked_candidates(
    grid_xyz: np.ndarray,
    soft_potential: np.ndarray,
    *,
    input_points_norm: np.ndarray,
    scale_mm: float,
    max_input_distance_mm: float | None,
    candidate_count: int,
    minimum_required: int,
    step_distance: np.ndarray | None = None,
    branch_ambiguity: np.ndarray | None = None,
    ranking_mode: str = "abs_potential_ambiguity",
    ambiguity_penalty_mm: float = 1.0,
) -> np.ndarray:
    """Return stable, optionally spatially balanced indices after support gating."""

    grid_xyz = _validated_xyz(grid_xyz, "grid_xyz")
    soft_potential = np.asarray(soft_potential, dtype=np.float32)
    if soft_potential.shape != (len(grid_xyz),):
        raise ValueError(
            "soft_potential must have one scalar per grid point, got "
            f"{soft_potential.shape}"
        )
    if not np.all(np.isfinite(soft_potential)):
        raise ValueError("soft_potential contains non-finite values")
    if candidate_count <= 0 or minimum_required <= 0:
        raise ValueError("candidate_count and minimum_required must be positive")
    if candidate_count < minimum_required:
        raise ValueError("candidate_count must be at least minimum_required")

    gated = _input_gate_indices(
        grid_xyz,
        input_points_norm,
        scale_mm,
        max_input_distance_mm,
    )
    if len(gated) < minimum_required:
        raise InsufficientEvidenceError(
            f"Only {len(gated)} grid points pass input-cloud gate; "
            f"minimum is {minimum_required}. Ungated fallback is prohibited."
        )

    score = candidate_ranking_score(
        soft_potential,
        scale_mm=scale_mm,
        ranking_mode=ranking_mode,
        step_distance=step_distance,
        branch_ambiguity=branch_ambiguity,
        ambiguity_penalty_mm=ambiguity_penalty_mm,
    )
    if ranking_mode == "spatial_balanced":
        local_order = _spatially_balanced_order(
            grid_xyz[gated], score[gated], min(candidate_count, len(gated))
        )
        return gated[local_order]
    # Stable sorting makes tied selection reproducible in grid order.
    order = np.argsort(score[gated], kind="stable")
    return gated[order[: min(candidate_count, len(gated))]]


def bounded_project(
    xyz: np.ndarray,
    *,
    soft_potential: np.ndarray,
    step_distance: np.ndarray,
    toward_direction: np.ndarray,
    scale_mm: float,
    max_step_mm: float,
    branch_ambiguity: np.ndarray | None = None,
    ambiguity_damping: float = 0.0,
) -> np.ndarray:
    """Project once using only non-negative step and the toward direction.

    ``soft_potential`` is accepted and validated to make the safety boundary
    explicit, but it is never used as displacement magnitude or sign.
    ``ambiguity_damping=1`` multiplies the step by ``1 - ambiguity``;
    ``ambiguity_damping=0`` disables ambiguity attenuation.
    """

    xyz = _validated_xyz(xyz, "xyz")
    n_points = len(xyz)
    soft_potential = np.asarray(soft_potential, dtype=np.float32)
    step_distance = np.asarray(step_distance, dtype=np.float32)
    toward_direction = np.asarray(toward_direction, dtype=np.float32)
    if soft_potential.shape != (n_points,):
        raise ValueError("soft_potential must have shape [N]")
    if step_distance.shape != (n_points,):
        raise ValueError("step_distance must have shape [N]")
    if toward_direction.shape != (n_points, 3):
        raise ValueError("toward_direction must have shape [N,3]")
    if not (
        np.all(np.isfinite(soft_potential))
        and np.all(np.isfinite(step_distance))
        and np.all(np.isfinite(toward_direction))
    ):
        raise ValueError("projection guidance contains non-finite values")
    if not np.isfinite(scale_mm) or scale_mm <= 0.0:
        raise ValueError("scale_mm must be finite and positive")
    if not np.isfinite(max_step_mm) or max_step_mm < 0.0:
        raise ValueError("max_step_mm must be finite and non-negative")
    if (
        not np.isfinite(ambiguity_damping)
        or ambiguity_damping < 0.0
        or ambiguity_damping > 1.0
    ):
        raise ValueError("ambiguity_damping must be in [0,1]")

    if branch_ambiguity is None:
        ambiguity = np.zeros(n_points, dtype=np.float32)
    else:
        ambiguity = np.asarray(branch_ambiguity, dtype=np.float32)
        if ambiguity.shape != (n_points,):
            raise ValueError("branch_ambiguity must have shape [N]")
        if not np.all(np.isfinite(ambiguity)):
            raise ValueError("branch_ambiguity contains non-finite values")
        ambiguity = np.clip(ambiguity, 0.0, 1.0)

    direction_norm = np.linalg.norm(toward_direction, axis=1, keepdims=True)
    unit_toward = np.divide(
        toward_direction,
        direction_norm,
        out=np.zeros_like(toward_direction),
        where=direction_norm > 1.0e-8,
    )

    # Negative step predictions are treated as zero.  The signed potential is
    # intentionally absent from this calculation.
    non_negative_step = np.maximum(step_distance, 0.0)
    attenuation = 1.0 - ambiguity_damping * ambiguity
    capped_step = np.minimum(
        non_negative_step * attenuation,
        np.float32(max_step_mm / scale_mm),
    )
    return (xyz + capped_step[:, None] * unit_toward).astype(
        np.float32, copy=False
    )


def project_candidates(
    model: SoftminGuidanceModel,
    latent: torch.Tensor,
    candidates: np.ndarray,
    device: str | torch.device,
    *,
    scale_mm: float,
    max_step_mm: float,
    iterations: int,
    ambiguity_damping: float = 0.0,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Repeatedly decode and bounded-project a candidate cloud."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    projected = _validated_xyz(candidates, "candidates").copy()
    for _ in range(iterations):
        potential, step, direction, ambiguity = decode_guidance_chunked(
            model, latent, projected, device, chunk_size
        )
        projected = bounded_project(
            projected,
            soft_potential=potential,
            step_distance=step,
            toward_direction=direction,
            scale_mm=scale_mm,
            max_step_mm=max_step_mm,
            branch_ambiguity=ambiguity,
            ambiguity_damping=ambiguity_damping,
        )
    return projected


def reconstruct_coarse_surface(
    model: SoftminGuidanceModel,
    cloud: InputCloud,
    device: str | torch.device,
    *,
    grid_res: int = 48,
    candidate_factor: float = 3.0,
    projection_iterations: int = 3,
    nbr_sz: int = 20,
    max_input_distance_mm: float | None = 10.0,
    max_step_mm: float = 5.0,
    ambiguity_damping: float = 1.0,
    bbox_pad_fraction: float = 0.15,
    prune_distance_mm: float | None = 20.0,
    chunk_size: int = 100_000,
    ranking_mode: str = "abs_potential_ambiguity",
    ambiguity_penalty_mm: float = 1.0,
):
    """Build one coarse normalized-coordinate PyVista surface."""

    if candidate_factor <= 0.0:
        raise ValueError("candidate_factor must be positive")
    if nbr_sz <= 0:
        raise ValueError("nbr_sz must be positive")

    latent = encode_input(model, cloud, device)
    guidance = evaluate_grid(
        model,
        latent,
        cloud.points_norm,
        grid_res,
        device,
        bbox_pad_fraction=bbox_pad_fraction,
        chunk_size=chunk_size,
    )

    minimum_required = max(3, nbr_sz * 2)
    candidate_count = max(
        minimum_required, int(candidate_factor * grid_res * grid_res)
    )
    candidate_indices = select_ranked_candidates(
        guidance.xyz,
        guidance.soft_potential,
        input_points_norm=cloud.points_norm,
        scale_mm=cloud.scale_mm,
        max_input_distance_mm=max_input_distance_mm,
        candidate_count=candidate_count,
        minimum_required=minimum_required,
        step_distance=guidance.step_distance,
        branch_ambiguity=guidance.branch_ambiguity,
        ranking_mode=ranking_mode,
        ambiguity_penalty_mm=ambiguity_penalty_mm,
    )
    projected = project_candidates(
        model,
        latent,
        guidance.xyz[candidate_indices],
        device,
        scale_mm=cloud.scale_mm,
        max_step_mm=max_step_mm,
        iterations=projection_iterations,
        ambiguity_damping=ambiguity_damping,
        chunk_size=chunk_size,
    )

    # Projection is bounded, but a wrong direction head could still move away.
    # Re-apply the independent geometric gate and fail closed if evidence drops.
    projected_gate = _input_gate_indices(
        projected,
        cloud.points_norm,
        cloud.scale_mm,
        max_input_distance_mm,
    )
    if len(projected_gate) < minimum_required:
        raise InsufficientEvidenceError(
            f"Only {len(projected_gate)} projected candidates remain within "
            f"the input-cloud gate; minimum is {minimum_required}."
        )
    projected = projected[projected_gate]

    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError(
            "pyvista is required for surface reconstruction and PLY output"
        ) from exc

    surface = pv.PolyData(projected).reconstruct_surface(nbr_sz=nbr_sz)
    if surface.n_points < 3 or surface.n_cells < 1:
        raise InsufficientEvidenceError(
            "pyvista surface reconstruction produced no usable triangles"
        )

    try:
        from .reconstruct import orient_components_consistently, prune_far_vertices
    except ImportError:
        from reconstruct import (  # type: ignore[no-redef]
            orient_components_consistently,
            prune_far_vertices,
        )

    points = np.asarray(surface.points, dtype=np.float32)
    triangles = np.asarray(surface.faces).reshape(-1, 4)[:, 1:4]
    triangles = orient_components_consistently(
        points, triangles, cloud.points_norm, cloud.normals
    )

    if prune_distance_mm is not None:
        if prune_distance_mm < 0.0:
            raise ValueError("prune_distance_mm must be non-negative or None")
        points, triangles, _ = prune_far_vertices(
            points,
            triangles,
            projected,
            prune_distance_mm / cloud.scale_mm,
        )
        if len(points) < 3 or len(triangles) < 1:
            raise InsufficientEvidenceError(
                "post-reconstruction prune removed all usable triangles"
            )

    faces = np.column_stack(
        [np.full(len(triangles), 3, dtype=np.int64), triangles]
    ).ravel()
    return pv.PolyData(points, faces)


def process(
    checkpoint_path: str | pathlib.Path,
    data_h5: str | pathlib.Path,
    output_ply: str | pathlib.Path,
    **reconstruction_options: Any,
) -> pathlib.Path:
    """Run the coarse reconstruction and save a world-coordinate PLY mesh."""

    device_option = reconstruction_options.pop("device", "auto")
    device = (
        "cuda"
        if device_option == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_option == "auto"
        else device_option
    )
    model = load_model(checkpoint_path, device)
    cloud = load_input_h5(data_h5)
    surface_norm = reconstruct_coarse_surface(
        model, cloud, device, **reconstruction_options
    )
    surface_world = surface_norm.copy(deep=True)
    surface_world.points = (
        np.asarray(surface_norm.points) * cloud.scale_mm + cloud.center
    )

    output_ply = pathlib.Path(output_ply)
    if output_ply.suffix.lower() != ".ply":
        output_ply = output_ply.with_suffix(".ply")
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    surface_world.save(str(output_ply))
    return output_ply


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Coarse mesh reconstruction with Softmin Guidance"
    )
    parser.add_argument("--checkpoint", "--ckpt", dest="checkpoint", required=True)
    parser.add_argument("--data", required=True, help="Input H5 with points/normals")
    parser.add_argument("--out", required=True, help="World-coordinate output PLY")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--grid-res", type=int, default=48)
    parser.add_argument("--candidate-factor", type=float, default=3.0)
    parser.add_argument("--projection-iterations", type=int, default=3)
    parser.add_argument("--nbr-sz", type=int, default=20)
    parser.add_argument("--input-dist-threshold-mm", type=float, default=10.0)
    parser.add_argument(
        "--ranking-mode", choices=RANKING_MODES,
        default="abs_potential_ambiguity",
    )
    parser.add_argument("--ambiguity-penalty-mm", type=float, default=1.0)
    parser.add_argument(
        "--no-input-gate", action="store_true",
        help="Analysis-only: disable the input-cloud support gate.",
    )
    parser.add_argument("--max-step-mm", type=float, default=5.0)
    parser.add_argument("--ambiguity-damping", type=float, default=1.0)
    parser.add_argument("--bbox-pad-fraction", type=float, default=0.15)
    parser.add_argument("--prune-dist-threshold-mm", type=float, default=20.0)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output = process(
        args.checkpoint,
        args.data,
        args.out,
        device=args.device,
        grid_res=args.grid_res,
        candidate_factor=args.candidate_factor,
        projection_iterations=args.projection_iterations,
        nbr_sz=args.nbr_sz,
        max_input_distance_mm=(
            None if args.no_input_gate else args.input_dist_threshold_mm
        ),
        max_step_mm=args.max_step_mm,
        ambiguity_damping=args.ambiguity_damping,
        bbox_pad_fraction=args.bbox_pad_fraction,
        prune_distance_mm=args.prune_dist_threshold_mm,
        chunk_size=args.chunk_size,
        ranking_mode=args.ranking_mode,
        ambiguity_penalty_mm=args.ambiguity_penalty_mm,
    )
    print(f"saved: {output}", flush=True)


if __name__ == "__main__":
    main()

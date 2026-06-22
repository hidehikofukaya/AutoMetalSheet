from __future__ import annotations

import pathlib
import sys

import h5py
import numpy as np
import pytest
import trimesh

PREPROCESS_DIR = pathlib.Path(__file__).parents[1] / "src" / "preprocess"
MODEL_DIR = pathlib.Path(__file__).parents[1] / "src" / "model"
sys.path.insert(0, str(PREPROCESS_DIR))
sys.path.insert(0, str(MODEL_DIR))

import midsurface_sampler as sampler
from dataset import MidSurfaceUDFDataset
from softmin_guidance_dataset import SoftminGuidanceDataset


def _single_triangle_mesh() -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        process=False,
    )


def _two_branch_mesh() -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.array(
            [
                [-2.0, -2.0, -1.0],
                [2.0, -2.0, -1.0],
                [0.0, 2.0, -1.0],
                [-2.0, -2.0, 1.0],
                [0.0, 2.0, 1.0],
                [2.0, -2.0, 1.0],
            ],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
        process=False,
    )


def _subdivided_square(divisions: int) -> trimesh.Trimesh:
    vertices = [
        [x / divisions, y / divisions, 0.0]
        for y in range(divisions + 1)
        for x in range(divisions + 1)
    ]
    faces = []
    stride = divisions + 1
    for y in range(divisions):
        for x in range(divisions):
            v0 = y * stride + x
            v1 = v0 + 1
            v2 = v0 + stride
            v3 = v2 + 1
            faces.extend([[v0, v1, v3], [v0, v3, v2]])
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )


def test_softmin_is_invariant_to_coplanar_tessellation_density():
    query = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
    coarse = sampler._softmin_guidance(
        _subdivided_square(1), query, tau_mm=0.5, k_faces=8, clip_radius_mm=None
    )
    fine = sampler._softmin_guidance(
        _subdivided_square(4), query, tau_mm=0.5, k_faces=32, clip_radius_mm=None
    )

    np.testing.assert_allclose(
        coarse.soft_potential, fine.soft_potential, atol=1.0e-7
    )
    np.testing.assert_allclose(coarse.soft_potential, [0.5], atol=1.0e-7)
    np.testing.assert_allclose(coarse.branch_ambiguity, [0.0], atol=1.0e-7)
    np.testing.assert_allclose(fine.branch_ambiguity, [0.0], atol=1.0e-7)


def test_softmin_guidance_ranges_and_potential_ordering():
    mesh = _two_branch_mesh()
    query_xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.1, 0.8],
            [0.2, 0.1, -0.8],
        ],
        dtype=np.float64,
    )

    guidance = sampler._softmin_guidance(
        mesh,
        query_xyz,
        tau_mm=0.5,
        k_faces=2,
        clip_radius_mm=None,
    )

    assert guidance.soft_potential.shape == (3,)
    assert guidance.step_distance.shape == (3,)
    assert guidance.soft_direction.shape == (3, 3)
    assert guidance.branch_ambiguity.shape == (3,)
    assert guidance.direction_strength.shape == (3,)
    assert np.all(np.isfinite(guidance.soft_potential))
    assert np.all(guidance.step_distance >= 0.0)
    assert np.all(guidance.soft_potential <= guidance.step_distance + 1e-6)
    assert np.all((guidance.branch_ambiguity >= 0.0) & (guidance.branch_ambiguity <= 1.0))
    assert np.all((guidance.direction_strength >= 0.0) & (guidance.direction_strength <= 1.0))
    assert guidance.branch_ambiguity[0] == pytest.approx(1.0, abs=1e-6)
    assert guidance.direction_strength[0] == pytest.approx(0.0, abs=1e-6)


def test_single_branch_has_near_zero_ambiguity_and_full_direction_strength():
    mesh = _single_triangle_mesh()
    query_xyz = np.array([[0.25, 0.25, 0.5]], dtype=np.float64)

    guidance = sampler._softmin_guidance(
        mesh,
        query_xyz,
        tau_mm=0.5,
        k_faces=8,
        clip_radius_mm=None,
    )

    assert guidance.branch_ambiguity[0] == pytest.approx(0.0, abs=1e-7)
    assert guidance.direction_strength[0] == pytest.approx(1.0, abs=1e-7)
    assert guidance.soft_potential[0] == pytest.approx(guidance.step_distance[0])
    assert np.linalg.norm(guidance.soft_direction[0]) == pytest.approx(1.0)


def test_legacy_softmin_api_still_returns_raw_and_direction_pair():
    mesh = _single_triangle_mesh()
    query_xyz = np.array([[0.25, 0.25, 0.5]], dtype=np.float64)

    raw, grad = sampler._softmin_udf_and_grad(
        mesh,
        query_xyz,
        tau_mm=0.5,
        k_faces=1,
        clip_radius_mm=None,
    )

    assert raw.shape == (1,)
    assert grad.shape == (1, 3)


def test_sample_query_legacy_return_and_alias_values_are_preserved():
    mesh = _single_triangle_mesh()

    result = sampler.sample_udf_queries(
        mesh,
        n_near=3,
        n_far=2,
        rng=np.random.default_rng(7),
        softmin_tau_mm=0.5,
        softmin_k_faces=1,
        softmin_clip_radius_mm=None,
    )

    assert len(result) == 3
    query_xyz, query_udf, query_grad = result
    guidance = sampler._softmin_guidance(
        mesh,
        query_xyz,
        tau_mm=0.5,
        k_faces=1,
        clip_radius_mm=None,
    )
    np.testing.assert_allclose(query_udf, guidance.soft_potential, atol=1e-7)
    np.testing.assert_allclose(query_grad, guidance.soft_direction, atol=1e-7)


def test_query_sampling_is_deterministic_for_the_same_generator_seed():
    mesh = _single_triangle_mesh()
    kwargs = dict(
        n_near=8,
        n_far=4,
        softmin_tau_mm=0.5,
        softmin_k_faces=1,
        softmin_clip_radius_mm=None,
    )

    first = sampler.sample_udf_queries(
        mesh, rng=np.random.default_rng(17), **kwargs
    )
    second = sampler.sample_udf_queries(
        mesh, rng=np.random.default_rng(17), **kwargs
    )

    for first_array, second_array in zip(first, second):
        np.testing.assert_array_equal(first_array, second_array)


def test_softmin_h5_guidance_contract(tmp_path: pathlib.Path):
    out_h5 = tmp_path / "guidance.h5"
    guidance = sampler.SoftminGuidance(
        soft_potential=np.array([-0.1, 0.4], dtype=np.float32),
        step_distance=np.array([0.2, 0.5], dtype=np.float32),
        soft_direction=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        branch_ambiguity=np.array([0.8, 0.1], dtype=np.float32),
        direction_strength=np.array([0.2, 0.95], dtype=np.float32),
    )

    sampler._write_dataset_h5(
        out_h5,
        points=np.zeros((2, 3), dtype=np.float32),
        normals=np.zeros((2, 3), dtype=np.float32),
        query_xyz=np.zeros((2, 3), dtype=np.float32),
        query_udf=guidance.soft_potential,
        query_grad=guidance.soft_direction,
        query_category=np.zeros(2, dtype=np.int64),
        source_stp=pathlib.Path("body001_filled.stp"),
        thickness_mm=1.2,
        softmin_enabled=True,
        softmin_tau_mm=0.5,
        softmin_k_faces=2,
        softmin_clip_radius_mm=5.0,
        softmin_guidance=guidance,
    )

    expected = {
        "query_soft_potential": ("softmin_potential", "mm"),
        "query_step_distance": ("hard_step_distance", "mm"),
        "query_soft_direction": ("soft_toward_surface_direction", "1"),
        "query_branch_ambiguity": ("normalized_branch_entropy", "1"),
        "query_direction_strength": ("pre_normalization_direction_norm", "1"),
    }
    with h5py.File(out_h5, "r") as h5:
        assert h5.attrs["schema_version"] == sampler.SOFTMIN_GUIDANCE_SCHEMA_VERSION
        assert h5.attrs["gradient_convention"] == sampler.TOWARD_SURFACE_GRADIENT_CONVENTION
        for field, (field_role, unit) in expected.items():
            assert field in h5
            assert h5[field].attrs["schema_version"] == sampler.SOFTMIN_GUIDANCE_SCHEMA_VERSION
            assert h5[field].attrs["field_role"] == field_role
            assert h5[field].attrs["unit"] == unit
        assert (
            h5["query_soft_direction"].attrs["gradient_convention"]
            == sampler.TOWARD_SURFACE_GRADIENT_CONVENTION
        )
        np.testing.assert_array_equal(h5["query_udf"][:], h5["query_soft_potential"][:])
        np.testing.assert_array_equal(h5["query_grad"][:], h5["query_soft_direction"][:])

    dataset = SoftminGuidanceDataset(out_h5, n_query_sample=None)
    assert len(dataset) == 1
    item = dataset[0]
    np.testing.assert_array_equal(
        item["query_soft_potential"].numpy(), guidance.soft_potential
    )

    with pytest.raises(ValueError, match="Softmin Guidance dataset"):
        MidSurfaceUDFDataset(out_h5, n_query_sample=None)[0]

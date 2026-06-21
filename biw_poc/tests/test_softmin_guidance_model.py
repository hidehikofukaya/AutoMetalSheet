from __future__ import annotations

import json
import pathlib
import sys

import h5py
import numpy as np
import pytest
import torch


MODEL_DIR = pathlib.Path(__file__).parents[1] / "src" / "model"
sys.path.insert(0, str(MODEL_DIR))

from softmin_guidance import (
    SoftminGuidanceForwardOutput,
    SoftminGuidanceModel,
    SoftminGuidancePrediction,
    build_checkpoint_metadata,
)
from softmin_guidance_dataset import SoftminGuidanceDataset


def _small_model() -> SoftminGuidanceModel:
    return SoftminGuidanceModel(
        token_dim=24,
        n_tokens=4,
        k_neighbors=3,
        n_latents=3,
        enc_layers=1,
        dec_layers=1,
        n_heads=4,
        n_freqs=2,
    )


def test_forward_contract_shapes_ranges_and_backprop():
    torch.manual_seed(7)
    model = _small_model()
    points = torch.randn(2, 12, 3)
    normals = torch.nn.functional.normalize(torch.randn(2, 12, 3), dim=-1)
    query_xyz = torch.randn(2, 7, 3)

    output = model(points, normals, query_xyz)
    assert isinstance(output, SoftminGuidanceForwardOutput)
    assert output.soft_potential.shape == (2, 7)
    assert output.step_distance.shape == (2, 7)
    assert output.direction_toward_surface.shape == (2, 7, 3)
    assert output.branch_ambiguity.shape == (2, 7)
    assert output.latent.shape == (2, 3, 24)
    assert torch.all(output.step_distance > 0)
    assert torch.all(
        (output.branch_ambiguity >= 0) & (output.branch_ambiguity <= 1)
    )
    assert torch.allclose(
        torch.linalg.vector_norm(output.direction_toward_surface, dim=-1),
        torch.ones(2, 7),
        atol=1.0e-5,
    )

    decoded = model.decode(query_xyz, output.latent)
    assert isinstance(decoded, SoftminGuidancePrediction)
    loss = (
        output.soft_potential.square().mean()
        + output.step_distance.mean()
        + output.direction_toward_surface.square().mean()
        + output.branch_ambiguity.mean()
        + output.latent.square().mean()
    )
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert model.tokenizer.pointnet.mlp1[0].weight.grad is not None
    assert model.decoder.head[-1].weight.grad is not None


def test_decoder_activations_include_signed_and_zero_vector_fallback():
    model = _small_model()
    with torch.no_grad():
        final = model.decoder.head[-1]
        final.weight.zero_()
        final.bias.copy_(torch.tensor([-2.0, -5.0, 0.0, 0.0, 0.0, 0.0]))

    prediction = model.decode(torch.zeros(1, 2, 3), torch.zeros(1, 3, 24))
    assert torch.equal(prediction.soft_potential, torch.full((1, 2), -2.0))
    assert torch.all(prediction.step_distance > 0)
    assert torch.equal(
        prediction.direction_toward_surface,
        torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
    )
    assert torch.equal(prediction.branch_ambiguity, torch.full((1, 2), 0.5))


def test_checkpoint_metadata_is_json_serializable_and_explicit():
    metadata = build_checkpoint_metadata(_small_model())
    json.dumps(metadata)
    assert metadata["schema_version"] == "model.softmin_guidance.v1"
    assert metadata["stage_a_schema_version"] == "stage_a.softmin_guidance.v1"
    assert metadata["output_contract"]["soft_potential"]["activation"] == "linear"
    assert metadata["output_contract"]["step_distance"]["activation"] == "softplus"
    assert (
        metadata["output_contract"]["direction_toward_surface"][
            "gradient_convention"
        ]
        == "toward_surface"
    )
    assert metadata["output_contract"]["branch_ambiguity"]["activation"] == "sigmoid"


def _write_guidance_h5(path: pathlib.Path, include_ambiguity: bool = True) -> None:
    n_query = 10
    query_index = np.arange(n_query, dtype=np.float32)
    points = np.array(
        [
            [-2.0, -1.0, 0.0],
            [2.0, -1.0, 0.0],
            [-2.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )
    query_xyz = np.stack(
        [query_index, query_index + 0.25, query_index - 0.5], axis=1
    )
    toward_surface = np.zeros((n_query, 3), dtype=np.float32)
    toward_surface[np.arange(n_query), np.arange(n_query) % 3] = 1.0

    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("points", data=points)
        h5_file.create_dataset("normals", data=np.tile([0.0, 0.0, 1.0], (6, 1)))
        h5_file.create_dataset("query_xyz", data=query_xyz)
        h5_file.create_dataset(
            "query_soft_potential", data=100.0 + query_index
        )
        h5_file.create_dataset("query_step_distance", data=200.0 + query_index)
        h5_file.create_dataset("query_soft_direction", data=toward_surface)
        if include_ambiguity:
            h5_file.create_dataset(
                "query_branch_ambiguity", data=query_index / 10.0
            )
        h5_file.create_dataset(
            "query_direction_strength", data=1.0 - query_index / 20.0
        )
        h5_file.create_dataset(
            "query_category", data=np.arange(n_query, dtype=np.int64)
        )
        h5_file.attrs["thickness_mm"] = 1.25
        h5_file.attrs["schema_version"] = "stage_a.softmin_guidance.v1"


def test_dataset_subsamples_all_query_fields_with_one_index_and_keeps_mm(tmp_path):
    path = tmp_path / "part_dataset.h5"
    _write_guidance_h5(path)
    dataset = SoftminGuidanceDataset(
        [path], n_query_sample=5, n_points_sample=4, seed=13
    )

    item = dataset[0]
    potential = item["query_soft_potential"]
    selected_index = potential - 100.0
    assert torch.equal(item["query_step_distance"], potential + 100.0)
    assert torch.allclose(item["query_branch_ambiguity"], selected_index / 10.0)
    assert torch.allclose(
        item["query_direction_strength"], 1.0 - selected_index / 20.0
    )
    assert torch.equal(item["query_category"], selected_index.to(torch.int64))
    assert torch.equal(
        item["query_soft_direction"].argmax(dim=-1),
        selected_index.to(torch.int64) % 3,
    )

    # Targets remain world-mm values while coordinates can be exactly restored
    # with the returned per-part transform.
    restored_query = item["query_xyz"] * item["scale"] + item["center"]
    expected_query = torch.stack(
        [selected_index, selected_index + 0.25, selected_index - 0.5], dim=1
    )
    assert torch.allclose(restored_query, expected_query, atol=1.0e-6)
    assert potential.min() >= 100.0
    assert item["points"].shape == (4, 3)
    assert item["normals"].shape == (4, 3)
    assert item["center"].shape == (3,)
    assert item["scale"].ndim == 0
    assert item["thickness_mm"].item() == pytest.approx(1.25)


def test_dataset_missing_new_stage_a_field_fails_at_construction(tmp_path):
    path = tmp_path / "missing_dataset.h5"
    _write_guidance_h5(path, include_ambiguity=False)

    with pytest.raises(KeyError, match="query_branch_ambiguity"):
        SoftminGuidanceDataset([path], n_query_sample=None)

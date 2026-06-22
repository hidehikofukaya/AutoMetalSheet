from __future__ import annotations

import pathlib
import sys

import h5py
import numpy as np
import pytest
import torch


MODEL_DIR = pathlib.Path(__file__).parents[1] / "src" / "model"
sys.path.insert(0, str(MODEL_DIR))

import train_softmin_guidance as trainer
from softmin_guidance import SoftminGuidancePrediction
from train_softmin_guidance import (
    GuidanceLossConfig,
    TrainConfig,
    _build_datasets,
    compute_guidance_losses,
    evaluate_epoch,
    potential_gradient_consistency_loss,
    split_query_indices,
    train_softmin_guidance,
)


def _loss_batch(
    *,
    potential: torch.Tensor,
    step: torch.Tensor,
    direction: torch.Tensor,
    ambiguity: torch.Tensor,
    strength: torch.Tensor,
    category: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "query_soft_potential": potential,
        "query_step_distance": step,
        "query_soft_direction": direction,
        "query_branch_ambiguity": ambiguity,
        "query_direction_strength": strength,
        "query_category": category,
        "scale": torch.ones(potential.shape[0]),
    }


def test_independent_losses_keep_signed_targets_and_apply_weights_and_clips():
    query_xyz = torch.zeros(1, 3, 3)
    direction = torch.tensor([[[1.0, 0.0, 0.0]] * 3])
    prediction = SoftminGuidancePrediction(
        soft_potential=torch.tensor([[-2.0, 0.0, 0.0]]),
        step_distance=torch.tensor([[0.5, 0.0, 0.0]]),
        direction_toward_surface=direction,
        branch_ambiguity=torch.zeros(1, 3),
    )
    batch = _loss_batch(
        potential=torch.tensor([[-2.0, 10.0, 10.0]]),
        step=torch.tensor([[0.5, 10.0, 10.0]]),
        direction=torch.tensor(
            [[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]]
        ),
        ambiguity=torch.tensor([[0.0, 1.0, 1.0]]),
        strength=torch.tensor([[1.0, 0.25, 0.0]]),
        category=torch.tensor([[0, 1, 2]]),
    )
    config = GuidanceLossConfig(
        category_weights=(1.0, 2.0, 3.0),
        distance_clips_mm=(1.0, 2.0, 3.0),
        lambda_potential=1.0,
        lambda_step=1.0,
        lambda_direction=1.0,
        lambda_ambiguity=1.0,
    )

    losses = compute_guidance_losses(
        prediction, batch, query_xyz, config, create_graph=False
    )

    # No floor/softplus is applied. The deliberately tiny 1 mm near clip
    # truncates the -2 mm target to -1 mm, while predictions remain
    # unclamped so they always receive a recovery gradient.
    assert losses["potential"].item() == pytest.approx(14.0 / 6.0)
    assert losses["step"].item() == pytest.approx(13.0 / 6.0)
    assert losses["direction"].item() == pytest.approx(2.0 / 3.0)
    assert losses["ambiguity"].item() == pytest.approx(5.0 / 6.0)
    assert losses["near_potential_abs_sum"].item() == pytest.approx(0.0)
    assert losses["near_step_abs_sum"].item() == pytest.approx(0.0)


def test_potential_gradient_consistency_uses_negative_gradient():
    query_xyz = torch.zeros(1, 2, 3, requires_grad=True)
    signed_potential = query_xyz[..., 0]
    toward_negative_x = torch.tensor(
        [[[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]]
    )
    weights = torch.ones(1, 2)

    aligned = potential_gradient_consistency_loss(
        signed_potential,
        query_xyz,
        toward_negative_x,
        weights,
        create_graph=True,
    )
    opposed = potential_gradient_consistency_loss(
        signed_potential,
        query_xyz,
        -toward_negative_x,
        weights,
        create_graph=True,
    )

    assert aligned.item() == pytest.approx(0.0, abs=1.0e-7)
    assert opposed.item() == pytest.approx(2.0, abs=1.0e-7)


def test_consistency_ignores_exactly_flat_initial_potential():
    query_xyz = torch.zeros(1, 2, 3, requires_grad=True)
    signed_potential = query_xyz[..., 0] * 0.0
    direction = torch.tensor([[[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]])

    loss = potential_gradient_consistency_loss(
        signed_potential,
        query_xyz,
        direction,
        torch.ones(1, 2),
        create_graph=True,
    )
    loss.backward()

    assert loss.item() == 0.0
    assert torch.isfinite(query_xyz.grad).all()


def test_potential_prediction_outside_clip_still_receives_gradient():
    predicted_potential = torch.tensor([[2.0]], requires_grad=True)
    prediction = SoftminGuidancePrediction(
        soft_potential=predicted_potential,
        step_distance=torch.tensor([[0.1]]),
        direction_toward_surface=torch.tensor([[[1.0, 0.0, 0.0]]]),
        branch_ambiguity=torch.tensor([[0.5]]),
    )
    batch = _loss_batch(
        potential=torch.tensor([[0.0]]),
        step=torch.tensor([[0.1]]),
        direction=torch.tensor([[[1.0, 0.0, 0.0]]]),
        ambiguity=torch.tensor([[0.5]]),
        strength=torch.tensor([[1.0]]),
        category=torch.tensor([[0]]),
    )
    batch["scale"] = torch.tensor([100.0])
    config = GuidanceLossConfig(
        distance_clips_mm=(5.0, 5.0, 40.0),
        lambda_step=0.0,
        lambda_direction=0.0,
        lambda_ambiguity=0.0,
    )

    losses = compute_guidance_losses(
        prediction, batch, torch.zeros(1, 1, 3), config, create_graph=False
    )
    losses["total"].backward()

    assert predicted_potential.grad is not None
    assert predicted_potential.grad.abs().item() > 0.0


def test_step_prediction_outside_clip_still_receives_gradient():
    predicted_step = torch.tensor([[10.0]], requires_grad=True)
    prediction = SoftminGuidancePrediction(
        soft_potential=torch.tensor([[0.0]]),
        step_distance=predicted_step,
        direction_toward_surface=torch.tensor([[[1.0, 0.0, 0.0]]]),
        branch_ambiguity=torch.tensor([[0.5]]),
    )
    batch = _loss_batch(
        potential=torch.tensor([[0.0]]),
        step=torch.tensor([[0.1]]),
        direction=torch.tensor([[[1.0, 0.0, 0.0]]]),
        ambiguity=torch.tensor([[0.5]]),
        strength=torch.tensor([[1.0]]),
        category=torch.tensor([[0]]),
    )
    config = GuidanceLossConfig(
        distance_clips_mm=(5.0, 5.0, 40.0),
        lambda_potential=0.0,
        lambda_direction=0.0,
        lambda_ambiguity=0.0,
    )

    losses = compute_guidance_losses(
        prediction, batch, torch.zeros(1, 1, 3), config, create_graph=False
    )
    losses["total"].backward()

    assert predicted_step.grad is not None
    assert predicted_step.grad.abs().item() > 0.0


def _write_training_h5(path: pathlib.Path) -> None:
    rng = np.random.default_rng(17)
    points = rng.normal(size=(8, 3)).astype(np.float32)
    normals = points / np.maximum(
        np.linalg.norm(points, axis=1, keepdims=True), 1.0e-6
    )
    query_xyz = rng.normal(scale=0.6, size=(9, 3)).astype(np.float32)
    step = np.linalg.norm(query_xyz, axis=1).astype(np.float32)
    toward = -query_xyz / np.maximum(step[:, None], 1.0e-6)
    potential = (step - 0.5).astype(np.float32)
    ambiguity = np.linspace(0.0, 0.8, len(query_xyz), dtype=np.float32)
    strength = np.linspace(1.0, 0.4, len(query_xyz), dtype=np.float32)
    category = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)

    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("points", data=points)
        h5_file.create_dataset("normals", data=normals)
        h5_file.create_dataset("query_xyz", data=query_xyz)
        h5_file.create_dataset("query_soft_potential", data=potential)
        h5_file.create_dataset("query_step_distance", data=step)
        h5_file.create_dataset("query_soft_direction", data=toward)
        h5_file.create_dataset("query_branch_ambiguity", data=ambiguity)
        h5_file.create_dataset("query_direction_strength", data=strength)
        h5_file.create_dataset("query_category", data=category)
        h5_file.attrs["schema_version"] = "stage_a.softmin_guidance.v1"
        h5_file.attrs["gradient_convention"] = "toward_surface"
        h5_file.attrs["coordinate_unit"] = "mm"
        h5_file.attrs["thickness_mm"] = 1.0


def test_query_holdout_indices_are_disjoint_and_seed_deterministic(tmp_path):
    data_path = tmp_path / "tiny_dataset.h5"
    _write_training_h5(data_path)

    categories = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)
    train_a, val_a = split_query_indices(
        9, val_fraction=1.0 / 3.0, seed=41, categories=categories
    )
    train_b, val_b = split_query_indices(
        9, val_fraction=1.0 / 3.0, seed=41, categories=categories
    )
    train_c, val_c = split_query_indices(
        9, val_fraction=1.0 / 3.0, seed=42, categories=categories
    )

    assert np.array_equal(train_a, train_b)
    assert np.array_equal(val_a, val_b)
    assert not (
        np.array_equal(train_a, train_c) and np.array_equal(val_a, val_c)
    )
    assert set(train_a).isdisjoint(set(val_a))
    assert sorted(np.concatenate([train_a, val_a]).tolist()) == list(range(9))
    assert sorted(categories[val_a].tolist()) == [0, 1, 2]

    config = TrainConfig(
        data=data_path,
        val_fraction=1.0 / 3.0,
        n_query_sample=None,
        seed=41,
    )
    train_dataset, val_dataset, split_metadata = _build_datasets(config)
    assert val_dataset is not None
    train_item = train_dataset[0]
    val_item = val_dataset[0]
    assert set(train_item["query_index"].tolist()).isdisjoint(
        set(val_item["query_index"].tolist())
    )
    assert split_metadata["strategy"] == "query_holdout"
    assert split_metadata["query_overlap_count"] == 0
    assert split_metadata["shared_point_cloud_input"] is True


def test_val_data_uses_disjoint_part_holdout(tmp_path):
    train_path = tmp_path / "train_dataset.h5"
    val_path = tmp_path / "val_dataset.h5"
    _write_training_h5(train_path)
    _write_training_h5(val_path)

    train_dataset, val_dataset, split_metadata = _build_datasets(
        TrainConfig(data=train_path, val_data=val_path, n_query_sample=None)
    )

    assert len(train_dataset) == 1
    assert val_dataset is not None and len(val_dataset) == 1
    assert split_metadata["strategy"] == "part_holdout"
    assert split_metadata["part_overlap_count"] == 0
    assert split_metadata["query_overlap_count"] == 0

    with pytest.raises(ValueError, match="overlap"):
        _build_datasets(
            TrainConfig(data=train_path, val_data=train_path, n_query_sample=None)
        )

    with h5py.File(train_path, "a") as h5_file:
        h5_file.attrs["source_stp"] = "shared/body002_filled.stp"
    with h5py.File(val_path, "a") as h5_file:
        h5_file.attrs["source_stp"] = "shared/body002_filled.stp"
    with pytest.raises(ValueError, match="source part identities overlap"):
        _build_datasets(
            TrainConfig(data=train_path, val_data=val_path, n_query_sample=None)
        )


def _epoch_metrics(near_composite: float) -> dict[str, float]:
    return {
        "total_loss": near_composite,
        "potential_loss": near_composite,
        "step_loss": near_composite,
        "direction_loss": 0.0,
        "ambiguity_loss": 0.0,
        "consistency_loss": 0.0,
        "near_potential_mae_mm": near_composite / 2.0,
        "near_step_mae_mm": near_composite / 2.0,
        "near_direction_cosine_error": 0.0,
        "near_ambiguity_mae": 0.0,
        "near_composite": near_composite,
    }


def test_evaluate_epoch_uses_eval_mode_and_no_grad():
    class EvalContractModel(torch.nn.Module):
        def forward(self, points, normals, query_xyz):
            assert self.training is False
            assert torch.is_grad_enabled() is False
            batch_size, n_query, _ = query_xyz.shape
            return SoftminGuidancePrediction(
                soft_potential=torch.zeros(batch_size, n_query),
                step_distance=torch.ones(batch_size, n_query),
                direction_toward_surface=torch.nn.functional.normalize(
                    torch.ones(batch_size, n_query, 3), dim=-1
                ),
                branch_ambiguity=torch.full((batch_size, n_query), 0.5),
            )

    batch = {
        "points": torch.zeros(1, 4, 3),
        "normals": torch.zeros(1, 4, 3),
        "query_xyz": torch.zeros(1, 3, 3),
        "query_soft_potential": torch.zeros(1, 3),
        "query_step_distance": torch.ones(1, 3),
        "query_soft_direction": torch.nn.functional.normalize(
            torch.ones(1, 3, 3), dim=-1
        ),
        "query_branch_ambiguity": torch.full((1, 3), 0.5),
        "query_direction_strength": torch.ones(1, 3),
        "query_category": torch.tensor([[0, 1, 2]]),
        "scale": torch.ones(1),
    }
    model = EvalContractModel().train()

    metrics = evaluate_epoch(
        model,
        [batch],
        torch.device("cpu"),
        TrainConfig(data="unused.h5", lambda_consistency=1.0),
    )

    assert metrics["consistency_loss"] == 0.0
    assert model.training is False


def test_best_checkpoint_uses_validation_metric_and_saves_prefixed_history(
    tmp_path, monkeypatch
):
    data_path = tmp_path / "tiny_dataset.h5"
    checkpoint_dir = tmp_path / "checkpoints"
    _write_training_h5(data_path)
    train_metrics = iter([_epoch_metrics(4.0), _epoch_metrics(1.0)])
    val_metrics = iter([_epoch_metrics(2.0), _epoch_metrics(3.0)])

    def fake_train_epoch(model, loader, device, optimizer, config):
        optimizer.step()
        return next(train_metrics)

    monkeypatch.setattr(
        trainer, "run_epoch", fake_train_epoch
    )
    monkeypatch.setattr(
        trainer, "evaluate_epoch", lambda *args, **kwargs: next(val_metrics)
    )

    summary = train_softmin_guidance(
        TrainConfig(
            data=data_path,
            val_fraction=1.0 / 3.0,
            ckpt_dir=checkpoint_dir,
            epochs=2,
            batch_size=1,
            n_query_sample=None,
            seed=7,
            device="cpu",
            token_dim=8,
            n_tokens=2,
            k_neighbors=2,
            n_latents=2,
            enc_layers=1,
            dec_layers=1,
            n_heads=2,
            n_freqs=1,
            ckpt_every=0,
            log_every=0,
        )
    )

    assert summary["best_metrics"]["epoch"] == 1.0
    assert summary["best_metrics"]["selection_split"] == "val"
    checkpoint = torch.load(
        summary["best_checkpoint"], map_location="cpu", weights_only=False
    )
    assert checkpoint["metadata"]["best_checkpoint_selection"]["metric"] == (
        "val_near_composite"
    )
    assert (
        checkpoint["metadata"]["evidence_scope"]["generalization_valid"]
        is False
    )
    assert (
        checkpoint["metadata"]["evidence_scope"][
            "query_holdout_is_same_part_interpolation_only"
        ]
        is True
    )
    assert checkpoint["metrics"]["train_near_composite"] == pytest.approx(4.0)
    assert checkpoint["metrics"]["val_near_composite"] == pytest.approx(2.0)
    assert checkpoint["metrics"]["best_near_composite"] == pytest.approx(2.0)
    assert len(checkpoint["history"]) == 1
    assert "train_total_loss" in checkpoint["history"][0]
    assert "val_total_loss" in checkpoint["history"][0]


def test_tiny_cpu_training_runs_one_epoch_and_saves_explicit_best_checkpoint(
    tmp_path,
):
    data_path = tmp_path / "tiny_dataset.h5"
    checkpoint_dir = tmp_path / "checkpoints"
    _write_training_h5(data_path)
    config = TrainConfig(
        data=data_path,
        ckpt_dir=checkpoint_dir,
        epochs=1,
        batch_size=1,
        lr=1.0e-3,
        n_query_sample=6,
        n_points_sample=8,
        seed=5,
        device="cpu",
        lambda_consistency=0.05,
        token_dim=8,
        n_tokens=2,
        k_neighbors=2,
        n_latents=2,
        enc_layers=1,
        dec_layers=1,
        n_heads=2,
        n_freqs=1,
        ckpt_every=0,
        log_every=0,
    )

    summary = train_softmin_guidance(config)

    assert summary["device"] == "cpu"
    assert len(summary["history"]) == 1
    best_path = pathlib.Path(summary["best_checkpoint"])
    last_path = pathlib.Path(summary["last_checkpoint"])
    assert best_path.is_file()
    assert last_path.is_file()

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    assert {"metadata", "cfg", "metrics", "model"} <= checkpoint.keys()
    assert checkpoint["metadata"]["trainer_schema_version"] == (
        "train.softmin_guidance.v1"
    )
    potential_contract = checkpoint["metadata"]["loss_contract"]["potential"]
    assert potential_contract["target"] == "signed_world_mm"
    assert potential_contract["softplus_or_floor"] is False
    selection = checkpoint["metadata"]["best_checkpoint_selection"]
    assert selection["metric"] == "train_near_composite"
    assert selection["source"] == "train_fallback"
    assert "near_potential_mae_mm" in selection["formula"]
    assert "near_step_mae_mm" in selection["formula"]
    assert "near_direction_cosine_error" in selection["formula"]
    assert "near_ambiguity_mae" in selection["formula"]
    assert selection["requires_post_training_field_gate"] is True
    assert checkpoint["metadata"]["evidence_scope"]["train_fallback"] is True

    metrics = checkpoint["metrics"]
    expected_composite = (
        config.best_potential_weight * metrics["train_near_potential_mae_mm"]
        + config.best_step_weight * metrics["train_near_step_mae_mm"]
        + config.best_direction_weight_mm
        * metrics["train_near_direction_cosine_error"]
        + config.best_ambiguity_weight_mm
        * metrics["train_near_ambiguity_mae"]
    )
    assert metrics["train_near_composite"] == pytest.approx(expected_composite)
    assert metrics["best_near_composite"] == pytest.approx(expected_composite)
    assert "val_near_composite" not in metrics
    assert checkpoint["history"] == summary["history"]
    assert checkpoint["cfg"]["device"] == "cpu"

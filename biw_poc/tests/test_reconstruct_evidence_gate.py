from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

MODEL_DIR = pathlib.Path(__file__).parents[1] / "src" / "model"
sys.path.insert(0, str(MODEL_DIR))

import reconstruct


def test_candidate_pool_fails_closed_when_evidence_is_insufficient():
    with pytest.raises(reconstruct.InsufficientEvidenceError):
        reconstruct.select_candidate_pool(np.array([1, 2, 3]), grid_count=100, minimum_required=4)


def test_candidate_pool_never_falls_back_to_ungated_grid():
    selected = reconstruct.select_candidate_pool(
        np.array([2, 7, 9, 11]), grid_count=20, minimum_required=4
    )
    assert selected.tolist() == [2, 7, 9, 11]


"""CPU-only tests for shared train/validation split helpers."""

from __future__ import annotations

import numpy as np
import pytest

from ecg_adv_gen.training import random_split_indices


def test_random_split_indices_matches_direct_finetune_zero_fraction_policy():
    train, val = random_split_indices(10, 0.0, seed=5, zero_val_policy="identity")

    assert train.dtype == np.int64
    assert val.dtype == np.int64
    assert train.tolist() == list(range(10))
    assert val.tolist() == list(range(10))


def test_random_split_indices_matches_ecgfounder_kshot_zero_fraction_policy():
    train, val = random_split_indices(10, 0.0, seed=5, zero_val_policy="permuted")

    assert train.tolist() == [7, 6, 1, 3, 2, 4, 0, 9, 5, 8]
    assert val.tolist() == [7, 6, 1, 3, 2, 4, 0, 9, 5, 8]


def test_random_split_indices_matches_legacy_positive_fraction():
    train, val = random_split_indices(10, 0.3, seed=5)
    train_min, val_min = random_split_indices(10, 0.01, seed=5)

    assert train.tolist() == [0, 2, 3, 4, 5, 8, 9]
    assert val.tolist() == [1, 6, 7]
    assert train_min.tolist() == [0, 1, 2, 3, 4, 5, 6, 8, 9]
    assert val_min.tolist() == [7]


def test_random_split_indices_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="non-negative"):
        random_split_indices(-1, 0.2, seed=1)
    with pytest.raises(ValueError, match="unknown zero_val_policy"):
        random_split_indices(2, 0.0, seed=1, zero_val_policy="bad")  # type: ignore[arg-type]

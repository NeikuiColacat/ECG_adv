"""CPU-only tests for paper-safe model-selection policy semantics."""

from __future__ import annotations

import pytest
import numpy as np

from ecg_adv_gen.evaluation import (
    ALLOWED_SELECTION_DATA,
    SELECTION_POLICY,
    SelectionPolicyError,
    compute_source_target_selection_score,
    has_forbidden_selection_reference,
    split_target_train_val_indices,
    validate_selection_policy,
)


def _valid_selection() -> dict:
    return {
        "policy": SELECTION_POLICY,
        "allowed_data": list(ALLOWED_SELECTION_DATA),
        "forbid_heldout_target_labels": True,
        "forbid_full_target_distribution_tuning": True,
    }


def test_validate_selection_policy_accepts_k500_internal_source_floor_contract():
    validated = validate_selection_policy(_valid_selection())

    assert validated["policy"] == SELECTION_POLICY
    assert validated["allowed_data"] == list(ALLOWED_SELECTION_DATA)
    assert validated["forbid_heldout_target_labels"] is True


def test_validate_selection_policy_rejects_heldout_target_selection_references():
    bad = _valid_selection()
    bad["allowed_data"] = [*ALLOWED_SELECTION_DATA, "pn2021_heldout"]

    with pytest.raises(SelectionPolicyError, match="allowed_data"):
        validate_selection_policy(bad)

    assert has_forbidden_selection_reference(["--selection_source", "pn2021_heldout"]) is True


def test_validate_selection_policy_rejects_relaxed_forbid_flags():
    bad = _valid_selection()
    bad["forbid_full_target_distribution_tuning"] = False

    with pytest.raises(SelectionPolicyError, match="forbid_full_target_distribution_tuning"):
        validate_selection_policy(bad)


def test_compute_source_target_selection_score_preserves_fullft_modes():
    source = {"macro_auprc": 0.7}
    target = {"macro_auprc": 0.9}

    assert compute_source_target_selection_score(
        selection_metric="source_auprc",
        source_metrics=source,
        target_val_metrics=None,
    ) == pytest.approx(0.7)
    assert compute_source_target_selection_score(
        selection_metric="target_val_auprc",
        source_metrics=source,
        target_val_metrics=target,
    ) == pytest.approx(0.9)
    assert compute_source_target_selection_score(
        selection_metric="source_plus_target_val_auprc",
        source_metrics=source,
        target_val_metrics=target,
        target_val_score_weight=0.25,
    ) == pytest.approx(0.75)

    with pytest.raises(SelectionPolicyError, match="requires target validation"):
        compute_source_target_selection_score(
            selection_metric="target_val_auprc",
            source_metrics=source,
            target_val_metrics=None,
        )
    with pytest.raises(SelectionPolicyError, match="unknown selection_metric"):
        compute_source_target_selection_score(
            selection_metric="heldout_target_auprc",
            source_metrics=source,
            target_val_metrics=target,
        )


def test_split_target_train_val_indices_preserves_random_legacy_behavior():
    record_ids = np.asarray([f"R{i}" for i in range(8)], dtype=str)
    labels = np.asarray(
        [
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
        ],
        dtype=np.float32,
    )
    target_idx = np.arange(8, dtype=np.int64)

    train_idx, val_idx, train_ids, val_ids = split_target_train_val_indices(
        target_idx,
        record_ids,
        labels,
        val_count=3,
        split_seed=11,
        split_mode="random",
    )

    assert train_idx.tolist() == [0, 1, 2, 6, 7]
    assert val_idx.tolist() == [3, 4, 5]
    assert train_ids == {"R0", "R1", "R2", "R6", "R7"}
    assert val_ids == {"R3", "R4", "R5"}


def test_split_target_train_val_indices_preserves_stratified_legacy_behavior():
    record_ids = np.asarray([f"R{i}" for i in range(8)], dtype=str)
    labels = np.asarray(
        [
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
        ],
        dtype=np.float32,
    )
    target_idx = np.arange(8, dtype=np.int64)

    train_idx, val_idx, train_ids, val_ids = split_target_train_val_indices(
        target_idx,
        record_ids,
        labels,
        val_count=3,
        split_seed=11,
        split_mode="stratified",
    )

    assert train_idx.tolist() == [1, 2, 5, 6, 7]
    assert val_idx.tolist() == [0, 3, 4]
    assert train_ids == {"R1", "R2", "R5", "R6", "R7"}
    assert val_ids == {"R0", "R3", "R4"}


def test_split_target_train_val_indices_handles_zero_and_rejects_invalid_requests():
    record_ids = np.asarray([f"R{i}" for i in range(4)], dtype=str)
    labels = np.eye(5, dtype=np.float32)[:4]
    target_idx = np.arange(4, dtype=np.int64)

    train_idx, val_idx, train_ids, val_ids = split_target_train_val_indices(
        target_idx,
        record_ids,
        labels,
        val_count=0,
        split_seed=11,
    )

    assert train_idx.tolist() == [0, 1, 2, 3]
    assert val_idx.tolist() == []
    assert train_ids == {"R0", "R1", "R2", "R3"}
    assert val_ids == set()
    with pytest.raises(ValueError, match="must be smaller"):
        split_target_train_val_indices(target_idx, record_ids, labels, 4, 11)
    with pytest.raises(ValueError, match="unknown target_val_split_mode"):
        split_target_train_val_indices(target_idx, record_ids, labels, 1, 11, "heldout")

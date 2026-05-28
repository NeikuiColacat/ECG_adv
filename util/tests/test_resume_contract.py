"""CPU-only tests for resume contract helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from ecg_adv_gen.training.resume_contract import (
    resume_contract_mismatches,
    should_save_initial_best_model,
    validate_resume_contract,
)


def test_should_save_initial_best_model_only_for_fresh_runs():
    assert should_save_initial_best_model(None) is True
    assert should_save_initial_best_model(Path("checkpoints/checkpoint_latest.pt")) is False


def test_resume_contract_rejects_critical_drift_by_default():
    saved = {"center_name": "ningbo", "seed": 1, "classes_in_scope": ["NORM", "STTC"]}
    current = {"center_name": "georgia", "seed": 1, "classes_in_scope": ["NORM", "STTC"]}
    with pytest.raises(ValueError, match="center_name"):
        validate_resume_contract(saved, current, allow_drift=False)


def test_resume_contract_reports_allowed_drift():
    saved = {"center_name": "ningbo", "quick_eval_centers": ("ningbo",)}
    current = {"center_name": "georgia", "quick_eval_centers": ["georgia"]}
    mismatches = validate_resume_contract(saved, current, allow_drift=True)
    assert [item["key"] for item in mismatches] == ["center_name", "quick_eval_centers"]


def test_resume_contract_normalizes_tuple_list_values():
    saved = {"quick_eval_centers": ("ningbo", "georgia")}
    current = {"quick_eval_centers": ["ningbo", "georgia"]}
    assert resume_contract_mismatches(saved, current) == []

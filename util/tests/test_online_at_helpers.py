"""CPU-only tests for online-AT orchestration helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from ecg_adv_gen.training.online_at import resolve_quick_eval_plan


def test_resolve_quick_eval_plan_uses_target_real_val_without_heldout_selection():
    plan = resolve_quick_eval_plan(
        quick_eval_source="target_real_val",
        center_name="ningbo",
        quick_eval_centers=["ningbo", "georgia"],
        output_dir=Path("/tmp/run"),
        quick_eval_n_per_center=1000,
        target_val_available=True,
        target_val_n=100,
    )

    assert plan.source == "target_real_val"
    assert plan.metric_view == "target_k500_internal_val"
    assert plan.uses_heldout_selection is False
    assert plan.cache_path is None
    assert plan.centers == ("ningbo",)
    assert plan.message == "quick_eval_source=target_real_val; n=100 (K500-internal validation, no PN2021 held-out selection)"


def test_resolve_quick_eval_plan_rejects_missing_target_real_val_split():
    with pytest.raises(ValueError, match="target_real_npz"):
        resolve_quick_eval_plan(
            quick_eval_source="target_real_val",
            center_name="ningbo",
            quick_eval_centers=["ningbo"],
            output_dir=Path("/tmp/run"),
            quick_eval_n_per_center=1000,
            target_val_available=False,
            target_val_n=0,
        )


def test_resolve_quick_eval_plan_preserves_pn2021_cache_path_for_exploration():
    plan = resolve_quick_eval_plan(
        quick_eval_source="pn2021",
        center_name="ningbo",
        quick_eval_centers=["ningbo", "georgia"],
        output_dir=Path("/tmp/run"),
        quick_eval_n_per_center=64,
        target_val_available=False,
        target_val_n=0,
    )

    assert plan.source == "pn2021"
    assert plan.metric_view == "pn2021_refexcluded_quick_subset"
    assert plan.uses_heldout_selection is True
    assert plan.cache_path == Path("/tmp/run/quick_eval_subset_n64.cache")
    assert plan.centers == ("ningbo", "georgia")

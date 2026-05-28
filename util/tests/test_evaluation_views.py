"""CPU-only tests for canonical evaluation view semantics."""

from __future__ import annotations

from ecg_adv_gen.evaluation import (
    PN2021_ALL_ZERO_KEPT_REFEXCLUDED,
    PN2021_DROP_ALL_ZERO_REFEXCLUDED,
    TARGET_ALL_ZERO_KEPT_REFEXCLUDED,
    TARGET_DROP_ALL_ZERO_REFEXCLUDED,
    canonicalize_view,
    normalize_macro_metric,
)
from ecg_adv_gen.reporting import canonicalize_view as reporting_canonicalize_view


def test_target_views_canonicalize_to_pn2021_refexcluded_views():
    assert canonicalize_view(TARGET_ALL_ZERO_KEPT_REFEXCLUDED) == PN2021_ALL_ZERO_KEPT_REFEXCLUDED
    assert canonicalize_view(TARGET_DROP_ALL_ZERO_REFEXCLUDED) == PN2021_DROP_ALL_ZERO_REFEXCLUDED
    assert canonicalize_view(PN2021_ALL_ZERO_KEPT_REFEXCLUDED) == PN2021_ALL_ZERO_KEPT_REFEXCLUDED


def test_macro_metric_aliases_keep_allzero_and_drop_views_separate():
    assert normalize_macro_metric("macro_auroc") == "macro_auroc"
    assert normalize_macro_metric("avg_macro_auprc") == "macro_auprc"
    assert normalize_macro_metric("drop_all_zero_macro_auroc") == "macro_auroc"
    assert normalize_macro_metric("avg_drop_all_zero_macro_auprc") == "macro_auprc"
    assert normalize_macro_metric("per_class_auroc") is None


def test_reporting_reexports_same_canonicalizer():
    assert reporting_canonicalize_view(TARGET_ALL_ZERO_KEPT_REFEXCLUDED) == PN2021_ALL_ZERO_KEPT_REFEXCLUDED

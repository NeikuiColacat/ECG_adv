"""Canonical evaluation views and metric aliases.

The raw artifacts use slightly different view names across legacy EfficientNet
and ECGFounder scripts. This module defines the paper-level canonical views
used by metrics export and paper-table generation.
"""

from __future__ import annotations


PTBXL_FOLD10_SOURCE_FLOOR = "ptbxl_fold10_source_floor"
PN2021_ALL_ZERO_KEPT_REFEXCLUDED = "pn2021_all_zero_kept_refexcluded"
PN2021_DROP_ALL_ZERO_REFEXCLUDED = "pn2021_drop_all_zero_refexcluded"
TARGET_K500_INTERNAL_VAL = "target_k500_internal_val"
TARGET_ALL_ZERO_KEPT_REFEXCLUDED = "target_all_zero_kept_refexcluded"
TARGET_DROP_ALL_ZERO_REFEXCLUDED = "target_drop_all_zero_refexcluded"

VIEW_ALIASES = {
    TARGET_ALL_ZERO_KEPT_REFEXCLUDED: PN2021_ALL_ZERO_KEPT_REFEXCLUDED,
    TARGET_DROP_ALL_ZERO_REFEXCLUDED: PN2021_DROP_ALL_ZERO_REFEXCLUDED,
}

MACRO_METRIC_ALIASES = {
    "macro_auroc": "macro_auroc",
    "macro_auprc": "macro_auprc",
    "avg_macro_auroc": "macro_auroc",
    "avg_macro_auprc": "macro_auprc",
    "avg_drop_all_zero_macro_auroc": "macro_auroc",
    "avg_drop_all_zero_macro_auprc": "macro_auprc",
    "drop_all_zero_macro_auroc": "macro_auroc",
    "drop_all_zero_macro_auprc": "macro_auprc",
}


def canonicalize_view(view: str) -> str:
    return VIEW_ALIASES.get(str(view), str(view))


def normalize_macro_metric(metric: str) -> str | None:
    return MACRO_METRIC_ALIASES.get(str(metric))

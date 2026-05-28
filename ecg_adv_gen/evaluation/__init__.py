"""Evaluation view semantics for paper-safe metric exports."""

from .views import (
    MACRO_METRIC_ALIASES,
    PN2021_ALL_ZERO_KEPT_REFEXCLUDED,
    PN2021_DROP_ALL_ZERO_REFEXCLUDED,
    PTBXL_FOLD10_SOURCE_FLOOR,
    TARGET_ALL_ZERO_KEPT_REFEXCLUDED,
    TARGET_DROP_ALL_ZERO_REFEXCLUDED,
    TARGET_K500_INTERNAL_VAL,
    VIEW_ALIASES,
    canonicalize_view,
    normalize_macro_metric,
)
from .pn2021_metric_views import (
    DROP_ALL_ZERO_POLICY,
    assemble_pn2021_center_metrics,
    assemble_target_refexcluded_views,
    summarize_center_view,
)
from .metrics import MetricRow, compute_macro_metric_dict, compute_macro_metrics
from .selection import (
    ALLOWED_SELECTION_DATA,
    FORBIDDEN_SELECTION_REFERENCES,
    SELECTION_POLICY,
    SelectionPolicyError,
    compute_source_target_selection_score,
    has_forbidden_selection_reference,
    validate_selection_policy,
)
from .target_splits import TargetSplitMode, split_target_train_val_indices

__all__ = [
    "MACRO_METRIC_ALIASES",
    "PN2021_ALL_ZERO_KEPT_REFEXCLUDED",
    "PN2021_DROP_ALL_ZERO_REFEXCLUDED",
    "PTBXL_FOLD10_SOURCE_FLOOR",
    "TARGET_ALL_ZERO_KEPT_REFEXCLUDED",
    "TARGET_DROP_ALL_ZERO_REFEXCLUDED",
    "TARGET_K500_INTERNAL_VAL",
    "VIEW_ALIASES",
    "DROP_ALL_ZERO_POLICY",
    "ALLOWED_SELECTION_DATA",
    "FORBIDDEN_SELECTION_REFERENCES",
    "SELECTION_POLICY",
    "MetricRow",
    "SelectionPolicyError",
    "TargetSplitMode",
    "assemble_pn2021_center_metrics",
    "assemble_target_refexcluded_views",
    "canonicalize_view",
    "compute_macro_metric_dict",
    "compute_macro_metrics",
    "compute_source_target_selection_score",
    "has_forbidden_selection_reference",
    "normalize_macro_metric",
    "summarize_center_view",
    "split_target_train_val_indices",
    "validate_selection_policy",
]

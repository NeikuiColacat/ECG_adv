"""Paper-safe model-selection policy helpers."""

from __future__ import annotations

from typing import Any


SELECTION_POLICY = "k500_internal_val_plus_source_floor"
LAST_CHECKPOINT_SELECTION_POLICY = "last_checkpoint_only"
ALLOWED_SELECTION_DATA = (
    "target_k500_train_split",
    "target_k500_internal_val",
    "ptbxl_source_floor",
)
LAST_CHECKPOINT_ALLOWED_SELECTION_DATA = (
    "target_k500_train_all",
    "ptbxl_source_sanity",
)
FORBIDDEN_SELECTION_REFERENCES = (
    "pn2021_heldout",
    "heldout_target",
    "target_test",
    "target_eval",
    "full_target_distribution",
)


class SelectionPolicyError(ValueError):
    """Raised when model-selection policy would violate the paper protocol."""


def _walk_strings(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_strings(child)
    elif isinstance(value, tuple):
        for child in value:
            yield from _walk_strings(child)
    elif isinstance(value, str):
        yield value


def has_forbidden_selection_reference(value: Any) -> bool:
    """Return true if a config/argv value names held-out target selection data."""
    for text in _walk_strings(value):
        lowered = text.lower()
        if any(token in lowered for token in FORBIDDEN_SELECTION_REFERENCES):
            return True
    return False


def validate_selection_policy(selection: dict[str, Any]) -> dict[str, Any]:
    """Validate the current K500-internal/source-floor selection contract."""
    if not isinstance(selection, dict):
        raise SelectionPolicyError("paper_protocol.selection must be a mapping")
    policy = str(selection.get("policy", ""))
    if policy == LAST_CHECKPOINT_SELECTION_POLICY:
        allowed_data = tuple(str(item) for item in selection.get("allowed_data", []))
        if set(allowed_data) != set(LAST_CHECKPOINT_ALLOWED_SELECTION_DATA):
            raise SelectionPolicyError(
                "selection.allowed_data must be exactly "
                f"{list(LAST_CHECKPOINT_ALLOWED_SELECTION_DATA)!r}, got {list(allowed_data)!r}"
            )
        if selection.get("forbid_k500_validation_split") is not True:
            raise SelectionPolicyError("selection.forbid_k500_validation_split must be true")
        if selection.get("forbid_best_checkpoint_selection") is not True:
            raise SelectionPolicyError("selection.forbid_best_checkpoint_selection must be true")
        if selection.get("forbid_heldout_target_labels") is not True:
            raise SelectionPolicyError("selection.forbid_heldout_target_labels must be true")
        if selection.get("forbid_full_target_distribution_tuning") is not True:
            raise SelectionPolicyError("selection.forbid_full_target_distribution_tuning must be true")
        if has_forbidden_selection_reference(selection):
            raise SelectionPolicyError("selection policy references held-out target selection data")
        return {
            "policy": LAST_CHECKPOINT_SELECTION_POLICY,
            "allowed_data": list(LAST_CHECKPOINT_ALLOWED_SELECTION_DATA),
            "forbid_k500_validation_split": True,
            "forbid_best_checkpoint_selection": True,
            "forbid_heldout_target_labels": True,
            "forbid_full_target_distribution_tuning": True,
        }
    if policy != SELECTION_POLICY:
        raise SelectionPolicyError(
            f"selection.policy={policy!r}, expected {SELECTION_POLICY!r} or {LAST_CHECKPOINT_SELECTION_POLICY!r}"
        )
    allowed_data = tuple(str(item) for item in selection.get("allowed_data", []))
    if set(allowed_data) != set(ALLOWED_SELECTION_DATA):
        raise SelectionPolicyError(
            "selection.allowed_data must be exactly "
            f"{list(ALLOWED_SELECTION_DATA)!r}, got {list(allowed_data)!r}"
        )
    if selection.get("forbid_heldout_target_labels") is not True:
        raise SelectionPolicyError("selection.forbid_heldout_target_labels must be true")
    if selection.get("forbid_full_target_distribution_tuning") is not True:
        raise SelectionPolicyError("selection.forbid_full_target_distribution_tuning must be true")
    if has_forbidden_selection_reference(selection):
        raise SelectionPolicyError("selection policy references held-out target selection data")
    return {
        "policy": SELECTION_POLICY,
        "allowed_data": list(ALLOWED_SELECTION_DATA),
        "forbid_heldout_target_labels": True,
        "forbid_full_target_distribution_tuning": True,
    }

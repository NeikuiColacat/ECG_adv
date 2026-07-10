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


def update_matched_checkpoint_selection(
    *,
    best_metric: float,
    candidate_metric: float,
    source_metric: float,
    source_baseline_metric: float,
    source_max_drop: float,
) -> dict[str, float | bool]:
    threshold = float(source_baseline_metric) - float(source_max_drop)
    floor_passed = float(source_metric) >= threshold
    return {
        "candidate_metric": float(candidate_metric),
        "best_metric_before": float(best_metric),
        "source_metric": float(source_metric),
        "source_baseline_metric": float(source_baseline_metric),
        "source_floor_threshold": threshold,
        "source_floor_passed": floor_passed,
        "selected": floor_passed and float(candidate_metric) > float(best_metric),
    }


def build_matched_training_record(
    *,
    comparison_arm: str,
    source_checkpoint_path: str,
    source_checkpoint_sha256: str,
    split: dict[str, Any],
    selection_metric: str,
    source_floor_max_drop: float,
    epochs: int,
    optimizer_steps_per_epoch: int,
    realized_optimizer_steps: int,
    scheduler_steps: int,
    source_floor_result: dict[str, Any],
) -> dict[str, Any]:
    if comparison_arm not in {"a0", "a5"}:
        raise ValueError(f"invalid matched comparison arm: {comparison_arm!r}")
    method_enabled = comparison_arm == "a5"
    return {
        "contract": "matched_effnet_a0_a5_v1",
        "comparison_arm": comparison_arm,
        "method_components": dict.fromkeys(("vae_lhat", "three_chain_augmix", "jsd"), method_enabled),
        "source_checkpoint": dict(
            stage="ptbxl_source", path=str(source_checkpoint_path), sha256=str(source_checkpoint_sha256)
        ),
        "k500_split": {
            "train_count": len(split["train_record_ids"]),
            "val_count": len(split["val_record_ids"]),
            "train_record_ids_sha256": split["train_record_ids_sha256"],
            "val_record_ids_sha256": split["val_record_ids_sha256"],
            "validation_fraction": float(split["val_fraction"]),
            "seed": int(split["seed"]),
        },
        "selection": dict(
            metric=str(selection_metric), source_floor_metric=str(selection_metric),
            source_floor_max_drop=float(source_floor_max_drop), checkpoint="best_model.pt",
            source_floor_result=source_floor_result,
        ),
        "budget": dict(
            epochs=int(epochs), optimizer_steps_per_epoch=int(optimizer_steps_per_epoch),
            realized_optimizer_steps=int(realized_optimizer_steps), scheduler_steps=int(scheduler_steps),
        ),
    }


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
    validation_fraction = float(selection.get("validation_fraction", 0.0))
    if not 0.0 < validation_fraction < 0.5:
        raise SelectionPolicyError("selection.validation_fraction must be in (0, 0.5)")
    seed = selection.get("seed")
    if type(seed) is not int:
        raise SelectionPolicyError("selection.seed must be an integer")
    metric = str(selection.get("metric") or "")
    if metric not in {"macro_auroc", "macro_auprc"}:
        raise SelectionPolicyError("selection.metric must be macro_auroc or macro_auprc")
    source_floor = selection.get("source_floor")
    if not isinstance(source_floor, dict) or str(source_floor.get("metric") or "") != metric:
        raise SelectionPolicyError("selection.source_floor.metric must match selection.metric")
    max_drop = float(source_floor.get("max_drop", -1.0))
    if max_drop < 0.0:
        raise SelectionPolicyError("selection.source_floor.max_drop must be non-negative")
    if selection.get("checkpoint") != "best_model.pt":
        raise SelectionPolicyError("selection.checkpoint must be best_model.pt")
    if has_forbidden_selection_reference(selection):
        raise SelectionPolicyError("selection policy references held-out target selection data")
    return {
        "policy": SELECTION_POLICY,
        "allowed_data": list(ALLOWED_SELECTION_DATA),
        "validation_fraction": validation_fraction,
        "seed": seed,
        "metric": metric,
        "source_floor": {"metric": metric, "max_drop": max_drop},
        "checkpoint": "best_model.pt",
        "forbid_heldout_target_labels": True,
        "forbid_full_target_distribution_tuning": True,
    }

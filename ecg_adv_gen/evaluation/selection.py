"""Paper-safe model-selection policy helpers."""

from __future__ import annotations

from typing import Any, Mapping


SELECTION_POLICY = "k500_internal_val_plus_source_floor"
ALLOWED_SELECTION_DATA = (
    "target_k500_train_split",
    "target_k500_internal_val",
    "ptbxl_source_floor",
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
    if policy != SELECTION_POLICY:
        raise SelectionPolicyError(f"selection.policy={policy!r}, expected {SELECTION_POLICY!r}")
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


def validate_ecgfounder_lhat_runtime_selection(
    *,
    selection_source: str,
    selection_metric: str,
    target_real_val_fraction: float,
    target_real_val_seed: int,
    allow_pn2021_heldout_selection: bool = False,
) -> dict[str, Any]:
    """Return a runtime selection-safety record for ECGFounder VAE-LHAT.

    The legacy runner may still expose held-out PN2021 selection for explicit
    exploratory diagnostics, but paper-safe launches must select checkpoints
    from the K-shot target internal validation split plus a PTB-XL source floor.
    """

    source = str(selection_source)
    metric = str(selection_metric)
    if source == "pn2021_heldout" and not allow_pn2021_heldout_selection:
        raise SelectionPolicyError(
            "selection_source='pn2021_heldout' uses held-out target labels; "
            "use --selection_source target_real_val for paper-safe selection "
            "or pass --allow_pn2021_heldout_selection for explicit diagnostics"
        )
    if source != "target_real_val" and source != "pn2021_heldout":
        raise SelectionPolicyError(f"unknown selection_source={source!r}")
    if source == "target_real_val":
        fraction = float(target_real_val_fraction)
        if not 0.0 < fraction < 1.0:
            raise SelectionPolicyError(
                f"target_real_val_fraction={fraction!r} must be between 0 and 1"
            )
    if metric.startswith("target_") and source == "pn2021_heldout" and not allow_pn2021_heldout_selection:
        raise SelectionPolicyError(f"selection_metric={metric!r} requires paper-safe target validation data")
    return {
        "selection_policy": SELECTION_POLICY,
        "selection_source": source,
        "selection_metric": metric,
        "target_real_val_fraction": float(target_real_val_fraction),
        "target_real_val_seed": int(target_real_val_seed),
        "selection_safety": {
            "heldout_target_labels_used_for_selection": source == "pn2021_heldout",
            "full_target_distribution_used_for_tuning": False,
            "forbidden_reference_found": has_forbidden_selection_reference(source),
            "allow_pn2021_heldout_selection": bool(allow_pn2021_heldout_selection),
        },
    }


def compute_source_target_selection_score(
    *,
    selection_metric: str,
    source_metrics: dict[str, Any],
    target_val_metrics: dict[str, Any] | None,
    target_val_score_weight: float = 0.5,
) -> float:
    """Compute the active full-FT source/target-internal-val selection score."""
    if selection_metric == "source_auprc":
        return float(source_metrics["macro_auprc"])
    if target_val_metrics is None:
        raise SelectionPolicyError(
            f"selection_metric={selection_metric!r} requires target validation metrics"
        )
    if selection_metric == "target_val_auprc":
        return float(target_val_metrics["macro_auprc"])
    if selection_metric == "source_plus_target_val_auprc":
        w = float(target_val_score_weight)
        return (
            (1.0 - w) * float(source_metrics["macro_auprc"])
            + w * float(target_val_metrics["macro_auprc"])
        )
    raise SelectionPolicyError(f"unknown selection_metric={selection_metric!r}")


def compute_ecgfounder_lhat_selection_score(
    *,
    selection_metric: str,
    target_metrics: Mapping[str, Any],
    source_metrics: Mapping[str, Any],
    source_selection_weight: float = 0.25,
    source_auprc_floor: float = 0.0,
    source_floor_penalty: float = 10.0,
) -> float:
    """Compute ECGFounder VAE-LHAT selection scores used by the 20260523 runner."""
    target_auroc = float(target_metrics["macro_auroc"])
    target_auprc = float(target_metrics["macro_auprc"])
    source_auroc = float(source_metrics["macro_auroc"])
    source_auprc = float(source_metrics["macro_auprc"])
    if selection_metric == "target_auroc":
        return target_auroc
    if selection_metric == "target_auprc":
        return target_auprc
    if selection_metric == "target_plus_source_auroc":
        return target_auroc + float(source_selection_weight) * source_auroc
    if selection_metric == "target_plus_source_auprc":
        return target_auprc + float(source_selection_weight) * source_auprc
    if selection_metric == "target_source_hmean_auroc":
        denom = target_auroc + source_auroc
        return (2.0 * target_auroc * source_auroc / denom) if denom > 0 else -float("inf")
    if selection_metric == "target_source_hmean_auprc":
        denom = target_auprc + source_auprc
        return (2.0 * target_auprc * source_auprc / denom) if denom > 0 else -float("inf")
    if selection_metric == "target_under_source_floor":
        floor = float(source_auprc_floor)
        if source_auprc < floor:
            return target_auprc - float(source_floor_penalty) * (floor - source_auprc)
        return target_auprc
    raise SelectionPolicyError(f"unknown selection_metric={selection_metric!r}")

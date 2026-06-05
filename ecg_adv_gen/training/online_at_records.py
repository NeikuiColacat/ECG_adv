"""Small CPU-safe record builders for online AT runs."""

from __future__ import annotations

from typing import Any


def build_checkpoint_selection_record(
    *,
    center: str,
    best_epoch: int,
    metric_name: str,
    metric_value: float,
    selection_source: str,
    heldout_target_labels_used: bool,
) -> dict[str, Any]:
    """Build a paper-safety record for online-AT checkpoint selection."""

    paper_safe = selection_source == "target_real_val" and not heldout_target_labels_used
    return {
        "center": str(center),
        "best_epoch": int(best_epoch),
        "metric_name": str(metric_name),
        "metric_value": float(metric_value),
        "selection_source": str(selection_source),
        "heldout_target_labels_used_for_selection": bool(heldout_target_labels_used),
        "paper_safe_selection": bool(paper_safe),
    }

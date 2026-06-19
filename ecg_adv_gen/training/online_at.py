"""Pure orchestration helpers for legacy online-AT runners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class QuickEvalPlan:
    """Resolved quick-eval branch metadata.

    This object intentionally contains no arrays or DataLoaders. Legacy scripts
    use it to keep paper-safety semantics visible before building data.
    """

    source: str
    metric_view: str
    centers: tuple[str, ...]
    uses_heldout_selection: bool
    cache_path: Path | None
    message: str


def resolve_quick_eval_plan(
    *,
    quick_eval_source: str,
    center_name: str,
    quick_eval_centers: Sequence[str],
    output_dir: str | Path,
    quick_eval_n_per_center: int,
    target_val_available: bool,
    target_val_n: int,
) -> QuickEvalPlan:
    """Resolve the online-AT quick-eval branch without touching datasets."""

    source = str(quick_eval_source)
    if source == "none":
        return QuickEvalPlan(
            source=source,
            metric_view="disabled",
            centers=(),
            uses_heldout_selection=False,
            cache_path=None,
            message="quick_eval_source=none; checkpoint selection eval disabled",
        )
    if source == "target_real_val":
        if not target_val_available:
            raise ValueError("--quick_eval_source target_real_val requires --target_real_npz")
        return QuickEvalPlan(
            source=source,
            metric_view="target_k500_internal_val",
            centers=(str(center_name),),
            uses_heldout_selection=False,
            cache_path=None,
            message=(
                f"quick_eval_source=target_real_val; n={int(target_val_n)} "
                "(K500-internal validation, no PN2021 held-out selection)"
            ),
        )
    if source == "pn2021":
        return QuickEvalPlan(
            source=source,
            metric_view="pn2021_refexcluded_quick_subset",
            centers=tuple(str(center) for center in quick_eval_centers),
            uses_heldout_selection=True,
            cache_path=Path(output_dir) / f"quick_eval_subset_n{int(quick_eval_n_per_center)}.cache",
            message=(
                "quick_eval_source=pn2021; using ref-excluded PN2021 quick subset "
                "for exploration"
            ),
        )
    raise ValueError(f"unknown quick_eval_source={source!r}")

"""ECGFounder filesystem and run-layout contracts.

These helpers intentionally contain only deterministic path and naming rules.
They do not import ECGFounder, PyTorch, or legacy scripts, so they are safe for
CPU-only config validation and manifest generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal


ECGFOUNDER_PREPROCESS_POLICIES = frozenset({"official_ptbxl_eval", "filtered_dataset"})
ECGFOUNDER_FEATURE_DATASETS = ("ptbxl", "pn2021")


def _require_preprocess_policy(preprocess_policy: str) -> None:
    if preprocess_policy not in ECGFOUNDER_PREPROCESS_POLICIES:
        allowed = ", ".join(sorted(ECGFOUNDER_PREPROCESS_POLICIES))
        raise ValueError(f"Unsupported ECGFounder preprocess policy {preprocess_policy!r}; expected one of {allowed}")


def ecgfounder_linear_probe_head_path(linear_probe_dir: str | Path) -> Path:
    return Path(linear_probe_dir) / "best_head.pt"


def ecgfounder_feature_cache_path(
    linear_probe_dir: str | Path,
    *,
    dataset: Literal["ptbxl", "pn2021"],
    preprocess_policy: str,
) -> Path:
    if dataset not in ECGFOUNDER_FEATURE_DATASETS:
        allowed = ", ".join(ECGFOUNDER_FEATURE_DATASETS)
        raise ValueError(f"Unsupported ECGFounder feature dataset {dataset!r}; expected one of {allowed}")
    _require_preprocess_policy(preprocess_policy)
    return Path(linear_probe_dir) / f"{dataset}_ecgfounder_features_{preprocess_policy}.npz"


def ecgfounder_feature_cache_paths(
    linear_probe_dir: str | Path,
    *,
    preprocess_policy: str,
) -> dict[str, Path]:
    return {
        dataset: ecgfounder_feature_cache_path(
            linear_probe_dir,
            dataset=dataset,  # type: ignore[arg-type]
            preprocess_policy=preprocess_policy,
        )
        for dataset in ECGFOUNDER_FEATURE_DATASETS
    }


def ecgfounder_kshot_head_run_dir(
    out_dir: str | Path,
    *,
    center: str,
    k: int,
    source_k: int,
    epochs: int,
    seed: int,
) -> Path:
    return Path(out_dir) / "runs" / f"{center}_K{k}_fromK{source_k}_headft_ep{epochs}_seed{seed}"


def ecgfounder_lhat_run_dir(
    out_dir: str | Path,
    *,
    center: str,
    k: int,
    hull_m: int,
    hull_lambda: str | float,
    epochs: int,
    seed: int,
) -> Path:
    lambda_tag = str(hull_lambda).replace(".", "p")
    return Path(out_dir) / "runs" / f"{center}_K{k}_M{hull_m}_lam{lambda_tag}_ep{epochs}_seed{seed}"


def ecgfounder_k500_head_path(
    run_root: str | Path,
    *,
    center: str,
    k: int,
    seed: int,
    source_k: int | None = None,
) -> Path:
    """Return the expected direct-K head checkpoint path.

    Historical ECGFounder K500 head runs encode both target K and source pool K
    in the run directory. If exactly one matching run exists under ``run_root``,
    use it. Otherwise fall back to the canonical ep50/seed path, which keeps
    manifests deterministic even before the artifact exists.
    """
    run_root = Path(run_root)
    source_k = k if source_k is None else source_k
    matches = sorted(run_root.glob(f"{center}_K{k}_fromK{source_k}_headft_*/best_head.pt"))
    if len(matches) == 1:
        return matches[0]
    return run_root / f"{center}_K{k}_fromK{source_k}_headft_ep50_seed{seed}" / "best_head.pt"

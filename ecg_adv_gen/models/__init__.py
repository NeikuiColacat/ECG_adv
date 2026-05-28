"""Model-specific contracts and path helpers."""

from .ecgfounder import (
    ECGFOUNDER_FEATURE_DATASETS,
    ECGFOUNDER_PREPROCESS_POLICIES,
    ecgfounder_feature_cache_path,
    ecgfounder_feature_cache_paths,
    ecgfounder_k500_head_path,
    ecgfounder_kshot_head_run_dir,
    ecgfounder_linear_probe_head_path,
    ecgfounder_lhat_run_dir,
)

__all__ = [
    "ECGFOUNDER_FEATURE_DATASETS",
    "ECGFOUNDER_PREPROCESS_POLICIES",
    "ecgfounder_feature_cache_path",
    "ecgfounder_feature_cache_paths",
    "ecgfounder_k500_head_path",
    "ecgfounder_kshot_head_run_dir",
    "ecgfounder_linear_probe_head_path",
    "ecgfounder_lhat_run_dir",
]

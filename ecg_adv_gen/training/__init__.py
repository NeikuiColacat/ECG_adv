"""Training helpers shared by legacy ECG experiment entrypoints."""

from .losses import (
    attack_success_stats,
    compute_pos_weight,
    fullft_adv_batch_diagnostics,
    masked_bce_per_sample,
    masked_bce_with_logits,
    merge_attack_success_stats,
    multilabel_bce_per_sample,
    pairwise_rank_loss,
    stream_weighted_masked_bce,
    summarize_fullft_adv_epoch_diagnostics,
)
from ecg_adv_gen.run_naming import (
    build_ecgfounder_fullft_method_tag,
    build_ecgfounder_fullft_run_leaf,
    build_ecgfounder_fullft_selection_tag,
    tag_value,
)
from .signal_streams import (
    CachedSignalDataset,
    MemorySignalDataset,
    TaggedCachedSignalDataset,
    TaggedMemorySignalDataset,
    build_weighted_signal_stream_loader,
)
from .splits import random_split_indices
from .stream_sampling import (
    WeightedFeatureStream,
    build_weighted_feature_stream_loader,
    multilabel_stream_sample_weights,
)
from .torch_utils import set_module_requires_grad

__all__ = [
    "attack_success_stats",
    "CachedSignalDataset",
    "compute_pos_weight",
    "fullft_adv_batch_diagnostics",
    "build_weighted_signal_stream_loader",
    "build_ecgfounder_fullft_method_tag",
    "build_ecgfounder_fullft_run_leaf",
    "build_ecgfounder_fullft_selection_tag",
    "masked_bce_per_sample",
    "masked_bce_with_logits",
    "MemorySignalDataset",
    "merge_attack_success_stats",
    "multilabel_bce_per_sample",
    "TaggedCachedSignalDataset",
    "TaggedMemorySignalDataset",
    "WeightedFeatureStream",
    "build_weighted_feature_stream_loader",
    "multilabel_stream_sample_weights",
    "pairwise_rank_loss",
    "random_split_indices",
    "set_module_requires_grad",
    "stream_weighted_masked_bce",
    "summarize_fullft_adv_epoch_diagnostics",
    "tag_value",
]

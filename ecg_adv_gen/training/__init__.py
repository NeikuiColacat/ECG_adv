"""Training helpers shared by managed ECG experiment runners."""

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
    TaggedSignalDataset,
    build_weighted_signal_stream_loader,
    build_weighted_signal_stream_loader_from_datasets,
)
from .splits import random_split_indices
from .stream_sampling import (
    WeightedFeatureStream,
    build_weighted_feature_stream_loader,
    multilabel_stream_sample_weights,
)
from .online_at import QuickEvalPlan, resolve_quick_eval_plan
from .online_at_records import build_checkpoint_selection_record
from .resume_contract import (
    resume_contract_mismatches,
    should_save_initial_best_model,
    validate_resume_contract,
)
from .checkpoint_state import (
    append_jsonl,
    atomic_torch_save,
    capture_rng_state,
    quality_buffer_state,
    resolve_resume_path,
    restore_quality_buffer_state,
    restore_rng_state,
)
from .torch_utils import set_module_requires_grad
from .effnet_super5 import PTBXLDatasetScheme, SynthNPZDataset, evaluate, init_weights

__all__ = [
    "append_jsonl",
    "attack_success_stats",
    "atomic_torch_save",
    "CachedSignalDataset",
    "compute_pos_weight",
    "capture_rng_state",
    "PTBXLDatasetScheme",
    "evaluate",
    "fullft_adv_batch_diagnostics",
    "build_weighted_signal_stream_loader_from_datasets",
    "build_weighted_signal_stream_loader",
    "build_ecgfounder_fullft_method_tag",
    "build_ecgfounder_fullft_run_leaf",
    "build_ecgfounder_fullft_selection_tag",
    "masked_bce_per_sample",
    "masked_bce_with_logits",
    "MemorySignalDataset",
    "merge_attack_success_stats",
    "multilabel_bce_per_sample",
    "SynthNPZDataset",
    "TaggedCachedSignalDataset",
    "TaggedMemorySignalDataset",
    "TaggedSignalDataset",
    "WeightedFeatureStream",
    "build_weighted_feature_stream_loader",
    "build_checkpoint_selection_record",
    "multilabel_stream_sample_weights",
    "pairwise_rank_loss",
    "quality_buffer_state",
    "QuickEvalPlan",
    "random_split_indices",
    "resume_contract_mismatches",
    "resolve_resume_path",
    "restore_quality_buffer_state",
    "restore_rng_state",
    "resolve_quick_eval_plan",
    "set_module_requires_grad",
    "init_weights",
    "should_save_initial_best_model",
    "stream_weighted_masked_bce",
    "summarize_fullft_adv_epoch_diagnostics",
    "tag_value",
    "validate_resume_contract",
]

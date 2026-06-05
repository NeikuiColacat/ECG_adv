"""Command audit adapters for YAML-managed legacy entrypoints."""

from .direct_finetune import build_direct_finetune_argv
from .direct import audit_direct_finetune_command
from .pn2021_eval import build_pn2021_eval_argv
from .pn2021c_eval import build_pn2021c_eval_argv
from .prompt_token_online_at import build_prompt_token_online_at_argv
from .source_training import audit_train_ptbxl_command

__all__ = [
    "audit_direct_finetune_command",
    "audit_train_ptbxl_command",
    "build_direct_finetune_argv",
    "build_pn2021_eval_argv",
    "build_pn2021c_eval_argv",
    "build_prompt_token_online_at_argv",
]

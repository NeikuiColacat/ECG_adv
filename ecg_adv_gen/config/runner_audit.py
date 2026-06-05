"""Command audit dispatcher for YAML-managed runner commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


DISPATCHED_RUNNER_AUDIT_SCRIPT_NAMES = frozenset(
    {
        "eval_crosscenter.py",
        "eval_pn2021_corruptions.py",
        "run_direct_finetune_k500_20260516.py",
        "run_benchmark_direct_finetune_v7_20260530.py",
        "synth_online_at_super5.py",
    }
)


def script_name(command: Mapping[str, Any]) -> str:
    argv = [str(x) for x in command.get("argv", [])]
    if len(argv) < 2:
        return ""
    return Path(argv[1]).name


def audit_runner_command(command: Mapping[str, Any], *, config: Mapping[str, Any]) -> dict[str, list[str]]:
    """Dispatch script-specific protocol audits away from the config loader."""

    name = script_name(command)
    if name == "eval_crosscenter.py":
        from ecg_adv_gen.config.adapters.pn2021_eval import audit_pn2021_eval_command

        return audit_pn2021_eval_command(command, config=config)
    if name == "eval_pn2021_corruptions.py":
        from ecg_adv_gen.config.adapters.pn2021c_eval import audit_pn2021c_eval_command

        return audit_pn2021c_eval_command(command, config=config)
    if name in {"run_direct_finetune_k500_20260516.py", "run_benchmark_direct_finetune_v7_20260530.py"}:
        from ecg_adv_gen.config.adapters.direct_finetune import audit_direct_finetune_command

        return audit_direct_finetune_command(command, config=config)
    if name == "synth_online_at_super5.py":
        from ecg_adv_gen.config.adapters.prompt_token_online_at import audit_prompt_token_online_at_command

        return audit_prompt_token_online_at_command(command, config=config)
    return {"errors": [], "warnings": [f"no dedicated audit for {name or '<empty>'}"]}

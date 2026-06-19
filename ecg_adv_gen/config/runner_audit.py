"""Command audit dispatcher for YAML-managed runner commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


DISPATCHED_RUNNER_AUDIT_SCRIPT_NAMES = frozenset(
    {
        "eval_crosscenter.py",
        "eval_ecgfounder_pn2021_corruptions.py",
        "eval_pn2021_corruptions.py",
        "run_direct_finetune_k500_20260516.py",
        "run_benchmark_direct_finetune_v7_20260530.py",
        "run_effnet_latent_augmix_stage3_20260524.py",
        "run_ecgfounder_fullft_super5_pilot_20260523.py",
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
    runner_adapter = str(((config.get("runner") or {}).get("adapter")) or "")
    if name == "run_effnet_latent_augmix_stage3_20260524.py" and runner_adapter == "benchmark_vae_lhat":
        from ecg_adv_gen.config.adapters.benchmark_vae_lhat import audit_benchmark_vae_lhat_command

        return audit_benchmark_vae_lhat_command(command, config=config)
    if name == "run_effnet_latent_augmix_stage3_20260524.py":
        from ecg_adv_gen.config.adapters.effnet_vae_lhat import audit_effnet_vae_lhat_command

        kshot = config["paper_protocol"]["kshot"]
        return audit_effnet_vae_lhat_command(
            command,
            expected_k=int(kshot["k"]),
            expected_seed=int(kshot.get("subset_seed", kshot["seed"])),
            target_centers=set(config["paper_protocol"]["centers"]["target_4"]),
        )
    if name == "eval_crosscenter.py":
        from ecg_adv_gen.config.adapters.pn2021_eval import audit_pn2021_eval_command

        return audit_pn2021_eval_command(command, config=config)
    if name == "eval_pn2021_corruptions.py":
        from ecg_adv_gen.config.adapters.pn2021c_eval import audit_pn2021c_eval_command

        return audit_pn2021c_eval_command(command, config=config)
    if name == "eval_ecgfounder_pn2021_corruptions.py":
        from ecg_adv_gen.config.adapters.ecgfounder_pn2021c_eval import audit_ecgfounder_pn2021c_eval_command

        return audit_ecgfounder_pn2021c_eval_command(command, config=config)
    if name in {"run_direct_finetune_k500_20260516.py", "run_benchmark_direct_finetune_v7_20260530.py"}:
        from ecg_adv_gen.config.adapters.direct_finetune import audit_direct_finetune_command

        return audit_direct_finetune_command(command, config=config)
    if name == "run_ecgfounder_fullft_super5_pilot_20260523.py":
        from ecg_adv_gen.config.adapters.ecgfounder_fullft import audit_ecgfounder_fullft_command

        return audit_ecgfounder_fullft_command(command, config=config)
    if name == "synth_online_at_super5.py":
        from ecg_adv_gen.config.adapters.prompt_token_online_at import audit_prompt_token_online_at_command

        return audit_prompt_token_online_at_command(command, config=config)
    return {"errors": [], "warnings": [f"no dedicated audit for {name or '<empty>'}"]}

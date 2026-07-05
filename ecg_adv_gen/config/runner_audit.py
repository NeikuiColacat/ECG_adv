"""Command audit dispatcher for YAML-managed runner commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


DISPATCHED_RUNNER_AUDIT_SCRIPT_NAMES = frozenset(
    {
        "pn2021_clean_eval.py",
        "ecgfounder_pn2021c_eval.py",
        "pn2021c_eval.py",
        "effnet_direct_finetune.py",
        "effnet_vae_lhat_augmix.py",
        "ecgfounder_fullft.py",
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
    if name == "effnet_vae_lhat_augmix.py":
        from ecg_adv_gen.config.adapters.effnet_vae_lhat import audit_effnet_vae_lhat_command

        kshot = config["paper_protocol"]["kshot"]
        return audit_effnet_vae_lhat_command(
            command,
            expected_k=int(kshot["k"]),
            expected_seed=int(kshot.get("subset_seed", kshot["seed"])),
            target_centers=set(config["paper_protocol"]["centers"]["target_4"]),
        )
    if name == "pn2021_clean_eval.py":
        from ecg_adv_gen.config.adapters.pn2021_eval import audit_pn2021_eval_command

        return audit_pn2021_eval_command(command, config=config)
    if name == "pn2021c_eval.py":
        from ecg_adv_gen.config.adapters.pn2021c_eval import audit_pn2021c_eval_command

        return audit_pn2021c_eval_command(command, config=config)
    if name == "ecgfounder_pn2021c_eval.py":
        from ecg_adv_gen.config.adapters.ecgfounder_pn2021c_eval import audit_ecgfounder_pn2021c_eval_command

        return audit_ecgfounder_pn2021c_eval_command(command, config=config)
    if name == "effnet_direct_finetune.py":
        from ecg_adv_gen.config.adapters.direct_finetune import audit_direct_finetune_command

        return audit_direct_finetune_command(command, config=config)
    if name == "ecgfounder_fullft.py":
        from ecg_adv_gen.config.adapters.ecgfounder_fullft import audit_ecgfounder_fullft_command

        return audit_ecgfounder_fullft_command(command, config=config)
    return {"errors": [], "warnings": [f"no dedicated audit for {name or '<empty>'}"]}

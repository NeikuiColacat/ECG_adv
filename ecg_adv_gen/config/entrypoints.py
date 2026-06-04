"""Managed legacy entrypoint registry for YAML launch adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManagedScriptProfile:
    """Static metadata for a legacy script managed through YAML launches."""

    script_name: str
    relative_path: str
    wrapper_root: str
    family: str


_PROFILES: tuple[ManagedScriptProfile, ...] = (
    ManagedScriptProfile(
        script_name="eval_crosscenter.py",
        relative_path="scripts/triple_labels/eval_crosscenter.py",
        wrapper_root="scripts/triple_labels",
        family="pn2021_refexcluded_eval",
    ),
    ManagedScriptProfile(
        script_name="eval_pn2021_corruptions.py",
        relative_path="scripts/triple_labels/eval_pn2021_corruptions.py",
        wrapper_root="scripts/triple_labels",
        family="pn2021_corruption_eval",
    ),
    ManagedScriptProfile(
        script_name="export_percent_kshot_v7_sjr_rgq_20260530.py",
        relative_path="scripts/paper/export_percent_kshot_v7_sjr_rgq_20260530.py",
        wrapper_root="scripts/paper",
        family="kshot_subset_export",
    ),
    ManagedScriptProfile(
        script_name="gate_prompt_token_synth.py",
        relative_path="scripts/ecgtwin_gen/gate_prompt_token_synth.py",
        wrapper_root="scripts/ecgtwin_gen",
        family="ecgtwin_prompt_token_gate",
    ),
    ManagedScriptProfile(
        script_name="generate_center_prompt_token_synth.py",
        relative_path="scripts/ecgtwin_gen/generate_center_prompt_token_synth.py",
        wrapper_root="scripts/ecgtwin_gen",
        family="ecgtwin_prompt_token_generate",
    ),
    ManagedScriptProfile(
        script_name="merge_gated_prompt_token_pools.py",
        relative_path="scripts/ecgtwin_gen/merge_gated_prompt_token_pools.py",
        wrapper_root="scripts/ecgtwin_gen",
        family="ecgtwin_prompt_token_pool_merge",
    ),
    ManagedScriptProfile(
        script_name="run_benchmark_direct_finetune_v7_20260530.py",
        relative_path="scripts/paper/run_benchmark_direct_finetune_v7_20260530.py",
        wrapper_root="scripts/paper",
        family="benchmark_direct_target_finetune",
    ),
    ManagedScriptProfile(
        script_name="run_direct_finetune_k500_20260516.py",
        relative_path="scripts/paper/run_direct_finetune_k500_20260516.py",
        wrapper_root="scripts/paper",
        family="effnet_direct_target_finetune",
    ),
    ManagedScriptProfile(
        script_name="run_ecgfounder_fullft_super5_pilot_20260523.py",
        relative_path="scripts/paper/run_ecgfounder_fullft_super5_pilot_20260523.py",
        wrapper_root="scripts/paper",
        family="ecgfounder_full_finetune",
    ),
    ManagedScriptProfile(
        script_name="run_ecgfounder_kshot_head_ft_20260517.py",
        relative_path="scripts/paper/run_ecgfounder_kshot_head_ft_20260517.py",
        wrapper_root="scripts/paper",
        family="ecgfounder_direct_head_finetune",
    ),
    ManagedScriptProfile(
        script_name="run_ecgfounder_vae_only_lhat_head_ft_20260523.py",
        relative_path="scripts/paper/run_ecgfounder_vae_only_lhat_head_ft_20260523.py",
        wrapper_root="scripts/paper",
        family="ecgfounder_vae_lhat",
    ),
    ManagedScriptProfile(
        script_name="run_effnet_latent_augmix_stage3_20260524.py",
        relative_path="scripts/paper/run_effnet_latent_augmix_stage3_20260524.py",
        wrapper_root="scripts/paper",
        family="effnet_vae_lhat",
    ),
    ManagedScriptProfile(
        script_name="select_quality_prompt_token_pool.py",
        relative_path="scripts/ecgtwin_gen/select_quality_prompt_token_pool.py",
        wrapper_root="scripts/ecgtwin_gen",
        family="ecgtwin_prompt_token_pool_select",
    ),
    ManagedScriptProfile(
        script_name="synth_online_at_super5.py",
        relative_path="scripts/pgd_cross_center/synth_online_at_super5.py",
        wrapper_root="scripts/pgd_cross_center",
        family="synthetic_online_adversarial_training",
    ),
    ManagedScriptProfile(
        script_name="train_center_prompt_tokens.py",
        relative_path="scripts/ecgtwin_gen/train_center_prompt_tokens.py",
        wrapper_root="scripts/ecgtwin_gen",
        family="ecgtwin_prompt_token_train",
    ),
    ManagedScriptProfile(
        script_name="train_dit_repro.py",
        relative_path="scripts/ecgtwin_author_repro/train_dit_repro.py",
        wrapper_root="scripts/ecgtwin_author_repro",
        family="ecgtwin_author_dit_repro",
    ),
    ManagedScriptProfile(
        script_name="train_ibe_repro.py",
        relative_path="scripts/ecgtwin_author_repro/train_ibe_repro.py",
        wrapper_root="scripts/ecgtwin_author_repro",
        family="ecgtwin_author_ibe_repro",
    ),
    ManagedScriptProfile(
        script_name="train_ptbxl.py",
        relative_path="scripts/triple_labels/train_ptbxl.py",
        wrapper_root="scripts/triple_labels",
        family="ptbxl_source_training",
    ),
)

_PROFILE_BY_NAME = {profile.script_name: profile for profile in _PROFILES}


def managed_runner_profiles() -> tuple[ManagedScriptProfile, ...]:
    """Return all legacy scripts accepted by the managed YAML launcher."""

    return _PROFILES


def managed_runner_script_names() -> frozenset[str]:
    """Return the managed runner allowlist as script basenames."""

    return frozenset(_PROFILE_BY_NAME)


def managed_script_profile(script_name_or_path: str | Path) -> ManagedScriptProfile:
    """Return the profile for a managed legacy script basename or path."""

    script_name = Path(script_name_or_path).name
    try:
        return _PROFILE_BY_NAME[script_name]
    except KeyError as exc:
        raise KeyError(f"Unmanaged runner script: {script_name}") from exc

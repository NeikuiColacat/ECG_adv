"""Managed entrypoint registry for YAML launch adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManagedScriptProfile:
    """Static metadata for a script managed through YAML launches."""

    script_name: str
    relative_path: str
    runner_root: str
    family: str


_PROFILES: tuple[ManagedScriptProfile, ...] = (
    ManagedScriptProfile(
        script_name="pn2021_clean_eval.py",
        relative_path="ecg_adv_gen/runner/pn2021_clean_eval.py",
        runner_root="ecg_adv_gen/runner",
        family="pn2021_refexcluded_eval",
    ),
    ManagedScriptProfile(
        script_name="ecgfounder_pn2021c_eval.py",
        relative_path="ecg_adv_gen/runner/ecgfounder_pn2021c_eval.py",
        runner_root="ecg_adv_gen/runner",
        family="ecgfounder_pn2021_corruption_eval",
    ),
    ManagedScriptProfile(
        script_name="pn2021c_eval.py",
        relative_path="ecg_adv_gen/runner/pn2021c_eval.py",
        runner_root="ecg_adv_gen/runner",
        family="pn2021_corruption_eval",
    ),
    ManagedScriptProfile(
        script_name="effnet_direct_finetune.py",
        relative_path="ecg_adv_gen/runner/effnet_direct_finetune.py",
        runner_root="ecg_adv_gen/runner",
        family="effnet_direct_target_finetune",
    ),
    ManagedScriptProfile(
        script_name="ecgfounder_fullft.py",
        relative_path="ecg_adv_gen/runner/ecgfounder_fullft.py",
        runner_root="ecg_adv_gen/runner",
        family="ecgfounder_full_finetune",
    ),
    ManagedScriptProfile(
        script_name="effnet_vae_lhat_augmix.py",
        relative_path="ecg_adv_gen/runner/effnet_vae_lhat_augmix.py",
        runner_root="ecg_adv_gen/runner",
        family="effnet_vae_lhat",
    ),
)

_PROFILE_BY_NAME = {profile.script_name: profile for profile in _PROFILES}


def managed_runner_profiles() -> tuple[ManagedScriptProfile, ...]:
    """Return all package runners accepted by the managed YAML launcher."""

    return _PROFILES


def managed_runner_script_names() -> frozenset[str]:
    """Return the managed runner allowlist as script basenames."""

    return frozenset(_PROFILE_BY_NAME)


def managed_script_profile(script_name_or_path: str | Path) -> ManagedScriptProfile:
    """Return the profile for a managed runner basename or path."""

    script_name = Path(script_name_or_path).name
    try:
        return _PROFILE_BY_NAME[script_name]
    except KeyError as exc:
        raise KeyError(f"Unmanaged runner script: {script_name}") from exc

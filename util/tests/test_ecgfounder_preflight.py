"""Focused CPU-only tests for ECGFounder startup validation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ecg_adv_gen.config import build_runner_commands, load_experiment_config, validate_experiment_config
from ecg_adv_gen.config.adapters.ecgfounder_fullft import audit_ecgfounder_fullft_command
from ecg_adv_gen.runner import ecgfounder_fullft


REPO = Path(__file__).resolve().parents[2]
LOCAL_EXAMPLE = REPO / "configs" / "local" / "linbinhao_server.example.yaml"
TRACKED_CONFIG = REPO / "configs" / "experiments" / "ecgfounder_vae_lhat_augmix_threechain_locked_k500.yaml"


class _ReachedPostValidation(Exception):
    """Sentinel proving main() reached the first post-validation call."""


def _locked_k500_argv(*extra: str) -> list[str]:
    return [
        "ecgfounder_fullft.py",
        "--stage",
        "k500",
        "--center",
        "ningbo",
        "--ref_meta_json",
        "/tmp/unused.json",
        "--enable_vae_adv_stream",
        "--enable_latent_augmix_branch",
        "--latent_augmix_chain_base_mode",
        "all_clean_plus_vae_adv",
        "--adv_weight",
        "20",
        "--latent_augmix_width",
        "3",
        "--latent_augmix_copies",
        "1",
        *extra,
    ]


def _stop_after_validation(_seed: int) -> None:
    raise _ReachedPostValidation


def _tracked_locked_command() -> tuple[dict, dict]:
    config = load_experiment_config(
        TRACKED_CONFIG,
        LOCAL_EXAMPLE,
        runtime_context={"run_id": "pytest_ecgfounder_cross_layer"},
    )
    validate_experiment_config(config, repo_root=REPO)
    command = build_runner_commands(config)[0]
    assert command["matrix"]["center"] == "ningbo"
    return config, {**command, "argv": list(command["argv"])}


def _set_option(argv: list[str], flag: str, value: str) -> None:
    if flag in argv:
        argv[argv.index(flag) + 1] = value
    else:
        argv.extend([flag, value])


def test_zero_weight_default_jsd_with_one_copy_reaches_post_validation(monkeypatch):
    monkeypatch.setattr(sys, "argv", _locked_k500_argv())
    monkeypatch.setattr(ecgfounder_fullft, "set_seed", _stop_after_validation)

    with pytest.raises(_ReachedPostValidation):
        ecgfounder_fullft.main()


@pytest.mark.parametrize(
    "weight_flag",
    ["--latent_augmix_consistency_weight", "--latent_augmix_bce_weight"],
)
def test_enabled_jsd_with_one_copy_still_fails_preflight(monkeypatch, weight_flag):
    monkeypatch.setattr(
        sys,
        "argv",
        _locked_k500_argv(weight_flag, "1.0"),
    )
    monkeypatch.setattr(ecgfounder_fullft, "set_seed", _stop_after_validation)

    with pytest.raises(ValueError, match="jsd requires --latent_augmix_copies >= 2"):
        ecgfounder_fullft.main()


def test_tracked_zero_weight_jsd_copies_one_reaches_runner_post_validation(monkeypatch):
    _, command = _tracked_locked_command()
    monkeypatch.setattr(sys, "argv", [str(value) for value in command["argv"][1:]])
    monkeypatch.setattr(ecgfounder_fullft, "set_seed", _stop_after_validation)

    with pytest.raises(_ReachedPostValidation):
        ecgfounder_fullft.main()


@pytest.mark.parametrize(
    "weight_flag",
    ["--latent_augmix_consistency_weight", "--latent_augmix_bce_weight"],
)
def test_tracked_positive_jsd_weight_with_one_copy_fails_adapter_and_runner(
    monkeypatch,
    weight_flag,
):
    config, command = _tracked_locked_command()
    _set_option(command["argv"], weight_flag, "1.0")

    audit = audit_ecgfounder_fullft_command(command, config=config)
    assert any(
        "jsd requires --latent_augmix_copies >= 2" in error
        for error in audit["errors"]
    ), audit

    monkeypatch.setattr(sys, "argv", [str(value) for value in command["argv"][1:]])
    monkeypatch.setattr(ecgfounder_fullft, "set_seed", _stop_after_validation)
    with pytest.raises(ValueError, match="jsd requires --latent_augmix_copies >= 2"):
        ecgfounder_fullft.main()


@pytest.mark.parametrize(
    "flag",
    [
        "--latent_augmix_copies",
        "--latent_augmix_consistency_weight",
        "--latent_augmix_bce_weight",
    ],
)
def test_tracked_adapter_reports_malformed_latent_augmix_numbers(flag):
    config, command = _tracked_locked_command()
    _set_option(command["argv"], flag, "not-a-number")

    audit = audit_ecgfounder_fullft_command(command, config=config)

    assert any(f"{flag} has invalid numeric value" in error for error in audit["errors"]), audit

"""Focused CPU-only tests for ECGFounder startup validation."""

from __future__ import annotations

import sys

import pytest

from ecg_adv_gen.runner import ecgfounder_fullft


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


def test_zero_weight_default_jsd_with_one_copy_reaches_post_validation(monkeypatch):
    monkeypatch.setattr(sys, "argv", _locked_k500_argv())
    monkeypatch.setattr(ecgfounder_fullft, "set_seed", _stop_after_validation)

    with pytest.raises(_ReachedPostValidation):
        ecgfounder_fullft.main()


def test_enabled_jsd_with_one_copy_still_fails_preflight(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _locked_k500_argv("--latent_augmix_consistency_weight", "1.0"),
    )
    monkeypatch.setattr(ecgfounder_fullft, "set_seed", _stop_after_validation)

    with pytest.raises(ValueError, match="jsd requires --latent_augmix_copies >= 2"):
        ecgfounder_fullft.main()

import pytest

from ecg_adv_gen.training.resume_contract import (
    resume_contract_mismatches,
    validate_resume_contract,
)


def test_resume_contract_rejects_legacy_free_pgd_checkpoint():
    saved = {"attack_mode": "pgd", "center_name": "ningbo"}
    current = {"center_name": "ningbo"}

    with pytest.raises(ValueError, match="attack_mode"):
        validate_resume_contract(saved, current, allow_drift=False)

    assert resume_contract_mismatches(saved, current) == [
        {"key": "attack_mode", "saved": "pgd", "current": "latent_hull"}
    ]


def test_resume_contract_accepts_previous_latent_hull_checkpoint():
    saved = {"attack_mode": "latent_hull", "center_name": "ningbo"}
    current = {"center_name": "ningbo"}

    assert validate_resume_contract(saved, current, allow_drift=False) == []


def test_resume_contract_rejects_disabled_legacy_latent_augmix_checkpoint():
    saved = {
        "attack_mode": "latent_hull",
        "center_name": "ningbo",
        "enable_latent_augmix_branch": False,
        "enable_latent_augmix_consistency": False,
        "latent_augmix_topology": "legacy_optional",
    }
    current = {"center_name": "ningbo"}

    with pytest.raises(ValueError, match="enable_latent_augmix_branch"):
        validate_resume_contract(saved, current, allow_drift=False)

    assert resume_contract_mismatches(saved, current) == [
        {"key": "enable_latent_augmix_branch", "saved": False, "current": True},
        {"key": "enable_latent_augmix_consistency", "saved": False, "current": True},
        {"key": "latent_augmix_topology", "saved": "legacy_optional", "current": "locked_three_chain"},
    ]


def test_resume_contract_allows_default_removed_latent_augmix_knobs():
    saved = {
        "attack_mode": "latent_hull",
        "center_name": "ningbo",
        "latent_augmix_mixture_mode": "beta",
        "latent_augmix_mixture_prob": 0.5,
        "latent_augmix_mixture_beta_a": 0.0,
        "latent_augmix_mixture_beta_b": 0.0,
        "latent_augmix_op_schedule": "random",
        "latent_augmix_chain_weights": "",
        "latent_augmix_signal_space": "model_zscore",
        "latent_augmix_corruption_source": "vae_decode",
        "latent_augmix_severity_params_file": "",
        "latent_augmix_severity_params_name": "",
        "no_latent_augmix_renorm": False,
        "latent_augmix_clip_abs": 6.0,
    }
    current = {"center_name": "ningbo"}

    assert validate_resume_contract(saved, current, allow_drift=False) == []


def test_resume_contract_rejects_nondefault_removed_latent_augmix_knobs():
    saved = {
        "attack_mode": "latent_hull",
        "center_name": "ningbo",
        "latent_augmix_mixture_mode": "fixed",
        "latent_augmix_signal_space": "raw_pre_zscore",
        "no_latent_augmix_renorm": True,
    }
    current = {"center_name": "ningbo"}

    with pytest.raises(ValueError, match="latent_augmix_mixture_mode"):
        validate_resume_contract(saved, current, allow_drift=False)

    assert resume_contract_mismatches(saved, current) == [
        {"key": "latent_augmix_mixture_mode", "saved": "fixed", "current": "beta"},
        {"key": "latent_augmix_signal_space", "saved": "raw_pre_zscore", "current": "model_zscore"},
        {"key": "no_latent_augmix_renorm", "saved": True, "current": False},
    ]

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

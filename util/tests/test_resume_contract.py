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

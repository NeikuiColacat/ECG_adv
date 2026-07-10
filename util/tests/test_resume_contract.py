import pytest

from ecg_adv_gen.training.resume_contract import (
    RESUME_CONTRACT_KEYS,
    resume_contract_mismatches,
    validate_resume_contract,
)


def test_resume_contract_rejects_legacy_free_pgd_checkpoint():
    saved = {
        "attack_mode": "pgd",
        "center_name": "ningbo",
        "latent_augmix_signal_space": "raw_pre_zscore",
    }
    current = {
        "center_name": "ningbo",
        "latent_augmix_signal_space": "raw_pre_zscore",
    }

    with pytest.raises(ValueError, match="attack_mode"):
        validate_resume_contract(saved, current, allow_drift=False)

    assert resume_contract_mismatches(saved, current) == [
        {"key": "attack_mode", "saved": "pgd", "current": "latent_hull"}
    ]


def test_resume_contract_accepts_raw_first_latent_hull_checkpoint():
    saved = {
        "attack_mode": "latent_hull",
        "center_name": "ningbo",
        "latent_augmix_signal_space": "raw_pre_zscore",
    }
    current = {
        "center_name": "ningbo",
        "latent_augmix_signal_space": "raw_pre_zscore",
    }

    assert validate_resume_contract(saved, current, allow_drift=False) == []


def test_resume_contract_rejects_checkpoint_without_signal_space_contract():
    saved = {"attack_mode": "latent_hull", "center_name": "ningbo"}
    current = {
        "center_name": "ningbo",
        "latent_augmix_signal_space": "raw_pre_zscore",
    }

    with pytest.raises(ValueError, match="latent_augmix_signal_space"):
        validate_resume_contract(saved, current, allow_drift=False)


def test_resume_contract_accepts_missing_signal_space_for_normalized_ablation():
    saved = {"attack_mode": "latent_hull", "center_name": "ningbo"}
    current = {
        "center_name": "ningbo",
        "latent_augmix_signal_space": "model_zscore",
    }

    assert validate_resume_contract(saved, current, allow_drift=False) == []


def test_resume_contract_drops_dead_latent_weight_cap():
    assert "latent_augmix_latent_weight_cap" not in RESUME_CONTRACT_KEYS


def test_resume_contract_preserves_normalized_signal_space_ablation():
    saved = {
        "attack_mode": "latent_hull",
        "center_name": "ningbo",
        "latent_augmix_signal_space": "model_zscore",
    }
    current = {
        "center_name": "ningbo",
        "latent_augmix_signal_space": "model_zscore",
    }

    assert validate_resume_contract(saved, current, allow_drift=False) == []


def test_resume_contract_accepts_matching_explicit_chain_weights():
    weights = "0.475,0.475,0.05"
    saved = {
        "attack_mode": "latent_hull",
        "center_name": "ningbo",
        "latent_augmix_chain_weights": weights,
        "latent_augmix_signal_space": "raw_pre_zscore",
    }
    current = {
        "center_name": "ningbo",
        "latent_augmix_chain_weights": weights,
        "latent_augmix_signal_space": "raw_pre_zscore",
    }

    assert validate_resume_contract(saved, current, allow_drift=False) == []


def test_resume_contract_rejects_different_explicit_chain_weights():
    saved = {
        "attack_mode": "latent_hull",
        "center_name": "ningbo",
        "latent_augmix_chain_weights": "0.475,0.475,0.05",
        "latent_augmix_signal_space": "raw_pre_zscore",
    }
    current = {
        "center_name": "ningbo",
        "latent_augmix_chain_weights": "0.45,0.45,0.10",
        "latent_augmix_signal_space": "raw_pre_zscore",
    }

    with pytest.raises(ValueError, match="latent_augmix_chain_weights"):
        validate_resume_contract(saved, current, allow_drift=False)

    assert resume_contract_mismatches(saved, current) == [
        {
            "key": "latent_augmix_chain_weights",
            "saved": "0.475,0.475,0.05",
            "current": "0.45,0.45,0.10",
        }
    ]


def test_resume_contract_rejects_disabled_legacy_latent_augmix_checkpoint():
    saved = {
        "attack_mode": "latent_hull",
        "center_name": "ningbo",
        "enable_latent_augmix_branch": False,
        "enable_latent_augmix_consistency": False,
        "latent_augmix_topology": "legacy_optional",
        "latent_augmix_signal_space": "raw_pre_zscore",
    }
    current = {
        "center_name": "ningbo",
        "latent_augmix_signal_space": "raw_pre_zscore",
    }

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
        "latent_augmix_signal_space": "raw_pre_zscore",
        "latent_augmix_corruption_source": "vae_decode",
        "latent_augmix_severity_params_file": "",
        "latent_augmix_severity_params_name": "",
        "no_latent_augmix_renorm": False,
        "latent_augmix_clip_abs": 6.0,
    }
    current = {
        "center_name": "ningbo",
        "latent_augmix_signal_space": "raw_pre_zscore",
    }

    assert validate_resume_contract(saved, current, allow_drift=False) == []


def test_resume_contract_rejects_nondefault_removed_latent_augmix_knobs():
    saved = {
        "attack_mode": "latent_hull",
        "center_name": "ningbo",
        "latent_augmix_mixture_mode": "fixed",
        "latent_augmix_signal_space": "model_zscore",
        "no_latent_augmix_renorm": True,
    }
    current = {
        "center_name": "ningbo",
        "latent_augmix_signal_space": "raw_pre_zscore",
    }

    with pytest.raises(ValueError, match="latent_augmix_mixture_mode"):
        validate_resume_contract(saved, current, allow_drift=False)

    assert resume_contract_mismatches(saved, current) == [
        {"key": "latent_augmix_mixture_mode", "saved": "fixed", "current": "beta"},
        {"key": "no_latent_augmix_renorm", "saved": True, "current": False},
        {"key": "latent_augmix_signal_space", "saved": "model_zscore", "current": "raw_pre_zscore"},
    ]


@pytest.mark.parametrize(
    ("key", "saved_value", "current_value"),
    [
        ("init_ckpt", "/tmp/old.pt", "/tmp/new.pt"),
        ("final_checkpoint_only", False, True),
        ("hull_include_anchor", True, False),
        ("hull_init_logit_gap", 4.0, 0.0),
        ("pgd_eps", 2.0, 3.25),
        ("asr_low_threshold", 0.3, 0.2),
        ("asr_high_threshold", 0.7, 0.8),
        ("enable_latent_augmix_consistency", True, False),
    ],
)
def test_resume_contract_rejects_operational_protocol_field_drift(
    key: str,
    saved_value,
    current_value,
):
    saved = {
        "attack_mode": "latent_hull",
        "center_name": "ningbo",
        "latent_augmix_signal_space": "raw_pre_zscore",
        key: saved_value,
    }
    current = {
        "center_name": "ningbo",
        "latent_augmix_signal_space": "raw_pre_zscore",
        key: current_value,
    }

    with pytest.raises(ValueError, match=key):
        validate_resume_contract(saved, current, allow_drift=False)

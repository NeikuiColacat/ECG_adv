import pytest

from scripts.triple_labels.eval_pn2021_corruptions import (
    STRESS_PROFILE_CHOICES,
    _build_corruption_op,
)


def test_calibrated_10to20pp_profile_builds_verified_strong_ops():
    assert "calibrated_10to20pp" in STRESS_PROFILE_CHOICES

    powerline = _build_corruption_op("powerline_noise", 5, "calibrated_10to20pp")
    emg = _build_corruption_op("emg_noise", 5, "calibrated_10to20pp")
    wander = _build_corruption_op("baseline_wander", 5, "calibrated_10to20pp")
    shift = _build_corruption_op("baseline_shift", 5, "calibrated_10to20pp")
    mask = _build_corruption_op("random_leads_masking", 5, "calibrated_10to20pp")

    assert powerline.max_amplitude == 8.0
    assert powerline.freq == 100
    assert powerline.dependency is False
    assert emg.max_amplitude == 2.3
    assert emg.dependency is False
    assert wander.max_amplitude == 2.5
    assert wander.k == 6
    assert wander.max_freq == 0.8
    assert wander.freq == 100
    assert wander.dependency is False
    assert shift.max_amplitude == 2.4
    assert shift.shift_ratio == 0.9
    assert shift.num_segment == 6
    assert shift.freq == 100
    assert shift.dependency is False
    assert mask.mask_leads_prob == 0.57
    assert mask.mask_leads_selection == "random"


def test_calibrated_10to20pp_profile_rejects_unverified_severities():
    with pytest.raises(ValueError, match="only for severities"):
        _build_corruption_op("emg_noise", 4, "calibrated_10to20pp")

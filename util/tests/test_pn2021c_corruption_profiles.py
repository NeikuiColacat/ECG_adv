import pytest
import numpy as np
import torch

from scripts.triple_labels.eval_pn2021_corruptions import (
    PN2021IndexedCenterDataset,
    STRESS_PROFILE_CHOICES,
    StreamingCorruptedPN2021Dataset,
    apply_effnet_input_stabilizer,
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


def test_effnet_input_stabilizer_is_noop_by_default():
    x = torch.randn(12, 250)
    y = apply_effnet_input_stabilizer(x, {})
    assert y is x


def test_effnet_clean_dataset_applies_stabilizer_and_renorms():
    t = np.linspace(0.0, 1.0, 250, dtype=np.float32)
    signal = np.stack([(lead + 1.0) * t + 10.0 for lead in range(12)], axis=1)
    signals = signal[None, :, :]
    labels = np.zeros((1, 5), dtype=np.float32)
    ds = PN2021IndexedCenterDataset(
        signals,
        labels,
        indices=np.array([0]),
        crop_len=250,
        input_stabilizer_config={"renorm_after_stabilizer": True},
    )

    x, y = ds[0]

    assert tuple(x.shape) == (12, 250)
    assert tuple(y.shape) == (5,)
    assert abs(float(x.mean())) < 1e-5
    assert abs(float(x.reshape(-1).std() - 1.0)) < 1e-5


def test_effnet_clean_dataset_can_stabilize_before_center_crop():
    t = np.linspace(0.0, 1.0, 1000, dtype=np.float32)
    signal = np.stack([(lead + 1.0) * t + 10.0 for lead in range(12)], axis=1)
    signals = signal[None, :, :]
    labels = np.zeros((1, 5), dtype=np.float32)
    ds = PN2021IndexedCenterDataset(
        signals,
        labels,
        indices=np.array([0]),
        crop_len=250,
        input_stabilizer_config={
            "renorm_after_stabilizer": True,
            "stage": "pre_crop",
        },
    )

    x, _ = ds[0]

    assert tuple(x.shape) == (12, 250)
    assert torch.isfinite(x).all()


def test_effnet_streaming_corrupted_dataset_applies_stabilizer_to_corrupted_view():
    t = np.linspace(0.0, 1.0, 250, dtype=np.float32)
    signal = np.stack([np.sin(2 * np.pi * (lead + 1) * t) for lead in range(12)], axis=1)
    signals = signal[None, :, :]
    labels = np.ones((1, 5), dtype=np.float32)
    ds = StreamingCorruptedPN2021Dataset(
        signals,
        labels,
        corruption="powerline_noise",
        public_severity=5,
        seed=123,
        crop_len=250,
        severity_profile="calibrated_10to20pp",
        indices=np.array([0]),
        input_stabilizer_config={"renorm_after_stabilizer": True},
    )

    x, y = ds[0]

    assert tuple(x.shape) == (12, 250)
    assert tuple(y.shape) == (5,)
    assert abs(float(x.mean())) < 1e-5
    assert abs(float(x.reshape(-1).std() - 1.0)) < 1e-5

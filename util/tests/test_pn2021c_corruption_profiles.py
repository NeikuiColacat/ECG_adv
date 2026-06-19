import pytest
import numpy as np
import torch
from argparse import Namespace

from scripts.triple_labels.eval_pn2021_corruptions import (
    PN2021IndexedCenterDataset,
    NativeRawFirstCorruptedPN2021Dataset,
    STRESS_PROFILE_CHOICES,
    StreamingCorruptedPN2021Dataset,
    apply_effnet_input_stabilizer,
    _build_corruption_op,
)
import scripts.triple_labels.eval_pn2021_corruptions as pn2021c_eval


def test_custom_profile_loader_canonicalizes_severity_keys(tmp_path):
    profile_path = tmp_path / "profiles.yaml"
    profile_path.write_text(
        """
profiles:
  emg_amp2p3:
    emg_noise:
      "5":
        max_amplitude: 2.3
        min_amplitude: 0.0
        p: 1.0
        dependency: false
""",
        encoding="utf-8",
    )

    profile = pn2021c_eval._load_custom_severity_profile(
        str(profile_path),
        "emg_amp2p3",
    )

    assert profile == {
        "emg_noise": {
            5: {
                "max_amplitude": 2.3,
                "min_amplitude": 0.0,
                "p": 1.0,
                "dependency": False,
            }
        }
    }


def test_custom_profile_builds_ops_from_external_params():
    profile = {
        "emg_noise": {
            5: {
                "max_amplitude": 2.3,
                "min_amplitude": 0.0,
                "p": 1.0,
                "dependency": False,
            }
        }
    }

    op = _build_corruption_op(
        "emg_noise",
        5,
        "custom",
        severity_profile_params=profile,
    )

    assert op.max_amplitude == 2.3
    assert op.min_amplitude == 0.0
    assert op.p == 1.0
    assert op.dependency is False


def test_custom_profile_builds_random_mask_with_max_masked_leads():
    profile = {
        "random_leads_masking": {
            5: {
                "max_masked_leads": 2,
                "mask_leads_selection": "random",
                "p": 1.0,
            }
        }
    }

    op = _build_corruption_op(
        "random_leads_masking",
        5,
        "custom",
        severity_profile_params=profile,
    )

    assert op.max_masked_leads == 2
    assert op.mask_leads_selection == "random"
    assert op.p == 1.0


def test_custom_profile_requires_stream_mode(tmp_path):
    profile_path = tmp_path / "profiles.yaml"
    profile_path.write_text(
        """
profiles:
  emg_amp2p3:
    emg_noise:
      5:
        max_amplitude: 2.3
""",
        encoding="utf-8",
    )
    args = Namespace(
        severity_profile="custom",
        severity_params_file=str(profile_path),
        severity_params_name="emg_amp2p3",
        mode="cache",
    )

    with pytest.raises(ValueError, match="requires --mode stream"):
        pn2021c_eval._resolve_severity_profile_args(args)


def test_non_custom_profile_rejects_external_param_file(tmp_path):
    args = Namespace(
        severity_profile="calibrated_10to20pp",
        severity_params_file=str(tmp_path / "profiles.yaml"),
        severity_params_name="emg_amp2p3",
        mode="stream",
    )

    with pytest.raises(ValueError, match="only valid with --severity_profile custom"):
        pn2021c_eval._resolve_severity_profile_args(args)


def test_streaming_dataset_passes_custom_profile_params_to_builder(monkeypatch):
    seen = []

    class RecordingOp:
        def __call__(self, sample):
            return sample

    def fake_build(corruption, public_severity, severity_profile, *, sample_rate_hz=None, severity_profile_params=None):
        seen.append((corruption, public_severity, severity_profile, severity_profile_params))
        return RecordingOp()

    monkeypatch.setattr(pn2021c_eval, "_build_corruption_op", fake_build)
    signals = np.zeros((1, 250, 12), dtype=np.float32)
    labels = np.ones((1, 5), dtype=np.float32)
    profile = {"emg_noise": {5: {"max_amplitude": 2.3}}}
    ds = StreamingCorruptedPN2021Dataset(
        signals,
        labels,
        corruption="emg_noise",
        public_severity=5,
        seed=123,
        crop_len=250,
        severity_profile="custom",
        severity_profile_params=profile,
        indices=np.array([0]),
    )

    ds[0]

    assert seen == [("emg_noise", 5, "custom", profile)]


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


def test_corruption_op_can_use_native_sample_rate():
    powerline = _build_corruption_op(
        "powerline_noise",
        5,
        "calibrated_10to20pp",
        sample_rate_hz=500,
    )
    wander = _build_corruption_op(
        "baseline_wander",
        5,
        "calibrated_10to20pp",
        sample_rate_hz=500,
    )

    assert powerline.freq == 500
    assert wander.freq == 500


def test_native_raw_first_corrupts_before_resample_and_model_zscore(monkeypatch):
    calls = []

    class RecordingOp:
        def __call__(self, sample):
            calls.append(("shape", tuple(sample.shape)))
            return sample + 1.0

    def fake_build(corruption, public_severity, severity_profile, *, sample_rate_hz=None, severity_profile_params=None):
        calls.append(("fs", sample_rate_hz))
        return RecordingOp()

    monkeypatch.setattr(pn2021c_eval, "_build_corruption_op", fake_build)
    t = np.linspace(0.0, 1.0, 5000, dtype=np.float32)
    signal = np.stack([np.sin(2 * np.pi * (lead + 1) * t) for lead in range(12)], axis=1)
    labels = np.ones((1, 5), dtype=np.float32)
    ds = NativeRawFirstCorruptedPN2021Dataset(
        [signal],
        labels,
        sample_rates=np.array([500.0], dtype=np.float32),
        corruption="powerline_noise",
        public_severity=5,
        seed=123,
        crop_len=1000,
        severity_profile="calibrated_10to20pp",
        indices=np.array([0]),
    )

    x, y = ds[0]

    assert calls == [("fs", 500.0), ("shape", (12, 5000))]
    assert tuple(x.shape) == (12, 1000)
    assert tuple(y.shape) == (5,)
    assert abs(float(x.mean())) < 1e-5
    assert abs(float(x.reshape(-1).std(unbiased=False) - 1.0)) < 1e-5


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

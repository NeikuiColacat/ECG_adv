"""
Severity [1..10] -> DeepECG op params mapping.

Calibrated for PTBXL @ 100Hz after per-sample z-score (signal std ~= 1).
Linear interpolation between severity=1 (mild) and severity=10 (aggressive).

All ops use:
  - freq=100 (PTBXL sample rate, not DeepECG's default 500)
  - p=1.0 (AugMix meta-algorithm samples ops externally; no double-gating)
  - dependency=False (AugMix is sample-level; avoid lead-level coupling surprises)
"""
from typing import Callable

from methods.augmix.ecg_ops import (
    PowerlineNoise,
    EMGNoise,
    BaselineShift,
    BaselineWander,
    RandomLeadsMask,
)


# (name, severity=1 params, severity=10 params) linearly interpolated on each field.
# Severity does not affect fixed params (freq, p, dependency, etc.).
_SEVERITY_TABLE = {
    # PowerlineNoise: 50/60Hz. z-score units -> 0.3 ~ SNR 10dB.
    "powerline_noise": {
        "s1": {"max_amplitude": 0.03, "min_amplitude": 0.0},
        "s10": {"max_amplitude": 0.30, "min_amplitude": 0.0},
    },
    # EMGNoise: white gaussian per lead. More morphology-damaging -> lower cap.
    "emg_noise": {
        "s1": {"max_amplitude": 0.02, "min_amplitude": 0.0},
        "s10": {"max_amplitude": 0.20, "min_amplitude": 0.0},
    },
    # BaselineShift: stepwise DC offset. DeepECG's upstream formula is
    # `amp = ±1 - U[0, max_amp]`, so max_amp barely changes offset magnitude
    # (dominated by the ±1 term). We scale `shift_ratio` (fraction of signal
    # duration affected) to make severity meaningfully monotonic.
    "baseline_shift": {
        "s1": {"max_amplitude": 0.05, "shift_ratio": 0.05},
        "s10": {"max_amplitude": 0.40, "shift_ratio": 0.40},
    },
    # BaselineWander: k=3 low-freq cosines in [0.01, 0.2] Hz.
    "baseline_wander": {
        "s1": {"max_amplitude": 0.05, "min_amplitude": 0.0},
        "s10": {"max_amplitude": 0.40, "min_amplitude": 0.0},
    },
    # RandomLeadsMask: probability each lead is zeroed out.
    "random_leads_masking": {
        "s1": {"mask_leads_prob": 0.05},
        "s10": {"mask_leads_prob": 0.50},
    },
}

# Fixed params per op (not affected by severity).
_FIXED_PARAMS = {
    "powerline_noise": {"freq": 100, "p": 1.0, "dependency": False},
    "emg_noise": {"p": 1.0, "dependency": False},
    "baseline_shift": {
        "freq": 100, "p": 1.0, "dependency": False,
        "num_segment": 1,
    },
    "baseline_wander": {
        "freq": 100, "p": 1.0, "dependency": False,
        "max_freq": 0.2, "min_freq": 0.01, "k": 3,
    },
    "random_leads_masking": {"p": 1.0, "mask_leads_selection": "random"},
}

_OP_CLASSES = {
    "powerline_noise": PowerlineNoise,
    "emg_noise": EMGNoise,
    "baseline_shift": BaselineShift,
    "baseline_wander": BaselineWander,
    "random_leads_masking": RandomLeadsMask,
}


def _lerp(s1_val, s10_val, severity: int) -> float:
    frac = (severity - 1) / 9.0  # severity=1 -> 0, severity=10 -> 1
    return s1_val + frac * (s10_val - s1_val)


def build_op(name: str, severity: int) -> Callable:
    """Build a DeepECG op instance with severity-scaled parameters.

    Args:
        name: one of "powerline_noise", "emg_noise", "baseline_shift",
              "baseline_wander", "random_leads_masking".
        severity: integer in [1, 10].

    Returns:
        Callable op with signature op(sample: Tensor[12, L]) -> Tensor[12, L].
    """
    if name not in _SEVERITY_TABLE:
        raise ValueError(f"unknown op: {name}")
    if not (1 <= severity <= 10):
        raise ValueError(f"severity must be in [1, 10], got {severity}")

    table = _SEVERITY_TABLE[name]
    s1_params, s10_params = table["s1"], table["s10"]

    scaled = {}
    for k in s1_params:
        scaled[k] = _lerp(s1_params[k], s10_params[k], severity)

    kwargs = {**_FIXED_PARAMS[name], **scaled}
    return _OP_CLASSES[name](**kwargs)


AVAILABLE_OPS = list(_SEVERITY_TABLE.keys())

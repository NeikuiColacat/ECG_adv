"""PTB-XL-layout adapters for the fairseq-signals ECG augmentations.

Upstream source
---------------
Repository: https://github.com/Jwoo5/fairseq-signals
File: fairseq_signals/data/ecg/augmentations.py
Pinned revision: f8f0ff1c788a82c2059cb452cd5462898867489e

The five operator equations, parameter names, defaults and random draw order
match the upstream MIT-licensed implementation. This module makes only these
adaptations:

1. NumPy input/output uses this project's ``(time, 12)`` cache layout instead
   of the upstream PyTorch ``(12, time)`` layout.
2. Output is a NumPy array rather than a PyTorch tensor.
3. Conditional lead masking reads ``mask_leads_condition`` rather than trying
   to unpack the string ``mask_leads_selection``.
4. Stochastic calls require an explicit isolated RNG from
   ``util.random_seed``; no module-global mutable stream exists.

Like upstream, outputs are ``float32`` and amplitudes use the same unit/scale
as the input. For this project's raw waveform caches that unit is mV. Severity
mapping and YAML loading deliberately remain outside this module.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from numpy.typing import NDArray

UPSTREAM_COMMIT = "f8f0ff1c788a82c2059cb452cd5462898867489e"
UPSTREAM_SOURCE_URL = (
    "https://github.com/Jwoo5/fairseq-signals/blob/"
    f"{UPSTREAM_COMMIT}/fairseq_signals/data/ecg/augmentations.py"
)

ECGArray = NDArray[np.floating]
RandomLike = Any

def _validate_ecg(signal: ECGArray) -> np.ndarray:
    """Validate one channel-last 12-lead ECG without copying it."""

    array = np.asarray(signal)
    if array.ndim != 2 or array.shape[1] != 12:
        raise ValueError(
            "signal must have shape (time, 12) in PTB-XL lead order; "
            f"got {array.shape}"
        )
    if array.shape[0] < 2:
        raise ValueError("signal must contain at least two time samples")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"signal must use a floating-point dtype; got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError("signal contains NaN or infinity")
    return array


def _validate_common_params(
    *,
    min_amplitude: float,
    max_amplitude: float,
    p: float,
) -> None:
    if float(min_amplitude) < 0.0:
        raise ValueError("min_amplitude must be non-negative")
    if float(max_amplitude) < float(min_amplitude):
        raise ValueError("max_amplitude must be >= min_amplitude")
    if not 0.0 <= float(p) <= 1.0:
        raise ValueError("p must be in [0, 1]")


def _random_source(rng: RandomLike | None) -> RandomLike:
    """Return upstream-compatible NumPy random functions.

    A ``RandomState`` reproduces upstream draws without mutating global state.
    ``Generator`` is also accepted, although it uses NumPy's newer bit
    generator and therefore does not produce the same samples for a numeric
    seed.
    """

    if rng is None:
        raise ValueError(
            "rng is required; create an isolated stream with util.random_seed"
        )
    required = ("uniform", "normal", "choice")
    if not all(callable(getattr(rng, name, None)) for name in required):
        raise TypeError(
            "rng must provide NumPy-compatible uniform, normal and choice methods"
        )
    return rng


def _adjust_channel_dependency(ecg_ct: np.ndarray) -> np.ndarray:
    """Reconstruct III/aVR/aVL/aVF from I and II exactly as upstream."""

    ecg_ct[2] = ecg_ct[1] - ecg_ct[0]
    ecg_ct[3] = -(ecg_ct[1] + ecg_ct[0]) / 2
    ecg_ct[4] = ecg_ct[0] - ecg_ct[1] / 2
    ecg_ct[5] = ecg_ct[1] - ecg_ct[0] / 2
    return ecg_ct


def powerline_noise(
    signal: ECGArray,
    *,
    max_amplitude: float = 0.5,
    min_amplitude: float = 0.0,
    p: float = 1.0,
    freq: float = 500.0,
    dependency: bool = True,
    rng: RandomLike | None = None,
) -> np.ndarray:
    """Apply upstream ``PowerlineNoise`` (random 50/60 Hz cosine)."""

    array = _validate_ecg(signal)
    _validate_common_params(
        min_amplitude=min_amplitude,
        max_amplitude=max_amplitude,
        p=p,
    )
    if float(freq) <= 0.0:
        raise ValueError("freq must be positive")
    random_source = _random_source(rng)
    output_ct = array.T.copy()

    if float(p) > random_source.uniform(0, 1):
        _, time_size = output_ct.shape
        amplitude = random_source.uniform(
            min_amplitude, max_amplitude, size=(1, 1)
        )
        powerline_hz = 50 if random_source.uniform(0, 1) > 0.5 else 60
        time = np.linspace(0, time_size - 1, time_size)
        phase = random_source.uniform(0, 2 * np.pi)
        noise = np.cos(2 * np.pi * powerline_hz * (time / freq) + phase)
        output_ct = output_ct + noise * amplitude
        if dependency:
            output_ct = _adjust_channel_dependency(output_ct)
    return np.ascontiguousarray(output_ct.T, dtype=np.float32)


def emg_noise(
    signal: ECGArray,
    *,
    max_amplitude: float = 0.5,
    min_amplitude: float = 0.0,
    dependency: bool = True,
    p: float = 1.0,
    rng: RandomLike | None = None,
) -> np.ndarray:
    """Apply upstream ``EMGNoise`` (per-lead Gaussian noise)."""

    array = _validate_ecg(signal)
    _validate_common_params(
        min_amplitude=min_amplitude,
        max_amplitude=max_amplitude,
        p=p,
    )
    random_source = _random_source(rng)
    output_ct = array.T.copy()

    if float(p) > random_source.uniform(0, 1):
        channel_size, time_size = output_ct.shape
        amplitude = random_source.uniform(
            min_amplitude, max_amplitude, size=(channel_size, 1)
        )
        noise = random_source.normal(0, 1, [channel_size, time_size])
        output_ct = output_ct + noise * amplitude
        if dependency:
            output_ct = _adjust_channel_dependency(output_ct)
    return np.ascontiguousarray(output_ct.T, dtype=np.float32)


def baseline_shift(
    signal: ECGArray,
    *,
    max_amplitude: float = 0.25,
    min_amplitude: float = 0.0,
    shift_ratio: float = 0.2,
    num_segment: int = 1,
    freq: float = 500.0,
    dependency: bool = False,
    p: float = 1.0,
    amplitude_mode: str = "legacy_upstream",
    rng: RandomLike | None = None,
) -> np.ndarray:
    """Apply upstream ``BaselineShift`` step-segment perturbation.

    ``freq`` is retained for exact upstream API compatibility even though the
    upstream implementation does not use it in this operator.
    """

    array = _validate_ecg(signal)
    _validate_common_params(
        min_amplitude=min_amplitude,
        max_amplitude=max_amplitude,
        p=p,
    )
    if not 0.0 < float(shift_ratio) <= 1.0:
        raise ValueError("shift_ratio must be in (0, 1]")
    if int(num_segment) < 1:
        raise ValueError("num_segment must be >= 1")
    if float(freq) <= 0.0:
        raise ValueError("freq must be positive")
    if amplitude_mode not in {"legacy_upstream", "signed_shared_uniform"}:
        raise ValueError(f"unknown baseline-shift amplitude_mode: {amplitude_mode!r}")
    random_source = _random_source(rng)
    output_ct = array.T.copy()

    if float(p) > random_source.uniform(0, 1):
        channel_size, time_size = output_ct.shape
        shift_length = time_size * float(shift_ratio)
        amp_channel = random_source.choice([1, -1], size=(channel_size, 1))
        amp_general = random_source.uniform(
            min_amplitude, max_amplitude, size=(1, 1)
        )
        if amplitude_mode == "signed_shared_uniform":
            amplitude = amp_channel * amp_general
        else:
            amplitude = amp_channel - amp_general
        noise = np.zeros(shape=(channel_size, time_size))
        for _ in range(int(num_segment)):
            segment_len = random_source.normal(shift_length, shift_length * 0.2)
            start = int(random_source.uniform(0, time_size - segment_len))
            stop = int(start + segment_len)
            noise[:, start:stop] = 1
        output_ct = output_ct + noise * amplitude
        if dependency:
            output_ct = _adjust_channel_dependency(output_ct)
    return np.ascontiguousarray(output_ct.T, dtype=np.float32)


def baseline_wander(
    signal: ECGArray,
    *,
    max_amplitude: float = 0.5,
    min_amplitude: float = 0.0,
    p: float = 1.0,
    max_freq: float = 0.2,
    min_freq: float = 0.01,
    k: int = 3,
    freq: float = 500.0,
    dependency: bool = True,
    rng: RandomLike | None = None,
) -> np.ndarray:
    """Apply upstream ``BaselineWander`` summed-cosine perturbation."""

    array = _validate_ecg(signal)
    _validate_common_params(
        min_amplitude=min_amplitude,
        max_amplitude=max_amplitude,
        p=p,
    )
    if float(min_freq) <= 0.0 or float(max_freq) < float(min_freq):
        raise ValueError("frequency bounds must satisfy 0 < min_freq <= max_freq")
    if int(k) < 1:
        raise ValueError("k must be >= 1")
    if float(freq) <= 0.0:
        raise ValueError("freq must be positive")
    random_source = _random_source(rng)
    output_ct = array.T.copy()

    if float(p) > random_source.uniform(0, 1):
        channel_size, time_size = output_ct.shape
        amp_channel = random_source.normal(1, 0.5, size=(channel_size, 1))
        amp_general = random_source.uniform(
            min_amplitude, max_amplitude, size=int(k)
        )
        noise = np.zeros(shape=(1, time_size))
        time = np.linspace(0, time_size - 1, time_size)
        for component in range(int(k)):
            component_freq = random_source.uniform(min_freq, max_freq)
            phase = random_source.uniform(0, 2 * np.pi)
            noise += (
                np.cos(2 * np.pi * component_freq * (time / freq) + phase)
                * amp_general[component]
            )
        noise = (noise * amp_channel).astype(np.float32)
        output_ct = output_ct + noise
        if dependency:
            output_ct = _adjust_channel_dependency(output_ct)
    return np.ascontiguousarray(output_ct.T, dtype=np.float32)


def random_leads_masking(
    signal: ECGArray,
    *,
    p: float = 1.0,
    mask_leads_selection: str = "random",
    mask_leads_prob: float = 0.5,
    mask_leads_condition: tuple[int, int] | None = None,
    ensure_at_least_one_lead: bool = False,
    rng: RandomLike | None = None,
    python_rng: random.Random | None = None,
) -> np.ndarray:
    """Apply upstream ``RandomLeadsMask`` with its conditional-mode bug fixed.

    In ``conditional`` mode, ``mask_leads_condition=(n_limb, n_chest)`` gives
    the number of leads to mask in the first and second six-lead groups.
    """

    array = _validate_ecg(signal)
    if not 0.0 <= float(p) <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if not 0.0 <= float(mask_leads_prob) <= 1.0:
        raise ValueError("mask_leads_prob must be in [0, 1]")
    if mask_leads_selection not in {"random", "conditional"}:
        raise ValueError(
            "mask_leads_selection must be either 'random' or 'conditional'"
        )
    random_source = _random_source(rng)
    python_random = python_rng
    output_ct = array.T.copy()

    if float(p) >= random_source.uniform(0, 1):
        masked_output = np.zeros_like(output_ct, dtype=np.float32)
        if mask_leads_selection == "random":
            survivors = (
                random_source.uniform(0, 1, size=12) >= float(mask_leads_prob)
            )
            if bool(ensure_at_least_one_lead) and not np.any(survivors):
                survivor = int(random_source.choice(np.arange(12)))
                survivors[survivor] = True
            masked_output[survivors] = output_ct[survivors]
        else:
            if python_random is None:
                raise ValueError(
                    "python_rng is required for conditional lead masking"
                )
            if mask_leads_condition is None:
                raise ValueError(
                    "mask_leads_condition is required for conditional masking"
                )
            n_limb, n_chest = (int(value) for value in mask_leads_condition)
            if not (0 <= n_limb <= 6 and 0 <= n_chest <= 6):
                raise ValueError(
                    "conditional masked-lead counts must both be in [0, 6]"
                )
            limb_survivors = np.asarray(
                python_random.sample(list(np.arange(6)), 6 - n_limb)
            )
            chest_survivors = np.asarray(
                python_random.sample(list(np.arange(6)), 6 - n_chest)
            ) + 6
            masked_output[limb_survivors] = output_ct[limb_survivors]
            masked_output[chest_survivors] = output_ct[chest_survivors]
        output_ct = masked_output
    return np.ascontiguousarray(output_ct.T, dtype=np.float32)

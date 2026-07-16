"""Reference-alignment tests for the fairseq-signals ECG operators."""

import random

import numpy as np
import pytest
import torch

from methods.augmix.ecg_ops import (
    BaselineShift,
    BaselineWander,
    EMGNoise,
    PowerlineNoise,
    RandomLeadsMask,
)
from util.augmentations import (
    baseline_shift,
    baseline_wander,
    emg_noise,
    powerline_noise,
    random_leads_masking,
)


def _signal(dtype=np.float32):
    time = np.linspace(0.0, 10.0, 1000, endpoint=False, dtype=dtype)
    return np.stack(
        [np.sin(2.0 * np.pi * (1.0 + lead / 20.0) * time) for lead in range(12)],
        axis=1,
    ).astype(dtype)


@pytest.mark.parametrize(
    ("function", "reference_class", "kwargs"),
    [
        (
            powerline_noise,
            PowerlineNoise,
            {"max_amplitude": 0.3, "freq": 100, "dependency": True},
        ),
        (
            emg_noise,
            EMGNoise,
            {"max_amplitude": 0.2, "dependency": True},
        ),
        (
            baseline_shift,
            BaselineShift,
            {
                "max_amplitude": 0.4,
                "shift_ratio": 0.3,
                "num_segment": 2,
                "freq": 100,
                "dependency": False,
            },
        ),
        (
            baseline_wander,
            BaselineWander,
            {
                "max_amplitude": 0.4,
                "max_freq": 0.2,
                "min_freq": 0.01,
                "k": 3,
                "freq": 100,
                "dependency": True,
            },
        ),
        (
            random_leads_masking,
            RandomLeadsMask,
            {"mask_leads_prob": 0.4},
        ),
    ],
)
def test_matches_vendored_fairseq_signals_reference(
    function,
    reference_class,
    kwargs,
):
    signal = _signal()
    reference = reference_class(p=1.0, **kwargs)

    np.random.seed(20260715)
    expected = reference(torch.from_numpy(signal.T.copy())).numpy().T
    np.random.seed(20260715)
    actual = function(signal, p=1.0, rng=np.random, **kwargs)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (powerline_noise, {"freq": 100}),
        (emg_noise, {}),
        (baseline_wander, {"freq": 100}),
        (baseline_shift, {"freq": 100}),
        (random_leads_masking, {}),
    ],
)
def test_operator_contract_and_no_in_place_change(function, kwargs):
    signal = _signal(np.float64)
    original = signal.copy()
    output = function(signal, rng=np.random.RandomState(7), **kwargs)

    assert output.shape == signal.shape
    assert output.dtype == np.float32
    assert output.flags.c_contiguous
    assert np.isfinite(output).all()
    assert np.array_equal(signal, original)
    assert not np.shares_memory(output, signal)


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (powerline_noise, {"freq": 100}),
        (emg_noise, {}),
        (baseline_wander, {"freq": 100}),
        (baseline_shift, {"freq": 100}),
        (random_leads_masking, {}),
    ],
)
def test_random_state_makes_operator_reproducible(function, kwargs):
    signal = _signal()
    first = function(signal, rng=np.random.RandomState(123), **kwargs)
    second = function(signal, rng=np.random.RandomState(123), **kwargs)
    assert np.array_equal(first, second)


def test_conditional_masking_uses_condition_counts():
    signal = np.ones((1000, 12), dtype=np.float32)
    output = random_leads_masking(
        signal,
        mask_leads_selection="conditional",
        mask_leads_condition=(2, 3),
        rng=np.random.RandomState(1),
        python_rng=random.Random(1),
    )
    masked_by_group = np.all(output == 0.0, axis=0).reshape(2, 6).sum(axis=1)
    assert masked_by_group.tolist() == [2, 3]


def test_wrong_layout_is_rejected():
    channel_first = np.zeros((12, 1000), dtype=np.float32)
    with pytest.raises(ValueError, match=r"shape \(time, 12\)"):
        emg_noise(channel_first)

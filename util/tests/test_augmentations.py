"""Golden-contract tests for the pinned fairseq-signals ECG operators."""

import random

import numpy as np
import pytest

from util.augmentations import (
    baseline_shift,
    baseline_wander,
    emg_noise,
    powerline_noise,
    random_leads_masking,
)
from util.augmentations.operators import UPSTREAM_COMMIT, UPSTREAM_SOURCE_URL


def _signal(dtype=np.float32):
    time = np.linspace(0.0, 10.0, 1000, endpoint=False, dtype=dtype)
    return np.stack(
        [np.sin(2.0 * np.pi * (1.0 + lead / 20.0) * time) for lead in range(12)],
        axis=1,
    ).astype(dtype)


@pytest.mark.parametrize(
    ("function", "kwargs", "expected_summary"),
    [
        (
            powerline_noise,
            {"max_amplitude": 0.3, "freq": 100, "dependency": True},
            (126.62981554470025, 0.01055248462872502, 0.747640072052713),
        ),
        (
            emg_noise,
            {"max_amplitude": 0.2, "dependency": True},
            (130.7209862279051, 0.01089341551899209, 0.7470555287952432),
        ),
        (
            baseline_shift,
            {
                "max_amplitude": 0.4,
                "shift_ratio": 0.3,
                "num_segment": 2,
                "freq": 100,
                "dependency": False,
            },
            (726.3211853300302, 0.060526765444169184, 0.9136473937863147),
        ),
        (
            baseline_wander,
            {
                "max_amplitude": 0.4,
                "max_freq": 0.2,
                "min_freq": 0.01,
                "k": 3,
                "freq": 100,
                "dependency": True,
            },
            (-333.70971428090706, -0.027809142856742256, 0.7449737181900838),
        ),
        (
            random_leads_masking,
            {"mask_leads_prob": 0.4},
            (127.50714974911433, 0.010625595812426194, 0.6454097615581246),
        ),
    ],
)
def test_matches_pinned_fairseq_signals_golden_contract(
    function,
    kwargs,
    expected_summary,
):
    signal = _signal()
    actual = function(
        signal,
        p=1.0,
        rng=np.random.RandomState(20260715),
        **kwargs,
    )
    summary = (
        float(actual.sum(dtype=np.float64)),
        float(actual.mean(dtype=np.float64)),
        float(actual.std(dtype=np.float64)),
    )
    np.testing.assert_allclose(summary, expected_summary, rtol=0.0, atol=1e-9)


def test_reference_revision_is_explicitly_pinned():
    assert UPSTREAM_COMMIT == "f8f0ff1c788a82c2059cb452cd5462898867489e"
    assert UPSTREAM_COMMIT in UPSTREAM_SOURCE_URL


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

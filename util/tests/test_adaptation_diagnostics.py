from __future__ import annotations

import numpy as np

from ecg_adv_gen.adaptation.diagnostics import (
    agent_attack_decision,
    decoded_signal_invalid_stats,
)


def test_decoded_signal_invalid_stats_handles_valid_nan_and_flatline():
    signals = np.zeros((4, 12, 1000), dtype=np.float32)
    signals[0, :, :] = np.linspace(0.0, 1.0, 1000, dtype=np.float32)
    signals[1, 0, 0] = np.nan
    signals[2, :, :] = 0.0
    signals[3, :, :] = np.linspace(-1.0, 1.0, 1000, dtype=np.float32)

    stats = decoded_signal_invalid_stats(signals)

    assert stats["decoded_invalid_rate"] == 0.5
    assert stats["nan_rate"] == 0.25
    assert stats["flatline_rate"] == 0.5


def test_decoded_signal_invalid_stats_empty_returns_nan():
    stats = decoded_signal_invalid_stats(np.zeros((0, 12, 1000), dtype=np.float32))

    assert np.isnan(stats["decoded_invalid_rate"])
    assert np.isnan(stats["nan_rate"])
    assert np.isnan(stats["flatline_rate"])


def test_agent_attack_decision_states():
    assert agent_attack_decision(
        {},
        asr_low_threshold=0.3,
        consecutive_low_asr=0,
    )["attack_state"] == "no_attack_or_disabled"

    assert agent_attack_decision(
        {"asr_overall": 0.4, "decoded_invalid_rate": 0.06},
        asr_low_threshold=0.3,
        consecutive_low_asr=0,
    )["stop_or_continue"] == "review_before_continue"

    assert agent_attack_decision(
        {"asr_overall": 0.2, "decoded_invalid_rate": 0.0},
        asr_low_threshold=0.3,
        consecutive_low_asr=0,
    )["attack_state"] == "attack_too_weak"

    assert agent_attack_decision(
        {
            "asr_overall": 0.9,
            "decoded_invalid_rate": 0.0,
            "attack_vs_anchor": {"loss_gain_mean": 0.1},
        },
        asr_low_threshold=0.3,
        consecutive_low_asr=0,
    )["attack_state"] == "attack_too_strong"

    assert agent_attack_decision(
        {"asr_overall": 0.5, "decoded_invalid_rate": 0.0},
        asr_low_threshold=0.3,
        consecutive_low_asr=1,
    )["attack_state"] == "watch_low_asr"

    assert agent_attack_decision(
        {"asr_overall": 0.5, "decoded_invalid_rate": 0.0},
        asr_low_threshold=0.3,
        consecutive_low_asr=0,
    )["attack_state"] == "healthy"

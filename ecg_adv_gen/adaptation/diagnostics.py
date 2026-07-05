"""CPU-testable diagnostics for latent-hull online AT runs."""

from __future__ import annotations

from typing import Any

import numpy as np


def _float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def decoded_signal_invalid_stats(signals: np.ndarray) -> dict[str, float]:
    """Return invalid, NaN, and flatline rates for decoded ECG batches."""
    if signals.size == 0:
        return {
            "decoded_invalid_rate": float("nan"),
            "nan_rate": float("nan"),
            "flatline_rate": float("nan"),
        }
    finite = np.isfinite(signals).all(axis=tuple(range(1, signals.ndim)))
    clean = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
    p2p = np.ptp(clean, axis=-1).max(axis=1)
    flatline = p2p < 1e-6
    invalid = (~finite) | flatline
    return {
        "decoded_invalid_rate": float(np.mean(invalid)),
        "nan_rate": float(np.mean(~finite)),
        "flatline_rate": float(np.mean(flatline)),
    }


def agent_attack_decision(
    entry: dict[str, Any],
    *,
    asr_low_threshold: float,
    consecutive_low_asr: int,
) -> dict[str, Any]:
    """Classify attack health from epoch diagnostics."""
    asr = _float_or_nan(entry.get("asr_overall", float("nan")))
    invalid = _float_or_nan(entry.get("decoded_invalid_rate", float("nan")))
    loss_gain = entry.get("attack_vs_anchor", {}).get("loss_gain_mean")
    if asr != asr:
        state = "no_attack_or_disabled"
        action = "continue_if_this_is_an_ablation"
    elif invalid == invalid and invalid > 0.05:
        state = "attack_too_strong_or_decode_invalid"
        action = "lower_hull_lambda_or_attack_strength_before_paper_run"
    elif asr < asr_low_threshold:
        state = "attack_too_weak"
        action = "increase_attack_strength_only_if_repeated_and_source_floor_is_safe"
    elif asr > 0.85 and (loss_gain is None or _float_or_nan(loss_gain) > 0.05):
        state = "attack_too_strong"
        action = "lower_adv_weight_or_attack_strength_if_target/source_metrics_drop"
    elif consecutive_low_asr > 0:
        state = "watch_low_asr"
        action = "continue_but_watch_next_epoch"
    else:
        state = "healthy"
        action = "continue"
    return {
        "attack_state": state,
        "action": action,
        "stop_or_continue": (
            "continue"
            if state not in {"attack_too_strong_or_decode_invalid"}
            else "review_before_continue"
        ),
        "asr_low_threshold": float(asr_low_threshold),
        "consecutive_low_asr": int(consecutive_low_asr),
    }

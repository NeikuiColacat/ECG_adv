"""Tests for util/ecg_viz.py (12-lead ECG debug visualization + sanity checks)."""
from pathlib import Path

import numpy as np
import pytest
import torch

from util.ecg_viz import (
    LEAD_ORDERS,
    plot_comparison,
    plot_ecg,
    plot_with_report,
    sanity_check,
)


def _synth_einthoven_valid(length: int = 1000, sample_rate: float = 100.0, seed: int = 0) -> np.ndarray:
    """生成满足 Einthoven (II=I+III) 且有清晰 QRS 峰的合成 12-lead 样本，ECGTwin lead order。

    lead II 每 1 秒一个窄峰 -> HR=60 bpm。
    """
    rng = np.random.default_rng(seed)
    t = np.arange(length) / sample_rate
    # QRS-like 峰：高斯脉冲每秒一个
    beat_period_s = 1.0
    lead_ii = np.zeros(length, dtype=np.float32)
    for beat_t in np.arange(beat_period_s, length / sample_rate, beat_period_s):
        lead_ii += 1.5 * np.exp(-((t - beat_t) ** 2) / (2 * 0.02 ** 2))
    lead_ii += rng.normal(0, 0.05, length).astype(np.float32)
    # I 随机基础 + 小脉冲，III = II - I
    lead_i = 0.3 * np.sin(2 * np.pi * 0.8 * t).astype(np.float32) + rng.normal(0, 0.03, length).astype(np.float32)
    lead_i += 0.5 * np.exp(-((t - 1.0) ** 2) / (2 * 0.02 ** 2))  # 小 QRS on I 也有
    lead_iii = (lead_ii - lead_i).astype(np.float32)
    # ECGTwin lead order: [I, II, III, aVR, aVF, aVL, V1..V6]
    sig = np.zeros((12, length), dtype=np.float32)
    sig[0] = lead_i
    sig[1] = lead_ii
    sig[2] = lead_iii
    sig[3] = -(lead_i + lead_ii) / 2.0  # aVR
    sig[4] = (lead_ii + lead_iii) / 2.0  # aVF ~ (II+III)/2
    sig[5] = (lead_i - lead_iii) / 2.0  # aVL ~ (I-III)/2
    # V1-V6 随便填噪声（不影响 Einthoven 检查）
    for j in range(6, 12):
        sig[j] = rng.normal(0, 0.2, length).astype(np.float32)
    return sig


# ---------------- plot_ecg ----------------

def test_plot_ecg_channels_first(tmp_path: Path):
    sig = np.random.randn(12, 1000).astype(np.float32) * 0.3
    out = plot_ecg(sig, 100.0, tmp_path / "a.png", lead_order="ecgtwin")
    assert out.exists()
    assert out.stat().st_size > 1024


def test_plot_ecg_time_first_autotranspose(tmp_path: Path):
    sig = np.random.randn(1000, 12).astype(np.float32) * 0.3
    out = plot_ecg(sig, 100.0, tmp_path / "b.png")
    assert out.exists() and out.stat().st_size > 1024


def test_plot_ecg_batched_input(tmp_path: Path):
    sig = torch.randn(1, 12, 1000) * 0.3
    out = plot_ecg(sig, 100.0, tmp_path / "c.png")
    assert out.exists() and out.stat().st_size > 1024


def test_plot_ecg_invalid_sample_rate(tmp_path: Path):
    sig = np.random.randn(12, 500).astype(np.float32)
    with pytest.raises(ValueError):
        plot_ecg(sig, 0.0, tmp_path / "x.png")


def test_plot_ecg_unknown_lead_order(tmp_path: Path):
    sig = np.random.randn(12, 500).astype(np.float32)
    with pytest.raises(ValueError):
        plot_ecg(sig, 100.0, tmp_path / "x.png", lead_order="foo")  # type: ignore


# ---------------- plot_comparison ----------------

def test_plot_comparison_overlay(tmp_path: Path):
    signals = [np.random.randn(12, 1000).astype(np.float32) * 0.3 for _ in range(3)]
    out = plot_comparison(
        signals, ["clean", "aug1", "aug2"], 100.0, tmp_path / "cmp_ov.png", mode="overlay"
    )
    assert out.exists() and out.stat().st_size > 1024


def test_plot_comparison_stacked(tmp_path: Path):
    signals = [np.random.randn(12, 500).astype(np.float32) * 0.3 for _ in range(2)]
    out = plot_comparison(
        signals, ["a", "b"], 100.0, tmp_path / "cmp_st.png", mode="stacked"
    )
    assert out.exists() and out.stat().st_size > 1024


def test_plot_comparison_length_alignment(tmp_path: Path):
    signals = [
        np.random.randn(12, 1000).astype(np.float32) * 0.3,
        np.random.randn(12, 2500).astype(np.float32) * 0.3,  # 更长 -> 会被截断
        np.random.randn(12, 500).astype(np.float32) * 0.3,   # 更短 -> 会被插值
    ]
    out = plot_comparison(signals, ["a", "b", "c"], 100.0, tmp_path / "cmp_align.png")
    assert out.exists()


def test_plot_comparison_label_mismatch(tmp_path: Path):
    signals = [np.random.randn(12, 500).astype(np.float32) for _ in range(3)]
    with pytest.raises(ValueError):
        plot_comparison(signals, ["only-one"], 100.0, tmp_path / "x.png")


# ---------------- sanity_check ----------------

def test_sanity_check_clean_synthetic():
    sig = _synth_einthoven_valid()
    rep = sanity_check(sig, 100.0, lead_order="ecgtwin")
    assert rep["has_nan"] is False
    assert rep["has_inf"] is False
    assert rep["einthoven_residual"] < 0.05
    assert rep["avR_residual"] < 0.10
    assert rep["hr_estimate_bpm"] is not None
    assert 40 < rep["hr_estimate_bpm"] < 120


def test_sanity_check_all_zeros():
    sig = np.zeros((12, 1000), dtype=np.float32)
    rep = sanity_check(sig, 100.0)
    assert len(rep["flatline_leads"]) == 12
    assert rep["hr_estimate_bpm"] is None
    assert any("flatline" in w for w in rep["warnings"])


def test_sanity_check_nan_detected():
    sig = _synth_einthoven_valid()
    sig[0, 100] = np.nan
    rep = sanity_check(sig, 100.0)
    assert rep["has_nan"] is True
    assert any("NaN" in w for w in rep["warnings"])


def test_sanity_check_einthoven_broken():
    sig = _synth_einthoven_valid()
    # 在 II 上加大偏移，破坏 II = I + III
    sig[1] = sig[1] + 2.0
    rep = sanity_check(sig, 100.0, lead_order="ecgtwin")
    assert rep["einthoven_residual"] > 0.2
    assert any("Einthoven" in w for w in rep["warnings"])


# ---------------- plot_with_report ----------------

def test_plot_with_report_returns_tuple(tmp_path: Path):
    sig = _synth_einthoven_valid()
    path, report = plot_with_report(
        sig, 100.0, tmp_path / "rep.png", title_prefix="synth", lead_order="ecgtwin"
    )
    assert path.exists()
    assert isinstance(report, dict)
    assert "warnings" in report
    assert "einthoven_residual" in report
    # synth 样本应 HR 大约 60 bpm，warnings 应很少
    assert report["hr_estimate_bpm"] is not None
    assert len(report["warnings"]) <= 1


def test_lead_orders_have_12_leads():
    for lo, leads in LEAD_ORDERS.items():
        assert len(leads) == 12, f"{lo} has {len(leads)} leads"

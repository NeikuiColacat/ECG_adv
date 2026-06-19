from __future__ import annotations

import numpy as np

from scripts.agent import pn2021c_realism_diagnostics as diag


def test_compute_signal_diagnostics_reports_rms_and_max_diff():
    clean = np.ones((12, 1000), dtype=np.float32)
    corrupt = clean + 0.5

    stats = diag.compute_signal_diagnostics(clean, corrupt, sample_rate_hz=100.0)

    assert stats["rms_clean"] == 1.0
    assert stats["rms_diff"] == 0.5
    assert stats["rms_ratio"] == 0.5
    assert stats["max_abs_diff"] == 0.5


def test_compute_signal_diagnostics_counts_masked_leads():
    t = np.linspace(0.0, 1.0, 1000, dtype=np.float32)
    clean = np.stack([np.sin(2 * np.pi * (i + 1) * t) for i in range(12)]).astype(np.float32)
    corrupt = clean.copy()
    corrupt[[1, 7]] = 0.0

    stats = diag.compute_signal_diagnostics(clean, corrupt, sample_rate_hz=100.0)

    assert stats["masked_lead_count"] == 2
    assert stats["corrupt_flatline_lead_count"] == 2


def test_compute_signal_diagnostics_handles_nonfinite_values():
    clean = np.zeros((12, 100), dtype=np.float32)
    corrupt = clean.copy()
    corrupt[0, 0] = np.nan
    corrupt[1, 0] = np.inf

    stats = diag.compute_signal_diagnostics(clean, corrupt, sample_rate_hz=100.0)

    assert stats["has_nan"] is True
    assert stats["has_inf"] is True

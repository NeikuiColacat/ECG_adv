from __future__ import annotations

import numpy as np

from util.ecg_viz import sanity_check


def run_quality_gate(signal_ct: np.ndarray, sample_rate: float = 100.0) -> dict:
    report = sanity_check(signal_ct, sample_rate=sample_rate, lead_order="ptbxl")
    warnings = list(report.get("warnings", []))
    status = "pass"
    if warnings:
        status = "warning"
    if report.get("has_nan") or report.get("has_inf") or len(report.get("flatline_leads", [])) >= 6:
        status = "fail"
    return {
        "status": status,
        "warnings": warnings,
        "einthoven_residual": report.get("einthoven_residual"),
        "avR_residual": report.get("avR_residual"),
        "hr_estimate_bpm": report.get("hr_estimate_bpm"),
        "flatline_leads": report.get("flatline_leads", []),
        "has_nan": report.get("has_nan", False),
        "has_inf": report.get("has_inf", False),
    }

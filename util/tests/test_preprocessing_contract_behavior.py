"""Behavior gates for package-owned ECG preprocessing and protocol ownership."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from ecg_adv_gen.data import ECGTWIN_TO_PTBXL_INDICES, get_data_preprocess_contract
from ecg_adv_gen.preprocessing import EXPECTED_LEADS, unified_preprocess_to_1000


REPO = Path(__file__).resolve().parents[2]
DATA_PROTOCOL = REPO / "configs" / "data" / "data_protocol_default.yaml"


def _canonical_band_limited_sines(time_s: np.ndarray) -> np.ndarray:
    columns = []
    for lead_index in range(1, 13):
        frequency_hz = 0.5 * lead_index
        amplitude = 0.4 + 0.03 * lead_index
        offset = 0.2 * lead_index
        phase = np.pi * lead_index / 17.0
        columns.append(
            offset + amplitude * np.sin(2.0 * np.pi * frequency_hz * time_s + phase)
        )
    return np.stack(columns, axis=1).astype(np.float32)


def test_unified_preprocess_resamples_and_restores_canonical_lead_order():
    source_time_s = np.arange(2000, dtype=np.float64) / 200.0
    target_time_s = np.arange(1000, dtype=np.float64) / 100.0
    source_leads = list(reversed(EXPECTED_LEADS))
    waveform = _canonical_band_limited_sines(source_time_s)[:, ::-1].copy()
    expected = _canonical_band_limited_sines(target_time_s)

    result = unified_preprocess_to_1000(
        waveform,
        fs=200,
        source_leads=source_leads,
        target_fs=100,
        target_len=1000,
        apply_filter=False,
        apply_zscore=False,
    )

    assert result is not None
    assert result.shape == (1000, 12)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert target_time_s[-1] == 9.99
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=2e-6)

    wrong_without_resampling = waveform[:1000, ::-1]
    assert not np.allclose(wrong_without_resampling, expected, rtol=1e-6, atol=2e-6)


def test_unified_preprocess_sanitizes_nonfinite_values_before_global_zscore():
    time = np.linspace(-2.0, 2.0, 1000, dtype=np.float32)[:, None]
    leads = np.arange(12, dtype=np.float32)[None, :] / 10.0
    waveform = time + leads
    waveform[0, 0] = np.nan
    waveform[1, 1] = np.inf
    waveform[2, 2] = -np.inf
    sanitized = np.nan_to_num(
        waveform.copy(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    global_mean = float(np.mean(sanitized))
    global_std = float(np.std(sanitized))
    expected = ((sanitized - global_mean) / (global_std + 1e-8)).astype(np.float32)

    result = unified_preprocess_to_1000(
        waveform,
        fs=100,
        target_fs=100,
        target_len=1000,
        apply_filter=False,
        apply_zscore=True,
    )

    assert result is not None
    assert result.shape == (1000, 12)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    per_lead_mean = np.mean(sanitized, axis=0, keepdims=True)
    per_lead_std = np.std(sanitized, axis=0, keepdims=True)
    wrong_per_lead = ((sanitized - per_lead_mean) / (per_lead_std + 1e-8)).astype(np.float32)
    assert not np.allclose(wrong_per_lead, expected, rtol=1e-6, atol=1e-6)


def test_unified_preprocess_rejects_missing_or_invalid_leads():
    waveform = np.zeros((1000, 12), dtype=np.float32)

    assert unified_preprocess_to_1000(
        waveform,
        fs=100,
        source_leads=list(EXPECTED_LEADS[:-1]),
        apply_filter=False,
    ) is None
    assert unified_preprocess_to_1000(
        waveform[:, :11],
        fs=100,
        apply_filter=False,
    ) is None


def test_data_preprocess_contract_carries_locked_model_views():
    contract = get_data_preprocess_contract()

    assert contract.classifier_fs == 100
    assert contract.classifier_len == 1000
    assert contract.lead_order == "ptbxl"
    assert tuple(EXPECTED_LEADS) == (
        "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"
    )
    assert contract.ecgtwin_to_ptbxl_indices == (0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11)
    assert contract.ecgtwin_to_ptbxl_indices == ECGTWIN_TO_PTBXL_INDICES
    assert contract.ecgfounder_input_fs == 500
    assert contract.ecgfounder_input_len == 5000
    assert contract.ecgfounder_input_shape == (12, 5000)


def test_data_protocol_default_is_mapping_evidence_only():
    protocol = yaml.safe_load(DATA_PROTOCOL.read_text(encoding="utf-8"))

    assert protocol["status"] == "mapping-evidence-only"
    assert set(protocol) == {
        "schema_version",
        "protocol_id",
        "status",
        "owner",
        "description",
        "labels",
    }
    assert "mapping evidence" in protocol["description"].lower()
    assert protocol["labels"]["mapping_version"] == "v7_super5_sjr_rgq_review_20260528"
    assert protocol["labels"]["mapping_hash"] == "555ec85d5b51"
    assert protocol["labels"]["class_order"] == ["CD", "HYP", "MI", "NORM", "STTC"]

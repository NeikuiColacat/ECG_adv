"""CPU-only tests for ECG waveform preprocessing helpers."""

from __future__ import annotations

import numpy as np
import pytest

from ecg_adv_gen.preprocessing import (
    PreprocessingError,
    ecgtwin_to_ptbxl,
    infer_lead_axis,
    infer_time_axis,
    linear_resample,
    move_to_channel_last,
    prepare_ecgtwin_decoded_for_classifier,
)
from util.lead_utils import ECGTWIN_TO_PTBXL_INDICES as LEGACY_REORDER


def test_infer_axes_for_channel_last_and_first():
    channel_last = np.zeros((2, 1024, 12), dtype=np.float32)
    channel_first = np.zeros((2, 12, 1024), dtype=np.float32)

    assert infer_lead_axis(channel_last) == 2
    assert infer_time_axis(channel_last) == 1
    assert infer_lead_axis(channel_first) == 1
    assert infer_time_axis(channel_first) == 2


def test_infer_lead_axis_rejects_ambiguous_shape():
    with pytest.raises(PreprocessingError, match="Expected exactly one lead axis"):
        infer_lead_axis(np.zeros((12, 1000, 12), dtype=np.float32))


def test_ecgtwin_to_ptbxl_matches_legacy_reorder_indices():
    sig = np.arange(2 * 4 * 12, dtype=np.float32).reshape(2, 4, 12)

    out = ecgtwin_to_ptbxl(sig)

    np.testing.assert_array_equal(out, sig[:, :, LEGACY_REORDER])


def test_linear_resample_preserves_endpoints_and_dtype():
    sig = np.stack([np.linspace(0, 1, 5), np.linspace(1, 3, 5)], axis=1).astype(np.float32)

    out = linear_resample(sig, target_length=9, time_axis=0)

    assert out.shape == (9, 2)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0], sig[0])
    np.testing.assert_allclose(out[-1], sig[-1])
    np.testing.assert_allclose(out[4], [0.5, 2.0], atol=1e-6)


def test_move_to_channel_last_from_channel_first():
    sig = np.zeros((3, 12, 1000), dtype=np.float32)

    out = move_to_channel_last(sig)

    assert out.shape == (3, 1000, 12)


def test_prepare_ecgtwin_decoded_for_classifier_channel_last():
    sig = np.zeros((2, 1024, 12), dtype=np.float32)
    for lead in range(12):
        sig[:, :, lead] = lead

    out = prepare_ecgtwin_decoded_for_classifier(sig)

    assert out.shape == (2, 1000, 12)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[:, :, 4], 5.0)
    np.testing.assert_allclose(out[:, :, 5], 4.0)


def test_prepare_ecgtwin_decoded_for_classifier_channel_first():
    sig = np.zeros((2, 12, 1024), dtype=np.float32)
    for lead in range(12):
        sig[:, lead, :] = lead

    out = prepare_ecgtwin_decoded_for_classifier(sig)

    assert out.shape == (2, 1000, 12)
    np.testing.assert_allclose(out[:, :, 4], 5.0)
    np.testing.assert_allclose(out[:, :, 5], 4.0)

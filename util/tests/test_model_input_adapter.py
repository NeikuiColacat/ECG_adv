"""CPU-only tests for the canonical raw-100-Hz model input adapter."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from models import prepare_canonical_model_input
from models.contracts import (
    ECGFOUNDER_SPEC,
    EFFICIENTNET1DV2_SPEC,
    ModelSpec,
)


def _global_zscore_bct(value_btc: torch.Tensor, epsilon: float) -> torch.Tensor:
    mean = value_btc.mean(dim=(1, 2), keepdim=True)
    scale = value_btc.std(dim=(1, 2), correction=0, keepdim=True).clamp_min(
        epsilon
    )
    return ((value_btc - mean) / scale).transpose(1, 2).contiguous()


def test_efficientnet_adapter_sanitizes_then_global_zscores_without_mutation() -> None:
    raw = torch.linspace(-2.0, 3.0, steps=2 * 1000 * 12).reshape(2, 1000, 12)
    raw[0, 0, 0] = float("nan")
    raw[0, 1, 1] = float("inf")
    raw[0, 2, 2] = float("-inf")
    original = raw.clone()
    sanitized = torch.nan_to_num(
        raw,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    expected = _global_zscore_bct(sanitized, 1e-5)

    actual = prepare_canonical_model_input(
        raw,
        EFFICIENTNET1DV2_SPEC,
        epsilon=1e-5,
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(raw, original, equal_nan=True)
    assert actual.shape == (2, 12, 1000)
    assert actual.dtype == torch.float32
    assert actual.is_contiguous()
    assert bool(torch.isfinite(actual).all())


def test_ecgfounder_adapter_linearly_upsamples_before_global_zscore() -> None:
    raw = torch.zeros((1, 1000, 12), dtype=torch.float64)
    raw[:, 337:, :] = 2.0
    raw[:, :, 1::2] += torch.linspace(0.0, 1.0, 1000).view(1, 1000, 1)
    raw_float = raw.float()
    upsampled = F.interpolate(
        raw_float.transpose(1, 2),
        size=5000,
        mode="linear",
        align_corners=True,
    ).transpose(1, 2).contiguous()
    expected = _global_zscore_bct(upsampled, 1e-6)

    actual = prepare_canonical_model_input(raw, ECGFOUNDER_SPEC)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    wrong_order = F.interpolate(
        _global_zscore_bct(raw_float, 1e-6),
        size=5000,
        mode="linear",
        align_corners=True,
    )
    assert not torch.allclose(actual, wrong_order, rtol=1e-6, atol=1e-6)
    assert actual.shape == (1, 12, 5000)
    assert actual.device == raw.device
    assert actual.dtype == torch.float32
    assert actual.is_contiguous()


@pytest.mark.parametrize("epsilon", [0.0, -1.0, float("nan"), float("inf")])
def test_adapter_rejects_invalid_epsilon(epsilon: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        prepare_canonical_model_input(
            torch.zeros((1, 1000, 12)),
            EFFICIENTNET1DV2_SPEC,
            epsilon=epsilon,
        )


def test_adapter_rejects_noncanonical_shape_and_unknown_model_spec() -> None:
    with pytest.raises(ValueError, match=r"\(B,1000,12\)"):
        prepare_canonical_model_input(
            torch.zeros((1, 12, 1000)), EFFICIENTNET1DV2_SPEC
        )
    unsupported = ModelSpec(
        name="unsupported",
        input_channels=12,
        input_points=1000,
        sampling_rate_hz=100,
    )
    with pytest.raises(ValueError, match="supports only"):
        prepare_canonical_model_input(
            torch.zeros((1, 1000, 12)), unsupported
        )

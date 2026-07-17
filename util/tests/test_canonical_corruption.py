"""Contracts for the canonical online GPU corruption domain."""

from __future__ import annotations

from itertools import combinations

import pytest
import torch
import torch.nn.functional as F

from core.corruption import (
    CANONICAL_OPERATORS,
    CORRUPTION_DOMAIN_SAMPLING_RATE_HZ,
    INPUT_SAMPLING_RATE_HZ,
    OUTPUT_SAMPLING_RATE_HZ,
    generate_canonical_corruption,
)


def _operator_params(*, probability: float) -> dict[str, dict[str, object]]:
    return {
        "powerline_noise": {
            "min_amplitude": 0.0,
            "max_amplitude": 0.5,
            "p": probability,
            "dependency": False,
        },
        "emg_noise": {
            "min_amplitude": 0.0,
            "max_amplitude": 0.2,
            "p": probability,
            "dependency": False,
        },
        "baseline_wander": {
            "min_amplitude": 0.0,
            "max_amplitude": 0.4,
            "min_freq": 0.01,
            "max_freq": 0.2,
            "k": 3,
            "p": probability,
            "dependency": False,
        },
        "baseline_shift": {
            "min_amplitude": 0.0,
            "max_amplitude": 0.5,
            "shift_ratio": 0.3,
            "num_segment": 1,
            "amplitude_mode": "signed_shared_uniform",
            "p": probability,
            "dependency": False,
        },
        "random_leads_masking": {
            "mask_leads_selection": "random",
            "mask_leads_prob": 0.5,
            "ensure_at_least_one_lead": True,
            "p": probability,
        },
    }


def test_canonical_corruption_is_100_to_500_to_100_and_non_inplace() -> None:
    batch = 20
    clean = torch.linspace(
        -1.0, 1.0, batch * 1000 * 12, dtype=torch.float32
    ).reshape(batch, 1000, 12)
    before = clean.clone()
    composition_indices = torch.arange(batch, dtype=torch.int64)

    result = generate_canonical_corruption(
        clean,
        operator_params=_operator_params(probability=0.0),
        composition_indices=composition_indices,
        generator=torch.Generator(device="cpu").manual_seed(17),
    )

    expected = F.interpolate(
        clean.transpose(1, 2),
        size=5000,
        mode="linear",
        align_corners=True,
    )
    expected = F.interpolate(
        expected,
        size=1000,
        mode="linear",
        align_corners=True,
    ).transpose(1, 2).contiguous()

    assert result.waveform_raw_100hz.shape == clean.shape
    assert result.waveform_raw_100hz.dtype == torch.float32
    assert result.waveform_raw_100hz.is_contiguous()
    assert result.waveform_raw_100hz.data_ptr() != clean.data_ptr()
    torch.testing.assert_close(result.waveform_raw_100hz, expected)
    torch.testing.assert_close(clean, before)
    assert result.diagnostics.input_sampling_rate_hz == INPUT_SAMPLING_RATE_HZ == 100
    assert (
        result.diagnostics.operator_domain_sampling_rate_hz
        == CORRUPTION_DOMAIN_SAMPLING_RATE_HZ
        == 500
    )
    assert result.diagnostics.output_sampling_rate_hz == OUTPUT_SAMPLING_RATE_HZ == 100
    assert result.diagnostics.interpolation_mode == "linear"
    assert result.diagnostics.align_corners is True
    assert result.diagnostics.operator_call_count == len(CANONICAL_OPERATORS)
    assert result.diagnostics.composition_index.tolist() == list(range(20))
    assert result.diagnostics.depth.tolist() == [2] * 10 + [3] * 10
    assert torch.equal(
        result.diagnostics.operator_mask.sum(dim=1),
        result.diagnostics.depth,
    )
    assert torch.count_nonzero(result.diagnostics.output_nonfinite_count) == 0


def test_canonical_corruption_is_deterministic_and_samples_depth2_or_depth3() -> None:
    clean = torch.linspace(-0.5, 0.5, 4 * 1000 * 12).reshape(4, 1000, 12)
    kwargs = {
        "operator_params": _operator_params(probability=1.0),
    }
    first = generate_canonical_corruption(
        clean,
        generator=torch.Generator(device="cpu").manual_seed(31),
        **kwargs,
    )
    second = generate_canonical_corruption(
        clean,
        generator=torch.Generator(device="cpu").manual_seed(31),
        **kwargs,
    )

    torch.testing.assert_close(first.waveform_raw_100hz, second.waveform_raw_100hz)
    assert torch.equal(
        first.diagnostics.composition_index,
        second.diagnostics.composition_index,
    )
    assert set(first.diagnostics.depth.tolist()).issubset({2, 3})
    assert torch.isfinite(first.waveform_raw_100hz).all()


def test_canonical_corruption_rejects_noncanonical_input() -> None:
    with pytest.raises(ValueError, match=r"\(B,1000,12\)"):
        generate_canonical_corruption(
            torch.zeros(2, 5000, 12),
            operator_params=_operator_params(probability=1.0),
            generator=torch.Generator(device="cpu").manual_seed(1),
        )


def test_canonical_composition_order_is_all_depth2_then_depth3() -> None:
    expected = tuple(combinations(CANONICAL_OPERATORS, 2)) + tuple(
        combinations(CANONICAL_OPERATORS, 3)
    )
    result = generate_canonical_corruption(
        torch.zeros(20, 1000, 12),
        operator_params=_operator_params(probability=0.0),
        composition_indices=torch.arange(20),
        generator=torch.Generator(device="cpu").manual_seed(9),
    )
    decoded = tuple(
        tuple(
            operator
            for operator, enabled in zip(CANONICAL_OPERATORS, row.tolist())
            if enabled
        )
        for row in result.diagnostics.operator_mask
    )
    assert decoded == expected

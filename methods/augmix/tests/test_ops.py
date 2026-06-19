"""Tests for methods/augmix/ecg_ops.py via severity.build_op."""
import numpy as np
import pytest
import torch

from methods.augmix.ecg_ops import RandomLeadsMask
from methods.augmix.severity import AVAILABLE_OPS, build_op


def _make_signal(seed: int = 0) -> torch.Tensor:
    # PTBXL crop: (12, 250) at 100Hz after z-score (std ~= 1).
    g = torch.Generator().manual_seed(seed)
    return torch.randn(12, 250, generator=g)


@pytest.mark.parametrize("op_name", AVAILABLE_OPS)
def test_shape_preserved(op_name):
    x = _make_signal()
    op = build_op(op_name, severity=5)
    y = op(x)
    assert y.shape == x.shape, f"{op_name}: {y.shape} != {x.shape}"


@pytest.mark.parametrize("op_name", AVAILABLE_OPS)
def test_dtype_float32(op_name):
    x = _make_signal()
    op = build_op(op_name, severity=5)
    y = op(x)
    assert y.dtype == torch.float32, f"{op_name}: dtype {y.dtype}"


@pytest.mark.parametrize("op_name", AVAILABLE_OPS)
def test_no_nan_inf(op_name):
    x = _make_signal()
    op = build_op(op_name, severity=10)
    y = op(x)
    assert not torch.isnan(y).any(), f"{op_name}: NaN present"
    assert not torch.isinf(y).any(), f"{op_name}: Inf present"


@pytest.mark.parametrize("op_name", AVAILABLE_OPS)
def test_severity_1_mild(op_name):
    """At severity=1, perturbation should be mild (< 0.3 avg abs diff)."""
    x = _make_signal()
    op = build_op(op_name, severity=1)
    # Average over many runs for stability (ops are stochastic).
    diffs = []
    for _ in range(20):
        y = op(x)
        diffs.append((y - x).abs().mean().item())
    avg_diff = float(np.mean(diffs))
    # random_leads_masking at s1 zeroes ~5% of leads -> avg |diff| ~= 0.05 * mean |x| ~= 0.04.
    # Other noise ops at s1 with max_amp ~= 0.03-0.05 -> avg |diff| ~= half of that.
    # Upper bound 0.3 is very loose; just catching pathological explosions.
    assert avg_diff < 0.3, f"{op_name}: severity=1 too strong (avg |diff| = {avg_diff:.3f})"


@pytest.mark.parametrize("op_name", AVAILABLE_OPS)
def test_severity_monotonic(op_name):
    """Severity 10 should produce noticeably larger perturbation than severity 1."""
    x = _make_signal()
    op1 = build_op(op_name, severity=1)
    op10 = build_op(op_name, severity=10)
    diffs1 = [(op1(x) - x).abs().mean().item() for _ in range(20)]
    diffs10 = [(op10(x) - x).abs().mean().item() for _ in range(20)]
    assert np.mean(diffs10) > np.mean(diffs1), (
        f"{op_name}: severity=10 should be > severity=1 "
        f"(got {np.mean(diffs10):.4f} vs {np.mean(diffs1):.4f})"
    )


def test_build_op_bad_name():
    with pytest.raises(ValueError):
        build_op("nonexistent_op", severity=5)


def test_build_op_bad_severity():
    with pytest.raises(ValueError):
        build_op("powerline_noise", severity=0)
    with pytest.raises(ValueError):
        build_op("powerline_noise", severity=11)


def test_random_leads_mask_can_cap_masked_lead_count():
    x = torch.ones(12, 250)
    op = RandomLeadsMask(
        p=1.0,
        mask_leads_selection="random",
        max_masked_leads=3,
    )

    np.random.seed(123)
    masked_counts = []
    for _ in range(40):
        y = op(x)
        masked_counts.append(int((y.abs().sum(dim=1) == 0).sum().item()))

    assert min(masked_counts) >= 1
    assert max(masked_counts) <= 3

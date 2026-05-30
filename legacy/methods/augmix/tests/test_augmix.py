"""Tests for methods/augmix/augmix.py meta-algorithm."""
import numpy as np
import pytest
import torch

from methods.augmix.augmix import augmix, DEFAULT_OPS


def _make_signal(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(12, 250, generator=g)


def test_shape_preserved():
    x = _make_signal()
    y = augmix(x, severity=3, width=3, depth=-1, alpha=1.0)
    assert y.shape == x.shape
    assert y.dtype == torch.float32


def test_no_nan_inf():
    x = _make_signal()
    for _ in range(10):
        y = augmix(x, severity=10, width=3, depth=-1, alpha=1.0)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()


def test_severity_3_reasonable_perturbation():
    """severity=3 should yield moderate perturbation (not identity, not explosion)."""
    x = _make_signal()
    diffs = []
    for _ in range(30):
        y = augmix(x, severity=3, width=3, depth=-1, alpha=1.0)
        diffs.append(((y - x).norm() / x.norm()).item())
    avg = float(np.mean(diffs))
    assert 0.01 < avg < 1.0, f"severity=3 relative diff out of range: {avg:.3f}"


def test_depth_and_width_variants():
    x = _make_signal()
    for depth in [1, 2, 3]:
        for width in [1, 2, 3, 5]:
            y = augmix(x, severity=3, width=width, depth=depth)
            assert y.shape == x.shape


def test_custom_ops_subset():
    x = _make_signal()
    y = augmix(x, severity=3, ops=["powerline_noise"])
    assert y.shape == x.shape


def test_unknown_op_rejected():
    x = _make_signal()
    with pytest.raises(ValueError):
        augmix(x, ops=["nonexistent_op"])


def test_default_ops_is_all_five():
    assert set(DEFAULT_OPS) == {
        "powerline_noise", "emg_noise", "baseline_shift",
        "baseline_wander", "random_leads_masking",
    }

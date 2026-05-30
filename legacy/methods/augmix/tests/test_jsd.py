"""Tests for methods/augmix/jsd_loss.py."""
import pytest
import torch

from methods.augmix.jsd_loss import jsd_multilabel


def test_identical_inputs_zero_loss():
    logits = torch.randn(4, 26)
    loss = jsd_multilabel(logits, logits.clone(), logits.clone())
    assert loss.item() < 1e-6, f"JSD on identical inputs should be ~0, got {loss.item()}"


def test_different_inputs_positive_loss():
    lc = torch.randn(4, 26)
    l1 = torch.randn(4, 26)
    l2 = torch.randn(4, 26)
    loss = jsd_multilabel(lc, l1, l2)
    assert loss.item() > 0.0, "JSD on different inputs should be > 0"


def test_gradient_flows():
    lc = torch.randn(4, 26, requires_grad=True)
    l1 = torch.randn(4, 26, requires_grad=True)
    l2 = torch.randn(4, 26, requires_grad=True)
    loss = jsd_multilabel(lc, l1, l2)
    loss.backward()
    assert lc.grad is not None and lc.grad.abs().sum() > 0
    assert l1.grad is not None and l1.grad.abs().sum() > 0
    assert l2.grad is not None and l2.grad.abs().sum() > 0


def test_reduction_modes():
    lc = torch.randn(4, 26)
    l1 = torch.randn(4, 26)
    l2 = torch.randn(4, 26)
    mean_loss = jsd_multilabel(lc, l1, l2, reduction="mean")
    sum_loss = jsd_multilabel(lc, l1, l2, reduction="sum")
    none_loss = jsd_multilabel(lc, l1, l2, reduction="none")
    assert mean_loss.shape == torch.Size([])
    assert sum_loss.shape == torch.Size([])
    assert none_loss.shape == torch.Size([4])
    assert abs(sum_loss.item() - none_loss.sum().item()) < 1e-5
    assert abs(mean_loss.item() - none_loss.mean().item()) < 1e-5


def test_non_negative():
    """JSD is always non-negative."""
    for _ in range(20):
        lc = torch.randn(4, 26) * 5.0
        l1 = torch.randn(4, 26) * 5.0
        l2 = torch.randn(4, 26) * 5.0
        loss = jsd_multilabel(lc, l1, l2)
        assert loss.item() >= -1e-6, f"JSD negative: {loss.item()}"

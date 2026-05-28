"""CPU-only tests for small torch training utilities."""

from __future__ import annotations

import torch.nn as nn

from ecg_adv_gen.training import set_module_requires_grad


def test_set_module_requires_grad_toggles_all_parameters():
    module = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))

    set_module_requires_grad(module, False)
    assert all(not param.requires_grad for param in module.parameters())

    set_module_requires_grad(module, True)
    assert all(param.requires_grad for param in module.parameters())

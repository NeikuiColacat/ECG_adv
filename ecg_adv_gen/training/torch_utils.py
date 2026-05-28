"""Small torch module utilities used by training entrypoints."""

from __future__ import annotations

import torch.nn as nn


def set_module_requires_grad(module: nn.Module, flag: bool) -> None:
    """Set ``requires_grad`` on all parameters in ``module``."""
    for param in module.parameters():
        param.requires_grad_(bool(flag))


__all__ = ["set_module_requires_grad"]

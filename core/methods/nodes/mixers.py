"""View-mixing node adapters."""

from __future__ import annotations

from core.methods.contracts import NodeContext, ViewValue


def augmix(
    context: NodeContext,
    inputs: tuple[ViewValue, ...],
) -> ViewValue:
    """Delegate AugMix arithmetic and branch generation to an explicit adapter."""

    return context.call_adapter("augmix", inputs)


__all__ = ["augmix"]

"""Canonical waveform-operation adapters."""

from __future__ import annotations

from core.methods.contracts import NodeContext, ViewValue


def canonical_corruption(
    context: NodeContext,
    inputs: tuple[ViewValue, ...],
) -> ViewValue:
    """Delegate to the existing canonical corruption implementation."""

    return context.call_adapter("canonical_corruption", inputs)


__all__ = ["canonical_corruption"]

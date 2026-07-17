"""Latent-operation adapters."""

from __future__ import annotations

from core.methods.contracts import NodeContext, ViewValue


def lhat_attack(
    context: NodeContext,
    inputs: tuple[ViewValue, ...],
) -> ViewValue:
    """Delegate to the existing LHAT implementation through a runner adapter."""

    return context.call_adapter("lhat_attack", inputs)


def paired_latent_bridge(
    context: NodeContext,
    inputs: tuple[ViewValue, ...],
) -> ViewValue:
    """Delegate strict paired encoding/interpolation/decoding to an adapter."""

    return context.call_adapter("paired_latent_bridge", inputs)


__all__ = ["lhat_attack", "paired_latent_bridge"]

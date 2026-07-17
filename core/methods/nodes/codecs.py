"""VAE codec node adapters.

No ECGTwin implementation is duplicated here.  Execution delegates to
callables supplied by the managed runner.
"""

from __future__ import annotations

from core.methods.contracts import NodeContext, ViewValue


def vae_encode(
    context: NodeContext,
    inputs: tuple[ViewValue, ...],
) -> ViewValue:
    return context.call_adapter("vae_encode", inputs)


def vae_decode(
    context: NodeContext,
    inputs: tuple[ViewValue, ...],
) -> ViewValue:
    return context.call_adapter("vae_decode", inputs)


__all__ = ["vae_decode", "vae_encode"]

"""Replay-buffer source and sink adapters."""

from __future__ import annotations

from core.methods.contracts import NodeContext, ViewValue


def replay_write(
    context: NodeContext,
    inputs: tuple[ViewValue, ...],
) -> ViewValue:
    """Write one typed view and return the adapter-provided routed view."""

    return context.call_adapter("replay_write", inputs)


def replay_read(
    context: NodeContext,
    inputs: tuple[ViewValue, ...],
) -> ViewValue:
    """Read a typed waveform batch from an explicitly supplied replay buffer."""

    return context.call_adapter("replay_read", inputs)


__all__ = ["replay_read", "replay_write"]

"""Source, pairing, and candidate-selection nodes."""

from __future__ import annotations

from core.methods.contracts import (
    NodeContext,
    PairedView,
    Provenance,
    ViewValue,
    WaveformView,
)


def clean_source(
    context: NodeContext,
    inputs: tuple[ViewValue, ...],
) -> ViewValue:
    """Return one explicitly supplied canonical clean source."""

    if inputs:
        raise ValueError("source.clean does not accept graph inputs")
    unknown = set(context.params) - {"source"}
    if unknown:
        raise ValueError(f"source.clean received unsupported params: {sorted(unknown)}")
    source_name = str(context.params.get("source", "clean"))
    value = context.source(source_name)
    if not isinstance(value, WaveformView):
        raise TypeError("source.clean requires a WaveformView source")
    return WaveformView(
        name=context.node_id,
        waveform=value.waveform,
        labels=value.labels,
        sample_ids=value.sample_ids,
        valid_mask=value.valid_mask,
        provenance=Provenance(
            node_id=context.node_id,
            operation=context.node_type,
            parent_names=(value.name,),
            rng_namespace=None,
            parameters={"source": source_name},
        ),
        metadata={**dict(value.metadata), "source_view": value.name},
    )


def strict_pair(
    context: NodeContext,
    inputs: tuple[ViewValue, ...],
) -> ViewValue:
    """Create a strict one-to-one waveform pair."""

    if context.params:
        raise ValueError("pair.strict does not accept params")
    if len(inputs) != 2 or not all(isinstance(value, WaveformView) for value in inputs):
        raise TypeError("pair.strict requires exactly two WaveformView inputs")
    left, right = inputs
    assert isinstance(left, WaveformView) and isinstance(right, WaveformView)
    return PairedView(
        name=context.node_id,
        left=left,
        right=right,
        provenance=Provenance(
            node_id=context.node_id,
            operation=context.node_type,
            parent_names=(left.name, right.name),
            parameters={},
        ),
    )


def select_candidates(
    context: NodeContext,
    inputs: tuple[ViewValue, ...],
) -> ViewValue:
    """Delegate candidate lookup to the registered latent-pool adapter."""

    return context.call_adapter("candidate_selector", inputs)


__all__ = ["clean_source", "select_candidates", "strict_pair"]

from __future__ import annotations

import pytest
import torch

from util.ecg_image_renderer import PulseECGTensorRenderer, PulseRenderGeometry


@pytest.fixture()
def renderer() -> PulseECGTensorRenderer:
    geometry = PulseRenderGeometry(resolution_dpi=20)
    background = torch.ones(
        (3, geometry.height_pixels, geometry.width_pixels), dtype=torch.float32
    )
    return PulseECGTensorRenderer(background, geometry=geometry)


def test_render_returns_deterministic_image_tensor_without_mutating_input(
    renderer: PulseECGTensorRenderer,
) -> None:
    time = torch.arange(5000, dtype=torch.float32) / 500.0
    signal = 0.2 * torch.sin(2.0 * torch.pi * 1.3 * time)
    waveforms = signal.view(1, 5000, 1).expand(2, -1, 12).clone()
    before = waveforms.clone()

    first = renderer.render(waveforms)
    second = renderer.render(waveforms)

    assert first.shape == (2, 3, 170, 220)
    assert first.dtype == torch.float32
    assert first.device == waveforms.device
    assert torch.equal(first, second)
    assert torch.equal(first[:1], renderer.render(waveforms[:1]))
    mixed_batch = torch.cat((waveforms[:1], 3 * waveforms[:1]), dim=0)
    assert torch.equal(first[:1], renderer.render(mixed_batch)[:1])
    assert torch.equal(waveforms, before)
    assert torch.isfinite(first).all()
    assert 0.0 <= float(first.min()) < float(first.max()) <= 1.0


def test_render_for_pulse_returns_five_normalized_tiles(
    renderer: PulseECGTensorRenderer,
) -> None:
    waveforms = torch.zeros((1, 5000, 12), dtype=torch.float32)
    tiles, image_sizes = renderer.render_for_pulse(waveforms)

    assert tiles.shape == (1, 5, 3, 336, 336)
    assert tiles.dtype == torch.float32
    assert tiles.is_contiguous()
    assert torch.isfinite(tiles).all()
    assert image_sizes == [(220, 170)]


@pytest.mark.parametrize(
    "waveforms,error",
    [
        (torch.zeros((5000, 12)), "shape"),
        (torch.zeros((1, 1000, 12)), "shape"),
        (torch.zeros((1, 5000, 11)), "shape"),
        (torch.zeros((1, 5000, 12), dtype=torch.int64), "floating-point"),
    ],
)
def test_render_rejects_invalid_contract(
    renderer: PulseECGTensorRenderer,
    waveforms: torch.Tensor,
    error: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        renderer.render(waveforms)


def test_render_rejects_nonfinite_waveform(
    renderer: PulseECGTensorRenderer,
) -> None:
    waveforms = torch.zeros((1, 5000, 12), dtype=torch.float32)
    waveforms[0, 10, 3] = torch.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        renderer.render(waveforms)

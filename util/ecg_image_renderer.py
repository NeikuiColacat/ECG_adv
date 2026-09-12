"""GPU ECG-paper rasterization for the exploratory PULSE-7B evaluator.

The frozen PULSE path uses ECG-Image-Kit to create a 2200 x 1700 PNG and then
turns that image into five normalized CLIP tiles.  Repeating Matplotlib and PNG
encoding for every record is expensive and prevents a fully online waveform
pipeline.

This module keeps the invariant parts of the official renderer (paper grid,
labels, calibration pulses, separators and scale text) in one background
tensor.  Only the waveform traces are rasterized for each input, on the input
tensor's device.  The resulting image can either be returned as an RGB Torch
tensor or converted directly to the five tiles consumed by PULSE.

The background is created once with the frozen ECG-Image-Kit implementation.
No per-record PIL, Matplotlib, PNG, or filesystem work occurs after that one
initialization step.
"""

from __future__ import annotations

import math
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional
import yaml
from PIL import Image


PTBXL_LEADS: tuple[str, ...] = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)
PULSE_LEAD_NAMES: tuple[str, ...] = (
    "III",
    "aVF",
    "V3",
    "V6",
    "II",
    "aVL",
    "V2",
    "V5",
    "I",
    "aVR",
    "V1",
    "V4",
)
CLIP_IMAGE_MEAN: tuple[float, float, float] = (
    0.48145466,
    0.4578275,
    0.40821073,
)
CLIP_IMAGE_STD: tuple[float, float, float] = (
    0.26862954,
    0.26130258,
    0.27577711,
)


@dataclass(frozen=True)
class PulseRenderGeometry:
    """Geometry of the exact ECG-Image-Kit layout used by the PULSE probe."""

    sample_rate_hz: int = 500
    duration_seconds: int = 10
    columns: int = 4
    resolution_dpi: int = 200
    width_inches: float = 11.0
    height_inches: float = 8.5
    grid_inch: float = 5.0 / 25.4
    x_grid_seconds: float = 0.2
    y_grid_mv: float = 0.5
    lead_seconds: float = 2.5
    dc_offset_seconds: float = 0.2
    lead_name_offset_mv: float = 0.5

    @property
    def width_pixels(self) -> int:
        return int(self.width_inches * self.resolution_dpi)

    @property
    def height_pixels(self) -> int:
        return int(self.height_inches * self.resolution_dpi)

    @property
    def samples(self) -> int:
        return self.sample_rate_hz * self.duration_seconds

    @property
    def segment_samples(self) -> int:
        return int(round(self.sample_rate_hz * self.lead_seconds))

    @property
    def x_max(self) -> float:
        return self.width_inches * self.x_grid_seconds / self.grid_inch

    @property
    def y_max(self) -> float:
        return self.height_inches * self.y_grid_mv / self.grid_inch

    @property
    def x_gap(self) -> float:
        unused = self.x_max - self.columns * self.lead_seconds
        return math.floor((unused / 2.0) / self.x_grid_seconds) * self.x_grid_seconds

    @property
    def row_height(self) -> float:
        # Three short-lead rows, one full-II row, plus the toolkit's two-row
        # vertical margin convention.
        return self.y_max / 6.0

    def validate(self) -> None:
        if self.sample_rate_hz != 500 or self.samples != 5000:
            raise ValueError("PULSE native renderer requires 500 Hz and 5000 samples")
        if self.columns != 4 or self.lead_seconds != 2.5:
            raise ValueError("PULSE renderer requires four 2.5-second columns")
        if self.width_pixels <= 0 or self.height_pixels <= 0:
            raise ValueError("render dimensions must be positive")


def _read_toolkit_config(toolkit_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    generator_dir = Path(toolkit_dir).resolve() / "codes" / "ecg-image-generator"
    plot_path = generator_dir / "ecg_plot.py"
    config_path = generator_dir / "config.yaml"
    if not plot_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"incomplete ECG-Image-Kit checkout: {generator_dir}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"invalid ECG-Image-Kit config: {config_path}")
    return generator_dir, config


def build_official_background(
    toolkit_dir: str | Path,
    *,
    geometry: PulseRenderGeometry = PulseRenderGeometry(),
    tmp_root: str | Path = "/dev/shm",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Create the invariant official paper as a CPU ``float32 CHW`` tensor.

    NaN traces cause Matplotlib to draw no dynamic signal while retaining the
    official grid, labels, separators, calibration pulses and scale text.  The
    temporary PNG is removed before this function returns.
    """

    geometry.validate()
    generator_dir, config = _read_toolkit_config(toolkit_dir)
    expected_names = list(PULSE_LEAD_NAMES)
    if list(config.get("leadNames_12", [])) != expected_names:
        raise ValueError(
            "ECG-Image-Kit leadNames_12 drifted: "
            f"{config.get('leadNames_12')!r} != {expected_names!r}"
        )
    if float(config.get("paper_len", -1)) != float(geometry.duration_seconds):
        raise ValueError("ECG-Image-Kit paper length is not 10 seconds")

    inserted = False
    if str(generator_dir) not in sys.path:
        sys.path.insert(0, str(generator_dir))
        inserted = True
    try:
        from ecg_plot import ecg_plot

        segment = np.full((geometry.segment_samples,), np.nan, dtype=np.float32)
        full = np.full((geometry.samples,), np.nan, dtype=np.float32)
        frame = {lead: segment for lead in PTBXL_LEADS}
        frame["fullII"] = full
        with tempfile.TemporaryDirectory(
            dir=Path(tmp_root), prefix="pulse_gpu_background_"
        ) as temporary:
            output_dir = Path(temporary)
            stem = "official_static_background"
            prior_state = random.getstate()
            random.seed(0)
            try:
                ecg_plot(
                    frame,
                    configs=config,
                    sample_rate=geometry.sample_rate_hz,
                    columns=geometry.columns,
                    rec_file_name=stem,
                    output_dir=str(output_dir),
                    resolution=geometry.resolution_dpi,
                    pad_inches=0,
                    lead_index=PTBXL_LEADS,
                    full_mode="II",
                    store_text_bbox=False,
                    full_header_file="",
                    style=None,
                    show_lead_name=True,
                    show_grid=True,
                    show_dc_pulse=True,
                    standard_colours=5,
                    bbox=False,
                    print_txt=False,
                    json_dict={},
                    start_index=0,
                    store_configs=0,
                    lead_length_in_seconds=geometry.lead_seconds,
                )
            finally:
                random.setstate(prior_state)
            image_path = output_dir / f"{stem}.png"
            with Image.open(image_path) as opened:
                pixels = np.asarray(opened.convert("RGB"), dtype=np.uint8).copy()
    finally:
        if inserted:
            sys.path.remove(str(generator_dir))

    expected_shape = (geometry.height_pixels, geometry.width_pixels, 3)
    if pixels.shape != expected_shape:
        raise ValueError(f"official background shape {pixels.shape} != {expected_shape}")
    background = torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255.0)
    metadata = {
        "schema_version": 1,
        "source": "frozen_ecg_image_kit_static_background",
        "toolkit_dir": str(Path(toolkit_dir).resolve()),
        "shape_chw": list(background.shape),
        "dtype": str(background.dtype),
        "value_range": [float(background.min()), float(background.max())],
        "lead_order": list(PTBXL_LEADS),
        "display_lead_names_bottom_to_top": list(PULSE_LEAD_NAMES),
        "sample_rate_hz": geometry.sample_rate_hz,
        "duration_seconds": geometry.duration_seconds,
        "resolution_dpi": geometry.resolution_dpi,
    }
    return background.contiguous(), metadata


class PulseECGTensorRenderer:
    """Render native-500-Hz ECG batches into PULSE image tensors."""

    def __init__(
        self,
        background_chw: torch.Tensor,
        *,
        geometry: PulseRenderGeometry = PulseRenderGeometry(),
        trace_gray: float | None = None,
        line_width_points: float = 0.75,
    ) -> None:
        geometry.validate()
        expected = (3, geometry.height_pixels, geometry.width_pixels)
        if tuple(background_chw.shape) != expected:
            raise ValueError(
                f"background must have shape {expected}, "
                f"got {tuple(background_chw.shape)}"
            )
        if not background_chw.is_floating_point():
            raise TypeError("background must be a floating-point RGB tensor")
        if not torch.isfinite(background_chw).all():
            raise ValueError("background contains NaN or infinity")
        if float(background_chw.min()) < 0.0 or float(background_chw.max()) > 1.0:
            raise ValueError("background must be in the [0, 1] range")
        self.geometry = geometry
        self.background = background_chw.detach().contiguous()
        # The frozen CPU path resets Python random to zero before each image.
        self.trace_gray = (
            float(random.Random(0).uniform(0.0, 0.2))
            if trace_gray is None
            else float(trace_gray)
        )
        if not 0.0 <= self.trace_gray <= 1.0:
            raise ValueError("trace_gray must lie in [0, 1]")
        self.line_width_points = float(line_width_points)
        if self.line_width_points <= 0.0:
            raise ValueError("line_width_points must be positive")
        self._lead_to_channel = {
            lead: index for index, lead in enumerate(PTBXL_LEADS)
        }
        self._segment_start = {
            "I": 0,
            "II": 0,
            "III": 0,
            "aVR": geometry.segment_samples,
            "aVL": geometry.segment_samples,
            "aVF": geometry.segment_samples,
            "V1": 2 * geometry.segment_samples,
            "V2": 2 * geometry.segment_samples,
            "V3": 2 * geometry.segment_samples,
            "V4": 3 * geometry.segment_samples,
            "V5": 3 * geometry.segment_samples,
            "V6": 3 * geometry.segment_samples,
        }

    @classmethod
    def from_ecg_image_kit(
        cls,
        toolkit_dir: str | Path,
        *,
        device: torch.device | str,
        geometry: PulseRenderGeometry = PulseRenderGeometry(),
        tmp_root: str | Path = "/dev/shm",
        line_width_points: float = 0.75,
    ) -> tuple["PulseECGTensorRenderer", dict[str, Any]]:
        background, metadata = build_official_background(
            toolkit_dir, geometry=geometry, tmp_root=tmp_root
        )
        renderer = cls(
            background.to(device=device),
            geometry=geometry,
            line_width_points=line_width_points,
        )
        metadata = {
            **metadata,
            "device": str(renderer.background.device),
            "trace_gray": renderer.trace_gray,
            "line_width_points": renderer.line_width_points,
        }
        return renderer, metadata

    def _validate_waveforms(self, waveforms_btc: torch.Tensor) -> None:
        expected = (self.geometry.samples, len(PTBXL_LEADS))
        if not isinstance(waveforms_btc, torch.Tensor):
            raise TypeError("waveforms must be a torch.Tensor")
        if waveforms_btc.ndim != 3 or tuple(waveforms_btc.shape[1:]) != expected:
            raise ValueError(
                f"waveforms must have shape (B,{expected[0]},{expected[1]}), "
                f"got {tuple(waveforms_btc.shape)}"
            )
        if not waveforms_btc.is_floating_point():
            raise TypeError("waveforms must be floating-point physical-mV values")
        if waveforms_btc.device != self.background.device:
            raise ValueError(
                f"waveforms are on {waveforms_btc.device}, background is on "
                f"{self.background.device}"
            )
        if not torch.isfinite(waveforms_btc).all():
            raise ValueError("waveforms contain NaN or infinity")

    def _trace_coordinates(
        self,
        values_bt: torch.Tensor,
        *,
        x_offset_seconds: float,
        y_offset_mv: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        geometry = self.geometry
        count = values_bt.shape[1]
        x_data = (
            torch.arange(count, device=values_bt.device, dtype=torch.float32)
            / float(geometry.sample_rate_hz)
            + geometry.x_gap
            + geometry.dc_offset_seconds
            + x_offset_seconds
        )
        x_pixels = x_data * (float(geometry.width_pixels) / geometry.x_max)
        y_pixels = float(geometry.height_pixels) * (
            1.0 - (values_bt.float() + y_offset_mv) / geometry.y_max
        )
        return x_pixels, y_pixels

    def _draw_trace(
        self,
        canvas_bchw: torch.Tensor,
        values_bt: torch.Tensor,
        *,
        x_offset_seconds: float,
        y_offset_mv: float,
    ) -> None:
        """Draw one batched lead with a per-column min/max envelope."""

        x_pixels, y_pixels = self._trace_coordinates(
            values_bt,
            x_offset_seconds=x_offset_seconds,
            y_offset_mv=y_offset_mv,
        )
        x_indices = torch.floor(x_pixels).to(torch.int64)
        x_min = max(int(x_indices.min()), 0)
        x_max = min(int(x_indices.max()), self.geometry.width_pixels - 1)
        if x_max < x_min:
            return
        width = x_max - x_min + 1
        local_indices = (x_indices - x_min).clamp(0, width - 1)
        expanded_indices = local_indices.unsqueeze(0).expand(values_bt.shape[0], -1)
        low = torch.full(
            (values_bt.shape[0], width),
            torch.inf,
            device=values_bt.device,
            dtype=torch.float32,
        )
        high = torch.full_like(low, -torch.inf)
        low.scatter_reduce_(1, expanded_indices, y_pixels, reduce="amin", include_self=True)
        high.scatter_reduce_(1, expanded_indices, y_pixels, reduce="amax", include_self=True)

        # Join neighboring column envelopes so steep QRS edges remain connected.
        previous_low = torch_functional.pad(low[:, :-1], (1, 0), value=torch.inf)
        previous_high = torch_functional.pad(high[:, :-1], (1, 0), value=-torch.inf)
        low = torch.minimum(low, previous_low)
        high = torch.maximum(high, previous_high)
        valid = torch.isfinite(low) & torch.isfinite(high)

        rows = torch.arange(
            self.geometry.height_pixels,
            device=values_bt.device,
            dtype=torch.float32,
        ).view(1, -1, 1)
        distance = torch.maximum(low.unsqueeze(1) - rows, rows - high.unsqueeze(1))
        distance.clamp_min_(0.0)
        # 0.75 points at the selected DPI, plus a one-pixel antialias fringe.
        radius = max(
            0.5,
            self.line_width_points * self.geometry.resolution_dpi / 72.0 / 2.0,
        )
        coverage = (radius + 0.5 - distance).clamp_(0.0, 1.0)
        coverage.mul_(valid.unsqueeze(1))
        region = canvas_bchw[:, :, :, x_min : x_max + 1]
        alpha = coverage.unsqueeze(1)
        region.mul_(1.0 - alpha).add_(self.trace_gray * alpha)

    @torch.inference_mode()
    def render(self, waveforms_btc: torch.Tensor) -> torch.Tensor:
        """Return ``float32 (B,3,1700,2200)`` RGB tensors in ``[0,1]``."""

        self._validate_waveforms(waveforms_btc)
        batch = waveforms_btc.shape[0]
        canvas = self.background.unsqueeze(0).expand(batch, -1, -1, -1).clone()
        segment = self.geometry.segment_samples

        for display_index, lead in enumerate(PULSE_LEAD_NAMES):
            channel = self._lead_to_channel[lead]
            start = self._segment_start[lead]
            values = waveforms_btc[:, start : start + segment, channel]
            row_from_bottom = display_index // self.geometry.columns
            y_offset = self.geometry.row_height * (1.5 + row_from_bottom)
            x_offset = (
                display_index % self.geometry.columns
            ) * self.geometry.lead_seconds
            self._draw_trace(
                canvas,
                values,
                x_offset_seconds=x_offset,
                y_offset_mv=y_offset,
            )

        full_ii_offset = (
            self.geometry.row_height / 2.0
            - self.geometry.lead_name_offset_mv
            + 0.8
        )
        self._draw_trace(
            canvas,
            waveforms_btc[:, :, self._lead_to_channel["II"]],
            x_offset_seconds=0.0,
            y_offset_mv=full_ii_offset,
        )
        return canvas.clamp_(0.0, 1.0)

    @staticmethod
    def _bicubic_resize(images: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        resized = torch_functional.interpolate(
            images,
            size=size,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        # PIL performs the corresponding resize in uint8 and cannot overshoot.
        return resized.clamp_(0.0, 1.0)

    @torch.inference_mode()
    def preprocess_for_pulse(self, images_bchw: torch.Tensor) -> torch.Tensor:
        """Return normalized ``(B,5,3,336,336)`` PULSE any-res inputs."""

        expected = (
            3,
            self.geometry.height_pixels,
            self.geometry.width_pixels,
        )
        if images_bchw.ndim != 4 or tuple(images_bchw.shape[1:]) != expected:
            raise ValueError(
                f"images must have shape (B,{expected}), "
                f"got {tuple(images_bchw.shape)}"
            )
        if images_bchw.device != self.background.device:
            raise ValueError("images and renderer background must use the same device")
        if not images_bchw.is_floating_point() or not torch.isfinite(images_bchw).all():
            raise ValueError("images must be finite floating-point tensors")

        # PULSE chooses 672x672 for a 2200x1700 ECG page, preserving aspect
        # ratio for the four spatial tiles and using a distorted 336x336 global
        # thumbnail as tile zero.
        target = 672
        patch = 336
        source_height = self.geometry.height_pixels
        source_width = self.geometry.width_pixels
        scale = min(target / source_width, target / source_height)
        resized_width = min(math.ceil(source_width * scale), target)
        resized_height = min(math.ceil(source_height * scale), target)
        spatial = self._bicubic_resize(
            images_bchw, (resized_height, resized_width)
        )
        padded = torch.zeros(
            (images_bchw.shape[0], 3, target, target),
            device=images_bchw.device,
            dtype=images_bchw.dtype,
        )
        paste_x = (target - resized_width) // 2
        paste_y = (target - resized_height) // 2
        padded[
            :,
            :,
            paste_y : paste_y + resized_height,
            paste_x : paste_x + resized_width,
        ] = spatial
        spatial_tiles = (
            padded.reshape(images_bchw.shape[0], 3, 2, patch, 2, patch)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(images_bchw.shape[0], 4, 3, patch, patch)
        )
        global_tile = self._bicubic_resize(
            images_bchw, (patch, patch)
        ).unsqueeze(1)
        tiles = torch.cat((global_tile, spatial_tiles), dim=1)
        mean = torch.tensor(
            CLIP_IMAGE_MEAN, device=tiles.device, dtype=tiles.dtype
        ).view(1, 1, 3, 1, 1)
        std = torch.tensor(
            CLIP_IMAGE_STD, device=tiles.device, dtype=tiles.dtype
        ).view(1, 1, 3, 1, 1)
        return tiles.sub(mean).div_(std).contiguous()

    @torch.inference_mode()
    def render_for_pulse(
        self, waveforms_btc: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """Render and directly return PULSE tiles plus PIL-style image sizes."""

        images = self.render(waveforms_btc)
        tiles = self.preprocess_for_pulse(images)
        image_sizes = [
            (self.geometry.width_pixels, self.geometry.height_pixels)
            for _ in range(waveforms_btc.shape[0])
        ]
        return tiles, image_sizes

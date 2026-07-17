"""ECG image helpers adapted from the ECGTwin author's plotting path.

The ECGTwin repository visualizes decoded waveforms by transposing them to
``(lead, time)`` and calling :mod:`ecg_plot` with one display column and a row
height of four.  This module preserves that implementation while making the
sampling rate, input layout, and lead identity explicit for the rebuilt data
contracts.

Author reference:
``model/ECGTwin/utils/inference_utils.py`` in the external ECGTwin checkout.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
import torch

if TYPE_CHECKING:
    from matplotlib.figure import Figure


ECGLayout = Literal["time_channel", "channel_time"]
CANONICAL_LEAD_ORDER = (
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
ECGTWIN_AUTHOR_LEAD_ORDER = (
    "I",
    "II",
    "III",
    "aVR",
    "aVF",
    "aVL",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)


def _lead_time_array(
    waveform: np.ndarray | torch.Tensor,
    *,
    layout: ECGLayout,
    lead_order: Sequence[str],
) -> np.ndarray:
    if isinstance(waveform, torch.Tensor):
        value = waveform.detach().cpu().numpy()
    else:
        value = np.asarray(waveform)
    if value.ndim != 2:
        raise ValueError("one ECG waveform must be a two-dimensional array")
    if layout not in {"time_channel", "channel_time"}:
        raise ValueError("layout must be time_channel or channel_time")
    leads = tuple(str(lead) for lead in lead_order)
    if not leads or len(set(leads)) != len(leads):
        raise ValueError("lead_order must contain unique non-empty lead names")
    lead_time = value.T if layout == "time_channel" else value
    if lead_time.shape[0] != len(leads):
        raise ValueError(
            "waveform lead dimension does not match lead_order: "
            f"shape={tuple(value.shape)}, layout={layout}, leads={len(leads)}"
        )
    result = np.array(lead_time, dtype=np.float32, order="C", copy=True)
    if result.shape[1] <= 1:
        raise ValueError("ECG waveform must contain at least two time samples")
    if not np.isfinite(result).all():
        raise ValueError("ECG visualization requires finite waveform values")
    return result


def create_ecg_figure(
    waveform: np.ndarray | torch.Tensor,
    *,
    sampling_rate_hz: float,
    layout: ECGLayout = "time_channel",
    lead_order: Sequence[str] = CANONICAL_LEAD_ORDER,
    title: str | None = None,
    columns: int = 1,
    row_height: int = 4,
    style: str | None = None,
    show_grid: bool = True,
) -> Figure:
    """Return the ECGTwin-author-style ECG paper-grid figure.

    Unlike the author script, this function does not hard-code ``102.4`` Hz or
    the ECGTwin/MIMIC aVF/aVL ordering.  Callers must supply the actual sampling
    rate and may explicitly pass :data:`ECGTWIN_AUTHOR_LEAD_ORDER` before a
    waveform has been converted to the project's canonical lead order.
    """

    if isinstance(sampling_rate_hz, bool) or not isinstance(
        sampling_rate_hz, (int, float)
    ):
        raise TypeError("sampling_rate_hz must be numeric")
    rate = float(sampling_rate_hz)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("sampling_rate_hz must be finite and positive")
    if isinstance(columns, bool) or int(columns) <= 0:
        raise ValueError("columns must be a positive integer")
    if isinstance(row_height, bool) or int(row_height) <= 0:
        raise ValueError("row_height must be a positive integer")
    leads = tuple(str(lead) for lead in lead_order)
    lead_time = _lead_time_array(
        waveform,
        layout=layout,
        lead_order=leads,
    )

    try:
        import ecg_plot
    except ImportError as exc:
        raise RuntimeError(
            "ECGTwin-author-style visualization requires the ecg-plot package"
        ) from exc
    from matplotlib import pyplot as plt

    ecg_plot.plot(
        lead_time,
        rate,
        lead_index=list(leads),
        title=title,
        columns=int(columns),
        row_height=int(row_height),
        style=style,
        show_grid=bool(show_grid),
    )
    return plt.gcf()


def save_ecg_image(
    waveform: np.ndarray | torch.Tensor,
    output_path: str | Path,
    *,
    sampling_rate_hz: float,
    layout: ECGLayout = "time_channel",
    lead_order: Sequence[str] = CANONICAL_LEAD_ORDER,
    title: str | None = None,
    columns: int = 1,
    row_height: int = 4,
    style: str | None = None,
    show_grid: bool = True,
    dpi: int = 150,
) -> Path:
    """Render one ECG to PNG and close the Matplotlib figure."""

    target = Path(output_path).expanduser().resolve()
    if target.suffix.lower() != ".png":
        raise ValueError("ECG image output_path must end in .png")
    if isinstance(dpi, bool) or int(dpi) <= 0:
        raise ValueError("dpi must be a positive integer")
    target.parent.mkdir(parents=True, exist_ok=True)
    from matplotlib import pyplot as plt

    figure = create_ecg_figure(
        waveform,
        sampling_rate_hz=sampling_rate_hz,
        layout=layout,
        lead_order=lead_order,
        title=title,
        columns=columns,
        row_height=row_height,
        style=style,
        show_grid=show_grid,
    )
    try:
        figure.savefig(target, dpi=int(dpi), bbox_inches="tight")
    finally:
        plt.close(figure)
    return target


__all__ = [
    "CANONICAL_LEAD_ORDER",
    "ECGTWIN_AUTHOR_LEAD_ORDER",
    "ECGLayout",
    "create_ecg_figure",
    "save_ecg_image",
]

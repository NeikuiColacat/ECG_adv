from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from util.lead_utils import PTBXL_LEADS


def make_ecg_figure(
    signal_ct: np.ndarray,
    sample_rate: float = 100.0,
    title: str = "",
    *,
    style: str = "paper",
):
    arr = np.asarray(signal_ct, dtype=np.float32)
    if arr.shape[0] != 12 and arr.shape[1] == 12:
        arr = arr.T
    if style == "grid":
        return _make_grid_figure(arr, sample_rate=sample_rate, title=title)
    return _make_paper_figure(arr, sample_rate=sample_rate, title=title)


def _make_grid_figure(arr: np.ndarray, sample_rate: float = 100.0, title: str = ""):
    time = np.arange(arr.shape[1]) / float(sample_rate)
    fig, axes = plt.subplots(6, 2, figsize=(14, 10), sharex=True)
    axes = axes.reshape(-1)
    for i, ax in enumerate(axes[:12]):
        ax.plot(time, arr[i], linewidth=0.8, color="#1f6f78")
        ax.set_title(PTBXL_LEADS[i], loc="left", fontsize=9)
        ax.grid(True, linewidth=0.3, alpha=0.45)
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def _make_paper_figure(
    arr: np.ndarray,
    sample_rate: float = 100.0,
    title: str = "",
    row_gap: float = 6.0,
    target_peak: float = 1.8,
):
    """Return the thesis-style 12-lead ECG gallery figure used in the app."""
    arr = np.nan_to_num(arr.astype(np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
    denom = float(np.percentile(np.abs(arr), 95))
    if denom > 1e-8:
        arr *= float(target_peak) / denom
    clip_abs = row_gap * 0.42
    arr = np.clip(arr, -clip_abs, clip_abs)

    length = arr.shape[1]
    time = np.arange(length, dtype=np.float32) / float(sample_rate)
    offsets = np.arange(12, dtype=np.float32)[::-1] * float(row_gap)
    fig, ax = plt.subplots(figsize=(16, 10.5))
    for i, lead_name in enumerate(PTBXL_LEADS):
        ax.plot(time, arr[i] + offsets[i], color="#111111", linewidth=0.85)
        ax.text(
            -0.015 * max(float(time[-1]), 1.0),
            offsets[i],
            lead_name,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#202020",
        )

    ax.set_xlim(0, float(time[-1]) if length > 1 else 1.0)
    ax.set_ylim(offsets[-1] - row_gap * 0.6, offsets[0] + row_gap * 0.6)
    ax.set_yticks(offsets)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("time (s)")
    if title:
        ax.set_title(title, fontsize=13, pad=12)

    x_minor = np.arange(0, float(time[-1]) + 0.2, 0.2)
    x_major = np.arange(0, float(time[-1]) + 1.0, 1.0)
    for x in x_minor:
        ax.axvline(x, color="#f6cbd1", linewidth=0.45, zorder=0)
    for x in x_major:
        ax.axvline(x, color="#e8a5ae", linewidth=0.75, zorder=0)
    for base in offsets:
        ax.axhline(base, color="#f0b6bf", linewidth=0.55, zorder=0)
        ax.axhline(base + clip_abs, color="#f8d9de", linewidth=0.35, zorder=0)
        ax.axhline(base - clip_abs, color="#f8d9de", linewidth=0.35, zorder=0)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig

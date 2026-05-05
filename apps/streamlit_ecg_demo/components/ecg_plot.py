from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from util.lead_utils import PTBXL_LEADS


def make_ecg_figure(signal_ct: np.ndarray, sample_rate: float = 100.0, title: str = ""):
    arr = np.asarray(signal_ct, dtype=np.float32)
    if arr.shape[0] != 12 and arr.shape[1] == 12:
        arr = arr.T
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


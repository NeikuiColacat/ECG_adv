"""
ECG 样本可视化 + 合理性检查工具（通用 debug 场景）。

三类输入场景:
- ECG 数据增强算子输出
- ECGTwin 条件生成样本 (util/ecgtwin_utils.py)
- AdvDiff 对抗样本 (adversarial/adv_generate.py)

API:
    plot_ecg(signal, sample_rate, save_path, ...) -> Path
    plot_comparison(signals, labels, sample_rate, save_path, ...) -> Path
    sanity_check(signal, sample_rate, lead_order=...) -> dict
    plot_with_report(signal, sample_rate, save_path, ...) -> (Path, dict)

专用对抗样本保存（GT-Target-Pred 命名）见 util/save_tool.py，本模块不覆盖。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Sequence, Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from util.lead_utils import ECGTWIN_LEADS, PTBXL_LEADS

LEAD_ORDERS: dict[str, list[str]] = {
    "ecgtwin": list(ECGTWIN_LEADS),
    "ptbxl": list(PTBXL_LEADS),
}

SignalLike = Union[torch.Tensor, np.ndarray]
LeadOrder = Literal["ptbxl", "ecgtwin"]


def _import_ecg_plot():
    """Pip 安装优先；失败时回退到 ECGTwin 的 sys.path 注入。返回 None 表示不可用。"""
    try:
        import ecg_plot
        return ecg_plot
    except ImportError:
        pass
    ecgtwin_root = Path(__file__).resolve().parent.parent / "model" / "ECGTwin"
    if ecgtwin_root.exists():
        sys.path.insert(0, str(ecgtwin_root))
        try:
            import ecg_plot
            return ecg_plot
        except ImportError:
            pass
    return None


def _to_channels_first_2d(sig: SignalLike) -> np.ndarray:
    """接受 Tensor/ndarray，返回 (12, L) float32 ndarray；batch 维 >1 会报错。"""
    if isinstance(sig, torch.Tensor):
        sig = sig.detach().cpu().numpy()
    elif not isinstance(sig, np.ndarray):
        sig = np.asarray(sig)

    if sig.ndim == 3:
        if sig.shape[0] != 1:
            raise ValueError(f"batch 维必须 ==1，得到 shape={sig.shape}")
        sig = sig[0]
    if sig.ndim != 2:
        raise ValueError(f"期望 2D 信号，得到 shape={sig.shape}")

    if sig.shape[0] == 12 and sig.shape[1] != 12:
        arr = sig
    elif sig.shape[1] == 12 and sig.shape[0] != 12:
        arr = sig.T
    elif sig.shape == (12, 12):
        raise ValueError("12x12 shape 歧义，无法判断导联维")
    else:
        raise ValueError(f"无法判断导联维，shape={sig.shape}，两维均非 12")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _validate_lead_order(lead_order: str) -> list[str]:
    if lead_order not in LEAD_ORDERS:
        raise ValueError(
            f"unknown lead_order={lead_order!r}；支持: {list(LEAD_ORDERS)}"
        )
    return LEAD_ORDERS[lead_order]


def _validate_sample_rate(sample_rate: float) -> float:
    if sample_rate is None or sample_rate <= 0:
        raise ValueError(f"sample_rate 必须 >0，得到 {sample_rate}")
    return float(sample_rate)


def _lead_idx(lead_order: str, name: str) -> int:
    leads = LEAD_ORDERS[lead_order]
    target = name.upper() if lead_order == "ptbxl" else name
    return leads.index(target)


def _align_length(signals: list[np.ndarray]) -> tuple[list[np.ndarray], bool]:
    """把所有信号对齐到第一条的长度。短则线性插值拉伸，长则截断。返回 (对齐后, 是否触发对齐)."""
    target_len = signals[0].shape[1]
    aligned = [signals[0]]
    changed = False
    for s in signals[1:]:
        L = s.shape[1]
        if L == target_len:
            aligned.append(s)
            continue
        changed = True
        if L > target_len:
            aligned.append(s[:, :target_len])
        else:
            x_old = np.linspace(0.0, 1.0, L)
            x_new = np.linspace(0.0, 1.0, target_len)
            resampled = np.stack([np.interp(x_new, x_old, s[ch]) for ch in range(12)])
            aligned.append(resampled.astype(np.float32))
    return aligned, changed


def plot_ecg(
    signal: SignalLike,
    sample_rate: float,
    save_path: Union[str, Path],
    *,
    title: str | None = None,
    lead_order: LeadOrder = "ecgtwin",
    columns: int = 4,
    row_height: float = 2.5,
    show_grid: bool = True,
    engine: Literal["auto", "ecgplot", "matplotlib"] = "auto",
) -> Path:
    """画单个 12-lead ECG 样本并保存为 PNG。

    shape 自动检测: (12,L) / (L,12) / (1,12,L) / (1,L,12) 均可。
    默认 3x4 网格（12/columns=3 行）。

    engine:
      - "auto": 优先 ecg_plot 库（医疗纸带样式），没装则回退 matplotlib 子图
      - "ecgplot": 强制用 ecg_plot 库（适合打印/医生审阅，图会很宽）
      - "matplotlib": 强制 3x4 子图（屏幕预览/调试友好）
    """
    sr = _validate_sample_rate(sample_rate)
    leads = _validate_lead_order(lead_order)
    arr = _to_channels_first_2d(signal)  # (12, L)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ecg_plot = _import_ecg_plot() if engine in ("auto", "ecgplot") else None
    if engine == "ecgplot" and ecg_plot is None:
        raise RuntimeError("engine='ecgplot' requested but ecg_plot is not importable")
    if ecg_plot is not None:
        ecg_plot.plot(
            arr,
            sample_rate=sr,
            title=title or "",
            lead_index=leads,
            columns=columns,
            row_height=row_height,
            show_grid=show_grid,
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(plt.gcf())
    else:
        _plot_ecg_matplotlib_fallback(
            arr, sr, leads, columns, row_height, title, show_grid, save_path
        )
    return save_path


def _plot_ecg_matplotlib_fallback(
    arr: np.ndarray,
    sample_rate: float,
    leads: list[str],
    columns: int,
    row_height: float,
    title: str | None,
    show_grid: bool,
    save_path: Path,
) -> None:
    rows = int(np.ceil(12 / columns))
    L = arr.shape[1]
    t = np.arange(L) / sample_rate
    fig, axes = plt.subplots(
        rows, columns, figsize=(3.5 * columns, row_height * rows), squeeze=False
    )
    for i in range(12):
        ax = axes[i // columns][i % columns]
        ax.plot(t, arr[i], color="black", linewidth=0.7)
        ax.set_title(leads[i], fontsize=9, pad=2)
        ax.set_xlim(0, t[-1] if L > 1 else 1)
        if show_grid:
            ax.grid(True, which="major", alpha=0.4, color="pink")
    for i in range(12, rows * columns):
        axes[i // columns][i % columns].set_visible(False)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ecg_ecgtwin_style(
    signal: SignalLike,
    save_path: Union[str, Path],
    *,
    sample_rate: float = 102.4,
    lead_order: LeadOrder = "ecgtwin",
    row_height: float = 4,
    title: str | None = None,
    per_lead_norm: bool = False,
    amplitude_scale: float = 1.0,
) -> Path:
    """One-column rhythm-strip ECG plot. Assumes input is in mV.

    per_lead_norm: peak-rescale each lead independently to fit its row (kills
    inter-lead amplitude comparisons; use when input is z-scored, not mV).
    amplitude_scale: uniform multiplier (preserves cross-lead relationships).
    """
    ecg_plot = _import_ecg_plot()
    if ecg_plot is None:
        raise RuntimeError(
            "ecg_plot not importable; cannot use ECGTwin-style plot. "
            "Install `ecg_plot` or fall back to plot_ecg(engine='matplotlib')."
        )

    sr = _validate_sample_rate(sample_rate)
    leads = _validate_lead_order(lead_order)
    arr = _to_channels_first_2d(signal)  # (12, L)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if per_lead_norm:
        # Scale each lead so its |peak| lands at 80% of the row half-height.
        # A std-based rescale (std → 0.4 "mV") is insufficient because typical
        # R peaks are 4–8σ; those still overflow a 2 mV row by 30–60%. Using
        # peak-based scaling guarantees no bleed into adjacent leads while
        # keeping a 20% margin.
        row_half = float(row_height) / 2.0
        peaks = np.max(np.abs(arr), axis=1, keepdims=True)
        peaks = np.where(peaks > 1e-8, peaks, 1.0)
        arr = arr / peaks * (0.8 * row_half)
    if amplitude_scale != 1.0:
        arr = arr * float(amplitude_scale)

    ecg_plot.plot(
        arr,
        sample_rate=sr,
        title=title if title is not None else "",
        lead_index=leads,
        columns=1,
        row_height=row_height,
    )
    # Thin out the x-axis tick labels (0.2s major -> 1s major; 0.2s as minor).
    from matplotlib.ticker import AutoMinorLocator
    ax = plt.gca()
    secs = arr.shape[1] / sr
    ax.set_xticks(np.arange(0, secs + 1e-6, 1.0))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis="x", labelsize=8)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(plt.gcf())
    return save_path


def plot_ecg_ecgtwin_gallery(
    signal: SignalLike,
    save_path: Union[str, Path],
    *,
    sample_rate: float = 100.0,
    lead_order: LeadOrder = "ptbxl",
    title: str | None = None,
    row_gap: float = 6.0,
    target_peak: float = 1.8,
    scale_mode: Literal["global_p95", "global_peak", "per_lead_peak", "none"] = "global_p95",
    clip_fraction: float = 0.42,
    figsize: tuple[float, float] = (18.0, 14.0),
    linewidth: float = 0.8,
    show_grid: bool = True,
    grid_seconds: float = 0.2,
    major_seconds: float = 1.0,
    color: str = "black",
) -> Path:
    """ECGTwin-gallery-style 12-lead rhythm strip in one large figure.

    This function is intentionally independent of the `ecg_plot` package so the
    vertical spacing is deterministic. Each lead is drawn on a separate
    horizontal baseline. Signals are optionally rescaled and then clipped to a
    fixed fraction of `row_gap`, preventing tall R peaks from entering adjacent
    leads.

    Recommended for generated/z-scored ECG debug figures:
      plot_ecg_ecgtwin_gallery(sig, out, lead_order="ptbxl",
                               row_gap=7.0, target_peak=2.0,
                               scale_mode="global_p95")

    scale_mode:
      - global_p95: one global p95 amplitude scale; preserves inter-lead ratios
        for most signal mass while keeping outliers controlled.
      - global_peak: one global max-abs scale; fully preserves ratios.
      - per_lead_peak: each lead peak-normalized; best for visual inspection,
        but cross-lead amplitude comparison is not valid.
      - none: no rescaling; only clipping applies.
    """
    sr = _validate_sample_rate(sample_rate)
    leads = _validate_lead_order(lead_order)
    arr = _to_channels_first_2d(signal).astype(np.float32, copy=True)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    row_gap = float(row_gap)
    if row_gap <= 0:
        raise ValueError(f"row_gap must be positive, got {row_gap}")
    clip_fraction = float(clip_fraction)
    if not (0 < clip_fraction <= 0.5):
        raise ValueError(f"clip_fraction must be in (0, 0.5], got {clip_fraction}")

    finite = np.isfinite(arr)
    arr = np.where(finite, arr, 0.0)

    if scale_mode == "global_p95":
        denom = float(np.percentile(np.abs(arr), 95))
        if denom > 1e-8:
            arr *= float(target_peak) / denom
    elif scale_mode == "global_peak":
        denom = float(np.max(np.abs(arr)))
        if denom > 1e-8:
            arr *= float(target_peak) / denom
    elif scale_mode == "per_lead_peak":
        denom = np.max(np.abs(arr), axis=1, keepdims=True)
        denom = np.where(denom > 1e-8, denom, 1.0)
        arr = arr / denom * float(target_peak)
    elif scale_mode == "none":
        pass
    else:
        raise ValueError(f"unknown scale_mode={scale_mode!r}")

    clip_abs = row_gap * clip_fraction
    arr = np.clip(arr, -clip_abs, clip_abs)

    length = arr.shape[1]
    t = np.arange(length, dtype=np.float32) / sr
    offsets = np.arange(12, dtype=np.float32)[::-1] * row_gap

    fig, ax = plt.subplots(figsize=figsize)
    for i, lead_name in enumerate(leads):
        y = arr[i] + offsets[i]
        ax.plot(t, y, color=color, linewidth=linewidth)
        ax.text(
            -0.015 * max(t[-1], 1.0),
            offsets[i],
            lead_name,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlim(0, t[-1] if length > 1 else 1)
    ax.set_ylim(offsets[-1] - row_gap * 0.6, offsets[0] + row_gap * 0.6)
    ax.set_yticks(offsets)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("time (s)")
    if title:
        ax.set_title(title, fontsize=13, pad=12)

    if show_grid:
        x_minor = np.arange(0, t[-1] + grid_seconds, grid_seconds)
        x_major = np.arange(0, t[-1] + major_seconds, major_seconds)
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
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_comparison(
    signals: Sequence[SignalLike],
    labels: Sequence[str],
    sample_rate: float,
    save_path: Union[str, Path],
    *,
    mode: Literal["overlay", "stacked"] = "overlay",
    lead_order: LeadOrder = "ecgtwin",
    alpha_sequence: Sequence[float] | None = None,
    row_height: float = 2.2,
) -> Path:
    """对比多条 ECG（clean vs augmented / adv），用于判断扰动是否合理。

    mode="overlay": 12 个子图，每格叠所有版本（不同颜色/alpha）。
    mode="stacked": 每个版本占一行 12-lead（行 x 4 列）。
    """
    if len(signals) != len(labels):
        raise ValueError(
            f"signals ({len(signals)}) 与 labels ({len(labels)}) 数量必须一致"
        )
    if len(signals) == 0:
        raise ValueError("signals 不能为空")
    if mode not in ("overlay", "stacked"):
        raise ValueError(f"mode 必须为 'overlay' 或 'stacked'，得到 {mode!r}")

    sr = _validate_sample_rate(sample_rate)
    leads = _validate_lead_order(lead_order)
    arrs = [_to_channels_first_2d(s) for s in signals]
    arrs, length_changed = _align_length(arrs)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    L = arrs[0].shape[1]
    t = np.arange(L) / sr
    alphas = (
        list(alpha_sequence)
        if alpha_sequence is not None
        else [max(0.3, 1.0 - 0.2 * i) for i in range(len(arrs))]
    )
    cmap = plt.get_cmap("tab10")
    warn = " [aligned]" if length_changed else ""

    if mode == "overlay":
        rows, cols = 3, 4
        fig, axes = plt.subplots(
            rows, cols, figsize=(3.5 * cols, row_height * rows), squeeze=False
        )
        for i in range(12):
            ax = axes[i // cols][i % cols]
            for k, arr in enumerate(arrs):
                ax.plot(
                    t,
                    arr[i],
                    color=cmap(k % 10),
                    alpha=alphas[k],
                    linewidth=0.8,
                    label=labels[k] if i == 0 else None,
                )
            ax.set_title(leads[i], fontsize=9, pad=2)
            ax.set_xlim(0, t[-1] if L > 1 else 1)
            ax.grid(True, alpha=0.3, color="pink")
        axes[0][0].legend(fontsize=7, loc="upper right")
        fig.suptitle(f"Comparison (overlay){warn}", fontsize=11)
    else:  # stacked
        cols = 4
        rows_per_block = 3
        n = len(arrs)
        fig, axes = plt.subplots(
            rows_per_block * n,
            cols,
            figsize=(3.5 * cols, row_height * rows_per_block * n),
            squeeze=False,
        )
        for k, arr in enumerate(arrs):
            for i in range(12):
                row = k * rows_per_block + (i // cols)
                ax = axes[row][i % cols]
                ax.plot(t, arr[i], color=cmap(k % 10), linewidth=0.7)
                prefix = f"[{labels[k]}] " if i == 0 and (i % cols) == 0 else ""
                ax.set_title(f"{prefix}{leads[i]}", fontsize=8, pad=2)
                ax.set_xlim(0, t[-1] if L > 1 else 1)
                ax.grid(True, alpha=0.3, color="pink")
        fig.suptitle(f"Comparison (stacked){warn}", fontsize=11)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def sanity_check(
    signal: SignalLike,
    sample_rate: float,
    lead_order: LeadOrder = "ecgtwin",
) -> dict:
    """跑一组客观可验证的 ECG 合理性检查。返回 JSON 可序列化 dict。

    不做医学诊断（ST/QT/节律），只标记物理/数值异常 + Einthoven 定律残差 + HR 估计。
    """
    sr = _validate_sample_rate(sample_rate)
    leads = _validate_lead_order(lead_order)
    arr = _to_channels_first_2d(signal)  # (12, L)

    report: dict = {
        "has_nan": bool(np.isnan(arr).any()),
        "has_inf": bool(np.isinf(arr).any()),
        "flatline_leads": [],
        "saturated_leads": [],
        "dc_offset_per_lead": {},
        "amplitude_p2p_per_lead": {},
        "einthoven_residual": float("nan"),
        "avR_residual": float("nan"),
        "hr_estimate_bpm": None,
        "warnings": [],
    }

    if report["has_nan"]:
        report["warnings"].append("signal contains NaN")
    if report["has_inf"]:
        report["warnings"].append("signal contains Inf")

    arr_finite = np.where(np.isfinite(arr), arr, 0.0)

    for i, name in enumerate(leads):
        lead = arr_finite[i]
        std = float(lead.std())
        if std < 1e-4:
            report["flatline_leads"].append(name)
        lead_max, lead_min = float(lead.max()), float(lead.min())
        if lead_max != lead_min:
            frac_at_max = float((lead == lead_max).mean())
            frac_at_min = float((lead == lead_min).mean())
            if frac_at_max > 0.95 or frac_at_min > 0.95:
                report["saturated_leads"].append(name)
        report["dc_offset_per_lead"][name] = float(lead.mean())
        report["amplitude_p2p_per_lead"][name] = float(lead_max - lead_min)

    if report["flatline_leads"]:
        report["warnings"].append(
            f"flatline leads: {report['flatline_leads']}"
        )
    if report["saturated_leads"]:
        report["warnings"].append(
            f"saturated leads: {report['saturated_leads']}"
        )
    big_dc = [
        n for n, v in report["dc_offset_per_lead"].items() if abs(v) > 0.5
    ]
    if big_dc:
        report["warnings"].append(f"large DC offset (|mean|>0.5): {big_dc}")

    # Einthoven: II ≈ I + III
    i_idx = _lead_idx(lead_order, "I")
    ii_idx = _lead_idx(lead_order, "II")
    iii_idx = _lead_idx(lead_order, "III")
    denom = float(np.mean(np.abs(arr_finite[ii_idx]))) + 1e-8
    eres = float(
        np.mean(np.abs(arr_finite[ii_idx] - (arr_finite[i_idx] + arr_finite[iii_idx])))
        / denom
    )
    report["einthoven_residual"] = eres
    if eres > 0.2:
        report["warnings"].append(
            f"Einthoven II=I+III violated (residual={eres:.3f})"
        )

    # aVR ≈ -(I+II)/2
    avr_name = "AVR" if lead_order == "ptbxl" else "aVR"
    avr_idx = _lead_idx(lead_order, avr_name)
    avr_denom = float(np.mean(np.abs(arr_finite[avr_idx]))) + 1e-8
    ares = float(
        np.mean(
            np.abs(arr_finite[avr_idx] + (arr_finite[i_idx] + arr_finite[ii_idx]) / 2.0)
        )
        / avr_denom
    )
    report["avR_residual"] = ares
    if ares > 0.3:
        report["warnings"].append(
            f"aVR = -(I+II)/2 violated (residual={ares:.3f})"
        )

    # HR via find_peaks on lead II (z-scored)
    lead_ii = arr_finite[ii_idx]
    lead_ii_std = float(lead_ii.std())
    if lead_ii_std > 1e-4:
        lead_ii_z = (lead_ii - lead_ii.mean()) / lead_ii_std
        min_distance = max(1, int(0.25 * sr))  # 240 bpm 上限
        peaks, _ = find_peaks(lead_ii_z, height=0.5, distance=min_distance)
        if len(peaks) >= 3:
            duration_sec = (peaks[-1] - peaks[0]) / sr
            if duration_sec > 0:
                hr = 60.0 * (len(peaks) - 1) / duration_sec
                report["hr_estimate_bpm"] = float(hr)
                if hr < 30 or hr > 200:
                    report["warnings"].append(
                        f"HR {hr:.0f} bpm outside physiological range [30,200]"
                    )

    return report


def plot_with_report(
    signal: SignalLike,
    sample_rate: float,
    save_path: Union[str, Path],
    *,
    title_prefix: str = "",
    lead_order: LeadOrder = "ecgtwin",
    columns: int = 4,
    row_height: float = 2.5,
    engine: Literal["auto", "ecgplot", "matplotlib"] = "auto",
) -> tuple[Path, dict]:
    """跑 sanity_check，把摘要叠到标题里，再调 plot_ecg 保存图像。返回 (path, report)."""
    report = sanity_check(signal, sample_rate, lead_order=lead_order)
    hr = report["hr_estimate_bpm"]
    hr_str = f"HR≈{hr:.0f}bpm" if hr is not None else "HR=N/A"
    n_warn = len(report["warnings"])
    title = f"{title_prefix} | {hr_str} | {n_warn} warnings".strip(" |")
    path = plot_ecg(
        signal,
        sample_rate,
        save_path,
        title=title,
        lead_order=lead_order,
        columns=columns,
        row_height=row_height,
        engine=engine,
    )
    return path, report

#!/usr/bin/env python3
"""Generate PN2021-C corruption realism diagnostics for a frozen profile."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.data.kshot import load_ref_record_ids_by_center
from ecg_adv_gen.data.pn2021 import (
    load_native_raw_first_center as _load_native_raw_first_center,
    load_raw_first_center as _load_raw_first_center,
)
from ecg_adv_gen.evaluation.pn2021_corruptions import (
    filter_record_indices as _filter_indices,
    stable_corruption_seed,
)
from ecg_adv_gen.evaluation.pn2021c import (
    build_corruption_op as _build_corruption_op,
    load_custom_severity_profile as _load_custom_severity_profile,
)
from ecg_adv_gen.data.waveform_datasets import (
    ecg1000_ct_to_ecgfounder_5000_no_zscore,
)
from ecg_adv_gen.labels import get_super5_scheme
from util.ecg_viz import plot_comparison, plot_ecg_ecgtwin_gallery, sanity_check


PUBLIC_SEVERITY = 5
DEFAULT_DATA_ROOT = Path("/home/linbinhao/ECG_adv_data")
DEFAULT_PROFILE = (
    Path("/home/linbinhao/ECG_adv_Gen")
    / "configs/corruption_profiles/pn2021c_dual_model_10to15pp_v1.yaml"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_DATA_ROOT
    / "runs/pn2021c_dual_model_10to15pp_sweep_20260618/diagnostics"
)


def _to_ct(signal: np.ndarray) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D signal, got shape={arr.shape}")
    if arr.shape[0] == 12:
        return arr
    if arr.shape[1] == 12:
        return arr.T
    raise ValueError(f"expected one ECG dimension to be 12, got shape={arr.shape}")


def _rms(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return float(np.sqrt(np.mean(np.square(arr))))


def _psd_band_fraction(signal_ct: np.ndarray, sample_rate_hz: float, lo: float, hi: float) -> float:
    arr = np.asarray(signal_ct, dtype=np.float32)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = arr - arr.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(arr, axis=1)) ** 2
    freqs = np.fft.rfftfreq(arr.shape[1], d=1.0 / float(sample_rate_hz))
    total = float(np.sum(spectrum)) + 1e-12
    mask = (freqs >= float(lo)) & (freqs <= float(hi))
    return float(np.sum(spectrum[:, mask]) / total)


def _near_frequency_fraction(signal_ct: np.ndarray, sample_rate_hz: float, freq_hz: float, width_hz: float = 1.0) -> float:
    nyquist = float(sample_rate_hz) / 2.0
    target = min(float(freq_hz), nyquist)
    return _psd_band_fraction(
        signal_ct,
        sample_rate_hz,
        max(0.0, target - float(width_hz)),
        min(nyquist, target + float(width_hz)),
    )


def compute_signal_diagnostics(
    clean_signal: np.ndarray,
    corrupted_signal: np.ndarray,
    *,
    sample_rate_hz: float,
) -> dict[str, Any]:
    clean_ct = _to_ct(clean_signal)
    corrupt_ct = _to_ct(corrupted_signal)
    if clean_ct.shape != corrupt_ct.shape:
        length = min(clean_ct.shape[1], corrupt_ct.shape[1])
        clean_ct = clean_ct[:, :length]
        corrupt_ct = corrupt_ct[:, :length]

    diff = corrupt_ct - clean_ct
    clean_rms = _rms(clean_ct)
    diff_rms = _rms(diff)
    corrupt_finite = np.where(np.isfinite(corrupt_ct), corrupt_ct, 0.0)
    clean_p2p = np.ptp(np.where(np.isfinite(clean_ct), clean_ct, 0.0), axis=1)
    corrupt_p2p = np.ptp(corrupt_finite, axis=1)
    corrupt_std = np.std(corrupt_finite, axis=1)
    masked = (clean_p2p > 1e-4) & (corrupt_p2p < 1e-4)
    flatline = corrupt_std < 1e-4
    nyquist = float(sample_rate_hz) / 2.0

    return {
        "sample_rate_hz": float(sample_rate_hz),
        "length": int(clean_ct.shape[1]),
        "has_nan": bool(np.isnan(corrupt_ct).any()),
        "has_inf": bool(np.isinf(corrupt_ct).any()),
        "rms_clean": clean_rms,
        "rms_corrupted": _rms(corrupt_ct),
        "rms_diff": diff_rms,
        "rms_ratio": float(diff_rms / max(clean_rms, 1e-8)),
        "max_abs_diff": float(np.max(np.abs(np.where(np.isfinite(diff), diff, 0.0)))),
        "clean_max_abs": float(np.max(np.abs(np.where(np.isfinite(clean_ct), clean_ct, 0.0)))),
        "corrupted_max_abs": float(np.max(np.abs(corrupt_finite))),
        "masked_lead_count": int(np.sum(masked)),
        "corrupt_flatline_lead_count": int(np.sum(flatline)),
        "low_freq_0p03_1hz_fraction": _psd_band_fraction(corrupt_ct, sample_rate_hz, 0.03, min(1.0, nyquist)),
        "emg_high_freq_fraction": _psd_band_fraction(corrupt_ct, sample_rate_hz, min(20.0, nyquist), nyquist),
        "powerline_50hz_fraction": _near_frequency_fraction(corrupt_ct, sample_rate_hz, 50.0),
        "powerline_60hz_fraction": _near_frequency_fraction(corrupt_ct, sample_rate_hz, 60.0),
    }


def _load_profile(path: Path, profile_name: str) -> tuple[dict[str, dict[int, dict]], dict[str, Any]]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"invalid profile file: {path}")
    profile = _load_custom_severity_profile(str(path), profile_name)
    metadata = data.get("metadata", {}) or {}
    return profile, metadata


def _build_loader_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        pn2021_root=str(args.pn2021_root),
        scheme=str(args.scheme),
        crop_len=1000,
    )


def _apply_custom_corruption(
    clean_ct: np.ndarray,
    *,
    operator: str,
    profile: dict[str, dict[int, dict]],
    sample_rate_hz: float,
    seed: int,
    seed_parts: tuple[object, ...],
) -> np.ndarray:
    sample_seed = stable_corruption_seed(int(seed), *seed_parts)
    np.random.seed(sample_seed)
    random.seed(sample_seed)
    torch.manual_seed(sample_seed)
    op = _build_corruption_op(
        operator,
        PUBLIC_SEVERITY,
        "custom",
        sample_rate_hz=float(sample_rate_hz),
        severity_profile_params=profile,
    )
    with torch.no_grad():
        out = op(torch.from_numpy(np.ascontiguousarray(clean_ct)).float())
    return out.detach().cpu().numpy().astype(np.float32, copy=False)


def _raw_first_samples(args: argparse.Namespace, exclude_ids: set[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scheme = get_super5_scheme()
    loader_args = _build_loader_args(args)
    signals, _labels, record_ids, _metadata, _kind = _load_raw_first_center(loader_args, scheme, args.center)
    indices = _filter_indices(record_ids, exclude_ids, args.max_source_records)
    return signals, record_ids, indices


def _native_samples(args: argparse.Namespace, exclude_ids: set[str]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    scheme = get_super5_scheme()
    loader_args = _build_loader_args(args)
    signals, _labels, record_ids, _metadata, _kind, sample_rates = _load_native_raw_first_center(
        loader_args,
        scheme,
        args.center,
    )
    indices = _filter_indices(record_ids, exclude_ids, args.max_source_records)
    return signals, sample_rates, record_ids, indices


def _choose_indices(indices: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    if len(indices) <= n_samples:
        return indices
    rng = np.random.default_rng(int(seed))
    picked = rng.choice(indices, size=int(n_samples), replace=False)
    return np.asarray(sorted(int(i) for i in picked), dtype=np.int64)


def _clean_ct_for_mode(
    *,
    mode: str,
    raw_signals: np.ndarray,
    native_signals: list[np.ndarray] | None,
    sample_rates: np.ndarray | None,
    real_idx: int,
) -> tuple[np.ndarray, float]:
    if mode == "raw_first":
        return np.asarray(raw_signals[real_idx], dtype=np.float32).T, 100.0
    if mode == "bottleneck5000":
        sig_ct = torch.from_numpy(np.ascontiguousarray(raw_signals[real_idx].T)).float()
        return ecg1000_ct_to_ecgfounder_5000_no_zscore(sig_ct).detach().cpu().numpy(), 500.0
    if mode == "native_raw_first":
        if native_signals is None or sample_rates is None:
            raise ValueError("native signals are required for native_raw_first diagnostics")
        return np.asarray(native_signals[real_idx], dtype=np.float32).T, float(sample_rates[real_idx])
    raise ValueError(f"unknown corruption input mode: {mode}")


def run_diagnostics(args: argparse.Namespace) -> list[dict[str, Any]]:
    profile, metadata = _load_profile(args.profile_file, args.profile_name)
    input_modes = metadata.get("input_modes", {}) or {}
    operators = list(args.operators or profile.keys())
    models = list(args.models)
    exclude_by_center = load_ref_record_ids_by_center([args.ref_meta_json]) if args.ref_meta_json else {}
    exclude_ids = exclude_by_center.get(args.center, set())
    raw_signals, raw_record_ids, raw_indices = _raw_first_samples(args, exclude_ids)

    needs_native = any(
        input_modes.get(op, {}).get(model) == "native_raw_first"
        for op in operators
        for model in models
    )
    native_signals = sample_rates = native_record_ids = native_indices = None
    if needs_native:
        native_signals, sample_rates, native_record_ids, native_indices = _native_samples(args, exclude_ids)

    rows: list[dict[str, Any]] = []
    for operator in operators:
        for model in models:
            mode = input_modes.get(operator, {}).get(model)
            if mode is None:
                mode = "raw_first" if model == "effnet" else "bottleneck5000"
            source_indices = native_indices if mode == "native_raw_first" else raw_indices
            record_ids = native_record_ids if mode == "native_raw_first" else raw_record_ids
            for real_idx in _choose_indices(np.asarray(source_indices), args.n_samples, args.seed):
                record_id = str(record_ids[int(real_idx)])
                clean_ct, sample_rate_hz = _clean_ct_for_mode(
                    mode=mode,
                    raw_signals=raw_signals,
                    native_signals=native_signals,
                    sample_rates=sample_rates,
                    real_idx=int(real_idx),
                )
                corrupt_ct = _apply_custom_corruption(
                    clean_ct,
                    operator=operator,
                    profile=profile,
                    sample_rate_hz=sample_rate_hz,
                    seed=args.seed,
                    seed_parts=(mode, operator, PUBLIC_SEVERITY, int(real_idx)),
                )
                stats = compute_signal_diagnostics(
                    clean_ct,
                    corrupt_ct,
                    sample_rate_hz=sample_rate_hz,
                )
                rel_dir = Path(model) / operator / mode / record_id
                out_dir = args.output_dir / rel_dir
                clean_png = out_dir / "clean_gallery.png"
                corrupt_png = out_dir / "corrupted_gallery.png"
                compare_png = out_dir / "comparison_overlay.png"
                plot_ecg_ecgtwin_gallery(
                    clean_ct,
                    clean_png,
                    sample_rate=sample_rate_hz,
                    lead_order="ptbxl",
                    title=f"{model} {operator} {mode} clean {record_id}",
                )
                plot_ecg_ecgtwin_gallery(
                    corrupt_ct,
                    corrupt_png,
                    sample_rate=sample_rate_hz,
                    lead_order="ptbxl",
                    title=f"{model} {operator} {mode} corrupted {record_id}",
                )
                plot_comparison(
                    [clean_ct, corrupt_ct],
                    ["clean", "corrupted"],
                    sample_rate_hz,
                    compare_png,
                    mode="overlay",
                    lead_order="ptbxl",
                )
                sanity = sanity_check(corrupt_ct, sample_rate_hz, lead_order="ptbxl")
                row = {
                    "model": model,
                    "operator": operator,
                    "corruption_input": mode,
                    "center": args.center,
                    "record_id": record_id,
                    "real_index": int(real_idx),
                    "clean_png": str(clean_png),
                    "corrupted_png": str(corrupt_png),
                    "comparison_png": str(compare_png),
                    "sanity_warnings_json": json.dumps(sanity.get("warnings", []), sort_keys=True),
                    **stats,
                }
                rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "diagnostics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[diagnostics] rows={len(rows)} csv={csv_path}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_file", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--profile_name", default="dual_model_10to15pp_v1")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--center", default="cpsc_2018")
    parser.add_argument("--scheme", default="super5", choices=["super5"])
    parser.add_argument("--operators", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=["effnet", "ecgfounder"])
    parser.add_argument("--n_samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--max_source_records", type=int, default=2000)
    parser.add_argument("--pn2021_root", type=Path, default=DEFAULT_DATA_ROOT / "physionet2021")
    parser.add_argument(
        "--ref_meta_json",
        type=Path,
        default=(
            DEFAULT_DATA_ROOT
            / "paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets"
            / "cpsc_2018/k500_seed20260531/cpsc_2018_real_k500_seed20260531.ref_meta.json"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_diagnostics(args)


if __name__ == "__main__":
    main()

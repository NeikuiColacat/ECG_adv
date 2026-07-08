"""Package-owned waveform datasets for PN2021 evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ecg_adv_gen.evaluation.pn2021c_protocol import LOCKED_ECGFOUNDER_CORRUPTION_INPUT
from ecg_adv_gen.evaluation.pn2021c import (
    PUBLIC_TO_INTERNAL_SEVERITY,
    apply_corruption_sequence,
)
from ecg_adv_gen.models.ecgfounder_torch import (
    ecg1000_to_ecgfounder_input,
    global_zscore_torch,
    stabilize_ecg_torch,
)
from ecg_adv_gen.preprocessing import crop_signal_tc, reorder_leads_tc, unified_preprocess_to_1000


def has_input_stabilizer(config: dict | None) -> bool:
    config = dict(config or {})
    return bool(
        config.get("bandpass_low_hz") is not None
        or config.get("bandpass_high_hz") is not None
        or config.get("repair_flat_leads")
        or config.get("clip_abs") is not None
        or config.get("renorm_after_stabilizer")
    )


def apply_effnet_input_stabilizer(ecg_ct: torch.Tensor, config: dict | None) -> torch.Tensor:
    """Apply optional EfficientNet ECG input stabilizer to ``(C,T)`` or ``(B,C,T)`` tensors."""
    config = dict(config or {})
    if not has_input_stabilizer(config):
        return ecg_ct
    squeeze = False
    x = ecg_ct
    if x.ndim == 2:
        x = x.unsqueeze(0)
        squeeze = True
    if x.ndim != 3:
        raise ValueError(f"expected ECG tensor with shape (C,T) or (B,C,T), got {tuple(ecg_ct.shape)}")
    y = stabilize_ecg_torch(
        x,
        sample_rate_hz=float(config.get("sample_rate_hz", 100.0)),
        bandpass_low_hz=config.get("bandpass_low_hz"),
        bandpass_high_hz=config.get("bandpass_high_hz"),
        repair_flat_leads=bool(config.get("repair_flat_leads", False)),
        clip_abs=config.get("clip_abs"),
    )
    if bool(config.get("renorm_after_stabilizer", False)):
        y = global_zscore_torch(y)
    if squeeze:
        y = y.squeeze(0)
    return y.to(dtype=ecg_ct.dtype)


def center_crop_ct(ecg_ct: torch.Tensor, crop_len: int | None) -> torch.Tensor:
    if crop_len and int(crop_len) < ecg_ct.shape[-1]:
        start = max((ecg_ct.shape[-1] - int(crop_len)) // 2, 0)
        return ecg_ct[..., start : start + int(crop_len)]
    return ecg_ct


class PN2021CachedCenterDataset(Dataset):
    def __init__(self, signals: np.ndarray, labels: np.ndarray, crop_len: int = 250):
        self.signals = signals
        self.labels = labels.astype(np.float32, copy=False)
        self.crop_len = crop_len
        self._fail = 0
        self._load_time = 0.0

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sig_tc = self.signals[idx]
        crop = crop_signal_tc(sig_tc, self.crop_len, mode="center")
        label = np.array(self.labels[idx], dtype=np.float32, copy=True)
        return (
            torch.from_numpy(np.ascontiguousarray(crop.T)).float(),
            torch.from_numpy(label).float(),
        )


class PN2021IndexedCenterDataset(Dataset):
    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        crop_len: int = 250,
        input_stabilizer_config: dict | None = None,
    ):
        self.signals = signals
        self.labels = labels.astype(np.float32, copy=False)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.crop_len = crop_len
        self.input_stabilizer_config = dict(input_stabilizer_config or {})

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        real_idx = int(self.indices[idx])
        sig_tc = self.signals[real_idx]
        stage = self.input_stabilizer_config.get("stage", "post_crop")
        if stage == "pre_crop":
            sig_ct_np = sig_tc.T
        else:
            start = max((sig_tc.shape[0] - self.crop_len) // 2, 0)
            crop = sig_tc[start : start + self.crop_len]
            sig_ct_np = crop.T
        sig_ct = torch.from_numpy(np.ascontiguousarray(sig_ct_np)).float()
        sig_ct = apply_effnet_input_stabilizer(sig_ct, self.input_stabilizer_config)
        if stage == "pre_crop":
            sig_ct = center_crop_ct(sig_ct, self.crop_len)
        label = np.array(self.labels[real_idx], dtype=np.float32, copy=True)
        return sig_ct.float(), torch.from_numpy(label).float()


class StreamingCorruptedPN2021Dataset(Dataset):
    def __init__(
        self,
        signals,
        labels,
        corruption,
        public_severity,
        seed=20260501,
        crop_len=250,
        severity_profile="standard",
        indices=None,
        input_stabilizer_config=None,
        severity_profile_params=None,
    ):
        self.signals = signals
        self.labels = labels.astype(np.float32, copy=False)
        self.indices = (
            np.arange(len(signals), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        self.corruption = corruption
        self.public_severity = int(public_severity)
        self.internal_severity = PUBLIC_TO_INTERNAL_SEVERITY[self.public_severity]
        self.severity_profile = severity_profile
        self.severity_profile_params = severity_profile_params
        self.seed = int(seed)
        self.crop_len = crop_len
        self.input_stabilizer_config = dict(input_stabilizer_config or {})

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        sig_tc = self.signals[real_idx]
        stage = self.input_stabilizer_config.get("stage", "post_crop")
        if stage == "pre_crop":
            sig_ct = torch.from_numpy(np.ascontiguousarray(sig_tc.T)).float()
        else:
            start = max((sig_tc.shape[0] - self.crop_len) // 2, 0)
            crop = sig_tc[start : start + self.crop_len]
            sig_ct = torch.from_numpy(np.ascontiguousarray(crop.T)).float()
        corrupt_ct = apply_corruption_sequence(
            sig_ct,
            self.corruption,
            self.public_severity,
            self.severity_profile,
            base_seed=self.seed,
            seed_parts=(self.corruption, self.public_severity, real_idx),
            severity_profile_params=self.severity_profile_params,
        )
        corrupt_ct = apply_effnet_input_stabilizer(
            corrupt_ct,
            self.input_stabilizer_config,
        )
        if stage == "pre_crop":
            corrupt_ct = center_crop_ct(corrupt_ct, self.crop_len)
        label = np.array(self.labels[real_idx], dtype=np.float32, copy=True)
        return corrupt_ct.float(), torch.from_numpy(label).float()


def global_zscore_ct(ecg_ct):
    return global_zscore_torch(ecg_ct.unsqueeze(0)).squeeze(0)


def apply_stabilizer_at_configured_stage(ecg_ct, input_stabilizer_config, crop_len):
    config = dict(input_stabilizer_config or {})
    if str(config.get("stage", "post_crop")) == "pre_crop":
        ecg_ct = apply_effnet_input_stabilizer(ecg_ct, config)
        return center_crop_ct(ecg_ct, crop_len)
    cropped = center_crop_ct(ecg_ct, crop_len)
    return apply_effnet_input_stabilizer(cropped, config)


class RawFirstCleanPN2021Dataset(Dataset):
    """Clean PN2021 view for raw-first corruption evaluation."""

    def __init__(self, signals, labels, indices, crop_len=250, input_stabilizer_config=None):
        self.signals = signals.astype(np.float32, copy=False)
        self.labels = labels.astype(np.float32, copy=False)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.crop_len = crop_len
        self.input_stabilizer_config = dict(input_stabilizer_config or {})

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        sig_tc = self.signals[real_idx]
        sig_ct = torch.from_numpy(np.ascontiguousarray(sig_tc.T)).float()
        sig_ct = global_zscore_ct(sig_ct)
        sig_ct = apply_stabilizer_at_configured_stage(
            sig_ct,
            self.input_stabilizer_config,
            self.crop_len,
        )
        label = np.array(self.labels[real_idx], dtype=np.float32, copy=True)
        return sig_ct.float(), torch.from_numpy(label).float()


class RawFirstCorruptedPN2021Dataset(Dataset):
    """Apply corruption before model-facing z-score normalization."""

    def __init__(
        self,
        signals,
        labels,
        corruption,
        public_severity,
        seed=20260501,
        crop_len=250,
        severity_profile="standard",
        indices=None,
        input_stabilizer_config=None,
        severity_profile_params=None,
    ):
        self.signals = signals.astype(np.float32, copy=False)
        self.labels = labels.astype(np.float32, copy=False)
        self.indices = (
            np.arange(len(signals), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        self.corruption = corruption
        self.public_severity = int(public_severity)
        self.internal_severity = PUBLIC_TO_INTERNAL_SEVERITY[self.public_severity]
        self.severity_profile = severity_profile
        self.severity_profile_params = severity_profile_params
        self.seed = int(seed)
        self.crop_len = crop_len
        self.input_stabilizer_config = dict(input_stabilizer_config or {})

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        sig_tc = self.signals[real_idx]
        sig_ct = torch.from_numpy(np.ascontiguousarray(sig_tc.T)).float()
        corrupt_ct = apply_corruption_sequence(
            sig_ct,
            self.corruption,
            self.public_severity,
            self.severity_profile,
            base_seed=self.seed,
            seed_parts=("raw_first", self.corruption, self.public_severity, real_idx),
            severity_profile_params=self.severity_profile_params,
        )
        corrupt_ct = global_zscore_ct(corrupt_ct)
        corrupt_ct = apply_stabilizer_at_configured_stage(
            corrupt_ct,
            self.input_stabilizer_config,
            self.crop_len,
        )
        label = np.array(self.labels[real_idx], dtype=np.float32, copy=True)
        return corrupt_ct.float(), torch.from_numpy(label).float()


def prepare_native_raw_signal_tc(signal_tc, source_leads):
    signal_tc = np.asarray(signal_tc, dtype=np.float32)
    if signal_tc.ndim != 2:
        return None
    if not np.isfinite(signal_tc).all():
        signal_tc = np.nan_to_num(signal_tc, nan=0.0, posinf=0.0, neginf=0.0)
    if source_leads is not None:
        signal_tc = reorder_leads_tc(signal_tc, source_leads)
        if signal_tc is None:
            return None
    if signal_tc.shape[1] != 12:
        return None
    return signal_tc.astype(np.float32, copy=False)


def model_preprocess_native_tc(signal_tc, sample_rate_hz):
    return unified_preprocess_to_1000(
        signal_tc,
        fs=float(sample_rate_hz),
        source_leads=None,
        target_fs=100,
        target_len=1000,
        preprocess_mode="minimal_resample",
        norm_mode="per_sample_global",
    )


class NativeRawFirstCleanPN2021Dataset(Dataset):
    """Clean PN2021 view that starts from native-fs raw ECG records."""

    def __init__(self, signals, labels, sample_rates, indices, crop_len=250, input_stabilizer_config=None):
        self.signals = list(signals)
        self.labels = labels.astype(np.float32, copy=False)
        self.sample_rates = np.asarray(sample_rates, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.crop_len = crop_len
        self.input_stabilizer_config = dict(input_stabilizer_config or {})

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        proc = model_preprocess_native_tc(
            self.signals[real_idx],
            float(self.sample_rates[real_idx]),
        )
        if proc is None:
            raise RuntimeError(f"native raw clean preprocessing failed for index {real_idx}")
        sig_ct = torch.from_numpy(np.ascontiguousarray(proc.T)).float()
        sig_ct = apply_stabilizer_at_configured_stage(
            sig_ct,
            self.input_stabilizer_config,
            self.crop_len,
        )
        label = np.array(self.labels[real_idx], dtype=np.float32, copy=True)
        return sig_ct.float(), torch.from_numpy(label).float()


class NativeRawFirstCorruptedPN2021Dataset(Dataset):
    """Apply corruption on native-fs raw ECG, then run model preprocessing."""

    def __init__(
        self,
        signals,
        labels,
        sample_rates,
        corruption,
        public_severity,
        seed=20260501,
        crop_len=250,
        severity_profile="standard",
        indices=None,
        input_stabilizer_config=None,
        severity_profile_params=None,
    ):
        self.signals = list(signals)
        self.labels = labels.astype(np.float32, copy=False)
        self.sample_rates = np.asarray(sample_rates, dtype=np.float32)
        self.indices = (
            np.arange(len(signals), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        self.corruption = corruption
        self.public_severity = int(public_severity)
        self.internal_severity = PUBLIC_TO_INTERNAL_SEVERITY[self.public_severity]
        self.severity_profile = severity_profile
        self.severity_profile_params = severity_profile_params
        self.seed = int(seed)
        self.crop_len = crop_len
        self.input_stabilizer_config = dict(input_stabilizer_config or {})

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        sample_rate_hz = float(self.sample_rates[real_idx])
        sig_ct = torch.from_numpy(np.ascontiguousarray(self.signals[real_idx].T)).float()
        corrupt_ct = apply_corruption_sequence(
            sig_ct,
            self.corruption,
            self.public_severity,
            self.severity_profile,
            base_seed=self.seed,
            seed_parts=(
                "native_raw_first",
                self.corruption,
                self.public_severity,
                real_idx,
            ),
            sample_rate_hz=sample_rate_hz,
            severity_profile_params=self.severity_profile_params,
        )
        corrupt_tc = corrupt_ct.detach().cpu().numpy().T.astype(np.float32, copy=False)
        proc = model_preprocess_native_tc(corrupt_tc, sample_rate_hz)
        if proc is None:
            raise RuntimeError(f"native raw corrupted preprocessing failed for index {real_idx}")
        corrupt_ct = torch.from_numpy(np.ascontiguousarray(proc.T)).float()
        corrupt_ct = apply_stabilizer_at_configured_stage(
            corrupt_ct,
            self.input_stabilizer_config,
            self.crop_len,
        )
        label = np.array(self.labels[real_idx], dtype=np.float32, copy=True)
        return corrupt_ct.float(), torch.from_numpy(label).float()


class ECGFounderStreamingCorruptedDataset(Dataset):
    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        corruption: str,
        public_severity: int,
        *,
        seed: int,
        crop_len: int,
        severity_profile: str,
        severity_profile_params: dict[str, Any] | None = None,
    ) -> None:
        self.signals = signals
        self.labels = labels.astype(np.float32, copy=False)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.corruption = str(corruption)
        self.public_severity = int(public_severity)
        self.internal_severity = PUBLIC_TO_INTERNAL_SEVERITY[self.public_severity]
        self.seed = int(seed)
        self.crop_len = int(crop_len)
        self.severity_profile = str(severity_profile)
        self.severity_profile_params = severity_profile_params

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        real_idx = int(self.indices[idx])
        sig_tc = crop_signal_tc(self.signals[real_idx], self.crop_len, mode="center")
        sig_ct = torch.from_numpy(np.ascontiguousarray(sig_tc.T)).float()
        corrupt_ct = apply_corruption_sequence(
            sig_ct,
            self.corruption,
            self.public_severity,
            self.severity_profile,
            base_seed=self.seed,
            seed_parts=(
                "ecgfounder_preprocessed_cache",
                self.corruption,
                self.public_severity,
                real_idx,
            ),
            severity_profile_params=self.severity_profile_params,
        )
        label = torch.from_numpy(
            np.array(self.labels[real_idx], dtype=np.float32, copy=True)
        ).float()
        return corrupt_ct.float(), label


class ECGFounderCleanDataset(Dataset):
    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        *,
        crop_len: int,
    ) -> None:
        self.signals = signals
        self.labels = labels.astype(np.float32, copy=False)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.crop_len = int(crop_len)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        real_idx = int(self.indices[idx])
        sig_tc = crop_signal_tc(self.signals[real_idx], self.crop_len, mode="center")
        sig_ct = torch.from_numpy(np.ascontiguousarray(sig_tc.T)).float()
        label = torch.from_numpy(
            np.array(self.labels[real_idx], dtype=np.float32, copy=True)
        ).float()
        return sig_ct.float(), label


def ecg1000_ct_to_ecgfounder_5000_no_zscore(
    sig_ct: torch.Tensor,
    *,
    target_points: int = 5000,
) -> torch.Tensor:
    return ecg1000_to_ecgfounder_input(
        sig_ct.unsqueeze(0),
        target_points=int(target_points),
        apply_global_zscore=False,
    ).squeeze(0)


class ECGFounderBottleneck5000CleanDataset(Dataset):
    """Clean ECGFounder view for the locked 100Hz->500Hz->zscore order."""

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        *,
        crop_len: int,
        target_points: int = 5000,
    ) -> None:
        self.signals = signals.astype(np.float32, copy=False)
        self.labels = labels.astype(np.float32, copy=False)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.crop_len = int(crop_len)
        self.target_points = int(target_points)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        real_idx = int(self.indices[idx])
        sig_tc = crop_signal_tc(self.signals[real_idx], self.crop_len, mode="center")
        sig_ct = torch.from_numpy(np.ascontiguousarray(sig_tc.T)).float()
        sig_ct = ecg1000_ct_to_ecgfounder_5000_no_zscore(
            sig_ct,
            target_points=self.target_points,
        )
        label = torch.from_numpy(
            np.array(self.labels[real_idx], dtype=np.float32, copy=True)
        ).float()
        return sig_ct.float(), label


class ECGFounderBottleneck5000CorruptedDataset(Dataset):
    """Apply PN2021-C corruption after ECGFounder's 500 Hz interpolation."""

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        corruption: str,
        public_severity: int,
        *,
        seed: int,
        crop_len: int,
        severity_profile: str,
        severity_profile_params: dict[str, Any] | None = None,
        target_points: int = 5000,
    ) -> None:
        self.signals = signals.astype(np.float32, copy=False)
        self.labels = labels.astype(np.float32, copy=False)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.corruption = str(corruption)
        self.public_severity = int(public_severity)
        self.internal_severity = PUBLIC_TO_INTERNAL_SEVERITY[self.public_severity]
        self.seed = int(seed)
        self.crop_len = int(crop_len)
        self.severity_profile = str(severity_profile)
        self.severity_profile_params = severity_profile_params
        self.target_points = int(target_points)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        real_idx = int(self.indices[idx])
        sig_tc = crop_signal_tc(self.signals[real_idx], self.crop_len, mode="center")
        sig_ct = torch.from_numpy(np.ascontiguousarray(sig_tc.T)).float()
        sig_ct = ecg1000_ct_to_ecgfounder_5000_no_zscore(
            sig_ct,
            target_points=self.target_points,
        )
        corrupt_ct = apply_corruption_sequence(
            sig_ct,
            self.corruption,
            self.public_severity,
            self.severity_profile,
            base_seed=self.seed,
            seed_parts=(
                LOCKED_ECGFOUNDER_CORRUPTION_INPUT,
                self.corruption,
                self.public_severity,
                real_idx,
            ),
            sample_rate_hz=500.0,
            severity_profile_params=self.severity_profile_params,
        )
        label = torch.from_numpy(
            np.array(self.labels[real_idx], dtype=np.float32, copy=True)
        ).float()
        return corrupt_ct.float(), label


__all__ = [
    "ECGFounderBottleneck5000CleanDataset",
    "ECGFounderBottleneck5000CorruptedDataset",
    "ECGFounderCleanDataset",
    "ECGFounderStreamingCorruptedDataset",
    "NativeRawFirstCleanPN2021Dataset",
    "NativeRawFirstCorruptedPN2021Dataset",
    "PN2021CachedCenterDataset",
    "PN2021IndexedCenterDataset",
    "RawFirstCleanPN2021Dataset",
    "RawFirstCorruptedPN2021Dataset",
    "StreamingCorruptedPN2021Dataset",
    "apply_effnet_input_stabilizer",
    "apply_stabilizer_at_configured_stage",
    "center_crop_ct",
    "ecg1000_ct_to_ecgfounder_5000_no_zscore",
    "global_zscore_ct",
    "has_input_stabilizer",
    "model_preprocess_native_tc",
    "prepare_native_raw_signal_tc",
]

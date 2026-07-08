"""Raw 100 Hz ECG signal helpers for locked PN2021 adaptation runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from ecg_adv_gen.data.real_anchors import find_real_anchor_base
from ecg_adv_gen.evaluation.pn2021c import build_corruption_op
from methods.augmix.severity import build_op


class RawSignalDataset(Dataset):
    """Memory-backed raw ECG dataset returning channels-first signals."""

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        *,
        indices: np.ndarray | None = None,
        target_len: int = 1000,
    ) -> None:
        self.signals = signals
        self.labels = labels.astype(np.float32, copy=False)
        self.target_len = int(target_len)
        self.indices = (
            np.arange(labels.shape[0], dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        if self.labels.shape[0] != self.signals.shape[0]:
            raise ValueError(
                f"signals/labels length mismatch: {self.signals.shape[0]} vs {self.labels.shape[0]}"
            )

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        real_idx = int(self.indices[idx])
        sig = np.asarray(self.signals[real_idx], dtype=np.float32)
        if sig.ndim != 2:
            raise ValueError(f"raw signal must be 2D, got {sig.shape}")
        if sig.shape[0] == 12:
            sig_ct = sig
        elif sig.shape[-1] == 12:
            sig_ct = sig.T
        else:
            raise ValueError(f"raw signal must be channels-first/last with 12 leads, got {sig.shape}")

        if self.target_len > 0 and sig_ct.shape[-1] != self.target_len:
            src_len = int(sig_ct.shape[-1])
            if src_len % self.target_len == 0:
                sig_ct = sig_ct[:, :: src_len // self.target_len]
            else:
                src_x = np.linspace(0.0, 1.0, num=src_len, dtype=np.float32)
                dst_x = np.linspace(0.0, 1.0, num=self.target_len, dtype=np.float32)
                sig_ct = np.stack(
                    [np.interp(dst_x, src_x, lead).astype(np.float32) for lead in sig_ct],
                    axis=0,
                )
        return sig_ct.astype(np.float32, copy=True), self.labels[real_idx].astype(np.float32, copy=True)


def real_anchor_base(
    center: str,
    args: Any,
    *,
    default_roots: Iterable[str | Path] = (),
) -> Path:
    return find_real_anchor_base(
        center,
        anchor_base_root=getattr(args, "anchor_base_root", ""),
        k=int(getattr(args, "k", 500)),
        seed=int(getattr(args, "seed", 42)),
        default_roots=default_roots,
    )


def anchor_signal_npz_path(
    center: str,
    args: Any,
    *,
    default_roots: Iterable[str | Path] = (),
) -> Path:
    base = real_anchor_base(center, args, default_roots=default_roots)
    candidates = [
        Path(str(base) + ".signals.npz"),
        base.with_suffix(".signals.npz"),
        base,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"missing target raw anchor signals for {center}; checked: "
        + ", ".join(str(p) for p in candidates)
    )


def apply_augmix_op_np(
    sig_ct: np.ndarray,
    op_name: str,
    op_severity: int,
    severity_profile: str = "standard",
    severity_profile_params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Apply one ECG AugMix op to a channels-first 100 Hz ECG sample."""

    if str(severity_profile) == "custom":
        op = build_corruption_op(
            op_name,
            int(op_severity),
            str(severity_profile),
            severity_profile_params=severity_profile_params,
        )
    else:
        if severity_profile_params is not None:
            raise ValueError("severity_profile_params are only valid for severity_profile=custom")
        op = build_op(op_name, int(op_severity)) if str(severity_profile) == "standard" else build_corruption_op(
            op_name,
            int(op_severity),
            str(severity_profile),
        )
    sig_t = torch.from_numpy(sig_ct.copy()).float()
    return op(sig_t).cpu().numpy().astype(np.float32, copy=False)

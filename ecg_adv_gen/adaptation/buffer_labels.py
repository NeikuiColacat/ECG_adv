"""Label construction helpers for adversarial ECG buffers."""

from __future__ import annotations

import numpy as np


def _as_1d_float32(value: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape={arr.shape!r}")
    return arr


def _target_index(target_label: np.ndarray) -> int:
    if target_label.size == 0:
        raise ValueError("target_label must not be empty")
    return int(np.argmax(target_label))


def build_adv_buffer_label(
    target_label: np.ndarray,
    *,
    label_mode: str,
    teacher_probs: np.ndarray | None = None,
    teacher_mix: float = 0.7,
    soft_target_floor: float = 0.0,
) -> np.ndarray:
    """Build one buffer label using the legacy masked-BCE label modes.

    ``hard`` keeps only the primary target class and uses ``-1`` as the masked
    sentinel for non-target classes. Other modes return full multi-label or soft
    labels in ``[0, 1]``.
    """

    target = _as_1d_float32(target_label, name="target_label")
    target_idx = _target_index(target)
    if label_mode == "hard":
        label = np.full((target.shape[0],), -1.0, dtype=np.float32)
        label[target_idx] = 1.0
        return label
    if label_mode == "multi_hot_hard":
        return (target > 0.5).astype(np.float32)
    if label_mode == "latent_soft":
        return np.clip(target, 0.0, 1.0).astype(np.float32, copy=False)

    if teacher_probs is None:
        raise ValueError(f"teacher_probs required for label_mode={label_mode}")
    teacher = np.clip(_as_1d_float32(teacher_probs, name="teacher_probs"), 0.0, 1.0)
    if teacher.shape != target.shape:
        raise ValueError(
            f"teacher_probs must have the same shape as target_label; "
            f"got {teacher.shape!r} vs {target.shape!r}"
        )

    if label_mode == "teacher_soft":
        label = teacher.copy()
    elif label_mode == "mixed_soft":
        hard_full = np.zeros((target.shape[0],), dtype=np.float32)
        hard_full[target_idx] = 1.0
        mix = float(np.clip(float(teacher_mix), 0.0, 1.0))
        label = mix * teacher + (1.0 - mix) * hard_full
    elif label_mode == "latent_mixed_teacher":
        latent_soft = np.clip(target, 0.0, 1.0)
        mix = float(np.clip(float(teacher_mix), 0.0, 1.0))
        label = mix * teacher + (1.0 - mix) * latent_soft
    else:
        raise ValueError(f"unsupported adv label_mode={label_mode}")

    if soft_target_floor > 0.0:
        label[target_idx] = max(float(label[target_idx]), float(soft_target_floor))
    return label.astype(np.float32, copy=False)

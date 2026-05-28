"""Deterministic train/validation split helpers for adaptation runners."""

from __future__ import annotations

from typing import Literal

import numpy as np

ZeroValPolicy = Literal["identity", "permuted"]


def random_split_indices(
    n: int,
    val_fraction: float,
    seed: int,
    *,
    zero_val_policy: ZeroValPolicy = "identity",
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``range(n)`` into train/validation indices with legacy semantics.

    For ``val_fraction > 0``, this matches the active paper runners: a seeded
    permutation is drawn, ``round(n * val_fraction)`` is used with a minimum of
    one validation sample, and train/val indices are sorted before returning.

    For ``val_fraction <= 0``, historical runners differed. ``identity`` returns
    sorted ``arange(n)`` for both train and validation; ``permuted`` returns the
    same seeded permutation for both.
    """
    n = int(n)
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    frac = float(val_fraction)
    if frac <= 0:
        if zero_val_policy == "identity":
            indices = np.arange(n, dtype=np.int64)
            return indices, indices
        if zero_val_policy == "permuted":
            order = np.random.default_rng(int(seed)).permutation(n).astype(np.int64)
            return order, order
        raise ValueError(f"unknown zero_val_policy={zero_val_policy!r}")

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n)
    n_val = max(1, int(round(n * frac)))
    val_idx = np.sort(order[:n_val]).astype(np.int64)
    train_idx = np.sort(order[n_val:]).astype(np.int64)
    return train_idx, val_idx

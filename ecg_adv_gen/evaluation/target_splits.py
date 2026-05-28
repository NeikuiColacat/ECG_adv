"""Target-center internal split helpers for paper-safe adaptation."""

from __future__ import annotations

from typing import Literal

import numpy as np

TargetSplitMode = Literal["random", "stratified"]


def split_target_train_val_indices(
    target_idx: np.ndarray,
    record_ids: np.ndarray,
    labels: np.ndarray,
    val_count: int,
    split_seed: int,
    split_mode: TargetSplitMode | str = "random",
) -> tuple[np.ndarray, np.ndarray, set[str], set[str]]:
    """Split K-shot target rows into train/validation without held-out data.

    This preserves the ECGFounder full-FT runner's legacy behavior, including
    the ``split_seed + 1701`` offset and the rarity/coverage-biased stratified
    validation selection. Inputs are row indices into the already-selected
    K-shot target set plus the corresponding full-cache record ids and labels.
    """
    target_idx = np.asarray(target_idx, dtype=np.int64)
    if val_count <= 0:
        train_ids = {str(record_ids[i]) for i in target_idx}
        return target_idx, np.empty(0, dtype=np.int64), train_ids, set()
    if val_count >= len(target_idx):
        raise ValueError(f"target_val_count={val_count} must be smaller than target K={len(target_idx)}")

    rng = np.random.default_rng(int(split_seed) + 1701)
    perm = np.asarray(target_idx, dtype=np.int64).copy()
    rng.shuffle(perm)
    if split_mode == "random":
        val_idx = np.sort(perm[:val_count])
        train_idx = np.sort(perm[val_count:])
    elif split_mode == "stratified":
        y = np.asarray(labels, dtype=np.float32)[target_idx] > 0.5
        pos_counts = y.sum(axis=0)
        selected: list[int] = []
        selected_set: set[int] = set()
        class_order = [
            int(c)
            for c in np.argsort(pos_counts)
            if int(pos_counts[int(c)]) >= 2
        ]
        for class_i in class_order:
            candidates = [
                int(i)
                for i, row in zip(target_idx, y)
                if row[class_i] and int(i) not in selected_set
            ]
            if not candidates:
                continue
            candidate_scores = []
            for idx in candidates:
                row = np.asarray(labels[idx], dtype=np.float32) > 0.5
                covered_new = sum(
                    1
                    for c in class_order
                    if row[c] and not any(np.asarray(labels[j], dtype=np.float32)[c] > 0.5 for j in selected)
                )
                rarity = float(
                    np.sum(
                        [
                            1.0 / max(float(pos_counts[c]), 1.0)
                            for c, present in enumerate(row)
                            if present
                        ]
                    )
                )
                candidate_scores.append((-covered_new, -rarity, rng.random(), idx))
            candidate_scores.sort()
            pick = int(candidate_scores[0][3])
            selected.append(pick)
            selected_set.add(pick)
            if len(selected) >= val_count:
                break
        for idx in perm:
            idx = int(idx)
            if len(selected) >= val_count:
                break
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)
        val_idx = np.sort(np.asarray(selected, dtype=np.int64))
        val_set = set(int(i) for i in val_idx)
        train_idx = np.sort(
            np.asarray([int(i) for i in target_idx if int(i) not in val_set], dtype=np.int64)
        )
    else:
        raise ValueError(f"unknown target_val_split_mode={split_mode}")

    train_ids = {str(record_ids[i]) for i in train_idx}
    val_ids = {str(record_ids[i]) for i in val_idx}
    return train_idx, val_idx, train_ids, val_ids

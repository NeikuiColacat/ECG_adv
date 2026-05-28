"""Real-anchor latent pool helpers for target-center adaptation.

These helpers keep dated runner scripts from owning filesystem lookup,
record-id matching, and K-shot anchor subset assembly. They intentionally avoid
waveform decoding, torch, and model imports so config/audit tests can exercise
the protocol on CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ecg_adv_gen.labels import CLASS_NAMES_SUPER5


def real_anchor_base_candidates(
    center: str,
    anchor_base_root: str | Path,
    *,
    k: int,
    seed: int,
) -> list[Path]:
    """Return current explicit-root anchor filename candidates."""
    root = Path(anchor_base_root)
    return [
        root / center / f"k{k}_seed{seed}" / f"{center}_real_k{k}_seed{seed}",
        root / center / f"{center}_real_k{k}_seed{seed}",
        root / center / f"{center}_real_k500_seed{seed}",
    ]


def find_real_anchor_base(
    center: str,
    *,
    anchor_base_root: str | Path | None = None,
    k: int = 500,
    seed: int = 42,
    default_roots: Iterable[str | Path] = (),
) -> Path:
    """Find the extension-free real-anchor base path for one center.

    The explicit-root search preserves the v6 K500 managed-layout candidates.
    The default-root fallback preserves the older seed42 prompt-token anchor
    layout used by legacy scripts when no explicit root is provided.
    """
    if anchor_base_root:
        candidates = real_anchor_base_candidates(center, anchor_base_root, k=k, seed=seed)
        for base in candidates:
            if base.with_suffix(".latent.npz").exists():
                return base
        searched = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"missing real-anchor files for {center}; searched: {searched}")

    candidates = [
        Path(root) / center / f"{center}_real_k500_seed42"
        for root in default_roots
    ]
    for base in candidates:
        if base.with_suffix(".latent.npz").exists():
            return base
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"missing real-anchor files for {center}; searched: {searched}")


def select_primary_label_proportional_min1(
    labels: np.ndarray,
    k: int,
    seed: int,
    *,
    num_classes: int | None = None,
) -> np.ndarray:
    """Legacy-compatible proportional primary-class K-shot subset selection.

    This matches the ECGFounder VAE-LHAT script's local behavior: classes that
    are present receive at least one provisional slot, then allocations are
    reduced or filled to exactly ``k`` before selected indices are sorted.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2D, got shape {labels.shape}")
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if k > len(labels):
        raise ValueError(f"K={k} exceeds matched anchors {len(labels)}")
    if k == len(labels):
        return np.arange(len(labels), dtype=np.int64)

    n_classes = int(num_classes if num_classes is not None else labels.shape[1])
    rng = np.random.default_rng(seed)
    primary = np.argmax(labels, axis=1)
    counts = np.bincount(primary, minlength=n_classes)
    raw = counts / max(int(counts.sum()), 1) * k
    alloc = np.floor(raw).astype(int)

    for cls in range(n_classes):
        if counts[cls] > 0 and alloc[cls] == 0:
            alloc[cls] = 1
    while alloc.sum() > k:
        cls = int(np.argmax(alloc))
        alloc[cls] -= 1
    while alloc.sum() < k:
        before = int(alloc.sum())
        for cls in np.argsort(-(raw - np.floor(raw))):
            if alloc.sum() >= k:
                break
            if alloc[cls] < counts[cls]:
                alloc[cls] += 1
        if int(alloc.sum()) == before:
            break

    pieces: list[np.ndarray] = []
    for cls, n_take in enumerate(alloc):
        idx = np.where(primary == cls)[0]
        if n_take > 0 and len(idx) > 0:
            pieces.append(rng.permutation(idx)[: min(int(n_take), len(idx))])
    selected = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
    if len(selected) < k:
        rest = np.setdiff1d(np.arange(len(labels)), selected, assume_unique=False)
        selected = np.concatenate([selected, rng.permutation(rest)[: k - len(selected)]])
    return np.sort(selected[:k].astype(np.int64, copy=False))


def _load_anchor_latents(base: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(Path(base).with_suffix(".latent.npz"), allow_pickle=True) as data:
        latents = data["latents"].astype(np.float32)
        record_ids = data["record_ids"].astype(str)
    return latents, record_ids


def _pn_label_lookup(
    center: str,
    pn_payload: dict[str, np.ndarray],
) -> dict[tuple[str, str], np.ndarray]:
    pn_record_ids = pn_payload["record_ids"].astype(str)
    pn_labels = pn_payload["labels"].astype(np.float32)
    if "centers" in pn_payload:
        pn_centers = pn_payload["centers"].astype(str)
    else:
        pn_centers = np.asarray([center] * len(pn_record_ids), dtype=str)
    return {
        (str(c), str(rid)): pn_labels[i]
        for i, (c, rid) in enumerate(zip(pn_centers, pn_record_ids))
    }


def _pn_row_lookup(
    center: str,
    pn_payload: dict[str, np.ndarray],
) -> dict[tuple[str, str], int]:
    pn_record_ids = pn_payload["record_ids"].astype(str)
    if "centers" in pn_payload:
        pn_centers = pn_payload["centers"].astype(str)
    else:
        pn_centers = np.asarray([center] * len(pn_record_ids), dtype=str)
    return {
        (str(c), str(rid)): i
        for i, (c, rid) in enumerate(zip(pn_centers, pn_record_ids))
    }


def anchor_label_counts(
    labels: np.ndarray,
    *,
    class_names: Sequence[str] = CLASS_NAMES_SUPER5,
) -> dict[str, int]:
    counts = np.asarray(labels).sum(axis=0).astype(int).tolist()
    return dict(zip(class_names, counts))


def anchor_classes_in_scope(
    labels: np.ndarray,
    *,
    class_names: Sequence[str] = CLASS_NAMES_SUPER5,
) -> list[str]:
    present = np.asarray(labels).sum(axis=0) > 0
    return [c for c, ok in zip(class_names, present) if bool(ok)]


def load_real_anchor_pool(
    center: str,
    *,
    k: int,
    seed: int,
    pn_payload: dict[str, np.ndarray],
    anchor_base_root: str | Path | None = None,
    default_roots: Iterable[str | Path] = (),
    class_names: Sequence[str] = CLASS_NAMES_SUPER5,
) -> dict[str, Any]:
    """Load matched target-center latent anchors and choose a K-shot subset."""
    base = find_real_anchor_base(
        center,
        anchor_base_root=anchor_base_root,
        k=k,
        seed=seed,
        default_roots=default_roots,
    )
    latents_all, record_ids_all = _load_anchor_latents(base)
    label_by_key = _pn_label_lookup(center, pn_payload)

    labels_all: list[np.ndarray] = []
    keep: list[int] = []
    for i, rid in enumerate(record_ids_all):
        label = label_by_key.get((center, str(rid)))
        if label is None:
            continue
        labels_all.append(label)
        keep.append(i)
    if not keep:
        raise RuntimeError(f"{center}: no anchor record ids matched PN2021 feature cache")

    keep_idx = np.asarray(keep, dtype=np.int64)
    latents_all = latents_all[keep_idx]
    record_ids_all = record_ids_all[keep_idx]
    labels_arr = np.stack(labels_all).astype(np.float32)

    selected = select_primary_label_proportional_min1(
        labels_arr,
        k,
        seed,
        num_classes=len(class_names),
    )
    labels = labels_arr[selected]
    return {
        "latents": latents_all[selected],
        "labels": labels,
        "record_ids": record_ids_all[selected],
        "classes_in_scope": anchor_classes_in_scope(labels, class_names=class_names),
        "label_counts": anchor_label_counts(labels, class_names=class_names),
        "source_base": str(base),
    }


def load_real_anchor_pool_for_record_ids(
    center: str,
    *,
    selected_ids: set[str] | Sequence[str],
    pn_payload: dict[str, np.ndarray],
    anchor_base_root: str | Path | None = None,
    k: int = 500,
    seed: int = 42,
    default_roots: Iterable[str | Path] = (),
    class_names: Sequence[str] = CLASS_NAMES_SUPER5,
    include_signals: bool = False,
) -> dict[str, Any]:
    """Load latent anchors for an explicit selected record-id set."""
    base = find_real_anchor_base(
        center,
        anchor_base_root=anchor_base_root,
        k=k,
        seed=seed,
        default_roots=default_roots,
    )
    latents_all, record_ids_all = _load_anchor_latents(base)

    want = {str(x) for x in selected_ids}
    rid_to_i = {str(rid): i for i, rid in enumerate(record_ids_all)}
    missing = sorted(want - set(rid_to_i))
    if missing:
        raise RuntimeError(
            f"{center}: {len(missing)} selected ids are missing from {Path(base).with_suffix('.latent.npz')}; "
            f"first missing={missing[:5]}"
        )
    chosen = np.asarray([rid_to_i[rid] for rid in sorted(want)], dtype=np.int64)

    label_by_key = _pn_label_lookup(center, pn_payload)
    row_by_key = _pn_row_lookup(center, pn_payload)
    labels: list[np.ndarray] = []
    signals: list[np.ndarray] = []
    for rid in record_ids_all[chosen]:
        key = (center, str(rid))
        if key not in label_by_key:
            raise RuntimeError(f"{center}: selected id {rid} missing from PN cache labels")
        labels.append(label_by_key[key])
        if include_signals:
            if "signals" not in pn_payload:
                raise RuntimeError(f"{center}: PN cache has no signals array")
            if key not in row_by_key:
                raise RuntimeError(f"{center}: selected id {rid} missing from PN cache signals")
            signals.append(np.asarray(pn_payload["signals"][row_by_key[key]], dtype=np.float32))

    labels_arr = np.stack(labels).astype(np.float32)
    out = {
        "latents": latents_all[chosen],
        "labels": labels_arr,
        "record_ids": record_ids_all[chosen],
        "classes_in_scope": anchor_classes_in_scope(labels_arr, class_names=class_names),
        "label_counts": anchor_label_counts(labels_arr, class_names=class_names),
        "source_base": str(base),
    }
    if include_signals:
        out["signals"] = np.stack(signals).astype(np.float32)
    return out

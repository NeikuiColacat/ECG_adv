"""K-shot metadata helpers for target-center adaptation runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

INCLUDE_RECORD_ID_KEYS = (
    "heldout_record_ids",
    "eval_record_ids",
    "include_record_ids",
    "test_record_ids",
)
SIGNAL_NPZ_REQUIRED_KEYS = ("signals", "labels", "record_ids")
LATENT_NPZ_REQUIRED_KEYS = ("latents", "labels", "record_ids")


@dataclass(frozen=True)
class KShotArtifactPaths:
    """Expected file paths for one K-shot real-anchor artifact group."""

    signals: Path
    latents: Path
    meta: Path
    trust: Path | None = None

    @classmethod
    def from_base(cls, base: str | Path, *, include_trust: bool = False) -> "KShotArtifactPaths":
        base_path = Path(base)
        return cls(
            signals=base_path.with_suffix(".signals.npz"),
            latents=base_path.with_suffix(".latent.npz"),
            meta=base_path.with_suffix(".ref_meta.json"),
            trust=base_path.with_suffix(".class_trust.json") if include_trust else None,
        )

    def missing(self) -> list[Path]:
        paths = [self.signals, self.latents, self.meta]
        if self.trust is not None:
            paths.append(self.trust)
        return [p for p in paths if not p.exists()]


def _load_json_any(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = _load_json_any(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def load_ref_record_ids_from_meta(path: str | Path) -> tuple[str, set[str]]:
    """Load ``(center, ref_record_ids)`` from one ref-meta JSON file."""
    meta = _load_json(path)
    center = str(meta["center"])
    return center, set(str(v) for v in meta.get("ref_record_ids", []))


def load_include_record_ids_from_meta(
    path: str | Path,
    *,
    include_keys: Sequence[str] = INCLUDE_RECORD_ID_KEYS,
) -> tuple[str, set[str], str | None]:
    """Load explicit eval/include IDs from one meta JSON file."""
    meta = _load_json(path)
    center = str(meta["center"])
    for key in include_keys:
        if meta.get(key):
            return center, set(str(v) for v in meta[key]), key
    return center, set(), None


def _merge_center_ids(items: Iterable[tuple[str, set[str]]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for center, ids in items:
        if not ids:
            continue
        out.setdefault(center, set()).update(ids)
    return out


def load_ref_record_ids_by_center(meta_paths: Iterable[str | Path]) -> dict[str, set[str]]:
    return _merge_center_ids(load_ref_record_ids_from_meta(path) for path in meta_paths)


def load_include_record_ids_by_center(meta_paths: Iterable[str | Path]) -> dict[str, set[str]]:
    return _merge_center_ids(
        (center, ids)
        for center, ids, _ in (load_include_record_ids_from_meta(path) for path in meta_paths)
    )


def load_selected_record_ids_from_meta(
    path: str | Path,
    center: str,
    *,
    id_keys: Sequence[str] = ("record_ids", "ref_record_ids", "selected_ref_record_ids"),
) -> set[str]:
    """Load selected record ids from legacy K-shot ref-meta JSON variants.

    Supported shapes match the ECGFounder full-FT runner's historical parser:
    a top-level list of item dictionaries, a mapping with one of ``id_keys``,
    or a mapping with an ``items`` list. Top-level list rows are filtered by
    ``center`` when a row-level center is present; mapping-level item rows keep
    legacy behavior and accept every listed row.
    """
    payload = _load_json_any(path)
    selected: set[str] = set()
    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            if str(row.get("center", center)) != center:
                continue
            rid = row.get("record_id") or row.get("record")
            if rid is not None:
                selected.add(str(rid))
    elif isinstance(payload, dict):
        for key in id_keys:
            if key in payload:
                selected.update(str(x) for x in payload[key])
        if not selected and "items" in payload:
            for row in payload["items"]:
                if not isinstance(row, dict):
                    continue
                rid = row.get("record_id") or row.get("record")
                if rid is not None:
                    selected.add(str(rid))
    else:
        raise ValueError(f"Expected JSON object or list: {path}")
    if not selected:
        raise RuntimeError(f"could not parse selected ref ids from {path}")
    return selected


def kshot_ref_meta_path(ref_root: str | Path, center: str, k: int, seed: int) -> Path:
    """Return the canonical K-shot ref-meta path under a real-anchor root."""

    return (
        Path(ref_root)
        / center
        / f"k{int(k)}_seed{int(seed)}"
        / f"{center}_real_k{int(k)}_seed{int(seed)}.ref_meta.json"
    )


def load_kshot_ref_record_ids(
    ref_root: str | Path,
    center: str,
    *,
    k: int,
    seed: int,
    source_k: int | None = None,
) -> list[str]:
    """Load ref ids from an exact K ref-meta file, falling back to source-K.

    This preserves the legacy ECGFounder direct-head behavior: if an exact
    target K metadata file exists, use it; otherwise use the larger source-K
    pool and perform deterministic sub-selection downstream.
    """

    path = kshot_ref_meta_path(ref_root, center, k, seed)
    if not path.exists() and source_k is not None:
        path = kshot_ref_meta_path(ref_root, center, source_k, seed)
    meta = _load_json(path)
    if "ref_record_ids" not in meta:
        raise KeyError(f"{path} missing required ref_record_ids")
    return [str(rid) for rid in meta["ref_record_ids"]]


def center_offset_seed(center: str) -> int:
    """Stable legacy center offset used for center-specific K-shot draws."""

    return sum((i + 1) * ord(ch) for i, ch in enumerate(center))


def select_kshot_indices_from_ref_ids(
    labels: np.ndarray,
    centers: np.ndarray,
    record_ids: np.ndarray,
    *,
    center: str,
    ref_record_ids: Iterable[str],
    k: int,
    seed: int,
    n_classes: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Select K target-center rows from a legacy ref-id pool.

    The allocation matches the ECGFounder frozen-feature K-shot head runner:
    candidate rows are restricted to ``center`` and the supplied ref ids; if the
    pool is larger than K, select by primary-label proportions, forcing at least
    one sample from every present primary class when possible.
    """

    labels = np.asarray(labels)
    centers = np.asarray(centers).astype(str)
    record_ids = np.asarray(record_ids).astype(str)
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2D, got shape {labels.shape}")
    if len(centers) != len(labels) or len(record_ids) != len(labels):
        raise ValueError("labels, centers, and record_ids must have matching first dimensions")
    k = int(k)
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    ref_set = set(str(rid) for rid in ref_record_ids)
    mask = (centers == center) & np.asarray([rid in ref_set for rid in record_ids], dtype=bool)
    candidates = np.nonzero(mask)[0]
    if len(candidates) < k:
        raise RuntimeError(f"{center}: need K={k}, found only {len(candidates)} candidates")
    if len(candidates) == k:
        selected = np.sort(candidates)
        return selected, record_ids[selected].astype(str).tolist()

    rng = np.random.default_rng(int(seed) + center_offset_seed(center))
    y = labels[candidates]
    primary = np.argmax(y, axis=1)
    n_classes = labels.shape[1] if n_classes is None else int(n_classes)
    selected_parts = []
    counts = np.bincount(primary, minlength=n_classes)
    raw = counts / counts.sum() * k
    alloc = np.floor(raw).astype(int)
    for cls in range(n_classes):
        if counts[cls] > 0 and alloc[cls] == 0:
            alloc[cls] = 1
    while alloc.sum() > k:
        cls = int(np.argmax(alloc))
        alloc[cls] -= 1
    while alloc.sum() < k:
        remainder = raw - np.floor(raw)
        for cls in np.argsort(-remainder):
            if alloc.sum() >= k:
                break
            if alloc[cls] < counts[cls]:
                alloc[cls] += 1
    for cls, n_take in enumerate(alloc):
        cls_idx = candidates[primary == cls]
        if n_take <= 0 or len(cls_idx) == 0:
            continue
        selected_parts.append(rng.permutation(cls_idx)[: min(n_take, len(cls_idx))])
    selected = np.concatenate(selected_parts) if selected_parts else np.asarray([], dtype=np.int64)
    if len(selected) < k:
        remaining = np.setdiff1d(candidates, selected, assume_unique=False)
        selected = np.concatenate([selected, rng.permutation(remaining)[: k - len(selected)]])
    selected = np.sort(selected[:k])
    return selected, record_ids[selected].astype(str).tolist()


def select_kshot_indices_from_ref_root(
    labels: np.ndarray,
    centers: np.ndarray,
    record_ids: np.ndarray,
    *,
    center: str,
    ref_root: str | Path,
    k: int,
    source_k: int,
    subset_seed: int,
    seed: int,
    n_classes: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Select K target rows using exact-K ref metadata or source-K fallback."""

    exact_path = kshot_ref_meta_path(ref_root, center, k, subset_seed)
    source_ids = load_kshot_ref_record_ids(
        ref_root,
        center,
        k=k if exact_path.exists() else source_k,
        seed=subset_seed,
    )
    return select_kshot_indices_from_ref_ids(
        labels,
        centers,
        record_ids,
        center=center,
        ref_record_ids=source_ids,
        k=k,
        seed=seed,
        n_classes=n_classes,
    )


def proportional_stratified_indices(labels: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Legacy-compatible primary-label proportional K-shot selection."""
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2D, got shape {labels.shape}")
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    rng = np.random.default_rng(seed)
    primary = labels.argmax(axis=1)
    pools = []
    for c in range(labels.shape[1]):
        idx = np.where(primary == c)[0]
        rng.shuffle(idx)
        pools.append(idx)

    counts = np.asarray([len(p) for p in pools], dtype=np.float64)
    raw = counts / max(counts.sum(), 1.0) * k
    take = np.floor(raw).astype(int)
    remainders = raw - take
    for c in np.argsort(-remainders):
        if take.sum() >= k:
            break
        if take[c] < len(pools[c]):
            take[c] += 1
    for c in range(len(take)):
        take[c] = min(take[c], len(pools[c]))

    selected = np.concatenate([pools[c][: take[c]] for c in range(len(pools))])
    if selected.size < k:
        already = set(int(i) for i in selected)
        rest = np.asarray([i for i in range(labels.shape[0]) if i not in already])
        rng.shuffle(rest)
        selected = np.concatenate([selected, rest[: k - selected.size]])
    rng.shuffle(selected)
    return selected[:k].astype(np.int64)


def validate_npz_keys(path: str | Path, required_keys: Sequence[str]) -> dict[str, Any]:
    """Validate required NPZ keys and return lightweight shape metadata."""
    with np.load(path, allow_pickle=True) as npz:
        keys = set(npz.files)
        missing = [key for key in required_keys if key not in keys]
        if missing:
            raise ValueError(f"{path} missing required NPZ keys: {missing}")
        return {
            "path": str(Path(path)),
            "keys": sorted(keys),
            "shapes": {key: tuple(npz[key].shape) for key in required_keys},
        }


def validate_kshot_artifacts(
    paths: KShotArtifactPaths,
    *,
    expected_k: int | None = None,
) -> dict[str, Any]:
    """Validate K-shot signal/latent/meta files without interpreting waveforms."""
    missing = paths.missing()
    if missing:
        raise FileNotFoundError(f"Missing K-shot artifacts: {[str(p) for p in missing]}")
    signal_info = validate_npz_keys(paths.signals, SIGNAL_NPZ_REQUIRED_KEYS)
    latent_info = validate_npz_keys(paths.latents, LATENT_NPZ_REQUIRED_KEYS)
    meta = _load_json(paths.meta)
    n_signal = int(signal_info["shapes"]["labels"][0])
    n_latent = int(latent_info["shapes"]["labels"][0])
    n_meta = len(meta.get("ref_record_ids", []))
    if n_signal != n_latent:
        raise ValueError(f"signal/latent label count mismatch: {n_signal} vs {n_latent}")
    if n_meta and n_meta != n_signal:
        raise ValueError(f"meta ref_record_ids count mismatch: {n_meta} vs {n_signal}")
    if expected_k is not None and n_signal != int(expected_k):
        raise ValueError(f"expected K={expected_k}, found {n_signal}")
    return {
        "signals": signal_info,
        "latents": latent_info,
        "meta_path": str(paths.meta),
        "center": meta.get("center", ""),
        "k": n_signal,
        "n_ref_record_ids": n_meta,
    }

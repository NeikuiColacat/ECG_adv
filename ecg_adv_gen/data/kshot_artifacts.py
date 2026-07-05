"""Canonical K-shot artifact path and ref-meta helpers.

These helpers are intentionally CPU-only and path-focused. Training/evaluation
code can use them to avoid hand-writing K/seed/center artifact names in every
managed runner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REF_RECORD_ID_KEYS = ("ref_record_ids", "record_ids", "selected_ref_record_ids")


def canonical_kshot_base(ref_root: str | Path, center: str, *, k: int, seed: int) -> Path:
    """Return the canonical base path for one target-center K-shot group."""

    k = int(k)
    seed = int(seed)
    return Path(ref_root) / center / f"k{k}_seed{seed}" / f"{center}_real_k{k}_seed{seed}"


@dataclass(frozen=True)
class KShotArtifactGroup:
    """Canonical artifact group for one center/K/seed real-anchor subset."""

    center: str
    k: int
    seed: int
    base: Path
    include_latent: bool = False
    include_trust: bool = False

    @classmethod
    def from_root(
        cls,
        ref_root: str | Path,
        *,
        center: str,
        k: int,
        seed: int,
        include_latent: bool = False,
        include_trust: bool = False,
    ) -> "KShotArtifactGroup":
        return cls.from_base(
            canonical_kshot_base(ref_root, center, k=k, seed=seed),
            center=center,
            k=k,
            seed=seed,
            include_latent=include_latent,
            include_trust=include_trust,
        )

    @classmethod
    def from_base(
        cls,
        base: str | Path,
        *,
        center: str,
        k: int,
        seed: int,
        include_latent: bool = False,
        include_trust: bool = False,
    ) -> "KShotArtifactGroup":
        base_path = Path(base)
        expected_name = f"{center}_real_k{int(k)}_seed{int(seed)}"
        if base_path.name != expected_name:
            raise ValueError(
                f"{base_path} does not match expected K-shot base name {expected_name!r}"
            )
        return cls(
            center=center,
            k=int(k),
            seed=int(seed),
            base=base_path,
            include_latent=bool(include_latent),
            include_trust=bool(include_trust),
        )

    @property
    def signals(self) -> Path:
        return self.base.with_suffix(".signals.npz")

    @property
    def latents(self) -> Path | None:
        return self.base.with_suffix(".latent.npz") if self.include_latent else None

    @property
    def ref_meta(self) -> Path:
        return self.base.with_suffix(".ref_meta.json")

    @property
    def class_trust(self) -> Path | None:
        return self.base.with_suffix(".class_trust.json") if self.include_trust else None

    def manifest_paths(self) -> dict[str, str | Path | None]:
        """Return raw paths for a launcher/manifest layer to wrap as records."""

        return {
            "anchor_base": str(self.base),
            "signals_npz": self.signals,
            "latent_npz": self.latents,
            "ref_meta_json": self.ref_meta,
            "class_trust_json": self.class_trust,
        }


@dataclass(frozen=True)
class RefMetaRecordIds:
    """Ordered and sorted ref IDs parsed from one K-shot metadata file."""

    center: str
    record_ids: list[str]

    @property
    def sorted_record_ids(self) -> list[str]:
        return sorted(self.record_ids)


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _first_list(meta: dict[str, Any], keys: Sequence[str]) -> list[str]:
    for key in keys:
        value = meta.get(key)
        if isinstance(value, list):
            return [str(v) for v in value]
    raise ValueError(f"ref meta has no record-id list under {list(keys)!r}")


def read_ref_meta_record_ids(
    path: str | Path,
    *,
    expected_center: str | None = None,
    expected_k: int | None = None,
    expected_seed: int | None = None,
    id_keys: Sequence[str] = REF_RECORD_ID_KEYS,
) -> RefMetaRecordIds:
    """Parse and validate ref IDs from one K-shot ref-meta JSON file."""

    meta_path = Path(path)
    meta = _read_json_object(meta_path)
    center = str(meta.get("center") or "")
    if expected_center is not None and center != expected_center:
        raise ValueError(f"{meta_path} center={center!r}, expected {expected_center!r}")
    record_ids = _first_list(meta, id_keys)
    if expected_k is not None and len(record_ids) != int(expected_k):
        raise ValueError(f"{meta_path} has {len(record_ids)} record ids, expected K={int(expected_k)}")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError(f"{meta_path} contains duplicate record ids")

    if expected_k is not None:
        meta_k = meta.get("K", meta.get("k", expected_k))
        if int(meta_k) != int(expected_k):
            raise ValueError(f"{meta_path} has K={meta_k}, expected {int(expected_k)}")
    if expected_seed is not None:
        meta_seed = meta.get("selection_seed", meta.get("subset_seed", meta.get("seed", expected_seed)))
        if int(meta_seed) != int(expected_seed):
            raise ValueError(f"{meta_path} has seed={meta_seed}, expected {int(expected_seed)}")

    return RefMetaRecordIds(center=center, record_ids=record_ids)


def _class_names_array(class_names: Sequence[str]) -> np.ndarray:
    return np.asarray([str(c) for c in class_names])


def _primary_class(labels: np.ndarray, class_names: Sequence[str]) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float32)
    names = list(class_names)
    out: list[str] = []
    for row in labels:
        if float(np.max(row)) <= 0.0:
            out.append("ALL_ZERO")
        else:
            out.append(names[int(np.argmax(row))])
    return np.asarray(out)


def _label_counts(labels: np.ndarray, class_names: Sequence[str]) -> dict[str, int]:
    counts = np.asarray(labels).sum(axis=0).astype(int).tolist()
    return {str(cls): int(count) for cls, count in zip(class_names, counts)}


def _read_optional_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _copy_npz_with_labels(
    source_path: Path,
    output_path: Path,
    *,
    labels: np.ndarray,
    mapping_metadata: Mapping[str, Any],
    class_names: Sequence[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(source_path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    payload["labels"] = labels.astype(np.float32, copy=False)
    payload["class_names"] = _class_names_array(class_names)
    payload["mapping_version"] = np.asarray(str(mapping_metadata["mapping_version"]))
    payload["mapping_hash"] = np.asarray(str(mapping_metadata["mapping_hash"]))
    if "latents" in payload:
        payload["primary_class"] = _primary_class(labels, class_names)
    np.savez_compressed(output_path, **payload)


def relabel_kshot_artifact_group(
    *,
    center: str,
    source_base: Path,
    output_base: Path,
    label_by_record_id: Mapping[str, np.ndarray],
    mapping_metadata: Mapping[str, Any],
    class_names: Sequence[str],
) -> dict[str, Any]:
    signal_path = source_base.with_suffix(".signals.npz")
    latent_path = source_base.with_suffix(".latent.npz")
    meta_path = source_base.with_suffix(".ref_meta.json")
    if not signal_path.exists() or not latent_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing source K-shot artifact group for {source_base}")

    with np.load(signal_path, allow_pickle=True) as data:
        signals = data["signals"]
        record_ids = data["record_ids"].astype(str)
    labels = []
    missing = []
    for rid in record_ids:
        label = label_by_record_id.get(str(rid))
        if label is None:
            missing.append(str(rid))
        else:
            labels.append(np.asarray(label, dtype=np.float32))
    if missing:
        raise RuntimeError(f"{center}: {len(missing)} record ids missing PN2021 labels; first={missing[:5]}")
    labels_arr = np.stack(labels).astype(np.float32)
    if labels_arr.shape != (len(record_ids), len(class_names)):
        raise ValueError(f"{center}: labels shape {labels_arr.shape} does not match records/classes")
    if len(signals) != len(record_ids):
        raise ValueError(f"{center}: signals and record_ids length mismatch")

    _copy_npz_with_labels(
        signal_path,
        output_base.with_suffix(".signals.npz"),
        labels=labels_arr,
        mapping_metadata=mapping_metadata,
        class_names=class_names,
    )
    _copy_npz_with_labels(
        latent_path,
        output_base.with_suffix(".latent.npz"),
        labels=labels_arr,
        mapping_metadata=mapping_metadata,
        class_names=class_names,
    )

    source_meta = _read_optional_json_object(meta_path)
    label_counts = _label_counts(labels_arr, class_names)
    output_meta = dict(source_meta)
    output_meta.update(
        {
            "center": center,
            "ref_record_ids": record_ids.astype(str).tolist(),
            "parent": str(meta_path),
            "policy": "same K-shot record ids relabeled under active PN2021 Super5 mapping",
            "mapping_version": str(mapping_metadata["mapping_version"]),
            "mapping_hash": str(mapping_metadata["mapping_hash"]),
            "class_names": [str(c) for c in class_names],
            "label_counts": label_counts,
        }
    )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    output_base.with_suffix(".ref_meta.json").write_text(
        json.dumps(output_meta, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    class_trust = {
        str(cls): (1.0 if int(label_counts[str(cls)]) > 0 else 0.0)
        for cls in class_names
    }
    output_base.with_suffix(".class_trust.json").write_text(
        json.dumps(
            {
                "center": center,
                "mapping_version": str(mapping_metadata["mapping_version"]),
                "mapping_hash": str(mapping_metadata["mapping_hash"]),
                "policy": "real_all_present under v7 relabeled K-shot subset",
                "class_trust": class_trust,
                "label_counts": label_counts,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "center": center,
        "n": int(len(record_ids)),
        "source_base": str(source_base),
        "output_base": str(output_base),
        "mapping_version": str(mapping_metadata["mapping_version"]),
        "mapping_hash": str(mapping_metadata["mapping_hash"]),
        "label_counts": label_counts,
    }

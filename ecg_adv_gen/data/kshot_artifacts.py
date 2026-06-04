"""Canonical K-shot artifact path and ref-meta helpers.

These helpers are intentionally CPU-only and path-focused. Training/evaluation
code can use them to avoid hand-writing K/seed/center artifact names in every
legacy wrapper.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


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

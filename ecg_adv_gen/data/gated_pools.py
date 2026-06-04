"""Prompt-token gated pool artifact contracts and small-file writers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ecg_adv_gen.labels.super5 import CLASS_NAMES_SUPER5


class GatedPoolError(ValueError):
    """Raised when gated pool artifacts violate the expected contract."""


@dataclass(frozen=True)
class GatedPoolArtifactPaths:
    """Canonical file layout for prompt-token gated pools."""

    out_dir: Path

    @classmethod
    def from_dir(cls, out_dir: str | Path) -> "GatedPoolArtifactPaths":
        return cls(Path(out_dir))

    @property
    def samples(self) -> Path:
        return self.out_dir / "gated_samples.npz"

    @property
    def latents(self) -> Path:
        return self.out_dir / "gated_samples.latent.npz"

    @property
    def class_trust(self) -> Path:
        return self.out_dir / "gated_samples.class_trust.json"

    @property
    def ref_meta(self) -> Path:
        return self.out_dir / "gated_samples.ref_meta.json"

    @property
    def gate_report(self) -> Path:
        return self.out_dir / "gate_report.json"

    def manifest_paths(self) -> dict[str, Path]:
        return {
            "gated_samples_npz": self.samples,
            "gated_latent_npz": self.latents,
            "class_trust_json": self.class_trust,
            "ref_meta_json": self.ref_meta,
            "gate_report_json": self.gate_report,
        }


@dataclass(frozen=True)
class GatedClassTrust:
    class_trust: dict[str, float]
    hardcoded_zero: tuple[str, ...]


@dataclass(frozen=True)
class MergedGatedPoolResult:
    paths: GatedPoolArtifactPaths
    report_path: Path
    center: str
    tag: str
    n_samples: int
    counts: dict[str, int]
    class_trust: dict[str, float]
    ref_record_ids: list[str]


@dataclass(frozen=True)
class SelectedGatedPoolResult:
    paths: GatedPoolArtifactPaths
    report_path: Path
    center: str
    tag: str
    n_samples: int
    counts: dict[str, int]
    class_trust: dict[str, float]
    ref_record_ids: list[str]


def _scalar_text(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(value)


def _as_float32_array(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape[0] < 1:
        raise GatedPoolError(f"{name} must be non-empty")
    return arr


def _validate_same_length(*arrays: tuple[str, np.ndarray]) -> int:
    lengths = {name: arr.shape[0] for name, arr in arrays}
    unique = set(lengths.values())
    if len(unique) != 1:
        raise GatedPoolError(f"gated pool arrays have inconsistent first dimension: {lengths}")
    return next(iter(unique))


def class_counts_from_labels(
    labels: np.ndarray,
    class_names: tuple[str, ...] | list[str] = CLASS_NAMES_SUPER5,
) -> dict[str, int]:
    labels = np.asarray(labels)
    class_names = tuple(class_names)
    if labels.ndim != 2 or labels.shape[1] != len(class_names):
        raise GatedPoolError(f"labels must have shape (N, {len(class_names)}), got {labels.shape}")
    class_ids = labels.argmax(axis=1)
    return {
        cls: int(np.count_nonzero(class_ids == i))
        for i, cls in enumerate(class_names)
    }


def build_gated_class_trust(
    passed_counts: Mapping[str, int],
    *,
    min_pass_per_class: int = 1,
    allow_hyp_cd_trust: bool = False,
    class_names: tuple[str, ...] | list[str] = CLASS_NAMES_SUPER5,
) -> GatedClassTrust:
    class_names = tuple(class_names)
    class_trust = {
        cls: 1.0 if int(passed_counts.get(cls, 0)) >= int(min_pass_per_class) else 0.0
        for cls in class_names
    }
    hardcoded_zero: tuple[str, ...] = ()
    if not allow_hyp_cd_trust:
        for cls in ("HYP", "CD"):
            if cls in class_trust:
                class_trust[cls] = 0.0
        hardcoded_zero = ("HYP", "CD")
    return GatedClassTrust(class_trust=class_trust, hardcoded_zero=hardcoded_zero)


def write_gated_pool_npzs(
    paths: GatedPoolArtifactPaths | str | Path,
    *,
    signals: np.ndarray,
    raw_signal_ct: np.ndarray,
    latents: np.ndarray,
    labels: np.ndarray,
    center_name: str,
    class_names: tuple[str, ...] | list[str] = CLASS_NAMES_SUPER5,
    source_indices: np.ndarray | list[int] | None = None,
) -> GatedPoolArtifactPaths:
    if not isinstance(paths, GatedPoolArtifactPaths):
        paths = GatedPoolArtifactPaths.from_dir(paths)
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    signals = _as_float32_array(signals, name="signals")
    raw_signal_ct = _as_float32_array(raw_signal_ct, name="raw_signal_ct")
    latents = _as_float32_array(latents, name="latents")
    labels = _as_float32_array(labels, name="labels")
    n = _validate_same_length(
        ("signals", signals),
        ("raw_signal_ct", raw_signal_ct),
        ("latents", latents),
        ("labels", labels),
    )
    if source_indices is not None:
        source_indices = np.asarray(source_indices, dtype=np.int64)
        if source_indices.shape[0] != n:
            raise GatedPoolError("source_indices length does not match samples")

    payload = {
        "signals": signals,
        "raw_signal_ct": raw_signal_ct,
        "latents": latents,
        "labels": labels,
        "center_name": str(center_name),
        "class_names": np.asarray(list(class_names)),
    }
    if source_indices is not None:
        payload["source_indices"] = source_indices
    np.savez_compressed(paths.samples, **payload)
    np.savez_compressed(
        paths.latents,
        latents=latents,
        labels=labels,
        center_name=str(center_name),
        class_names=np.asarray(list(class_names)),
        **({"source_indices": source_indices} if source_indices is not None else {}),
    )
    return paths


def load_gated_pool_npzs(paths: GatedPoolArtifactPaths | str | Path) -> dict[str, Any]:
    if not isinstance(paths, GatedPoolArtifactPaths):
        paths = GatedPoolArtifactPaths.from_dir(paths)
    if not paths.samples.exists():
        raise FileNotFoundError(paths.samples)
    if not paths.latents.exists():
        raise FileNotFoundError(paths.latents)
    with np.load(paths.samples, allow_pickle=True) as samples:
        sample_blob = {key: samples[key] for key in samples.files}
    with np.load(paths.latents, allow_pickle=True) as latents:
        latent_blob = {key: latents[key] for key in latents.files}

    required = ("signals", "raw_signal_ct", "latents", "labels")
    for key in required:
        if key not in sample_blob:
            raise GatedPoolError(f"{paths.samples} missing required key {key!r}")
    for key in ("latents", "labels"):
        if key not in latent_blob:
            raise GatedPoolError(f"{paths.latents} missing required key {key!r}")
    if sample_blob["latents"].shape != latent_blob["latents"].shape:
        raise GatedPoolError("sample and latent artifacts disagree on latent shape")
    if sample_blob["labels"].shape != latent_blob["labels"].shape:
        raise GatedPoolError("sample and latent artifacts disagree on label shape")
    if not np.allclose(sample_blob["latents"], latent_blob["latents"]):
        raise GatedPoolError("sample and latent artifacts store different latents")
    if not np.allclose(sample_blob["labels"], latent_blob["labels"]):
        raise GatedPoolError("sample and latent artifacts store different labels")

    sample_center = _scalar_text(sample_blob.get("center_name", "?"))
    latent_center = _scalar_text(latent_blob.get("center_name", sample_center))
    if sample_center != latent_center:
        raise GatedPoolError(
            f"sample and latent artifacts disagree on center_name: {sample_center!r} != {latent_center!r}"
        )
    sample_class_names = tuple(
        _scalar_text(x) for x in sample_blob.get("class_names", CLASS_NAMES_SUPER5)
    )
    latent_class_names = tuple(
        _scalar_text(x) for x in latent_blob.get("class_names", sample_class_names)
    )
    if sample_class_names != latent_class_names:
        raise GatedPoolError(
            "sample and latent artifacts disagree on class_names: "
            f"{sample_class_names!r} != {latent_class_names!r}"
        )

    return {
        "signals": sample_blob["signals"].astype(np.float32),
        "raw_signal_ct": sample_blob["raw_signal_ct"].astype(np.float32),
        "sample_latents": sample_blob["latents"].astype(np.float32),
        "sample_labels": sample_blob["labels"].astype(np.float32),
        "latents": latent_blob["latents"].astype(np.float32),
        "labels": latent_blob["labels"].astype(np.float32),
        "center_name": sample_center,
        "class_names": sample_class_names,
        "source_indices": sample_blob.get("source_indices"),
    }


def write_gated_class_trust_json(
    paths: GatedPoolArtifactPaths | str | Path,
    *,
    tag: str,
    center: str,
    source_samples: str | Path,
    class_trust: Mapping[str, float],
    counts_total: Mapping[str, int],
    counts_passed: Mapping[str, int],
    policy: Mapping[str, Any] | str,
    gated_latent_pool: str | Path | None = None,
) -> Path:
    if not isinstance(paths, GatedPoolArtifactPaths):
        paths = GatedPoolArtifactPaths.from_dir(paths)
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    blob = {
        "tag": str(tag),
        "center": str(center),
        "source_samples": str(source_samples),
        "gated_latent_pool": str(gated_latent_pool or paths.latents),
        "class_trust": {cls: float(value) for cls, value in class_trust.items()},
        "policy": policy,
        "counts_total": {cls: int(value) for cls, value in counts_total.items()},
        "counts_passed": {cls: int(value) for cls, value in counts_passed.items()},
    }
    paths.class_trust.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return paths.class_trust


def write_gated_ref_meta_json(
    paths: GatedPoolArtifactPaths | str | Path,
    *,
    center: str,
    ref_record_ids: list[str] | tuple[str, ...],
    k: int | None = None,
    selection_seed: int | None = None,
    policy: str = "Exclude full prompt-token K-ref selection from downstream quick eval when available.",
) -> Path:
    if not isinstance(paths, GatedPoolArtifactPaths):
        paths = GatedPoolArtifactPaths.from_dir(paths)
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    ref_ids = sorted({str(record_id) for record_id in ref_record_ids if str(record_id)})
    blob: dict[str, Any] = {
        "center": str(center),
        "ref_record_ids": ref_ids,
        "policy": str(policy),
    }
    if k is not None:
        blob["K"] = int(k)
    if selection_seed is not None:
        blob["selection_seed"] = int(selection_seed)
    paths.ref_meta.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return paths.ref_meta


def write_selected_gated_pool_artifacts(
    out_dir: str | Path,
    *,
    tag: str,
    center: str,
    signals: np.ndarray,
    raw_signal_ct: np.ndarray,
    latents: np.ndarray,
    labels: np.ndarray,
    ref_record_ids: list[str] | tuple[str, ...],
    report_name: str,
    report_payload: Mapping[str, Any],
    trust_policy: str,
    ref_policy: str,
    allow_hyp_cd_trust: bool = False,
    class_names: tuple[str, ...] | list[str] = CLASS_NAMES_SUPER5,
) -> SelectedGatedPoolResult:
    """Write selector-style gated artifacts plus a selector-specific report."""
    paths = GatedPoolArtifactPaths.from_dir(out_dir)
    class_names = tuple(class_names)
    write_gated_pool_npzs(
        paths,
        signals=signals,
        raw_signal_ct=raw_signal_ct,
        latents=latents,
        labels=labels,
        center_name=center,
        class_names=class_names,
    )
    counts = class_counts_from_labels(labels, class_names=class_names)
    trust = build_gated_class_trust(
        counts,
        min_pass_per_class=1,
        allow_hyp_cd_trust=allow_hyp_cd_trust,
        class_names=class_names,
    )
    paths.class_trust.write_text(
        json.dumps(
            {
                "tag": str(tag),
                "center": str(center),
                "class_trust": trust.class_trust,
                "counts": counts,
                "policy": str(trust_policy),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_gated_ref_meta_json(
        paths,
        center=center,
        ref_record_ids=list(ref_record_ids),
        policy=ref_policy,
    )
    report_path = paths.out_dir / report_name
    report = dict(report_payload)
    report.update(
        {
            "tag": str(tag),
            "center": str(center),
            "n_samples": int(np.asarray(labels).shape[0]),
            "counts": counts,
            "samples": str(paths.samples),
            "latents": str(paths.latents),
            "class_trust": str(paths.class_trust),
            "ref_meta": str(paths.ref_meta),
        }
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    ref_ids = sorted({str(record_id) for record_id in ref_record_ids if str(record_id)})
    return SelectedGatedPoolResult(
        paths=paths,
        report_path=report_path,
        center=str(center),
        tag=str(tag),
        n_samples=int(np.asarray(labels).shape[0]),
        counts=counts,
        class_trust=trust.class_trust,
        ref_record_ids=ref_ids,
    )


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _merged_class_trust_from_sources(
    counts: Mapping[str, int],
    trust_blobs: list[Mapping[str, Any]],
    *,
    allow_hyp_cd_trust: bool = False,
    class_names: tuple[str, ...] | list[str] = CLASS_NAMES_SUPER5,
) -> GatedClassTrust:
    class_names = tuple(class_names)
    class_trust: dict[str, float] = {}
    for cls in class_names:
        source_values = [
            float(blob.get("class_trust", {}).get(cls, 0.0))
            for blob in trust_blobs
        ]
        class_trust[cls] = (
            1.0
            if int(counts.get(cls, 0)) > 0 and max(source_values or [0.0]) > 0.0
            else 0.0
        )
    hardcoded_zero: tuple[str, ...] = ()
    if not allow_hyp_cd_trust:
        for cls in ("HYP", "CD"):
            if cls in class_trust:
                class_trust[cls] = 0.0
        hardcoded_zero = ("HYP", "CD")
    return GatedClassTrust(class_trust=class_trust, hardcoded_zero=hardcoded_zero)


def merge_gated_pool_artifacts(
    input_dirs: list[str | Path] | tuple[str | Path, ...],
    out_dir: str | Path,
    *,
    tag: str | None = None,
    allow_hyp_cd_trust: bool = False,
    class_names: tuple[str, ...] | list[str] = CLASS_NAMES_SUPER5,
) -> MergedGatedPoolResult:
    """Merge canonical gated pool directories into one canonical output."""
    if not input_dirs:
        raise GatedPoolError("input_dirs must be non-empty")
    input_paths = [GatedPoolArtifactPaths.from_dir(path) for path in input_dirs]
    loaded = [load_gated_pool_npzs(paths) for paths in input_paths]

    center_names = [str(blob["center_name"]) for blob in loaded]
    if len(set(center_names)) != 1:
        raise GatedPoolError(f"input centers differ: {center_names}")
    center_name = center_names[0]
    tag = str(tag or center_name)
    class_names = tuple(class_names)
    loaded_class_names = [tuple(blob["class_names"]) for blob in loaded]
    if any(names != class_names for names in loaded_class_names):
        raise GatedPoolError(
            "input class_names differ from expected class order: "
            f"expected {class_names!r}, got {loaded_class_names!r}"
        )

    signals = np.concatenate([blob["signals"].astype(np.float32) for blob in loaded], axis=0)
    raw_signal_ct = np.concatenate([blob["raw_signal_ct"].astype(np.float32) for blob in loaded], axis=0)
    latents = np.concatenate([blob["latents"].astype(np.float32) for blob in loaded], axis=0)
    labels = np.concatenate([blob["labels"].astype(np.float32) for blob in loaded], axis=0)
    counts = class_counts_from_labels(labels, class_names=class_names)

    trust_blobs = [_read_json_if_exists(paths.class_trust) for paths in input_paths]
    trust = _merged_class_trust_from_sources(
        counts,
        trust_blobs,
        allow_hyp_cd_trust=allow_hyp_cd_trust,
        class_names=class_names,
    )
    ref_blobs = [_read_json_if_exists(paths.ref_meta) for paths in input_paths]
    ref_ids = sorted({
        str(record_id)
        for blob in ref_blobs
        for record_id in blob.get("ref_record_ids", [])
        if str(record_id)
    })

    paths = GatedPoolArtifactPaths.from_dir(out_dir)
    write_gated_pool_npzs(
        paths,
        signals=signals,
        raw_signal_ct=raw_signal_ct,
        latents=latents,
        labels=labels,
        center_name=center_name,
        class_names=class_names,
    )
    paths.class_trust.write_text(
        json.dumps(
            {
                "tag": tag,
                "center": center_name,
                "source_dirs": [str(Path(path)) for path in input_dirs],
                "gated_latent_pool": str(paths.latents),
                "class_trust": trust.class_trust,
                "counts": counts,
                "policy": (
                    "trust is enabled for classes with merged samples and any trusted source; "
                    f"{'/'.join(trust.hardcoded_zero)} hardcoded 0"
                    if trust.hardcoded_zero
                    else "trust is enabled for classes with merged samples and any trusted source"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_gated_ref_meta_json(
        paths,
        center=center_name,
        ref_record_ids=ref_ids,
        policy="Union of source ref ids for downstream quick-eval exclusion.",
    )
    report_path = paths.out_dir / "merge_report.json"
    report_path.write_text(
        json.dumps(
            {
                "tag": tag,
                "center": center_name,
                "source_dirs": [str(Path(path)) for path in input_dirs],
                "n_samples": int(labels.shape[0]),
                "counts": counts,
                "samples": str(paths.samples),
                "latents": str(paths.latents),
                "class_trust": str(paths.class_trust),
                "ref_meta": str(paths.ref_meta),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return MergedGatedPoolResult(
        paths=paths,
        report_path=report_path,
        center=center_name,
        tag=tag,
        n_samples=int(labels.shape[0]),
        counts=counts,
        class_trust=trust.class_trust,
        ref_record_ids=ref_ids,
    )

"""Canonical train400 latent-pool bridge for the isolated sandbox.

The whitelist trainers intentionally bind a latent pool to their outer loader.
For Diverse Latent AugMax the scientific resource is narrower: both tuning and
full-K500 refit must use the *same* frozen train400 geometry, while refit still
optimizes all 500 K500 anchors.  This module supplies that bridge without
modifying whitelist source.

The encoder batch is fixed at 64 independent of classifier/backbone batch size.
Pool encoding also runs under a scoped fail-closed deterministic policy whose
process-global PyTorch/cuDNN flags are restored in ``finally``.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
import torch.nn as nn

from core.latent_pool import LatentPool, build_latent_pool as _whitelist_build_latent_pool
from core.lhat import load_lhat_config
from data_preprocess.data_runtime import (
    RuntimeDataLoader,
    _hash_id_set_sha256 as _runtime_hash_id_set_sha256,
    get_dataloader,
)
from util.config_bundle import resolve_config_reference


CANONICAL_ENCODER_CHUNK_SIZE = 64
CANONICAL_POOL_PARTITION = "k500_tune_train"
CANONICAL_POOL_RECORD_COUNT = 400
CANONICAL_POOL_POLICY = "train400_fixed_encode_chunk64_deterministic_v1"
POOL_SOURCE_IDENTITY_KEYS = frozenset(
    {
        "split_id",
        "split_manifest_sha256",
        "source_manifest_sha256",
        "hash_id_set_sha256",
        "partition",
        "center",
        "record_count",
        "identity_sha256",
    }
)
DIVERSE_METHOD_IDS = frozenset(
    {
        "diverse_augmax_d0_random_v1",
        "diverse_augmax_d1_hard_v1",
        "diverse_augmax_d2_diverse_hard_v1",
        "diverse_augmax_d3_endpoint_hard_v1",
        "diverse_augmax_d6_endpointset_boundary",
    }
)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, description: str) -> str:
    resolved = str(value).strip().lower()
    if len(resolved) != 64 or any(
        character not in "0123456789abcdef" for character in resolved
    ):
        raise ValueError(f"{description} must be a 64-character SHA256")
    return resolved


def _hash_ids_from_selection(selection: Any) -> tuple[str, ...]:
    raw = getattr(selection, "hash_ids", None)
    if raw is None:
        raise TypeError("canonical pool loader selection lacks hash_ids")
    values = tuple(str(value) for value in raw)
    if not values or len(values) != len(set(values)):
        raise ValueError("canonical pool selection hash IDs must be non-empty and unique")
    return values


@dataclass(frozen=True)
class CanonicalPoolSourceIdentity:
    """Exact managed-loader identity bound to the train400 geometry."""

    split_id: str
    split_manifest_sha256: str
    source_manifest_sha256: str
    hash_id_set_sha256: str
    partition: str
    center: str
    record_count: int
    identity_sha256: str

    @classmethod
    def from_loader(
        cls,
        loader: Any,
        *,
        expected_partition: str | None = None,
        expected_center: str | None = None,
        expected_count: int | None = None,
    ) -> "CanonicalPoolSourceIdentity":
        dataset = getattr(loader, "dataset", None)
        selection = getattr(dataset, "selection", None)
        if selection is None:
            raise TypeError("canonical pool loader lacks a managed ECGSelection")
        selection_hash_ids = _hash_ids_from_selection(selection)
        claimed_hash_id_set_sha256 = _require_sha256(
            getattr(selection, "hash_id_set_sha256", ""),
            "selection.hash_id_set_sha256",
        )
        # Recompute with the exact data_runtime algorithm instead of trusting
        # the selection's self-reported digest.  This is the independent bridge
        # between the managed source selection and sandbox geometry identity.
        actual_hash_id_set_sha256 = _runtime_hash_id_set_sha256(selection_hash_ids)
        if claimed_hash_id_set_sha256 != actual_hash_id_set_sha256:
            raise ValueError(
                "canonical pool selection hash-ID set SHA256 mismatch: "
                f"claimed={claimed_hash_id_set_sha256}, "
                f"actual={actual_hash_id_set_sha256}"
            )
        values = {
            "split_id": str(getattr(selection, "split_id", "")),
            "split_manifest_sha256": _require_sha256(
                getattr(selection, "split_manifest_sha256", ""),
                "selection.split_manifest_sha256",
            ),
            "source_manifest_sha256": _require_sha256(
                getattr(selection, "source_manifest_sha256", ""),
                "selection.source_manifest_sha256",
            ),
            "hash_id_set_sha256": actual_hash_id_set_sha256,
            "partition": str(getattr(selection, "partition", "")),
            "center": str(getattr(selection, "logical_center", "")),
            "record_count": int(len(selection_hash_ids)),
        }
        if not values["split_id"] or not values["partition"] or not values["center"]:
            raise ValueError("canonical pool selection identity is incomplete")
        if expected_partition is not None and values["partition"] != expected_partition:
            raise ValueError(
                "canonical pool partition mismatch: "
                f"expected={expected_partition!r}, actual={values['partition']!r}"
            )
        if expected_center is not None and values["center"] != expected_center:
            raise ValueError(
                "canonical pool center mismatch: "
                f"expected={expected_center!r}, actual={values['center']!r}"
            )
        if expected_count is not None and values["record_count"] != expected_count:
            raise ValueError(
                "canonical pool record count mismatch: "
                f"expected={expected_count}, actual={values['record_count']}"
            )
        identity_sha256 = _canonical_sha256(values)
        return cls(identity_sha256=identity_sha256, **values)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CanonicalPoolSourceIdentity":
        payload = dict(value)
        if set(payload) != POOL_SOURCE_IDENTITY_KEYS:
            raise ValueError(
                "pool source identity keys must be exactly "
                f"{sorted(POOL_SOURCE_IDENTITY_KEYS)}"
            )
        identity_sha256 = _require_sha256(
            payload.pop("identity_sha256"), "pool source identity SHA256"
        )
        record_count = payload.get("record_count")
        if isinstance(record_count, bool) or not isinstance(record_count, int):
            raise TypeError("pool source identity record_count must be an integer")
        if record_count <= 0:
            raise ValueError("pool source identity record_count must be positive")
        for name in (
            "split_manifest_sha256",
            "source_manifest_sha256",
            "hash_id_set_sha256",
        ):
            payload[name] = _require_sha256(payload[name], f"pool source {name}")
        for name in ("split_id", "partition", "center"):
            payload[name] = str(payload[name])
            if not payload[name]:
                raise ValueError(f"pool source {name} must be non-empty")
        expected = _canonical_sha256(payload)
        if identity_sha256 != expected:
            raise ValueError("pool source identity SHA256 mismatch")
        return cls(identity_sha256=identity_sha256, **payload)

    def describe(self) -> dict[str, Any]:
        return {
            "split_id": self.split_id,
            "split_manifest_sha256": self.split_manifest_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "hash_id_set_sha256": self.hash_id_set_sha256,
            "partition": self.partition,
            "center": self.center,
            "record_count": self.record_count,
            "identity_sha256": self.identity_sha256,
        }


def _selection_hash_ids(loader: Any) -> tuple[str, ...]:
    selection = getattr(getattr(loader, "dataset", None), "selection", None)
    if selection is None:
        raise TypeError("canonical pool loader lacks a managed ECGSelection")
    return _hash_ids_from_selection(selection)


@dataclass(frozen=True)
class CanonicalGeometryPool:
    """A train400 geometry pool with an explicit outer-training membership.

    During tuning the outer membership is the same train400 selection.  During
    refit it is the complete K500 selection, allowing the whitelist trainer's
    coverage guards to remain active while the standardizer/radius resource
    stays frozen to train400.  Diverse runtime code consumes only geometry;
    inherited neighbor lookup still fails for K500-only hashes by design.
    """

    geometry_pool: LatentPool
    canonical_source_identity: CanonicalPoolSourceIdentity
    canonical_source_hash_ids: tuple[str, ...]
    outer_training_selection_identity: CanonicalPoolSourceIdentity
    outer_training_hash_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.canonical_source_identity.partition != CANONICAL_POOL_PARTITION:
            raise ValueError("geometry pool must be sourced from k500_tune_train")
        if self.canonical_source_identity.record_count != CANONICAL_POOL_RECORD_COUNT:
            raise ValueError("geometry pool source must contain exactly train400")
        if len(self.geometry_pool.hash_ids) != CANONICAL_POOL_RECORD_COUNT:
            raise ValueError("geometry pool must contain exactly 400 latent records")
        if (
            len(self.canonical_source_hash_ids) != CANONICAL_POOL_RECORD_COUNT
            or len(self.canonical_source_hash_ids)
            != len(set(self.canonical_source_hash_ids))
        ):
            raise ValueError("canonical geometry source must contain 400 unique hashes")
        source_hash_id_set_sha256 = _runtime_hash_id_set_sha256(
            self.canonical_source_hash_ids
        )
        if (
            source_hash_id_set_sha256
            != self.canonical_source_identity.hash_id_set_sha256
        ):
            raise ValueError(
                "canonical geometry source hashes differ from managed source identity"
            )
        if set(self.geometry_pool.hash_ids) != set(self.canonical_source_hash_ids):
            raise ValueError("geometry pool hashes differ from attached source selection")
        if (
            _runtime_hash_id_set_sha256(self.geometry_pool.hash_ids)
            != source_hash_id_set_sha256
        ):
            raise ValueError("geometry pool hash-set digest differs from canonical source")
        if len(self.outer_training_hash_ids) != len(set(self.outer_training_hash_ids)):
            raise ValueError("outer training membership contains duplicate hash IDs")
        if len(self.outer_training_hash_ids) != self.outer_training_selection_identity.record_count:
            raise ValueError("outer training hashes differ from selection record count")
        if (
            _runtime_hash_id_set_sha256(self.outer_training_hash_ids)
            != self.outer_training_selection_identity.hash_id_set_sha256
        ):
            raise ValueError(
                "outer training hashes differ from managed selection identity"
            )
        if not set(self.geometry_pool.hash_ids).issubset(set(self.outer_training_hash_ids)):
            raise ValueError("train400 geometry must be a subset of outer training membership")

    @property
    def hash_ids(self) -> tuple[str, ...]:
        """Membership expected by the whitelist outer-epoch coverage guard."""

        return self.outer_training_hash_ids

    @property
    def identity(self):
        return self.geometry_pool.identity

    @property
    def standardizer(self):
        return self.geometry_pool.standardizer

    def __len__(self) -> int:
        return len(self.geometry_pool)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.geometry_pool, name)

    def to(self, device: str | torch.device) -> "CanonicalGeometryPool":
        return replace(self, geometry_pool=self.geometry_pool.to(device))

    def bind_outer_training_loader(self, loader: Any) -> "CanonicalGeometryPool":
        identity = CanonicalPoolSourceIdentity.from_loader(
            loader,
            expected_center=self.canonical_source_identity.center,
        )
        hashes = _selection_hash_ids(loader)
        return replace(
            self,
            outer_training_selection_identity=identity,
            outer_training_hash_ids=hashes,
        )

    def describe(self) -> dict[str, Any]:
        payload = self.geometry_pool.describe()
        payload["canonical_pool_policy"] = CANONICAL_POOL_POLICY
        payload["canonical_encoder_chunk_size"] = CANONICAL_ENCODER_CHUNK_SIZE
        payload["canonical_geometry_source"] = self.canonical_source_identity.describe()
        payload["canonical_geometry_source_hash_count"] = len(
            self.canonical_source_hash_ids
        )
        payload["outer_training_selection"] = (
            self.outer_training_selection_identity.describe()
        )
        payload["outer_training_membership_count"] = len(self.outer_training_hash_ids)
        return payload


@contextmanager
def deterministic_pool_encoding() -> Iterator[None]:
    """Temporarily require deterministic encoder kernels and restore state."""

    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = bool(torch.backends.cudnn.benchmark)
    cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=False)
        yield
    finally:
        # Restore every process-global flag even when a deterministic kernel is
        # unavailable and pool construction fails closed.
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = benchmark


def _as_batch_tensor(value: Any, *, name: str, batch_size: int) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 0:
        tensor = tensor.repeat(batch_size)
    if int(tensor.shape[0]) != batch_size:
        raise ValueError(f"canonical pool {name} batch length mismatch")
    return tensor.detach().to(device="cpu").contiguous()


def _batch_hashes(value: Any, batch_size: int) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    else:
        values = tuple(value)
    hashes = tuple(str(item) for item in values)
    if len(hashes) != batch_size or any(not item for item in hashes):
        raise ValueError("canonical pool batch hash_id values are invalid")
    return hashes


def _fixed_chunk_batches(loader: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    rows: list[tuple[int, str, int, torch.Tensor, torch.Tensor]] = []
    for raw_batch in loader:
        if not isinstance(raw_batch, Mapping):
            raise TypeError("canonical pool loader batches must be mappings")
        present = [name for name in ("waveform_raw", "waveform") if name in raw_batch]
        if not present:
            raise KeyError("canonical pool batch lacks waveform_raw/waveform")
        waveform = torch.as_tensor(raw_batch[present[0]])
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 3 or tuple(waveform.shape[1:]) != (1000, 12):
            raise ValueError("canonical pool waveform must have shape (B,1000,12)")
        waveform = waveform.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not bool(torch.isfinite(waveform).all()):
            raise ValueError("canonical pool waveform contains NaN or infinity")
        batch_size = int(waveform.shape[0])
        labels = _as_batch_tensor(
            raw_batch["label"], name="label", batch_size=batch_size
        ).to(dtype=torch.float32)
        selection = _as_batch_tensor(
            raw_batch["selection_index"],
            name="selection_index",
            batch_size=batch_size,
        ).to(dtype=torch.int64)
        cache = _as_batch_tensor(
            raw_batch["cache_index"], name="cache_index", batch_size=batch_size
        ).to(dtype=torch.int64)
        hashes = _batch_hashes(raw_batch["hash_id"], batch_size)
        rows.extend(
            (
                int(selection[index]),
                hashes[index],
                int(cache[index]),
                waveform[index].clone(),
                labels[index].clone(),
            )
            for index in range(batch_size)
        )
    if not rows:
        raise ValueError("canonical pool loader is empty")
    rows.sort(key=lambda item: item[0])
    selections = tuple(item[0] for item in rows)
    hashes = tuple(item[1] for item in rows)
    caches = tuple(item[2] for item in rows)
    if len(selections) != len(set(selections)):
        raise ValueError("canonical pool has duplicate selection_index values")
    if len(hashes) != len(set(hashes)):
        raise ValueError("canonical pool has duplicate hash_id values")
    if len(caches) != len(set(caches)):
        raise ValueError("canonical pool has duplicate cache_index values")
    for start in range(0, len(rows), CANONICAL_ENCODER_CHUNK_SIZE):
        chunk = rows[start : start + CANONICAL_ENCODER_CHUNK_SIZE]
        yield {
            "waveform_raw": torch.stack([item[3] for item in chunk], dim=0),
            "label": torch.stack([item[4] for item in chunk], dim=0),
            "selection_index": torch.tensor(
                [item[0] for item in chunk], dtype=torch.int64
            ),
            "cache_index": torch.tensor(
                [item[2] for item in chunk], dtype=torch.int64
            ),
            "hash_id": tuple(item[1] for item in chunk),
            "sampling_rate_hz": torch.full((len(chunk),), 100, dtype=torch.int64),
            "physical_unit": tuple("mV" for _ in chunk),
            "normalization": tuple("none" for _ in chunk),
        }


def build_canonical_latent_pool(
    encoder: nn.Module,
    loader: Any,
    *,
    encoder_identity: str,
    device: str | torch.device | None = None,
    num_candidates: int = 20,
    standardizer_epsilon: float = 1.0e-6,
    expected_center: str | None = None,
) -> CanonicalGeometryPool:
    """Build the one canonical train400 pool using fixed encoder chunks."""

    source_identity = CanonicalPoolSourceIdentity.from_loader(
        loader,
        expected_partition=CANONICAL_POOL_PARTITION,
        expected_center=expected_center,
        expected_count=CANONICAL_POOL_RECORD_COUNT,
    )
    selection_hashes = _selection_hash_ids(loader)
    fixed_batches = tuple(_fixed_chunk_batches(loader))
    observed_hashes = tuple(
        hash_id for batch in fixed_batches for hash_id in batch["hash_id"]
    )
    if set(observed_hashes) != set(selection_hashes):
        raise RuntimeError("canonical pool iteration differs from managed selection")
    with deterministic_pool_encoding():
        geometry = _whitelist_build_latent_pool(
            encoder,
            fixed_batches,
            encoder_identity=encoder_identity,
            device=device,
            num_candidates=num_candidates,
            standardizer_epsilon=standardizer_epsilon,
        )
    return CanonicalGeometryPool(
        geometry_pool=geometry,
        canonical_source_identity=source_identity,
        canonical_source_hash_ids=selection_hashes,
        outer_training_selection_identity=source_identity,
        outer_training_hash_ids=tuple(geometry.hash_ids),
    )


def open_canonical_selection_loader(
    *,
    center: str,
    partition: str,
    config_root: str | Path,
    split_config_path: str | Path,
    data_load_config_path: str | Path,
    seed_config_path: str | Path,
    validate_values: str = "full",
    selection_resident: bool | None = None,
) -> RuntimeDataLoader:
    """Open a backbone-independent, non-shuffled managed selection loader."""

    if partition not in {"k500_tune_train", "k500"}:
        raise ValueError("canonical selection loader supports train400 or K500 only")
    resident = partition == CANONICAL_POOL_PARTITION if selection_resident is None else bool(selection_resident)
    return get_dataloader(
        dataset="pn2021",
        partition=partition,
        logical_center=center,
        sampling_rate_hz=100,
        batch_size=CANONICAL_ENCODER_CHUNK_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        cache_mode="mmap",
        validate_values=validate_values,
        selection_resident=resident,
        selection_resident_pin_memory=False,
        prepare_for_model=False,
        sanitize=False,
        global_zscore=False,
        output_layout="time_channel",
        seed_namespace=(
            "diverse_latent_augmax:canonical_pool:"
            f"{CANONICAL_POOL_POLICY}:{center}:{partition}"
        ),
        seed_config_path=seed_config_path,
        config_root=config_root,
        split_config_path=split_config_path,
        data_load_config_path=data_load_config_path,
    )


@contextmanager
def patch_canonical_pool_runtime() -> Iterator[None]:
    """Scope canonical tuning/refit pool construction to sandbox delegates."""

    import core.pn2021_tuning as tuning
    import core.train_PN2021 as train_module

    original_tuning_builder = tuning.build_latent_pool
    original_refit_builder = train_module.build_pn2021_latent_pool

    def tuning_builder(
        encoder: nn.Module,
        batches: Any,
        *,
        encoder_identity: str,
        device: str | torch.device | None = None,
        num_candidates: int = 20,
        standardizer_epsilon: float = 1.0e-6,
    ) -> CanonicalGeometryPool:
        selection = getattr(getattr(batches, "dataset", None), "selection", None)
        if getattr(selection, "partition", None) != CANONICAL_POOL_PARTITION:
            return original_tuning_builder(
                encoder,
                batches,
                encoder_identity=encoder_identity,
                device=device,
                num_candidates=num_candidates,
                standardizer_epsilon=standardizer_epsilon,
            )
        return build_canonical_latent_pool(
            encoder,
            batches,
            encoder_identity=encoder_identity,
            device=device,
            num_candidates=num_candidates,
            standardizer_epsilon=standardizer_epsilon,
            expected_center=str(getattr(selection, "logical_center", "")),
        )

    def refit_builder(
        encoder: nn.Module,
        *,
        center: str,
        model_name: str,
        method_config_path: str | Path,
        config_path: str | Path,
        config_root: str | Path | None = None,
        device: str | torch.device | None = None,
        training_parameters: Mapping[str, Any] | None = None,
        dataloader_parameters: Mapping[str, Any] | None = None,
    ) -> Any:
        method, online = train_module.load_pn2021_method_profile(
            method_config_path,
            config_path=config_path,
            config_root=config_root,
        )
        if method.profile_name not in DIVERSE_METHOD_IDS:
            return original_refit_builder(
                encoder,
                center=center,
                model_name=model_name,
                method_config_path=method_config_path,
                config_path=config_path,
                config_root=config_root,
                device=device,
                training_parameters=training_parameters,
                dataloader_parameters=dataloader_parameters,
            )
        if not method.requirements.latent_pool:
            raise ValueError("Diverse AugMax method must require a latent pool")
        resource = method.resources.get("lhat_config")
        if not isinstance(resource, Mapping):
            raise ValueError("Diverse AugMax method lacks lhat_config")
        lhat_path = resolve_config_reference(
            resource.get("path"),
            owner_config_path=method.source_path,
            config_root=online.config_root,
            description="method.resources.lhat_config",
            must_exist=True,
        )
        lhat = load_lhat_config(lhat_path, config_root=online.config_root)
        pool_loader = open_canonical_selection_loader(
            center=center,
            partition=CANONICAL_POOL_PARTITION,
            config_root=online.config_root,
            split_config_path=online.references["split_config"],
            data_load_config_path=online.references["data_load_config"],
            seed_config_path=online.references["random_seed_config"],
            validate_values="full",
            selection_resident=True,
        )
        outer_loader: RuntimeDataLoader | None = None
        try:
            pool = build_canonical_latent_pool(
                encoder,
                pool_loader,
                encoder_identity=train_module._encoder_sha256(encoder),
                device=device,
                num_candidates=lhat.num_candidates,
                standardizer_epsilon=lhat.standardizer_epsilon,
                expected_center=center,
            )
            outer_loader = open_canonical_selection_loader(
                center=center,
                partition="k500",
                config_root=online.config_root,
                split_config_path=online.references["split_config"],
                data_load_config_path=online.references["data_load_config"],
                seed_config_path=online.references["random_seed_config"],
                validate_values="sample",
                selection_resident=False,
            )
            outer_identity = CanonicalPoolSourceIdentity.from_loader(
                outer_loader,
                expected_partition="k500",
                expected_center=center,
                expected_count=500,
            )
            if outer_identity.split_id != pool.canonical_source_identity.split_id:
                raise RuntimeError("train400 and K500 use different split IDs")
            if (
                outer_identity.split_manifest_sha256
                != pool.canonical_source_identity.split_manifest_sha256
                or outer_identity.source_manifest_sha256
                != pool.canonical_source_identity.source_manifest_sha256
            ):
                raise RuntimeError("train400 and K500 use different manifest identities")
            return pool.bind_outer_training_loader(outer_loader)
        finally:
            if outer_loader is not None:
                outer_loader.close()
            pool_loader.close()

    tuning.build_latent_pool = tuning_builder
    train_module.build_pn2021_latent_pool = refit_builder
    try:
        yield
    finally:
        train_module.build_pn2021_latent_pool = original_refit_builder
        tuning.build_latent_pool = original_tuning_builder


__all__ = [
    "CANONICAL_ENCODER_CHUNK_SIZE",
    "CANONICAL_POOL_PARTITION",
    "CANONICAL_POOL_POLICY",
    "CANONICAL_POOL_RECORD_COUNT",
    "CanonicalGeometryPool",
    "CanonicalPoolSourceIdentity",
    "POOL_SOURCE_IDENTITY_KEYS",
    "build_canonical_latent_pool",
    "deterministic_pool_encoding",
    "open_canonical_selection_loader",
    "patch_canonical_pool_runtime",
]

"""Train-only ECGTwin latent pool for K500 online adversarial training.

This module is deliberately a consumer of caller-owned raw K500 batches.  It
does not open caches, resolve splits, or import the retired data layer.  The
caller must provide raw 100 Hz, 10-second, PTB-XL-order mV waveforms together
with their already-verified hash and integer identities.

The pool stores the deterministic scaled posterior mean returned by the frozen
ECGTwin encoder.  Eligibility is exact-positive-set, distinct, and non-self.
Records with fewer than the configured ``M`` candidates remain visible for
clean training, but attack access fails explicitly instead of padding or
silently dropping them.  The default mainline remains ``M=20``; bounded
candidate-count ablations use the same explicit registry as the LHAT config.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from core.lhat import LatentStandardizer, SUPPORTED_NUM_CANDIDATES
from models.vae import prepare_ecgtwin_encoder_input


LATENT_POOL_SCHEMA_VERSION = 2
MAIN_NUM_CANDIDATES = 20
EXPECTED_RAW_SHAPE = (1000, 12)
EXPECTED_LATENT_SHAPE = (4, 128)
EXPECTED_CLASS_COUNT = 5


def _update_text(digest: Any, value: Any) -> None:
    encoded = str(value).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
    digest.update(encoded)


def _sha256_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    _update_text(digest, len(values))
    for value in values:
        _update_text(digest, value)
    return digest.hexdigest()


def _sha256_tensor(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    _update_text(digest, str(tensor.dtype))
    _update_text(digest, tuple(int(size) for size in tensor.shape))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _validate_encoder_identity(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("encoder_identity must be a SHA256 string")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("encoder_identity must be a 64-character SHA256 string")
    return normalized


def _validate_generator(generator: torch.Generator, device: torch.device) -> None:
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    generator_device = torch.device(generator.device)
    if generator_device.type != device.type:
        raise ValueError("generator device must match latent device")
    if device.type == "cuda":
        current_index = torch.cuda.current_device()
        latent_index = current_index if device.index is None else device.index
        generator_index = (
            current_index
            if generator_device.index is None
            else generator_device.index
        )
        if generator_index != latent_index:
            raise ValueError("generator CUDA index must match latent CUDA index")


def _module_device(
    module: nn.Module,
    requested: str | torch.device | None,
) -> torch.device:
    devices = {
        value.device
        for value in tuple(module.parameters()) + tuple(module.buffers())
    }
    if len(devices) > 1:
        raise ValueError("VAE encoder parameters and buffers must share one device")
    current = next(iter(devices), torch.device("cpu"))
    resolved = current if requested is None else torch.device(requested)
    if devices:
        same_device = resolved == current
        if resolved.type == current.type == "cuda":
            default_index = torch.cuda.current_device()
            resolved_index = (
                default_index if resolved.index is None else resolved.index
            )
            current_index = default_index if current.index is None else current.index
            same_device = resolved_index == current_index
        if not same_device:
            raise ValueError(
                f"requested device {resolved} does not match encoder device {current}"
            )
        return current
    return resolved


def _batch_tensor(value: Any, *, name: str) -> torch.Tensor:
    try:
        result = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be tensor-compatible") from exc
    return result


def _raw_waveforms(batch: Mapping[str, Any]) -> torch.Tensor:
    present = [name for name in ("waveform", "waveform_raw") if name in batch]
    if len(present) != 1:
        raise ValueError("each batch must contain exactly one of waveform/waveform_raw")
    value = _batch_tensor(batch[present[0]], name=present[0])
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or tuple(value.shape[1:]) != EXPECTED_RAW_SHAPE:
        raise ValueError("raw K500 waveform must have shape (B,1000,12)")
    if not value.is_floating_point():
        raise TypeError("raw K500 waveform must be floating-point mV")
    value = value.to(dtype=torch.float32)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("raw K500 waveform contains NaN or Inf")
    return value


def _labels(value: Any, batch_size: int) -> torch.Tensor:
    labels = _batch_tensor(value, name="label")
    if labels.ndim == 1:
        labels = labels.unsqueeze(0)
    if labels.shape != (batch_size, EXPECTED_CLASS_COUNT):
        raise ValueError(f"label must have shape (B,{EXPECTED_CLASS_COUNT})")
    if labels.dtype == torch.bool:
        labels = labels.to(dtype=torch.float32)
    elif not labels.is_floating_point() and labels.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("label must be binary numeric values")
    labels = labels.to(dtype=torch.float32)
    if not bool(torch.isfinite(labels).all()) or not bool(
        torch.all((labels == 0.0) | (labels == 1.0))
    ):
        raise ValueError("label must contain only binary 0/1 values")
    return labels


def _indices(value: Any, batch_size: int, *, name: str) -> torch.Tensor:
    result = _batch_tensor(value, name=name)
    if result.ndim == 0:
        result = result.unsqueeze(0)
    if result.ndim != 1 or int(result.shape[0]) != batch_size:
        raise ValueError(f"{name} must have shape (B,)")
    if result.dtype == torch.bool or result.is_floating_point():
        raise TypeError(f"{name} must contain integers")
    result = result.to(dtype=torch.int64)
    if bool(torch.any(result < 0)):
        raise ValueError(f"{name} must be non-negative")
    return result


def _hash_ids(value: Any, batch_size: int) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence):
        raw_values = value
    else:
        try:
            raw_values = tuple(value)
        except TypeError as exc:
            raise TypeError("hash_id must be a string sequence") from exc
    if len(raw_values) != batch_size:
        raise ValueError("hash_id must have one value per waveform")
    if any(not isinstance(item, str) for item in raw_values):
        raise TypeError("hash_id values must be strings")
    values = tuple(str(item) for item in raw_values)
    if any(not item for item in values):
        raise ValueError("hash_id values must not be empty")
    return values


def _validate_optional_domain(batch: Mapping[str, Any], batch_size: int) -> None:
    if "sampling_rate_hz" in batch:
        rate = _batch_tensor(batch["sampling_rate_hz"], name="sampling_rate_hz")
        if rate.ndim == 0:
            rate = rate.repeat(batch_size)
        if rate.ndim != 1 or int(rate.shape[0]) != batch_size:
            raise ValueError("sampling_rate_hz must be scalar or have shape (B,)")
        if not bool(torch.all(rate == 100)):
            raise ValueError("latent-pool input must be the raw 100 Hz bottleneck")
    if "physical_unit" in batch:
        units = batch["physical_unit"]
        values = (units,) if isinstance(units, str) else tuple(units)
        if len(values) not in {1, batch_size} or any(
            str(value).lower() != "mv" for value in values
        ):
            raise ValueError("latent-pool input physical_unit must be mV")
    if "normalization" in batch:
        normalization = batch["normalization"]
        values = (
            (normalization,)
            if isinstance(normalization, str)
            else tuple(normalization)
        )
        if len(values) not in {1, batch_size} or any(
            str(value).lower() not in {"none", "raw", "raw_mv"}
            for value in values
        ):
            raise ValueError("latent-pool input must be raw, not normalized")


def _encode_mean(
    encoder: nn.Module,
    waveform: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    bridged = prepare_ecgtwin_encoder_input(waveform.to(device=device))
    with torch.inference_mode():
        result = encoder(bridged)
    if not isinstance(result, tuple) or len(result) != 3:
        raise TypeError("VAE encoder must return (scaled_latent, mean, log_variance)")
    latent, mean, log_variance = result
    if not all(isinstance(item, torch.Tensor) for item in result):
        raise TypeError("VAE encoder outputs must be torch.Tensor values")
    expected = (int(waveform.shape[0]),) + EXPECTED_LATENT_SHAPE
    if (
        latent.shape != expected
        or mean.shape != expected
        or log_variance.shape != expected
    ):
        raise ValueError(
            "VAE encoder outputs must have shape (B,4,128), "
            f"got {tuple(latent.shape)}/{tuple(mean.shape)}/{tuple(log_variance.shape)}"
        )
    if not all(bool(torch.isfinite(item).all()) for item in result):
        raise ValueError("VAE deterministic encoder output contains NaN or Inf")
    return latent.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _candidate_counts(labels: torch.Tensor) -> torch.Tensor:
    keys = [tuple(int(value) for value in row) for row in labels.tolist()]
    group_sizes: dict[tuple[int, ...], int] = {}
    for key in keys:
        group_sizes[key] = group_sizes.get(key, 0) + 1
    return torch.tensor(
        [group_sizes[key] - 1 for key in keys],
        dtype=torch.int64,
    )


def _exact_neighbor_table(
    standardized_latents: torch.Tensor,
    labels: torch.Tensor,
    candidate_counts: torch.Tensor,
) -> torch.Tensor:
    """Precompute the exact stable neighbor order once for the immutable pool.

    The calculation deliberately mirrors ``select_exact_label_candidates``:
    exact multi-hot labels, non-self candidates, flattened L2 distance and a
    stable argsort.  Padding is ``-1`` and is never exposed for an eligible
    request.  Keeping this work at pool construction removes one full-pool
    distance scan and sort per anchor from every online LHAT batch without
    changing candidate semantics.
    """

    if standardized_latents.ndim < 2 or labels.ndim != 2:
        raise ValueError("latent pool and labels must have batch axes")
    if standardized_latents.shape[0] != labels.shape[0]:
        raise ValueError("latent pool and labels must have equal record counts")
    if candidate_counts.shape != (standardized_latents.shape[0],):
        raise ValueError("candidate_counts must have one value per latent")
    if not bool(torch.isfinite(standardized_latents).all()):
        raise ValueError("standardized latent pool contains NaN or Inf")

    width = int(candidate_counts.max().item())
    table = torch.full(
        (int(standardized_latents.shape[0]), width),
        -1,
        dtype=torch.int64,
        device=standardized_latents.device,
    )
    labels_bool = labels > 0.5
    for anchor in range(int(standardized_latents.shape[0])):
        eligible = torch.all(labels_bool == labels_bool[anchor], dim=1)
        eligible[anchor] = False
        pool = torch.nonzero(eligible, as_tuple=False).flatten()
        expected = int(candidate_counts[anchor].item())
        if int(pool.numel()) != expected:
            raise RuntimeError("exact-label candidate count changed during precompute")
        if expected == 0:
            continue
        delta = standardized_latents[pool] - standardized_latents[anchor]
        distance = delta.flatten(1).norm(p=2, dim=1)
        ordered = pool[torch.argsort(distance, stable=True)]
        table[anchor, :expected] = ordered
    return table.contiguous()


@dataclass(frozen=True)
class LatentPoolIdentity:
    """Stable, manifest-ready identity for one encoded train-only pool."""

    schema_version: int
    encoder_identity: str
    record_count: int
    latent_shape: tuple[int, ...]
    num_candidates: int
    ordered_hash_ids_sha256: str
    labels_sha256: str
    latents_sha256: str
    eligibility_sha256: str
    exact_neighbor_indices_sha256: str
    standardizer_epsilon: float
    standardizer_sha256: str
    identity_sha256: str

    def describe(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "encoder_identity": self.encoder_identity,
            "record_count": self.record_count,
            "latent_shape": list(self.latent_shape),
            "num_candidates": self.num_candidates,
            "ordered_hash_ids_sha256": self.ordered_hash_ids_sha256,
            "labels_sha256": self.labels_sha256,
            "latents_sha256": self.latents_sha256,
            "eligibility_sha256": self.eligibility_sha256,
            "exact_neighbor_indices_sha256": self.exact_neighbor_indices_sha256,
            "standardizer_epsilon": self.standardizer_epsilon,
            "standardizer_sha256": self.standardizer_sha256,
            "identity_sha256": self.identity_sha256,
        }


@dataclass(frozen=True)
class LatentAttackBatch:
    """Aligned standardized anchors and exact-label non-self candidates."""

    anchor_pool_indices: torch.Tensor
    selection_indices: torch.Tensor
    cache_indices: torch.Tensor
    hash_ids: tuple[str, ...]
    labels: torch.Tensor
    anchor_standardized: torch.Tensor
    candidate_pool_indices: torch.Tensor
    candidate_hash_ids: tuple[tuple[str, ...], ...]
    candidates_standardized: torch.Tensor


@dataclass(frozen=True)
class LatentPool:
    """Immutable identity mapping plus device-movable latent tensors."""

    hash_ids: tuple[str, ...]
    selection_indices: torch.Tensor
    cache_indices: torch.Tensor
    labels: torch.Tensor
    latents: torch.Tensor
    standardized_latents: torch.Tensor
    candidate_counts: torch.Tensor
    exact_neighbor_indices: torch.Tensor
    standardizer: LatentStandardizer
    identity: LatentPoolIdentity
    _hash_to_pool: dict[str, int] = field(init=False, repr=False, compare=False)
    _selection_to_pool: dict[int, int] = field(
        init=False, repr=False, compare=False
    )
    _cache_to_pool: dict[int, int] = field(init=False, repr=False, compare=False)
    _eligible_hash_ids: tuple[str, ...] = field(
        init=False, repr=False, compare=False
    )
    _ineligible_hash_ids: tuple[str, ...] = field(
        init=False, repr=False, compare=False
    )
    _eligible_hash_id_set: frozenset[str] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_hash_to_pool",
            {value: index for index, value in enumerate(self.hash_ids)},
        )
        object.__setattr__(
            self,
            "_selection_to_pool",
            {
                int(value): index
                for index, value in enumerate(self.selection_indices.tolist())
            },
        )
        object.__setattr__(
            self,
            "_cache_to_pool",
            {
                int(value): index
                for index, value in enumerate(self.cache_indices.tolist())
            },
        )
        eligibility = (
            self.candidate_counts >= self.identity.num_candidates
        ).detach().to(device="cpu").tolist()
        eligible_hash_ids = tuple(
            value
            for value, is_eligible in zip(self.hash_ids, eligibility, strict=True)
            if is_eligible
        )
        ineligible_hash_ids = tuple(
            value
            for value, is_eligible in zip(self.hash_ids, eligibility, strict=True)
            if not is_eligible
        )
        object.__setattr__(self, "_eligible_hash_ids", eligible_hash_ids)
        object.__setattr__(self, "_ineligible_hash_ids", ineligible_hash_ids)
        object.__setattr__(
            self, "_eligible_hash_id_set", frozenset(eligible_hash_ids)
        )

    def __len__(self) -> int:
        return len(self.hash_ids)

    @property
    def eligible_mask(self) -> torch.Tensor:
        return self.candidate_counts >= self.identity.num_candidates

    @property
    def eligible_hash_ids(self) -> tuple[str, ...]:
        return self._eligible_hash_ids

    @property
    def ineligible_hash_ids(self) -> tuple[str, ...]:
        return self._ineligible_hash_ids

    @property
    def eligible_hash_id_set(self) -> frozenset[str]:
        return self._eligible_hash_id_set

    def eligibility_manifest(self) -> dict[str, Any]:
        counts = self.candidate_counts.detach().to(device="cpu").tolist()
        ineligible = [
            {"hash_id": hash_id, "candidate_count": int(count)}
            for hash_id, count in zip(self.hash_ids, counts)
            if int(count) < self.identity.num_candidates
        ]
        return {
            "policy": "exact_positive_set_nonself_distinct",
            "num_candidates": self.identity.num_candidates,
            "eligible_count": len(self.eligible_hash_ids),
            "ineligible_count": len(ineligible),
            "eligible_hash_ids": list(self.eligible_hash_ids),
            "ineligible": ineligible,
            "eligibility_sha256": self.identity.eligibility_sha256,
        }

    def describe(self) -> dict[str, Any]:
        return {
            "identity": self.identity.describe(),
            "standardizer": self.standardizer.describe(),
            "eligibility": self.eligibility_manifest(),
            "device": str(self.latents.device),
        }

    def to(self, device: str | torch.device) -> "LatentPool":
        resolved = torch.device(device)
        standardizer = LatentStandardizer(
            mean=self.standardizer.mean.to(device=resolved),
            scale=self.standardizer.scale.to(device=resolved),
            count=self.standardizer.count,
            epsilon=self.standardizer.epsilon,
            identity_sha256=self.standardizer.identity_sha256,
        )
        return LatentPool(
            hash_ids=self.hash_ids,
            selection_indices=self.selection_indices.to(device=resolved),
            cache_indices=self.cache_indices.to(device=resolved),
            labels=self.labels.to(device=resolved),
            latents=self.latents.to(device=resolved),
            standardized_latents=self.standardized_latents.to(device=resolved),
            candidate_counts=self.candidate_counts.to(device=resolved),
            exact_neighbor_indices=self.exact_neighbor_indices.to(device=resolved),
            standardizer=standardizer,
            identity=self.identity,
        )

    def _indices_from_mapping(
        self,
        values: Sequence[Any],
        mapping: Mapping[Any, int],
        *,
        name: str,
    ) -> torch.Tensor:
        if isinstance(values, (str, bytes)):
            raise TypeError(f"{name} values must be a sequence")
        requested = tuple(values)
        if not requested:
            raise ValueError(f"{name} values must not be empty")
        missing = [value for value in requested if value not in mapping]
        if missing:
            raise KeyError(
                f"{name} values are outside this latent pool: "
                f"count={len(missing)}, preview={missing[:5]}"
            )
        return torch.tensor(
            [mapping[value] for value in requested],
            dtype=torch.int64,
            device=self.latents.device,
        )

    def indices_for_hashes(self, hash_ids: Sequence[str]) -> torch.Tensor:
        if isinstance(hash_ids, (str, bytes)):
            raise TypeError("hash_ids must be a sequence of complete hash strings")
        values = tuple(hash_ids)
        if any(not isinstance(value, str) for value in values):
            raise TypeError("hash_id values must be strings")
        return self._indices_from_mapping(
            values, self._hash_to_pool, name="hash_id"
        )

    def indices_for_selection_indices(
        self, selection_indices: Sequence[int] | torch.Tensor
    ) -> torch.Tensor:
        values = _batch_tensor(selection_indices, name="selection_indices")
        if values.ndim == 0:
            values = values.unsqueeze(0)
        if (
            values.ndim != 1
            or values.dtype == torch.bool
            or values.is_floating_point()
        ):
            raise TypeError(
                "selection_index values must be a one-dimensional integer sequence"
            )
        return self._indices_from_mapping(
            tuple(int(value) for value in values.detach().to(device="cpu").tolist()),
            self._selection_to_pool,
            name="selection_index",
        )

    def indices_for_cache_indices(
        self, cache_indices: Sequence[int] | torch.Tensor
    ) -> torch.Tensor:
        values = _batch_tensor(cache_indices, name="cache_indices")
        if values.ndim == 0:
            values = values.unsqueeze(0)
        if (
            values.ndim != 1
            or values.dtype == torch.bool
            or values.is_floating_point()
        ):
            raise TypeError(
                "cache_index values must be a one-dimensional integer sequence"
            )
        return self._indices_from_mapping(
            tuple(int(value) for value in values.detach().to(device="cpu").tolist()),
            self._cache_to_pool,
            name="cache_index",
        )

    def get_attack_batch_by_pool_indices(
        self,
        pool_indices: Sequence[int] | torch.Tensor,
        *,
        mode: str,
        local_pool_size: int,
        generator: torch.Generator,
    ) -> LatentAttackBatch:
        anchors = _batch_tensor(pool_indices, name="pool_indices")
        if anchors.ndim == 0:
            anchors = anchors.unsqueeze(0)
        if (
            anchors.ndim != 1
            or anchors.dtype == torch.bool
            or anchors.is_floating_point()
        ):
            raise TypeError("pool_indices must be a one-dimensional integer sequence")
        anchors = anchors.to(device=self.latents.device, dtype=torch.int64)
        if anchors.numel() == 0:
            raise ValueError("pool_indices must not be empty")
        if bool(torch.any(anchors < 0)) or bool(torch.any(anchors >= len(self))):
            raise IndexError("pool index is outside this latent pool")

        counts = self.candidate_counts.index_select(0, anchors)
        bad_positions = torch.nonzero(
            counts < self.identity.num_candidates, as_tuple=False
        ).flatten()
        if bad_positions.numel():
            details = []
            for position in bad_positions.detach().to(device="cpu").tolist():
                pool_index = int(anchors[position].item())
                details.append(
                    f"{self.hash_ids[pool_index]} has {int(counts[position].item())}"
                )
            raise ValueError(
                "ineligible latent anchors requested: "
                + ", ".join(details)
                + f" distinct non-self candidates; requires {self.identity.num_candidates}"
            )

        if mode not in {"nearest", "local_random"}:
            raise ValueError("mode must be nearest or local_random")
        if int(local_pool_size) < self.identity.num_candidates:
            raise ValueError("local_pool_size must be at least num_candidates")
        _validate_generator(generator, self.standardized_latents.device)
        neighbor_rows = self.exact_neighbor_indices.index_select(0, anchors)
        if mode == "nearest":
            candidate_indices = neighbor_rows[:, : self.identity.num_candidates]
        else:
            # Preserve the historical generator contract: one randperm call
            # per anchor in caller order, with the same local-pool length.
            selected_rows: list[torch.Tensor] = []
            for row, count in zip(
                neighbor_rows,
                counts.detach().to(device="cpu").tolist(),
                strict=True,
            ):
                local_count = min(int(local_pool_size), int(count))
                local = row[:local_count]
                order = torch.randperm(
                    local_count,
                    device=local.device,
                    generator=generator,
                )
                selected_rows.append(
                    local[order[: self.identity.num_candidates]]
                )
            candidate_indices = torch.stack(selected_rows, dim=0)
        anchor_list = anchors.detach().to(device="cpu").tolist()
        candidate_list = candidate_indices.detach().to(device="cpu").tolist()
        return LatentAttackBatch(
            anchor_pool_indices=anchors,
            selection_indices=self.selection_indices.index_select(0, anchors),
            cache_indices=self.cache_indices.index_select(0, anchors),
            hash_ids=tuple(self.hash_ids[index] for index in anchor_list),
            labels=self.labels.index_select(0, anchors),
            anchor_standardized=self.standardized_latents.index_select(0, anchors),
            candidate_pool_indices=candidate_indices,
            candidate_hash_ids=tuple(
                tuple(self.hash_ids[index] for index in row)
                for row in candidate_list
            ),
            candidates_standardized=self.standardized_latents[candidate_indices],
        )

    def get_attack_batch_by_hashes(
        self,
        hash_ids: Sequence[str],
        *,
        mode: str,
        local_pool_size: int,
        generator: torch.Generator,
    ) -> LatentAttackBatch:
        return self.get_attack_batch_by_pool_indices(
            self.indices_for_hashes(hash_ids),
            mode=mode,
            local_pool_size=local_pool_size,
            generator=generator,
        )

    def get_attack_batch_by_selection_indices(
        self,
        selection_indices: Sequence[int] | torch.Tensor,
        *,
        mode: str,
        local_pool_size: int,
        generator: torch.Generator,
    ) -> LatentAttackBatch:
        return self.get_attack_batch_by_pool_indices(
            self.indices_for_selection_indices(selection_indices),
            mode=mode,
            local_pool_size=local_pool_size,
            generator=generator,
        )

    def get_attack_batch_by_cache_indices(
        self,
        cache_indices: Sequence[int] | torch.Tensor,
        *,
        mode: str,
        local_pool_size: int,
        generator: torch.Generator,
    ) -> LatentAttackBatch:
        return self.get_attack_batch_by_pool_indices(
            self.indices_for_cache_indices(cache_indices),
            mode=mode,
            local_pool_size=local_pool_size,
            generator=generator,
        )


def _build_identity(
    *,
    encoder_identity: str,
    hash_ids: tuple[str, ...],
    selection_indices: torch.Tensor,
    cache_indices: torch.Tensor,
    labels: torch.Tensor,
    latents: torch.Tensor,
    candidate_counts: torch.Tensor,
    eligible_mask: torch.Tensor,
    exact_neighbor_indices: torch.Tensor,
    standardizer: LatentStandardizer,
    num_candidates: int,
) -> LatentPoolIdentity:
    ordered_hash_ids_sha256 = _sha256_strings(hash_ids)
    labels_sha256 = _sha256_tensor(labels)
    latents_sha256 = _sha256_tensor(latents)
    eligibility_digest = hashlib.sha256()
    for hash_id, count, eligible in zip(
        hash_ids,
        candidate_counts.tolist(),
        eligible_mask.tolist(),
    ):
        _update_text(eligibility_digest, hash_id)
        _update_text(eligibility_digest, int(count))
        _update_text(eligibility_digest, int(bool(eligible)))
    eligibility_sha256 = eligibility_digest.hexdigest()
    exact_neighbor_indices_sha256 = _sha256_tensor(exact_neighbor_indices)

    identity_digest = hashlib.sha256()
    for value in (
        LATENT_POOL_SCHEMA_VERSION,
        encoder_identity,
        len(hash_ids),
        EXPECTED_LATENT_SHAPE,
        num_candidates,
        ordered_hash_ids_sha256,
        _sha256_tensor(selection_indices),
        _sha256_tensor(cache_indices),
        labels_sha256,
        latents_sha256,
        eligibility_sha256,
        exact_neighbor_indices_sha256,
        standardizer.epsilon,
        standardizer.identity_sha256,
    ):
        _update_text(identity_digest, value)
    return LatentPoolIdentity(
        schema_version=LATENT_POOL_SCHEMA_VERSION,
        encoder_identity=encoder_identity,
        record_count=len(hash_ids),
        latent_shape=EXPECTED_LATENT_SHAPE,
        num_candidates=num_candidates,
        ordered_hash_ids_sha256=ordered_hash_ids_sha256,
        labels_sha256=labels_sha256,
        latents_sha256=latents_sha256,
        eligibility_sha256=eligibility_sha256,
        exact_neighbor_indices_sha256=exact_neighbor_indices_sha256,
        standardizer_epsilon=standardizer.epsilon,
        standardizer_sha256=standardizer.identity_sha256,
        identity_sha256=identity_digest.hexdigest(),
    )


def build_latent_pool(
    encoder: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    *,
    encoder_identity: str,
    device: str | torch.device | None = None,
    num_candidates: int = MAIN_NUM_CANDIDATES,
    standardizer_epsilon: float = 1e-6,
) -> LatentPool:
    """Encode caller-supplied raw K500 records into a strict train-only pool.

    ``batches`` may contain DataLoader-style batches or scalar records.  Every
    mapping must expose ``waveform`` (or ``waveform_raw``), ``label``,
    ``hash_id``, ``selection_index``, and ``cache_index``.  Pool order is
    canonicalized by ``selection_index`` so batching and mmap access order do
    not change its identity.
    """

    if not isinstance(encoder, nn.Module):
        raise TypeError("encoder must be a torch.nn.Module")
    if encoder.training or any(
        parameter.requires_grad for parameter in encoder.parameters()
    ):
        raise ValueError("latent-pool construction requires a frozen eval-mode encoder")
    if (
        not isinstance(num_candidates, int)
        or isinstance(num_candidates, bool)
        or num_candidates not in SUPPORTED_NUM_CANDIDATES
    ):
        raise ValueError(
            "the latent-pool contract requires num_candidates in "
            f"{sorted(SUPPORTED_NUM_CANDIDATES)}"
        )
    resolved_num_candidates = int(num_candidates)
    epsilon = float(standardizer_epsilon)
    if not bool(torch.isfinite(torch.tensor(epsilon))) or epsilon <= 0.0:
        raise ValueError("standardizer_epsilon must be finite and positive")
    resolved_encoder_identity = _validate_encoder_identity(encoder_identity)
    resolved_device = _module_device(encoder, device)

    all_hash_ids: list[str] = []
    label_batches: list[torch.Tensor] = []
    selection_batches: list[torch.Tensor] = []
    cache_batches: list[torch.Tensor] = []
    latent_batches: list[torch.Tensor] = []
    for raw_batch in batches:
        if not isinstance(raw_batch, Mapping):
            raise TypeError("each latent-pool input batch must be a mapping")
        missing = {
            name
            for name in ("label", "hash_id", "selection_index", "cache_index")
            if name not in raw_batch
        }
        if missing:
            raise KeyError(
                f"latent-pool input batch missing fields: {sorted(missing)}"
            )
        waveform = _raw_waveforms(raw_batch)
        batch_size = int(waveform.shape[0])
        _validate_optional_domain(raw_batch, batch_size)
        label = _labels(raw_batch["label"], batch_size)
        selection = _indices(
            raw_batch["selection_index"], batch_size, name="selection_index"
        )
        cache = _indices(raw_batch["cache_index"], batch_size, name="cache_index")
        hash_ids = _hash_ids(raw_batch["hash_id"], batch_size)

        latent_batches.append(
            _encode_mean(encoder, waveform, device=resolved_device)
        )
        label_batches.append(label.to(device="cpu"))
        selection_batches.append(selection.to(device="cpu"))
        cache_batches.append(cache.to(device="cpu"))
        all_hash_ids.extend(hash_ids)

    if not latent_batches:
        raise ValueError("cannot build a latent pool from zero records")
    labels = torch.cat(label_batches, dim=0).contiguous()
    selection_indices = torch.cat(selection_batches, dim=0).contiguous()
    cache_indices = torch.cat(cache_batches, dim=0).contiguous()
    latents = torch.cat(latent_batches, dim=0).contiguous()
    if len(set(all_hash_ids)) != len(all_hash_ids):
        raise ValueError("duplicate hash_id values are not allowed in a latent pool")
    if torch.unique(selection_indices).numel() != selection_indices.numel():
        raise ValueError("duplicate selection_index values are not allowed")
    if torch.unique(cache_indices).numel() != cache_indices.numel():
        raise ValueError("duplicate cache_index values are not allowed")

    order = torch.argsort(selection_indices, stable=True)
    labels = labels.index_select(0, order)
    selection_indices = selection_indices.index_select(0, order)
    cache_indices = cache_indices.index_select(0, order)
    latents = latents.index_select(0, order)
    hash_ids = tuple(all_hash_ids[int(index)] for index in order.tolist())

    candidate_counts = _candidate_counts(labels)
    eligible_mask = candidate_counts >= resolved_num_candidates
    eligible_latents = latents[eligible_mask]
    if eligible_latents.shape[0] < 2:
        raise ValueError(
            "fewer than two "
            f"M={resolved_num_candidates}-eligible train records; "
            "cannot fit LatentStandardizer"
        )
    standardizer = LatentStandardizer.fit(
        eligible_latents,
        epsilon=epsilon,
    )
    standardized_latents = standardizer.transform(latents).contiguous()
    exact_neighbor_indices = _exact_neighbor_table(
        standardized_latents,
        labels,
        candidate_counts,
    )
    identity = _build_identity(
        encoder_identity=resolved_encoder_identity,
        hash_ids=hash_ids,
        selection_indices=selection_indices,
        cache_indices=cache_indices,
        labels=labels,
        latents=latents,
        candidate_counts=candidate_counts,
        eligible_mask=eligible_mask,
        exact_neighbor_indices=exact_neighbor_indices,
        standardizer=standardizer,
        num_candidates=resolved_num_candidates,
    )
    return LatentPool(
        hash_ids=hash_ids,
        selection_indices=selection_indices,
        cache_indices=cache_indices,
        labels=labels,
        latents=latents,
        standardized_latents=standardized_latents,
        candidate_counts=candidate_counts,
        exact_neighbor_indices=exact_neighbor_indices,
        standardizer=standardizer,
        identity=identity,
    )


__all__ = [
    "LATENT_POOL_SCHEMA_VERSION",
    "MAIN_NUM_CANDIDATES",
    "LatentAttackBatch",
    "LatentPool",
    "LatentPoolIdentity",
    "build_latent_pool",
]

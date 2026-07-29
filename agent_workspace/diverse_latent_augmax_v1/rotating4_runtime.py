"""Scoped deterministic rotating-4 corruption schedule for the A7 sandbox.

The whitelist trainer owns model construction, normalization, optimization,
BatchNorm accounting and run records.  This adapter changes only the
supervised corruption exposure plan for method profiles that explicitly carry
the ``rotating4_schedule`` contract:

``clean + 2 depth-2 + 2 depth-3 -> one optimizer update``.

Per-record SHA256 offsets and an epoch stride of two cover every canonical
depth-2/depth-3 composition once in five epochs.  The four host-side sentinel
indices exist only to delimit one accumulated optimizer group; the wrapped
runtime receives the actual per-record canonical composition tensor.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn as nn

from core.methods import CompiledMethod


ROTATING4_EXPOSURE_POLICY = "clean_aux_once_then_rotating_depth23_2plus2"
ROTATING4_SCHEDULE_POLICY = "stable_hash_depth_balanced_rotating_2plus2_v1"
ROTATING4_OFFSET_NAMESPACE = "pn2021_rotating_depth23_2plus2_v1"
ROTATING4_CONTRACT_KEY = "rotating4_schedule"
ROTATING4_EXECUTION_SLOT_SENTINELS = (0, 1, 10, 19)
ROTATING4_SLOT_BY_SENTINEL = {
    0: (0, 0),
    1: (0, 1),
    10: (10, 0),
    19: (10, 1),
}
ROTATING4_CYCLE_EPOCHS = 5
ROTATING4_EPOCH_STRIDE = 2


def _mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a mapping")
    return dict(value)


def is_rotating4_method(method: Any) -> bool:
    return (
        isinstance(method, CompiledMethod)
        and ROTATING4_CONTRACT_KEY in method.contracts
    )


def validate_rotating4_method(method: CompiledMethod) -> dict[str, Any]:
    """Fail closed on schedule, family weighting and optimizer semantics."""

    contracts = method.contracts
    schedule = _mapping(
        contracts.get(ROTATING4_CONTRACT_KEY),
        ROTATING4_CONTRACT_KEY,
    )
    expected_schedule = {
        "schema_version": 1,
        "policy": ROTATING4_SCHEDULE_POLICY,
        "depth2_composition_indices": list(range(10)),
        "depth3_composition_indices": list(range(10, 20)),
        "views_per_family_per_epoch": 2,
        "cycle_epochs": ROTATING4_CYCLE_EPOCHS,
        "epoch_stride_within_family": ROTATING4_EPOCH_STRIDE,
        "record_offset": "sha256(namespace_pipe_hash_id)_uint64_be_mod10",
        "offset_namespace": ROTATING4_OFFSET_NAMESPACE,
        "same_offset_for_depth2_and_depth3": True,
        "coverage": (
            "each_record_sees_all_10_depth2_and_all_10_depth3_"
            "compositions_once_per_5_epochs"
        ),
        "parameter_rng": (
            "existing_comparison_rng_identity_and_exposure_identity"
        ),
    }
    if schedule != expected_schedule:
        raise ValueError("rotating4 schedule contract drifted")
    if contracts.get("exposure_policy") != ROTATING4_EXPOSURE_POLICY:
        raise ValueError("rotating4 exposure policy drifted")
    if tuple(contracts.get("composition_indices", ())) != tuple(range(20)):
        raise ValueError("rotating4 canonical composition universe drifted")
    if tuple(contracts.get("rotating4_execution_slot_sentinels", ())) != (
        ROTATING4_EXECUTION_SLOT_SENTINELS
    ):
        raise ValueError("rotating4 execution-slot sentinels drifted")
    if contracts.get("clean_exposures_per_base_record") != 1:
        raise ValueError("rotating4 requires one clean exposure")
    if contracts.get("corrupted_exposures_per_base_record") != 4:
        raise ValueError("rotating4 requires four corruption exposures")
    if contracts.get("total_exposures_per_base_record") != 5:
        raise ValueError("rotating4 requires five total exposure executions")
    expected_family = {
        "clean": 0.5,
        "corrupted_total": 0.5,
        "corrupted_per_composition": 0.125,
    }
    if dict(contracts.get("family_loss_weights", {})) != expected_family:
        raise ValueError("rotating4 family-loss weights drifted")
    if contracts.get("optimizer_step_policy") != (
        "accumulate_family_balanced_once_per_base_batch"
    ):
        raise ValueError("rotating4 optimizer-step policy drifted")
    if contracts.get("batch_norm_running_stats_policy") != (
        "family_loss_weighted_once_per_base_batch"
    ):
        raise ValueError("rotating4 BatchNorm policy drifted")
    if contracts.get("heldout_target_feedback_allowed") is not False:
        raise ValueError("rotating4 must forbid held-out target feedback")
    return schedule


def stable_record_offset(hash_id: str) -> int:
    value = str(hash_id)
    if not value:
        raise ValueError("rotating4 hash IDs must be non-empty")
    payload = f"{ROTATING4_OFFSET_NAMESPACE}|{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 10


def _epoch_from_rng_identity(rng_identity: Sequence[str]) -> int:
    if isinstance(rng_identity, (str, bytes)):
        raise TypeError("rotating4 rng_identity must be a sequence")
    epochs = [
        value.split("=", 1)[1]
        for raw in rng_identity
        for value in (str(raw),)
        if value.startswith("epoch=")
    ]
    if len(epochs) != 1:
        raise ValueError("rotating4 RNG identity must contain exactly one epoch")
    try:
        epoch = int(epochs[0])
    except ValueError as exc:
        raise ValueError("rotating4 epoch identity must be an integer") from exc
    if epoch <= 0:
        raise ValueError("rotating4 epoch must be positive")
    return epoch


def rotating_composition_indices(
    hash_ids: Sequence[str],
    *,
    epoch: int,
    slot_sentinel: int,
    device: torch.device,
) -> torch.Tensor:
    """Return one canonical composition per record for an execution slot."""

    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ValueError("rotating4 epoch must be a positive integer")
    try:
        family_base, within_family = ROTATING4_SLOT_BY_SENTINEL[
            int(slot_sentinel)
        ]
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            "rotating4 slot sentinel must be one of "
            f"{ROTATING4_EXECUTION_SLOT_SENTINELS}"
        ) from None
    cycle_epoch = (epoch - 1) % ROTATING4_CYCLE_EPOCHS
    indices = [
        family_base
        + (
            stable_record_offset(str(hash_id))
            + ROTATING4_EPOCH_STRIDE * cycle_epoch
            + within_family
        )
        % 10
        for hash_id in hash_ids
    ]
    return torch.tensor(indices, device=device, dtype=torch.int64)


class _Rotating4MethodRuntime:
    def __init__(self, delegate: Any, method: CompiledMethod) -> None:
        self._delegate = delegate
        self.method = method
        self.augmix_config = getattr(delegate, "augmix_config", None)
        self.rng_seed_config_paths = getattr(
            delegate, "rng_seed_config_paths", {}
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def describe(self) -> dict[str, Any]:
        payload = dict(self._delegate.describe())
        payload["rotating4_supervised_schedule"] = validate_rotating4_method(
            self.method
        )
        payload["rotating4_execution_slot_sentinels"] = list(
            ROTATING4_EXECUTION_SLOT_SENTINELS
        )
        payload["wrapped_runtime_type"] = payload.get("type")
        payload["type"] = "sandbox_rotating4_supervised_runtime"
        return payload

    def set_performance_timer(self, timer: Any | None) -> None:
        setter = getattr(self._delegate, "set_performance_timer", None)
        if setter is not None:
            if not callable(setter):
                raise TypeError(
                    "wrapped runtime set_performance_timer must be callable"
                )
            setter(timer)

    def generate(
        self,
        *,
        composition_index_hint: int | None = None,
        **kwargs: Any,
    ):
        if composition_index_hint in ROTATING4_SLOT_BY_SENTINEL:
            hash_ids = kwargs.get("hash_ids")
            rng_identity = kwargs.get("rng_identity")
            current = kwargs.get("composition_indices")
            if isinstance(hash_ids, (str, bytes)) or not isinstance(
                hash_ids, Sequence
            ):
                raise TypeError("rotating4 generate requires hash_ids")
            if isinstance(rng_identity, (str, bytes)) or not isinstance(
                rng_identity, Sequence
            ):
                raise TypeError("rotating4 generate requires rng_identity")
            if not isinstance(current, torch.Tensor):
                raise TypeError(
                    "rotating4 corruption exposure requires composition_indices"
                )
            if current.ndim != 1 or current.shape[0] != len(hash_ids):
                raise ValueError(
                    "rotating4 composition sentinel tensor must have shape (B,)"
                )
            kwargs["composition_indices"] = rotating_composition_indices(
                tuple(str(value) for value in hash_ids),
                epoch=_epoch_from_rng_identity(rng_identity),
                slot_sentinel=int(composition_index_hint),
                device=current.device,
            )
        return self._delegate.generate(
            composition_index_hint=composition_index_hint,
            **kwargs,
        )


@contextmanager
def patch_online_trainer_runtime() -> Iterator[None]:
    """Install the rotating schedule after all existing sandbox adapters."""

    import core.online_trainer as online_trainer

    original_factory = online_trainer.build_method_runtime
    original_exposures = online_trainer._method_exposure_steps
    original_grouped_policies = (
        online_trainer.FIXED20_GROUPED_EXPOSURE_POLICIES
    )

    def patched_exposures(method: CompiledMethod):
        if not is_rotating4_method(method):
            return original_exposures(method)
        validate_rotating4_method(method)
        objective_names = {term.name for term in method.objective.terms}
        auxiliary_terms = tuple(
            str(value)
            for value in method.contracts.get("pcgrad_auxiliary_terms", ())
        )
        clean_terms = ("clean_bce", *auxiliary_terms)
        if set(clean_terms) - objective_names:
            raise ValueError("rotating4 clean/auxiliary objective terms drifted")
        if "corrupted_bce" not in objective_names:
            raise ValueError("rotating4 requires corrupted_bce")
        names = (
            "rotating_depth2_a",
            "rotating_depth2_b",
            "rotating_depth3_a",
            "rotating_depth3_b",
        )
        return (
            online_trainer._ExposureStep(
                "clean",
                -1,
                clean_terms,
                loss_scale=0.5,
            ),
            *tuple(
                online_trainer._ExposureStep(
                    name,
                    sentinel,
                    ("corrupted_bce",),
                    loss_scale=0.125,
                )
                for name, sentinel in zip(
                    names,
                    ROTATING4_EXECUTION_SLOT_SENTINELS,
                    strict=True,
                )
            ),
        )

    def patched_factory(
        method: CompiledMethod,
        *,
        model_name: str,
        config_root: str | Path,
        latent_pool: Any | None = None,
        encoder: nn.Module | None = None,
        decoder: nn.Module | None = None,
        minimum_std_mV: float = 1.0e-4,
        maximum_abs_mV: float = 20.0,
    ):
        runtime = original_factory(
            method,
            model_name=model_name,
            config_root=config_root,
            latent_pool=latent_pool,
            encoder=encoder,
            decoder=decoder,
            minimum_std_mV=minimum_std_mV,
            maximum_abs_mV=maximum_abs_mV,
        )
        if not is_rotating4_method(method):
            return runtime
        validate_rotating4_method(method)
        return _Rotating4MethodRuntime(runtime, method)

    online_trainer.FIXED20_GROUPED_EXPOSURE_POLICIES = frozenset(
        (*original_grouped_policies, ROTATING4_EXPOSURE_POLICY)
    )
    online_trainer._method_exposure_steps = patched_exposures
    online_trainer.build_method_runtime = patched_factory
    try:
        yield
    finally:
        online_trainer.build_method_runtime = original_factory
        online_trainer._method_exposure_steps = original_exposures
        online_trainer.FIXED20_GROUPED_EXPOSURE_POLICIES = (
            original_grouped_policies
        )


__all__ = [
    "ROTATING4_CYCLE_EPOCHS",
    "ROTATING4_EXECUTION_SLOT_SENTINELS",
    "ROTATING4_EXPOSURE_POLICY",
    "ROTATING4_SCHEDULE_POLICY",
    "is_rotating4_method",
    "patch_online_trainer_runtime",
    "rotating_composition_indices",
    "stable_record_offset",
    "validate_rotating4_method",
]

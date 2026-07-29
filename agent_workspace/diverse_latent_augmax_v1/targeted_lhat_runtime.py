"""Scoped class-targeted generated-view search objectives for sandbox experiments.

The Diverse AugMax runtime performs its own coefficient search and does not call
``core.lhat`` for its hard view.  For multi-label records, ordinary BCE lets the
many negative labels dominate that search.  This adapter changes only the
Diverse AugMax hard-view coefficient-search objective; fixed20 supervision,
auxiliary supervision, optimizer steps, decoder, geometry and quality gates
remain unchanged.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.methods import CompiledMethod


POSITIVE_ONLY_METHOD_ID = "fixed20_pcgrad_lhat_poshide_augmix_v1"
BALANCED_METHOD_ID = "fixed20_pcgrad_lhat_balanced_augmix_v1"
METHOD_MODES = {
    POSITIVE_ONLY_METHOD_ID: "positive_labels_only",
    BALANCED_METHOD_ID: "equal_positive_negative_families",
}
METHOD_IDS = frozenset(METHOD_MODES)
SEARCH_MODES = frozenset(
    {
        "multilabel_bce",
        *METHOD_MODES.values(),
        "cyclic_all_five_classes",
        "cyclic_abnormal_four_classes",
    }
)


@dataclass
class _TargetedState:
    generation_calls: int = 0
    hard_search_loss_calls: int = 0
    hard_search_records: int = 0
    records: int = 0
    positive_labels: int = 0
    negative_labels: int = 0
    class_search_evaluations: list[int] | None = None

    def clear(self) -> None:
        self.generation_calls = 0
        self.hard_search_loss_calls = 0
        self.hard_search_records = 0
        self.records = 0
        self.positive_labels = 0
        self.negative_labels = 0
        self.class_search_evaluations = [0, 0, 0, 0, 0]


_STATE = _TargetedState()


def _is_targeted(method: Any) -> bool:
    return isinstance(method, CompiledMethod) and (
        method.profile_name in METHOD_IDS
        or "augmax_hard_search_loss" in method.contracts
        or "augmax_hard_search_geometry_override" in method.contracts
    )


def validate_targeted_method(method: CompiledMethod) -> str:
    mode = METHOD_MODES.get(method.profile_name)
    if mode is None:
        configured = method.contracts.get("augmax_hard_search_loss")
        if configured not in SEARCH_MODES:
            raise ValueError(
                "targeted AugMax method lacks an allowlisted hard-search loss: "
                f"{method.profile_name!r}"
            )
        mode = str(configured)
    if method.contracts.get("augmax_hard_search_loss") != mode:
        raise ValueError(f"{method.profile_name} has an invalid AugMax search loss")
    if method.contracts.get("augmax_hard_search_loss_scope") != (
        "coefficient_search_only"
    ):
        raise ValueError(f"{method.profile_name} changed supervision outside search")
    if method.contracts.get("augmax_hard_search_positive_requirement") != (
        "at_least_one_positive_per_record"
    ):
        raise ValueError(f"{method.profile_name} lacks the positive-label contract")
    return mode


def _targeted_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mode: str,
    class_offset: int = 0,
) -> torch.Tensor:
    if logits.shape != targets.shape or logits.ndim != 2:
        raise ValueError("targeted LHAT logits/targets must share shape (B,C)")
    elementwise = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    if mode == "multilabel_bce":
        return elementwise.mean(dim=1)
    if mode in {"cyclic_all_five_classes", "cyclic_abnormal_four_classes"}:
        classes = (
            torch.arange(logits.shape[1], device=logits.device, dtype=torch.long)
            if mode == "cyclic_all_five_classes"
            else torch.tensor((0, 1, 2, 4), device=logits.device, dtype=torch.long)
        )
        selected = classes[
            (torch.arange(logits.shape[0], device=logits.device) + int(class_offset))
            % int(classes.numel())
        ]
        if _STATE.class_search_evaluations is None:
            _STATE.class_search_evaluations = [0, 0, 0, 0, 0]
        counts = torch.bincount(selected.detach().cpu(), minlength=5).tolist()
        for index, count in enumerate(counts):
            _STATE.class_search_evaluations[index] += int(count)
        return elementwise.gather(1, selected.unsqueeze(1)).squeeze(1)
    positive = targets >= 0.5
    negative = ~positive
    positive_count = positive.sum(dim=1)
    if bool((positive_count <= 0).any().item()):
        raise ValueError("targeted LHAT requires at least one positive label per record")
    positive_loss = (elementwise * positive).sum(dim=1) / positive_count
    if mode == "positive_labels_only":
        return positive_loss
    if mode != "equal_positive_negative_families":
        raise ValueError(f"unsupported targeted LHAT loss mode: {mode!r}")
    negative_count = negative.sum(dim=1)
    negative_loss = (elementwise * negative).sum(dim=1) / negative_count.clamp_min(1)
    return torch.where(
        negative_count > 0,
        0.5 * positive_loss + 0.5 * negative_loss,
        positive_loss,
    )


class _TargetedLHATRuntime:
    def __init__(self, runtime: Any, method: CompiledMethod) -> None:
        self.runtime = runtime
        self.method = method
        self.augmix_config = runtime.augmix_config
        self.rng_seed_config_paths = runtime.rng_seed_config_paths

    def describe(self) -> dict[str, Any]:
        payload = dict(self.runtime.describe())
        geometry = self.method.contracts.get("augmax_hard_search_geometry_override")
        payload["targeted_augmax_hard_search"] = {
            "mode": validate_targeted_method(self.method),
            "scope": "coefficient_search_only",
            "supervised_objective_changed": False,
            "geometry_override": dict(geometry) if isinstance(geometry, dict) else None,
        }
        return payload

    def generate(self, **kwargs: Any):
        import compatible_runtime

        mode = validate_targeted_method(self.method)
        epoch = 0
        for value in kwargs.get("rng_identity", ()):
            text = str(value)
            if text.startswith("epoch="):
                epoch = int(text.split("=", 1)[1])
                break
        labels = kwargs.get("targets")
        if isinstance(labels, torch.Tensor):
            _STATE.records += int(labels.shape[0])
            _STATE.positive_labels += int((labels >= 0.5).sum().detach().cpu())
            _STATE.negative_labels += int((labels < 0.5).sum().detach().cpu())
        original_loss = compatible_runtime._generated_search_loss
        original_geometry = (
            compatible_runtime.HULL_LAMBDA,
            compatible_runtime.HARD_STEPS,
            compatible_runtime.HARD_LEARNING_RATE,
        )
        geometry = self.method.contracts.get("augmax_hard_search_geometry_override")
        if geometry is not None:
            if not isinstance(geometry, dict):
                raise TypeError("AugMax hard-search geometry override must be a mapping")
            if geometry.get("scope") != "compatible_d19_generated_views_only":
                raise ValueError("AugMax geometry override has an invalid scope")
            hull_lambda = float(geometry.get("hull_lambda", -1.0))
            hard_steps = int(geometry.get("hard_steps", -1))
            hard_learning_rate = float(geometry.get("hard_learning_rate", -1.0))
            if hull_lambda not in {0.10, 0.15}:
                raise ValueError("sandbox hull_lambda must be 0.10 or 0.15")
            if hard_steps != 5 or hard_learning_rate != 0.25:
                raise ValueError("sandbox geometry requires steps=5 and lr=0.25")
            if float(geometry.get("pgd_epsilon", -1.0)) != 2.0:
                raise ValueError("sandbox geometry may not change PGD epsilon")
            compatible_runtime.HULL_LAMBDA = hull_lambda
            compatible_runtime.HARD_STEPS = hard_steps
            compatible_runtime.HARD_LEARNING_RATE = hard_learning_rate

        def targeted_loss(
            logits: torch.Tensor,
            targets: torch.Tensor,
        ) -> torch.Tensor:
            safe_logits = torch.nan_to_num(
                logits.float(), nan=0.0, posinf=30.0, neginf=-30.0
            )
            loss = _targeted_loss(
                safe_logits,
                targets.float(),
                mode=mode,
                class_offset=epoch,
            )
            if not bool(torch.isfinite(loss).all().item()):
                raise ValueError("targeted AugMax search loss became non-finite")
            _STATE.hard_search_loss_calls += 1
            _STATE.hard_search_records += int(targets.shape[0])
            return loss

        compatible_runtime._generated_search_loss = targeted_loss
        try:
            result = self.runtime.generate(**kwargs)
            _STATE.generation_calls += 1
            return result
        finally:
            compatible_runtime._generated_search_loss = original_loss
            (
                compatible_runtime.HULL_LAMBDA,
                compatible_runtime.HARD_STEPS,
                compatible_runtime.HARD_LEARNING_RATE,
            ) = original_geometry


def diagnostics_payload() -> dict[str, Any] | None:
    if _STATE.generation_calls == 0:
        return None
    return {
        "schema_version": 1,
        "generation_calls": _STATE.generation_calls,
        "hard_search_loss_calls": _STATE.hard_search_loss_calls,
        "hard_search_records": _STATE.hard_search_records,
        "records_seen_across_calls": _STATE.records,
        "positive_labels_seen": _STATE.positive_labels,
        "negative_labels_seen": _STATE.negative_labels,
        "class_search_evaluations": list(
            _STATE.class_search_evaluations or [0, 0, 0, 0, 0]
        ),
    }


def write_diagnostics(output_dir: str | Path) -> Path | None:
    payload = diagnostics_payload()
    if payload is None:
        return None
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "targeted_lhat_diagnostics.json"
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


@contextmanager
def patch_online_trainer_runtime() -> Iterator[None]:
    """Wrap only allowlisted runtimes with targeted AugMax hard search."""

    import core.online_trainer as online_trainer

    original_factory = online_trainer.build_method_runtime
    _STATE.clear()

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
        if not _is_targeted(method):
            return runtime
        validate_targeted_method(method)
        return _TargetedLHATRuntime(runtime, method)

    online_trainer.build_method_runtime = patched_factory
    try:
        yield
    finally:
        online_trainer.build_method_runtime = original_factory


__all__ = [
    "BALANCED_METHOD_ID",
    "METHOD_IDS",
    "POSITIVE_ONLY_METHOD_ID",
    "diagnostics_payload",
    "patch_online_trainer_runtime",
    "validate_targeted_method",
    "write_diagnostics",
]

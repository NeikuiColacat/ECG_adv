"""PTB-XL DataLoader adapter for the generic supervised trainer.

This module owns only PTB-XL split/cache selection. Optimisation, checkpointing
and metrics remain in :mod:`core.supervised_trainer`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from core.supervised_trainer import (
    DEFAULT_TRAIN_CONFIG,
    TrainingResult,
    load_train_config,
    train_model,
)
from data_preprocess.data_runtime import RuntimeDataLoader, get_dataloader
from models.contracts import ModelSpec
from util.config_bundle import resolve_config_reference


PTBXL_DATALOADER_PARAMETER_NAMES = frozenset(
    {
        "train_batch_size",
        "eval_batch_size",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "cache_mode",
        "validate_values",
        "drop_last",
    }
)


def _model_spec(model: nn.Module) -> ModelSpec | None:
    spec = getattr(model, "model_spec", None)
    if spec is None and hasattr(model, "module"):
        spec = getattr(model.module, "model_spec", None)
    if spec is not None and not isinstance(spec, ModelSpec):
        raise TypeError("model.model_spec must be a models.contracts.ModelSpec")
    return spec


def _sampling_rate(model: nn.Module, requested: int | None) -> int:
    spec = _model_spec(model)
    if spec is None and requested is None:
        raise ValueError(
            "sampling_rate_hz is required when model has no ModelSpec"
        )
    resolved = spec.sampling_rate_hz if requested is None else int(requested)
    if resolved not in {100, 500}:
        raise ValueError("PTB-XL sampling_rate_hz must be 100 or 500")
    if spec is not None and resolved != spec.sampling_rate_hz:
        raise ValueError(
            f"sampling_rate_hz={resolved} conflicts with {spec.name} "
            f"ModelSpec={spec.sampling_rate_hz}"
        )
    return resolved


def _resolve_dataloader_parameters(
    training: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    supplied = {} if overrides is None else dict(overrides)
    unknown = sorted(set(supplied) - PTBXL_DATALOADER_PARAMETER_NAMES)
    if unknown:
        raise ValueError(f"unknown dataloader_parameters: {unknown}")
    resolved = {
        "train_batch_size": supplied.get(
            "train_batch_size", training["train_batch_size"]
        ),
        "eval_batch_size": supplied.get(
            "eval_batch_size", training["eval_batch_size"]
        ),
        "num_workers": supplied.get("num_workers", training["num_workers"]),
        "pin_memory": supplied.get("pin_memory", training["pin_memory"]),
        "persistent_workers": supplied.get(
            "persistent_workers", training["persistent_workers"]
        ),
        "prefetch_factor": supplied.get(
            "prefetch_factor", training["prefetch_factor"]
        ),
        "cache_mode": supplied.get("cache_mode", training["cache_mode"]),
        "validate_values": supplied.get(
            "validate_values", training["validate_values"]
        ),
        "drop_last": supplied.get("drop_last", False),
    }
    if int(resolved["num_workers"]) == 0 and "persistent_workers" not in supplied:
        resolved["persistent_workers"] = False
    for key in ("train_batch_size", "eval_batch_size", "prefetch_factor"):
        value = resolved[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"resolved {key} must be a positive integer")
    workers = resolved["num_workers"]
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("resolved num_workers must be a nonnegative integer")
    for key in ("pin_memory", "persistent_workers", "drop_last"):
        if not isinstance(resolved[key], bool):
            raise ValueError(f"resolved {key} must be a boolean")
    if workers == 0 and resolved["persistent_workers"]:
        raise ValueError("persistent_workers requires num_workers > 0")
    if resolved["cache_mode"] not in {"auto", "ram", "mmap"}:
        raise ValueError("resolved cache_mode must be auto, ram or mmap")
    if workers > 0 and resolved["cache_mode"] != "mmap":
        raise ValueError("num_workers > 0 requires cache_mode='mmap'")
    if resolved["validate_values"] not in {"none", "sample", "full"}:
        raise ValueError("resolved validate_values must be none, sample or full")
    return resolved, supplied


@dataclass(frozen=True)
class PTBXLDataLoaders:
    train: RuntimeDataLoader
    validation: RuntimeDataLoader
    test: RuntimeDataLoader | None
    sampling_rate_hz: int
    resolved_parameters: dict[str, Any]
    explicit_overrides: dict[str, Any]

    def close(self) -> None:
        self.train.close()
        self.validation.close()
        if self.test is not None:
            self.test.close()

    def describe(self) -> dict[str, Any]:
        return {
            "sampling_rate_hz": self.sampling_rate_hz,
            "resolved_parameters": self.resolved_parameters,
            "explicit_overrides": self.explicit_overrides,
            "train": self.train.describe(),
            "validation": self.validation.describe(),
            "test": None if self.test is None else self.test.describe(),
        }


def build_ptbxl_dataloaders(
    model: nn.Module,
    *,
    config_path: str | Path = DEFAULT_TRAIN_CONFIG,
    config_root: str | Path | None = None,
    sampling_rate_hz: int | None = None,
    dataloader_parameters: Mapping[str, Any] | None = None,
) -> PTBXLDataLoaders:
    """Build official folds 1-8/9/10 loaders for one model contract."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    config = load_train_config(config_path, config_root=config_root)
    data = config.payload["data"]
    if data.get("dataset") != "ptbxl":
        raise ValueError("training config data.dataset must be ptbxl")
    resolved_rate = _sampling_rate(model, sampling_rate_hz)
    resolved, supplied = _resolve_dataloader_parameters(
        config.payload["training"], dataloader_parameters
    )
    references = config.payload["references"]
    split_config_path = resolve_config_reference(
        references["split_config"],
        owner_config_path=config.path,
        config_root=config.config_root,
        description="references.split_config",
        must_exist=True,
    )
    data_load_config_path = resolve_config_reference(
        references["data_load_config"],
        owner_config_path=config.path,
        config_root=config.config_root,
        description="references.data_load_config",
        must_exist=True,
    )
    common = {
        "dataset": "ptbxl",
        "sampling_rate_hz": resolved_rate,
        "num_workers": int(resolved["num_workers"]),
        "pin_memory": bool(resolved["pin_memory"]),
        "persistent_workers": bool(resolved["persistent_workers"]),
        "prefetch_factor": int(resolved["prefetch_factor"]),
        "cache_mode": resolved["cache_mode"],
        "validate_values": resolved["validate_values"],
        "prepare_for_model": True,
        "sanitize": True,
        "global_zscore": True,
        "output_layout": "channel_time",
        "seed_config_path": config.random_seed_config_path,
        "config_root": config.config_root,
        "split_config_path": split_config_path,
        "data_load_config_path": data_load_config_path,
    }
    opened: list[RuntimeDataLoader] = []
    try:
        train = get_dataloader(
            partition=str(data["train_partition"]),
            batch_size=int(resolved["train_batch_size"]),
            shuffle=True,
            drop_last=bool(resolved["drop_last"]),
            **common,
        )
        opened.append(train)
        validation = get_dataloader(
            partition=str(data["validation_partition"]),
            batch_size=int(resolved["eval_batch_size"]),
            shuffle=False,
            drop_last=False,
            **common,
        )
        opened.append(validation)
        test = None
        if bool(config.payload["selection"]["evaluate_test_at_end"]):
            test = get_dataloader(
                partition=str(data["test_partition"]),
                batch_size=int(resolved["eval_batch_size"]),
                shuffle=False,
                drop_last=False,
                **common,
            )
            opened.append(test)
    except Exception:
        for loader in opened:
            loader.close()
        raise
    return PTBXLDataLoaders(
        train=train,
        validation=validation,
        test=test,
        sampling_rate_hz=resolved_rate,
        resolved_parameters=resolved,
        explicit_overrides=supplied,
    )


def train_ptbxl(
    model: nn.Module,
    *,
    config_path: str | Path = DEFAULT_TRAIN_CONFIG,
    config_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    pos_weight: torch.Tensor | Sequence[float] | None = None,
    class_names: Sequence[str] | None = None,
    sampling_rate_hz: int | None = None,
    training_parameters: Mapping[str, Any] | None = None,
    dataloader_parameters: Mapping[str, Any] | None = None,
) -> TrainingResult:
    """Train any compatible Torch model on official PTB-XL Super5 splits."""

    config = load_train_config(config_path, config_root=config_root)
    loaders = build_ptbxl_dataloaders(
        model,
        config_path=config.path,
        config_root=config.config_root,
        sampling_rate_hz=sampling_rate_hz,
        dataloader_parameters=dataloader_parameters,
    )
    try:
        return train_model(
            model,
            loaders.train,
            validation_dataloader=loaders.validation,
            test_dataloader=loaders.test,
            config_path=config.path,
            config_root=config.config_root,
            output_dir=output_dir,
            device=device,
            pos_weight=pos_weight,
            class_names=class_names,
            training_parameters=training_parameters,
        )
    finally:
        loaders.close()


__all__ = [
    "PTBXL_DATALOADER_PARAMETER_NAMES",
    "PTBXLDataLoaders",
    "build_ptbxl_dataloaders",
    "train_ptbxl",
]

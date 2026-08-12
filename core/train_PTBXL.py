"""PTB-XL DataLoader adapter for the generic supervised trainer.

This module owns only PTB-XL split/cache selection. Optimisation, checkpointing
and metrics remain in :mod:`core.supervised_trainer`.
"""

from __future__ import annotations

import json
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
from data_preprocess.data_runtime import PTBXLLoaderPlan, RuntimeDataLoader
from models.contracts import (
    ECGFOUNDER_SPEC,
    EFFICIENTNET1DV2_SPEC,
    ModelSpec,
)
from models.input_adapter import (
    CANONICAL_SAMPLING_RATE_HZ,
    prepare_canonical_model_input,
)
from models.factory import build_model, get_model_spec
from util.config_bundle import resolve_config_reference


def _model_spec(model: nn.Module) -> ModelSpec | None:
    spec = getattr(model, "model_spec", None)
    if spec is None and hasattr(model, "module"):
        spec = getattr(model.module, "model_spec", None)
    if spec is not None and not isinstance(spec, ModelSpec):
        raise TypeError("model.model_spec must be a models.contracts.ModelSpec")
    return spec


def _managed_model_spec(model: nn.Module) -> ModelSpec:
    spec = _model_spec(model)
    if spec is None:
        raise ValueError("managed PTB-XL models must expose a ModelSpec")
    if spec not in {EFFICIENTNET1DV2_SPEC, ECGFOUNDER_SPEC}:
        raise ValueError("unsupported managed PTB-XL ModelSpec")
    return spec


@dataclass(frozen=True)
class _CanonicalInputAdapter:
    spec: ModelSpec

    def __call__(self, raw_100hz_btc: torch.Tensor) -> torch.Tensor:
        return prepare_canonical_model_input(raw_100hz_btc, self.spec)

    def describe(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": "prepare_canonical_model_input",
            "source_sampling_rate_hz": CANONICAL_SAMPLING_RATE_HZ,
            "source_layout": "time_channel",
            "model": self.spec.describe(),
        }


def _plan_model_spec(value: ModelSpec | str) -> ModelSpec:
    if isinstance(value, str):
        matches = {
            EFFICIENTNET1DV2_SPEC.name: EFFICIENTNET1DV2_SPEC,
            ECGFOUNDER_SPEC.name: ECGFOUNDER_SPEC,
        }
        try:
            return matches[value]
        except KeyError:
            raise ValueError("unsupported managed PTB-XL model name") from None
    if not isinstance(value, ModelSpec):
        raise TypeError("model_spec must be a ModelSpec or managed model name")
    if value not in {EFFICIENTNET1DV2_SPEC, ECGFOUNDER_SPEC}:
        raise ValueError("unsupported managed PTB-XL ModelSpec")
    return value


def _loader_profile(config: Any, spec: ModelSpec) -> dict[str, Any]:
    profiles = config.payload.get("boot_models")
    if not isinstance(profiles, Mapping):
        raise ValueError("training config boot_models must be a mapping")
    profile = profiles.get(spec.name)
    if not isinstance(profile, Mapping):
        raise ValueError(f"training config boot_models.{spec.name} is required")
    loader = profile.get("dataloader_parameters")
    if not isinstance(loader, Mapping):
        raise ValueError(
            f"training config boot_models.{spec.name}.dataloader_parameters "
            "must be a mapping"
        )
    expected = {"train_batch_size", "eval_batch_size"}
    if set(loader) != expected:
        raise ValueError(
            f"boot_models.{spec.name}.dataloader_parameters keys mismatch: "
            f"missing={sorted(expected - set(loader))}, "
            f"unexpected={sorted(set(loader) - expected)}"
        )
    return dict(loader)


def _build_loader_plan(config: Any, spec: ModelSpec) -> PTBXLLoaderPlan:
    data = config.payload["data"]
    training = config.payload["training"]
    profile = _loader_profile(config, spec)
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
    evaluate_test = config.payload["selection"]["evaluate_test_at_end"]
    if type(evaluate_test) is not bool:
        raise TypeError("selection.evaluate_test_at_end must be boolean")
    test_partition = data["test_partition"] if evaluate_test else None
    return PTBXLLoaderPlan(
        train_partition=data["train_partition"],
        validation_partition=data["validation_partition"],
        test_partition=test_partition,
        train_batch_size=profile["train_batch_size"],
        eval_batch_size=profile["eval_batch_size"],
        num_workers=training["num_workers"],
        pin_memory=training["pin_memory"],
        persistent_workers=training["persistent_workers"],
        prefetch_factor=training["prefetch_factor"],
        cache_mode=training["cache_mode"],
        validate_values=training["validate_values"],
        drop_last=False,
        split_config_path=split_config_path,
        data_load_config_path=data_load_config_path,
        seed_config_path=config.random_seed_config_path,
    )


def build_ptbxl_loader_plan(
    model_spec: ModelSpec | str,
    *,
    config_path: str | Path = DEFAULT_TRAIN_CONFIG,
    config_root: str | Path | None = None,
) -> PTBXLLoaderPlan:
    """Resolve and validate a finite loader plan without opening data."""

    config = load_train_config(config_path, config_root=config_root)
    if config.payload["data"].get("dataset") != "ptbxl":
        raise ValueError("training config data.dataset must be ptbxl")
    return _build_loader_plan(config, _plan_model_spec(model_spec))


def run_ptbxl_boot(model_name: str, args: Any) -> int:
    """Resolve, print, or execute one of the two finite PTB-XL boot profiles."""

    config = load_train_config(args.config, config_root=args.config_root)
    profiles = config.payload.get("boot_models")
    profile = profiles.get(model_name) if isinstance(profiles, Mapping) else None
    if not isinstance(profile, Mapping):
        raise ValueError(f"training config boot_models.{model_name} is required")
    profile = dict(profile)
    training_parameters = dict(profile.get("training_parameters") or {})
    if getattr(args, "epochs", None) is not None:
        if model_name != ECGFOUNDER_SPEC.name or args.epochs != 10:
            raise ValueError("only ECGFounder --epochs 10 is supported")
        training_parameters["epochs"] = 10
    spec = get_model_spec(model_name)
    output_dir = args.output_dir or profile.get("output_dir")
    plan = {
        "model": spec.describe(),
        "config": config.describe(),
        "output_dir": None if output_dir is None else str(Path(output_dir).resolve()),
        "device": config.payload["training"]["device"],
        "training_parameters": training_parameters,
        "loader_plan": build_ptbxl_loader_plan(
            spec, config_path=config.path, config_root=config.config_root
        ).describe(),
        "pos_weight": None,
    }
    if model_name == EFFICIENTNET1DV2_SPEC.name:
        fields = (("checkpoint_path", None),)
    elif model_name == ECGFOUNDER_SPEC.name:
        fields = (("pretrained_checkpoint_path", None), ("task_checkpoint_path", None),
                  ("trainable_scope", "full"))
    else:
        raise ValueError("unsupported managed PTB-XL model name")
    model_kwargs = {name: profile.get(name, default) for name, default in fields}
    plan.update(
        {
            name: (None if value is None else str(Path(value).resolve()))
            if name.endswith("_path") else value
            for name, value in model_kwargs.items()
        }
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    model = build_model(
        model_name,
        config_root=config.config_root,
        seed_namespace="ptbxl_model_initialization",
        seed_identity=(config.profile_name,),
        map_location="cpu",
        **model_kwargs,
    )
    result = train_ptbxl(
        model,
        config_path=config.path,
        config_root=config.config_root,
        output_dir=output_dir,
        training_parameters=training_parameters,
    )
    print(json.dumps(result.describe(), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


@dataclass(frozen=True)
class PTBXLDataLoaders:
    train: RuntimeDataLoader
    validation: RuntimeDataLoader
    test: RuntimeDataLoader | None
    plan: PTBXLLoaderPlan

    @property
    def sampling_rate_hz(self) -> int:
        return CANONICAL_SAMPLING_RATE_HZ

    def close(self) -> None:
        self.train.close()
        self.validation.close()
        if self.test is not None:
            self.test.close()

    def describe(self) -> dict[str, Any]:
        return {
            "loader_plan": self.plan.describe(),
            "train": self.train.describe(),
            "validation": self.validation.describe(),
            "test": None if self.test is None else self.test.describe(),
        }


def build_ptbxl_dataloaders(
    model: nn.Module,
    *,
    config_path: str | Path = DEFAULT_TRAIN_CONFIG,
    config_root: str | Path | None = None,
) -> PTBXLDataLoaders:
    """Build raw canonical-100 Hz loaders for official folds 1-8/9/10."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    config = load_train_config(config_path, config_root=config_root)
    data = config.payload["data"]
    if data.get("dataset") != "ptbxl":
        raise ValueError("training config data.dataset must be ptbxl")
    plan = _build_loader_plan(config, _managed_model_spec(model))
    opened: list[RuntimeDataLoader] = []
    try:
        train = plan.open_train()
        opened.append(train)
        validation = plan.open_validation()
        opened.append(validation)
        test = None
        if plan.test_partition is not None:
            test = plan.open_test()
            opened.append(test)
    except Exception:
        for loader in opened:
            loader.close()
        raise
    return PTBXLDataLoaders(
        train=train,
        validation=validation,
        test=test,
        plan=plan,
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
    training_parameters: Mapping[str, Any] | None = None,
) -> TrainingResult:
    """Train any compatible Torch model on official PTB-XL Super5 splits."""

    config = load_train_config(config_path, config_root=config_root)
    spec = _managed_model_spec(model)
    loaders = build_ptbxl_dataloaders(
        model,
        config_path=config.path,
        config_root=config.config_root,
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
            input_adapter=_CanonicalInputAdapter(spec),
        )
    finally:
        loaders.close()


__all__ = [
    "PTBXLDataLoaders",
    "build_ptbxl_dataloaders",
    "build_ptbxl_loader_plan",
    "run_ptbxl_boot",
    "train_ptbxl",
]

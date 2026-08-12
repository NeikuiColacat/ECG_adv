"""Generic YAML-managed supervised trainer for Torch ECG classifiers.

Dataset selection and preprocessing deliberately stay outside this module.
Callers build loaders through a finite dataset-specific plan and pass them
together with a model to :func:`train_model`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.checkpoints import load_model_checkpoint, sha256_file
from models.contracts import (
    CLASS_ORDER,
    ModelSpec,
    validate_model_input,
    validate_model_output,
)
from util.config_bundle import (
    config_bundle_root,
    resolve_config_reference,
    resolve_entry_config_path,
)
from util.evaluation.metrics import compute_classification_metrics
from util.random_seed import seed_process


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "train" / "PTBXL.yaml"
TRAINING_PARAMETER_NAMES = frozenset(
    {
        "epochs",
        "scheduler_horizon_epochs",
        "learning_rate",
        "weight_decay",
        "minimum_learning_rate_ratio",
        "gradient_clip_norm",
        "amp_enabled",
        "amp_dtype",
    }
)


def _section(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"training config section {key!r} must be a mapping")
    return value


@dataclass(frozen=True)
class SupervisedTrainConfig:
    path: Path
    sha256: str
    config_root: Path
    random_seed_config_path: Path
    payload: dict[str, Any]

    @property
    def profile_name(self) -> str:
        return str(self.payload["profile_name"])

    @property
    def selection_metric(self) -> str:
        return str(self.payload["selection"]["metric"])

    def describe(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "config_root": str(self.config_root),
            "random_seed_config": str(self.random_seed_config_path),
            "random_seed_config_sha256": sha256_file(self.random_seed_config_path),
            "resolved": self.payload,
        }


@dataclass(frozen=True)
class TrainingResult:
    model: nn.Module
    output_dir: Path
    selected_epoch: int
    selected_metric: float | None
    selected_checkpoint_path: Path
    last_checkpoint_path: Path
    history: tuple[dict[str, Any], ...]
    test_metrics: dict[str, Any] | None
    model_identity: dict[str, Any]
    config_identity: dict[str, Any]
    seed_identity: dict[str, Any]
    input_adapter_identity: dict[str, Any] | None = None

    def describe(self) -> dict[str, Any]:
        selected_checkpoint = {
            "path": str(self.selected_checkpoint_path),
            "sha256": sha256_file(self.selected_checkpoint_path),
        }
        resolved_config = self.config_identity.get("resolved", {})
        selection_config = (
            resolved_config.get("selection", {})
            if isinstance(resolved_config, Mapping)
            else {}
        )
        partition = selection_config.get("partition")
        metric = selection_config.get("metric")
        selection_policy = "last" if metric == "last" else "validation_metric"
        result = {
            "output_dir": str(self.output_dir),
            "selected_epoch": self.selected_epoch,
            "selected_metric": self.selected_metric,
            "selected_checkpoint": selected_checkpoint,
            "last_checkpoint": {
                "path": str(self.last_checkpoint_path),
                "sha256": sha256_file(self.last_checkpoint_path),
            },
            "epochs_completed": len(self.history),
            "test_metrics": self.test_metrics,
            "model": self.model_identity,
            "config": self.config_identity,
            "seed": self.seed_identity,
            "selection": {
                "policy": selection_policy,
                "partition": partition,
                "metric": metric,
                "selected_epoch": self.selected_epoch,
                "selected_metric": self.selected_metric,
                "selected_checkpoint": selected_checkpoint,
                "final_model_selected": True,
                "heldout_evaluation_used_for_selection": False,
            },
        }
        if self.input_adapter_identity is not None:
            result["input_adapter"] = self.input_adapter_identity
        return result


def load_train_config(
    path: str | Path = DEFAULT_TRAIN_CONFIG,
    *,
    config_root: str | Path | None = None,
) -> SupervisedTrainConfig:
    """Load one supervised-training profile from a portable config bundle."""

    config_path = resolve_entry_config_path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"training config not found: {config_path}") from None
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("training config must be a schema_version=1 mapping")
    if not isinstance(payload.get("profile_name"), str) or not payload["profile_name"]:
        raise ValueError("training profile_name must be non-empty")
    references = _section(payload, "references")
    random_seed = _section(payload, "random_seed")
    training = _section(payload, "training")
    selection = _section(payload, "selection")
    output = _section(payload, "output")

    batch_keys = _section(training, "batch_keys")
    if set(batch_keys) != {"input", "target"} or not all(
        isinstance(value, str) and value for value in batch_keys.values()
    ):
        raise ValueError("training.batch_keys must define non-empty input and target")
    optimizer = _section(training, "optimizer")
    scheduler = _section(training, "scheduler")
    loss = _section(training, "loss")
    amp = _section(training, "amp")
    if optimizer.get("name") != "adamw":
        raise ValueError("supervised trainer currently supports AdamW only")
    if scheduler.get("name") != "cosine_annealing":
        raise ValueError("supervised trainer currently supports cosine annealing only")
    if loss != {"name": "bce_with_logits", "pos_weight": "optional_argument"}:
        raise ValueError(
            "training.loss must be BCE-with-logits with optional_argument pos_weight"
        )
    if amp.get("dtype") not in {"bfloat16", "float16"}:
        raise ValueError("training.amp.dtype must be bfloat16 or float16")
    if not isinstance(random_seed.get("stream_namespace"), str) or not str(
        random_seed["stream_namespace"]
    ):
        raise ValueError("random_seed.stream_namespace must be a non-empty string")
    if not isinstance(random_seed.get("comparison_group"), str) or not str(
        random_seed["comparison_group"]
    ):
        raise ValueError("random_seed.comparison_group must be a non-empty string")
    replicate_id = random_seed.get("replicate_id")
    if (
        isinstance(replicate_id, bool)
        or not isinstance(replicate_id, int)
        or replicate_id < 0
    ):
        raise ValueError("random_seed.replicate_id must be a nonnegative integer")
    if not isinstance(random_seed.get("deterministic_algorithms"), bool):
        raise ValueError("random_seed.deterministic_algorithms must be a boolean")

    metric = selection.get("metric")
    mode = selection.get("mode")
    if metric not in {"macro_auprc", "loss", "last"}:
        raise ValueError("selection.metric must be macro_auprc, loss or last")
    expected_mode = "min" if metric == "loss" else "max" if metric == "macro_auprc" else "last"
    if mode != expected_mode:
        raise ValueError(f"selection.mode for {metric} must be {expected_mode}")
    if output.get("if_exists") != "error":
        raise ValueError("supervised trainer does not overwrite existing runs")
    if not isinstance(output.get("run_dir"), str) or not output["run_dir"]:
        raise ValueError("output.run_dir must be a non-empty path")

    root = config_bundle_root(config_path, config_root=config_root)
    return SupervisedTrainConfig(
        path=config_path,
        sha256=sha256_file(config_path),
        config_root=root,
        random_seed_config_path=resolve_config_reference(
            references["random_seed_config"],
            owner_config_path=config_path,
            config_root=root,
            description="references.random_seed_config",
            must_exist=True,
        ),
        payload=payload,
    )


def _model_spec(model: nn.Module) -> ModelSpec | None:
    spec = getattr(model, "model_spec", None)
    if spec is None and hasattr(model, "module"):
        spec = getattr(model.module, "model_spec", None)
    if spec is not None and not isinstance(spec, ModelSpec):
        raise TypeError("model.model_spec must be a models.contracts.ModelSpec")
    return spec


def _batch_tensors(
    batch: Any,
    *,
    input_key: str,
    target_key: str,
    model_spec: ModelSpec | None,
    input_adapter_present: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, Mapping):
        raise TypeError("dataloader batches must be mappings")
    inputs = batch.get(input_key)
    targets = batch.get(target_key)
    if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("configured input and target batch values must be tensors")
    if model_spec is not None and not input_adapter_present:
        validate_model_input(inputs, model_spec)
    elif inputs.ndim < 2 or not inputs.is_floating_point():
        raise ValueError("generic model inputs must be rank-2-or-higher floating tensors")
    if targets.ndim != 2 or targets.shape[0] != inputs.shape[0]:
        raise ValueError("supervised targets must have shape (B,num_classes)")
    if not targets.is_floating_point() or not bool(torch.isfinite(targets).all()):
        raise ValueError("supervised targets must be finite floating tensors")
    return inputs, targets


def _describe_input_adapter(
    input_adapter: Callable[[torch.Tensor], torch.Tensor] | None,
) -> dict[str, Any] | None:
    """Return a stable, JSON-ready identity for an optional input adapter."""

    if input_adapter is None:
        return None
    if not callable(input_adapter):
        raise TypeError("input_adapter must be callable")
    describe = getattr(input_adapter, "describe", None)
    if callable(describe):
        identity = describe()
        if not isinstance(identity, Mapping) or not identity:
            raise TypeError("input_adapter.describe() must return a non-empty mapping")
        return dict(identity)
    owner = input_adapter.__class__
    module = getattr(input_adapter, "__module__", owner.__module__)
    qualname = getattr(input_adapter, "__qualname__", owner.__qualname__)
    return {
        "schema_version": 1,
        "callable": f"{module}.{qualname}",
    }


def _apply_input_adapter(
    inputs: torch.Tensor,
    *,
    input_adapter: Callable[[torch.Tensor], torch.Tensor] | None,
    model_spec: ModelSpec | None,
) -> torch.Tensor:
    """Apply a caller-owned adapter after H2D and validate its model-domain output."""

    if input_adapter is None:
        return inputs
    batch_size = int(inputs.shape[0])
    adapted = input_adapter(inputs)
    if not isinstance(adapted, torch.Tensor):
        raise TypeError("input_adapter must return a torch.Tensor")
    if adapted.device != inputs.device:
        raise ValueError("input_adapter must preserve the input device")
    if int(adapted.shape[0]) != batch_size:
        raise ValueError("input_adapter must preserve batch size")
    if model_spec is not None:
        return validate_model_input(adapted, model_spec)
    if adapted.ndim < 2 or not adapted.is_floating_point():
        raise ValueError("adapted generic model inputs must be floating rank-2-or-higher")
    if not bool(torch.isfinite(adapted).all()):
        raise ValueError("adapted generic model inputs contain NaN or Inf")
    return adapted


def _model_logits(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    model_spec: ModelSpec | None,
) -> torch.Tensor:
    logits = model(inputs)
    if model_spec is not None:
        logits = validate_model_output(
            logits,
            model_spec,
            batch_size=int(inputs.shape[0]),
        )
    if not isinstance(logits, torch.Tensor) or logits.shape != targets.shape:
        raise ValueError("model logits must have the same shape as targets")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError("model logits must be finite floating tensors")
    return logits


def _classification_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    metrics = compute_classification_metrics(
        logits,
        targets,
        class_order=CLASS_ORDER,
        undefined_class_policy="strict",
    )
    per_class = metrics["per_class"]
    return {
        **metrics,
        "per_class_auroc": {
            class_name: per_class[class_name]["auroc"] for class_name in CLASS_ORDER
        },
        "per_class_auprc": {
            class_name: per_class[class_name]["auprc"] for class_name in CLASS_ORDER
        },
    }


def _run_epoch(
    model: nn.Module,
    dataloader: Any,
    *,
    device: torch.device,
    model_spec: ModelSpec | None,
    input_key: str,
    target_key: str,
    pos_weight: torch.Tensor | None,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    gradient_clip_norm: float = 1.0,
    on_optimizer_step: Callable[[], None] | None = None,
    input_adapter: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    collected_logits: list[np.ndarray] = []
    collected_targets: list[np.ndarray] = []
    grad_context = torch.enable_grad if training else torch.no_grad
    with grad_context():
        for batch in dataloader:
            inputs, targets = _batch_tensors(
                batch,
                input_key=input_key,
                target_key=target_key,
                model_spec=model_spec,
                input_adapter_present=input_adapter is not None,
            )
            inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device=device, dtype=torch.float32, non_blocking=True)
            inputs = _apply_input_adapter(
                inputs,
                input_adapter=input_adapter,
                model_spec=model_spec,
            )
            if pos_weight is not None and pos_weight.numel() != targets.shape[1]:
                raise ValueError("pos_weight length must match target width")
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                logits = _model_logits(model, inputs, targets, model_spec)
                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    targets,
                    pos_weight=pos_weight,
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("supervised loss became NaN or Inf")
            if training:
                assert scaler is not None
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    optimizer.step()
                if on_optimizer_step is not None:
                    on_optimizer_step()
            batch_size = int(inputs.shape[0])
            total_loss += float(loss.detach()) * batch_size
            total_samples += batch_size
            if not training:
                collected_logits.append(logits.float().cpu().numpy())
                collected_targets.append(targets.float().cpu().numpy())
    if total_samples == 0:
        raise ValueError("dataloader is empty")
    result: dict[str, Any] = {"loss": total_loss / total_samples}
    if not training:
        epoch_logits = np.concatenate(collected_logits, axis=0)
        epoch_targets = np.concatenate(collected_targets, axis=0)
        classification = _classification_metrics(epoch_logits, epoch_targets)
        result.update(classification)
    return result


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _output_member(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("output filenames must be non-empty strings")
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError("output filenames must stay inside the run directory") from None
    return path


def _describe_dataloader(dataloader: Any) -> Any:
    describe = getattr(dataloader, "describe", None)
    if callable(describe):
        return describe()
    dataset = getattr(dataloader, "dataset", None)
    return {
        "type": dataloader.__class__.__name__,
        "dataset_type": None if dataset is None else dataset.__class__.__name__,
        "sample_count": None if dataset is None else len(dataset),
        "batch_size": getattr(dataloader, "batch_size", None),
        "num_workers": getattr(dataloader, "num_workers", None),
        "drop_last": getattr(dataloader, "drop_last", None),
    }


def _describe_model(
    model: nn.Module,
    spec: ModelSpec | None,
    trainable_parameters: Sequence[nn.Parameter],
) -> dict[str, Any]:
    owner = model.module if hasattr(model, "module") else model
    result: dict[str, Any] = {
        "name": spec.name if spec is not None else owner.__class__.__name__,
        "spec": None if spec is None else spec.describe(),
        "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameter_count": int(
            sum(p.numel() for p in trainable_parameters)
        ),
    }
    for attribute in (
        "checkpoint_identity",
        "pretrained_checkpoint_identity",
        "task_checkpoint_identity",
        "random_seed_identity",
    ):
        identity = getattr(owner, attribute, None)
        describe = getattr(identity, "describe", None)
        result[attribute] = describe() if callable(describe) else None
    return result


def _resolve_training_parameters(
    training: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    supplied = {} if overrides is None else dict(overrides)
    unknown = sorted(set(supplied) - TRAINING_PARAMETER_NAMES)
    if unknown:
        raise ValueError(f"unknown training_parameters: {unknown}")
    optimizer = training["optimizer"]
    scheduler = training["scheduler"]
    amp = training["amp"]
    resolved = {
        "epochs": supplied.get("epochs", training["epochs"]),
        "scheduler_horizon_epochs": supplied.get(
            "scheduler_horizon_epochs",
            supplied.get("epochs", training["epochs"]),
        ),
        "learning_rate": supplied.get(
            "learning_rate", optimizer["learning_rate"]
        ),
        "weight_decay": supplied.get("weight_decay", optimizer["weight_decay"]),
        "minimum_learning_rate_ratio": supplied.get(
            "minimum_learning_rate_ratio",
            scheduler["minimum_learning_rate_ratio"],
        ),
        "gradient_clip_norm": supplied.get(
            "gradient_clip_norm", training["gradient_clip_norm"]
        ),
        "amp_enabled": supplied.get("amp_enabled", amp["enabled"]),
        "amp_dtype": supplied.get("amp_dtype", amp["dtype"]),
    }
    for key in ("epochs", "scheduler_horizon_epochs"):
        value = resolved[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"resolved {key} must be a positive integer")
    if resolved["scheduler_horizon_epochs"] < resolved["epochs"]:
        raise ValueError(
            "scheduler_horizon_epochs must be greater than or equal to epochs"
        )
    for key in (
        "learning_rate",
        "minimum_learning_rate_ratio",
        "gradient_clip_norm",
    ):
        value = resolved[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"resolved {key} must be numeric")
        resolved[key] = float(value)
        if not math.isfinite(resolved[key]) or resolved[key] <= 0:
            raise ValueError(f"resolved {key} must be finite and positive")
    weight_decay = resolved["weight_decay"]
    if isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float)):
        raise ValueError("resolved weight_decay must be numeric")
    resolved["weight_decay"] = float(weight_decay)
    if not math.isfinite(resolved["weight_decay"]) or resolved["weight_decay"] < 0:
        raise ValueError("resolved weight_decay must be finite and nonnegative")
    if not isinstance(resolved["amp_enabled"], bool):
        raise ValueError("resolved amp_enabled must be a boolean")
    if resolved["amp_dtype"] not in {"bfloat16", "float16"}:
        raise ValueError("resolved amp_dtype must be bfloat16 or float16")
    return resolved, supplied


def seed_training_process(config: SupervisedTrainConfig):
    """Seed one matched training replicate without method/config-name drift.

    The config SHA and profile name remain part of the run record, but are
    intentionally excluded from seed derivation.  Therefore copied YAMLs for
    Direct, LHAT, and LHAT+AugMix receive the same process seed whenever their
    explicit comparison group and replicate id match.
    """

    random_seed = config.payload["random_seed"]
    return seed_process(
        str(random_seed["stream_namespace"]),
        str(random_seed["comparison_group"]),
        int(random_seed["replicate_id"]),
        config_path=config.random_seed_config_path,
        deterministic_algorithms=bool(random_seed["deterministic_algorithms"]),
    )


def train_model(
    model: nn.Module,
    train_dataloader: Any,
    *,
    validation_dataloader: Any | None = None,
    test_dataloader: Any | None = None,
    config_path: str | Path = DEFAULT_TRAIN_CONFIG,
    config_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    pos_weight: torch.Tensor | Sequence[float] | None = None,
    class_names: Sequence[str] | None = None,
    training_parameters: Mapping[str, Any] | None = None,
    input_adapter: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> TrainingResult:
    """Train ``model`` using caller-supplied supervised dataloaders.

    The model is updated in place. Dataset/split semantics belong to the
    dataloaders, so the same function can train PTB-XL source models, PN2021
    K500 fine-tuning arms, EfficientNet1DV2, or ECGFounder. The trainer assumes
    multi-label targets and raw logits suitable for BCE-with-logits.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if train_dataloader is None:
        raise ValueError("train_dataloader is required")
    config = load_train_config(config_path, config_root=config_root)
    training = config.payload["training"]
    resolved_training, supplied_training = _resolve_training_parameters(
        training, training_parameters
    )
    selection = config.payload["selection"]
    output_config = config.payload["output"]
    metric_name = config.selection_metric
    if metric_name in {"macro_auprc", "loss"} and validation_dataloader is None:
        raise ValueError(f"selection.metric={metric_name} requires validation_dataloader")

    requested_device = str(device) if device is not None else str(training["device"])
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_device = torch.device(requested_device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if resolved_device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be CPU or CUDA")

    raw_output = output_config["run_dir"] if output_dir is None else output_dir
    output = Path(raw_output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    try:
        output.relative_to(PROJECT_ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("training outputs must remain outside the Git worktree")
    checkpoints = _output_member(output, output_config["checkpoints_dir"])
    checkpoints.mkdir(parents=True, exist_ok=False)
    selected_path = _output_member(checkpoints, output_config["best_checkpoint_file"])
    last_path = _output_member(checkpoints, output_config["last_checkpoint_file"])
    history_path = _output_member(output, output_config["history_file"])
    result_path = _output_member(output, output_config["result_file"])

    spec = _model_spec(model)
    input_adapter_identity = _describe_input_adapter(input_adapter)
    seed = seed_training_process(config)
    model.to(resolved_device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")

    optimizer = AdamW(
        trainable,
        lr=resolved_training["learning_rate"],
        weight_decay=resolved_training["weight_decay"],
    )
    epochs = int(resolved_training["epochs"])
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(resolved_training["scheduler_horizon_epochs"]),
        eta_min=(
            resolved_training["learning_rate"]
            * resolved_training["minimum_learning_rate_ratio"]
        ),
    )
    amp_enabled = resolved_training["amp_enabled"] and resolved_device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if resolved_training["amp_dtype"] == "bfloat16"
        else torch.float16
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled and amp_dtype == torch.float16
    )
    resolved_pos_weight = (
        None
        if pos_weight is None
        else torch.as_tensor(pos_weight, dtype=torch.float32, device=resolved_device)
    )
    if resolved_pos_weight is not None and resolved_pos_weight.ndim != 1:
        raise ValueError("pos_weight must be one-dimensional")

    if class_names is None:
        configured = config.payload.get("data", {}).get("class_order")
        class_names = tuple(configured) if configured else ()
    class_names = tuple(str(value) for value in class_names)
    if class_names != CLASS_ORDER:
        raise ValueError(
            f"class_names must be the canonical Super5 order {CLASS_ORDER}, "
            f"got {class_names}"
        )
    input_key = str(training["batch_keys"]["input"])
    target_key = str(training["batch_keys"]["target"])
    model_identity = _describe_model(model, spec, trainable)
    train_dataloader_description = _describe_dataloader(train_dataloader)
    validation_dataloader_description = (
        None
        if validation_dataloader is None
        else _describe_dataloader(validation_dataloader)
    )
    test_dataloader_description = (
        None if test_dataloader is None else _describe_dataloader(test_dataloader)
    )
    run_identity = {
        "config": config.describe(),
        "seed": seed.describe(),
        "model": model_identity,
        "device": str(resolved_device),
        "output_dir": str(output),
        "training_parameters": {
            "resolved": resolved_training,
            "explicit_overrides": supplied_training,
        },
        "pos_weight": (
            None
            if resolved_pos_weight is None
            else resolved_pos_weight.cpu().tolist()
        ),
        "dataloaders": {
            "train": train_dataloader_description,
            "validation": validation_dataloader_description,
            "test": test_dataloader_description,
        },
    }
    if input_adapter_identity is not None:
        run_identity["input_adapter"] = input_adapter_identity

    history: list[dict[str, Any]] = []
    selected_epoch = 0
    selected_metric: float | None = None
    best_score = math.inf if metric_name == "loss" else -math.inf
    global_step = 0

    def on_optimizer_step() -> None:
        nonlocal global_step
        global_step += 1

    for epoch in range(1, epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = _run_epoch(
            model,
            train_dataloader,
            device=resolved_device,
            model_spec=spec,
            input_key=input_key,
            target_key=target_key,
            pos_weight=resolved_pos_weight,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            optimizer=optimizer,
            scaler=scaler,
            gradient_clip_norm=resolved_training["gradient_clip_norm"],
            on_optimizer_step=on_optimizer_step,
            input_adapter=input_adapter,
        )
        validation_metrics = (
            None
            if validation_dataloader is None
            else _run_epoch(
                model,
                validation_dataloader,
                device=resolved_device,
                model_spec=spec,
                input_key=input_key,
                target_key=target_key,
                pos_weight=resolved_pos_weight,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                input_adapter=input_adapter,
            )
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        checkpoint = {
            "schema_version": 1,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": record,
            "run_identity": run_identity,
        }
        _save_checkpoint(last_path, checkpoint)
        if metric_name == "last":
            is_selected = True
            current_metric = None
        else:
            assert validation_metrics is not None
            current_metric = float(validation_metrics[metric_name])
            if not math.isfinite(current_metric):
                raise FloatingPointError(
                    f"validation {metric_name} is not finite"
                )
            is_selected = (
                current_metric < best_score
                if metric_name == "loss"
                else current_metric > best_score
            )
        if is_selected:
            selected_epoch = epoch
            selected_metric = current_metric
            if current_metric is not None:
                best_score = current_metric
            if selected_path != last_path:
                _save_checkpoint(selected_path, checkpoint)
        _write_json(
            history_path,
            {
                "schema_version": 1,
                "selection_metric": metric_name,
                "selected_epoch": selected_epoch,
                "selected_metric": selected_metric,
                "optimizer_steps": global_step,
                "epochs": history,
            },
        )

    if selected_epoch <= 0:
        raise RuntimeError("no training checkpoint was selected")
    if bool(selection["restore_best_at_end"]):
        load_model_checkpoint(model, selected_path, map_location=resolved_device)
    test_metrics = (
        None
        if test_dataloader is None
        or not bool(selection["evaluate_test_at_end"])
        else _run_epoch(
            model,
            test_dataloader,
            device=resolved_device,
            model_spec=spec,
            input_key=input_key,
            target_key=target_key,
            pos_weight=resolved_pos_weight,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            input_adapter=input_adapter,
        )
    )
    result = TrainingResult(
        model=model,
        output_dir=output,
        selected_epoch=selected_epoch,
        selected_metric=selected_metric,
        selected_checkpoint_path=selected_path,
        last_checkpoint_path=last_path,
        history=tuple(history),
        test_metrics=test_metrics,
        model_identity=run_identity["model"],
        config_identity=config.describe(),
        seed_identity=seed.describe(),
        input_adapter_identity=input_adapter_identity,
    )
    _write_json(result_path, result.describe())
    return result


__all__ = [
    "DEFAULT_TRAIN_CONFIG",
    "load_train_config",
    "train_model",
]

"""Family-balanced PN2021 Direct tuning on the managed 400/100 split.

Training uses the existing online trainer and Torch corruption operators on the
400 training records.  Validation is deliberately different: the clean 100
records and the twenty hash-seeded PN2021-C cache views are loaded once into
resident CPU tensors and reused unchanged for every epoch and backbone.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml

from core.methods import compile_method_profile
from core.online_trainer import (
    OnlineTrainConfig,
    OnlineTrainingResult,
    load_online_train_config,
    resolve_online_training_parameters,
    train_online_model,
)
from data_preprocess.data_runtime import RuntimeDataLoader, get_dataloader
from models.checkpoints import CheckpointIdentity
from models.contracts import (
    CLASS_ORDER,
    ECGFOUNDER_SPEC,
    EFFICIENTNET1DV2_SPEC,
    ModelSpec,
    validate_model_output,
)
from models.input_adapter import prepare_canonical_model_input
from util.config_bundle import (
    config_bundle_root,
    resolve_config_reference,
    resolve_entry_config_path,
)
from util.evaluation.metrics import compute_classification_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECT_TUNE_CONFIG = (
    PROJECT_ROOT / "configs" / "train" / "PN2021_direct_tune.yaml"
)
ALLOWED_CENTERS = (
    "ningbo",
    "chapman_shaoxing",
    "cpsc_2018",
    "georgia",
)
DIRECT_TUNING_PROTOCOL_ID = "pn2021_direct_family_balanced_tuning"
DIRECT_METHOD_ID = "direct_depth23_fixed20"
VALIDATION_ARTIFACT_TYPE = "pn2021_family_balanced_validation_predictions"
VALIDATION_ARTIFACT_SCHEMA = "pn2021_family_balanced_validation_predictions"
COMPOSITION_INDICES = tuple(range(20))
PN2021_TUNING_DATALOADER_PARAMETER_NAMES = frozenset(
    {
        "train_batch_size",
        "eval_batch_size",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "cache_mode",
        "validate_values",
        "selection_resident",
        "selection_resident_pin_memory",
        "drop_last",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_lines(values: Sequence[str], *, sort: bool) -> str:
    normalized = [str(value) for value in values]
    if sort:
        normalized.sort()
    payload = "".join(f"{value}\n" for value in normalized).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_label_set_sha256(hash_ids: Sequence[str], targets: np.ndarray) -> str:
    rows = sorted(
        (str(hash_id), "".join(str(int(value)) for value in target))
        for hash_id, target in zip(hash_ids, targets, strict=True)
    )
    payload = "".join(f"{hash_id}|{label}\n" for hash_id, label in rows).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a mapping")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _model_spec(model: nn.Module) -> ModelSpec:
    owner = model.module if hasattr(model, "module") else model
    spec = getattr(owner, "model_spec", None)
    if spec not in {EFFICIENTNET1DV2_SPEC, ECGFOUNDER_SPEC}:
        raise ValueError("model must expose a managed EfficientNet/ECGFounder spec")
    return spec


@dataclass(frozen=True)
class CanonicalTuningInputAdapter:
    model_spec: ModelSpec
    epsilon: float = 1.0e-6

    def __call__(self, raw_100hz_btc: torch.Tensor) -> torch.Tensor:
        return prepare_canonical_model_input(
            raw_100hz_btc,
            self.model_spec,
            epsilon=self.epsilon,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "implementation": "models.input_adapter.prepare_canonical_model_input",
            "model": self.model_spec.name,
            "input": "raw_mV_100hz_time_channel",
            "resampling": (
                "linear_100_to_500hz_align_corners_true"
                if self.model_spec == ECGFOUNDER_SPEC
                else "identity_100hz"
            ),
            "normalization": "per_sample_global_zscore_after_resampling",
            "epsilon": self.epsilon,
            "output_layout": "channel_time",
        }


@dataclass(frozen=True)
class PN2021TuningConfig:
    path: Path
    sha256: str
    config_root: Path
    references: dict[str, Path]
    payload: dict[str, Any]
    online: OnlineTrainConfig

    @property
    def profile_name(self) -> str:
        return str(self.payload["profile_name"])

    def describe(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "config_root": str(self.config_root),
            "references": {
                name: {"path": str(path), "sha256": _sha256_file(path)}
                for name, path in self.references.items()
            },
            "online_training": self.online.describe(),
            "resolved": self.payload,
        }


def load_pn2021_tuning_config(
    path: str | Path = DEFAULT_DIRECT_TUNE_CONFIG,
    *,
    config_root: str | Path | None = None,
) -> PN2021TuningConfig:
    config_path = resolve_entry_config_path(path)
    root = config_bundle_root(config_path, config_root=config_root)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root_payload = _mapping(payload, "PN2021 tuning config")
    expected_keys = {
        "schema_version",
        "profile_name",
        "protocol_lock",
        "references",
        "data",
        "training",
        "validation_predictions",
        "pooled_selection",
        "refit_contract",
        "output",
    }
    if set(root_payload) != expected_keys or root_payload.get("schema_version") != 1:
        raise ValueError("PN2021 tuning config schema or root keys are invalid")
    if root_payload.get("profile_name") != DIRECT_TUNING_PROTOCOL_ID:
        raise ValueError(f"profile_name must be {DIRECT_TUNING_PROTOCOL_ID}")

    raw_references = _mapping(root_payload["references"], "references")
    expected_references = {
        "online_training_config",
        "method_config",
        "split_config",
        "data_load_config",
        "corruption_cache_config",
        "random_seed_config",
        "source_baseline_registry",
        "clean_baseline_registry",
    }
    if set(raw_references) != expected_references:
        raise ValueError("PN2021 tuning references are incomplete or unexpected")
    references = {
        name: resolve_config_reference(
            raw,
            owner_config_path=config_path,
            config_root=root,
            description=f"references.{name}",
            must_exist=True,
        )
        for name, raw in raw_references.items()
    }

    protocol = _mapping(root_payload["protocol_lock"], "protocol_lock")
    if protocol.get("status") != "k500_internal_tuning_only":
        raise ValueError("protocol status must be k500_internal_tuning_only")
    if protocol.get("protocol_id") != DIRECT_TUNING_PROTOCOL_ID:
        raise ValueError("protocol_id mismatch")
    if protocol.get("method_id") != DIRECT_METHOD_ID:
        raise ValueError(f"protocol method_id must be {DIRECT_METHOD_ID}")
    if tuple(protocol.get("centers", ())) != ALLOWED_CENTERS:
        raise ValueError(f"protocol centers must be {ALLOWED_CENTERS}")
    if protocol.get("merge_cpsc_2018_extra_into_cpsc_2018") is not True:
        raise ValueError("CPSC 2018 Extra must be merged into CPSC 2018")
    if protocol.get("heldout_evaluation_used_for_selection") is not False:
        raise ValueError("held-out PN2021 evaluation cannot select the epoch")
    if protocol.get("validation_records_allowed_in_training") is not False:
        raise ValueError("validation100 records cannot enter training")
    if protocol.get("validation_records_allowed_in_latent_pool") is not False:
        raise ValueError("validation100 records cannot enter latent pools")
    if protocol.get("mapping_version") != "v7_super5_sjr_rgq_review_20260528":
        raise ValueError("mapping version mismatch")
    if protocol.get("mapping_hash") != "555ec85d5b51":
        raise ValueError("mapping hash mismatch")

    data = _mapping(root_payload["data"], "data")
    required_data = {
        "dataset": "pn2021",
        "source_partition": "k500",
        "train_partition": "k500_tune_train",
        "validation_partition": "k500_tune_validation",
        "train_records_per_center": 400,
        "validation_records_per_center": 100,
        "waveform_domain": "canonical_raw_100hz",
        "sampling_rate_hz": 100,
        "waveform_points": 1000,
        "duration_seconds": 10,
        "physical_unit": "mV",
        "input_layout": "time_channel",
        "normalization": "none",
    }
    for key, expected in required_data.items():
        if data.get(key) != expected:
            raise ValueError(f"data.{key} must be {expected!r}")
    if tuple(data.get("class_order", ())) != CLASS_ORDER:
        raise ValueError(f"data.class_order must be {CLASS_ORDER}")

    validation = _mapping(root_payload["validation_predictions"], "validation_predictions")
    if validation.get("enabled") is not True:
        raise ValueError("validation prediction export must be enabled")
    if validation.get("artifact_schema") != VALIDATION_ARTIFACT_SCHEMA:
        raise ValueError("validation artifact schema mismatch")
    if tuple(validation.get("composition_indices", ())) != COMPOSITION_INDICES:
        raise ValueError("validation must use the canonical twenty compositions")
    selection = _mapping(root_payload["pooled_selection"], "pooled_selection")
    if selection.get("expected_records_per_center") != 100:
        raise ValueError("selection expects 100 records per center")
    if selection.get("expected_pooled_records") != 400:
        raise ValueError("selection expects 400 pooled records")
    if selection.get("expected_compositions") != 20:
        raise ValueError("selection expects twenty compositions")
    if selection.get("score") != {"clean_weight": 0.5, "robust_weight": 0.5}:
        raise ValueError("selection score must weight clean and robust equally")

    online = load_online_train_config(
        references["online_training_config"], config_root=root
    )
    method = compile_method_profile(references["method_config"])
    if method.profile_name != DIRECT_METHOD_ID or not method.executable:
        raise ValueError("tuning requires the executable family-balanced Direct method")
    if online.payload["protocol"].get("tuning") is None:
        raise ValueError("online training config does not permit the managed 400/100 split")
    return PN2021TuningConfig(
        path=config_path,
        sha256=_sha256_file(config_path),
        config_root=root,
        references=references,
        payload=root_payload,
        online=online,
    )


def _load_source_registry(config: Any) -> dict[str, Any]:
    references = getattr(config, "references", None)
    path = references.get("source_baseline_registry") if isinstance(references, dict) else None
    if not isinstance(path, Path):
        raise ValueError("config must resolve source_baseline_registry")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("source baseline registry must be schema_version=1")
    return payload


def _checkpoint_identity(model: nn.Module, spec: ModelSpec) -> CheckpointIdentity:
    owner = model.module if hasattr(model, "module") else model
    attribute = (
        "checkpoint_identity"
        if spec == EFFICIENTNET1DV2_SPEC
        else "task_checkpoint_identity"
    )
    identity = getattr(owner, attribute, None)
    if not isinstance(identity, CheckpointIdentity):
        raise ValueError(f"Direct tuning model must carry strict {attribute}")
    return identity


def validate_locked_source_checkpoint(model: nn.Module, config: Any) -> dict[str, Any]:
    spec = _model_spec(model)
    registry = _load_source_registry(config)
    expected = _mapping(registry.get("models"), "source registry models").get(spec.name)
    expected = _mapping(expected, f"source registry model {spec.name}")
    identity = _checkpoint_identity(model, spec)
    if identity.sha256 != expected.get("selected_checkpoint_sha256"):
        raise ValueError("model source checkpoint SHA256 differs from locked registry")
    if Path(identity.path).resolve() != Path(expected["selected_checkpoint"]).resolve():
        raise ValueError("model source checkpoint path differs from locked registry")
    return {
        "baseline_id": registry.get("baseline", {}).get("id"),
        "model_family": spec.name,
        "path": str(identity.path),
        "sha256": identity.sha256,
    }


def _resolve_loader_parameters(
    config: PN2021TuningConfig,
    model_name: str,
    resolved_training: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    supplied = {} if overrides is None else dict(overrides)
    unknown = sorted(set(supplied) - PN2021_TUNING_DATALOADER_PARAMETER_NAMES)
    if unknown:
        raise ValueError(f"unknown PN2021 tuning dataloader parameters: {unknown}")
    defaults = config.payload["training"]
    eval_sizes = _mapping(defaults["validation_batch_size"], "validation_batch_size")
    resolved = {
        "train_batch_size": supplied.get("train_batch_size", resolved_training["batch_size"]),
        "eval_batch_size": supplied.get("eval_batch_size", eval_sizes[model_name]),
        "num_workers": supplied.get("num_workers", defaults["num_workers"]),
        "pin_memory": supplied.get("pin_memory", defaults["pin_memory"]),
        "persistent_workers": supplied.get("persistent_workers", defaults["persistent_workers"]),
        "prefetch_factor": supplied.get("prefetch_factor", defaults["prefetch_factor"]),
        "cache_mode": supplied.get("cache_mode", defaults["cache_mode"]),
        "validate_values": supplied.get("validate_values", defaults["validate_values"]),
        "selection_resident": supplied.get("selection_resident", defaults["selection_resident"]),
        "selection_resident_pin_memory": supplied.get(
            "selection_resident_pin_memory",
            defaults["selection_resident_pin_memory"],
        ),
        "drop_last": supplied.get("drop_last", False),
    }
    for key in ("train_batch_size", "eval_batch_size", "prefetch_factor"):
        value = resolved[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"resolved {key} must be a positive integer")
    if int(resolved["train_batch_size"]) != int(resolved_training["batch_size"]):
        raise ValueError("train loader batch size must match online training batch size")
    if resolved["num_workers"] != 0:
        raise ValueError("resident frozen tuning loaders require num_workers=0")
    for key in (
        "pin_memory",
        "persistent_workers",
        "selection_resident",
        "selection_resident_pin_memory",
        "drop_last",
    ):
        if not isinstance(resolved[key], bool):
            raise ValueError(f"resolved {key} must be boolean")
    if resolved["persistent_workers"] or resolved["drop_last"]:
        raise ValueError("tuning requires persistent_workers=false and drop_last=false")
    if resolved["cache_mode"] != "mmap" or resolved["selection_resident"] is not True:
        raise ValueError("tuning requires mmap source plus selection residency")
    if resolved["validate_values"] not in {"none", "sample", "full"}:
        raise ValueError("validate_values must be none, sample or full")
    return resolved, supplied


def _selection_hashes(loader: RuntimeDataLoader) -> tuple[str, ...]:
    selection = getattr(loader.dataset, "selection", None)
    values = getattr(selection, "hash_ids", None)
    if values is None:
        raise ValueError("managed tuning loader does not expose selection hashes")
    return tuple(str(value) for value in values)


@dataclass(frozen=True)
class PN2021TuningDataLoaders:
    train: RuntimeDataLoader
    clean_validation: RuntimeDataLoader
    corrupted_validation: tuple[RuntimeDataLoader, ...]
    center: str
    model_name: str
    input_adapter: CanonicalTuningInputAdapter
    resolved_parameters: dict[str, Any]
    explicit_overrides: dict[str, Any]

    def close(self) -> None:
        self.train.close()
        self.clean_validation.close()
        for loader in self.corrupted_validation:
            loader.close()

    def describe(self) -> dict[str, Any]:
        return {
            "center": self.center,
            "model_name": self.model_name,
            "input_adapter": self.input_adapter.describe(),
            "resolved_parameters": self.resolved_parameters,
            "explicit_overrides": self.explicit_overrides,
            "train": self.train.describe(),
            "clean_validation": self.clean_validation.describe(),
            "corrupted_validation": [
                loader.describe() for loader in self.corrupted_validation
            ],
        }


def build_pn2021_tuning_dataloaders(
    model: nn.Module,
    *,
    center: str,
    config_path: str | Path = DEFAULT_DIRECT_TUNE_CONFIG,
    config_root: str | Path | None = None,
    training_parameters: Mapping[str, Any] | None = None,
    dataloader_parameters: Mapping[str, Any] | None = None,
) -> PN2021TuningDataLoaders:
    if center not in ALLOWED_CENTERS:
        raise ValueError(f"center must be one of {ALLOWED_CENTERS}")
    config = load_pn2021_tuning_config(config_path, config_root=config_root)
    spec = _model_spec(model)
    resolved_training, _ = resolve_online_training_parameters(
        config.online, spec.name, training_parameters
    )
    resolved, supplied = _resolve_loader_parameters(
        config, spec.name, resolved_training, dataloader_parameters
    )
    random_seed = config.online.payload["random_seed"]
    common = {
        "logical_center": center,
        "sampling_rate_hz": 100,
        "num_workers": 0,
        "pin_memory": bool(resolved["pin_memory"]),
        "persistent_workers": False,
        "prefetch_factor": int(resolved["prefetch_factor"]),
        "cache_mode": "mmap",
        "validate_values": resolved["validate_values"],
        "selection_resident": True,
        "selection_resident_pin_memory": bool(
            resolved["selection_resident_pin_memory"]
        ),
        "drop_last": False,
        "prepare_for_model": False,
        "sanitize": False,
        "global_zscore": False,
        "output_layout": "time_channel",
        "seed_config_path": config.references["random_seed_config"],
        "config_root": config.config_root,
        "split_config_path": config.references["split_config"],
        "data_load_config_path": config.references["data_load_config"],
    }
    opened: list[RuntimeDataLoader] = []
    try:
        train = get_dataloader(
            dataset="pn2021",
            partition="k500_tune_train",
            batch_size=int(resolved["train_batch_size"]),
            shuffle=True,
            seed_namespace=(
                f"pn2021_direct_tune:{random_seed['comparison_group']}:"
                f"{random_seed['replicate_id']}:{center}:{spec.name}:train"
            ),
            **common,
        )
        opened.append(train)
        clean_validation = get_dataloader(
            dataset="pn2021",
            partition="k500_tune_validation",
            batch_size=int(resolved["eval_batch_size"]),
            shuffle=False,
            seed_namespace=(
                f"pn2021_direct_tune:{center}:validation:clean"
            ),
            **common,
        )
        opened.append(clean_validation)
        corrupted: list[RuntimeDataLoader] = []
        for composition_index in COMPOSITION_INDICES:
            loader = get_dataloader(
                dataset="pn2021c",
                partition="k500_tune_validation",
                batch_size=int(resolved["eval_batch_size"]),
                view=composition_index,
                shuffle=False,
                seed_namespace=(
                    f"pn2021_direct_tune:{center}:validation:"
                    f"composition_{composition_index:02d}"
                ),
                corruption_cache_config_path=config.references[
                    "corruption_cache_config"
                ],
                **common,
            )
            opened.append(loader)
            corrupted.append(loader)
    except Exception:
        for loader in opened:
            loader.close()
        raise

    expected_hashes = _selection_hashes(clean_validation)
    if len(expected_hashes) != 100 or len(set(expected_hashes)) != 100:
        for loader in opened:
            loader.close()
        raise RuntimeError("clean validation must contain 100 unique hashes")
    for index, loader in enumerate(corrupted):
        if _selection_hashes(loader) != expected_hashes:
            for member in opened:
                member.close()
            raise RuntimeError(
                f"corruption composition {index} changed validation hash order"
            )
    return PN2021TuningDataLoaders(
        train=train,
        clean_validation=clean_validation,
        corrupted_validation=tuple(corrupted),
        center=center,
        model_name=spec.name,
        input_adapter=CanonicalTuningInputAdapter(spec),
        resolved_parameters=resolved,
        explicit_overrides=supplied,
    )


def _selection_identity(loader: RuntimeDataLoader) -> dict[str, Any]:
    description = loader.describe()
    dataset = description.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("runtime loader description lacks dataset identity")
    selection = dataset.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("runtime loader description lacks selection identity")
    return dict(selection)


class FrozenValidationEvaluator:
    """Evaluate one immutable clean+20 validation bank after every epoch."""

    def __init__(
        self,
        loaders: PN2021TuningDataLoaders,
        config: PN2021TuningConfig,
        *,
        output_dir: Path,
        amp_enabled: bool,
        amp_dtype: str,
    ) -> None:
        self.loaders = loaders
        self.config = config
        self.output_dir = output_dir / str(
            config.payload["validation_predictions"]["subdir"]
        )
        self.spec = loaders.input_adapter.model_spec
        self.amp_enabled = bool(amp_enabled)
        self.amp_dtype = (
            torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
        )
        self.clean_identity = _selection_identity(loaders.clean_validation)
        corrupted_descriptions = [
            loader.describe()["dataset"] for loader in loaders.corrupted_validation
        ]
        composition_ids = tuple(
            str(item.get("composition_id", "")) for item in corrupted_descriptions
        )
        if len(composition_ids) != 20 or any(not value for value in composition_ids):
            raise ValueError("frozen validation requires twenty composition IDs")
        if len(set(composition_ids)) != 20:
            raise ValueError("frozen validation composition IDs must be unique")
        cache_descriptions = [dict(item["cache"]) for item in corrupted_descriptions]
        first_cache = cache_descriptions[0]
        if any(
            item.get("manifest_sha256") != first_cache.get("manifest_sha256")
            or item.get("cache_version") != first_cache.get("cache_version")
            for item in cache_descriptions[1:]
        ):
            raise ValueError("frozen validation views use different corruption caches")
        manifest_path = Path(first_cache["cache_dir"]) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _sha256_file(manifest_path) != first_cache["manifest_sha256"]:
            raise ValueError("corruption cache manifest identity changed")
        self.composition_ids = composition_ids
        self.frozen_identity = {
            "corruption_cache_dir": first_cache["cache_dir"],
            "corruption_cache_version": first_cache["cache_version"],
            "corruption_cache_manifest_path": str(manifest_path),
            "corruption_cache_manifest_sha256": first_cache["manifest_sha256"],
            "corruption_config_identity_hash": manifest["identity"][
                "config_identity_hash"
            ],
            "composition_indices": list(COMPOSITION_INDICES),
            "composition_ids": list(composition_ids),
            "ordered_hash_ids_sha256": _sha256_lines(
                _selection_hashes(loaders.clean_validation), sort=False
            ),
            "source_manifest_sha256": self.clean_identity[
                "source_manifest_sha256"
            ],
        }

    def describe(self) -> dict[str, Any]:
        return {
            "type": "frozen_clean_plus_20_pn2021c_validation",
            "output_dir": str(self.output_dir),
            "clean_selection": self.clean_identity,
            "frozen_corruption": self.frozen_identity,
            "input_adapter": self.loaders.input_adapter.describe(),
        }

    def _predict(
        self,
        model: nn.Module,
        loader: RuntimeDataLoader,
        device: torch.device,
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        logits_parts: list[np.ndarray] = []
        targets_parts: list[np.ndarray] = []
        hashes: list[str] = []
        use_amp = self.amp_enabled and device.type == "cuda"
        with torch.inference_mode():
            for batch in loader:
                raw = batch.get("waveform")
                targets = batch.get("label")
                batch_hashes = batch.get("hash_id")
                if not isinstance(raw, torch.Tensor) or not isinstance(targets, torch.Tensor):
                    raise TypeError("validation batch waveform/label must be tensors")
                if not isinstance(batch_hashes, (list, tuple)):
                    raise TypeError("validation batch must provide hash_id strings")
                raw = raw.to(device=device, dtype=torch.float32, non_blocking=True)
                targets = targets.to(dtype=torch.float32)
                model_input = self.loaders.input_adapter(raw)
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=self.amp_dtype):
                    output = validate_model_output(
                        model(model_input),
                        self.spec,
                        batch_size=int(raw.shape[0]),
                    )
                logits_parts.append(output.detach().float().cpu().numpy())
                targets_parts.append(targets.cpu().numpy())
                hashes.extend(str(value) for value in batch_hashes)
        logits = np.concatenate(logits_parts, axis=0).astype(np.float32, copy=False)
        truth = np.concatenate(targets_parts, axis=0).astype(np.uint8, copy=False)
        if logits.shape != (100, 5) or truth.shape != (100, 5):
            raise RuntimeError("frozen validation prediction shape must be (100,5)")
        if len(hashes) != 100 or len(set(hashes)) != 100:
            raise RuntimeError("frozen validation hashes must be 100 unique records")
        return logits, truth, tuple(hashes)

    @staticmethod
    def _align(
        logits: np.ndarray,
        targets: np.ndarray,
        hashes: Sequence[str],
        reference_hashes: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        positions = {str(value): index for index, value in enumerate(hashes)}
        if set(positions) != set(reference_hashes):
            raise RuntimeError("clean and corrupted validation hash sets differ")
        order = np.asarray([positions[str(value)] for value in reference_hashes])
        return logits[order], targets[order]

    def __call__(
        self,
        model: nn.Module,
        epoch: int,
        device: torch.device,
        run_identity: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        arrays_path = self.output_dir / f"epoch_{int(epoch):04d}.npz"
        sidecar_path = self.output_dir / f"epoch_{int(epoch):04d}.json"
        if arrays_path.exists() or sidecar_path.exists():
            raise FileExistsError(f"validation artifact already exists for epoch {epoch}")

        clean_logits, targets, hashes = self._predict(
            model, self.loaders.clean_validation, device
        )
        corrupted_logits: list[np.ndarray] = []
        for loader in self.loaders.corrupted_validation:
            logits, current_targets, current_hashes = self._predict(model, loader, device)
            logits, current_targets = self._align(
                logits, current_targets, current_hashes, hashes
            )
            if not np.array_equal(current_targets, targets):
                raise RuntimeError("clean/corrupted validation labels differ")
            corrupted_logits.append(logits)
        corrupted = np.stack(corrupted_logits, axis=0).astype(np.float32, copy=False)

        clean_metrics = compute_classification_metrics(
            clean_logits,
            targets,
            class_order=CLASS_ORDER,
            undefined_class_policy="skip_undefined",
        )
        composition_metrics = [
            compute_classification_metrics(
                corrupted[index],
                targets,
                class_order=CLASS_ORDER,
                undefined_class_policy="skip_undefined",
            )
            for index in COMPOSITION_INDICES
        ]
        local_robust_auprc = float(
            np.mean([item["macro_auprc"] for item in composition_metrics])
        )
        local_score = 0.5 * float(clean_metrics["macro_auprc"]) + 0.5 * local_robust_auprc

        temporary = arrays_path.with_name(f".{arrays_path.name}.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                clean_logits=clean_logits,
                corrupted_logits=corrupted,
                targets=targets,
                hash_ids=np.asarray(hashes, dtype=np.str_),
                composition_ids=np.asarray(self.composition_ids, dtype=np.str_),
            )
        temporary.replace(arrays_path)

        model_identity = _mapping(run_identity.get("model"), "run model identity")
        source_checkpoint = (
            model_identity.get("task_checkpoint_identity")
            or model_identity.get("checkpoint_identity")
            or model_identity.get("pretrained_checkpoint_identity")
        )
        training_identity = _mapping(
            run_identity.get("training_parameters"), "run training parameters"
        )
        comparison_identity = {
            "protocol_id": DIRECT_TUNING_PROTOCOL_ID,
            "method_id": DIRECT_METHOD_ID,
            "model_family": self.spec.name,
            "model_spec": model_identity.get("spec"),
            "source_checkpoint_identity": source_checkpoint,
            "tuning_config_sha256": self.config.sha256,
            "online_training_config_sha256": self.config.online.sha256,
            "method_profile_sha256": _sha256_file(
                self.config.references["method_config"]
            ),
            "resolved_training_parameters": training_identity.get("resolved"),
            "pos_weight": run_identity.get("pos_weight"),
            "comparison_group": self.config.online.payload["random_seed"][
                "comparison_group"
            ],
            "replicate_id": self.config.online.payload["random_seed"][
                "replicate_id"
            ],
        }
        _write_json(
            sidecar_path,
            {
                "schema_version": 1,
                "artifact_type": VALIDATION_ARTIFACT_TYPE,
                "artifact_schema": VALIDATION_ARTIFACT_SCHEMA,
                "epoch": int(epoch),
                "center": self.loaders.center,
                "class_order": list(CLASS_ORDER),
                "record_count": 100,
                "composition_count": 20,
                "arrays_file": arrays_path.name,
                "arrays_sha256": _sha256_file(arrays_path),
                "clean_validation_identity": self.clean_identity,
                "frozen_corruption_identity": self.frozen_identity,
                "prediction_identity": {
                    "ordered_hash_ids_sha256": _sha256_lines(hashes, sort=False),
                    "hash_id_set_sha256": _sha256_lines(hashes, sort=True),
                    "hash_label_set_sha256": _hash_label_set_sha256(hashes, targets),
                },
                "comparison_identity": comparison_identity,
                "local_monitoring": {
                    "undefined_class_policy": "skip_undefined",
                    "clean": clean_metrics,
                    "composition_metrics": composition_metrics,
                    "robust_macro_auprc": local_robust_auprc,
                    "score": local_score,
                    "selection_use": False,
                },
            },
        )
        return {
            "artifact_path": str(sidecar_path),
            "artifact_sha256": _sha256_file(sidecar_path),
            "clean_macro_auprc_local": float(clean_metrics["macro_auprc"]),
            "robust_macro_auprc_local": local_robust_auprc,
            "score_local": local_score,
            "selection_deferred_to_four_center_pool": True,
        }


def train_pn2021_direct_tuning(
    model: nn.Module,
    *,
    center: str,
    config_path: str | Path = DEFAULT_DIRECT_TUNE_CONFIG,
    config_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    pos_weight: torch.Tensor | Sequence[float] | None = None,
    training_parameters: Mapping[str, Any] | None = None,
    dataloader_parameters: Mapping[str, Any] | None = None,
) -> OnlineTrainingResult:
    config = load_pn2021_tuning_config(config_path, config_root=config_root)
    validate_locked_source_checkpoint(model, config)
    spec = _model_spec(model)
    resolved_training, _ = resolve_online_training_parameters(
        config.online, spec.name, training_parameters
    )
    loaders = build_pn2021_tuning_dataloaders(
        model,
        center=center,
        config_path=config.path,
        config_root=config.config_root,
        training_parameters=training_parameters,
        dataloader_parameters=dataloader_parameters,
    )
    output = (
        Path(config.payload["output"]["run_dir"]).expanduser().resolve()
        / spec.name
        / center
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    evaluator = FrozenValidationEvaluator(
        loaders,
        config,
        output_dir=output,
        amp_enabled=bool(resolved_training["amp_enabled"]),
        amp_dtype=str(resolved_training["amp_dtype"]),
    )
    try:
        return train_online_model(
            model,
            loaders.train,
            center=center,
            method_config_path=config.references["method_config"],
            config_path=config.online.path,
            config_root=config.config_root,
            output_dir=output,
            device=device,
            training_parameters=training_parameters,
            pos_weight=pos_weight,
            epoch_evaluator=evaluator,
        )
    finally:
        loaders.close()


__all__ = [
    "ALLOWED_CENTERS",
    "COMPOSITION_INDICES",
    "CanonicalTuningInputAdapter",
    "DEFAULT_DIRECT_TUNE_CONFIG",
    "DIRECT_METHOD_ID",
    "DIRECT_TUNING_PROTOCOL_ID",
    "FrozenValidationEvaluator",
    "PN2021TuningConfig",
    "PN2021TuningDataLoaders",
    "VALIDATION_ARTIFACT_SCHEMA",
    "VALIDATION_ARTIFACT_TYPE",
    "build_pn2021_tuning_dataloaders",
    "load_pn2021_tuning_config",
    "train_pn2021_direct_tuning",
    "validate_locked_source_checkpoint",
]

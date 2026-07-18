"""Typed-method online trainer for PN2021 K500.

The trainer owns the outer optimization and exposure budget.  A restricted
method graph may generate canonical raw-mV views, but cannot alter the loader,
normalization, optimizer, scheduler, or checkpoint selection.  Standard
``matched_base`` methods use exactly one optimizer step per base batch.  The
fixed-20 Direct baseline evaluates one clean view plus all twenty canonical
corruption views, accumulates ``0.5 * clean + 0.5 * mean(corruptions)``, and
then performs that single step.  Every base record therefore contributes one
family-balanced objective without materializing a training cache.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from core.augmix import multilabel_jsd
from core.methods import (
    BASE_VIEW_NAME,
    CompiledMethod,
    WaveformView,
    build_method_runtime,
    compile_method_profile,
)
from models.checkpoints import sha256_file
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
from util.random_seed import seed_process
from util.tensorboard_logging import (
    build_tensorboard_monitor,
    parse_tensorboard_logging_config,
)

if TYPE_CHECKING:
    from core.latent_pool import LatentPool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONLINE_CONFIG_PATH = PROJECT_ROOT / "configs" / "train" / "PN2021.yaml"
DEFAULT_METHOD_CONFIG_DIR = PROJECT_ROOT / "configs" / "train" / "methods"
ALLOWED_CENTERS = ("ningbo", "chapman_shaoxing", "cpsc_2018", "georgia")
ONLINE_PARAMETER_NAMES = frozenset(
    {
        "epochs",
        "scheduler_horizon_epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "minimum_learning_rate_ratio",
        "gradient_clip_norm",
        "amp_enabled",
        "amp_dtype",
    }
)
FIXED20_EXPOSURE_POLICY = "clean_once_then_exhaustive_depth23"
FIXED20_COMPOSITION_ORDER = tuple(range(20))
FAMILY_BALANCED_BN_POLICY = "family_loss_weighted_once_per_base_batch"
OBJECTIVE_VIEW_BALANCED_BN_POLICY = "objective_view_weighted_once_per_base_batch"


@dataclass(frozen=True)
class _ExposureStep:
    name: str
    composition_index: int | None
    objective_terms: tuple[str, ...] | None


@dataclass
class _BatchNormMomentumPlan:
    policy: str
    named_modules: tuple[tuple[str, nn.Module], ...]
    original_momenta: tuple[float, ...]
    schedules: tuple[tuple[float, ...], ...]
    exposure_weights: tuple[float, ...]

    def apply(self, exposure_index: int) -> None:
        if exposure_index < 0 or exposure_index >= len(self.exposure_weights):
            raise IndexError("BatchNorm exposure index is outside the locked schedule")
        for (_, module), schedule in zip(self.named_modules, self.schedules):
            module.momentum = float(schedule[exposure_index])

    def restore(self) -> None:
        for (_, module), momentum in zip(
            self.named_modules, self.original_momenta
        ):
            module.momentum = float(momentum)

    def describe(self) -> dict[str, Any]:
        schedule_by_base_momentum: dict[str, list[float]] = {}
        for momentum, schedule in zip(self.original_momenta, self.schedules):
            schedule_by_base_momentum.setdefault(
                format(momentum, ".17g"), list(schedule)
            )
        return {
            "policy": self.policy,
            "module_count": len(self.named_modules),
            "module_names": [name for name, _ in self.named_modules],
            "base_momenta": list(self.original_momenta),
            "exposure_weights": list(self.exposure_weights),
            "schedule_by_base_momentum": schedule_by_base_momentum,
            "effective_total_momentum_per_base_batch": "preserve_each_module_base_momentum",
        }


def _family_balanced_batch_norm_momenta(
    base_momentum: float,
    exposure_weights: Sequence[float],
) -> tuple[float, ...]:
    """Convert family weights into exact sequential BatchNorm momenta.

    After all exposures, the old running statistic retains ``1-m`` and every
    exposure contributes ``m * weight`` where ``m`` is the module's original
    momentum. This prevents twenty corruption forwards from silently
    overriding the declared 0.5/0.5 family balance.
    """

    base = float(base_momentum)
    weights = tuple(float(value) for value in exposure_weights)
    if not 0.0 <= base <= 1.0:
        raise ValueError("BatchNorm momentum must be in [0,1]")
    if not weights or any(value < 0.0 for value in weights):
        raise ValueError("BatchNorm exposure weights must be non-negative")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("BatchNorm exposure weights must sum to one")
    target = tuple(base * value for value in weights)
    later_target = 0.0
    reversed_schedule: list[float] = []
    for contribution in reversed(target):
        retained_before = 1.0 - later_target
        if retained_before <= 0.0:
            raise ValueError("BatchNorm sequential momentum schedule is singular")
        reversed_schedule.append(contribution / retained_before)
        later_target += contribution
    return tuple(reversed(reversed_schedule))


def _build_batch_norm_momentum_plan(
    model: nn.Module,
    method: CompiledMethod,
    *,
    fixed20_exposure: bool,
) -> _BatchNormMomentumPlan | None:
    policy = str(method.contracts.get("batch_norm_running_stats_policy", ""))
    if fixed20_exposure:
        if policy != FAMILY_BALANCED_BN_POLICY:
            raise ValueError(
                "fixed20 must declare family-loss-weighted BatchNorm running stats"
            )
        family_weights = _mapping(
            method.contracts.get("family_loss_weights"),
            "fixed20 family_loss_weights",
        )
        exposure_weights = (
            float(family_weights["clean"]),
            *(
                float(family_weights["corrupted_per_composition"])
                for _ in FIXED20_COMPOSITION_ORDER
            ),
        )
    elif policy == OBJECTIVE_VIEW_BALANCED_BN_POLICY:
        ordered_views: list[str] = []
        for term in method.objective.terms:
            for name in term.views:
                if name not in ordered_views:
                    ordered_views.append(name)
        weights = _mapping(
            method.contracts.get("batch_norm_objective_view_weights"),
            "batch_norm_objective_view_weights",
        )
        if set(weights) != set(ordered_views):
            raise ValueError(
                "BatchNorm objective-view weights must cover every objective view"
            )
        exposure_weights = tuple(float(weights[name]) for name in ordered_views)
        if not math.isclose(
            sum(exposure_weights), 1.0, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("BatchNorm objective-view weights must sum to one")
    elif policy:
        raise ValueError(f"unsupported BatchNorm running-stats policy: {policy}")
    else:
        return None
    named_modules = tuple(
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        and bool(module.track_running_stats)
    )
    original_momenta: list[float] = []
    schedules: list[tuple[float, ...]] = []
    for name, module in named_modules:
        if module.momentum is None:
            raise ValueError(
                f"BatchNorm module {name!r} uses cumulative momentum, which is "
                "incompatible with the locked family-balanced policy"
            )
        momentum = float(module.momentum)
        original_momenta.append(momentum)
        schedules.append(
            _family_balanced_batch_norm_momenta(momentum, exposure_weights)
        )
    return _BatchNormMomentumPlan(
        policy=policy,
        named_modules=named_modules,
        original_momenta=tuple(original_momenta),
        schedules=tuple(schedules),
        exposure_weights=tuple(exposure_weights),
    )


def _method_exposure_steps(method: CompiledMethod) -> tuple[_ExposureStep, ...]:
    """Resolve the optimizer-exposure schedule declared by one method profile."""

    policy = method.contracts.get("exposure_policy")
    if policy is None:
        return (_ExposureStep("base", None, None),)
    if policy != FIXED20_EXPOSURE_POLICY:
        raise ValueError(f"unsupported method exposure_policy: {policy!r}")
    raw_order = method.contracts.get("composition_indices")
    if not isinstance(raw_order, (list, tuple)):
        raise ValueError("fixed20 method must declare contracts.composition_indices")
    order = tuple(int(value) for value in raw_order)
    if order != FIXED20_COMPOSITION_ORDER or set(order) != set(range(20)):
        raise ValueError(
            "fixed20 composition_indices must be the locked canonical 0..19 schedule"
        )
    if method.contracts.get("clean_exposures_per_base_record") != 1:
        raise ValueError("fixed20 method must expose every clean record exactly once")
    if method.contracts.get("corrupted_exposures_per_base_record") != 20:
        raise ValueError("fixed20 method must expose all twenty corruptions")
    if tuple(method.contracts.get("tensorboard_probe_composition_indices", ())) != (
        0,
        10,
        19,
    ):
        raise ValueError("fixed20 TensorBoard probes must be compositions 0, 10 and 19")
    names = {term.name for term in method.objective.terms}
    if names != {"clean_bce", "corrupted_bce"}:
        raise ValueError(
            "fixed20 method objective must contain clean_bce and corrupted_bce"
        )
    if (
        method.contracts.get("optimizer_step_policy")
        != "accumulate_family_balanced_once_per_base_batch"
    ):
        raise ValueError(
            "fixed20 must accumulate all family-balanced losses before one step"
        )
    if (
        method.contracts.get("batch_norm_running_stats_policy")
        != FAMILY_BALANCED_BN_POLICY
    ):
        raise ValueError(
            "fixed20 must weight BatchNorm running stats by the declared families"
        )
    family_weights = _mapping(
        method.contracts.get("family_loss_weights"),
        "fixed20 family_loss_weights",
    )
    expected_weights = {
        "clean": 0.5,
        "corrupted_total": 0.5,
        "corrupted_per_composition": 0.025,
    }
    if family_weights != expected_weights:
        raise ValueError(
            "fixed20 family weights must be clean=0.5 and 20x corruption=0.025"
        )
    return (
        _ExposureStep("clean", -1, ("clean_bce",)),
        *tuple(
            _ExposureStep(
                f"corruption_{index:02d}", index, ("corrupted_bce",)
            )
            for index in order
        ),
    )


def _iter_exposure_batches(
    train_dataloader: Any,
    exposure_steps: Sequence[_ExposureStep],
):
    """Expand one loaded base batch without reading it from storage again."""

    group = 0
    for raw_batch in train_dataloader:
        if len(exposure_steps) == 1 and exposure_steps[0].name == "base":
            yield raw_batch
            continue
        group += 1
        for exposure in exposure_steps:
            batch = dict(raw_batch)
            batch["__exposure_name"] = exposure.name
            batch["__exposure_group"] = group
            batch["__composition_index"] = exposure.composition_index
            batch["__objective_terms"] = exposure.objective_terms
            yield batch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a mapping")
    return value


def _positive_number(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{description} must be finite and positive")
    return result


def _nonnegative_number(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{description} must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class OnlineTrainConfig:
    path: Path
    sha256: str
    config_root: Path
    references: dict[str, Path]
    payload: dict[str, Any]

    @property
    def profile_name(self) -> str:
        return str(self.payload["profile_name"])

    def describe(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "config_root": str(self.config_root),
            "references": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in self.references.items()
            },
            "resolved": self.payload,
        }


@dataclass(frozen=True)
class OnlineTrainingResult:
    model: nn.Module
    output_dir: Path
    method_id: str
    scientific_arm: str
    center: str
    epochs_completed: int
    optimizer_steps: int
    last_checkpoint_path: Path
    last_checkpoint_sha256: str
    history: tuple[dict[str, Any], ...]
    model_identity: dict[str, Any]
    config_identity: dict[str, Any]
    seed_identity: dict[str, Any]
    latent_pool_identity: dict[str, Any] | None

    def describe(self) -> dict[str, Any]:
        if len(self.last_checkpoint_sha256) != 64:
            raise ValueError("last_checkpoint_sha256 must contain a SHA256 digest")
        last_checkpoint = {
            "path": str(self.last_checkpoint_path),
            "sha256": self.last_checkpoint_sha256,
        }
        return {
            "output_dir": str(self.output_dir),
            "method_id": self.method_id,
            "scientific_arm": self.scientific_arm,
            "center": self.center,
            "epochs_completed": self.epochs_completed,
            "optimizer_steps": self.optimizer_steps,
            "last_checkpoint": last_checkpoint,
            "selection": {
                "policy": "last",
                "selected_epoch": self.epochs_completed,
                "selected_metric": None,
                "selected_checkpoint": last_checkpoint,
                "heldout_evaluation_used_for_selection": False,
            },
            "model": self.model_identity,
            "config": self.config_identity,
            "seed": self.seed_identity,
            "latent_pool": self.latent_pool_identity,
        }


class _SampledStepTimer:
    """Measure one sampled step without synchronizing every training step."""

    _PHASES = ("h2d", "augmentation", "forward_backward")

    def __init__(
        self,
        *,
        device: torch.device,
        data_wait_ms: float,
        cuda_events: bool,
    ) -> None:
        self.device = device
        self.data_wait_ms = max(0.0, float(data_wait_ms))
        self.use_cuda_events = device.type == "cuda" and bool(cuda_events)
        self._cpu_started: dict[str, float] = {}
        self._cpu_elapsed: dict[str, float] = {}
        self._cuda_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}

    def start(self, phase: str) -> None:
        if phase not in self._PHASES:
            raise ValueError(f"unsupported timing phase: {phase}")
        if self.use_cuda_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self._cuda_events[phase] = (start, end)
        else:
            self._cpu_started[phase] = time.perf_counter()

    def stop(self, phase: str) -> None:
        if self.use_cuda_events:
            self._cuda_events[phase][1].record()
        else:
            started = self._cpu_started.pop(phase)
            self._cpu_elapsed[phase] = max(
                0.0, (time.perf_counter() - started) * 1000.0
            )

    def finish(self, *, batch_size: int, grad_norm: float) -> dict[str, float]:
        if self.use_cuda_events:
            # All phases use the current training stream. Synchronizing the last
            # event once makes every earlier event elapsed-time query valid.
            self._cuda_events[self._PHASES[-1]][1].synchronize()
            elapsed = {
                phase: max(0.0, float(start.elapsed_time(end)))
                for phase, (start, end) in self._cuda_events.items()
            }
        else:
            elapsed = dict(self._cpu_elapsed)
        values = {
            "data_wait_ms": self.data_wait_ms,
            "h2d_ms": elapsed.get("h2d", 0.0),
            "augmentation_ms": elapsed.get("augmentation", 0.0),
            "forward_backward_ms": elapsed.get("forward_backward", 0.0),
        }
        values["step_ms"] = max(
            1.0e-9,
            sum(values[name] for name in (
                "data_wait_ms",
                "h2d_ms",
                "augmentation_ms",
                "forward_backward_ms",
            )),
        )
        values["samples_per_sec"] = float(batch_size) * 1000.0 / values["step_ms"]
        values["grad_norm"] = max(0.0, float(grad_norm))
        return values


def load_online_train_config(
    path: str | Path = DEFAULT_ONLINE_CONFIG_PATH,
    *,
    config_root: str | Path | None = None,
) -> OnlineTrainConfig:
    """Load the strict matched PN2021 K500 experiment contract."""

    config_path = resolve_entry_config_path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"online training config not found: {config_path}") from None
    root_payload = _mapping(payload, "online training config")
    expected_root_keys = {
        "schema_version",
        "profile_name",
        "references",
        "protocol",
        "random_seed",
        "fairness",
        "data",
        "training",
        "diagnostics",
        "logging",
        "output",
    }
    if set(root_payload) != expected_root_keys:
        raise ValueError("online training config keys are incomplete or unexpected")
    if root_payload.get("schema_version") != 1:
        raise ValueError("online training config schema_version must be 1")
    if not isinstance(root_payload.get("profile_name"), str) or not root_payload[
        "profile_name"
    ]:
        raise ValueError("online training profile_name must be non-empty")
    references = _mapping(root_payload.get("references"), "references")
    expected_references = {
        "split_config",
        "data_load_config",
        "random_seed_config",
        "source_baseline_registry",
    }
    if set(references) != expected_references:
        raise ValueError("online training references are incomplete or unexpected")
    root = config_bundle_root(config_path, config_root=config_root)
    resolved_references = {
        name: resolve_config_reference(
            raw,
            owner_config_path=config_path,
            config_root=root,
            description=f"references.{name}",
            must_exist=True,
        )
        for name, raw in references.items()
    }

    protocol = _mapping(root_payload.get("protocol"), "protocol")
    if tuple(protocol.get("centers", ())) != ALLOWED_CENTERS:
        raise ValueError(f"protocol.centers must be {ALLOWED_CENTERS}")
    if protocol.get("partition") != "k500" or not bool(
        protocol.get("use_all_k500", False)
    ):
        raise ValueError("online adaptation must use the complete K500 partition")
    if bool(protocol.get("validation_split", True)):
        raise ValueError("locked exploratory K500 protocol has no validation split")
    if protocol.get("checkpoint_selection") != "last":
        raise ValueError("online K500 checkpoint selection must be last")
    if not bool(protocol.get("heldout_ref_exclusion_required", False)):
        raise ValueError("heldout reference exclusion must be required")
    if tuple(protocol.get("class_order", ())) != CLASS_ORDER:
        raise ValueError("online class order must be CD/HYP/MI/NORM/STTC")
    if protocol.get("mapping_version") != "v7_super5_sjr_rgq_review_20260528":
        raise ValueError("online config uses the wrong PN2021 mapping version")
    if protocol.get("mapping_hash") != "555ec85d5b51":
        raise ValueError("online config uses the wrong PN2021 mapping hash")
    tuning = protocol.get("tuning")
    if tuning is not None:
        tuning = _mapping(tuning, "protocol.tuning")
        if tuning != {
            "train_partition": "k500_tune_train",
            "validation_partition": "k500_tune_validation",
            "train_records_per_center": 400,
            "validation_records_per_center": 100,
        }:
            raise ValueError("protocol.tuning must lock the managed 400/100 split")

    random_seed = _mapping(root_payload.get("random_seed"), "random_seed")
    for key in ("stream_namespace", "comparison_group"):
        if not isinstance(random_seed.get(key), str) or not random_seed[key]:
            raise ValueError(f"random_seed.{key} must be a non-empty string")
    replicate = random_seed.get("replicate_id")
    if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 0:
        raise ValueError("random_seed.replicate_id must be a nonnegative integer")
    if not isinstance(random_seed.get("deterministic_algorithms"), bool):
        raise ValueError("random_seed.deterministic_algorithms must be boolean")

    fairness = _mapping(root_payload.get("fairness"), "fairness")
    common_fairness_keys = {
        "method_profile_required",
        "matched_outer_optimizer_steps",
        "same_source_checkpoint_required",
        "one_optimizer_step_per_base_batch",
        "loss_normalization",
        "ineligible_anchor_policy",
    }
    budget_policy = str(fairness.get("budget_policy", "matched_base"))
    expected_fairness_keys = common_fairness_keys | (
        {"budget_policy"} if "budget_policy" in fairness else set()
    )
    if set(fairness) != expected_fairness_keys:
        raise ValueError("fairness keys are incomplete or unexpected")
    if fairness.get("method_profile_required") is not True:
        raise ValueError("fairness.method_profile_required must be true")
    if fairness.get("same_source_checkpoint_required") is not True:
        raise ValueError("fairness.same_source_checkpoint_required must be true")
    if budget_policy != "matched_base":
        raise ValueError("fairness.budget_policy must be matched_base")
    if fairness.get("matched_outer_optimizer_steps") is not True:
        raise ValueError(
            "matched_base requires fairness.matched_outer_optimizer_steps=true"
        )
    if fairness.get("one_optimizer_step_per_base_batch") is not True:
        raise ValueError(
            "matched_base requires one_optimizer_step_per_base_batch=true"
        )
    if fairness.get("loss_normalization") != "per_origin_record_mean":
        raise ValueError("matched methods must use per-origin record means")
    if fairness.get("ineligible_anchor_policy") != "clean_loss_only":
        raise ValueError("ineligible anchors must remain in the clean loss")

    data = _mapping(root_payload.get("data"), "data")
    if (
        int(data.get("source_sampling_rate_hz", 0)) != 100
        or int(data.get("source_points", 0)) != 1000
        or data.get("source_layout") != "time_channel"
        or data.get("source_units") != "mV"
    ):
        raise ValueError("online source must be raw 100 Hz (1000,12) mV")
    if data.get("augmentation_timing") != "before_per_sample_global_zscore":
        raise ValueError("augmentation must occur before global z-score")
    if data.get("model_normalization") != "per_sample_global_zscore":
        raise ValueError("model normalization must be per-sample global z-score")
    _positive_number(data.get("normalization_epsilon"), "normalization_epsilon")
    quality_gate = _mapping(
        data.get("hard_sample_quality_gate"), "hard_sample_quality_gate"
    )
    if quality_gate.get("policy") != "reject_to_clean_only":
        raise ValueError("invalid hard-sample quality-gate policy")
    if quality_gate.get("require_finite") is not True:
        raise ValueError("hard-sample quality gate must require finite waveforms")
    _positive_number(
        quality_gate.get("minimum_global_std_mV"),
        "hard_sample_quality_gate.minimum_global_std_mV",
    )
    _positive_number(
        quality_gate.get("maximum_absolute_mV"),
        "hard_sample_quality_gate.maximum_absolute_mV",
    )
    upsampling = _mapping(data.get("ecgfounder_upsampling"), "ecgfounder_upsampling")
    if upsampling != {
        "target_sampling_rate_hz": 500,
        "target_points": 5000,
        "interpolation": "linear",
        "align_corners": True,
    }:
        raise ValueError("ECGFounder upsampling contract must be linear 100->500 Hz")

    training = _mapping(root_payload.get("training"), "training")
    if training.get("optimizer") != "adamw" or training.get("scheduler") != "cosine_annealing":
        raise ValueError("online trainer requires AdamW with cosine annealing")
    profiles = _mapping(training.get("model_profiles"), "training.model_profiles")
    if set(profiles) != {EFFICIENTNET1DV2_SPEC.name, ECGFOUNDER_SPEC.name}:
        raise ValueError("online config must define both supported model profiles")
    for model_name, raw_profile in profiles.items():
        profile = _mapping(raw_profile, f"model profile {model_name}")
        epochs = profile.get("epochs")
        batch_size = profile.get("batch_size")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError(f"{model_name}.epochs must be positive")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"{model_name}.batch_size must be positive")
        _positive_number(profile.get("learning_rate"), f"{model_name}.learning_rate")
        _nonnegative_number(profile.get("weight_decay"), f"{model_name}.weight_decay")
        _positive_number(
            profile.get("minimum_learning_rate_ratio"),
            f"{model_name}.minimum_learning_rate_ratio",
        )
    amp = _mapping(training.get("amp"), "training.amp")
    if not isinstance(amp.get("enabled"), bool) or amp.get("dtype") not in {
        "bfloat16",
        "float16",
    }:
        raise ValueError("training.amp must define enabled and a supported dtype")
    _positive_number(training.get("gradient_clip_norm"), "gradient_clip_norm")

    diagnostics = _mapping(root_payload.get("diagnostics"), "diagnostics")
    diagnostics_interval = diagnostics.get("diagnostics_every_steps")
    if (
        isinstance(diagnostics_interval, bool)
        or not isinstance(diagnostics_interval, int)
        or diagnostics_interval <= 0
    ):
        raise ValueError("diagnostics_every_steps must be a positive integer")
    if not isinstance(diagnostics.get("persist_epoch_ineligible_hashes"), bool):
        raise ValueError("persist_epoch_ineligible_hashes must be boolean")
    performance_timing = _mapping(
        diagnostics.get("performance_timing"), "diagnostics.performance_timing"
    )
    if set(performance_timing) != {
        "enabled",
        "interval_steps",
        "cuda_events",
        "scope",
    }:
        raise ValueError("performance_timing keys are incomplete or unexpected")
    if not isinstance(performance_timing.get("enabled"), bool):
        raise ValueError("performance_timing.enabled must be boolean")
    timing_interval = performance_timing.get("interval_steps")
    if (
        isinstance(timing_interval, bool)
        or not isinstance(timing_interval, int)
        or timing_interval <= 0
    ):
        raise ValueError("performance_timing.interval_steps must be positive")
    if performance_timing.get("cuda_events") is not True:
        raise ValueError("sampled CUDA timing must use cuda_events=true")
    if performance_timing.get("scope") != "sampled_core_step_excludes_logging":
        raise ValueError("performance_timing.scope must disclose logging exclusion")
    parse_tensorboard_logging_config(
        _mapping(root_payload.get("logging"), "logging")
    )
    output = _mapping(root_payload.get("output"), "output")
    if output.get("if_exists") != "error":
        raise ValueError("online training may not overwrite an existing run")
    checkpoint_write_policy = str(
        output.get("checkpoint_write_policy", "every_epoch")
    )
    if checkpoint_write_policy not in {"every_epoch", "final_only"}:
        raise ValueError(
            "output.checkpoint_write_policy must be every_epoch or final_only"
        )

    return OnlineTrainConfig(
        path=config_path,
        sha256=_sha256(config_path),
        config_root=root,
        references=resolved_references,
        payload=root_payload,
    )


def resolve_online_training_parameters(
    config: OnlineTrainConfig,
    model_name: str,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    supplied = {} if overrides is None else dict(overrides)
    unknown = sorted(set(supplied) - ONLINE_PARAMETER_NAMES)
    if unknown:
        raise ValueError(f"unknown online training parameters: {unknown}")
    training = config.payload["training"]
    profile = training["model_profiles"].get(model_name)
    if not isinstance(profile, dict):
        raise ValueError(f"unsupported online model_name: {model_name!r}")
    amp = training["amp"]
    resolved = {
        "epochs": supplied.get("epochs", profile["epochs"]),
        "scheduler_horizon_epochs": supplied.get(
            "scheduler_horizon_epochs",
            supplied.get("epochs", profile["epochs"]),
        ),
        "batch_size": supplied.get("batch_size", profile["batch_size"]),
        "learning_rate": supplied.get("learning_rate", profile["learning_rate"]),
        "weight_decay": supplied.get("weight_decay", profile["weight_decay"]),
        "minimum_learning_rate_ratio": supplied.get(
            "minimum_learning_rate_ratio", profile["minimum_learning_rate_ratio"]
        ),
        "gradient_clip_norm": supplied.get(
            "gradient_clip_norm", training["gradient_clip_norm"]
        ),
        "amp_enabled": supplied.get("amp_enabled", amp["enabled"]),
        "amp_dtype": supplied.get("amp_dtype", amp["dtype"]),
    }
    for key in ("epochs", "scheduler_horizon_epochs", "batch_size"):
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
        resolved[key] = _positive_number(resolved[key], f"resolved {key}")
    resolved["weight_decay"] = _nonnegative_number(
        resolved["weight_decay"], "resolved weight_decay"
    )
    if not isinstance(resolved["amp_enabled"], bool):
        raise ValueError("resolved amp_enabled must be boolean")
    if resolved["amp_dtype"] not in {"bfloat16", "float16"}:
        raise ValueError("resolved amp_dtype must be bfloat16 or float16")
    return resolved, supplied


def _model_spec(model: nn.Module) -> ModelSpec:
    owner = model.module if hasattr(model, "module") else model
    spec = getattr(owner, "model_spec", None)
    if spec not in {EFFICIENTNET1DV2_SPEC, ECGFOUNDER_SPEC}:
        raise ValueError("online trainer requires the managed EfficientNet or ECGFounder spec")
    return spec


def _model_identity(model: nn.Module, spec: ModelSpec) -> dict[str, Any]:
    owner = model.module if hasattr(model, "module") else model
    result: dict[str, Any] = {
        "spec": spec.describe(),
        "total_parameter_count": int(sum(value.numel() for value in model.parameters())),
        "trainable_parameter_count": int(
            sum(value.numel() for value in model.parameters() if value.requires_grad)
        ),
    }
    for name in (
        "checkpoint_identity",
        "pretrained_checkpoint_identity",
        "task_checkpoint_identity",
        "random_seed_identity",
    ):
        identity = getattr(owner, name, None)
        describe = getattr(identity, "describe", None)
        result[name] = describe() if callable(describe) else None
    return result


def _batch_hashes(batch: Mapping[str, Any], batch_size: int) -> tuple[str, ...]:
    raw = batch.get("hash_id")
    if not isinstance(raw, (list, tuple)) or len(raw) != batch_size:
        raise ValueError("online batches must provide one hash_id string per record")
    values = tuple(str(value) for value in raw)
    if any(not value for value in values) or len(set(values)) != len(values):
        raise ValueError("online batch hash_ids must be non-empty and unique")
    return values


def _loader_selection_hashes(train_dataloader: Any) -> tuple[str, ...] | None:
    """Return the immutable managed selection, when the loader exposes one."""

    dataset = getattr(train_dataloader, "dataset", None)
    selection = getattr(dataset, "selection", None)
    raw_hashes = getattr(selection, "hash_ids", None)
    if raw_hashes is None:
        return None
    loader_hashes = tuple(str(value) for value in raw_hashes)
    if len(set(loader_hashes)) != len(loader_hashes):
        raise RuntimeError("training loader selection contains duplicate hash IDs")
    return loader_hashes


def _loader_selection_partition(train_dataloader: Any) -> str | None:
    dataset = getattr(train_dataloader, "dataset", None)
    selection = getattr(dataset, "selection", None)
    raw = getattr(selection, "partition", None)
    return None if raw is None else str(raw)


def _preflight_loader_pool_hashes(train_dataloader: Any, latent_pool: Any) -> None:
    """Fail before optimization when a managed loader and pool differ."""

    loader_hashes = _loader_selection_hashes(train_dataloader)
    if loader_hashes is None:
        return
    pool_hashes = tuple(str(value) for value in latent_pool.hash_ids)
    if len(loader_hashes) != len(pool_hashes) or set(loader_hashes) != set(pool_hashes):
        missing = sorted(set(loader_hashes) - set(pool_hashes))
        extra = sorted(set(pool_hashes) - set(loader_hashes))
        raise RuntimeError(
            "training loader and latent pool K500 membership differ: "
            f"missing_from_pool={missing[:5]}, extra_in_pool={extra[:5]}"
        )


def _pool_identity(pool: Any | None) -> dict[str, Any] | None:
    if pool is None:
        return None
    describe = getattr(pool, "describe", None)
    if callable(describe):
        value = describe()
        if isinstance(value, dict):
            return value
    identity = getattr(pool, "identity", None)
    describe = getattr(identity, "describe", None)
    if callable(describe):
        return describe()
    if isinstance(identity, dict):
        return dict(identity)
    raise TypeError("latent_pool must expose a serializable identity")


def _module_checkpoint_identity(module: nn.Module | None) -> dict[str, Any] | None:
    if module is None:
        return None
    identity = getattr(module, "checkpoint_identity", None)
    describe = getattr(identity, "describe", None)
    if not callable(describe):
        raise ValueError("managed VAE components must expose checkpoint_identity")
    value = describe()
    if not isinstance(value, dict) or len(str(value.get("sha256", ""))) != 64:
        raise ValueError("managed VAE checkpoint identity is invalid")
    return value


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")


def _output_member(root: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError("output member names must be non-empty strings")
    path = (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError("output member escapes the run directory") from None
    return path


def _default_output_dir(
    config: OnlineTrainConfig,
    *,
    center: str,
    model_name: str,
    method_id: str,
) -> Path:
    root = Path(config.payload["output"]["root_dir"]).expanduser().resolve()
    replicate = int(config.payload["random_seed"]["replicate_id"])
    return root / center / model_name / method_id / f"replicate_{replicate}"


@dataclass(frozen=True)
class _ObjectiveBatch:
    total: torch.Tensor
    raw_terms: dict[str, torch.Tensor]
    weighted_terms: dict[str, torch.Tensor]
    valid_counts: dict[str, int]


def _compute_objective(
    *,
    method: CompiledMethod,
    bundle: Any,
    model: nn.Module,
    spec: ModelSpec,
    normalization_epsilon: float,
    pos_weight: torch.Tensor | None,
    objective_term_names: Sequence[str] | None = None,
    batch_norm_plan: _BatchNormMomentumPlan | None = None,
) -> _ObjectiveBatch:
    """Compute profile-declared losses with one mean contribution per origin.

    Each term is averaged over its valid records and then multiplied by E/B.
    Thus extra views cannot silently amplify records that lack that view, while
    the clean term continues to cover every base record.
    """

    clean = bundle.require(BASE_VIEW_NAME)
    if not isinstance(clean, WaveformView):
        raise TypeError("method clean_view must be a WaveformView")
    batch_size = clean.batch_size
    if not bool(clean.valid_mask.all().item()):
        raise ValueError("clean_view must be valid for every base record")

    selected_names = (
        None
        if objective_term_names is None
        else tuple(str(value) for value in objective_term_names)
    )
    if selected_names is not None:
        if not selected_names or len(set(selected_names)) != len(selected_names):
            raise ValueError("objective_term_names must be non-empty and unique")
        known_names = {term.name for term in method.objective.terms}
        unknown_names = sorted(set(selected_names) - known_names)
        if unknown_names:
            raise ValueError(f"unknown objective term names: {unknown_names}")
    selected_terms = tuple(
        term
        for term in method.objective.terms
        if selected_names is None or term.name in selected_names
    )

    ordered_views: list[str] = []
    for term in selected_terms:
        for name in term.views:
            if name not in ordered_views:
                ordered_views.append(name)
    if not ordered_views:
        raise ValueError("method objective must contain at least one term")

    views: dict[str, WaveformView] = {}
    positions: dict[str, torch.Tensor] = {}
    logits: dict[str, torch.Tensor] = {}
    full_to_local: dict[str, torch.Tensor] = {}
    if batch_norm_plan is not None and len(batch_norm_plan.exposure_weights) != len(
        ordered_views
    ):
        raise ValueError(
            "objective-view BatchNorm plan does not match the objective views"
        )
    for view_index, name in enumerate(ordered_views):
        value = bundle.require(name)
        if not isinstance(value, WaveformView):
            raise TypeError(f"objective view {name!r} must be a WaveformView")
        if value.sample_ids != clean.sample_ids:
            raise RuntimeError(f"objective view {name!r} changed origin order")
        if not torch.equal(value.labels, clean.labels):
            raise RuntimeError(f"objective view {name!r} changed labels")
        valid_positions = torch.nonzero(value.valid_mask, as_tuple=False).flatten()
        views[name] = value
        positions[name] = valid_positions
        forward_positions = (
            torch.arange(batch_size, device=value.waveform.device, dtype=torch.long)
            if batch_norm_plan is not None
            else valid_positions
        )
        lookup = torch.full(
            (batch_size,),
            -1,
            device=value.waveform.device,
            dtype=torch.long,
        )
        if forward_positions.numel():
            lookup[forward_positions] = torch.arange(
                forward_positions.numel(),
                device=value.waveform.device,
                dtype=torch.long,
            )
            raw_subset = value.waveform.index_select(0, forward_positions)
            model_input = prepare_canonical_model_input(
                raw_subset,
                spec,
                epsilon=normalization_epsilon,
            )
            if batch_norm_plan is not None:
                batch_norm_plan.apply(view_index)
            logits[name] = validate_model_output(
                model(model_input),
                spec,
                batch_size=int(forward_positions.numel()),
                check_finite=False,
            )
        else:
            logits[name] = clean.waveform.new_empty((0, 5))
        full_to_local[name] = lookup

    reference = logits[ordered_views[0]]
    total = reference.sum() * 0.0
    raw_terms: dict[str, torch.Tensor] = {}
    weighted_terms: dict[str, torch.Tensor] = {}
    valid_counts: dict[str, int] = {}
    for term in selected_terms:
        if term.kind == "bce":
            name = term.views[0]
            count = int(positions[name].numel())
            if count:
                targets = views[name].labels.index_select(0, positions[name])
                selected_logits = logits[name].index_select(
                    0,
                    full_to_local[name].index_select(0, positions[name]),
                )
                raw_loss = F.binary_cross_entropy_with_logits(
                    selected_logits, targets, pos_weight=pos_weight
                )
            else:
                raw_loss = reference.sum() * 0.0
        elif term.kind == "bernoulli_jsd":
            common = torch.ones(
                batch_size,
                device=clean.waveform.device,
                dtype=torch.bool,
            )
            for name in term.views:
                common &= views[name].valid_mask
            common_positions = torch.nonzero(common, as_tuple=False).flatten()
            count = int(common_positions.numel())
            if count:
                aligned_logits = tuple(
                    logits[name].index_select(
                        0,
                        full_to_local[name].index_select(0, common_positions),
                    )
                    for name in term.views
                )
                raw_loss = multilabel_jsd(aligned_logits)
            else:
                raw_loss = reference.sum() * 0.0
        else:
            raise RuntimeError(f"unsupported compiled objective kind: {term.kind}")
        contribution = float(term.weight) * (float(count) / batch_size) * raw_loss
        raw_terms[term.name] = raw_loss
        weighted_terms[term.name] = contribution
        valid_counts[term.name] = count
        total = total + contribution
    return _ObjectiveBatch(
        total=total,
        raw_terms=raw_terms,
        weighted_terms=weighted_terms,
        valid_counts=valid_counts,
    )


def _legacy_loss_aliases(
    raw_terms: Mapping[str, float],
    weighted_terms: Mapping[str, float],
) -> dict[str, float]:
    """Keep the locked A5 TensorBoard names while exposing generic terms."""

    aliases: dict[str, float] = {}
    raw_names = {
        "clean_bce": "clean_bce",
        "lhat_direct_bce": "lhat_hard_bce",
        "augmix_bce": "augmix_bce",
        "clean_lhat_augmix_jsd": "augmix_jsd",
    }
    weighted_names = {
        "clean_bce": "weighted_clean",
        "lhat_direct_bce": "weighted_lhat",
        "augmix_bce": "weighted_augmix",
        "clean_lhat_augmix_jsd": "weighted_jsd",
    }
    for source, target in raw_names.items():
        if source in raw_terms:
            aliases[target] = float(raw_terms[source])
    for source, target in weighted_names.items():
        if source in weighted_terms:
            aliases[target] = float(weighted_terms[source])
    return aliases


def _objective_host_scalars(
    objective: _ObjectiveBatch,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Transfer one sampled objective record to the host in one synchronization."""

    raw_names = tuple(objective.raw_terms)
    weighted_names = tuple(objective.weighted_terms)
    tensors = (
        objective.total,
        *(objective.raw_terms[name] for name in raw_names),
        *(objective.weighted_terms[name] for name in weighted_names),
    )
    values = (
        torch.stack(
            tuple(
                value.detach().to(dtype=torch.float32).reshape(())
                for value in tensors
            )
        )
        .cpu()
        .tolist()
    )
    raw_offset = 1
    weighted_offset = raw_offset + len(raw_names)
    return (
        float(values[0]),
        {
            name: float(values[raw_offset + index])
            for index, name in enumerate(raw_names)
        },
        {
            name: float(values[weighted_offset + index])
            for index, name in enumerate(weighted_names)
        },
    )


def train_online_model(
    model: nn.Module,
    train_dataloader: Any,
    *,
    center: str,
    method_config_path: str | Path,
    latent_pool: "LatentPool | None" = None,
    encoder: nn.Module | None = None,
    decoder: nn.Module | None = None,
    config_path: str | Path = DEFAULT_ONLINE_CONFIG_PATH,
    config_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    training_parameters: Mapping[str, Any] | None = None,
    pos_weight: torch.Tensor | Sequence[float] | None = None,
    epoch_evaluator: Callable[
        [nn.Module, int, torch.device, Mapping[str, Any]], Mapping[str, Any]
    ]
    | None = None,
) -> OnlineTrainingResult:
    """Train one file-backed typed method under its locked K500 budget.

    The ``matched_base`` budget performs one optimizer step per base batch.
    Fixed-20 expands the views used to form that objective but accumulates all
    gradients before the one step. ``epoch_evaluator`` is a caller-owned,
    read-only validation hook; it cannot select checkpoints or alter training.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if train_dataloader is None:
        raise ValueError("train_dataloader is required")
    if center not in ALLOWED_CENTERS:
        raise ValueError(f"center must be one of {ALLOWED_CENTERS}")

    config = load_online_train_config(config_path, config_root=config_root)
    method_path = resolve_config_reference(
        str(method_config_path),
        owner_config_path=config.path,
        config_root=config.config_root,
        description="method_config_path",
        must_exist=True,
    )
    try:
        method_path.relative_to(config.config_root)
    except ValueError:
        raise ValueError(
            "method profile must belong to the selected config bundle"
        ) from None
    method = compile_method_profile(method_path)
    if not method.executable:
        raise ValueError(
            f"method profile {method.profile_name!r} is audit-only and not executable"
        )
    clean_terms = [
        term
        for term in method.objective.terms
        if term.kind == "bce"
        and term.views == (BASE_VIEW_NAME,)
        and float(term.weight) == 1.0
    ]
    if len(clean_terms) != 1:
        raise ValueError(
            "every executable method must declare one unit-weight clean_view BCE"
        )
    exposure_steps = _method_exposure_steps(method)
    fixed20_exposure = len(exposure_steps) == 21
    fairness = config.payload["fairness"]
    budget_policy = str(fairness.get("budget_policy", "matched_base"))
    if budget_policy != "matched_base":
        raise ValueError("online methods require the matched_base budget")

    spec = _model_spec(model)
    resolved, supplied = resolve_online_training_parameters(
        config, spec.name, training_parameters
    )
    configured_batch = getattr(train_dataloader, "batch_size", None)
    if configured_batch is not None and int(configured_batch) != int(
        resolved["batch_size"]
    ):
        raise ValueError(
            "train_dataloader batch_size does not match the resolved online profile"
        )
    train_partition = _loader_selection_partition(train_dataloader)
    if train_partition == "k500":
        pass
    elif train_partition == "k500_tune_train":
        if config.payload["protocol"].get("tuning") is None:
            raise ValueError(
                "k500_tune_train requires a training config with protocol.tuning"
            )
    elif train_partition is not None:
        raise ValueError(
            "online training accepts only k500 or k500_tune_train, got "
            f"{train_partition!r}"
        )

    requires_latent = bool(method.requirements.latent_pool)
    requires_encoder = bool(method.requirements.vae_encoder)
    requires_decoder = bool(method.requirements.vae_decoder)
    if requires_latent != (latent_pool is not None):
        raise ValueError(
            "latent_pool presence must exactly match the method profile requirement"
        )
    if requires_decoder != (decoder is not None):
        raise ValueError(
            "VAE decoder presence must exactly match the method profile requirement"
        )
    if requires_encoder != (encoder is not None):
        raise ValueError(
            "VAE encoder presence must exactly match the method profile requirement"
        )

    requested_device = str(device or config.payload["training"]["device"])
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_device = torch.device(requested_device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if resolved_device.type not in {"cpu", "cuda"}:
        raise ValueError("online device must be CPU or CUDA")

    output = (
        _default_output_dir(
            config,
            center=center,
            model_name=spec.name,
            method_id=method.profile_name,
        )
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    try:
        output.relative_to(PROJECT_ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("online outputs must stay outside the Git worktree")
    output.mkdir(parents=True, exist_ok=False)
    output_config = config.payload["output"]
    checkpoint_write_policy = str(
        output_config.get("checkpoint_write_policy", "every_epoch")
    )
    checkpoint_dir = _output_member(output, output_config["checkpoints_dir"])
    checkpoint_dir.mkdir()
    last_checkpoint = _output_member(
        checkpoint_dir, output_config["last_checkpoint_file"]
    )
    history_path = _output_member(output, output_config["history_file"])
    diagnostics_path = _output_member(output, output_config["diagnostics_file"])
    result_path = _output_member(output, output_config["result_file"])
    method_resources_path = _output_member(
        output, output_config["method_resources_file"]
    )

    random_seed = config.payload["random_seed"]
    seed = seed_process(
        str(random_seed["stream_namespace"]),
        str(random_seed["comparison_group"]),
        int(random_seed["replicate_id"]),
        center,
        spec.name,
        config_path=config.references["random_seed_config"],
        deterministic_algorithms=bool(random_seed["deterministic_algorithms"]),
    )
    model.to(resolved_device)
    batch_norm_plan = _build_batch_norm_momentum_plan(
        model,
        method,
        fixed20_exposure=fixed20_exposure,
    )
    if encoder is not None:
        encoder.to(resolved_device).eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
    if decoder is not None:
        decoder.to(resolved_device).eval()
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)
    if latent_pool is not None:
        move_pool = getattr(latent_pool, "to", None)
        if not callable(move_pool):
            raise TypeError("latent_pool must provide to(device)")
        latent_pool = move_pool(resolved_device)
        _preflight_loader_pool_hashes(train_dataloader, latent_pool)

    managed_loader_hashes = _loader_selection_hashes(train_dataloader)
    expected_epoch_hashes = (
        set(str(value) for value in latent_pool.hash_ids)
        if latent_pool is not None
        else (
            set(managed_loader_hashes)
            if managed_loader_hashes is not None
            else None
        )
    )
    pool_identity = _pool_identity(latent_pool)
    encoder_identity = _module_checkpoint_identity(encoder)
    decoder_identity = _module_checkpoint_identity(decoder)

    quality = config.payload["data"]["hard_sample_quality_gate"]
    runtime = build_method_runtime(
        method,
        model_name=spec.name,
        config_root=config.config_root,
        latent_pool=latent_pool,
        encoder=encoder,
        decoder=decoder,
        minimum_std_mV=float(quality["minimum_global_std_mV"]),
        maximum_abs_mV=float(quality["maximum_absolute_mV"]),
    )
    if (
        runtime.augmix_config is not None
        and runtime.augmix_config.random_seed_config_path
        != config.references["random_seed_config"]
    ):
        raise ValueError(
            "online training and AugMix must use the same random_seed config"
        )
    mismatched_method_rngs = sorted(
        name
        for name, path in runtime.rng_seed_config_paths.items()
        if path != config.references["random_seed_config"]
    )
    if mismatched_method_rngs:
        raise ValueError(
            "online training and method RNG resources must use the same "
            "random_seed config: "
            + ", ".join(mismatched_method_rngs)
        )
    method_resource_identity = runtime.describe()
    method_resource_identity["latent_pool"] = pool_identity
    method_resource_identity["vae_encoder_checkpoint"] = encoder_identity
    method_resource_identity["vae_decoder_checkpoint"] = decoder_identity
    _write_json(method_resources_path, method_resource_identity)

    trainable = [value for value in model.parameters() if value.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    optimizer = AdamW(
        trainable,
        lr=float(resolved["learning_rate"]),
        weight_decay=float(resolved["weight_decay"]),
    )
    epochs = int(resolved["epochs"])
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(resolved["scheduler_horizon_epochs"]),
        eta_min=float(resolved["learning_rate"])
        * float(resolved["minimum_learning_rate_ratio"]),
    )
    amp_enabled = bool(resolved["amp_enabled"]) and resolved_device.type == "cuda"
    amp_dtype = (
        torch.bfloat16 if resolved["amp_dtype"] == "bfloat16" else torch.float16
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled and amp_dtype == torch.float16
    )
    resolved_pos_weight = (
        None
        if pos_weight is None
        else torch.as_tensor(
            pos_weight, dtype=torch.float32, device=resolved_device
        )
    )
    if resolved_pos_weight is not None and tuple(resolved_pos_weight.shape) != (5,):
        raise ValueError("pos_weight must contain five Super5 values")

    epsilon = float(config.payload["data"]["normalization_epsilon"])
    diagnostics_interval = int(
        config.payload["diagnostics"]["diagnostics_every_steps"]
    )
    timing_config = config.payload["diagnostics"]["performance_timing"]
    timing_enabled = bool(timing_config["enabled"])
    timing_interval = int(timing_config["interval_steps"])
    tensorboard_step_interval = int(
        config.payload["logging"]["tensorboard"]["scalars"][
            "batch_loss_interval_steps"
        ]
    )
    monitor = build_tensorboard_monitor(config.payload["logging"], output)
    run_identity = {
        "config": config.describe(),
        "method": method_resource_identity,
        "seed": seed.describe(),
        "center": center,
        "scientific_arm": method.scientific_arm,
        "model": _model_identity(model, spec),
        "vae_encoder_checkpoint": encoder_identity,
        "vae_decoder_checkpoint": decoder_identity,
        "training_parameters": {
            "resolved": resolved,
            "explicit_overrides": supplied,
        },
        "pos_weight": (
            None
            if resolved_pos_weight is None
            else resolved_pos_weight.detach().cpu().tolist()
        ),
        "exposure_plan": {
            "budget_policy": budget_policy,
            "optimizer_steps_per_base_batch": 1,
            "view_executions_per_base_batch": len(exposure_steps),
            "family_loss_weights": (
                method.contracts.get("family_loss_weights")
                if fixed20_exposure
                else None
            ),
            "steps": [
                {
                    "name": exposure.name,
                    "composition_index": exposure.composition_index,
                    "objective_terms": (
                        None
                        if exposure.objective_terms is None
                        else list(exposure.objective_terms)
                    ),
                }
                for exposure in exposure_steps
            ],
            "materialized_corruption_cache": False,
            "batch_norm_running_stats": (
                None if batch_norm_plan is None else batch_norm_plan.describe()
            ),
        },
        "method_rng_derivation": {
            "base_seed": seed.base_seed,
            "profile_namespace": method.rng_namespace,
            "execution_identity": [
                "comparison_group",
                "replicate_id",
                "center",
                "model",
                "epoch",
                "view_execution_step",
                "ordered_batch_hash_sha256",
            ],
            "persistent_generator_state_required": False,
        },
        "latent_pool": pool_identity,
        "dataloader": (
            train_dataloader.describe()
            if callable(getattr(train_dataloader, "describe", None))
            else {
                "type": train_dataloader.__class__.__name__,
                "batch_size": configured_batch,
            }
        ),
        "training_partition": train_partition,
        "logging": monitor.describe(),
        "epoch_evaluator": (
            None
            if epoch_evaluator is None
            else (
                epoch_evaluator.describe()
                if callable(getattr(epoch_evaluator, "describe", None))
                else {
                    "type": (
                        f"{epoch_evaluator.__class__.__module__}."
                        f"{epoch_evaluator.__class__.__qualname__}"
                    )
                }
            )
        ),
    }

    history: list[dict[str, Any]] = []
    view_execution_step = 0
    optimizer_steps = 0
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer_steps_at_epoch_start = optimizer_steps
            epoch_base_batches = 0
            epoch_loss_sum = torch.zeros((), device=resolved_device)
            epoch_samples = 0
            epoch_origin_samples = 0
            epoch_term_sums = {
                term.name: torch.zeros((), device=resolved_device)
                for term in method.objective.terms
            }
            epoch_weighted_sums = {
                term.name: torch.zeros((), device=resolved_device)
                for term in method.objective.terms
            }
            epoch_term_counts = {
                term.name: 0 for term in method.objective.terms
            }
            epoch_view_count_sums = {
                name: torch.zeros((), device=resolved_device, dtype=torch.int64)
                for name in method.outputs
            }
            epoch_candidate_eligible = 0
            epoch_quality_accepted = 0
            epoch_quality_view_total = 0
            epoch_quality_view_accepted = 0
            epoch_ineligible_hashes: list[str] = []
            epoch_quality_rejected: list[dict[str, str]] = []
            diagnostic_sums: dict[str, float] = {}
            diagnostic_weights: dict[str, int] = {}
            performance_sums: dict[str, float] = {}
            timed_step_count = 0
            epoch_seen_hashes: dict[str, set[str]] = {
                exposure.name: set() for exposure in exposure_steps
            }
            epoch_exposure_counts: dict[str, int] = {
                exposure.name: 0 for exposure in exposure_steps
            }
            learning_rate = float(optimizer.param_groups[0]["lr"])
            previous_step_end = time.perf_counter()
            cached_exposure_group: int | None = None
            cached_raw: torch.Tensor | None = None
            cached_targets: torch.Tensor | None = None

            for batch in _iter_exposure_batches(train_dataloader, exposure_steps):
                batch_received = time.perf_counter()
                data_wait_ms = max(
                    0.0, (batch_received - previous_step_end) * 1000.0
                )
                next_execution_step = view_execution_step + 1
                timer = (
                    _SampledStepTimer(
                        device=resolved_device,
                        data_wait_ms=data_wait_ms,
                        cuda_events=bool(timing_config["cuda_events"]),
                    )
                    if timing_enabled
                    and next_execution_step % timing_interval == 0
                    else None
                )
                if not isinstance(batch, Mapping):
                    raise TypeError("online dataloader batches must be mappings")
                exposure_name = str(batch.get("__exposure_name", "base"))
                if exposure_name not in epoch_seen_hashes:
                    raise RuntimeError(f"unknown online exposure: {exposure_name!r}")
                exposure_group_raw = batch.get("__exposure_group")
                exposure_group = (
                    None
                    if exposure_group_raw is None
                    else int(exposure_group_raw)
                )
                composition_index_raw = batch.get("__composition_index")
                composition_index = (
                    None
                    if composition_index_raw is None
                    else int(composition_index_raw)
                )
                objective_terms_raw = batch.get("__objective_terms")
                objective_terms = (
                    None
                    if objective_terms_raw is None
                    else tuple(str(value) for value in objective_terms_raw)
                )
                raw = batch.get("waveform")
                targets = batch.get("label")
                if not isinstance(raw, torch.Tensor) or not isinstance(
                    targets, torch.Tensor
                ):
                    raise TypeError(
                        "online batch waveform and label must be tensors"
                    )
                batch_size = int(raw.shape[0])
                if tuple(raw.shape[1:]) != (1000, 12):
                    raise ValueError(
                        "online waveform must be raw-mV (B,1000,12)"
                    )
                if targets.shape != (batch_size, 5):
                    raise ValueError("online labels must have shape (B,5)")
                hashes = _batch_hashes(batch, batch_size)
                exposure_seen = epoch_seen_hashes[exposure_name]
                repeated = exposure_seen.intersection(hashes)
                if repeated:
                    raise RuntimeError(
                        f"online exposure {exposure_name!r} contains repeated hash IDs: "
                        f"{sorted(repeated)[:5]}"
                    )
                exposure_seen.update(hashes)
                epoch_exposure_counts[exposure_name] += batch_size

                if timer is not None:
                    timer.start("h2d")
                if (
                    exposure_group is not None
                    and cached_exposure_group == exposure_group
                    and cached_raw is not None
                    and cached_targets is not None
                ):
                    raw = cached_raw
                    targets = cached_targets
                else:
                    raw = raw.to(
                        resolved_device, dtype=torch.float32, non_blocking=True
                    )
                    targets = targets.to(
                        resolved_device, dtype=torch.float32, non_blocking=True
                    )
                    if exposure_group is not None:
                        cached_exposure_group = exposure_group
                        cached_raw = raw
                        cached_targets = targets
                if timer is not None:
                    timer.stop("h2d")
                    timer.start("augmentation")

                group_start = not fixed20_exposure or exposure_name == "clean"
                group_end = (
                    not fixed20_exposure or composition_index == 19
                )
                if group_start:
                    optimizer.zero_grad(set_to_none=True)
                    epoch_origin_samples += batch_size
                    epoch_base_batches += 1
                family_loss_scale = (
                    1.0
                    if not fixed20_exposure
                    else (0.5 if exposure_name == "clean" else 0.025)
                )
                if batch_norm_plan is not None and fixed20_exposure:
                    batch_norm_plan.apply(
                        0 if composition_index == -1 else int(composition_index) + 1
                    )
                hash_digest = hashlib.sha256(
                    "\n".join(hashes).encode("utf-8")
                ).hexdigest()
                generated = runtime.generate(
                    clean_raw=raw,
                    targets=targets,
                    hash_ids=hashes,
                    classifier=model,
                    base_seed=seed.base_seed,
                    rng_identity=(
                        str(random_seed["comparison_group"]),
                        str(random_seed["replicate_id"]),
                        center,
                        spec.name,
                        f"epoch={epoch}",
                        f"view_execution_step={next_execution_step}",
                        f"exposure={exposure_name}",
                        f"batch_hash_sha256={hash_digest}",
                    ),
                    composition_indices=(
                        None
                        if composition_index is None
                        else torch.full(
                            (batch_size,),
                            composition_index,
                            device=resolved_device,
                            dtype=torch.int64,
                        )
                    ),
                )
                if timer is not None:
                    timer.stop("augmentation")
                    timer.start("forward_backward")

                with torch.cuda.amp.autocast(
                    enabled=amp_enabled, dtype=amp_dtype
                ):
                    objective = _compute_objective(
                        method=method,
                        bundle=generated.bundle,
                        model=model,
                        spec=spec,
                        normalization_epsilon=epsilon,
                        pos_weight=resolved_pos_weight,
                        objective_term_names=objective_terms,
                        batch_norm_plan=(
                            None if fixed20_exposure else batch_norm_plan
                        ),
                    )
                if not bool(torch.isfinite(objective.total).item()):
                    raise FloatingPointError(
                        "online training loss became NaN or Inf"
                    )
                scaled_objective = objective.total * family_loss_scale
                if scaler.is_enabled():
                    scaler.scale(scaled_objective).backward()
                    if group_end:
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            trainable, float(resolved["gradient_clip_norm"])
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grad_norm = torch.zeros((), device=resolved_device)
                else:
                    scaled_objective.backward()
                    if group_end:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            trainable, float(resolved["gradient_clip_norm"])
                        )
                        optimizer.step()
                    else:
                        grad_norm = torch.zeros((), device=resolved_device)
                if group_end:
                    optimizer_steps += 1
                if timer is not None:
                    timer.stop("forward_backward")

                view_execution_step += 1
                performance = (
                    None
                    if timer is None
                    else timer.finish(
                        batch_size=batch_size,
                        grad_norm=float(grad_norm),
                    )
                )
                if performance is not None:
                    timed_step_count += 1
                    for name, value in performance.items():
                        performance_sums[name] = (
                            performance_sums.get(name, 0.0) + float(value)
                        )
                    _append_jsonl(
                        diagnostics_path,
                        {
                            "kind": "performance_step",
                            "epoch": epoch,
                            "view_execution_step": view_execution_step,
                            "optimizer_step": optimizer_steps,
                            "metrics": performance,
                        },
                    )

                write_tensorboard_step = (
                    view_execution_step % tensorboard_step_interval == 0
                )
                write_diagnostic_step = (
                    view_execution_step % diagnostics_interval == 0
                )
                raw_scalars: dict[str, float] = {}
                weighted_scalars: dict[str, float] = {}
                objective_total_scalar: float | None = None
                if write_tensorboard_step or write_diagnostic_step:
                    (
                        objective_total_scalar,
                        raw_scalars,
                        weighted_scalars,
                    ) = _objective_host_scalars(objective)
                    objective_total_scalar *= family_loss_scale
                    weighted_scalars = {
                        name: value * family_loss_scale
                        for name, value in weighted_scalars.items()
                    }
                if write_tensorboard_step:
                    if objective_total_scalar is None:
                        raise RuntimeError("sampled objective scalar is unavailable")
                    step_components = {
                        **raw_scalars,
                        **{
                            f"weighted_{name}": value
                            for name, value in weighted_scalars.items()
                        },
                        **_legacy_loss_aliases(raw_scalars, weighted_scalars),
                    }
                    monitor.log_train_step(
                        loss=objective_total_scalar,
                        learning_rate=learning_rate,
                        global_step=view_execution_step,
                        loss_components=step_components,
                        performance=performance,
                    )

                probe_logger = getattr(monitor, "log_ecg_views", None)
                probe_preflight = getattr(monitor, "should_log_ecg_views", None)
                fixed20_probe_indices = tuple(
                    int(value)
                    for value in method.contracts.get(
                        "tensorboard_probe_composition_indices", ()
                    )
                )
                probe_allowed = (
                    not fixed20_exposure
                    or composition_index == -1
                    or composition_index in fixed20_probe_indices
                )
                probe_id = (
                    "default"
                    if not fixed20_exposure
                    else (
                        "clean"
                        if composition_index == -1
                        else f"composition_{int(composition_index):02d}"
                    )
                )
                if callable(probe_logger) and probe_allowed:
                    is_final_epoch = epoch == epochs
                    for batch_position, hash_id in enumerate(hashes):
                        if callable(probe_preflight) and not probe_preflight(
                            hash_id=hash_id,
                            epoch=epoch,
                            probe_id=probe_id,
                            is_final_epoch=is_final_epoch,
                        ):
                            continue
                        probe_logger(
                            hash_id=hash_id,
                            epoch=epoch,
                            sampling_rate_hz=100,
                            raw_views=generated.probe_views(batch_position),
                            probe_id=probe_id,
                            is_final_epoch=is_final_epoch,
                            metadata={
                                "center": center,
                                "model": spec.name,
                                "method_id": method.profile_name,
                                "method_profile_sha256": method.profile_sha256,
                                "exposure": exposure_name,
                                "composition_index": composition_index,
                            },
                        )

                epoch_samples += batch_size
                epoch_loss_sum += scaled_objective.detach() * batch_size
                for name, raw_loss in objective.raw_terms.items():
                    count = objective.valid_counts[name]
                    epoch_term_sums[name] += raw_loss.detach() * count
                    epoch_weighted_sums[name] += (
                        objective.weighted_terms[name].detach()
                        * family_loss_scale
                        * batch_size
                    )
                    epoch_term_counts[name] += count
                for name, value in generated.bundle.values.items():
                    if isinstance(value, WaveformView):
                        epoch_view_count_sums[name] += value.valid_mask.sum()
                epoch_candidate_eligible += len(
                    generated.candidate_eligible_positions
                )
                epoch_quality_accepted += len(generated.accepted_positions)
                epoch_quality_view_total += generated.quality_view_total_count
                epoch_quality_view_accepted += (
                    generated.quality_view_accepted_count
                )
                epoch_ineligible_hashes.extend(generated.ineligible_hash_ids)
                epoch_quality_rejected.extend(generated.quality_rejected)

                step_diagnostics: dict[str, float] = {}
                for name, value in generated.bundle.diagnostics.items():
                    if isinstance(value, bool) or not isinstance(
                        value, (int, float)
                    ):
                        continue
                    scalar = float(value)
                    if not math.isfinite(scalar):
                        continue
                    step_diagnostics[name] = scalar
                    weight = generated.diagnostic_weights.get(name, batch_size)
                    diagnostic_sums[name] = (
                        diagnostic_sums.get(name, 0.0) + scalar * weight
                    )
                    diagnostic_weights[name] = (
                        diagnostic_weights.get(name, 0) + weight
                    )
                if write_diagnostic_step:
                    _append_jsonl(
                        diagnostics_path,
                        {
                            "kind": "step",
                            "epoch": epoch,
                            "view_execution_step": view_execution_step,
                            "optimizer_step": optimizer_steps,
                            "method_id": method.profile_name,
                            "exposure": exposure_name,
                            "composition_index": composition_index,
                            "candidate_eligible_count": len(
                                generated.candidate_eligible_positions
                            ),
                            "quality_accepted_count": len(
                                generated.accepted_positions
                            ),
                            "quality_view_total_count": (
                                generated.quality_view_total_count
                            ),
                            "quality_view_accepted_count": (
                                generated.quality_view_accepted_count
                            ),
                            "quality_rejected_count": len(
                                generated.quality_rejected
                            ),
                            "ineligible_count": len(
                                generated.ineligible_hash_ids
                            ),
                            "objective_raw": raw_scalars,
                            "objective_weighted": weighted_scalars,
                            "metrics": step_diagnostics,
                        },
                    )
                previous_step_end = time.perf_counter()

            if epoch_samples == 0 or epoch_origin_samples == 0:
                raise ValueError("online train dataloader is empty")
            if optimizer_steps - optimizer_steps_at_epoch_start != epoch_base_batches:
                raise RuntimeError(
                    "matched-base optimizer-step invariant failed: "
                    f"steps={optimizer_steps - optimizer_steps_at_epoch_start}, "
                    f"base_batches={epoch_base_batches}"
                )
            if expected_epoch_hashes is not None:
                for exposure_name, seen_hashes in epoch_seen_hashes.items():
                    if seen_hashes == expected_epoch_hashes:
                        continue
                    missing = sorted(expected_epoch_hashes - seen_hashes)
                    extra = sorted(seen_hashes - expected_epoch_hashes)
                    raise RuntimeError(
                        f"online exposure {exposure_name!r} does not cover the "
                        "complete bound K500 selection: "
                        f"missing={missing[:5]}, extra={extra[:5]}"
                    )
            validation_metrics: dict[str, Any] | None = None
            if epoch_evaluator is not None:
                model.eval()
                evaluated = epoch_evaluator(
                    model,
                    epoch,
                    resolved_device,
                    run_identity,
                )
                if not isinstance(evaluated, Mapping):
                    raise TypeError("epoch_evaluator must return a mapping")
                validation_metrics = dict(evaluated)
            scheduler.step()
            ineligible_unique = tuple(sorted(set(epoch_ineligible_hashes)))
            quality_rejected = [
                {"node_id": node_id, "hash_id": hash_id, "reason": reason}
                for node_id, hash_id, reason in sorted(
                    {
                        (
                            item.get("node_id", ""),
                            item["hash_id"],
                            item["reason"],
                        )
                        for item in epoch_quality_rejected
                    }
                )
            ]
            view_names = tuple(epoch_view_count_sums)
            view_values = (
                torch.stack(
                    tuple(epoch_view_count_sums[name] for name in view_names)
                )
                .detach()
                .cpu()
                .tolist()
            )
            epoch_view_counts = {
                name: int(view_values[index])
                for index, name in enumerate(view_names)
            }
            raw_means = {
                name: (
                    float(epoch_term_sums[name].detach().cpu())
                    / epoch_term_counts[name]
                    if epoch_term_counts[name]
                    else 0.0
                )
                for name in epoch_term_sums
            }
            weighted_means = {
                name: float(value.detach().cpu()) / epoch_origin_samples
                for name, value in epoch_weighted_sums.items()
            }
            diagnostics_mean = {
                name: value / diagnostic_weights[name]
                for name, value in diagnostic_sums.items()
                if diagnostic_weights.get(name, 0) > 0
            }
            performance_mean = {
                name: value / timed_step_count
                for name, value in performance_sums.items()
            }
            performance_mean["timed_step_count"] = timed_step_count
            clean_exposure_count = int(
                epoch_exposure_counts.get(
                    "clean", epoch_exposure_counts.get("base", 0)
                )
            )
            corrupted_exposure_count = int(
                sum(
                    count
                    for name, count in epoch_exposure_counts.items()
                    if name.startswith("corruption_")
                )
            )
            exposure_metrics = {
                "budget_policy": budget_policy,
                "base_record_count": (
                    len(expected_epoch_hashes)
                    if expected_epoch_hashes is not None
                    else max(epoch_exposure_counts.values())
                ),
                "clean_count": clean_exposure_count,
                "corrupted_count": corrupted_exposure_count,
                "total_count": int(sum(epoch_exposure_counts.values())),
                "optimizer_steps_per_base_batch": 1,
                "optimizer_steps_this_epoch": epoch_base_batches,
                "view_executions_per_base_batch": len(exposure_steps),
                "family_loss_weights": (
                    method.contracts.get("family_loss_weights")
                    if fixed20_exposure
                    else None
                ),
                "per_exposure_counts": dict(epoch_exposure_counts),
                "materialized_corruption_cache": False,
            }
            train_metrics: dict[str, Any] = {
                "loss": (
                    float(epoch_loss_sum.detach().cpu()) / epoch_origin_samples
                ),
                "objective_terms": raw_means,
                "weighted_objective_terms": weighted_means,
                "objective_valid_counts": dict(epoch_term_counts),
                "view_valid_counts": dict(epoch_view_counts),
                "sample_count": epoch_origin_samples,
                "view_execution_sample_count": epoch_samples,
                "exposure": exposure_metrics,
            }
            legacy = _legacy_loss_aliases(raw_means, weighted_means)
            train_metrics.update(
                {
                    key: value
                    for key, value in legacy.items()
                    if not key.startswith("weighted_")
                }
            )
            legacy_weighted_names = {
                "weighted_clean": "weighted_clean_contribution",
                "weighted_lhat": "weighted_lhat_contribution",
                "weighted_augmix": "weighted_augmix_contribution",
                "weighted_jsd": "weighted_jsd_contribution",
            }
            for source, target in legacy_weighted_names.items():
                if source in legacy:
                    train_metrics[target] = legacy[source]
            if method.requirements.latent_pool or method.requirements.vae_encoder:
                train_metrics.update(
                    {
                        "eligible_count": epoch_quality_accepted,
                        "candidate_eligible_count": epoch_candidate_eligible,
                        "quality_accepted_count": epoch_quality_accepted,
                        "quality_view_total_count": epoch_quality_view_total,
                        "quality_view_accepted_count": (
                            epoch_quality_view_accepted
                        ),
                        "ineligible_count": len(ineligible_unique),
                        "quality_rejected_count": len(quality_rejected),
                        "candidate_eligible_fraction": (
                            epoch_candidate_eligible / epoch_samples
                        ),
                        "quality_accepted_fraction": (
                            epoch_quality_accepted / epoch_samples
                        ),
                        "quality_view_accepted_fraction": (
                            epoch_quality_view_accepted / epoch_quality_view_total
                            if epoch_quality_view_total
                            else 0.0
                        ),
                        "all_quality_views_accepted_fraction": (
                            epoch_quality_accepted / epoch_samples
                        ),
                    }
                )
                if epoch_quality_view_total == 2 * epoch_samples:
                    train_metrics.update(
                        {
                            "both_views_accepted_record_count": (
                                epoch_quality_accepted
                            ),
                            "both_views_accepted_fraction": (
                                epoch_quality_accepted / epoch_samples
                            ),
                        }
                    )
            epoch_record = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train": train_metrics,
                "validation": validation_metrics,
                "diagnostics": diagnostics_mean,
                "performance": performance_mean,
                "exposure": exposure_metrics,
                "ineligible_hash_ids": list(ineligible_unique),
                "quality_rejected": quality_rejected,
            }
            history.append(epoch_record)
            _append_jsonl(
                diagnostics_path,
                {
                    "kind": "epoch",
                    "epoch": epoch,
                    "method_id": method.profile_name,
                    "objective_terms": raw_means,
                    "weighted_objective_terms": weighted_means,
                    "view_valid_counts": epoch_view_counts,
                    "candidate_eligible_count": epoch_candidate_eligible,
                    "quality_accepted_count": epoch_quality_accepted,
                    "quality_view_total_count": epoch_quality_view_total,
                    "quality_view_accepted_count": epoch_quality_view_accepted,
                    "quality_rejected_count": len(quality_rejected),
                    "ineligible_count": len(ineligible_unique),
                    "ineligible_hash_ids": list(ineligible_unique),
                    "quality_rejected": quality_rejected,
                    "metrics": diagnostics_mean,
                    "performance": performance_mean,
                    "exposure": exposure_metrics,
                },
            )
            if checkpoint_write_policy == "every_epoch" or epoch == epochs:
                checkpoint = {
                    "schema_version": 2,
                    "epoch": epoch,
                    "scientific_arm": method.scientific_arm,
                    "method_id": method.profile_name,
                    "center": center,
                    "optimizer_steps": optimizer_steps,
                    "selection": "last",
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "grad_scaler_state_dict": scaler.state_dict(),
                    "metrics": epoch_record,
                    "run_identity": run_identity,
                }
                temporary = last_checkpoint.with_name(
                    f".{last_checkpoint.name}.tmp"
                )
                torch.save(checkpoint, temporary)
                temporary.replace(last_checkpoint)
            _write_json(
                history_path,
                {
                    "schema_version": 2,
                    "selection": "last",
                    "method_id": method.profile_name,
                    "optimizer_steps": optimizer_steps,
                    "epochs": history,
                },
            )
            monitor.log_epoch(epoch_record, is_selected=(epoch == epochs))

        if not last_checkpoint.is_file():
            raise RuntimeError("online training did not produce a final checkpoint")
        last_checkpoint_sha256 = sha256_file(last_checkpoint)
        result = OnlineTrainingResult(
            model=model,
            output_dir=output,
            method_id=method.profile_name,
            scientific_arm=method.scientific_arm,
            center=center,
            epochs_completed=len(history),
            optimizer_steps=optimizer_steps,
            last_checkpoint_path=last_checkpoint,
            last_checkpoint_sha256=last_checkpoint_sha256,
            history=tuple(history),
            model_identity=run_identity["model"],
            config_identity={
                "training": config.describe(),
                "method": method.describe(),
            },
            seed_identity=seed.describe(),
            latent_pool_identity=pool_identity,
        )
        _write_json(result_path, result.describe())
        return result
    finally:
        if batch_norm_plan is not None:
            batch_norm_plan.restore()
        monitor.close()


__all__ = [
    "ALLOWED_CENTERS",
    "DEFAULT_METHOD_CONFIG_DIR",
    "DEFAULT_ONLINE_CONFIG_PATH",
    "FIXED20_COMPOSITION_ORDER",
    "FIXED20_EXPOSURE_POLICY",
    "FAMILY_BALANCED_BN_POLICY",
    "ONLINE_PARAMETER_NAMES",
    "OnlineTrainConfig",
    "OnlineTrainingResult",
    "load_online_train_config",
    "resolve_online_training_parameters",
    "train_online_model",
]

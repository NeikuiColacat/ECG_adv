"""Finite-recipe online trainer for PN2021 K500.

The trainer owns the optimizer and exposure budget.  Every recipe takes one
outer optimizer step per base batch; recipes only choose the locked views and
losses accumulated into that step.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from core.augmix import (
    AugMixConfig,
    generate_two_chain_augmix_strong_view,
    multilabel_jsd,
)
from core.methods import (
    AuxiliaryVariant,
    BASE_VIEW_NAME,
    RecipeKind,
    RecipeSpec,
    WaveformView,
    build_method_runtime,
    load_recipe_spec,
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
from util.random_seed import make_torch_generator, seed_process
from util.pn2021_artifact_contract import validate_training_lineage

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
        "stage1_steps",
        "stage1_learning_rate",
        "stage1_weight_decay",
        "stage1_gradient_clip_norm",
    }
)
FIXED20_COMPOSITION_ORDER = tuple(range(20))
FAMILY_BALANCED_BN_POLICY = "family_loss_weighted_once_per_base_batch"
OBJECTIVE_VIEW_BALANCED_BN_POLICY = "objective_view_weighted_once_per_base_batch"


@dataclass(frozen=True)
class _ExposureStep:
    name: str
    composition_index: int | None
    objective_terms: tuple[str, ...] | None
    loss_scale: float = 1.0


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
    recipe: RecipeSpec,
    *,
    exposure_steps: Sequence[_ExposureStep],
) -> _BatchNormMomentumPlan | None:
    if recipe.kind in {RecipeKind.FIXED20, RecipeKind.TWO_STAGE_AUGMIX_LHAT}:
        policy = FAMILY_BALANCED_BN_POLICY
        exposure_weights = tuple(
            0.0 if step.name == "auxiliary" else float(step.loss_scale)
            for step in exposure_steps
        )
    elif recipe.kind is RecipeKind.LATENT_THREECHAIN:
        policy = OBJECTIVE_VIEW_BALANCED_BN_POLICY
        exposure_weights = (0.5, 0.25, 0.25)
    elif recipe.kind in {RecipeKind.CLEAN, RecipeKind.RANDOM_DEPTH23}:
        return None
    else:  # pragma: no cover - RecipeKind is closed.
        raise AssertionError(f"unsupported recipe kind: {recipe.kind}")
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


def _method_exposure_steps(
    recipe: RecipeSpec,
    *,
    epoch: int = 1,
) -> tuple[_ExposureStep, ...]:
    """Resolve the code-owned exposure schedule for one finite recipe."""

    if recipe.kind in {
        RecipeKind.CLEAN,
        RecipeKind.RANDOM_DEPTH23,
        RecipeKind.LATENT_THREECHAIN,
    }:
        return (_ExposureStep("base", None, None),)
    if recipe.kind is RecipeKind.FIXED20:
        return (
            _ExposureStep("clean", -1, ("clean_bce",), 0.5),
            *tuple(
                _ExposureStep(
                    f"corruption_{index:02d}",
                    index,
                    ("corrupted_bce",),
                    0.025,
                )
                for index in FIXED20_COMPOSITION_ORDER
            ),
        )
    if recipe.kind is RecipeKind.TWO_STAGE_AUGMIX_LHAT:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
            raise ValueError("rotating-four epoch must be a positive integer")
        slot = (epoch - 1) % 5
        compositions = (
            2 * slot,
            2 * slot + 1,
            10 + 2 * slot,
            10 + 2 * slot + 1,
        )
        steps: tuple[_ExposureStep, ...] = (
            _ExposureStep("clean", -1, ("clean_bce",), 0.5),
            *tuple(
                _ExposureStep(
                    f"corruption_{index:02d}",
                    index,
                    ("corrupted_bce",),
                    0.125,
                )
                for index in compositions
            ),
        )
        if recipe.auxiliary_variant is AuxiliaryVariant.CONTRACTED_LHAT:
            return (*steps, _ExposureStep("auxiliary", None, ("lhat_direct_bce",), 2.0))
        if recipe.auxiliary_variant is AuxiliaryVariant.MATCHED_NO_VAE:
            return steps
        raise ValueError("two-stage recipe requires a finite auxiliary variant")
    raise AssertionError(f"unsupported recipe kind: {recipe.kind}")


def _recipe_family_loss_weights(recipe: RecipeSpec) -> dict[str, float] | None:
    if recipe.kind is RecipeKind.FIXED20:
        return {
            "clean": 0.5,
            "corrupted_total": 0.5,
            "corrupted_per_composition": 0.025,
        }
    if recipe.kind is RecipeKind.TWO_STAGE_AUGMIX_LHAT:
        return {
            "clean": 0.5,
            "corrupted_total": 0.5,
            "corrupted_per_composition": 0.125,
        }
    return None


def _recipe_identity(recipe: RecipeSpec) -> dict[str, Any]:
    return {
        "recipe_id": recipe.recipe_id,
        "kind": recipe.kind.value,
        "auxiliary_variant": recipe.auxiliary_variant.value,
        "schema_version": recipe.schema_version,
        "recipe_version": recipe.recipe_version,
        "implementation_identity": recipe.implementation_identity,
        "recipe_spec_sha256": recipe.recipe_sha256,
    }


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
        for exposure_index, exposure in enumerate(exposure_steps):
            batch = dict(raw_batch)
            batch["__exposure_name"] = exposure.name
            batch["__exposure_group"] = group
            batch["__exposure_index"] = exposure_index
            batch["__composition_index"] = exposure.composition_index
            batch["__objective_terms"] = exposure.objective_terms
            batch["__loss_scale"] = exposure.loss_scale
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
    recipe_identity: dict[str, Any]
    model_identity: dict[str, Any]
    config_identity: dict[str, Any]
    seed_identity: dict[str, Any]
    latent_pool_identity: dict[str, Any] | None
    lineage: dict[str, Any]

    def describe(self) -> dict[str, Any]:
        if len(self.last_checkpoint_sha256) != 64:
            raise ValueError("last_checkpoint_sha256 must contain a SHA256 digest")
        last_checkpoint = {
            "path": str(self.last_checkpoint_path),
            "sha256": self.last_checkpoint_sha256,
        }
        return {
            "schema_version": 1,
            "artifact_type": "pn2021_train_result",
            "lineage": self.lineage,
            "output_dir": str(self.output_dir),
            "method_id": self.method_id,
            "recipe": self.recipe_identity,
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
    """Measure one step or one fixed20 group with a single final sync."""

    _CORE_PHASES = (
        "h2d",
        "augmentation",
        "forward_backward",
        "optimizer",
    )
    _DETAIL_PHASES = (
        "method_generation",
        "candidate_generation",
        "fixed20_generation",
        "raw_chain3_generation",
        "lhat_search",
        "augmix",
        "gradient_combine",
    )
    _PHASES = _CORE_PHASES + _DETAIL_PHASES

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
        self._wall_started = time.perf_counter() - self.data_wait_ms / 1000.0
        self._excluded_host_started: float | None = None
        self._excluded_host_ms = 0.0
        self._cpu_started: dict[str, float] = {}
        self._cpu_elapsed: dict[str, float] = {}
        self._cuda_events: dict[
            str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
        ] = {}

    def add_data_wait(self, milliseconds: float) -> None:
        self.data_wait_ms += max(0.0, float(milliseconds))

    def start_excluded_host(self) -> None:
        """Start host-only observer work excluded from training throughput.

        CUDA phase events remain the source of truth for device work.  This
        interval is only for optional logging/visualization which can otherwise
        make a sampled fixed20 group look compute-bound on the host.
        """

        if self._excluded_host_started is not None:
            raise RuntimeError("excluded host timing is already active")
        self._excluded_host_started = time.perf_counter()

    def stop_excluded_host(self) -> None:
        if self._excluded_host_started is None:
            raise RuntimeError("excluded host timing was not started")
        self._excluded_host_ms += max(
            0.0,
            (time.perf_counter() - self._excluded_host_started) * 1000.0,
        )
        self._excluded_host_started = None

    def start(self, phase: str) -> None:
        if phase not in self._PHASES:
            raise ValueError(f"unsupported timing phase: {phase}")
        if phase in self._cpu_started:
            raise RuntimeError(f"timing phase {phase!r} is already active")
        if self.use_cuda_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self._cuda_events.setdefault(phase, []).append((start, end))
        else:
            self._cpu_started[phase] = time.perf_counter()

    def stop(self, phase: str) -> None:
        if phase not in self._PHASES:
            raise ValueError(f"unsupported timing phase: {phase}")
        if self.use_cuda_events:
            if phase not in self._cuda_events or not self._cuda_events[phase]:
                raise RuntimeError(f"timing phase {phase!r} was not started")
            self._cuda_events[phase][-1][1].record()
        else:
            if phase not in self._cpu_started:
                raise RuntimeError(f"timing phase {phase!r} was not started")
            started = self._cpu_started.pop(phase)
            self._cpu_elapsed[phase] = self._cpu_elapsed.get(phase, 0.0) + max(
                0.0,
                (time.perf_counter() - started) * 1000.0,
            )

    def finish(
        self,
        *,
        batch_size: int,
        grad_norm: float | torch.Tensor,
        view_executions: int = 1,
    ) -> dict[str, float]:
        if view_executions <= 0:
            raise ValueError("view_executions must be positive")
        if self._excluded_host_started is not None:
            raise RuntimeError("cannot finish timing with active excluded host work")
        if self._cpu_started:
            raise RuntimeError(
                "cannot finish timing with active CPU phases: "
                + ", ".join(sorted(self._cpu_started))
            )
        if self.use_cuda_events:
            # All phases use the current training stream. Synchronizing the last
            # event once makes every earlier event elapsed-time query valid.
            last_event = next(
                (
                    self._cuda_events[phase][-1][1]
                    for phase in reversed(self._CORE_PHASES)
                    if self._cuda_events.get(phase)
                ),
                None,
            )
            if last_event is None:
                raise RuntimeError("sampled CUDA timing recorded no core phase")
            last_event.synchronize()
            elapsed = {
                phase: sum(
                    max(0.0, float(start.elapsed_time(end)))
                    for start, end in pairs
                )
                for phase, pairs in self._cuda_events.items()
            }
        else:
            elapsed = dict(self._cpu_elapsed)
        observed_wall_ms = max(
            0.0, (time.perf_counter() - self._wall_started) * 1000.0
        )
        grad_norm_value = (
            float(grad_norm.detach().cpu())
            if isinstance(grad_norm, torch.Tensor)
            else float(grad_norm)
        )
        phase_total_ms = sum(
            (
                self.data_wait_ms,
                elapsed.get("h2d", 0.0),
                elapsed.get("augmentation", 0.0),
                elapsed.get("forward_backward", 0.0),
                elapsed.get("optimizer", 0.0),
            )
        )
        wall_ms = max(
            phase_total_ms,
            observed_wall_ms - self._excluded_host_ms,
        )
        values = {
            "data_wait_ms": self.data_wait_ms / view_executions,
            "h2d_ms": elapsed.get("h2d", 0.0) / view_executions,
            "augmentation_ms": elapsed.get("augmentation", 0.0)
            / view_executions,
            "forward_backward_ms": elapsed.get("forward_backward", 0.0)
            / view_executions,
            "optimizer_ms": elapsed.get("optimizer", 0.0) / view_executions,
        }
        values.update(
            {
                f"{phase}_ms": elapsed.get(phase, 0.0)
                for phase in self._DETAIL_PHASES
                if phase in elapsed
            }
        )
        values["step_ms"] = max(
            1.0e-9,
            sum(values[name] for name in (
                "data_wait_ms",
                "h2d_ms",
                "augmentation_ms",
                "forward_backward_ms",
                "optimizer_ms",
            )),
        )
        values["samples_per_sec"] = float(batch_size) * 1000.0 / values["step_ms"]
        values["grad_norm"] = max(0.0, grad_norm_value)
        if view_executions > 1:
            values.update(
                {
                    "base_group_data_wait_ms": self.data_wait_ms,
                    "base_group_h2d_ms": elapsed.get("h2d", 0.0),
                    "base_group_augmentation_ms": elapsed.get(
                        "augmentation", 0.0
                    ),
                    "base_group_forward_backward_ms": elapsed.get(
                        "forward_backward", 0.0
                    ),
                    "base_group_optimizer_ms": elapsed.get("optimizer", 0.0),
                    "base_group_gpu_plus_data_ms": phase_total_ms,
                    "base_group_wall_ms": wall_ms,
                    "base_group_observed_wall_ms": observed_wall_ms,
                    "base_group_excluded_logging_ms": self._excluded_host_ms,
                    "base_group_host_gap_ms": max(0.0, wall_ms - phase_total_ms),
                    "base_records_per_sec": float(batch_size)
                    * 1000.0
                    / max(1.0e-9, wall_ms),
                    "view_executions": float(view_executions),
                }
            )
        return values


@contextmanager
def _excluded_observer_work(
    timer: _SampledStepTimer | None,
) -> Iterator[None]:
    """Exclude optional host logging from an open fixed20 group timer."""

    if timer is None:
        yield
        return
    timer.start_excluded_host()
    try:
        yield
    finally:
        timer.stop_excluded_host()


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
        scheduler_horizon = profile.get("scheduler_horizon_epochs", epochs)
        if (
            isinstance(scheduler_horizon, bool)
            or not isinstance(scheduler_horizon, int)
            or scheduler_horizon < epochs
        ):
            raise ValueError(
                f"{model_name}.scheduler_horizon_epochs must be >= epochs"
            )
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"{model_name}.batch_size must be positive")
        _positive_number(profile.get("learning_rate"), f"{model_name}.learning_rate")
        _nonnegative_number(profile.get("weight_decay"), f"{model_name}.weight_decay")
        _positive_number(
            profile.get("minimum_learning_rate_ratio"),
            f"{model_name}.minimum_learning_rate_ratio",
        )
        stage1_keys = {
            "stage1_steps",
            "stage1_learning_rate",
            "stage1_weight_decay",
            "stage1_gradient_clip_norm",
        }
        present_stage1_keys = stage1_keys.intersection(profile)
        if present_stage1_keys and present_stage1_keys != stage1_keys:
            raise ValueError(
                f"{model_name} Stage-1 parameters must be declared together"
            )
        if present_stage1_keys:
            stage1_steps = profile["stage1_steps"]
            if (
                isinstance(stage1_steps, bool)
                or not isinstance(stage1_steps, int)
                or stage1_steps <= 0
            ):
                raise ValueError(f"{model_name}.stage1_steps must be positive")
            _positive_number(
                profile["stage1_learning_rate"],
                f"{model_name}.stage1_learning_rate",
            )
            _nonnegative_number(
                profile["stage1_weight_decay"],
                f"{model_name}.stage1_weight_decay",
            )
            _positive_number(
                profile["stage1_gradient_clip_norm"],
                f"{model_name}.stage1_gradient_clip_norm",
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
    if performance_timing.get("scope") not in {
        "sampled_core_step_excludes_logging",
        "fixed20_base_group_excludes_logging_otherwise_sampled_step",
    }:
        raise ValueError(
            "performance_timing.scope must disclose sampled-step or fixed20 "
            "base-group timing and logging exclusion"
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
            profile.get(
                "scheduler_horizon_epochs",
                supplied.get("epochs", profile["epochs"]),
            ),
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
        "stage1_steps": supplied.get(
            "stage1_steps", profile.get("stage1_steps", 0)
        ),
        "stage1_learning_rate": supplied.get(
            "stage1_learning_rate", profile.get("stage1_learning_rate", 0.0)
        ),
        "stage1_weight_decay": supplied.get(
            "stage1_weight_decay", profile.get("stage1_weight_decay", 0.0)
        ),
        "stage1_gradient_clip_norm": supplied.get(
            "stage1_gradient_clip_norm",
            profile.get("stage1_gradient_clip_norm", 0.0),
        ),
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
    stage1_steps = resolved["stage1_steps"]
    if (
        isinstance(stage1_steps, bool)
        or not isinstance(stage1_steps, int)
        or stage1_steps < 0
    ):
        raise ValueError("resolved stage1_steps must be a nonnegative integer")
    if stage1_steps > 0:
        for key in ("stage1_learning_rate", "stage1_gradient_clip_norm"):
            resolved[key] = _positive_number(resolved[key], f"resolved {key}")
        resolved["stage1_weight_decay"] = _nonnegative_number(
            resolved["stage1_weight_decay"], "resolved stage1_weight_decay"
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


def _method_rng_identity(
    *,
    comparison_group: str,
    replicate_id: int,
    center: str,
    model_name: str,
    epoch: int,
    batch_hash_sha256: str,
    exposure_name: str,
    composition_index: int | None,
) -> tuple[str, ...]:
    """Derive method RNG from semantic stream identity, not execution order."""

    stream = (
        f"composition_index={composition_index}"
        if composition_index is not None
        else f"exposure={exposure_name}"
    )
    return (
        str(comparison_group),
        str(replicate_id),
        str(center),
        str(model_name),
        f"epoch={epoch}",
        f"batch_hash_sha256={batch_hash_sha256}",
        stream,
    )


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


def _training_lineage(
    *,
    model_identity: Mapping[str, Any],
    spec: ModelSpec,
    center: str,
    recipe: RecipeSpec,
    config: OnlineTrainConfig,
    seed_identity: Mapping[str, Any],
    train_dataloader: Any,
) -> dict[str, Any]:
    selection = getattr(getattr(train_dataloader, "dataset", None), "selection", None)
    describe = getattr(selection, "describe", None)
    if not callable(describe):
        raise ValueError("managed online dataloader must expose its selection identity")
    adaptation = describe()
    if not isinstance(adaptation, Mapping):
        raise ValueError("managed selection identity must be a mapping")
    source_key = (
        "checkpoint_identity"
        if spec.name == EFFICIENTNET1DV2_SPEC.name
        else "task_checkpoint_identity"
    )
    source = model_identity.get(source_key)
    if not isinstance(source, Mapping):
        raise ValueError("online model must expose its source checkpoint identity")
    random_seed = config.payload["random_seed"]
    return validate_training_lineage(
        {
            "schema_version": 1,
            "scope": "pn2021_k500_center_adaptation",
            "model": {"name": spec.name, "spec": spec.describe()},
            "center": center,
            "method": {
                "recipe_id": recipe.recipe_id,
                "scientific_arm": recipe.scientific_arm,
                "recipe_spec_sha256": recipe.recipe_sha256,
                "implementation_identity": recipe.implementation_identity,
                "recipe_version": recipe.recipe_version,
                "kind": recipe.kind.value,
                "auxiliary_variant": recipe.auxiliary_variant.value,
                "schema_version": recipe.schema_version,
            },
            "comparison": {
                "group": random_seed["comparison_group"],
                "replicate_id": random_seed["replicate_id"],
            },
            "seed": {
                key: seed_identity[key]
                for key in ("base_seed", "effective_seed", "namespace", "config_sha256")
            },
            "source_checkpoint": {"sha256": source["sha256"]},
            "training_config_sha256": config.sha256,
            "adaptation_data": {
                key: adaptation[key]
                for key in (
                    "dataset",
                    "partition",
                    "logical_center",
                    "source_centers",
                    "record_count",
                    "split_id",
                    "hash_id_set_sha256",
                    "split_manifest_sha256",
                    "source_manifest_sha256",
                    "mapping_version",
                    "mapping_hash",
                    "class_order",
                )
            },
            "selection": {
                "policy": "last",
                "heldout_evaluation_used_for_selection": False,
            },
        }
    )


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


def _backward_objective(
    objective: _ObjectiveBatch,
    *,
    loss_scale: float,
    scaler: Any,
    empty_lhat_auxiliary: bool,
) -> torch.Tensor:
    """Backpropagate one exposure, treating an empty LHAT view as an empty sum."""

    scaled = objective.total * loss_scale
    if empty_lhat_auxiliary:
        if objective.valid_counts != {"lhat_direct_bce": 0}:
            raise RuntimeError("empty LHAT auxiliary has an inconsistent objective")
        return scaled
    if not scaled.requires_grad:
        raise RuntimeError("non-empty online objective is detached from the model")
    if scaler.is_enabled():
        scaler.scale(scaled).backward()
    else:
        scaled.backward()
    return scaled


def _compute_objective(
    *,
    recipe: RecipeSpec,
    bundle: Any,
    model: nn.Module,
    spec: ModelSpec,
    normalization_epsilon: float,
    pos_weight: torch.Tensor | None,
    objective_term_names: Sequence[str] | None = None,
    batch_norm_plan: _BatchNormMomentumPlan | None = None,
) -> _ObjectiveBatch:
    """Compute recipe losses with one mean contribution per origin.

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
        known_names = {term.name for term in recipe.objective.terms}
        unknown_names = sorted(set(selected_names) - known_names)
        if unknown_names:
            raise ValueError(f"unknown objective term names: {unknown_names}")
    selected_terms = tuple(
        term
        for term in recipe.objective.terms
        if selected_names is None or term.name in selected_names
    )

    ordered_views: list[str] = []
    for term in selected_terms:
        for name in term.views:
            if name not in ordered_views:
                ordered_views.append(name)
    if not ordered_views:
        raise ValueError("recipe objective must contain at least one term")

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
            raise RuntimeError(f"unsupported recipe objective kind: {term.kind}")
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


def _materialize_finite_scalar_mapping(
    values: Mapping[str, float | int | torch.Tensor],
) -> dict[str, float]:
    """Transfer detached device scalars in one synchronization.

    Method runtimes may keep diagnostic reductions on-device throughout the
    fixed20 hot path.  Host conversion is deferred to a sampled logging boundary
    or epoch end instead of forcing several CUDA synchronizations per view.
    """

    resolved: dict[str, float] = {}
    tensor_names: list[str] = []
    tensors: list[torch.Tensor] = []
    target_device: torch.device | None = None
    for name, value in values.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            target_device = value.device if target_device is None else target_device
            tensor_names.append(name)
            tensors.append(value.detach().reshape(()))
            continue
        if isinstance(value, (int, float)):
            scalar = float(value)
            if math.isfinite(scalar):
                resolved[name] = scalar
    if tensors:
        assert target_device is not None
        host_values = (
            torch.stack(
                tuple(
                    value.to(device=target_device, dtype=torch.float32)
                    for value in tensors
                )
            )
            .cpu()
            .tolist()
        )
        for name, value in zip(tensor_names, host_values, strict=True):
            scalar = float(value)
            if math.isfinite(scalar):
                resolved[name] = scalar
    return resolved


_DIAGNOSTIC_RATE_COMPONENTS = (
    (
        "decoded_anchor_sample_anyflip_asr",
        "sample_anyflip_success",
        "sample_anyflip_eligible",
        "sample_anyflip_numerator",
        "sample_anyflip_denominator",
    ),
    (
        "decoded_anchor_positive_hide_asr",
        "positive_hide_numerator",
        "positive_hide_denominator",
        "positive_hide_numerator",
        "positive_hide_denominator",
    ),
    (
        "decoded_anchor_negative_add_asr",
        "negative_add_numerator",
        "negative_add_denominator",
        "negative_add_numerator",
        "negative_add_denominator",
    ),
)
_DIAGNOSTIC_RATE_SAMPLE_SUFFIXES = frozenset(
    suffix
    for _, numerator, denominator, _, _ in _DIAGNOSTIC_RATE_COMPONENTS
    for suffix in (numerator, denominator)
)


def _diagnostic_sample_summary(
    chunks: Mapping[str, Sequence[torch.Tensor]],
) -> tuple[
    dict[str, dict[str, float | int]],
    dict[str, float],
    dict[str, dict[str, float | None]],
]:
    """Compute exact epoch statistics with one device-to-host transfer."""

    samples: dict[str, torch.Tensor] = {}
    for name in sorted(chunks):
        values = tuple(chunks[name])
        if not values:
            continue
        combined = torch.cat(values, dim=0).detach().to(dtype=torch.float32)
        if combined.ndim != 1:
            raise ValueError("diagnostic sample chunks must concatenate to rank 1")
        if combined.numel() > 0:
            samples[name] = combined
    if not samples:
        return {}, {}, {}

    devices = {value.device for value in samples.values()}
    if len(devices) != 1:
        raise ValueError("diagnostic sample chunks must share one device")
    stat_names = tuple(samples)
    stat_tensors: list[torch.Tensor] = []
    counts: dict[str, int] = {}
    for name in stat_names:
        values = samples[name]
        counts[name] = int(values.numel())
        quantiles = torch.quantile(
            values,
            torch.tensor((0.5, 0.9), device=values.device, dtype=values.dtype),
        )
        stat_tensors.extend(
            (
                torch.isfinite(values).sum().to(dtype=torch.float32),
                values.sum(),
                values.mean(),
                quantiles[0],
                quantiles[1],
            )
        )
    host_values = torch.stack(stat_tensors).detach().cpu().tolist()
    distributions: dict[str, dict[str, float | int]] = {}
    flat: dict[str, float] = {}
    offset = 0
    for name in stat_names:
        finite_count, total, mean, median, p90 = (
            float(value) for value in host_values[offset : offset + 5]
        )
        offset += 5
        count = counts[name]
        if int(finite_count) != count or not all(
            math.isfinite(value) for value in (total, mean, median, p90)
        ):
            raise FloatingPointError(
                f"diagnostic sample {name!r} contains NaN or Inf"
            )
        distributions[name] = {
            "count": count,
            "sum": total,
            "mean": mean,
            "median": median,
            "p90": p90,
        }
        suffix = name.rsplit("/", 1)[-1]
        if suffix not in _DIAGNOSTIC_RATE_SAMPLE_SUFFIXES:
            flat[f"{name}_count"] = float(count)
            flat[f"{name}_mean"] = mean
            flat[f"{name}_median"] = median
            flat[f"{name}_p90"] = p90

    rates: dict[str, dict[str, float | None]] = {}
    for (
        rate_suffix,
        numerator_suffix,
        denominator_suffix,
        aggregate_numerator_suffix,
        aggregate_denominator_suffix,
    ) in _DIAGNOSTIC_RATE_COMPONENTS:
        for numerator_name in tuple(distributions):
            if numerator_name.rsplit("/", 1)[-1] != numerator_suffix:
                continue
            prefix = (
                numerator_name.rsplit("/", 1)[0]
                if "/" in numerator_name
                else ""
            )
            denominator_name = (
                f"{prefix}/{denominator_suffix}"
                if prefix
                else denominator_suffix
            )
            if denominator_name not in distributions:
                raise ValueError(
                    f"diagnostic rate {rate_suffix!r} lacks {denominator_name!r}"
                )
            numerator = float(distributions[numerator_name]["sum"])
            denominator = float(distributions[denominator_name]["sum"])
            if numerator < 0.0 or denominator < 0.0 or numerator > denominator:
                raise ValueError(
                    f"diagnostic rate {rate_suffix!r} has invalid counts"
                )
            rate = numerator / denominator if denominator > 0.0 else None
            rate_name = f"{prefix}/{rate_suffix}" if prefix else rate_suffix
            rates[rate_name] = {
                "numerator": numerator,
                "denominator": denominator,
                "rate": rate,
            }
            aggregate_numerator_name = (
                f"{prefix}/{aggregate_numerator_suffix}"
                if prefix
                else aggregate_numerator_suffix
            )
            aggregate_denominator_name = (
                f"{prefix}/{aggregate_denominator_suffix}"
                if prefix
                else aggregate_denominator_suffix
            )
            flat[aggregate_numerator_name] = numerator
            flat[aggregate_denominator_name] = denominator
            if rate is not None:
                flat[rate_name] = rate
    return distributions, flat, rates


def _update_framed_sha256(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, byteorder="little", signed=False))
    digest.update(payload)


def _stochastic_trace_summary(
    *,
    input_identity_sha256: str,
    input_record_count: int,
    chunks: Mapping[str, Sequence[torch.Tensor]],
) -> dict[str, Any]:
    """Hash compact realized RNG/outcome tensors at the epoch boundary."""

    if len(input_identity_sha256) != 64:
        raise ValueError("stochastic input identity must be a SHA256 hex digest")
    if input_record_count < 0:
        raise ValueError("stochastic input record count must be non-negative")
    tensor_records: dict[str, dict[str, Any]] = {}
    for name in sorted(chunks):
        values = tuple(chunks[name])
        if not values:
            continue
        trailing_shapes = {tuple(value.shape[1:]) for value in values}
        dtypes = {value.dtype for value in values}
        if len(trailing_shapes) != 1 or len(dtypes) != 1:
            raise ValueError(
                f"stochastic trace {name!r} changed shape or dtype within an epoch"
            )
        combined = torch.cat(values, dim=0).detach().contiguous().cpu()
        if combined.dtype not in {
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.float32,
            torch.float64,
        }:
            raise TypeError(
                f"stochastic trace {name!r} uses unsupported dtype {combined.dtype}"
            )
        header = {
            "schema_version": 1,
            "name": name,
            "dtype": str(combined.dtype),
            "shape": list(combined.shape),
        }
        tensor_digest = hashlib.sha256()
        _update_framed_sha256(
            tensor_digest,
            json.dumps(
                header, sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
        )
        _update_framed_sha256(tensor_digest, combined.numpy().tobytes(order="C"))
        tensor_records[name] = {
            "dtype": header["dtype"],
            "shape": header["shape"],
            "sha256": tensor_digest.hexdigest(),
        }
    aggregate_payload = {
        "schema_version": 1,
        "input_identity_sha256": input_identity_sha256,
        "input_record_count": input_record_count,
        "tensors": tensor_records,
    }
    aggregate_sha256 = hashlib.sha256(
        json.dumps(
            aggregate_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        **aggregate_payload,
        "aggregate_sha256": aggregate_sha256,
    }


def _feature_width(model: nn.Module, spec: ModelSpec) -> int:
    if spec is EFFICIENTNET1DV2_SPEC:
        classifier = getattr(model, "classifier", None)
        if not isinstance(classifier, nn.Sequential) or len(classifier) != 4:
            raise TypeError("EfficientNet Stage-1 requires the canonical classifier")
        head = classifier[3]
    elif spec is ECGFOUNDER_SPEC:
        head = getattr(model, "dense", None)
    else:
        raise ValueError(f"unsupported Stage-1 model spec: {spec.name}")
    if not isinstance(head, nn.Linear):
        raise TypeError("Stage-1 classifier head must be nn.Linear")
    return int(head.in_features)


def _head_parameters(
    model: nn.Module,
    spec: ModelSpec,
) -> tuple[nn.Parameter, ...]:
    if spec is EFFICIENTNET1DV2_SPEC:
        head = getattr(model, "classifier")[3]
    elif spec is ECGFOUNDER_SPEC:
        head = getattr(model, "dense")
    else:
        raise ValueError(f"unsupported Stage-1 model spec: {spec.name}")
    if not isinstance(head, nn.Linear):
        raise TypeError("Stage-1 classifier head must be nn.Linear")
    return tuple(head.parameters())


def _forward_logits_and_features(
    model: nn.Module,
    model_input: torch.Tensor,
    spec: ModelSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    forward_features = getattr(model, "forward_features", None)
    if not callable(forward_features):
        raise TypeError(f"{spec.name} must expose forward_features for Stage-1")
    features = forward_features(model_input)
    if features.ndim != 2 or features.shape[0] != model_input.shape[0]:
        raise ValueError("forward_features must return shape (B,D)")
    if spec is EFFICIENTNET1DV2_SPEC:
        classifier = getattr(model, "classifier")
        logits = classifier[3](classifier[2](features))
    elif spec is ECGFOUNDER_SPEC:
        logits = getattr(model, "dense")(features)
    else:
        raise ValueError(f"unsupported Stage-1 model spec: {spec.name}")
    return (
        validate_model_output(
            logits,
            spec,
            batch_size=int(model_input.shape[0]),
            check_finite=False,
        ),
        features,
    )


def _simclr_nt_xent(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("SimCLR projections must share shape (B,D)")
    if first.shape[0] < 2:
        raise ValueError("SimCLR requires at least two records per batch")
    features = F.normalize(torch.cat((first, second), dim=0).float(), dim=1)
    similarities = features @ features.transpose(0, 1)
    similarities = similarities / float(temperature)
    similarities.fill_diagonal_(-torch.inf)
    batch = int(first.shape[0])
    positives = torch.cat(
        (
            torch.arange(batch, 2 * batch, device=features.device),
            torch.arange(0, batch, device=features.device),
        )
    )
    rows = torch.arange(2 * batch, device=features.device)
    loss = (
        torch.logsumexp(similarities, dim=1)
        - similarities[rows, positives]
    ).mean()
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("Stage-1 SimCLR loss became NaN or Inf")
    return loss


def _weighted_logit_anchor_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    class_weights: Sequence[float],
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError("logit anchor expects aligned (B,5) logits")
    elements = F.binary_cross_entropy_with_logits(
        student_logits.float(),
        torch.sigmoid(teacher_logits.float()),
        reduction="none",
    )
    weights = torch.as_tensor(
        class_weights,
        device=elements.device,
        dtype=elements.dtype,
    )
    if tuple(weights.shape) != (5,) or float(weights.sum()) <= 0.0:
        raise ValueError("logit anchor must declare five positive-mass class weights")
    return (elements * weights.view(1, 5)).sum(dim=1).div(
        weights.sum()
    ).mean()


def _cache_k500_logits(
    model: nn.Module,
    train_dataloader: Any,
    *,
    spec: ModelSpec,
    device: torch.device,
    normalization_epsilon: float,
) -> dict[str, torch.Tensor]:
    """Cache frozen-teacher inference on the bound K500 records only."""

    was_training = model.training
    model.eval()
    logits_by_hash: dict[str, torch.Tensor] = {}
    try:
        with torch.no_grad():
            for batch in train_dataloader:
                if not isinstance(batch, Mapping):
                    raise TypeError("teacher-cache batches must be mappings")
                raw = batch.get("waveform")
                if not isinstance(raw, torch.Tensor):
                    raise TypeError("teacher-cache waveform must be a tensor")
                hashes = _batch_hashes(batch, int(raw.shape[0]))
                raw = raw.to(device, dtype=torch.float32, non_blocking=True)
                model_input = prepare_canonical_model_input(
                    raw,
                    spec,
                    epsilon=float(normalization_epsilon),
                )
                logits = validate_model_output(
                    model(model_input),
                    spec,
                    batch_size=int(raw.shape[0]),
                    check_finite=True,
                ).detach().cpu()
                for index, hash_id in enumerate(hashes):
                    if hash_id in logits_by_hash:
                        raise RuntimeError(
                            f"teacher cache repeated K500 hash {hash_id!r}"
                        )
                    logits_by_hash[hash_id] = logits[index].contiguous()
    finally:
        model.train(was_training)
    selected_hashes = _loader_selection_hashes(train_dataloader)
    if selected_hashes is not None:
        if set(logits_by_hash) != set(selected_hashes):
            raise RuntimeError("frozen-teacher cache differs from loader selection")
    elif not logits_by_hash:
        raise RuntimeError("frozen-teacher cache is empty")
    return logits_by_hash


def _teacher_logits_for_hashes(
    cache: Mapping[str, torch.Tensor],
    hashes: Sequence[str],
    *,
    device: torch.device,
) -> torch.Tensor:
    try:
        values = [cache[str(hash_id)] for hash_id in hashes]
    except KeyError as exc:
        raise RuntimeError(
            f"teacher cache is missing K500 hash {exc.args[0]!r}"
        ) from None
    return torch.stack(values).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )


@contextmanager
def _preserve_batch_norm_buffers(model: nn.Module) -> Iterator[None]:
    states: list[
        tuple[
            nn.Module,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
        ]
    ] = []
    for module in model.modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        states.append(
            (
                module,
                None if module.running_mean is None else module.running_mean.clone(),
                None if module.running_var is None else module.running_var.clone(),
                None
                if module.num_batches_tracked is None
                else module.num_batches_tracked.clone(),
            )
        )
    try:
        yield
    finally:
        with torch.no_grad():
            for module, mean, variance, tracked in states:
                if mean is not None:
                    module.running_mean.copy_(mean)
                if variance is not None:
                    module.running_var.copy_(variance)
                if tracked is not None:
                    module.num_batches_tracked.copy_(tracked)


@contextmanager
def _preserve_torch_rng(device: torch.device) -> Iterator[None]:
    cpu_state = torch.random.get_rng_state()
    cuda_state = (
        torch.cuda.get_rng_state(device)
        if device.type == "cuda"
        else None
    )
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)


def _run_augmix_simclr_stage1(
    model: nn.Module,
    train_dataloader: Any,
    *,
    spec: ModelSpec,
    device: torch.device,
    recipe: RecipeSpec,
    augmix_config: AugMixConfig,
    center: str,
    base_seed: int,
    resolved: Mapping[str, Any],
    normalization_epsilon: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    """Run the frozen K500-only AugMix-SimCLR representation stage."""

    if recipe.kind is not RecipeKind.TWO_STAGE_AUGMIX_LHAT:
        raise ValueError("Stage-1 belongs only to the two-stage recipe")
    temperature = augmix_config.stage1_simclr_temperature
    teacher_cache = _cache_k500_logits(
        model,
        train_dataloader,
        spec=spec,
        device=device,
        normalization_epsilon=normalization_epsilon,
    )
    feature_width = _feature_width(model, spec)
    projector = nn.Sequential(
        nn.Linear(feature_width, feature_width),
        nn.ReLU(inplace=True),
        nn.Linear(feature_width, min(128, feature_width)),
    ).to(device)
    head_states = tuple(
        (parameter, bool(parameter.requires_grad))
        for parameter in _head_parameters(model, spec)
    )
    for parameter, _ in head_states:
        parameter.requires_grad_(False)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("Stage-1 has no trainable backbone parameters")
    stage1_steps = int(resolved["stage1_steps"])
    optimizer = AdamW(
        [*trainable, *projector.parameters()],
        lr=float(resolved["stage1_learning_rate"]),
        weight_decay=float(resolved["stage1_weight_decay"]),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=stage1_steps,
        eta_min=float(resolved["stage1_learning_rate"]) * 0.01,
    )
    losses = torch.zeros(3, device=device, dtype=torch.float64)
    iterator = iter(train_dataloader)
    model.train()
    try:
        for step in range(stage1_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_dataloader)
                batch = next(iterator)
            if not isinstance(batch, Mapping):
                raise TypeError("Stage-1 batches must be mappings")
            raw = batch.get("waveform")
            if not isinstance(raw, torch.Tensor):
                raise TypeError("Stage-1 waveform must be a tensor")
            hashes = _batch_hashes(batch, int(raw.shape[0]))
            raw = raw.to(device, dtype=torch.float32, non_blocking=True)
            generator = make_torch_generator(
                device,
                augmix_config.random_namespace,
                recipe.comparison_rng_identity,
                center,
                spec.name,
                f"base_seed={base_seed}",
                f"stage1_step={step}",
                config_path=augmix_config.random_seed_config_path,
            )
            strong = generate_two_chain_augmix_strong_view(
                raw,
                sampling_rate_hz=100,
                config=augmix_config,
                generator=generator,
            ).mixed_raw
            clean_input = prepare_canonical_model_input(
                raw,
                spec,
                epsilon=normalization_epsilon,
            )
            strong_input = prepare_canonical_model_input(
                strong,
                spec,
                epsilon=normalization_epsilon,
            )
            teacher_logits = _teacher_logits_for_hashes(
                teacher_cache, hashes, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(
                enabled=amp_enabled,
                dtype=amp_dtype,
            ):
                clean_logits, clean_features = _forward_logits_and_features(
                    model, clean_input, spec
                )
                _, strong_features = _forward_logits_and_features(
                    model, strong_input, spec
                )
                simclr = _simclr_nt_xent(
                    projector(clean_features),
                    projector(strong_features),
                    temperature=temperature,
                )
                anchor = _weighted_logit_anchor_loss(
                    clean_logits,
                    teacher_logits,
                    (1.0, 1.0, 1.0, 1.0, 1.0),
                )
                total = simclr + 5.0 * anchor
            if not bool(torch.isfinite(total).item()):
                raise FloatingPointError("Stage-1 loss became NaN or Inf")
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                [*trainable, *projector.parameters()],
                float(resolved["stage1_gradient_clip_norm"]),
            )
            optimizer.step()
            scheduler.step()
            losses += torch.stack(
                (total.detach(), simclr.detach(), anchor.detach())
            ).to(dtype=torch.float64)
    finally:
        for parameter, requires_grad in head_states:
            parameter.requires_grad_(requires_grad)
    means = (losses / float(stage1_steps)).detach().cpu().tolist()
    return {
        "schema_version": 1,
        "steps": stage1_steps,
        "optimizer": "adamw",
        "learning_rate": float(resolved["stage1_learning_rate"]),
        "weight_decay": float(resolved["stage1_weight_decay"]),
        "gradient_clip_norm": float(resolved["stage1_gradient_clip_norm"]),
        "simclr_temperature": temperature,
        "augmix_internal_chains": augmix_config.stage1_width,
        "augmix_dirichlet_alpha": augmix_config.stage1_dirichlet_alpha,
        "augmix_beta_alpha": augmix_config.stage1_beta_alpha,
        "logit_anchor_weight": 5.0,
        "teacher_scope": "frozen_source_checkpoint_inference_on_bound_k500",
        "ptbxl_replay_weight": 0.0,
        "mean_total_loss": float(means[0]),
        "mean_simclr_loss": float(means[1]),
        "mean_logit_anchor_loss": float(means[2]),
    }


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
) -> OnlineTrainingResult:
    """Train one file-backed finite recipe under its locked K500 budget.

    The ``matched_base`` budget performs one optimizer step per base batch.
    Fixed-20 expands the views used to form that objective but accumulates all
    gradients before the one step.
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
            "recipe config must belong to the selected config bundle"
        ) from None
    recipe = load_recipe_spec(method_path)
    exposure_steps = _method_exposure_steps(recipe)
    objective_term_weights = {
        term.name: float(term.weight) for term in recipe.objective.terms
    }
    grouped_exposure = recipe.kind in {
        RecipeKind.FIXED20,
        RecipeKind.TWO_STAGE_AUGMIX_LHAT,
    }
    auxiliary_exposure = (
        recipe.auxiliary_variant is AuxiliaryVariant.CONTRACTED_LHAT
    )
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
    if train_partition not in {None, "k500"}:
        raise ValueError(
            f"online training accepts only k500, got {train_partition!r}"
        )

    requires_latent = bool(recipe.requirements.latent_pool)
    requires_encoder = bool(recipe.requirements.vae_encoder)
    requires_decoder = bool(recipe.requirements.vae_decoder)
    if requires_latent != (latent_pool is not None):
        raise ValueError(
            "latent_pool presence must exactly match the recipe requirement"
        )
    if requires_decoder != (decoder is not None):
        raise ValueError(
            "VAE decoder presence must exactly match the recipe requirement"
        )
    if requires_encoder != (encoder is not None):
        raise ValueError(
            "VAE encoder presence must exactly match the recipe requirement"
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
            method_id=recipe.recipe_id,
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
        recipe,
        exposure_steps=exposure_steps,
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
        recipe,
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
    method_resource_identity["composition_index_hint_supported"] = True
    method_resource_identity["performance_timer_hook_supported"] = False
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
    staged_method = recipe.kind is RecipeKind.TWO_STAGE_AUGMIX_LHAT
    stage1_summary: dict[str, Any] | None = None
    stage2_teacher_cache: dict[str, torch.Tensor] | None = None
    if staged_method:
        if int(resolved["stage1_steps"]) <= 0:
            raise ValueError("AugMix-SimCLR mainline requires Stage-1 steps")
        if runtime.augmix_config is None:
            raise RuntimeError("two-stage recipe lacks its resolved AugMix config")
        stage1_summary = _run_augmix_simclr_stage1(
            model,
            train_dataloader,
            spec=spec,
            device=resolved_device,
            recipe=recipe,
            augmix_config=runtime.augmix_config,
            center=center,
            base_seed=seed.base_seed,
            resolved=resolved,
            normalization_epsilon=epsilon,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        stage1_checkpoint = checkpoint_dir / "stage1.pt"
        temporary_stage1 = stage1_checkpoint.with_name(
            f".{stage1_checkpoint.name}.tmp"
        )
        torch.save(
            {
                "schema_version": 1,
                "stage": "augmix_simclr",
                "method_id": recipe.recipe_id,
                "recipe": _recipe_identity(recipe),
                "center": center,
                "model_state_dict": model.state_dict(),
                "summary": stage1_summary,
            },
            temporary_stage1,
        )
        temporary_stage1.replace(stage1_checkpoint)
        stage1_summary["checkpoint"] = {
            "path": str(stage1_checkpoint),
            "sha256": sha256_file(stage1_checkpoint),
        }
        stage2_teacher_cache = _cache_k500_logits(
            model,
            train_dataloader,
            spec=spec,
            device=resolved_device,
            normalization_epsilon=epsilon,
        )
        method_resource_identity["stage1"] = stage1_summary
        method_resource_identity["stage2_teacher"] = {
            "scope": "frozen_post_stage1_pre_stage2_logits_on_bound_k500",
            "record_count": len(stage2_teacher_cache),
        }
        _write_json(method_resources_path, method_resource_identity)
    model_identity = _model_identity(model, spec)
    lineage = _training_lineage(
        model_identity=model_identity,
        spec=spec,
        center=center,
        recipe=recipe,
        config=config,
        seed_identity=seed.describe(),
        train_dataloader=train_dataloader,
    )
    run_identity = {
        "config": config.describe(),
        "method": method_resource_identity,
        "recipe": _recipe_identity(recipe),
        "seed": seed.describe(),
        "center": center,
        "scientific_arm": recipe.scientific_arm,
        "model": model_identity,
        "vae_encoder_checkpoint": encoder_identity,
        "vae_decoder_checkpoint": decoder_identity,
        "training_parameters": {
            "resolved": resolved,
            "explicit_overrides": supplied,
        },
        "stage1": stage1_summary,
        "pos_weight": (
            None
            if resolved_pos_weight is None
            else resolved_pos_weight.detach().cpu().tolist()
        ),
        "exposure_plan": {
            "budget_policy": budget_policy,
            "optimizer_steps_per_base_batch": 1,
            "view_executions_per_base_batch": len(exposure_steps),
            "five_epoch_rotation_cycle": (
                [
                    {
                        "epoch_modulo_five_slot": epoch,
                        "composition_indices": [
                            step.composition_index
                            for step in _method_exposure_steps(recipe, epoch=epoch)
                            if step.name.startswith("corruption_")
                        ],
                    }
                    for epoch in range(1, 6)
                ]
                if recipe.kind is RecipeKind.TWO_STAGE_AUGMIX_LHAT
                else None
            ),
            "family_loss_weights": _recipe_family_loss_weights(recipe),
            "steps": [
                {
                    "name": exposure.name,
                    "composition_index": exposure.composition_index,
                    "objective_terms": (
                        None
                        if exposure.objective_terms is None
                        else list(exposure.objective_terms)
                    ),
                    "loss_scale": exposure.loss_scale,
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
            "profile_namespace": recipe.rng_namespace,
            "comparison_rng_identity": recipe.comparison_rng_identity,
            "execution_identity": [
                "comparison_group",
                "replicate_id",
                "center",
                "model",
                "epoch",
                "ordered_batch_hash_sha256",
                "composition_index_or_exposure_stream",
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
    }

    history: list[dict[str, Any]] = []
    view_execution_step = 0
    optimizer_steps = 0
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_exposure_steps = _method_exposure_steps(recipe, epoch=epoch)
            optimizer_steps_at_epoch_start = optimizer_steps
            epoch_base_batches = 0
            epoch_loss_sum = torch.zeros((), device=resolved_device)
            epoch_samples = 0
            epoch_origin_samples = 0
            epoch_term_sums = {
                term.name: torch.zeros((), device=resolved_device)
                for term in recipe.objective.terms
            }
            epoch_weighted_sums = {
                term.name: torch.zeros((), device=resolved_device)
                for term in recipe.objective.terms
            }
            epoch_effective_loss_mass_sums = {
                term.name: 0.0 for term in recipe.objective.terms
            }
            epoch_term_counts = {
                term.name: 0 for term in recipe.objective.terms
            }
            epoch_view_count_sums = {
                name: torch.zeros((), device=resolved_device, dtype=torch.int64)
                for name in recipe.output_names
            }
            epoch_candidate_eligible = 0
            epoch_quality_accepted = 0
            epoch_quality_view_total = 0
            epoch_quality_view_accepted = 0
            epoch_ineligible_hashes: list[str] = []
            epoch_quality_rejected: list[dict[str, str]] = []
            diagnostic_sums: dict[str, float | torch.Tensor] = {}
            diagnostic_weights: dict[str, int] = {}
            diagnostic_sample_chunks: dict[str, list[torch.Tensor]] = {}
            stochastic_trace_chunks: dict[str, list[torch.Tensor]] = {}
            stochastic_input_digest = hashlib.sha256()
            stochastic_input_record_count = 0
            performance_sums: dict[str, float] = {}
            timed_step_count = 0
            epoch_seen_hashes: dict[str, set[str]] = {
                exposure.name: set() for exposure in epoch_exposure_steps
            }
            epoch_exposure_counts: dict[str, int] = {
                exposure.name: 0 for exposure in epoch_exposure_steps
            }
            learning_rate = float(optimizer.param_groups[0]["lr"])
            previous_step_end = time.perf_counter()
            cached_exposure_group: int | None = None
            cached_raw: torch.Tensor | None = None
            cached_targets: torch.Tensor | None = None
            fixed20_group_timer: _SampledStepTimer | None = None
            fixed20_group_finite: torch.Tensor | None = None
            fixed20_group_view_executions = 0

            for batch in _iter_exposure_batches(
                train_dataloader, epoch_exposure_steps
            ):
                batch_received = time.perf_counter()
                data_wait_ms = max(
                    0.0, (batch_received - previous_step_end) * 1000.0
                )
                next_execution_step = view_execution_step + 1
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
                exposure_index = int(batch.get("__exposure_index", 0))
                group_start = not grouped_exposure or exposure_index == 0
                group_end = (
                    not grouped_exposure
                    or exposure_index == len(epoch_exposure_steps) - 1
                )
                if grouped_exposure and group_start:
                    if fixed20_group_finite is not None:
                        raise RuntimeError(
                            "grouped finite-loss batch started before the prior "
                            "group ended"
                        )
                    fixed20_group_finite = torch.ones(
                        (), device=resolved_device, dtype=torch.bool
                    )
                if grouped_exposure:
                    if group_start:
                        if fixed20_group_timer is not None:
                            raise RuntimeError(
                                "grouped timing batch started before the prior group ended"
                            )
                        sample_group = timing_enabled and (
                            (optimizer_steps + 1) % timing_interval == 0
                        )
                        fixed20_group_timer = (
                            _SampledStepTimer(
                                device=resolved_device,
                                data_wait_ms=data_wait_ms,
                                cuda_events=bool(timing_config["cuda_events"]),
                            )
                            if sample_group
                            else None
                        )
                        fixed20_group_view_executions = 0
                    elif fixed20_group_timer is not None:
                        fixed20_group_timer.add_data_wait(data_wait_ms)
                    timer = fixed20_group_timer
                    if timer is not None:
                        fixed20_group_view_executions += 1
                else:
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
                if group_start:
                    optimizer.zero_grad(set_to_none=True)
                    epoch_origin_samples += batch_size
                    epoch_base_batches += 1
                family_loss_scale = float(batch.get("__loss_scale", 1.0))
                if batch_norm_plan is not None and grouped_exposure:
                    batch_norm_plan.apply(exposure_index)
                hash_digest = hashlib.sha256(
                    "\n".join(hashes).encode("utf-8")
                ).hexdigest()
                composition_indices = (
                    None
                    if composition_index is None
                    else torch.full(
                        (batch_size,),
                        composition_index,
                        device=resolved_device,
                        dtype=torch.int64,
                    )
                )
                rng_identity = _method_rng_identity(
                    comparison_group=str(random_seed["comparison_group"]),
                    replicate_id=int(random_seed["replicate_id"]),
                    center=center,
                    model_name=spec.name,
                    epoch=epoch,
                    batch_hash_sha256=hash_digest,
                    exposure_name=exposure_name,
                    composition_index=composition_index,
                )
                trace_input_payload = json.dumps(
                    {
                        "schema_version": 1,
                        "base_seed": seed.base_seed,
                        "comparison_rng_identity": recipe.comparison_rng_identity,
                        "rng_identity": list(rng_identity),
                        "composition_index": composition_index,
                        "exposure_loss_scale": family_loss_scale,
                        "ordered_hash_ids": list(hashes),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                _update_framed_sha256(
                    stochastic_input_digest,
                    trace_input_payload,
                )
                stochastic_input_record_count += batch_size
                generation_phase = (
                    "method_generation"
                    if not grouped_exposure
                    else (
                        "candidate_generation"
                        if group_start or exposure_name == "auxiliary"
                        else "fixed20_generation"
                    )
                )
                if timer is not None:
                    timer.start(generation_phase)
                try:
                    generated = runtime.generate(
                        clean_raw=raw,
                        targets=targets,
                        hash_ids=hashes,
                        classifier=model,
                        base_seed=seed.base_seed,
                        rng_identity=rng_identity,
                        composition_indices=composition_indices,
                        composition_index_hint=composition_index,
                        objective_term_names=objective_terms,
                    )
                finally:
                    if timer is not None:
                        timer.stop(generation_phase)
                if timer is not None:
                    timer.stop("augmentation")
                    timer.start("forward_backward")

                with ExitStack() as auxiliary_state:
                    if staged_method and exposure_name == "auxiliary":
                        auxiliary_state.enter_context(
                            _preserve_batch_norm_buffers(model)
                        )
                        auxiliary_state.enter_context(
                            _preserve_torch_rng(resolved_device)
                        )
                    with torch.cuda.amp.autocast(
                        enabled=amp_enabled, dtype=amp_dtype
                    ):
                        objective = _compute_objective(
                            recipe=recipe,
                            bundle=generated.bundle,
                            model=model,
                            spec=spec,
                            normalization_epsilon=epsilon,
                            pos_weight=resolved_pos_weight,
                            objective_term_names=objective_terms,
                            batch_norm_plan=(
                                None if grouped_exposure else batch_norm_plan
                            ),
                        )
                        if staged_method and exposure_name == "clean":
                            if stage2_teacher_cache is None:
                                raise RuntimeError(
                                    "Stage-2 teacher cache is unavailable"
                                )
                            teacher_logits = _teacher_logits_for_hashes(
                                stage2_teacher_cache,
                                hashes,
                                device=resolved_device,
                            )
                            clean_input = prepare_canonical_model_input(
                                raw,
                                spec,
                                epsilon=epsilon,
                            )
                            # Keep the preservation contexts alive through
                            # backward. BatchNorm saves its running buffers for
                            # gradient computation, so restoring them before
                            # backward increments their version counter and
                            # invalidates the clean logit-anchor graph.
                            auxiliary_state.enter_context(
                                _preserve_batch_norm_buffers(model)
                            )
                            auxiliary_state.enter_context(
                                _preserve_torch_rng(resolved_device)
                            )
                            student_logits = validate_model_output(
                                model(clean_input),
                                spec,
                                batch_size=batch_size,
                                check_finite=False,
                            )
                            anchor_loss = _weighted_logit_anchor_loss(
                                student_logits,
                                teacher_logits,
                                (1.0, 1.0, 1.0, 1.0, 1.0),
                            )
                            anchor_weight = {
                                EFFICIENTNET1DV2_SPEC.name: 2.0,
                                ECGFOUNDER_SPEC.name: 0.5,
                            }[spec.name]
                            objective = _ObjectiveBatch(
                                total=objective.total
                                + anchor_loss
                                * anchor_weight
                                / family_loss_scale,
                                raw_terms=objective.raw_terms,
                                weighted_terms=objective.weighted_terms,
                                valid_counts=objective.valid_counts,
                            )
                    objective_finite = torch.isfinite(objective.total)
                    if grouped_exposure:
                        if fixed20_group_finite is None:
                            raise RuntimeError(
                                "grouped finite-loss check lacks an active batch"
                            )
                        fixed20_group_finite &= objective_finite.detach()
                        if group_end and not bool(fixed20_group_finite.item()):
                            raise FloatingPointError(
                                "online grouped loss became NaN or Inf"
                            )
                    elif not bool(objective_finite.item()):
                        raise FloatingPointError(
                            "online training loss became NaN or Inf"
                        )
                    scaled_objective = _backward_objective(
                        objective,
                        loss_scale=family_loss_scale,
                        scaler=scaler,
                        empty_lhat_auxiliary=(
                            staged_method
                            and exposure_name == "auxiliary"
                            and objective.valid_counts == {"lhat_direct_bce": 0}
                        ),
                    )
                if timer is not None:
                    timer.stop("forward_backward")

                grad_norm = torch.zeros((), device=resolved_device)
                if group_end:
                    if timer is not None:
                        timer.start("optimizer")
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            trainable, float(resolved["gradient_clip_norm"])
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            trainable, float(resolved["gradient_clip_norm"])
                        )
                        optimizer.step()
                    optimizer_steps += 1
                    if timer is not None:
                        timer.stop("optimizer")

                view_execution_step += 1
                finish_group_timer = grouped_exposure and group_end
                performance = None
                if timer is not None and (
                    not grouped_exposure or finish_group_timer
                ):
                    performance = timer.finish(
                        batch_size=batch_size,
                        grad_norm=grad_norm,
                        view_executions=(
                            fixed20_group_view_executions
                            if grouped_exposure
                            else 1
                        ),
                    )
                if finish_group_timer:
                    fixed20_group_timer = None
                    fixed20_group_finite = None
                    fixed20_group_view_executions = 0
                if performance is not None:
                    timed_step_count += 1
                    for name, value in performance.items():
                        performance_sums[name] = (
                            performance_sums.get(name, 0.0) + float(value)
                        )
                    _append_jsonl(
                        diagnostics_path,
                        {
                            "kind": (
                                "performance_base_group"
                                if grouped_exposure
                                else "performance_step"
                            ),
                            "epoch": epoch,
                            "view_execution_step": view_execution_step,
                            "optimizer_step": optimizer_steps,
                            "metrics": performance,
                        },
                    )

                write_diagnostic_step = (
                    view_execution_step % diagnostics_interval == 0
                )
                raw_scalars: dict[str, float] = {}
                weighted_scalars: dict[str, float] = {}
                observer_timer = (
                    timer
                    if grouped_exposure and timer is not None and not group_end
                    else None
                )
                with _excluded_observer_work(observer_timer):
                    if write_diagnostic_step:
                        _, raw_scalars, weighted_scalars = _objective_host_scalars(
                            objective
                        )
                        weighted_scalars = {
                            name: value * family_loss_scale
                            for name, value in weighted_scalars.items()
                        }

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
                    epoch_effective_loss_mass_sums[name] += (
                        family_loss_scale * objective_term_weights[name] * count
                    )
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
                for name, values in generated.diagnostic_samples.items():
                    if values.numel() > 0:
                        diagnostic_sample_chunks.setdefault(name, []).append(
                            values.detach()
                        )
                for name, values in generated.stochastic_trace.items():
                    stochastic_trace_chunks.setdefault(name, []).append(
                        values.detach()
                    )

                step_diagnostic_values: dict[
                    str, float | int | torch.Tensor
                ] = {}
                for name, value in generated.bundle.diagnostics.items():
                    if isinstance(value, bool):
                        continue
                    if isinstance(value, torch.Tensor):
                        if value.numel() != 1:
                            continue
                        scalar_value: float | torch.Tensor = value.detach().to(
                            device=resolved_device,
                            dtype=torch.float32,
                        ).reshape(())
                    elif isinstance(value, (int, float)):
                        scalar_value = float(value)
                        if not math.isfinite(scalar_value):
                            continue
                    else:
                        continue
                    weight = generated.diagnostic_weights.get(name, batch_size)
                    previous = diagnostic_sums.get(name)
                    weighted_value = scalar_value * weight
                    if previous is None:
                        diagnostic_sums[name] = weighted_value
                    elif isinstance(previous, torch.Tensor) or isinstance(
                        weighted_value, torch.Tensor
                    ):
                        previous_tensor = (
                            previous
                            if isinstance(previous, torch.Tensor)
                            else torch.tensor(
                                previous,
                                device=resolved_device,
                                dtype=torch.float32,
                            )
                        )
                        weighted_tensor = (
                            weighted_value
                            if isinstance(weighted_value, torch.Tensor)
                            else torch.tensor(
                                weighted_value,
                                device=resolved_device,
                                dtype=torch.float32,
                            )
                        )
                        diagnostic_sums[name] = previous_tensor + weighted_tensor
                    else:
                        diagnostic_sums[name] = previous + weighted_value
                    diagnostic_weights[name] = (
                        diagnostic_weights.get(name, 0) + weight
                    )
                    step_diagnostic_values[name] = scalar_value
                with _excluded_observer_work(observer_timer):
                    step_diagnostics = (
                        _materialize_finite_scalar_mapping(step_diagnostic_values)
                        if write_diagnostic_step
                        else {}
                    )
                    if write_diagnostic_step:
                        _append_jsonl(
                            diagnostics_path,
                            {
                                "kind": "step",
                                "epoch": epoch,
                                "view_execution_step": view_execution_step,
                                "optimizer_step": optimizer_steps,
                                "method_id": recipe.recipe_id,
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

            if fixed20_group_timer is not None or fixed20_group_finite is not None:
                raise RuntimeError("fixed20 timing/finite group did not terminate")
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
            effective_loss_masses = {
                name: value / epoch_origin_samples
                for name, value in epoch_effective_loss_mass_sums.items()
            }
            diagnostic_mean_values = {
                name: value / diagnostic_weights[name]
                for name, value in diagnostic_sums.items()
                if diagnostic_weights.get(name, 0) > 0
            }
            diagnostics_mean = _materialize_finite_scalar_mapping(
                diagnostic_mean_values
            )
            (
                diagnostic_distributions,
                diagnostic_distribution_scalars,
                diagnostic_rates,
            ) = _diagnostic_sample_summary(diagnostic_sample_chunks)
            diagnostics_mean.update(diagnostic_distribution_scalars)
            stochastic_trace = _stochastic_trace_summary(
                input_identity_sha256=stochastic_input_digest.hexdigest(),
                input_record_count=stochastic_input_record_count,
                chunks=stochastic_trace_chunks,
            )
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
                "family_loss_weights": _recipe_family_loss_weights(recipe),
                "per_exposure_counts": dict(epoch_exposure_counts),
                "materialized_corruption_cache": False,
            }
            if auxiliary_exposure:
                auxiliary_names = ("lhat_direct_bce",)
                nominal_auxiliary_mass = 2.0
                effective_auxiliary_mass = sum(
                    effective_loss_masses[name] for name in auxiliary_names
                )
                exposure_metrics.update(
                    {
                        "auxiliary_objective_terms": list(auxiliary_names),
                        "auxiliary_nominal_loss_mass": nominal_auxiliary_mass,
                        "auxiliary_effective_loss_mass": effective_auxiliary_mass,
                        "auxiliary_effective_fraction_of_nominal": (
                            effective_auxiliary_mass / nominal_auxiliary_mass
                        ),
                    }
                )
            train_metrics: dict[str, Any] = {
                "loss": (
                    float(epoch_loss_sum.detach().cpu()) / epoch_origin_samples
                ),
                "objective_terms": raw_means,
                "weighted_objective_terms": weighted_means,
                "objective_effective_loss_mass": effective_loss_masses,
                "objective_valid_counts": dict(epoch_term_counts),
                "view_valid_counts": dict(epoch_view_counts),
                "sample_count": epoch_origin_samples,
                "view_execution_sample_count": epoch_samples,
                "exposure": exposure_metrics,
            }
            if recipe.requirements.latent_pool or recipe.requirements.vae_encoder:
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
                            epoch_candidate_eligible / epoch_origin_samples
                        ),
                        "quality_accepted_fraction": (
                            epoch_quality_accepted / epoch_origin_samples
                        ),
                        "quality_view_accepted_fraction": (
                            epoch_quality_view_accepted / epoch_quality_view_total
                            if epoch_quality_view_total
                            else 0.0
                        ),
                        "all_quality_views_accepted_fraction": (
                            epoch_quality_accepted / epoch_origin_samples
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
                                epoch_quality_accepted / epoch_origin_samples
                            ),
                        }
                    )
            epoch_record = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train": train_metrics,
                "diagnostics": diagnostics_mean,
                "diagnostic_distributions": diagnostic_distributions,
                "diagnostic_rates": diagnostic_rates,
                "stochastic_trace": stochastic_trace,
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
                    "method_id": recipe.recipe_id,
                    "objective_terms": raw_means,
                    "weighted_objective_terms": weighted_means,
                    "objective_effective_loss_mass": effective_loss_masses,
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
                    "diagnostic_distributions": diagnostic_distributions,
                    "diagnostic_rates": diagnostic_rates,
                    "stochastic_trace": stochastic_trace,
                    "performance": performance_mean,
                    "exposure": exposure_metrics,
                },
            )
            if checkpoint_write_policy == "every_epoch" or epoch == epochs:
                checkpoint = {
                    "schema_version": 3,
                    "lineage": lineage,
                    "epoch": epoch,
                    "scientific_arm": recipe.scientific_arm,
                    "method_id": recipe.recipe_id,
                    "recipe": _recipe_identity(recipe),
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
                    "schema_version": 3,
                    "lineage": lineage,
                    "selection": "last",
                    "method_id": recipe.recipe_id,
                    "recipe": _recipe_identity(recipe),
                    "optimizer_steps": optimizer_steps,
                    "epochs": history,
                },
            )
        if not last_checkpoint.is_file():
            raise RuntimeError("online training did not produce a final checkpoint")
        last_checkpoint_sha256 = sha256_file(last_checkpoint)
        result = OnlineTrainingResult(
            model=model,
            output_dir=output,
            method_id=recipe.recipe_id,
            scientific_arm=recipe.scientific_arm,
            center=center,
            epochs_completed=len(history),
            optimizer_steps=optimizer_steps,
            last_checkpoint_path=last_checkpoint,
            last_checkpoint_sha256=last_checkpoint_sha256,
            history=tuple(history),
            recipe_identity=_recipe_identity(recipe),
            model_identity=run_identity["model"],
            config_identity={
                "training": config.describe(),
                "method": recipe.describe(),
            },
            seed_identity=seed.describe(),
            latent_pool_identity=pool_identity,
            lineage=lineage,
        )
        _write_json(result_path, result.describe())
        return result
    finally:
        if batch_norm_plan is not None:
            batch_norm_plan.restore()


__all__ = [
    "ALLOWED_CENTERS",
    "DEFAULT_METHOD_CONFIG_DIR",
    "DEFAULT_ONLINE_CONFIG_PATH",
    "FIXED20_COMPOSITION_ORDER",
    "FAMILY_BALANCED_BN_POLICY",
    "ONLINE_PARAMETER_NAMES",
    "OnlineTrainConfig",
    "OnlineTrainingResult",
    "load_online_train_config",
    "resolve_online_training_parameters",
    "train_online_model",
]

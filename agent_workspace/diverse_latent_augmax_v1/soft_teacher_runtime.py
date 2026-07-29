"""Frozen-initializer teacher targets for allowlisted LHAT/AugMix auxiliaries.

Legacy Fixed20 methods snapshot the source classifier on first objective use.
J4 methods snapshot the Stage2 initializer before the first runtime generation
branch. Hard-label BCE terms remain unchanged; only each method's explicitly
allowlisted auxiliary BCE receives a convex blend of hard labels and frozen
teacher probabilities.
"""

from __future__ import annotations

import copy
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.methods import BASE_VIEW_NAME, CompiledMethod, WaveformView
from models.contracts import ModelSpec, validate_model_output
from models.input_adapter import prepare_canonical_model_input


AUGMIX_METHOD_ID = "fixed20_pcgrad_teacher04_lhat_augmix_v1"
DIRECT_AUGMIX_METHOD_ID = "fixed20_pcgrad_teacher04_lhat_direct_augmix_v1"
J4_AUGMIX_METHOD_ID = "vae_lhat_chain3_teacher04_polish_v1"
J4_CONTROL_METHOD_ID = "clean_chain3_teacher04_polish_control_v1"
VAE_LHAT_TEACHER_DISTILL_METHOD_ID = (
    "vae_lhat_teacher_distill_polish_v1"
)
CLEAN_TEACHER_DISTILL_CONTROL_METHOD_ID = (
    "clean_teacher_distill_polish_control_v1"
)
METHOD_SOFT_BCE_CALLS = {
    AUGMIX_METHOD_ID: 1,
    DIRECT_AUGMIX_METHOD_ID: 2,
    J4_AUGMIX_METHOD_ID: 1,
    J4_CONTROL_METHOD_ID: 1,
    VAE_LHAT_TEACHER_DISTILL_METHOD_ID: 1,
    CLEAN_TEACHER_DISTILL_CONTROL_METHOD_ID: 1,
}
METHOD_IDS = frozenset(METHOD_SOFT_BCE_CALLS)
LEGACY_METHOD_IDS = frozenset({AUGMIX_METHOD_ID, DIRECT_AUGMIX_METHOD_ID})
J4_METHOD_IDS = frozenset({J4_AUGMIX_METHOD_ID, J4_CONTROL_METHOD_ID})
DISTILL_METHOD_IDS = frozenset(
    {
        VAE_LHAT_TEACHER_DISTILL_METHOD_ID,
        CLEAN_TEACHER_DISTILL_CONTROL_METHOD_ID,
    }
)
STAGE2_METHOD_IDS = J4_METHOD_IDS | DISTILL_METHOD_IDS
TEACHER_MIX = 0.40
DISTILL_TEACHER_MIX = 1.0
TEACHER_POLICY = "frozen_source_snapshot_before_first_optimizer_step"
J4_TEACHER_POLICY = (
    "frozen_stage2_initializer_snapshot_before_first_optimizer_step"
)
TARGET_POLICY = "teacher_mix_times_source_probability_plus_hard_remainder"
J4_TARGET_POLICY = (
    "teacher_mix_times_stage2_initializer_probability_plus_hard_remainder"
)
J4_TEACHER_SCOPE = "augmix_bce_only"
DISTILL_TARGET_POLICY = "frozen_stage2_initializer_probability_only"
J4_INITIALIZER_SHA_CONTRACT = "auxiliary_teacher_initializer_sha256"

_J4_EXPECTED_OBJECTIVES = {
    J4_AUGMIX_METHOD_ID: (
        ("clean_bce", "bce", ("clean_view",), 1.0),
        ("clean_preserve_bce", "bce", ("clean_view",), 0.5),
        ("augmix_bce", "bce", ("augmix_view",), 0.25),
        (
            "clean_lhat_augmix_jsd",
            "bernoulli_jsd",
            ("clean_view", "lhat_view", "augmix_view"),
            0.25,
        ),
    ),
    J4_CONTROL_METHOD_ID: (
        ("clean_bce", "bce", ("clean_view",), 1.0),
        ("clean_preserve_bce", "bce", ("clean_view",), 0.5),
        ("augmix_bce", "bce", ("augmix_view",), 0.25),
        (
            "clean_control_augmix_jsd",
            "bernoulli_jsd",
            ("clean_view", "chain3_control_view", "augmix_view"),
            0.25,
        ),
    ),
}
_DISTILL_EXPECTED_OBJECTIVES = {
    VAE_LHAT_TEACHER_DISTILL_METHOD_ID: (
        ("clean_bce", "bce", ("clean_view",), 1.0),
        ("clean_preserve_bce", "bce", ("clean_view",), 0.5),
        (
            "lhat_teacher_distill_bce",
            "bce",
            ("lhat_view",),
            0.5,
        ),
    ),
    CLEAN_TEACHER_DISTILL_CONTROL_METHOD_ID: (
        ("clean_bce", "bce", ("clean_view",), 1.0),
        ("clean_preserve_bce", "bce", ("clean_view",), 0.5),
        (
            "clean_teacher_distill_control_bce",
            "bce",
            ("lhat_control_view",),
            0.5,
        ),
    ),
}


@dataclass
class _TeacherState:
    teacher: nn.Module | None = None
    method_id: str | None = None
    teacher_policy: str | None = None
    target_policy: str | None = None
    teacher_scope: str | None = None
    teacher_mix: float | None = None
    initializer_sha256: str | None = None
    source_parameter_count: int = 0
    generate_calls: int = 0
    objective_calls: int = 0
    soft_bce_calls: int = 0
    target_shift_abs_means: list[float] = field(default_factory=list)
    target_minima: list[float] = field(default_factory=list)
    target_maxima: list[float] = field(default_factory=list)

    def clear(self) -> None:
        self.teacher = None
        self.method_id = None
        self.teacher_policy = None
        self.target_policy = None
        self.teacher_scope = None
        self.teacher_mix = None
        self.initializer_sha256 = None
        self.source_parameter_count = 0
        self.generate_calls = 0
        self.objective_calls = 0
        self.soft_bce_calls = 0
        self.target_shift_abs_means.clear()
        self.target_minima.clear()
        self.target_maxima.clear()


_STATE = _TeacherState()


def _same_float(value: Any, expected: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1.0e-12)
    )


def _is_teacher_method(method: Any) -> bool:
    return isinstance(method, CompiledMethod) and (
        method.profile_name in METHOD_IDS
        or "auxiliary_label_policy" in method.contracts
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_j4_objective(method: CompiledMethod) -> None:
    expected = _J4_EXPECTED_OBJECTIVES[method.profile_name]
    actual = tuple(
        (term.name, term.kind, tuple(term.views), float(term.weight))
        for term in method.objective.terms
    )
    if actual != expected:
        raise ValueError(
            f"{method.profile_name} objective/order drifted: "
            f"expected {expected!r}, got {actual!r}"
        )


def _validate_distill_objective(method: CompiledMethod) -> None:
    auxiliary_weight = method.contracts.get("auxiliary_effective_weight")
    if (
        isinstance(auxiliary_weight, bool)
        or not isinstance(auxiliary_weight, (int, float))
        or not math.isfinite(float(auxiliary_weight))
        or not 0.0 < float(auxiliary_weight) < 0.5
    ):
        raise ValueError(
            f"{method.profile_name} auxiliary_effective_weight must lie in "
            "(0,0.5)"
        )
    auxiliary_weight = float(auxiliary_weight)
    auxiliary_name = (
        "lhat_teacher_distill_bce"
        if method.profile_name == VAE_LHAT_TEACHER_DISTILL_METHOD_ID
        else "clean_teacher_distill_control_bce"
    )
    auxiliary_view = (
        "lhat_view"
        if method.profile_name == VAE_LHAT_TEACHER_DISTILL_METHOD_ID
        else "lhat_control_view"
    )
    expected = (
        ("clean_bce", "bce", ("clean_view",), 1.0),
        (
            "clean_preserve_bce",
            "bce",
            ("clean_view",),
            1.0 - 2.0 * auxiliary_weight,
        ),
        (
            auxiliary_name,
            "bce",
            (auxiliary_view,),
            2.0 * auxiliary_weight,
        ),
    )
    actual = tuple(
        (term.name, term.kind, tuple(term.views), float(term.weight))
        for term in method.objective.terms
    )
    if actual != expected:
        raise ValueError(
            f"{method.profile_name} objective/order drifted: "
            f"expected {expected!r}, got {actual!r}"
        )


def validate_teacher_method(method: CompiledMethod) -> int:
    if method.profile_name not in METHOD_IDS:
        raise ValueError(
            f"soft-teacher method is not allowlisted: {method.profile_name!r}"
        )
    contracts = method.contracts
    if method.profile_name in J4_METHOD_IDS:
        expected = {
            "auxiliary_label_policy": J4_TARGET_POLICY,
            "auxiliary_teacher_policy": J4_TEACHER_POLICY,
            "auxiliary_teacher_mix": TEACHER_MIX,
            "auxiliary_teacher_scope": J4_TEACHER_SCOPE,
            "non_augmix_bce_hard_labels_preserved": True,
        }
    elif method.profile_name in DISTILL_METHOD_IDS:
        expected = {
            "auxiliary_label_policy": DISTILL_TARGET_POLICY,
            "auxiliary_teacher_policy": J4_TEACHER_POLICY,
            "auxiliary_teacher_mix": DISTILL_TEACHER_MIX,
            "auxiliary_teacher_scope": (
                "lhat_distill_bce_only"
                if method.profile_name == VAE_LHAT_TEACHER_DISTILL_METHOD_ID
                else "clean_identity_distill_control_bce_only"
            ),
            "auxiliary_soft_bce_pos_weight": None,
            "clean_hard_label_bce_preserved": True,
        }
    else:
        expected = {
            "auxiliary_label_policy": TARGET_POLICY,
            "auxiliary_teacher_policy": TEACHER_POLICY,
            "auxiliary_teacher_mix": TEACHER_MIX,
            "auxiliary_teacher_scope": "lhat_and_augmix_bce_only",
            "fixed20_hard_labels_preserved": True,
        }
    for name, value in expected.items():
        actual = contracts.get(name)
        valid = _same_float(actual, value) if isinstance(value, float) else actual == value
        if not valid:
            raise ValueError(
                f"{method.profile_name} has invalid teacher contract {name}: "
                f"expected {value!r}, got {actual!r}"
            )
    if method.profile_name in STAGE2_METHOD_IDS:
        initializer_sha256 = contracts.get(J4_INITIALIZER_SHA_CONTRACT)
        if not _is_sha256(initializer_sha256):
            raise ValueError(
                f"{method.profile_name} contract {J4_INITIALIZER_SHA_CONTRACT} "
                "must be a lowercase 64-character SHA256"
            )
        if method.profile_name in J4_METHOD_IDS:
            _validate_j4_objective(method)
        else:
            _validate_distill_objective(method)
    return METHOD_SOFT_BCE_CALLS[method.profile_name]


def _identity_sha256(identity: Any) -> str | None:
    value = (
        identity.get("sha256")
        if isinstance(identity, Mapping)
        else getattr(identity, "sha256", None)
    )
    return str(value) if _is_sha256(value) else None


def _model_initializer_sha256(model: nn.Module) -> str:
    owners: list[nn.Module] = [model]
    seen = {id(model)}
    cursor = model
    while True:
        child = getattr(cursor, "module", None)
        if not isinstance(child, nn.Module) or id(child) in seen:
            child = getattr(cursor, "_orig_mod", None)
        if not isinstance(child, nn.Module) or id(child) in seen:
            break
        owners.append(child)
        seen.add(id(child))
        cursor = child
    for attribute in ("task_checkpoint_identity", "checkpoint_identity"):
        for owner in reversed(owners):
            resolved = _identity_sha256(getattr(owner, attribute, None))
            if resolved is not None:
                return resolved
    raise ValueError(
        "J4 frozen teacher requires a strict task/checkpoint initializer identity"
    )


def _teacher_metadata(method: CompiledMethod) -> tuple[str, str, str, str | None]:
    if method.profile_name in J4_METHOD_IDS:
        initializer_sha256 = str(
            method.contracts[J4_INITIALIZER_SHA_CONTRACT]
        )
        return (
            J4_TEACHER_POLICY,
            J4_TARGET_POLICY,
            J4_TEACHER_SCOPE,
            initializer_sha256,
        )
    if method.profile_name in DISTILL_METHOD_IDS:
        initializer_sha256 = str(
            method.contracts[J4_INITIALIZER_SHA_CONTRACT]
        )
        return (
            J4_TEACHER_POLICY,
            DISTILL_TARGET_POLICY,
            str(method.contracts["auxiliary_teacher_scope"]),
            initializer_sha256,
        )
    return (
        TEACHER_POLICY,
        TARGET_POLICY,
        "lhat_and_augmix_bce_only",
        None,
    )


def _teacher_mix(method: CompiledMethod) -> float:
    return (
        DISTILL_TEACHER_MIX
        if method.profile_name in DISTILL_METHOD_IDS
        else TEACHER_MIX
    )


def _frozen_teacher(
    model: nn.Module,
    method: CompiledMethod,
    *,
    require_preexisting: bool = False,
) -> nn.Module:
    validate_teacher_method(method)
    teacher_policy, target_policy, teacher_scope, initializer_sha256 = (
        _teacher_metadata(method)
    )
    teacher_mix = _teacher_mix(method)
    if method.profile_name in STAGE2_METHOD_IDS:
        actual_initializer_sha256 = _model_initializer_sha256(model)
        if actual_initializer_sha256 != initializer_sha256:
            raise ValueError(
                f"{method.profile_name} Stage2 initializer SHA256 drifted: "
                f"expected {initializer_sha256}, got {actual_initializer_sha256}"
            )
    if _STATE.teacher is None:
        if require_preexisting:
            raise RuntimeError(
                f"{method.profile_name} teacher was not frozen before objective "
                "evaluation"
            )
        teacher = copy.deepcopy(model)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        if teacher.training or any(
            parameter.requires_grad for parameter in teacher.parameters()
        ):
            raise RuntimeError("failed to freeze Stage2 initializer teacher")
        _STATE.teacher = teacher
        _STATE.method_id = method.profile_name
        _STATE.teacher_policy = teacher_policy
        _STATE.target_policy = target_policy
        _STATE.teacher_scope = teacher_scope
        _STATE.teacher_mix = teacher_mix
        _STATE.initializer_sha256 = initializer_sha256
        _STATE.source_parameter_count = sum(
            int(parameter.numel()) for parameter in teacher.parameters()
        )
    elif (
        _STATE.method_id != method.profile_name
        or _STATE.teacher_policy != teacher_policy
        or _STATE.target_policy != target_policy
        or _STATE.teacher_scope != teacher_scope
        or _STATE.teacher_mix != teacher_mix
        or _STATE.initializer_sha256 != initializer_sha256
    ):
        raise RuntimeError("soft-teacher process state crossed method/initializer scope")
    return _STATE.teacher


def _teacher_targets(
    *,
    bundle: Any,
    model: nn.Module,
    method: CompiledMethod,
    spec: ModelSpec,
    normalization_epsilon: float,
) -> torch.Tensor:
    clean = bundle.require(BASE_VIEW_NAME)
    if not isinstance(clean, WaveformView):
        raise TypeError("soft-teacher objective requires a clean WaveformView")
    teacher = _frozen_teacher(
        model,
        method,
        require_preexisting=method.profile_name in STAGE2_METHOD_IDS,
    )
    if teacher.training or any(
        parameter.requires_grad for parameter in teacher.parameters()
    ):
        raise RuntimeError("soft teacher must remain frozen in eval mode")
    with torch.inference_mode():
        teacher_input = prepare_canonical_model_input(
            clean.waveform,
            spec,
            epsilon=float(normalization_epsilon),
        )
        logits = validate_model_output(
            teacher(teacher_input),
            spec,
            batch_size=clean.batch_size,
            check_finite=True,
        )
        probabilities = torch.sigmoid(logits.float())
        hard = clean.labels.float()
        teacher_mix = _teacher_mix(method)
        targets = (
            teacher_mix * probabilities + (1.0 - teacher_mix) * hard
        )
    _STATE.target_shift_abs_means.append(
        float((targets - hard).abs().mean().detach().cpu())
    )
    _STATE.target_minima.append(float(targets.min().detach().cpu()))
    _STATE.target_maxima.append(float(targets.max().detach().cpu()))
    return targets.detach()


def diagnostics_payload() -> dict[str, Any] | None:
    if _STATE.teacher is None:
        return None
    target_count = len(_STATE.target_shift_abs_means)
    return {
        "schema_version": 1,
        "method_id": _STATE.method_id,
        "teacher_policy": _STATE.teacher_policy,
        "target_policy": _STATE.target_policy,
        "teacher_scope": _STATE.teacher_scope,
        "teacher_mix": _STATE.teacher_mix,
        "initializer_sha256": _STATE.initializer_sha256,
        "source_parameter_count": _STATE.source_parameter_count,
        "generate_calls": _STATE.generate_calls,
        "objective_calls": _STATE.objective_calls,
        "soft_bce_calls": _STATE.soft_bce_calls,
        "mean_absolute_target_shift": (
            None
            if target_count == 0
            else sum(_STATE.target_shift_abs_means) / target_count
        ),
        "minimum_soft_target": (
            None if target_count == 0 else min(_STATE.target_minima)
        ),
        "maximum_soft_target": (
            None if target_count == 0 else max(_STATE.target_maxima)
        ),
    }


def write_diagnostics(output_dir: str | Path) -> Path | None:
    payload = diagnostics_payload()
    if payload is None:
        return None
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "soft_teacher_diagnostics.json"
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


@contextmanager
def patch_online_trainer_runtime() -> Iterator[None]:
    """Apply frozen-teacher targets only inside allowlisted auxiliary BCE calls."""

    import core.online_trainer as online_trainer

    original_factory = online_trainer.build_method_runtime
    original_objective = online_trainer._compute_objective
    wrapped_runtimes: list[tuple[Any, Any]] = []
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
        if _is_teacher_method(method):
            validate_teacher_method(method)
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
        if method.profile_name not in STAGE2_METHOD_IDS:
            return runtime
        original_generate = getattr(runtime, "generate", None)
        if not callable(original_generate):
            raise TypeError(
                "Stage2 teacher method runtime must expose callable generate"
            )

        def teacher_primed_generate(
            *,
            clean_raw: torch.Tensor,
            targets: torch.Tensor,
            hash_ids: Sequence[str],
            classifier: nn.Module,
            base_seed: int,
            rng_identity: Sequence[str],
            composition_indices: torch.Tensor | None = None,
            composition_index_hint: int | None = None,
            objective_term_names: Sequence[str] | None = None,
        ):
            if not isinstance(classifier, nn.Module):
                raise TypeError(
                    "Stage2 teacher runtime requires a torch.nn.Module classifier"
                )
            if _STATE.teacher is None and (
                _STATE.generate_calls != 0 or _STATE.objective_calls != 0
            ):
                raise RuntimeError(
                    "teacher snapshot lifecycle started after Stage2 work"
                )
            _frozen_teacher(classifier, method)
            _STATE.generate_calls += 1
            return original_generate(
                clean_raw=clean_raw,
                targets=targets,
                hash_ids=hash_ids,
                classifier=classifier,
                base_seed=base_seed,
                rng_identity=rng_identity,
                composition_indices=composition_indices,
                composition_index_hint=composition_index_hint,
                objective_term_names=objective_term_names,
            )

        runtime.generate = teacher_primed_generate
        wrapped_runtimes.append((runtime, original_generate))
        return runtime

    def patched_objective(**kwargs: Any):
        method = kwargs.get("method")
        if not _is_teacher_method(method):
            return original_objective(**kwargs)
        assert isinstance(method, CompiledMethod)
        soft_call_limit = validate_teacher_method(method)
        if method.profile_name in DISTILL_METHOD_IDS:
            if kwargs.get("objective_term_names") is not None:
                raise ValueError(
                    f"{method.profile_name} requires one complete ordered "
                    "objective; objective_term_names must be None"
                )
            if _STATE.teacher is None:
                raise RuntimeError(
                    f"{method.profile_name} teacher was not frozen at "
                    "runtime.generate"
                )
            if _STATE.generate_calls != _STATE.objective_calls + 1:
                raise RuntimeError(
                    f"{method.profile_name} generate/objective lifecycle "
                    f"drifted: generate={_STATE.generate_calls}, "
                    f"objective={_STATE.objective_calls}"
                )
            model = kwargs.get("model")
            spec = kwargs.get("spec")
            bundle = kwargs.get("bundle")
            if not isinstance(model, nn.Module) or not isinstance(
                spec, ModelSpec
            ):
                raise TypeError(
                    "teacher-distill objective requires model and ModelSpec"
                )
            clean = bundle.require(BASE_VIEW_NAME)
            auxiliary_name = (
                "lhat_view"
                if method.profile_name == VAE_LHAT_TEACHER_DISTILL_METHOD_ID
                else "lhat_control_view"
            )
            auxiliary = bundle.require(auxiliary_name)
            if not isinstance(clean, WaveformView) or not isinstance(
                auxiliary, WaveformView
            ):
                raise TypeError(
                    "teacher-distill objective requires clean/auxiliary "
                    "WaveformViews"
                )
            if not bool(clean.valid_mask.all().item()):
                raise ValueError(
                    "teacher-distill clean BCE must retain the complete batch"
                )
            soft_targets = _teacher_targets(
                bundle=bundle,
                model=model,
                method=method,
                spec=spec,
                normalization_epsilon=float(
                    kwargs.get("normalization_epsilon")
                ),
            )
            auxiliary_positions = torch.nonzero(
                auxiliary.valid_mask, as_tuple=False
            ).flatten()
            hard_clean_targets = clean.labels
            hard_auxiliary_targets = auxiliary.labels.index_select(
                0, auxiliary_positions
            )
            soft_auxiliary_targets = soft_targets.index_select(
                0, auxiliary_positions
            )
            original_bce = F.binary_cross_entropy_with_logits
            bce_call_index = 0

            def patched_distill_bce(
                input: torch.Tensor,
                target: torch.Tensor,
                weight: torch.Tensor | None = None,
                size_average: bool | None = None,
                reduce: bool | None = None,
                reduction: str = "mean",
                pos_weight: torch.Tensor | None = None,
            ) -> torch.Tensor:
                nonlocal bce_call_index
                bce_call_index += 1
                if bce_call_index in {1, 2}:
                    expected_hard = hard_clean_targets
                    resolved_target = target
                    resolved_pos_weight = pos_weight
                elif bce_call_index == 3:
                    expected_hard = hard_auxiliary_targets
                    resolved_target = soft_auxiliary_targets.to(
                        device=input.device,
                        dtype=input.dtype,
                    )
                    resolved_pos_weight = None
                    _STATE.soft_bce_calls += 1
                else:
                    raise RuntimeError(
                        f"{method.profile_name} emitted more than three BCE calls"
                    )
                expected_hard = expected_hard.to(
                    device=target.device,
                    dtype=target.dtype,
                )
                if target.shape != expected_hard.shape or not torch.equal(
                    target, expected_hard
                ):
                    raise RuntimeError(
                        f"{method.profile_name} BCE call {bce_call_index} "
                        "hard-label routing drifted"
                    )
                if resolved_target.shape != input.shape:
                    raise ValueError(
                        "teacher-distill targets do not match auxiliary logits"
                    )
                return original_bce(
                    input,
                    resolved_target,
                    weight=weight,
                    size_average=size_average,
                    reduce=reduce,
                    reduction=reduction,
                    pos_weight=resolved_pos_weight,
                )

            F.binary_cross_entropy_with_logits = patched_distill_bce
            try:
                result = original_objective(**kwargs)
            finally:
                F.binary_cross_entropy_with_logits = original_bce
            if bce_call_index != 3:
                raise RuntimeError(
                    f"{method.profile_name} BCE call routing drifted: "
                    f"expected 3, got {bce_call_index}"
                )
            _STATE.objective_calls += 1
            return result

        if method.profile_name in J4_METHOD_IDS:
            if kwargs.get("objective_term_names") is not None:
                raise ValueError(
                    f"{method.profile_name} requires one complete ordered objective; "
                    "objective_term_names must be None"
                )
            if _STATE.teacher is None:
                raise RuntimeError(
                    f"{method.profile_name} teacher was not frozen at runtime.generate"
                )
            if _STATE.generate_calls != _STATE.objective_calls + 1:
                raise RuntimeError(
                    f"{method.profile_name} generate/objective lifecycle drifted: "
                    f"generate={_STATE.generate_calls}, "
                    f"objective={_STATE.objective_calls}"
                )
            model = kwargs.get("model")
            spec = kwargs.get("spec")
            bundle = kwargs.get("bundle")
            if not isinstance(model, nn.Module) or not isinstance(spec, ModelSpec):
                raise TypeError(
                    "J4 soft-teacher objective requires model and ModelSpec"
                )
            clean = bundle.require(BASE_VIEW_NAME)
            augmix = bundle.require("augmix_view")
            if not isinstance(clean, WaveformView) or not isinstance(
                augmix, WaveformView
            ):
                raise TypeError(
                    "J4 soft-teacher objective requires clean/augmix WaveformViews"
                )
            if not bool(clean.valid_mask.all().item()):
                raise ValueError("J4 clean BCE must retain the complete hard-label batch")
            soft_targets = _teacher_targets(
                bundle=bundle,
                model=model,
                method=method,
                spec=spec,
                normalization_epsilon=float(kwargs.get("normalization_epsilon")),
            )
            augmix_positions = torch.nonzero(
                augmix.valid_mask, as_tuple=False
            ).flatten()
            hard_clean_targets = clean.labels
            hard_augmix_targets = augmix.labels.index_select(
                0, augmix_positions
            )
            soft_augmix_targets = soft_targets.index_select(
                0, augmix_positions
            )
            original_bce = F.binary_cross_entropy_with_logits
            bce_call_index = 0

            def patched_j4_bce(
                input: torch.Tensor,
                target: torch.Tensor,
                weight: torch.Tensor | None = None,
                size_average: bool | None = None,
                reduce: bool | None = None,
                reduction: str = "mean",
                pos_weight: torch.Tensor | None = None,
            ) -> torch.Tensor:
                nonlocal bce_call_index
                bce_call_index += 1
                if bce_call_index in {1, 2}:
                    expected_hard = hard_clean_targets
                    resolved_target = target
                elif bce_call_index == 3:
                    expected_hard = hard_augmix_targets
                    resolved_target = soft_augmix_targets.to(
                        device=input.device,
                        dtype=input.dtype,
                    )
                    _STATE.soft_bce_calls += 1
                else:
                    raise RuntimeError(
                        f"{method.profile_name} emitted more than three BCE calls"
                    )
                expected_hard = expected_hard.to(
                    device=target.device,
                    dtype=target.dtype,
                )
                if target.shape != expected_hard.shape or not torch.equal(
                    target, expected_hard
                ):
                    raise RuntimeError(
                        f"{method.profile_name} BCE call {bce_call_index} "
                        "hard-label routing drifted"
                    )
                if resolved_target.shape != input.shape:
                    raise ValueError(
                        "J4 soft-teacher targets do not match AugMix logits"
                    )
                return original_bce(
                    input,
                    resolved_target,
                    weight=weight,
                    size_average=size_average,
                    reduce=reduce,
                    reduction=reduction,
                    pos_weight=pos_weight,
                )

            F.binary_cross_entropy_with_logits = patched_j4_bce
            try:
                result = original_objective(**kwargs)
            finally:
                F.binary_cross_entropy_with_logits = original_bce
            if bce_call_index != 3:
                raise RuntimeError(
                    f"{method.profile_name} BCE call routing drifted: "
                    f"expected 3, got {bce_call_index}"
                )
            _STATE.objective_calls += 1
            return result

        selected = tuple(kwargs.get("objective_term_names") or ())
        if selected == ("corrupted_bce",):
            return original_objective(**kwargs)
        expected_auxiliary = (
            ("lhat_direct_bce", "augmix_bce", "clean_lhat_augmix_jsd")
            if soft_call_limit == 2
            else ("augmix_bce", "clean_lhat_augmix_jsd")
        )
        if selected != ("clean_bce", *expected_auxiliary):
            raise ValueError(f"unexpected soft-teacher objective routing: {selected}")
        model = kwargs.get("model")
        spec = kwargs.get("spec")
        if not isinstance(model, nn.Module) or not isinstance(spec, ModelSpec):
            raise TypeError("soft-teacher objective requires model and ModelSpec")
        soft_targets = _teacher_targets(
            bundle=kwargs.get("bundle"),
            model=model,
            method=method,
            spec=spec,
            normalization_epsilon=float(kwargs.get("normalization_epsilon")),
        )
        bundle = kwargs.get("bundle")
        soft_view_names = (
            ("lhat_view", "augmix_view")
            if soft_call_limit == 2
            else ("augmix_view",)
        )
        soft_targets_by_call: list[torch.Tensor] = []
        for view_name in soft_view_names:
            view = bundle.require(view_name)
            if not isinstance(view, WaveformView):
                raise TypeError(f"{view_name} must be a WaveformView")
            positions = torch.nonzero(view.valid_mask, as_tuple=False).flatten()
            soft_targets_by_call.append(soft_targets.index_select(0, positions))
        original_bce = F.binary_cross_entropy_with_logits
        bce_call_index = 0

        def patched_bce(
            input: torch.Tensor,
            target: torch.Tensor,
            weight: torch.Tensor | None = None,
            size_average: bool | None = None,
            reduce: bool | None = None,
            reduction: str = "mean",
            pos_weight: torch.Tensor | None = None,
        ) -> torch.Tensor:
            nonlocal bce_call_index
            bce_call_index += 1
            use_soft = bce_call_index <= soft_call_limit
            resolved_target = (
                soft_targets_by_call[bce_call_index - 1].to(
                    device=input.device,
                    dtype=input.dtype,
                )
                if use_soft
                else target
            )
            if resolved_target.shape != input.shape:
                raise ValueError("soft-teacher targets do not match auxiliary logits")
            if use_soft:
                _STATE.soft_bce_calls += 1
            return original_bce(
                input,
                resolved_target,
                weight=weight,
                size_average=size_average,
                reduce=reduce,
                reduction=reduction,
                pos_weight=pos_weight,
            )

        F.binary_cross_entropy_with_logits = patched_bce
        try:
            result = original_objective(**kwargs)
        finally:
            F.binary_cross_entropy_with_logits = original_bce
        expected_total_calls = soft_call_limit + 1
        if bce_call_index != expected_total_calls:
            raise RuntimeError(
                "soft-teacher BCE call routing drifted: "
                f"expected {expected_total_calls}, got {bce_call_index}"
            )
        _STATE.objective_calls += 1
        return result

    online_trainer.build_method_runtime = patched_factory
    online_trainer._compute_objective = patched_objective
    try:
        yield
    finally:
        for runtime, original_generate in reversed(wrapped_runtimes):
            runtime.generate = original_generate
        online_trainer._compute_objective = original_objective
        online_trainer.build_method_runtime = original_factory
        # Keep the frozen snapshot alive until diagnostics are written by the
        # delegate immediately after this context exits.  The process-local
        # state is cleared at the next context entry.


__all__ = [
    "AUGMIX_METHOD_ID",
    "DIRECT_AUGMIX_METHOD_ID",
    "J4_AUGMIX_METHOD_ID",
    "J4_CONTROL_METHOD_ID",
    "VAE_LHAT_TEACHER_DISTILL_METHOD_ID",
    "CLEAN_TEACHER_DISTILL_CONTROL_METHOD_ID",
    "METHOD_IDS",
    "diagnostics_payload",
    "patch_online_trainer_runtime",
    "validate_teacher_method",
    "write_diagnostics",
]

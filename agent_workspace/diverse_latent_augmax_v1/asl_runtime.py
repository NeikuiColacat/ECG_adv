"""Scoped paper-default asymmetric loss for the sandbox comparison.

The implementation follows Alibaba-MIIL's official multi-label ASL formula
(``gamma_negative=4``, ``gamma_positive=0``, probability margin ``0.05``),
including detached focal weights.  The only deliberate project adaptation is
the reduction: mean over records and classes, matching the whitelist trainer's
existing BCE reduction and therefore keeping the effective learning-rate scale
comparable.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from core.methods import CompiledMethod


DIRECT_METHOD_ID = "fixed20_asl_official_gn4_gp0_m005_v1"
CANDIDATE_METHOD_ID = "fixed20_asl_pcgrad_strong_lhat_augmix_v1"
GROUPDRO_D19_METHOD_ID = "fixed20_groupdro_pcgrad_d19_aux_a050_v1"
GROUPDRO_D19_SUPCON_METHOD_ID = "fixed20_groupdro_pcgrad_d19_supcon_a050_v1"
SEARCH_METHOD_ID = "fixed20_search_d19_v1"
METHOD_IDS = frozenset(
    {
        DIRECT_METHOD_ID,
        CANDIDATE_METHOD_ID,
        GROUPDRO_D19_METHOD_ID,
        GROUPDRO_D19_SUPCON_METHOD_ID,
        SEARCH_METHOD_ID,
    }
)
LOSS_ID = "asymmetric_loss_multilabel_iccv2021"
GAMMA_NEGATIVE = 4.0
GAMMA_POSITIVE = 0.0
PROBABILITY_MARGIN = 0.05
EPSILON = 1.0e-8
REDUCTION = "mean_over_records_and_classes"


@dataclass
class _ASLState:
    calls: int = 0
    elements: int = 0
    losses: list[float] = field(default_factory=list)
    macro_balanced_bce_references: list[float] = field(default_factory=list)
    parameters: dict[str, float] | None = None

    def clear(self) -> None:
        self.calls = 0
        self.elements = 0
        self.losses.clear()
        self.macro_balanced_bce_references.clear()
        self.parameters = None


_STATE = _ASLState()


def _same_float(value: Any, expected: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1.0e-12)
    )


def _is_asl_method(method: Any) -> bool:
    return isinstance(method, CompiledMethod) and (
        method.profile_name in METHOD_IDS
        or method.contracts.get("classification_loss") == LOSS_ID
    )


def _bounded_contract(
    contracts: dict[str, Any], name: str, minimum: float, maximum: float
) -> float:
    value = contracts.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise ValueError(f"ASL contract {name} must lie in [{minimum},{maximum}]")
    return float(value)


def validate_asl_method(method: CompiledMethod) -> tuple[float, float, float]:
    """Fail closed if an ASL marker is copied into an unregistered profile."""

    if method.profile_name not in METHOD_IDS:
        raise ValueError(f"ASL method is not allowlisted: {method.profile_name!r}")
    contracts = method.contracts
    expected = {
        "classification_loss": LOSS_ID,
        "asl_epsilon": EPSILON,
        "asl_focal_weight_gradient": "detached",
        "asl_reduction": REDUCTION,
    }
    for name, value in expected.items():
        actual = contracts.get(name)
        if isinstance(value, float):
            valid = _same_float(actual, value)
        else:
            valid = actual == value
        if not valid:
            raise ValueError(
                f"{method.profile_name} has invalid ASL contract {name}: "
                f"expected {value!r}, got {actual!r}"
            )
    if method.profile_name == SEARCH_METHOD_ID:
        return (
            _bounded_contract(contracts, "asl_gamma_negative", 0.0, 12.0),
            _bounded_contract(contracts, "asl_gamma_positive", 0.0, 6.0),
            _bounded_contract(contracts, "asl_probability_margin", 0.0, 0.30),
        )
    expected_parameters = {
        "asl_gamma_negative": GAMMA_NEGATIVE,
        "asl_gamma_positive": GAMMA_POSITIVE,
        "asl_probability_margin": PROBABILITY_MARGIN,
    }
    for name, value in expected_parameters.items():
        if not _same_float(contracts.get(name), value):
            raise ValueError(
                f"{method.profile_name} has invalid ASL contract {name}: "
                f"expected {value!r}, got {contracts.get(name)!r}"
            )
    return GAMMA_NEGATIVE, GAMMA_POSITIVE, PROBABILITY_MARGIN


def asymmetric_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma_negative: float = GAMMA_NEGATIVE,
    gamma_positive: float = GAMMA_POSITIVE,
    probability_margin: float = PROBABILITY_MARGIN,
) -> torch.Tensor:
    """Official multi-label ASL terms with BCE-compatible mean reduction."""

    if logits.shape != targets.shape:
        raise ValueError("ASL logits and targets must have identical shapes")
    if not logits.is_floating_point() or not targets.is_floating_point():
        raise TypeError("ASL logits and targets must be floating-point tensors")
    probabilities_positive = torch.sigmoid(logits)
    probabilities_negative = (1.0 - probabilities_positive + probability_margin).clamp(
        max=1.0
    )
    log_likelihood = targets * torch.log(
        probabilities_positive.clamp(min=EPSILON)
    )
    log_likelihood = log_likelihood + (1.0 - targets) * torch.log(
        probabilities_negative.clamp(min=EPSILON)
    )
    # The official implementation disables autograd while constructing the
    # focal weights.  Detach is the local, exception-safe equivalent.
    with torch.no_grad():
        probability_true = (
            probabilities_positive * targets
            + probabilities_negative * (1.0 - targets)
        )
        gamma = (
            gamma_positive * targets
            + gamma_negative * (1.0 - targets)
        )
        asymmetric_weight = torch.pow(1.0 - probability_true, gamma)
    loss = -(log_likelihood * asymmetric_weight).mean()
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("ASL produced a non-finite loss")
    return loss


def macro_balanced_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return a detached class-balanced corruption difficulty surrogate.

    Each class contributes equally.  Within a class, present positive and
    negative partitions also contribute equally, preventing common negative
    labels from hiding rare positive failures.  The value is a GroupDRO update
    statistic only; ASL remains the differentiated training objective.
    """

    if logits.shape != targets.shape:
        raise ValueError("balanced BCE logits and targets must have identical shapes")
    if logits.ndim != 2:
        raise ValueError("balanced BCE expects [batch, class] tensors")
    with torch.no_grad():
        detached_logits = logits.detach()
        detached_targets = targets.detach()
        # Stable elementwise BCE without calling the temporarily patched
        # ``F.binary_cross_entropy_with_logits`` and recursing.
        elementwise = F.softplus(detached_logits) - detached_targets * detached_logits
        class_losses = []
        for class_index in range(int(targets.shape[1])):
            positive = targets[:, class_index] > 0.5
            partitions = []
            if bool(positive.any().item()):
                partitions.append(elementwise[positive, class_index].mean())
            negative = ~positive
            if bool(negative.any().item()):
                partitions.append(elementwise[negative, class_index].mean())
            if not partitions:
                raise RuntimeError("balanced BCE class has no valid records")
            class_losses.append(torch.stack(partitions).mean())
        loss = torch.stack(class_losses).mean()
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("balanced BCE produced a non-finite loss")
    return loss


def macro_balanced_bce_call_count() -> int:
    return len(_STATE.macro_balanced_bce_references)


def macro_balanced_bce_since(call_count: int) -> float:
    """Return the sole reference recorded after ``call_count``."""

    current = len(_STATE.macro_balanced_bce_references)
    if current != call_count + 1:
        raise RuntimeError(
            "GroupDRO corrupted objective must record exactly one balanced BCE "
            f"reference, observed {current - call_count}"
        )
    return _STATE.macro_balanced_bce_references[-1]


def diagnostics_payload() -> dict[str, Any] | None:
    if _STATE.calls == 0 and _STATE.parameters is None:
        return None
    disabled = _STATE.parameters == {
        "gamma_negative": 0.0,
        "gamma_positive": 0.0,
        "probability_margin": 0.0,
    }
    return {
        "schema_version": 1,
        "loss_id": LOSS_ID,
        "enabled": not disabled,
        "parameters": {
            **(_STATE.parameters or {
                "gamma_negative": GAMMA_NEGATIVE,
                "gamma_positive": GAMMA_POSITIVE,
                "probability_margin": PROBABILITY_MARGIN,
            }),
            "epsilon": EPSILON,
            "focal_weight_gradient": "detached",
            "reduction": REDUCTION,
        },
        "call_count": _STATE.calls,
        "element_count": _STATE.elements,
        "mean_call_loss": (
            sum(_STATE.losses) / len(_STATE.losses)
            if _STATE.losses
            else None
        ),
    }


def write_diagnostics(output_dir: str | Path) -> Path | None:
    payload = diagnostics_payload()
    if payload is None:
        return None
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "asl_diagnostics.json"
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


@contextmanager
def patch_online_trainer_runtime() -> Iterator[None]:
    """Replace BCE only while evaluating an allowlisted ASL method objective."""

    import core.online_trainer as online_trainer

    original_objective = online_trainer._compute_objective
    _STATE.clear()

    def patched_objective(**kwargs: Any):
        method = kwargs.get("method")
        if not _is_asl_method(method):
            return original_objective(**kwargs)
        assert isinstance(method, CompiledMethod)
        gamma_negative, gamma_positive, probability_margin = validate_asl_method(method)
        parameters = {
            "gamma_negative": gamma_negative,
            "gamma_positive": gamma_positive,
            "probability_margin": probability_margin,
        }
        if _STATE.parameters is None:
            _STATE.parameters = parameters
        elif _STATE.parameters != parameters:
            raise RuntimeError("ASL parameters changed within one training run")
        if parameters == {
            "gamma_negative": 0.0,
            "gamma_positive": 0.0,
            "probability_margin": 0.0,
        }:
            return original_objective(**kwargs)
        original_bce = F.binary_cross_entropy_with_logits

        def patched_bce(
            input: torch.Tensor,
            target: torch.Tensor,
            weight: torch.Tensor | None = None,
            size_average: bool | None = None,
            reduce: bool | None = None,
            reduction: str = "mean",
            pos_weight: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if weight is not None or pos_weight is not None:
                raise ValueError("paper-default ASL does not accept BCE weights")
            if size_average is not None or reduce is not None or reduction != "mean":
                raise ValueError("sandbox ASL requires the locked mean reduction")
            reference = macro_balanced_bce_with_logits(input, target)
            loss = asymmetric_loss_with_logits(
                input,
                target,
                gamma_negative=gamma_negative,
                gamma_positive=gamma_positive,
                probability_margin=probability_margin,
            )
            _STATE.calls += 1
            _STATE.elements += int(target.numel())
            _STATE.losses.append(float(loss.detach().float().cpu()))
            _STATE.macro_balanced_bce_references.append(
                float(reference.detach().float().cpu())
            )
            return loss

        # ``core.online_trainer.F`` and this module's ``F`` are the same
        # torch.nn.functional module.  The substitution is tightly scoped to
        # one synchronous objective call and restored even on failure.
        F.binary_cross_entropy_with_logits = patched_bce
        try:
            return original_objective(**kwargs)
        finally:
            F.binary_cross_entropy_with_logits = original_bce

    online_trainer._compute_objective = patched_objective
    try:
        yield
    finally:
        online_trainer._compute_objective = original_objective


__all__ = [
    "CANDIDATE_METHOD_ID",
    "DIRECT_METHOD_ID",
    "GROUPDRO_D19_METHOD_ID",
    "GROUPDRO_D19_SUPCON_METHOD_ID",
    "SEARCH_METHOD_ID",
    "METHOD_IDS",
    "asymmetric_loss_with_logits",
    "macro_balanced_bce_call_count",
    "macro_balanced_bce_since",
    "macro_balanced_bce_with_logits",
    "diagnostics_payload",
    "patch_online_trainer_runtime",
    "validate_asl_method",
    "write_diagnostics",
]

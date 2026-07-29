"""Scoped objective scaling for allowlisted sandbox graph methods.

The whitelist generic graph runtime generates all method views.  This sidecar
does exactly two things while installed: fail closed on each allowlisted
method's frozen contract and multiply the compiled objective total/weighted
terms by 0.5.  Raw terms and valid counts remain untouched.  It deliberately
has no endpoint or radius-calibration responsibility.
"""

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import core.methods.runtime as core_method_runtime
from core.augmix import (
    _load_operator_profile,
    generate_three_chain_augmix,
)
from core.corruption import generate_canonical_corruption
from core.methods.contracts import (
    NodeContext,
    Provenance,
    ViewBundle,
    ViewValue,
    WaveformView,
)
from core.methods.registry import CompiledMethod
from core.methods.runtime import GeneratedMethodBatch, MethodViewRuntime
from models.contracts import validate_model_output
from models.factory import get_model_spec
from models.input_adapter import prepare_canonical_model_input

try:  # Package import and direct delegate subprocess import are both supported.
    from .paired_path_runtime import (
        _decode_latents_chunked,
        apply_linear_endpoint_residual_correction,
        project_paired_latent_path,
    )
    from .runtime_adapter import _frozen_classifier_for_attack
except ImportError:  # pragma: no cover - direct sibling import in delegate.py
    from paired_path_runtime import (  # type: ignore[no-redef]
        _decode_latents_chunked,
        apply_linear_endpoint_residual_correction,
        project_paired_latent_path,
    )
    from runtime_adapter import _frozen_classifier_for_attack  # type: ignore[no-redef]


D8_METHOD_ID = "diverse_augmax_d8_dual_axis_v1"
D9_METHOD_ID = "diverse_augmax_d9_dual_axis_jsd06_v1"
R4_RESIDUAL_LATENT2_METHOD_ID = "r4_residual_latent2_v1"
R4_LHAT_AUGMIX_LOWDOSE_METHOD_ID = "r4_lhat_augmix_lowdose_v1"
R8_RESIDUAL_LATENT1_CLEAN55_METHOD_ID = "r8_residual_latent1_clean55_v1"
A5_M20_LHAT_AUGMIX_METHOD_ID = "a5_m20_lhat_augmix_v1"
PURE_M20_LHAT_AUGMIX_METHOD_ID = "pure_m20_lhat_augmix"
VAE_LHAT_BOUNDED_AUGMAX_POLISH_METHOD_ID = (
    "vae_lhat_bounded_augmax_polish_v1"
)
CLEAN_BOUNDED_AUGMAX_CONTROL_METHOD_ID = (
    "clean_bounded_augmax_polish_control_v1"
)
BOUNDED_AUGMAX_METHOD_IDS = frozenset(
    {
        VAE_LHAT_BOUNDED_AUGMAX_POLISH_METHOD_ID,
        CLEAN_BOUNDED_AUGMAX_CONTROL_METHOD_ID,
    }
)
PURE_M20_RUNTIME_METHOD_IDS = frozenset(
    {PURE_M20_LHAT_AUGMIX_METHOD_ID, *BOUNDED_AUGMAX_METHOD_IDS}
)
VAE_LHAT_CLEAN_POLISH_METHOD_ID = "vae_lhat_clean_polish_v1"
VAE_LHAT_CHAIN3_POLISH_METHOD_ID = "vae_lhat_chain3_polish_v1"
CLEAN_CHAIN3_POLISH_CONTROL_METHOD_ID = "clean_chain3_polish_control_v1"
J4_VAE_LHAT_CHAIN3_METHOD_ID = "vae_lhat_chain3_teacher04_polish_v1"
J4_CLEAN_CHAIN3_CONTROL_METHOD_ID = "clean_chain3_teacher04_polish_control_v1"
VAE_LHAT_TEACHER_DISTILL_METHOD_ID = (
    "vae_lhat_teacher_distill_polish_v1"
)
CLEAN_TEACHER_DISTILL_CONTROL_METHOD_ID = (
    "clean_teacher_distill_polish_control_v1"
)
VAE_LHAT_HARD_BCE_METHOD_ID = "vae_lhat_hard_bce_polish_v1"
CLEAN_HARD_BCE_CONTROL_METHOD_ID = "clean_hard_bce_polish_control_v1"
VAE_LHAT_PATH_BCE_METHOD_ID = "vae_lhat_path_bce_polish_v1"
CLEAN_LHAT_PATH_CONTROL_METHOD_ID = "clean_lhat_path_bce_polish_control_v1"
VAE_LHAT_CONSISTENCY_METHOD_ID = "vae_lhat_consistency_polish_v1"
CLEAN_CONSISTENCY_CONTROL_METHOD_ID = "clean_consistency_polish_control_v1"
VAE_LHAT_FEATURE_INVARIANCE_METHOD_ID = (
    "vae_lhat_feature_invariance_polish_v1"
)
CLEAN_FEATURE_INVARIANCE_CONTROL_METHOD_ID = (
    "clean_feature_invariance_polish_control_v1"
)
VAE_LHAT_LOCAL_ANCHOR_SOFT_METHOD_ID = (
    "vae_lhat_local_anchor_soft_polish_v1"
)
CLEAN_LOCAL_ANCHOR_SOFT_CONTROL_METHOD_ID = (
    "clean_local_anchor_soft_polish_control_v1"
)
VAE_CALIBRATED_LOCAL_ANCHOR_SOFT_METHOD_ID = (
    "vae_lhat_calibrated_local_anchor_soft_polish_v1"
)
CLEAN_CALIBRATED_LOCAL_ANCHOR_SOFT_CONTROL_METHOD_ID = (
    "clean_calibrated_local_anchor_soft_polish_control_v1"
)
LOCAL_ANCHOR_SOFT_CANDIDATE_IDS = frozenset(
    {
        VAE_LHAT_LOCAL_ANCHOR_SOFT_METHOD_ID,
        VAE_CALIBRATED_LOCAL_ANCHOR_SOFT_METHOD_ID,
    }
)
LOCAL_ANCHOR_SOFT_CONTROL_IDS = frozenset(
    {
        CLEAN_LOCAL_ANCHOR_SOFT_CONTROL_METHOD_ID,
        CLEAN_CALIBRATED_LOCAL_ANCHOR_SOFT_CONTROL_METHOD_ID,
    }
)
LOCAL_ANCHOR_SOFT_METHOD_IDS = (
    LOCAL_ANCHOR_SOFT_CANDIDATE_IDS | LOCAL_ANCHOR_SOFT_CONTROL_IDS
)
LHAT_PATH_T_VALUES = (0.4, 0.7, 1.0)
SCALE_ONLY_METHOD_IDS = frozenset(
    {
        D8_METHOD_ID,
        D9_METHOD_ID,
        R4_RESIDUAL_LATENT2_METHOD_ID,
        R4_LHAT_AUGMIX_LOWDOSE_METHOD_ID,
        R8_RESIDUAL_LATENT1_CLEAN55_METHOD_ID,
        A5_M20_LHAT_AUGMIX_METHOD_ID,
        PURE_M20_LHAT_AUGMIX_METHOD_ID,
        VAE_LHAT_BOUNDED_AUGMAX_POLISH_METHOD_ID,
        CLEAN_BOUNDED_AUGMAX_CONTROL_METHOD_ID,
        VAE_LHAT_CLEAN_POLISH_METHOD_ID,
        VAE_LHAT_CHAIN3_POLISH_METHOD_ID,
        CLEAN_CHAIN3_POLISH_CONTROL_METHOD_ID,
        J4_VAE_LHAT_CHAIN3_METHOD_ID,
        J4_CLEAN_CHAIN3_CONTROL_METHOD_ID,
        VAE_LHAT_TEACHER_DISTILL_METHOD_ID,
        CLEAN_TEACHER_DISTILL_CONTROL_METHOD_ID,
        VAE_LHAT_HARD_BCE_METHOD_ID,
        CLEAN_HARD_BCE_CONTROL_METHOD_ID,
        VAE_LHAT_PATH_BCE_METHOD_ID,
        CLEAN_LHAT_PATH_CONTROL_METHOD_ID,
        VAE_LHAT_CONSISTENCY_METHOD_ID,
        CLEAN_CONSISTENCY_CONTROL_METHOD_ID,
        VAE_LHAT_FEATURE_INVARIANCE_METHOD_ID,
        CLEAN_FEATURE_INVARIANCE_CONTROL_METHOD_ID,
        VAE_LHAT_LOCAL_ANCHOR_SOFT_METHOD_ID,
        CLEAN_LOCAL_ANCHOR_SOFT_CONTROL_METHOD_ID,
        VAE_CALIBRATED_LOCAL_ANCHOR_SOFT_METHOD_ID,
        CLEAN_CALIBRATED_LOCAL_ANCHOR_SOFT_CONTROL_METHOD_ID,
    }
)
OBJECTIVE_GLOBAL_SCALE = 0.5


def _managed_classifier_head(model: nn.Module) -> nn.Module:
    """Return the managed linear head whose input is the penultimate feature."""

    dense = getattr(model, "dense", None)
    if isinstance(dense, nn.Linear):
        return dense
    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Sequential) and len(classifier):
        head = classifier[-1]
        if isinstance(head, nn.Linear):
            return head
    raise TypeError(
        "feature-invariance polish requires the managed linear classifier head"
    )


class _PenultimateFeatureCaptureModel(nn.Module):
    """Capture classifier-head inputs without adding another model forward."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.features: list[torch.Tensor] = []

        def capture(
            _module: nn.Module, arguments: tuple[torch.Tensor, ...]
        ) -> None:
            if len(arguments) != 1 or arguments[0].ndim != 2:
                raise RuntimeError(
                    "feature-invariance polish captured an invalid feature tensor"
                )
            self.features.append(arguments[0])

        self._handle = _managed_classifier_head(model).register_forward_pre_hook(
            capture
        )

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        return self.model(model_input)

    def close(self) -> None:
        self._handle.remove()


class _LogitCaptureModel(nn.Module):
    """Capture ordered objective logits without adding a model forward."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.logits: list[torch.Tensor] = []

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        output = self.model(model_input)
        self.logits.append(output)
        return output


@contextmanager
def _disable_batch_norm_running_updates(model: nn.Module) -> Iterator[None]:
    """Keep recomputation on batch statistics without updating buffers twice."""

    modules = tuple(
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        and bool(module.track_running_stats)
    )
    original_momenta = tuple(module.momentum for module in modules)
    try:
        for module in modules:
            module.momentum = 0.0
        yield
    finally:
        for module, momentum in zip(modules, original_momenta, strict=True):
            module.momentum = momentum


class _CheckpointedModelForward(nn.Module):
    """Recompute one model view during backward while preserving BN buffers."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, model_input):
        def contexts():
            return nullcontext(), _disable_batch_norm_running_updates(self.model)

        return checkpoint(
            self.model,
            model_input,
            use_reentrant=False,
            preserve_rng_state=True,
            context_fn=contexts,
        )


def _cap_vae_chain_weight(
    weights: torch.Tensor, cap: float
) -> torch.Tensor:
    """Cap chain three and redistribute its excess across both raw chains."""

    if weights.ndim != 2 or weights.shape[1] != 3:
        raise ValueError("AugMix weights must have shape (B,3)")
    if not 0.0 <= float(cap) <= 1.0:
        raise ValueError("VAE chain weight cap must lie in [0,1]")
    vae = weights[:, 2].clamp(max=float(cap))
    raw = weights[:, :2]
    raw = raw / raw.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    raw = raw * (1.0 - vae).unsqueeze(1)
    return torch.cat((raw, vae.unsqueeze(1)), dim=1).contiguous()


def _floor_canonical_chain1_weight(
    weights: torch.Tensor, floor: float
) -> torch.Tensor:
    """Keep a minimum canonical dose while preserving a convex three-chain mix."""

    if weights.ndim != 2 or weights.shape[1] != 3:
        raise ValueError("AugMix weights must have shape (B,3)")
    if not 0.0 <= float(floor) <= 1.0:
        raise ValueError("canonical chain1 weight floor must lie in [0,1]")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()):
        raise ValueError("AugMix weights must be finite and non-negative")
    if not bool(
        torch.allclose(
            weights.sum(dim=1),
            torch.ones(weights.shape[0], device=weights.device, dtype=weights.dtype),
            atol=1.0e-5,
            rtol=1.0e-5,
        )
    ):
        raise ValueError("AugMix weights must sum to one")

    canonical = weights[:, 0].clamp_min(float(floor))
    remaining = (1.0 - canonical).clamp_min(0.0)
    tail = weights[:, 1:]
    tail_sum = tail.sum(dim=1, keepdim=True)
    tail_proportions = torch.where(
        tail_sum > 1.0e-12,
        tail / tail_sum.clamp_min(1.0e-12),
        torch.full_like(tail, 0.5),
    )
    return torch.cat(
        (canonical.unsqueeze(1), tail_proportions * remaining.unsqueeze(1)),
        dim=1,
    ).contiguous()


def _sparse_augmix_weights(
    template: torch.Tensor,
    vae_probability: float,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample one complete chain, the low-alpha limit of AugMix weights."""

    if template.ndim != 2 or template.shape[1] != 3:
        raise ValueError("AugMix weight template must have shape (B,3)")
    if not 0.0 <= float(vae_probability) <= 1.0:
        raise ValueError("VAE chain probability must lie in [0,1]")
    raw_probability = 0.5 * (1.0 - float(vae_probability))
    probabilities = torch.tensor(
        (raw_probability, raw_probability, float(vae_probability)),
        device=template.device,
        dtype=torch.float32,
    ).expand(template.shape[0], -1)
    selected = torch.multinomial(
        probabilities,
        1,
        replacement=True,
        generator=generator,
    )
    return torch.zeros_like(template).scatter_(1, selected, 1.0).contiguous()


def _sample_valid_path_indices(
    valid_mask: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Replayably sample one valid latent-path position per record."""

    if (
        not isinstance(valid_mask, torch.Tensor)
        or valid_mask.ndim != 2
        or valid_mask.dtype != torch.bool
        or valid_mask.shape[1] < 2
    ):
        raise ValueError("latent path valid_mask must be bool (B,K), K >= 2")
    if not bool(valid_mask.any(dim=1).all()):
        raise ValueError("every latent path row must contain a valid position")
    probabilities = valid_mask.to(dtype=torch.float32)
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
    return torch.multinomial(
        probabilities,
        1,
        replacement=True,
        generator=generator,
    ).flatten().contiguous()


def _orthogonal_logit_novelty(
    clean_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Measure candidate logit motion outside two AugMix reference directions."""

    if clean_logits.ndim != 2:
        raise ValueError("clean logits must have shape (B,C)")
    if reference_logits.shape != (clean_logits.shape[0], 2, clean_logits.shape[1]):
        raise ValueError("reference logits must have shape (B,2,C)")
    if (
        candidate_logits.ndim != 3
        or candidate_logits.shape[0] != clean_logits.shape[0]
        or candidate_logits.shape[2] != clean_logits.shape[1]
    ):
        raise ValueError("candidate logits must have shape (B,K,C)")
    if not 0.0 < float(epsilon) < 1.0:
        raise ValueError("novelty epsilon must lie in (0,1)")

    references = reference_logits.float() - clean_logits[:, None].float()
    first = references[:, 0]
    first_norm = first.norm(dim=1, keepdim=True)
    first_unit = torch.where(
        first_norm > epsilon,
        first / first_norm.clamp_min(epsilon),
        torch.zeros_like(first),
    )
    second = references[:, 1]
    second_residual = second - (
        (second * first_unit).sum(dim=1, keepdim=True) * first_unit
    )
    second_norm = second_residual.norm(dim=1, keepdim=True)
    second_unit = torch.where(
        second_norm > epsilon,
        second_residual / second_norm.clamp_min(epsilon),
        torch.zeros_like(second_residual),
    )

    candidate_delta = candidate_logits.float() - clean_logits[:, None].float()
    first_projection = (
        (candidate_delta * first_unit[:, None]).sum(dim=2, keepdim=True)
        * first_unit[:, None]
    )
    second_projection = (
        (candidate_delta * second_unit[:, None]).sum(dim=2, keepdim=True)
        * second_unit[:, None]
    )
    orthogonal = candidate_delta - first_projection - second_projection
    total_norm = candidate_delta.norm(dim=2)
    return torch.where(
        total_norm > epsilon,
        orthogonal.norm(dim=2) / total_norm.clamp_min(epsilon),
        torch.zeros_like(total_norm),
    ).clamp_(0.0, 1.0)


def _select_complementary_path_indices(
    valid_mask: torch.Tensor,
    bce_gain: torch.Tensor,
    novelty: torch.Tensor,
    *,
    minimum_bce_gain: float,
    maximum_bce_gain: float,
    minimum_novelty: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a replayable useful-hard path point or mark VAE unavailable."""

    if (
        valid_mask.ndim != 2
        or valid_mask.dtype != torch.bool
        or bce_gain.shape != valid_mask.shape
        or novelty.shape != valid_mask.shape
    ):
        raise ValueError("path selection tensors must align as (B,K)")
    if not (
        0.0 <= float(minimum_bce_gain) <= float(maximum_bce_gain)
        and 0.0 <= float(minimum_novelty) <= 1.0
    ):
        raise ValueError("path selection thresholds are invalid")
    admissible = (
        valid_mask
        & torch.isfinite(bce_gain)
        & torch.isfinite(novelty)
        & (bce_gain >= float(minimum_bce_gain))
        & (bce_gain <= float(maximum_bce_gain))
        & (novelty >= float(minimum_novelty))
    )
    score = torch.where(
        admissible,
        bce_gain.clamp_min(0.0) * novelty,
        torch.full_like(bce_gain, -torch.inf),
    )
    selected = score.argmax(dim=1)
    available = admissible.any(dim=1)
    return selected.contiguous(), available.contiguous()


def _select_lhat_guided_augmix_indices(
    lhat_direction: torch.Tensor,
    candidate_directions: torch.Tensor,
    previous_directions: torch.Tensor | None,
    available: torch.Tensor,
    valid_candidates: torch.Tensor,
    *,
    alignment_weight: float,
    diversity_weight: float,
    minimum_alignment: float = 0.0,
    epsilon: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select diverse raw AugMix candidates that follow the LHAT direction."""

    if lhat_direction.ndim != 2:
        raise ValueError("LHAT direction must have shape (B,C)")
    if (
        candidate_directions.ndim != 3
        or candidate_directions.shape[0] != lhat_direction.shape[0]
        or candidate_directions.shape[2] != lhat_direction.shape[1]
    ):
        raise ValueError("candidate directions must have shape (B,K,C)")
    if (
        available.shape != (lhat_direction.shape[0],)
        or available.dtype != torch.bool
        or valid_candidates.shape != candidate_directions.shape[:2]
        or valid_candidates.dtype != torch.bool
    ):
        raise ValueError("guided AugMix availability/validity lost alignment")
    if previous_directions is not None and (
        previous_directions.ndim != 3
        or previous_directions.shape[0] != lhat_direction.shape[0]
        or previous_directions.shape[2] != lhat_direction.shape[1]
    ):
        raise ValueError("previous directions must have shape (B,N,C)")
    if not (
        0.0 <= float(alignment_weight) <= 1.0
        and 0.0 <= float(diversity_weight) <= 1.0
        and math.isclose(
            float(alignment_weight) + float(diversity_weight),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-8,
        )
        and -1.0 <= float(minimum_alignment) <= 1.0
        and 0.0 < float(epsilon) < 1.0
    ):
        raise ValueError("guided AugMix score weights or epsilon are invalid")

    lhat_unit = F.normalize(lhat_direction.float(), dim=1, eps=float(epsilon))
    candidate_unit = F.normalize(
        candidate_directions.float(), dim=2, eps=float(epsilon)
    )
    alignment = (candidate_unit * lhat_unit[:, None]).sum(dim=2).clamp(-1.0, 1.0)
    if previous_directions is None or previous_directions.shape[1] == 0:
        diversity = torch.ones_like(alignment)
    else:
        previous_unit = F.normalize(
            previous_directions.float(), dim=2, eps=float(epsilon)
        )
        redundancy = torch.einsum(
            "bkc,bnc->bkn", candidate_unit, previous_unit
        ).clamp_min(0.0).amax(dim=2)
        diversity = (1.0 - redundancy).clamp(0.0, 1.0)
    score = (
        float(alignment_weight) * alignment
        + float(diversity_weight) * diversity
    )
    selectable = valid_candidates & available[:, None]
    score = torch.where(selectable, score, torch.full_like(score, -torch.inf))
    selected = score.argmax(dim=1)
    rows = torch.arange(
        candidate_directions.shape[0],
        device=candidate_directions.device,
        dtype=torch.long,
    )
    guided = selectable.any(dim=1) & (
        alignment[rows, selected] >= float(minimum_alignment)
    )
    # The first valid candidate is an independently generated raw AugMix
    # fallback.  This keeps the 20-view supervision budget complete without
    # inventing a VAE endpoint for records that have no valid LHAT direction.
    fallback = valid_candidates.to(dtype=torch.int64).argmax(dim=1)
    selected = torch.where(guided, selected, fallback)
    selected_alignment = torch.where(
        guided, alignment[rows, selected], torch.zeros_like(alignment[:, 0])
    )
    selected_diversity = torch.where(
        guided, diversity[rows, selected], torch.ones_like(diversity[:, 0])
    )
    return selected.contiguous(), selected_alignment, selected_diversity


@dataclass(frozen=True)
class _A5LatentPathBank:
    """One batch-local clean-to-LHAT path bank shared by all AugMix views."""

    accepted_positions: tuple[int, ...]
    waveforms: torch.Tensor
    valid_mask: torch.Tensor
    t_values: torch.Tensor

    def __post_init__(self) -> None:
        accepted = len(self.accepted_positions)
        if (
            self.waveforms.ndim != 4
            or tuple(self.waveforms.shape[:2]) != (accepted, len(self.t_values))
            or tuple(self.waveforms.shape[2:]) != (1000, 12)
        ):
            raise ValueError("A5 latent path waveforms must have shape (A,K,1000,12)")
        if self.valid_mask.shape != self.waveforms.shape[:2]:
            raise ValueError("A5 latent path validity must align with (A,K)")
        if self.valid_mask.dtype != torch.bool:
            raise TypeError("A5 latent path validity must be boolean")
        if not bool(self.valid_mask.any(dim=1).all()):
            raise ValueError("every accepted A5 record needs a valid path point")
        if not bool(torch.isfinite(self.waveforms).all()):
            raise ValueError("A5 latent path waveforms must be finite")

    def sample(
        self,
        *,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected = _sample_valid_path_indices(
            self.valid_mask,
            generator=generator,
        )
        rows = torch.arange(
            int(self.waveforms.shape[0]),
            device=self.waveforms.device,
            dtype=torch.long,
        )
        return (
            self.waveforms[rows, selected].contiguous(),
            self.t_values.index_select(0, selected).contiguous(),
        )


@dataclass(frozen=True)
class _A5ComplementaryPath:
    """One path point per accepted record selected against AugMix directions."""

    waveforms: torch.Tensor
    t_values: torch.Tensor
    available_mask: torch.Tensor
    bce_gain: torch.Tensor
    novelty: torch.Tensor

    def __post_init__(self) -> None:
        accepted = int(self.waveforms.shape[0])
        if self.waveforms.shape != (accepted, 1000, 12):
            raise ValueError("complementary path waveforms must have shape (A,1000,12)")
        for name, value in (
            ("t_values", self.t_values),
            ("available_mask", self.available_mask),
            ("bce_gain", self.bce_gain),
            ("novelty", self.novelty),
        ):
            if value.shape != (accepted,):
                raise ValueError(f"complementary path {name} must have shape (A,)")
        if self.available_mask.dtype != torch.bool:
            raise TypeError("complementary path availability must be boolean")
        if not bool(torch.isfinite(self.waveforms).all()):
            raise ValueError("complementary path waveforms must be finite")


@dataclass
class _A5GuidedAugMixState:
    """Batch-local LHAT direction and already selected AugMix directions."""

    clean_logits: torch.Tensor
    clean_features: torch.Tensor
    lhat_direction: torch.Tensor
    available_mask: torch.Tensor
    selected_directions: list[torch.Tensor]

    def __post_init__(self) -> None:
        if self.clean_logits.ndim != 2:
            raise ValueError("guided AugMix clean logits must have shape (B,C)")
        if (
            self.clean_features.ndim != 2
            or self.clean_features.shape[0] != self.clean_logits.shape[0]
        ):
            raise ValueError("guided AugMix clean features must have shape (B,F)")
        if self.lhat_direction.shape != self.clean_features.shape:
            raise ValueError("guided AugMix LHAT direction lost alignment")
        if (
            self.available_mask.shape != (self.clean_logits.shape[0],)
            or self.available_mask.dtype != torch.bool
        ):
            raise ValueError("guided AugMix availability must be boolean shape (B,)")
        if not bool(torch.isfinite(self.clean_logits).all()):
            raise ValueError("guided AugMix clean logits must be finite")
        if not bool(torch.isfinite(self.clean_features).all()):
            raise ValueError("guided AugMix clean features must be finite")
        if not bool(torch.isfinite(self.lhat_direction).all()):
            raise ValueError("guided AugMix LHAT direction must be finite")


class _A5CalibratedMixRuntime(MethodViewRuntime):
    """Matched M20 runtime for calibrated or ordinary convex AugMix."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        control_mode = self.method.contracts.get(
            "matched_chain3_control_mode"
        )
        self._matched_clean_chain3_control = (
            self.method.profile_name
            == CLEAN_BOUNDED_AUGMAX_CONTROL_METHOD_ID
        )
        if self._matched_clean_chain3_control:
            if (
                control_mode
                != "replace_lhat_with_clean_after_full_search_and_qc"
                or self.method.contracts.get(
                    "control_executes_full_lhat_search"
                )
                is not True
            ):
                raise ValueError(
                    "bounded AugMax control must execute full LHAT and replace "
                    "only its accepted waveform"
                )
        elif control_mode is not None:
            raise ValueError(
                "matched_chain3_control_mode is restricted to the bounded "
                "AugMax control"
            )
        raw = self.method.contracts.get("calibrated_augmix_mixing")
        if not isinstance(raw, Mapping):
            raise ValueError("A5-M20 requires calibrated_augmix_mixing")
        self._strength_min = float(raw.get("strength_min", -1.0))
        self._strength_max = float(raw.get("strength_max", -1.0))
        self._vae_weight_cap = float(raw.get("vae_chain_weight_cap", -1.0))
        self._mixing_mode = str(raw.get("mixing_mode", "convex_capped"))
        self._vae_chain_probability = float(
            raw.get("vae_chain_probability", self._vae_weight_cap)
        )
        self._canonical_chain1_weight_floor = float(
            raw.get("canonical_chain1_weight_floor", 0.0)
        )
        path_raw = self.method.contracts.get("latent_hard_path_sampling")
        self._path_t_values: tuple[float, ...] = ()
        self._path_selection: Mapping[str, Any] | None = None
        if path_raw is not None:
            if not isinstance(path_raw, Mapping):
                raise ValueError("latent_hard_path_sampling must be a mapping")
            expected_path_keys = {
                "enabled",
                "t_values",
                "sampling",
                "geometry",
                "residual_correction",
                "cache_scope",
            }
            if set(path_raw) != expected_path_keys:
                raise ValueError(
                    "latent_hard_path_sampling keys must be exactly "
                    f"{sorted(expected_path_keys)}"
                )
            if path_raw["enabled"] is not True:
                raise ValueError("declared latent hard path sampling must be enabled")
            if path_raw["sampling"] != "uniform_valid_per_augmix_node":
                raise ValueError("A5 latent path sampling policy is unsupported")
            if path_raw["geometry"] != "standardized_clean_to_lhat_ray":
                raise ValueError("A5 latent path geometry is unsupported")
            if path_raw["residual_correction"] != "linear_clean_hard_endpoint":
                raise ValueError("A5 latent path residual correction is unsupported")
            if path_raw["cache_scope"] != "one_generate_call":
                raise ValueError("A5 latent path cache must be batch-local")
            values = tuple(float(value) for value in path_raw["t_values"])
            if (
                len(values) < 2
                or any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in values)
                or any(right <= left for left, right in zip(values, values[1:]))
                or not math.isclose(values[-1], 1.0, rel_tol=0.0, abs_tol=1.0e-12)
            ):
                raise ValueError(
                    "A5 latent path t_values must be strictly increasing in (0,1] "
                    "and include endpoint 1"
                )
            self._path_t_values = values
        selection_raw = self.method.contracts.get("latent_hard_path_selection")
        if selection_raw is not None:
            if not isinstance(selection_raw, Mapping):
                raise ValueError("latent_hard_path_selection must be a mapping")
            expected_selection_keys = {
                "enabled",
                "method",
                "minimum_bce_gain",
                "maximum_bce_gain",
                "minimum_novelty",
                "reference",
                "fallback",
                "cache_scope",
            }
            if set(selection_raw) != expected_selection_keys:
                raise ValueError(
                    "latent_hard_path_selection keys must be exactly "
                    f"{sorted(expected_selection_keys)}"
                )
            if selection_raw["enabled"] is not True:
                raise ValueError("declared latent hard path selection must be enabled")
            if not self._path_t_values:
                raise ValueError("complementary selection requires a latent path bank")
            if selection_raw["method"] != "orthogonal_logit_residual_hardness":
                raise ValueError("A5 complementary path selection method is unsupported")
            if selection_raw["reference"] != "first_augmix_node_two_raw_chains":
                raise ValueError("A5 complementary path reference is unsupported")
            if selection_raw["fallback"] != "redistribute_vae_mass_to_raw_chains":
                raise ValueError("A5 complementary path fallback is unsupported")
            if selection_raw["cache_scope"] != "one_generate_call":
                raise ValueError("A5 complementary path selection must be batch-local")
            minimum_gain = float(selection_raw["minimum_bce_gain"])
            maximum_gain = float(selection_raw["maximum_bce_gain"])
            minimum_novelty = float(selection_raw["minimum_novelty"])
            if not (
                0.0 <= minimum_gain <= maximum_gain
                and 0.0 <= minimum_novelty <= 1.0
            ):
                raise ValueError("A5 complementary path thresholds are invalid")
            self._path_selection = dict(selection_raw)
        guided_raw = self.method.contracts.get("lhat_guided_augmix_selection")
        self._guided_selection: Mapping[str, Any] | None = None
        if guided_raw is not None:
            if not isinstance(guided_raw, Mapping):
                raise ValueError("lhat_guided_augmix_selection must be a mapping")
            expected_guided_keys = {
                "enabled",
                "method",
                "candidate_count",
                "candidate_topology",
                "selection_space",
                "alignment_weight",
                "diversity_weight",
                "minimum_alignment",
                "training_waveform_source",
                "unavailable_lhat_fallback",
                "cache_scope",
            }
            if set(guided_raw) != expected_guided_keys:
                raise ValueError(
                    "lhat_guided_augmix_selection keys must be exactly "
                    f"{sorted(expected_guided_keys)}"
                )
            if guided_raw["enabled"] is not True:
                raise ValueError("declared LHAT-guided AugMix selection must be enabled")
            if (
                guided_raw["method"]
                != "lhat_penultimate_feature_alignment_fixed_composition_coverage"
            ):
                raise ValueError("LHAT-guided AugMix selection method is unsupported")
            if (
                guided_raw["candidate_topology"]
                != "fixed20_composition_parameter_replay"
            ):
                raise ValueError("LHAT-guided AugMix candidate topology is unsupported")
            if guided_raw["selection_space"] != "efficientnet_penultimate_feature":
                raise ValueError("LHAT-guided AugMix selection space is unsupported")
            if guided_raw["training_waveform_source"] != "raw_augmix_only":
                raise ValueError("guided AugMix must train on raw AugMix waveforms only")
            if guided_raw["unavailable_lhat_fallback"] != "random_raw_augmix":
                raise ValueError("guided AugMix LHAT fallback is unsupported")
            if guided_raw["cache_scope"] != "one_generate_call":
                raise ValueError("guided AugMix state must be batch-local")
            candidate_count = int(guided_raw["candidate_count"])
            alignment_weight = float(guided_raw["alignment_weight"])
            diversity_weight = float(guided_raw["diversity_weight"])
            minimum_alignment = float(guided_raw["minimum_alignment"])
            if not (
                2 <= candidate_count <= 16
                and 0.0 <= alignment_weight <= 1.0
                and 0.0 <= diversity_weight <= 1.0
                and math.isclose(
                    alignment_weight + diversity_weight,
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1.0e-8,
                )
                and -1.0 <= minimum_alignment <= 1.0
            ):
                raise ValueError("guided AugMix candidate count or weights are invalid")
            if self._path_t_values or self._path_selection is not None:
                raise ValueError(
                    "guided raw AugMix and decoded latent-path training are mutually exclusive"
                )
            self._guided_selection = dict(guided_raw)
        hardness_raw = self.method.contracts.get("hardness_coupled_augmax")
        self._hardness_coupled_augmax: Mapping[str, Any] | None = None
        if hardness_raw is not None:
            if not isinstance(hardness_raw, Mapping):
                raise ValueError("hardness_coupled_augmax must be a mapping")
            expected_hardness_keys = {
                "enabled",
                "method",
                "candidates",
                "minimum_bce_gain",
                "maximum_bce_gain",
                "fallback",
                "cache_scope",
                "classifier_forward_chunk_size",
            }
            if set(hardness_raw) != expected_hardness_keys:
                raise ValueError(
                    "hardness_coupled_augmax keys must be exactly "
                    f"{sorted(expected_hardness_keys)}"
                )
            if hardness_raw["enabled"] is not True:
                raise ValueError("declared hardness-coupled AugMax must be enabled")
            if (
                hardness_raw["method"]
                != "hardest_per_record_bce_under_cap"
            ):
                raise ValueError("hardness-coupled AugMax method is unsupported")
            if hardness_raw["fallback"] != "classic_sampled_augmix":
                raise ValueError("hardness-coupled AugMax fallback is unsupported")
            if hardness_raw["cache_scope"] != "one_generate_call":
                raise ValueError(
                    "hardness-coupled AugMax clean reference must be batch-local"
                )
            if self.model_name != "efficientnet1dv2":
                raise ValueError(
                    "hardness-coupled AugMax currently supports EfficientNet only"
                )
            if (
                self._guided_selection is not None
                or self._path_selection is not None
                or self._path_t_values
            ):
                raise ValueError(
                    "hardness-coupled AugMax may not be combined with an older "
                    "guided or latent-path selector"
                )
            raw_candidates = hardness_raw["candidates"]
            if not isinstance(raw_candidates, Sequence) or isinstance(
                raw_candidates, (str, bytes)
            ):
                raise ValueError("hardness-coupled candidates must be a sequence")
            candidates: list[dict[str, Any]] = []
            for raw_candidate in raw_candidates:
                if not isinstance(raw_candidate, Mapping) or set(raw_candidate) != {
                    "weights",
                    "strength",
                }:
                    raise ValueError(
                        "each hardness-coupled candidate needs weights and strength"
                    )
                weights = tuple(float(value) for value in raw_candidate["weights"])
                strength = float(raw_candidate["strength"])
                if (
                    len(weights) != 3
                    or any(not math.isfinite(value) or value < 0.0 for value in weights)
                    or not math.isclose(
                        sum(weights), 1.0, rel_tol=0.0, abs_tol=1.0e-6
                    )
                    or not math.isfinite(strength)
                    or not 0.0 <= strength <= 1.0
                ):
                    raise ValueError(
                        "hardness-coupled candidates must be convex weights and "
                        "strength in [0,1]"
                    )
                candidates.append({"weights": weights, "strength": strength})
            if not 2 <= len(candidates) <= 8:
                raise ValueError(
                    "hardness-coupled AugMax requires 2-8 fixed candidates"
                )
            minimum_gain = float(hardness_raw["minimum_bce_gain"])
            maximum_gain = float(hardness_raw["maximum_bce_gain"])
            chunk_size = int(hardness_raw["classifier_forward_chunk_size"])
            if not (
                0.0 <= minimum_gain <= maximum_gain <= 1.0
                and 1 <= chunk_size <= 1024
            ):
                raise ValueError(
                    "hardness-coupled BCE bounds or classifier chunk size are invalid"
                )
            self._hardness_coupled_augmax = {
                **dict(hardness_raw),
                "candidates": tuple(candidates),
                "minimum_bce_gain": minimum_gain,
                "maximum_bce_gain": maximum_gain,
                "classifier_forward_chunk_size": chunk_size,
            }
        coverage_raw = self.method.contracts.get("canonical_chain1_coverage")
        self._canonical_chain1_coverage = False
        if coverage_raw is not None:
            if not isinstance(coverage_raw, Mapping):
                raise ValueError("canonical_chain1_coverage must be a mapping")
            expected_coverage_keys = {
                "enabled",
                "schedule",
                "chain1",
                "chain2",
                "chain3",
                "direct_fixed20_endpoint_supervision",
            }
            if set(coverage_raw) != expected_coverage_keys:
                raise ValueError(
                    "canonical_chain1_coverage keys must be exactly "
                    f"{sorted(expected_coverage_keys)}"
                )
            if coverage_raw != {
                "enabled": True,
                "schedule": "augmix_node_id_1_to_20",
                "chain1": "matching_canonical_depth23_composition",
                "chain2": "independent_random_depth23_composition",
                "chain3": "shared_vae_lhat_hard_view",
                "direct_fixed20_endpoint_supervision": False,
            }:
                raise ValueError(
                    "canonical_chain1_coverage contract is unsupported"
                )
            if self.method.profile_name not in PURE_M20_RUNTIME_METHOD_IDS:
                raise ValueError(
                    "canonical chain coverage is restricted to the pure method"
                )
            self._canonical_chain1_coverage = True
        fallback_raw = self.method.contracts.get("lhat_unavailable_fallback")
        self._lhat_unavailable_fallback = False
        if self.method.profile_name in PURE_M20_RUNTIME_METHOD_IDS:
            if fallback_raw != "augmix_only_chain3_clean":
                raise ValueError(
                    "pure M20 must preserve the matched supervised-view budget "
                    "with lhat_unavailable_fallback=augmix_only_chain3_clean"
                )
            self._lhat_unavailable_fallback = True
        elif fallback_raw is not None:
            raise ValueError(
                "lhat_unavailable_fallback is restricted to the pure M20 method"
            )
        self._latent_path_bank: _A5LatentPathBank | None = None
        self._complementary_path: _A5ComplementaryPath | None = None
        self._guided_state: _A5GuidedAugMixState | None = None
        self._hardness_clean_bce: torch.Tensor | None = None
        if not (
            0.0 <= self._strength_min <= self._strength_max <= 1.0
            and 0.0 <= self._vae_weight_cap <= 1.0
            and 0.0 <= self._vae_chain_probability <= 1.0
            and 0.0 <= self._canonical_chain1_weight_floor <= 1.0
        ):
            raise ValueError(
                "A5 calibrated mixing requires 0 <= strength_min <= "
                "strength_max <= 1, a VAE cap in [0,1], and a canonical "
                "chain1 floor in [0,1]"
            )
        if self._mixing_mode not in {"convex_capped", "sparse_onehot"}:
            raise ValueError("A5 mixing_mode must be convex_capped or sparse_onehot")

    def _build_latent_path_bank(
        self,
        source: WaveformView,
        hard: WaveformView,
        *,
        anchor_standardized: torch.Tensor,
        hard_standardized: torch.Tensor,
    ) -> _A5LatentPathBank | None:
        accepted_positions = tuple(
            int(value)
            for value in hard.metadata.get("accepted_positions", ())
        )
        candidate_positions = tuple(
            int(value)
            for value in hard.metadata.get("candidate_eligible_positions", ())
        )
        if not accepted_positions:
            return None
        local_by_batch_position = {
            batch_position: local_position
            for local_position, batch_position in enumerate(candidate_positions)
        }
        try:
            accepted_local = tuple(
                local_by_batch_position[position] for position in accepted_positions
            )
        except KeyError as exc:
            raise RuntimeError(
                "accepted LHAT position is absent from the candidate batch"
            ) from exc
        local = torch.as_tensor(
            accepted_local,
            device=anchor_standardized.device,
            dtype=torch.long,
        )
        batch_positions = torch.as_tensor(
            accepted_positions,
            device=source.waveform.device,
            dtype=torch.long,
        )
        anchor = anchor_standardized.index_select(0, local).float()
        endpoint = hard_standardized.index_select(0, local).float()
        projections = tuple(
            project_paired_latent_path(anchor, endpoint, t=value)
            for value in self._path_t_values
        )
        standardized_paths = torch.stack(
            tuple(value.standardized_latent for value in projections),
            dim=1,
        )
        standardizer = self.latent_pool.standardizer
        native = standardizer.inverse_transform(
            torch.cat(
                (
                    anchor,
                    standardized_paths.reshape(-1, 4, 128),
                ),
                dim=0,
            )
        )
        decoded, decoded_finite = _decode_latents_chunked(self.decoder, native)
        accepted_count = len(accepted_positions)
        clean_decoded = decoded[:accepted_count]
        decoded_paths = decoded[accepted_count:].reshape(
            accepted_count,
            len(self._path_t_values),
            1000,
            12,
        )
        decoded_path_finite = decoded_finite[accepted_count:].reshape(
            accepted_count,
            len(self._path_t_values),
        )
        clean_raw = source.waveform.index_select(0, batch_positions).float()
        hard_raw = hard.waveform.index_select(0, batch_positions).float()
        corrected = apply_linear_endpoint_residual_correction(
            decoded_paths,
            clean_raw=clean_raw,
            corrupt_raw=hard_raw,
            clean_decoded=clean_decoded,
            # LHAT's endpoint waveform is the deterministic decode of the
            # captured final latent, so its residual is exactly zero here.
            corrupt_decoded=hard_raw,
            t_values=self._path_t_values,
        )
        projection_finite = torch.stack(
            tuple(value.finite_mask for value in projections),
            dim=1,
        )
        finite = (
            decoded_path_finite
            & projection_finite
            & torch.isfinite(corrected).flatten(2).all(dim=2)
        )
        standard_deviation = corrected.flatten(2).std(dim=2, unbiased=False)
        maximum_absolute = corrected.abs().flatten(2).amax(dim=2)
        valid = (
            finite
            & (standard_deviation >= self.minimum_std_mV)
            & (maximum_absolute <= self.maximum_abs_mV)
        )
        # t=1 is residual-corrected to the already accepted LHAT waveform.
        # Keep this fail-closed assertion so an invalid bank never reaches an
        # AugMix node silently.
        if not bool(valid.any(dim=1).all()):
            raise RuntimeError("accepted LHAT endpoint produced no valid path point")
        return _A5LatentPathBank(
            accepted_positions=accepted_positions,
            waveforms=torch.nan_to_num(corrected).contiguous(),
            valid_mask=valid.contiguous(),
            t_values=torch.tensor(
                self._path_t_values,
                device=corrected.device,
                dtype=torch.float32,
            ),
        )

    def _classifier_logits_and_features(
        self,
        classifier: nn.Module,
        waveforms: torch.Tensor,
        *,
        chunk_size: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if waveforms.ndim != 3 or tuple(waveforms.shape[1:]) != (1000, 12):
            raise ValueError("A5 selection waveforms must have shape (N,1000,12)")
        if chunk_size <= 0:
            raise ValueError("A5 classifier chunk_size must be positive")
        if self.model_name != "efficientnet1dv2":
            raise ValueError("A5 penultimate-feature selection supports EfficientNet only")
        head_container = getattr(classifier, "classifier", None)
        if not isinstance(head_container, nn.Sequential) or not isinstance(
            head_container[-1], nn.Linear
        ):
            raise TypeError("managed EfficientNet classifier head is unavailable")
        model_spec = get_model_spec(self.model_name)
        outputs: list[torch.Tensor] = []
        features: list[torch.Tensor] = []

        def capture(
            _module: nn.Module, arguments: tuple[torch.Tensor, ...]
        ) -> None:
            if len(arguments) != 1 or arguments[0].ndim != 2:
                raise RuntimeError("EfficientNet head received unexpected features")
            features.append(arguments[0].detach().float())

        handle = head_container[-1].register_forward_pre_hook(capture)
        try:
            with _frozen_classifier_for_attack(classifier), torch.no_grad():
                for start in range(0, int(waveforms.shape[0]), int(chunk_size)):
                    batch = waveforms[start : start + int(chunk_size)]
                    with torch.autocast(
                        device_type=batch.device.type,
                        dtype=torch.bfloat16,
                        enabled=batch.device.type == "cuda",
                    ):
                        logits = validate_model_output(
                            classifier(
                                prepare_canonical_model_input(
                                    batch,
                                    model_spec,
                                    epsilon=(
                                        1.0e-6
                                        if self.lhat_config is None
                                        else self.lhat_config.normalization_epsilon
                                    ),
                                )
                            ),
                            model_spec,
                            batch_size=int(batch.shape[0]),
                            check_finite=False,
                        ).float()
                    outputs.append(logits)
        finally:
            handle.remove()
        if len(features) != len(outputs):
            raise RuntimeError("EfficientNet feature capture lost chunk alignment")
        return (
            torch.cat(outputs, dim=0).contiguous(),
            torch.cat(features, dim=0).contiguous(),
        )

    def _classifier_logits(
        self,
        classifier: nn.Module,
        waveforms: torch.Tensor,
        *,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        logits, _ = self._classifier_logits_and_features(
            classifier, waveforms, chunk_size=chunk_size
        )
        return logits

    def _select_hardness_coupled_augmix(
        self,
        *,
        clean: torch.Tensor,
        chain1: torch.Tensor,
        chain2: torch.Tensor,
        lhat: torch.Tensor,
        labels: torch.Tensor,
        clean_bce: torch.Tensor,
        classic_weights: torch.Tensor,
        classic_strength: torch.Tensor,
        classifier: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        """Select a bounded hard convex mixture without adding a new data source."""

        config = self._hardness_coupled_augmax
        if config is None:
            raise RuntimeError("hardness-coupled AugMax was not configured")
        accepted = int(clean.shape[0])
        expected_waveform_shape = (accepted, 1000, 12)
        if any(
            value.shape != expected_waveform_shape
            for value in (clean, chain1, chain2, lhat)
        ):
            raise ValueError("hardness-coupled AugMax waveforms lost alignment")
        if (
            labels.shape != (accepted, 5)
            or clean_bce.shape != (accepted,)
            or classic_weights.shape != (accepted, 3)
            or classic_strength.shape != (accepted,)
        ):
            raise ValueError("hardness-coupled AugMax metadata lost alignment")
        fixed = tuple(config["candidates"])
        fixed_weights = torch.tensor(
            tuple(candidate["weights"] for candidate in fixed),
            device=clean.device,
            dtype=torch.float32,
        )
        fixed_strength = torch.tensor(
            tuple(candidate["strength"] for candidate in fixed),
            device=clean.device,
            dtype=torch.float32,
        )
        weights = torch.cat(
            (
                classic_weights[:, None].float(),
                fixed_weights[None].expand(accepted, -1, -1),
            ),
            dim=1,
        )
        strengths = torch.cat(
            (
                classic_strength[:, None].float(),
                fixed_strength[None].expand(accepted, -1),
            ),
            dim=1,
        )
        chains = torch.stack((chain1.float(), chain2.float(), lhat.float()), dim=1)
        mixtures = (
            weights[:, :, :, None, None] * chains[:, None]
        ).sum(dim=2)
        candidates = clean[:, None].float() + strengths[:, :, None, None] * (
            mixtures - clean[:, None].float()
        )
        candidate_count = int(candidates.shape[1])
        finite = torch.isfinite(candidates).flatten(2).all(dim=2)
        standard_deviation = candidates.flatten(2).std(dim=2, unbiased=False)
        maximum_absolute = candidates.abs().flatten(2).amax(dim=2)
        quality_valid = (
            finite
            & (standard_deviation >= self.minimum_std_mV)
            & (maximum_absolute <= self.maximum_abs_mV)
        )
        candidate_logits = self._classifier_logits(
            classifier,
            candidates.reshape(accepted * candidate_count, 1000, 12),
            chunk_size=int(config["classifier_forward_chunk_size"]),
        ).reshape(accepted, candidate_count, 5)
        candidate_bce = F.binary_cross_entropy_with_logits(
            candidate_logits,
            labels[:, None].float().expand(-1, candidate_count, -1),
            reduction="none",
        ).mean(dim=2)
        gains = candidate_bce - clean_bce[:, None]
        admissible = (
            quality_valid
            & torch.isfinite(gains)
            & (gains >= float(config["minimum_bce_gain"]))
            & (gains <= float(config["maximum_bce_gain"]))
        )
        scores = torch.where(
            admissible,
            gains,
            torch.full_like(gains, -torch.inf),
        )
        selected = scores.argmax(dim=1)
        available = admissible.any(dim=1)
        selected = torch.where(available, selected, torch.zeros_like(selected))
        rows = torch.arange(accepted, device=clean.device, dtype=torch.long)
        selected_waveform = candidates[rows, selected]
        selected_weights = weights[rows, selected]
        selected_strength = strengths[rows, selected]
        selected_gain = gains[rows, selected]
        selected_valid = quality_valid[rows, selected]
        diagnostics = {
            "hardness_candidate_count": float(candidate_count),
            "hardness_available_fraction": float(available.float().mean().item()),
            "hardness_selected_nonclassic_fraction": float(
                (selected != 0).float().mean().item()
            ),
            "hardness_selected_bce_gain": float(
                selected_gain[selected_valid].mean().item()
                if bool(selected_valid.any())
                else 0.0
            ),
        }
        return (
            selected_waveform.contiguous(),
            selected_weights.contiguous(),
            selected_strength.contiguous(),
            diagnostics,
        )

    def _select_path_against_augmix(
        self,
        *,
        clean: torch.Tensor,
        raw_chain1: torch.Tensor,
        raw_chain2: torch.Tensor,
        labels: torch.Tensor,
        classifier: nn.Module,
    ) -> _A5ComplementaryPath:
        if self._latent_path_bank is None or self._path_selection is None:
            raise RuntimeError("A5 complementary selection resources are unavailable")
        bank = self._latent_path_bank
        accepted = int(bank.waveforms.shape[0])
        if (
            clean.shape != (accepted, 1000, 12)
            or raw_chain1.shape != clean.shape
            or raw_chain2.shape != clean.shape
            or labels.shape != (accepted, 5)
        ):
            raise ValueError("A5 complementary selection inputs lost alignment")
        candidate_count = int(bank.waveforms.shape[1])
        stacked = torch.cat(
            (
                clean.float(),
                raw_chain1.float(),
                raw_chain2.float(),
                bank.waveforms.reshape(accepted * candidate_count, 1000, 12).float(),
            ),
            dim=0,
        )
        logits = self._classifier_logits(classifier, stacked)
        clean_logits = logits[:accepted]
        raw_logits = torch.stack(
            (logits[accepted : 2 * accepted], logits[2 * accepted : 3 * accepted]),
            dim=1,
        )
        candidate_logits = logits[3 * accepted :].reshape(
            accepted, candidate_count, 5
        )
        repeated_labels = labels[:, None].expand(-1, candidate_count, -1)
        clean_bce = F.binary_cross_entropy_with_logits(
            clean_logits,
            labels.float(),
            reduction="none",
        ).mean(dim=1)
        candidate_bce = F.binary_cross_entropy_with_logits(
            candidate_logits,
            repeated_labels.float(),
            reduction="none",
        ).mean(dim=2)
        bce_gain = candidate_bce - clean_bce[:, None]
        novelty = _orthogonal_logit_novelty(
            clean_logits,
            raw_logits,
            candidate_logits,
        )
        finite = (
            torch.isfinite(clean_logits).all(dim=1)[:, None]
            & torch.isfinite(raw_logits).all(dim=2).all(dim=1)[:, None]
            & torch.isfinite(candidate_logits).all(dim=2)
        )
        selected, available = _select_complementary_path_indices(
            bank.valid_mask & finite,
            bce_gain,
            novelty,
            minimum_bce_gain=float(self._path_selection["minimum_bce_gain"]),
            maximum_bce_gain=float(self._path_selection["maximum_bce_gain"]),
            minimum_novelty=float(self._path_selection["minimum_novelty"]),
        )
        rows = torch.arange(accepted, device=clean.device, dtype=torch.long)
        selected_waveforms = bank.waveforms[rows, selected]
        selected_t = bank.t_values.index_select(0, selected)
        selected_gain = bce_gain[rows, selected]
        selected_novelty = novelty[rows, selected]
        selected_waveforms = torch.where(
            available[:, None, None], selected_waveforms, clean.float()
        )
        selected_t = torch.where(available, selected_t, torch.zeros_like(selected_t))
        selected_gain = torch.where(
            available, selected_gain, torch.zeros_like(selected_gain)
        )
        selected_novelty = torch.where(
            available, selected_novelty, torch.zeros_like(selected_novelty)
        )
        return _A5ComplementaryPath(
            waveforms=selected_waveforms.contiguous(),
            t_values=selected_t.contiguous(),
            available_mask=available.contiguous(),
            bce_gain=selected_gain.contiguous(),
            novelty=selected_novelty.contiguous(),
        )

    @staticmethod
    def _redistribute_unavailable_vae_mass(
        weights: torch.Tensor,
        available: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if weights.ndim != 2 or weights.shape[1] != 3:
            raise ValueError("A5 mixture weights must have shape (B,3)")
        if available.shape != (weights.shape[0],) or available.dtype != torch.bool:
            raise ValueError("A5 VAE availability must be boolean shape (B,)")
        unavailable = ~available
        if not bool(unavailable.any()):
            return weights
        result = weights.clone()
        raw = result[unavailable, :2]
        raw_sum = raw.sum(dim=1, keepdim=True)
        empty = raw_sum.squeeze(1) <= 1.0e-12
        raw = raw / raw_sum.clamp_min(1.0e-12)
        if bool(empty.any()):
            chosen = torch.randint(
                0,
                2,
                (int(empty.sum().item()),),
                device=weights.device,
                generator=generator,
            )
            raw[empty] = torch.zeros(
                int(empty.sum().item()), 2, device=weights.device, dtype=weights.dtype
            ).scatter_(1, chosen[:, None], 1.0)
        result[unavailable, :2] = raw
        result[unavailable, 2] = 0.0
        return result.contiguous()

    def _build_guided_augmix_state(
        self,
        *,
        clean: WaveformView,
        hard: WaveformView,
        classifier: nn.Module,
    ) -> _A5GuidedAugMixState:
        if clean.sample_ids != hard.sample_ids or not torch.equal(
            clean.labels, hard.labels
        ):
            raise RuntimeError("guided AugMix inputs lost origin/label alignment")
        logits, features = self._classifier_logits_and_features(
            classifier,
            torch.cat((clean.waveform.float(), hard.waveform.float()), dim=0),
        )
        batch = clean.batch_size
        clean_logits = logits[:batch]
        clean_features = features[:batch]
        hard_features = features[batch:]
        direction = hard_features - clean_features
        available = (
            hard.valid_mask
            & torch.isfinite(direction).all(dim=1)
            & (direction.norm(dim=1) > 1.0e-6)
        )
        direction = torch.where(
            available[:, None], direction, torch.zeros_like(direction)
        )
        return _A5GuidedAugMixState(
            clean_logits=clean_logits.detach().contiguous(),
            clean_features=clean_features.detach().contiguous(),
            lhat_direction=direction.detach().contiguous(),
            available_mask=available.detach().contiguous(),
            selected_directions=[],
        )

    def _guided_raw_augmix(
        self,
        context: NodeContext,
        clean: WaveformView,
        hard: WaveformView,
    ) -> WaveformView:
        if self._guided_selection is None or self.augmix_config is None:
            raise RuntimeError("guided raw AugMix was not configured")
        if self._guided_state is None:
            self._guided_state = self._build_guided_augmix_state(
                clean=clean,
                hard=hard,
                classifier=context.resource("classifier"),
            )
        state = self._guided_state
        batch = clean.batch_size
        candidate_count = int(self._guided_selection["candidate_count"])
        generator = context.torch_generator(
            "guided_raw_candidates", device=clean.waveform.device
        )
        expanded_clean = (
            clean.waveform[:, None]
            .expand(-1, candidate_count, -1, -1)
            .reshape(batch * candidate_count, 1000, 12)
            .float()
            .contiguous()
        )
        try:
            composition_index = int(context.node_id.rsplit("_", 1)[1]) - 1
        except (IndexError, ValueError) as exc:
            raise ValueError(
                "guided AugMix node id must end with a 1-based composition index"
            ) from exc
        if not 0 <= composition_index < 20:
            raise ValueError("guided AugMix composition index must lie in [0,19]")
        composition_indices = torch.full(
            (batch * candidate_count,),
            composition_index,
            device=clean.waveform.device,
            dtype=torch.int64,
        )
        corruption = generate_canonical_corruption(
            expanded_clean,
            operator_params=_load_operator_profile(self.augmix_config),
            generator=generator,
            composition_indices=composition_indices,
            _input_prevalidated=True,
        )
        if not bool(
            (corruption.diagnostics.composition_index == composition_index).all()
        ):
            raise RuntimeError("guided AugMix composition coverage drifted")
        candidates_full_strength = corruption.waveform_raw_100hz.reshape(
            batch, candidate_count, 1000, 12
        )
        depths = corruption.diagnostics.depth.reshape(batch, candidate_count)
        nonfinite = corruption.diagnostics.output_nonfinite_count.reshape(
            batch, candidate_count
        )
        if math.isclose(
            self._strength_min,
            self._strength_max,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            strength = torch.full(
                (batch, candidate_count),
                self._strength_min,
                device=clean.waveform.device,
                dtype=torch.float32,
            )
        else:
            strength = self._strength_min + (
                self._strength_max - self._strength_min
            ) * torch.rand(
                batch,
                candidate_count,
                device=clean.waveform.device,
                dtype=torch.float32,
                generator=generator,
            )
        candidates = clean.waveform[:, None].float() + strength[:, :, None, None] * (
            candidates_full_strength - clean.waveform[:, None].float()
        )
        valid_candidates = (
            (nonfinite == 0)
            & torch.isfinite(candidates).flatten(2).all(dim=2)
        )
        candidate_logits, candidate_features = self._classifier_logits_and_features(
            context.resource("classifier"),
            candidates.reshape(batch * candidate_count, 1000, 12),
        )
        candidate_logits = candidate_logits.reshape(batch, candidate_count, 5)
        candidate_features = candidate_features.reshape(
            batch, candidate_count, -1
        )
        candidate_directions = candidate_features - state.clean_features[:, None]
        finite_outputs = (
            torch.isfinite(candidate_logits).all(dim=2)
            & torch.isfinite(candidate_features).all(dim=2)
        )
        valid_candidates = valid_candidates & finite_outputs
        previous = (
            None
            if not state.selected_directions
            else torch.stack(state.selected_directions, dim=1)
        )
        selected, selected_alignment, selected_diversity = (
            _select_lhat_guided_augmix_indices(
                state.lhat_direction,
                candidate_directions,
                previous,
                state.available_mask,
                valid_candidates,
                alignment_weight=float(
                    self._guided_selection["alignment_weight"]
                ),
                diversity_weight=float(
                    self._guided_selection["diversity_weight"]
                ),
                minimum_alignment=float(
                    self._guided_selection["minimum_alignment"]
                ),
            )
        )
        rows = torch.arange(batch, device=clean.waveform.device, dtype=torch.long)
        selected_waveform = candidates[rows, selected].contiguous()
        selected_direction = candidate_directions[rows, selected].detach().contiguous()
        state.selected_directions.append(selected_direction)
        selected_valid = valid_candidates[rows, selected] & clean.valid_mask
        labels = clean.labels.float()
        clean_bce = F.binary_cross_entropy_with_logits(
            state.clean_logits, labels, reduction="none"
        ).mean(dim=1)
        selected_bce = F.binary_cross_entropy_with_logits(
            candidate_logits[rows, selected], labels, reduction="none"
        ).mean(dim=1)
        selected_gain = selected_bce - clean_bce
        selected_depth = depths[rows, selected].float()
        guided = state.available_mask & selected_valid
        diagnostics = {
            "accepted_count": float(selected_valid.sum().item()),
            "candidate_count": float(candidate_count),
            "composition_index": float(composition_index),
            "lhat_guided_fraction": float(guided.float().mean().item()),
            "selected_chain_depth": float(selected_depth[selected_valid].mean().item()),
            "selected_bce_gain": float(selected_gain[selected_valid].mean().item()),
            "selected_lhat_alignment": float(
                selected_alignment[guided].mean().item()
                if bool(guided.any())
                else 0.0
            ),
            "selected_direction_diversity": float(
                selected_diversity[guided].mean().item()
                if bool(guided.any())
                else 0.0
            ),
            "selected_invalid_fraction": float((~selected_valid).float().mean().item()),
            "vae_waveform_training_weight": 0.0,
            "strength": float(strength[rows, selected].mean().item()),
        }
        for name, value in diagnostics.items():
            context.record_diagnostic(name, value)
        return WaveformView(
            name=context.node_id,
            waveform=selected_waveform.to(dtype=clean.waveform.dtype),
            labels=clean.labels,
            sample_ids=clean.sample_ids,
            valid_mask=selected_valid.contiguous(),
            provenance=Provenance(
                node_id=context.node_id,
                operation="lhat_guided_diverse_low_alpha_augmix",
                parent_names=(clean.name, hard.name),
                rng_namespace=context.rng_namespace,
                parameters={
                    "candidate_count": candidate_count,
                    "candidate_topology": "four_parameter_replays_of_one_fixed20_composition",
                    "candidate_mixing": "low_alpha_onehot_full_strength",
                    "composition_index": composition_index,
                    "selection": "lhat_penultimate_feature_alignment_fixed_composition_coverage",
                    "selection_space": "efficientnet_penultimate_feature",
                    "alignment_weight": float(
                        self._guided_selection["alignment_weight"]
                    ),
                    "diversity_weight": float(
                        self._guided_selection["diversity_weight"]
                    ),
                    "minimum_alignment": float(
                        self._guided_selection["minimum_alignment"]
                    ),
                    "bce_gain_used_for_selection": False,
                    "training_waveform_source": "raw_augmix_only",
                    "unavailable_lhat_fallback": "random_raw_augmix",
                },
            ),
            metadata={"diagnostic_means": diagnostics},
        )

    def _lhat_attack(
        self,
        context: NodeContext,
        inputs: tuple[ViewValue, ...],
    ) -> ViewValue:
        if not self._path_t_values:
            hard = super()._lhat_attack(context, inputs)
            return self._replace_lhat_with_clean_for_matched_control(
                hard,
                inputs,
            )
        if self._latent_path_bank is not None:
            raise RuntimeError("A5 LHAT path bank was materialized twice")
        captured: dict[str, Any] = {}
        original = core_method_runtime.generate_lhat_adversarial

        def capture_lhat(**kwargs: Any):
            result = original(**kwargs)
            captured["anchor_standardized"] = kwargs[
                "anchor_standardized"
            ].detach()
            captured["result"] = result
            return result

        core_method_runtime.generate_lhat_adversarial = capture_lhat
        try:
            hard = super()._lhat_attack(context, inputs)
        finally:
            core_method_runtime.generate_lhat_adversarial = original
        if not isinstance(hard, WaveformView):
            raise TypeError("A5 LHAT node did not return a WaveformView")
        result = captured.get("result")
        anchor_standardized = captured.get("anchor_standardized")
        if result is None or not isinstance(anchor_standardized, torch.Tensor):
            if hard.metadata.get("candidate_eligible_positions"):
                raise RuntimeError("A5 failed to capture LHAT latent endpoints")
            return self._replace_lhat_with_clean_for_matched_control(
                hard,
                inputs,
            )
        if len(inputs) != 1 or not isinstance(inputs[0], WaveformView):
            raise TypeError("A5 LHAT path requires one clean WaveformView")
        self._latent_path_bank = self._build_latent_path_bank(
            inputs[0],
            hard,
            anchor_standardized=anchor_standardized,
            hard_standardized=result.latent_standardized,
        )
        return self._replace_lhat_with_clean_for_matched_control(
            hard,
            inputs,
        )

    def _replace_lhat_with_clean_for_matched_control(
        self,
        hard: ViewValue,
        inputs: tuple[ViewValue, ...],
    ) -> ViewValue:
        """Preserve full LHAT compute/QC but hide its waveform from control loss."""

        if not self._matched_clean_chain3_control:
            return hard
        if not isinstance(hard, WaveformView):
            raise TypeError("bounded AugMax LHAT control requires a WaveformView")
        if len(inputs) != 1 or not isinstance(inputs[0], WaveformView):
            raise TypeError(
                "bounded AugMax LHAT control requires one clean WaveformView"
            )
        clean = inputs[0]
        if clean.sample_ids != hard.sample_ids or not torch.equal(
            clean.labels, hard.labels
        ):
            raise RuntimeError(
                "bounded AugMax LHAT control lost origin/label alignment"
            )
        return replace(
            hard,
            waveform=clean.waveform.clone().contiguous(),
            provenance=replace(
                hard.provenance,
                operation="full_lhat_search_then_clean_chain3_control",
                parameters={
                    **dict(hard.provenance.parameters),
                    "waveform_replaced_after_full_search_and_qc": True,
                    "replacement_source": clean.name,
                },
            ),
            metadata={
                **dict(hard.metadata),
                "matched_chain3_control_mode": (
                    "replace_lhat_with_clean_after_full_search_and_qc"
                ),
                "full_lhat_search_executed": True,
            },
        )

    def _augmix(
        self, context: NodeContext, inputs: tuple[ViewValue, ...]
    ) -> ViewValue:
        if (
            len(inputs) != 2
            or not isinstance(inputs[0], WaveformView)
            or not isinstance(inputs[1], WaveformView)
        ):
            raise TypeError("A5 calibrated AugMix requires clean and LHAT views")
        if self.augmix_config is None:
            raise RuntimeError("A5 calibrated AugMix has no resolved config")
        clean, hard = inputs
        if clean.sample_ids != hard.sample_ids or not torch.equal(
            clean.labels, hard.labels
        ):
            raise RuntimeError("A5 calibrated inputs lost origin/label alignment")
        if self._guided_selection is not None:
            return self._guided_raw_augmix(context, clean, hard)
        lhat_available_positions = tuple(
            int(value)
            for value in torch.nonzero(hard.valid_mask, as_tuple=False)
            .flatten()
            .detach()
            .cpu()
            .tolist()
        )
        accepted_positions = (
            tuple(
                int(value)
                for value in torch.nonzero(clean.valid_mask, as_tuple=False)
                .flatten()
                .detach()
                .cpu()
                .tolist()
            )
            if self._lhat_unavailable_fallback
            else lhat_available_positions
        )
        full_waveform = clean.waveform.clone()
        diagnostics: dict[str, float] = {}
        if accepted_positions:
            indices = torch.as_tensor(
                accepted_positions, device=clean.waveform.device, dtype=torch.long
            )
            generator = context.torch_generator(
                "chains_and_mixing", device=clean.waveform.device
            )
            selected_hard = hard.waveform.index_select(0, indices)
            selected_t: torch.Tensor | None = None
            if self._latent_path_bank is not None:
                if accepted_positions != self._latent_path_bank.accepted_positions:
                    raise RuntimeError("A5 latent path bank lost sample alignment")
                path_generator = context.torch_generator(
                    "latent_path_choice", device=clean.waveform.device
                )
                selected_hard, selected_t = self._latent_path_bank.sample(
                    generator=path_generator
                )
            selected_clean = clean.waveform.index_select(0, indices).float()
            result = generate_three_chain_augmix(
                selected_clean,
                selected_hard,
                sampling_rate_hz=100,
                config=self.augmix_config,
                generator=generator,
                include_normalized=False,
            )
            chain1_raw = result.chain1_raw
            chain1_depth = result.chain1_depth
            chain1_composition_index = result.chain1_composition_index
            if self._canonical_chain1_coverage:
                try:
                    canonical_index = int(context.node_id.rsplit("_", 1)[1]) - 1
                except (IndexError, ValueError) as exc:
                    raise ValueError(
                        "canonical-coverage AugMix node id must end with a "
                        "1-based composition index"
                    ) from exc
                if not 0 <= canonical_index < 20:
                    raise ValueError(
                        "canonical-coverage composition index must lie in [0,19]"
                    )
                fixed_indices = torch.full(
                    (len(accepted_positions),),
                    canonical_index,
                    device=selected_clean.device,
                    dtype=torch.int64,
                )
                canonical = generate_canonical_corruption(
                    selected_clean,
                    operator_params=_load_operator_profile(self.augmix_config),
                    generator=generator,
                    composition_indices=fixed_indices,
                    _input_prevalidated=True,
                )
                if not bool(
                    (canonical.diagnostics.composition_index == canonical_index).all()
                ):
                    raise RuntimeError(
                        "canonical chain1 composition coverage drifted"
                    )
                chain1_raw = canonical.waveform_raw_100hz
                chain1_depth = canonical.diagnostics.depth
                chain1_composition_index = canonical.diagnostics.composition_index
            complementary_available: torch.Tensor | None = None
            if self._path_selection is not None:
                if self._complementary_path is None:
                    self._complementary_path = self._select_path_against_augmix(
                        clean=clean.waveform.index_select(0, indices),
                        raw_chain1=chain1_raw,
                        raw_chain2=result.chain2_raw,
                        labels=clean.labels.index_select(0, indices),
                        classifier=context.resource("classifier"),
                    )
                selected_hard = self._complementary_path.waveforms
                selected_t = self._complementary_path.t_values
                complementary_available = self._complementary_path.available_mask
            weights = (
                _sparse_augmix_weights(
                    result.mixture_weights,
                    self._vae_chain_probability,
                    generator=generator,
                )
                if self._mixing_mode == "sparse_onehot"
                else _cap_vae_chain_weight(
                    result.mixture_weights, self._vae_weight_cap
                )
            )
            if complementary_available is not None:
                weights = self._redistribute_unavailable_vae_mass(
                    weights,
                    complementary_available,
                    generator=generator,
                )
            if self._canonical_chain1_weight_floor > 0.0:
                weights = _floor_canonical_chain1_weight(
                    weights, self._canonical_chain1_weight_floor
                )
            mixture = (
                weights[:, 0, None, None] * chain1_raw
                + weights[:, 1, None, None] * result.chain2_raw
                + weights[:, 2, None, None] * selected_hard
            )
            strength = self._strength_min + (
                self._strength_max - self._strength_min
            ) * result.augmented_strength
            mixed = selected_clean + strength[:, None, None] * (
                mixture - selected_clean
            )
            hardness_diagnostics: dict[str, float] = {}
            if self._hardness_coupled_augmax is not None:
                if self._hardness_clean_bce is None:
                    clean_logits = self._classifier_logits(
                        context.resource("classifier"),
                        clean.waveform.float(),
                        chunk_size=int(
                            self._hardness_coupled_augmax[
                                "classifier_forward_chunk_size"
                            ]
                        ),
                    )
                    self._hardness_clean_bce = F.binary_cross_entropy_with_logits(
                        clean_logits,
                        clean.labels.float(),
                        reduction="none",
                    ).mean(dim=1).detach().contiguous()
                mixed, weights, strength, hardness_diagnostics = (
                    self._select_hardness_coupled_augmix(
                        clean=selected_clean,
                        chain1=chain1_raw,
                        chain2=result.chain2_raw,
                        lhat=selected_hard,
                        labels=clean.labels.index_select(0, indices),
                        clean_bce=self._hardness_clean_bce.index_select(0, indices),
                        classic_weights=weights,
                        classic_strength=strength,
                        classifier=context.resource("classifier"),
                    )
                )
            full_waveform.index_copy_(0, indices, mixed)
            values = torch.stack(
                (
                    result.chain1_depth.float().mean(),
                    chain1_depth.float().mean(),
                    result.chain2_depth.float().mean(),
                    weights[:, 0].mean(),
                    weights[:, 2].mean(),
                    strength.mean(),
                )
            ).detach().cpu().tolist()
            diagnostics = dict(
                zip(
                    (
                        "discarded_random_chain1_depth",
                        "chain1_depth",
                        "chain2_depth",
                        "canonical_chain1_weight",
                        "vae_weight",
                        "strength",
                    ),
                    (float(value) for value in values),
                    strict=True,
                )
            )
            if self._canonical_chain1_coverage:
                diagnostics["chain1_composition_index"] = float(
                    chain1_composition_index[0].item()
                )
            diagnostics.update(hardness_diagnostics)
            if selected_t is not None:
                diagnostics["latent_path_t"] = float(
                    selected_t.mean().detach().cpu()
                )
                diagnostics["latent_path_unique_t"] = float(
                    selected_t.unique().numel()
                )
            if self._complementary_path is not None:
                available = self._complementary_path.available_mask
                diagnostics["complementary_available_fraction"] = float(
                    available.float().mean().detach().cpu()
                )
                if bool(available.any()):
                    diagnostics["complementary_bce_gain"] = float(
                        self._complementary_path.bce_gain[available]
                        .mean()
                        .detach()
                        .cpu()
                    )
                    diagnostics["complementary_novelty"] = float(
                        self._complementary_path.novelty[available]
                        .mean()
                        .detach()
                        .cpu()
                    )
                else:
                    diagnostics["complementary_bce_gain"] = 0.0
                    diagnostics["complementary_novelty"] = 0.0
            for name, value in diagnostics.items():
                context.record_diagnostic(name, value)
        context.record_diagnostic("accepted_count", len(accepted_positions))
        context.record_diagnostic(
            "lhat_available_count", len(lhat_available_positions)
        )
        context.record_diagnostic(
            "lhat_fallback_fraction",
            (
                float(len(accepted_positions) - len(lhat_available_positions))
                / float(len(accepted_positions))
                if accepted_positions
                else 0.0
            ),
        )
        return WaveformView(
            name=context.node_id,
            waveform=full_waveform.contiguous(),
            labels=clean.labels,
            sample_ids=clean.sample_ids,
            valid_mask=(
                clean.valid_mask.clone()
                if self._lhat_unavailable_fallback
                else hard.valid_mask.clone()
            ),
            provenance=Provenance(
                node_id=context.node_id,
                operation=(
                    "pure_m20_lhat_hardness_coupled_augmax"
                    if self._hardness_coupled_augmax is not None
                    else "pure_m20_lhat_threechain_augmix"
                    if self.method.profile_name in PURE_M20_RUNTIME_METHOD_IDS
                    else "a5_m20_severity_preserving_threechain_augmix"
                ),
                parent_names=(clean.name, hard.name),
                rng_namespace=context.rng_namespace,
                parameters={
                    "chain3_source": hard.name,
                    "strength_min": self._strength_min,
                    "strength_max": self._strength_max,
                    "vae_chain_weight_cap": self._vae_weight_cap,
                    "vae_chain_probability": self._vae_chain_probability,
                    "canonical_chain1_weight_floor": (
                        self._canonical_chain1_weight_floor
                    ),
                    "mixing": self._mixing_mode,
                    "latent_path_t_values": list(self._path_t_values),
                    "latent_path_sampling": (
                        "uniform_valid_per_augmix_node"
                        if self._path_t_values
                        else "disabled"
                    ),
                    "latent_path_selection": (
                        None
                        if self._path_selection is None
                        else dict(self._path_selection)
                    ),
                    "hardness_coupled_augmax": (
                        None
                        if self._hardness_coupled_augmax is None
                        else dict(self._hardness_coupled_augmax)
                    ),
                    "canonical_chain1_coverage": self._canonical_chain1_coverage,
                    "lhat_unavailable_fallback": (
                        "augmix_only_chain3_clean"
                        if self._lhat_unavailable_fallback
                        else None
                    ),
                },
            ),
            metadata={
                "accepted_positions": accepted_positions,
                "lhat_available_positions": lhat_available_positions,
                "diagnostic_means": diagnostics,
            },
        )

    def generate(
        self,
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
        if (
            self._latent_path_bank is not None
            or self._complementary_path is not None
            or self._guided_state is not None
            or self._hardness_clean_bce is not None
        ):
            raise RuntimeError("A5 latent path runtime generation is not re-entrant")
        try:
            return super().generate(
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
        finally:
            self._latent_path_bank = None
            self._complementary_path = None
            self._guided_state = None
            self._hardness_clean_bce = None

class _J4MatchedEligibilityRuntime(MethodViewRuntime):
    """Keep allowlisted candidate/control auxiliary support record-identical.

    The candidate preserves the whitelist LHAT runtime semantics and fails
    closed if waveform QC rejects any eligible endpoint. The control executes
    its ordinary identity graph first, then applies the candidate pool's
    deterministic exact-label eligibility mask to the two auxiliary views.
    Waveforms, labels, record order, and stochastic traces are not regenerated.
    """

    _CANDIDATE_ID = J4_VAE_LHAT_CHAIN3_METHOD_ID
    _CONTROL_ID = J4_CLEAN_CHAIN3_CONTROL_METHOD_ID
    _DISTILL_CANDIDATE_ID = VAE_LHAT_TEACHER_DISTILL_METHOD_ID
    _DISTILL_CONTROL_ID = CLEAN_TEACHER_DISTILL_CONTROL_METHOD_ID
    _HARD_BCE_CANDIDATE_ID = VAE_LHAT_HARD_BCE_METHOD_ID
    _HARD_BCE_CONTROL_ID = CLEAN_HARD_BCE_CONTROL_METHOD_ID
    _PATH_BCE_CANDIDATE_ID = VAE_LHAT_PATH_BCE_METHOD_ID
    _PATH_BCE_CONTROL_ID = CLEAN_LHAT_PATH_CONTROL_METHOD_ID
    _CONSISTENCY_CANDIDATE_ID = VAE_LHAT_CONSISTENCY_METHOD_ID
    _CONSISTENCY_CONTROL_ID = CLEAN_CONSISTENCY_CONTROL_METHOD_ID
    _FEATURE_CANDIDATE_ID = VAE_LHAT_FEATURE_INVARIANCE_METHOD_ID
    _FEATURE_CONTROL_ID = CLEAN_FEATURE_INVARIANCE_CONTROL_METHOD_ID
    _ANCHOR_SOFT_CANDIDATE_ID = VAE_LHAT_LOCAL_ANCHOR_SOFT_METHOD_ID
    _ANCHOR_SOFT_CONTROL_ID = CLEAN_LOCAL_ANCHOR_SOFT_CONTROL_METHOD_ID
    _CALIBRATED_ANCHOR_SOFT_CANDIDATE_ID = (
        VAE_CALIBRATED_LOCAL_ANCHOR_SOFT_METHOD_ID
    )
    _CALIBRATED_ANCHOR_SOFT_CONTROL_ID = (
        CLEAN_CALIBRATED_LOCAL_ANCHOR_SOFT_CONTROL_METHOD_ID
    )

    def _replace_lhat_endpoint_with_sampled_path(
        self,
        context: NodeContext,
        source: WaveformView,
        hard: WaveformView,
        *,
        anchor_standardized: torch.Tensor,
        hard_standardized: torch.Tensor,
    ) -> WaveformView:
        """Sample one supervised point on each clean-to-LHAT latent ray.

        The three frozen path locations are decoded together.  One location is
        then selected independently per eligible record with an isolated
        node-local generator.  Both the candidate and its identity control run
        this function so attack, VAE decode and RNG budgets remain matched.
        """

        if self.latent_pool is None or self.decoder is None:
            raise RuntimeError("LHAT path polish requires latent pool and decoder")
        candidate_positions = tuple(
            int(value)
            for value in hard.metadata.get("candidate_eligible_positions", ())
        )
        accepted_positions = tuple(
            int(value) for value in hard.metadata.get("accepted_positions", ())
        )
        if not accepted_positions:
            return hard
        local_by_batch_position = {
            batch_position: local_position
            for local_position, batch_position in enumerate(candidate_positions)
        }
        try:
            accepted_local_positions = tuple(
                local_by_batch_position[position] for position in accepted_positions
            )
        except KeyError as exc:
            raise RuntimeError(
                "LHAT path accepted position is absent from the candidate batch"
            ) from exc
        local = torch.as_tensor(
            accepted_local_positions,
            device=anchor_standardized.device,
            dtype=torch.long,
        )
        batch_positions = torch.as_tensor(
            accepted_positions,
            device=source.waveform.device,
            dtype=torch.long,
        )
        anchor = anchor_standardized.index_select(0, local).float()
        endpoint = hard_standardized.index_select(0, local).float()
        projections = tuple(
            project_paired_latent_path(anchor, endpoint, t=value)
            for value in LHAT_PATH_T_VALUES
        )
        standardized_paths = torch.stack(
            tuple(value.standardized_latent for value in projections),
            dim=1,
        )
        standardizer = self.latent_pool.standardizer
        native_paths = standardizer.inverse_transform(
            standardized_paths.reshape(-1, 4, 128)
        )
        decoded_flat, decoded_finite_flat = _decode_latents_chunked(
            self.decoder, native_paths
        )
        accepted_count = len(accepted_positions)
        decoded = decoded_flat.reshape(
            accepted_count, len(LHAT_PATH_T_VALUES), 1000, 12
        )
        decoded_finite = decoded_finite_flat.reshape(
            accepted_count, len(LHAT_PATH_T_VALUES)
        )
        anchor_reconstruction = hard.metadata.get("anchor_waveform_raw")
        if not isinstance(anchor_reconstruction, torch.Tensor):
            raise RuntimeError("LHAT path polish lacks anchor reconstruction")
        clean_raw = source.waveform.index_select(0, batch_positions).float()
        clean_decoded = anchor_reconstruction.index_select(
            0, batch_positions
        ).float()
        hard_raw = hard.waveform.index_select(0, batch_positions).float()
        corrected = apply_linear_endpoint_residual_correction(
            decoded,
            clean_raw=clean_raw,
            corrupt_raw=hard_raw,
            clean_decoded=clean_decoded,
            corrupt_decoded=hard_raw,
            t_values=LHAT_PATH_T_VALUES,
        )
        projection_finite = torch.stack(
            tuple(value.finite_mask for value in projections), dim=1
        )
        finite = (
            decoded_finite
            & projection_finite
            & torch.isfinite(corrected).flatten(2).all(dim=2)
        )
        standard_deviation = corrected.flatten(2).std(dim=2, unbiased=False)
        maximum_absolute = corrected.abs().flatten(2).amax(dim=2)
        valid = (
            finite
            & (standard_deviation >= self.minimum_std_mV)
            & (maximum_absolute <= self.maximum_abs_mV)
        )
        if not bool(valid.any(dim=1).all()):
            raise RuntimeError(
                "an accepted LHAT endpoint produced no valid frozen path point"
            )

        generator = context.torch_generator(
            "latent_path_t_choice", device=corrected.device
        )
        uniform = torch.rand(
            accepted_count,
            len(LHAT_PATH_T_VALUES),
            device=corrected.device,
            generator=generator,
        )
        choices = torch.where(
            valid,
            uniform,
            torch.full_like(uniform, -1.0),
        ).argmax(dim=1)
        rows = torch.arange(
            accepted_count, device=corrected.device, dtype=torch.long
        )
        if not bool(valid[rows, choices].all()):
            raise RuntimeError("LHAT path sampler selected an invalid path point")
        selected = corrected[rows, choices].to(dtype=hard.waveform.dtype)
        selected_t = torch.as_tensor(
            LHAT_PATH_T_VALUES, device=corrected.device, dtype=torch.float32
        ).index_select(0, choices)
        waveform = hard.waveform.clone()
        waveform.index_copy_(0, batch_positions, selected)

        pair_radius = projections[0].pair_radius_rms
        diagnostics = {
            "latent_path_t_mean": float(selected_t.mean().detach().cpu()),
            "latent_path_pair_radius_rms_mean": float(
                pair_radius.mean().detach().cpu()
            ),
            "latent_path_bank_valid_fraction": float(
                valid.float().mean().detach().cpu()
            ),
            "latent_path_selected_valid_fraction": float(
                valid[rows, choices].float().mean().detach().cpu()
            ),
        }
        for index, value in enumerate(LHAT_PATH_T_VALUES):
            diagnostics[f"latent_path_t_{int(round(value * 100)):03d}_fraction"] = (
                float((choices == index).float().mean().detach().cpu())
            )
        for name, value in diagnostics.items():
            context.record_diagnostic(name, value)
        stochastic_trace = dict(hard.metadata.get("stochastic_trace", {}))
        stochastic_trace.update(
            {
                "latent_path_choice_index": choices.detach().contiguous(),
                "latent_path_t": selected_t.detach().contiguous(),
            }
        )
        diagnostic_means = dict(hard.metadata.get("diagnostic_means", {}))
        diagnostic_means.update(diagnostics)
        return replace(
            hard,
            waveform=waveform.contiguous(),
            provenance=replace(
                hard.provenance,
                operation="vae_lhat_standardized_latent_path_sample",
                parameters={
                    **dict(hard.provenance.parameters),
                    "path_t_values": list(LHAT_PATH_T_VALUES),
                    "path_sampling": (
                        "uniform_valid_per_eligible_record_per_epoch"
                    ),
                    "geometry": "standardized_clean_to_lhat_ray",
                    "residual_correction": "linear_clean_hard_endpoint",
                    "candidate_training_view": "one_sampled_path_point",
                },
            ),
            metadata={
                **dict(hard.metadata),
                "diagnostic_means": diagnostic_means,
                "stochastic_trace": stochastic_trace,
                "latent_path_t_values": LHAT_PATH_T_VALUES,
                "latent_path_selected_t": selected_t.detach().contiguous(),
            },
        )

    def _lhat_attack(
        self,
        context: NodeContext,
        inputs: tuple[ViewValue, ...],
    ) -> ViewValue:
        if self.method.profile_name not in {
            self._PATH_BCE_CANDIDATE_ID,
            self._PATH_BCE_CONTROL_ID,
        }:
            return super()._lhat_attack(context, inputs)
        captured: dict[str, Any] = {}
        original = core_method_runtime.generate_lhat_adversarial

        def capture_lhat(**kwargs: Any):
            result = original(**kwargs)
            captured["anchor_standardized"] = kwargs[
                "anchor_standardized"
            ].detach()
            captured["hard_standardized"] = result.latent_standardized.detach()
            return result

        core_method_runtime.generate_lhat_adversarial = capture_lhat
        try:
            hard = super()._lhat_attack(context, inputs)
        finally:
            core_method_runtime.generate_lhat_adversarial = original
        if not isinstance(hard, WaveformView):
            raise TypeError("LHAT path node did not return a WaveformView")
        if len(inputs) != 1 or not isinstance(inputs[0], WaveformView):
            raise TypeError("LHAT path requires one clean WaveformView")
        if not hard.metadata.get("candidate_eligible_positions"):
            return hard
        anchor = captured.get("anchor_standardized")
        endpoint = captured.get("hard_standardized")
        if not isinstance(anchor, torch.Tensor) or not isinstance(
            endpoint, torch.Tensor
        ):
            raise RuntimeError("failed to capture LHAT latent path endpoints")
        return self._replace_lhat_endpoint_with_sampled_path(
            context,
            inputs[0],
            hard,
            anchor_standardized=anchor,
            hard_standardized=endpoint,
        )

    def _control_eligibility(
        self,
        clean: WaveformView,
    ) -> tuple[torch.Tensor, tuple[int, ...], tuple[str, ...], str | None]:
        pool = self.latent_pool
        if pool is None:
            raise RuntimeError("J4 matched control requires the candidate latent pool")
        eligible_hash_ids = getattr(pool, "eligible_hash_id_set", None)
        pool_hash_ids = getattr(pool, "hash_ids", None)
        if not isinstance(eligible_hash_ids, frozenset):
            raise TypeError(
                "J4 latent pool must expose eligible_hash_id_set as a frozenset"
            )
        if not isinstance(pool_hash_ids, tuple):
            raise TypeError("J4 latent pool must expose its ordered hash_ids tuple")
        pool_hash_id_set = set(pool_hash_ids)
        unknown = tuple(
            hash_id for hash_id in clean.sample_ids if hash_id not in pool_hash_id_set
        )
        if unknown:
            raise RuntimeError(
                "J4 control batch contains hashes outside the matched latent pool: "
                f"{unknown}"
            )
        mask = torch.as_tensor(
            [hash_id in eligible_hash_ids for hash_id in clean.sample_ids],
            device=clean.waveform.device,
            dtype=torch.bool,
        )
        mask &= clean.valid_mask
        positions = tuple(
            int(value)
            for value in torch.nonzero(mask, as_tuple=False)
            .flatten()
            .detach()
            .cpu()
            .tolist()
        )
        ineligible = tuple(
            hash_id
            for hash_id, is_eligible in zip(
                clean.sample_ids,
                mask.detach().cpu().tolist(),
                strict=True,
            )
            if not bool(is_eligible)
        )
        identity = getattr(pool, "identity", None)
        eligibility_sha256 = getattr(identity, "eligibility_sha256", None)
        if eligibility_sha256 is not None and (
            not isinstance(eligibility_sha256, str)
            or len(eligibility_sha256) != 64
        ):
            raise ValueError("J4 latent-pool eligibility identity is invalid")
        return mask, positions, ineligible, eligibility_sha256

    @staticmethod
    def _replace_bundle_views(
        bundle: ViewBundle,
        replacements: Mapping[str, WaveformView],
        *,
        diagnostics: Mapping[str, Any],
    ) -> ViewBundle:
        def resolved(value: ViewValue) -> ViewValue:
            if isinstance(value, WaveformView):
                return replacements.get(value.name, value)
            return value

        return ViewBundle(
            values={
                name: resolved(value) for name, value in bundle.values.items()
            },
            node_values={
                name: resolved(value) for name, value in bundle.node_values.items()
            },
            diagnostics=dict(diagnostics),
        )

    def _match_control_eligibility(
        self,
        generated: GeneratedMethodBatch,
    ) -> GeneratedMethodBatch:
        if generated.quality_rejected:
            raise RuntimeError(
                "J4 clean control unexpectedly reported quality rejection"
            )
        clean = generated.bundle.require("clean_view")
        chain3 = generated.bundle.require("chain3_control_view")
        augmix = generated.bundle.require("augmix_view")
        if not all(isinstance(view, WaveformView) for view in (clean, chain3, augmix)):
            raise TypeError("J4 matched views must all be WaveformView instances")
        assert isinstance(clean, WaveformView)
        assert isinstance(chain3, WaveformView)
        assert isinstance(augmix, WaveformView)
        if not bool(clean.valid_mask.all()):
            raise RuntimeError("J4 clean view must remain valid for every base record")
        for name, view in (("chain3_control_view", chain3), ("augmix_view", augmix)):
            if view.sample_ids != clean.sample_ids or not torch.equal(
                view.labels, clean.labels
            ):
                raise RuntimeError(f"J4 {name} lost origin/label alignment")
            if not torch.equal(view.valid_mask, clean.valid_mask):
                raise RuntimeError(
                    f"J4 ordinary control {name} must be valid before eligibility masking"
                )

        mask, positions, ineligible, eligibility_sha256 = self._control_eligibility(
            clean
        )
        accounting = {
            "candidate_eligible_positions": positions,
            "accepted_positions": positions,
            "ineligible_hash_ids": ineligible,
            "quality_rejected": (),
            "eligibility_policy": "exact_positive_set_nonself_distinct",
            "eligibility_sha256": eligibility_sha256,
        }
        masked_chain3 = replace(
            chain3,
            valid_mask=mask.clone(),
            metadata={**dict(chain3.metadata), **accounting},
        )
        matched_augmix_waveform = torch.where(
            mask[:, None, None],
            augmix.waveform,
            clean.waveform,
        ).contiguous()
        masked_augmix = replace(
            augmix,
            waveform=matched_augmix_waveform,
            valid_mask=mask.clone(),
            metadata={
                **dict(augmix.metadata),
                "accepted_positions": positions,
                "ineligible_hash_ids": ineligible,
                "eligibility_policy": "exact_positive_set_nonself_distinct",
                "eligibility_sha256": eligibility_sha256,
            },
        )
        diagnostics = dict(generated.bundle.diagnostics)
        diagnostics[
            f"{chain3.provenance.node_id}/candidate_eligible_count"
        ] = len(positions)
        diagnostics[
            f"{chain3.provenance.node_id}/quality_accepted_count"
        ] = len(positions)
        diagnostics[f"{chain3.provenance.node_id}/ineligible_count"] = len(
            ineligible
        )
        diagnostics[f"{chain3.provenance.node_id}/quality_rejected_count"] = 0
        diagnostics[f"{augmix.provenance.node_id}/accepted_count"] = len(positions)
        diagnostic_weights = dict(generated.diagnostic_weights)
        for name in (
            f"{chain3.provenance.node_id}/candidate_eligible_count",
            f"{chain3.provenance.node_id}/quality_accepted_count",
            f"{chain3.provenance.node_id}/ineligible_count",
            f"{chain3.provenance.node_id}/quality_rejected_count",
        ):
            diagnostic_weights[name] = 1
        bundle = self._replace_bundle_views(
            generated.bundle,
            {
                chain3.name: masked_chain3,
                augmix.name: masked_augmix,
            },
            diagnostics=diagnostics,
        )
        return replace(
            generated,
            bundle=bundle,
            diagnostic_weights=diagnostic_weights,
            candidate_eligible_positions=positions,
            accepted_positions=positions,
            quality_view_total_count=clean.batch_size,
            quality_view_accepted_count=len(positions),
            ineligible_hash_ids=ineligible,
            quality_rejected=(),
        )

    def _match_distill_control_eligibility(
        self,
        generated: GeneratedMethodBatch,
    ) -> GeneratedMethodBatch:
        if generated.quality_rejected:
            raise RuntimeError(
                "teacher-distill identity control unexpectedly reported "
                "quality rejection"
            )
        clean = generated.bundle.require("clean_view")
        control = generated.bundle.require("lhat_control_view")
        if not isinstance(clean, WaveformView) or not isinstance(
            control, WaveformView
        ):
            raise TypeError(
                "teacher-distill clean/control views must be WaveformView"
            )
        if not bool(clean.valid_mask.all()):
            raise RuntimeError(
                "teacher-distill clean view must be valid for every record"
            )
        if (
            control.sample_ids != clean.sample_ids
            or not torch.equal(control.labels, clean.labels)
            or not torch.equal(control.valid_mask, clean.valid_mask)
        ):
            raise RuntimeError(
                "teacher-distill control lost origin, label, or validity "
                "alignment"
            )
        mask, positions, ineligible, eligibility_sha256 = (
            self._control_eligibility(clean)
        )
        accounting = {
            "candidate_eligible_positions": positions,
            "accepted_positions": positions,
            "ineligible_hash_ids": ineligible,
            "quality_rejected": (),
            "eligibility_policy": "exact_positive_set_nonself_distinct",
            "eligibility_sha256": eligibility_sha256,
        }
        masked_control = replace(
            control,
            waveform=clean.waveform.clone().contiguous(),
            valid_mask=mask.clone(),
            metadata={**dict(control.metadata), **accounting},
        )
        diagnostics = dict(generated.bundle.diagnostics)
        node_id = control.provenance.node_id
        diagnostics[f"{node_id}/candidate_eligible_count"] = len(positions)
        diagnostics[f"{node_id}/quality_accepted_count"] = len(positions)
        diagnostics[f"{node_id}/ineligible_count"] = len(ineligible)
        diagnostics[f"{node_id}/quality_rejected_count"] = 0
        diagnostic_weights = dict(generated.diagnostic_weights)
        for name in (
            f"{node_id}/candidate_eligible_count",
            f"{node_id}/quality_accepted_count",
            f"{node_id}/ineligible_count",
            f"{node_id}/quality_rejected_count",
        ):
            diagnostic_weights[name] = 1
        bundle = self._replace_bundle_views(
            generated.bundle,
            {control.name: masked_control},
            diagnostics=diagnostics,
        )
        return replace(
            generated,
            bundle=bundle,
            diagnostic_weights=diagnostic_weights,
            candidate_eligible_positions=positions,
            accepted_positions=positions,
            quality_view_total_count=clean.batch_size,
            quality_view_accepted_count=len(positions),
            ineligible_hash_ids=ineligible,
            quality_rejected=(),
        )

    def _normalize_distill_candidate_fallback(
        self,
        generated: GeneratedMethodBatch,
    ) -> GeneratedMethodBatch:
        if generated.quality_rejected:
            raise RuntimeError(
                "teacher-distill candidate quality rejection breaks matched "
                f"eligibility: {tuple(generated.quality_rejected)}"
            )
        clean = generated.bundle.require("clean_view")
        lhat = generated.bundle.require("lhat_view")
        if not isinstance(clean, WaveformView) or not isinstance(lhat, WaveformView):
            raise TypeError(
                "teacher-distill clean/LHAT views must be WaveformView"
            )
        if lhat.sample_ids != clean.sample_ids or not torch.equal(
            lhat.labels, clean.labels
        ):
            raise RuntimeError(
                "teacher-distill LHAT view lost origin or label alignment"
            )
        expected_mask, positions, ineligible, _ = self._control_eligibility(
            clean
        )
        if not torch.equal(lhat.valid_mask, expected_mask):
            raise RuntimeError(
                "teacher-distill candidate validity drifted from latent-pool "
                "eligibility"
            )
        fallback = torch.where(
            lhat.valid_mask[:, None, None],
            lhat.waveform,
            clean.waveform,
        ).contiguous()
        normalized = replace(lhat, waveform=fallback)
        bundle = self._replace_bundle_views(
            generated.bundle,
            {lhat.name: normalized},
            diagnostics=generated.bundle.diagnostics,
        )
        return replace(
            generated,
            bundle=bundle,
            candidate_eligible_positions=positions,
            accepted_positions=positions,
            quality_view_total_count=clean.batch_size,
            quality_view_accepted_count=len(positions),
            ineligible_hash_ids=ineligible,
            quality_rejected=(),
        )

    def _normalize_anchor_soft_quality_fallback(
        self,
        generated: GeneratedMethodBatch,
        *,
        identity_control: bool,
    ) -> GeneratedMethodBatch:
        """Preserve hard-QC accounting while matching the L8 compute graph.

        Both L8 arms execute the same LHAT node. The candidate keeps accepted
        decoded views; the control replaces those waveforms with clean
        identity. Ineligible or hard-QC-rejected records remain clean-only in
        both arms and are excluded from the auxiliary BCE by the same typed
        valid mask produced by the LHAT runtime.
        """

        clean = generated.bundle.require("clean_view")
        auxiliary_view_name = (
            "lhat_control_view" if identity_control else "lhat_view"
        )
        auxiliary = generated.bundle.require(auxiliary_view_name)
        if not isinstance(clean, WaveformView) or not isinstance(
            auxiliary, WaveformView
        ):
            raise TypeError(
                "local anchor-soft clean/auxiliary views must be WaveformView"
            )
        if not bool(clean.valid_mask.all()):
            raise RuntimeError(
                "local anchor-soft clean view must be valid for every record"
            )
        if (
            auxiliary.sample_ids != clean.sample_ids
            or not torch.equal(auxiliary.labels, clean.labels)
        ):
            raise RuntimeError(
                "local anchor-soft auxiliary lost origin or label alignment"
            )

        _, expected_positions, expected_ineligible, eligibility_sha256 = (
            self._control_eligibility(clean)
        )
        candidate_positions = tuple(
            int(value) for value in generated.candidate_eligible_positions
        )
        accepted_positions = tuple(
            int(value) for value in generated.accepted_positions
        )
        if candidate_positions != expected_positions:
            raise RuntimeError(
                "local anchor-soft candidate eligibility drifted from the "
                "frozen latent-pool identity"
            )
        if tuple(generated.ineligible_hash_ids) != expected_ineligible:
            raise RuntimeError(
                "local anchor-soft ineligible hash IDs drifted from the "
                "frozen latent-pool identity"
            )
        candidate_set = set(candidate_positions)
        accepted_set = set(accepted_positions)
        if not accepted_set <= candidate_set:
            raise RuntimeError(
                "local anchor-soft accepted positions are outside eligibility"
            )
        expected_mask = torch.zeros(
            clean.batch_size,
            dtype=torch.bool,
            device=clean.waveform.device,
        )
        if accepted_positions:
            expected_mask[
                torch.as_tensor(
                    accepted_positions,
                    dtype=torch.long,
                    device=clean.waveform.device,
                )
            ] = True
        if not torch.equal(auxiliary.valid_mask, expected_mask):
            raise RuntimeError(
                "local anchor-soft auxiliary valid mask drifted from LHAT "
                "accepted positions"
            )

        expected_rejected_hashes = {
            clean.sample_ids[position]
            for position in candidate_positions
            if position not in accepted_set
        }
        actual_rejected_hashes: set[str] = set()
        allowed_reasons = {"nonfinite", "flatline", "severe_amplitude"}
        for item in generated.quality_rejected:
            hash_id = item.get("hash_id")
            reason = item.get("reason")
            if not isinstance(hash_id, str) or not isinstance(reason, str):
                raise TypeError(
                    "local anchor-soft quality rejection is malformed"
                )
            reason_parts = set(reason.split("+"))
            if not reason_parts or not reason_parts <= allowed_reasons:
                raise RuntimeError(
                    "local anchor-soft quality rejection escaped the hard-QC "
                    f"allowlist: {reason!r}"
                )
            actual_rejected_hashes.add(hash_id)
        if actual_rejected_hashes != expected_rejected_hashes:
            raise RuntimeError(
                "local anchor-soft hard-QC hashes do not match rejected "
                "eligible positions"
            )
        if (
            generated.quality_view_total_count != clean.batch_size
            or generated.quality_view_accepted_count
            != len(accepted_positions)
        ):
            raise RuntimeError(
                "local anchor-soft quality accounting drifted"
            )

        waveform = (
            clean.waveform.clone().contiguous()
            if identity_control
            else torch.where(
                expected_mask[:, None, None],
                auxiliary.waveform,
                clean.waveform,
            ).contiguous()
        )
        accounting = {
            "candidate_eligible_positions": candidate_positions,
            "accepted_positions": accepted_positions,
            "ineligible_hash_ids": expected_ineligible,
            "quality_rejected": tuple(generated.quality_rejected),
            "eligibility_policy": "exact_positive_set_nonself_distinct",
            "eligibility_sha256": eligibility_sha256,
            "quality_rejection_policy": "clean_loss_only",
        }
        normalized = replace(
            auxiliary,
            waveform=waveform,
            valid_mask=expected_mask,
            metadata={**dict(auxiliary.metadata), **accounting},
        )
        bundle = self._replace_bundle_views(
            generated.bundle,
            {auxiliary.name: normalized},
            diagnostics=generated.bundle.diagnostics,
        )
        return replace(generated, bundle=bundle)

    def generate(self, **kwargs: Any) -> GeneratedMethodBatch:
        generated = super().generate(**kwargs)
        if self.method.profile_name == self._CANDIDATE_ID:
            if generated.quality_rejected:
                raise RuntimeError(
                    "J4 candidate quality rejection breaks matched eligibility: "
                    f"{tuple(generated.quality_rejected)}"
                )
            return generated
        if self.method.profile_name == self._CONTROL_ID:
            return self._match_control_eligibility(generated)
        if self.method.profile_name == self._DISTILL_CANDIDATE_ID:
            return self._normalize_distill_candidate_fallback(generated)
        if self.method.profile_name == self._DISTILL_CONTROL_ID:
            return self._match_distill_control_eligibility(generated)
        if self.method.profile_name == self._HARD_BCE_CANDIDATE_ID:
            return self._normalize_distill_candidate_fallback(generated)
        if self.method.profile_name == self._HARD_BCE_CONTROL_ID:
            return self._match_distill_control_eligibility(generated)
        if self.method.profile_name == self._PATH_BCE_CANDIDATE_ID:
            return self._normalize_distill_candidate_fallback(generated)
        if self.method.profile_name == self._PATH_BCE_CONTROL_ID:
            return self._match_distill_control_eligibility(generated)
        if self.method.profile_name == self._CONSISTENCY_CANDIDATE_ID:
            return self._normalize_distill_candidate_fallback(generated)
        if self.method.profile_name == self._CONSISTENCY_CONTROL_ID:
            return self._match_distill_control_eligibility(generated)
        if self.method.profile_name == self._FEATURE_CANDIDATE_ID:
            return self._normalize_distill_candidate_fallback(generated)
        if self.method.profile_name == self._FEATURE_CONTROL_ID:
            return self._match_distill_control_eligibility(generated)
        if self.method.profile_name in {
            self._ANCHOR_SOFT_CANDIDATE_ID,
            self._CALIBRATED_ANCHOR_SOFT_CANDIDATE_ID,
        }:
            return self._normalize_anchor_soft_quality_fallback(
                generated,
                identity_control=False,
            )
        if self.method.profile_name in {
            self._ANCHOR_SOFT_CONTROL_ID,
            self._CALIBRATED_ANCHOR_SOFT_CONTROL_ID,
        }:
            return self._normalize_anchor_soft_quality_fallback(
                generated,
                identity_control=True,
            )
        raise RuntimeError(
            "matched-eligibility runtime received an unsupported method"
        )


_EXPECTED_OBJECTIVES: Mapping[
    str, tuple[tuple[str, str, tuple[str, ...], float], ...]
] = MappingProxyType(
    {
        D8_METHOD_ID: (
            ("clean_bce", "bce", ("clean_view",), 1.0),
            ("corruption_1_bce", "bce", ("corruption_view_1",), 0.4),
            ("corruption_2_bce", "bce", ("corruption_view_2",), 0.4),
            ("lhat_bce", "bce", ("lhat_view",), 0.2),
        ),
        D9_METHOD_ID: (
            ("clean_bce", "bce", ("clean_view",), 1.0),
            ("corruption_1_bce", "bce", ("corruption_view_1",), 0.4),
            ("corruption_2_bce", "bce", ("corruption_view_2",), 0.4),
            ("lhat_bce", "bce", ("lhat_view",), 0.2),
            (
                "clean_corruption_lhat_jsd",
                "bernoulli_jsd",
                (
                    "clean_view",
                    "corruption_view_1",
                    "corruption_view_2",
                    "lhat_view",
                ),
                0.6,
            ),
        ),
        R4_RESIDUAL_LATENT2_METHOD_ID: (
            ("clean_bce", "bce", ("clean_view",), 1.0),
            ("corruption_1_bce", "bce", ("corruption_view_1",), 0.2),
            ("corruption_2_bce", "bce", ("corruption_view_2",), 0.2),
            ("corruption_3_bce", "bce", ("corruption_view_3",), 0.2),
            ("corruption_4_bce", "bce", ("corruption_view_4",), 0.2),
            ("latent_1_bce", "bce", ("latent_view_1",), 0.1),
            ("latent_2_bce", "bce", ("latent_view_2",), 0.1),
            (
                "clean_raw4_latent2_jsd",
                "bernoulli_jsd",
                (
                    "clean_view",
                    "corruption_view_1",
                    "corruption_view_2",
                    "corruption_view_3",
                    "corruption_view_4",
                    "latent_view_1",
                    "latent_view_2",
                ),
                0.3,
            ),
        ),
        R4_LHAT_AUGMIX_LOWDOSE_METHOD_ID: (
            ("clean_bce", "bce", ("clean_view",), 1.0),
            ("corruption_1_bce", "bce", ("corruption_view_1",), 0.2),
            ("corruption_2_bce", "bce", ("corruption_view_2",), 0.2),
            ("corruption_3_bce", "bce", ("corruption_view_3",), 0.2),
            ("corruption_4_bce", "bce", ("corruption_view_4",), 0.2),
            ("augmix_bce", "bce", ("augmix_view",), 0.2),
            (
                "clean_raw4_augmix_jsd",
                "bernoulli_jsd",
                (
                    "clean_view",
                    "corruption_view_1",
                    "corruption_view_2",
                    "corruption_view_3",
                    "corruption_view_4",
                    "augmix_view",
                ),
                0.3,
            ),
        ),
        R8_RESIDUAL_LATENT1_CLEAN55_METHOD_ID: (
            ("clean_bce", "bce", ("clean_view",), 1.0),
            ("clean_preserve_bce", "bce", ("clean_view",), 0.1),
            ("corruption_1_bce", "bce", ("corruption_view_1",), 0.1),
            ("corruption_2_bce", "bce", ("corruption_view_2",), 0.1),
            ("corruption_3_bce", "bce", ("corruption_view_3",), 0.1),
            ("corruption_4_bce", "bce", ("corruption_view_4",), 0.1),
            ("corruption_5_bce", "bce", ("corruption_view_5",), 0.1),
            ("corruption_6_bce", "bce", ("corruption_view_6",), 0.1),
            ("corruption_7_bce", "bce", ("corruption_view_7",), 0.1),
            ("corruption_8_bce", "bce", ("corruption_view_8",), 0.1),
            ("latent_bce", "bce", ("latent_view",), 0.1),
            (
                "clean_raw8_latent1_jsd",
                "bernoulli_jsd",
                (
                    "clean_view",
                    "corruption_view_1",
                    "corruption_view_2",
                    "corruption_view_3",
                    "corruption_view_4",
                    "corruption_view_5",
                    "corruption_view_6",
                    "corruption_view_7",
                    "corruption_view_8",
                    "latent_view",
                ),
                0.3,
            ),
        ),
        A5_M20_LHAT_AUGMIX_METHOD_ID: (
            ("clean_bce", "bce", ("clean_view",), 1.0),
            *tuple(
                (
                    f"augmix_{index:02d}_bce",
                    "bce",
                    (f"augmix_view_{index:02d}",),
                    0.05,
                )
                for index in range(1, 21)
            ),
            (
                "clean_augmix_01_02_lhat_jsd",
                "bernoulli_jsd",
                (
                    "clean_view",
                    "augmix_view_01",
                    "augmix_view_02",
                    "lhat_view",
                ),
                0.6,
            ),
        ),
        PURE_M20_LHAT_AUGMIX_METHOD_ID: (
            ("clean_bce", "bce", ("clean_view",), 1.0),
            *tuple(
                (
                    f"augmix_{index:02d}_bce",
                    "bce",
                    (f"augmix_view_{index:02d}",),
                    0.05,
                )
                for index in range(1, 21)
            ),
            (
                "clean_augmix_01_02_lhat_jsd",
                "bernoulli_jsd",
                (
                    "clean_view",
                    "augmix_view_01",
                    "augmix_view_02",
                    "lhat_view",
                ),
                0.6,
            ),
        ),
        **{
            method_id: (
                ("clean_bce", "bce", ("clean_view",), 1.0),
                *tuple(
                    (
                        f"augmix_{index:02d}_bce",
                        "bce",
                        (f"augmix_view_{index:02d}",),
                        0.05,
                    )
                    for index in range(1, 21)
                ),
                (
                    "clean_augmix_01_02_lhat_jsd",
                    "bernoulli_jsd",
                    (
                        "clean_view",
                        "augmix_view_01",
                        "augmix_view_02",
                        "lhat_view",
                    ),
                    0.6,
                ),
            )
            for method_id in BOUNDED_AUGMAX_METHOD_IDS
        },
        VAE_LHAT_CLEAN_POLISH_METHOD_ID: (
            ("clean_bce", "bce", ("clean_view",), 1.0),
            ("lhat_direct_bce", "bce", ("lhat_view",), 0.5),
            (
                "clean_lhat_jsd",
                "bernoulli_jsd",
                ("clean_view", "lhat_view"),
                0.5,
            ),
        ),
        VAE_LHAT_CHAIN3_POLISH_METHOD_ID: (
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
        J4_VAE_LHAT_CHAIN3_METHOD_ID: (
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
        CLEAN_CHAIN3_POLISH_CONTROL_METHOD_ID: (
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
        J4_CLEAN_CHAIN3_CONTROL_METHOD_ID: (
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
)

_DUAL_AXIS_BN_WEIGHTS = MappingProxyType(
    {
        "clean_view": 0.5,
        "corruption_view_1": 0.2,
        "corruption_view_2": 0.2,
        "lhat_view": 0.1,
    }
)
_R4_RESIDUAL_LATENT2_BN_WEIGHTS = MappingProxyType(
    {
        "clean_view": 0.5,
        "corruption_view_1": 0.1,
        "corruption_view_2": 0.1,
        "corruption_view_3": 0.1,
        "corruption_view_4": 0.1,
        "latent_view_1": 0.05,
        "latent_view_2": 0.05,
    }
)
_R4_LHAT_AUGMIX_LOWDOSE_BN_WEIGHTS = MappingProxyType(
    {
        "clean_view": 0.5,
        "corruption_view_1": 0.1,
        "corruption_view_2": 0.1,
        "corruption_view_3": 0.1,
        "corruption_view_4": 0.1,
        "augmix_view": 0.1,
    }
)
_R8_RESIDUAL_LATENT1_CLEAN55_BN_WEIGHTS = MappingProxyType(
    {
        "clean_view": 0.55,
        "corruption_view_1": 0.05,
        "corruption_view_2": 0.05,
        "corruption_view_3": 0.05,
        "corruption_view_4": 0.05,
        "corruption_view_5": 0.05,
        "corruption_view_6": 0.05,
        "corruption_view_7": 0.05,
        "corruption_view_8": 0.05,
        "latent_view": 0.05,
    }
)
_A5_M20_LHAT_AUGMIX_BN_WEIGHTS = MappingProxyType(
    {
        "clean_view": 0.5,
        **{
            f"augmix_view_{index:02d}": 0.025
            for index in range(1, 21)
        },
        "lhat_view": 0.0,
    }
)
_VAE_LHAT_CLEAN_POLISH_BN_WEIGHTS = MappingProxyType(
    {
        "clean_view": 0.5,
        "lhat_view": 0.5,
    }
)
_VAE_LHAT_CHAIN3_POLISH_BN_WEIGHTS = MappingProxyType(
    {
        "clean_view": 0.75,
        "lhat_view": 0.125,
        "augmix_view": 0.125,
    }
)
_CLEAN_CHAIN3_POLISH_CONTROL_BN_WEIGHTS = MappingProxyType(
    {
        "clean_view": 0.75,
        "chain3_control_view": 0.125,
        "augmix_view": 0.125,
    }
)
_VAE_LHAT_TEACHER_DISTILL_BN_WEIGHTS = MappingProxyType(
    {
        "clean_view": 0.75,
        "lhat_view": 0.25,
    }
)
_CLEAN_TEACHER_DISTILL_CONTROL_BN_WEIGHTS = MappingProxyType(
    {
        "clean_view": 0.75,
        "lhat_control_view": 0.25,
    }
)
_EXPECTED_BN_WEIGHTS = MappingProxyType(
    {
        D8_METHOD_ID: _DUAL_AXIS_BN_WEIGHTS,
        D9_METHOD_ID: _DUAL_AXIS_BN_WEIGHTS,
        R4_RESIDUAL_LATENT2_METHOD_ID: _R4_RESIDUAL_LATENT2_BN_WEIGHTS,
        R4_LHAT_AUGMIX_LOWDOSE_METHOD_ID: _R4_LHAT_AUGMIX_LOWDOSE_BN_WEIGHTS,
        R8_RESIDUAL_LATENT1_CLEAN55_METHOD_ID: _R8_RESIDUAL_LATENT1_CLEAN55_BN_WEIGHTS,
        A5_M20_LHAT_AUGMIX_METHOD_ID: _A5_M20_LHAT_AUGMIX_BN_WEIGHTS,
        PURE_M20_LHAT_AUGMIX_METHOD_ID: _A5_M20_LHAT_AUGMIX_BN_WEIGHTS,
        **{
            method_id: _A5_M20_LHAT_AUGMIX_BN_WEIGHTS
            for method_id in BOUNDED_AUGMAX_METHOD_IDS
        },
        VAE_LHAT_CLEAN_POLISH_METHOD_ID: _VAE_LHAT_CLEAN_POLISH_BN_WEIGHTS,
        VAE_LHAT_CHAIN3_POLISH_METHOD_ID: _VAE_LHAT_CHAIN3_POLISH_BN_WEIGHTS,
        J4_VAE_LHAT_CHAIN3_METHOD_ID: _VAE_LHAT_CHAIN3_POLISH_BN_WEIGHTS,
        CLEAN_CHAIN3_POLISH_CONTROL_METHOD_ID: (
            _CLEAN_CHAIN3_POLISH_CONTROL_BN_WEIGHTS
        ),
        J4_CLEAN_CHAIN3_CONTROL_METHOD_ID: (
            _CLEAN_CHAIN3_POLISH_CONTROL_BN_WEIGHTS
        ),
        VAE_LHAT_TEACHER_DISTILL_METHOD_ID: (
            _VAE_LHAT_TEACHER_DISTILL_BN_WEIGHTS
        ),
        CLEAN_TEACHER_DISTILL_CONTROL_METHOD_ID: (
            _CLEAN_TEACHER_DISTILL_CONTROL_BN_WEIGHTS
        ),
    }
)


def _declares_scale_only_adapter(method: CompiledMethod) -> bool:
    """Recognize allowlisted profiles and reject copied marker contracts."""

    return (
        method.profile_name in SCALE_ONLY_METHOD_IDS
        or "objective_scale_adapter_mode" in method.contracts
    )


def _same_finite_float(left: Any, right: float) -> bool:
    return (
        not isinstance(left, bool)
        and isinstance(left, (int, float))
        and math.isfinite(float(left))
        and math.isclose(float(left), right, rel_tol=0.0, abs_tol=1.0e-12)
    )


def validate_scale_only_method(method: CompiledMethod) -> None:
    """Fail closed unless ``method`` exactly matches its frozen semantics."""

    method_id = method.profile_name
    if method_id not in SCALE_ONLY_METHOD_IDS:
        raise ValueError(
            "objective scale-only method is not allowlisted: " f"{method_id!r}"
        )
    if method.contracts.get("objective_scale_adapter_mode") != "scale_only":
        raise ValueError(
            f"{method_id} contracts.objective_scale_adapter_mode must be "
            "'scale_only'"
        )
    if not _same_finite_float(
        method.contracts.get("objective_global_scale"), OBJECTIVE_GLOBAL_SCALE
    ):
        raise ValueError(
            f"{method_id} contracts.objective_global_scale must be 0.5"
        )
    if "diverse_augmax" in method.contracts:
        raise ValueError(
            f"{method_id} scale-only method may not request the Diverse runtime"
        )
    if method.contracts.get("fixed20_train_views") != 0:
        raise ValueError(f"{method_id} must not use fixed20 training views")
    if method.contracts.get("augmix_train_views") != 0:
        raise ValueError(f"{method_id} must not use prebuilt AugMix training views")
    if method.contracts.get("heldout_target_feedback_allowed") is not False:
        raise ValueError(f"{method_id} must forbid held-out target feedback")
    if method_id in BOUNDED_AUGMAX_METHOD_IDS:
        expected_hardness = {
            "enabled": True,
            "method": "hardest_per_record_bce_under_cap",
            "candidates": [
                {"weights": [0.50, 0.25, 0.25], "strength": 0.75},
                {"weights": [0.25, 0.50, 0.25], "strength": 0.75},
                {"weights": [0.25, 0.25, 0.50], "strength": 0.75},
                {
                    "weights": [
                        0.3333333333,
                        0.3333333333,
                        0.3333333334,
                    ],
                    "strength": 1.0,
                },
                {"weights": [0.20, 0.20, 0.60], "strength": 1.0},
                {"weights": [0.70, 0.15, 0.15], "strength": 1.0},
            ],
            "minimum_bce_gain": 0.0,
            "maximum_bce_gain": 0.15,
            "fallback": "classic_sampled_augmix",
            "cache_scope": "one_generate_call",
            "classifier_forward_chunk_size": 256,
        }
        actual_hardness = method.contracts.get("hardness_coupled_augmax")
        if not isinstance(actual_hardness, Mapping) or dict(
            actual_hardness
        ) != expected_hardness:
            raise ValueError(
                f"{method_id} bounded AugMax candidate set drifted"
            )
        expected_geometry = {
            "lhat_hull_lambda": 0.35,
            "lhat_steps": 10,
            "lhat_pgd_epsilon_l2_standardized": 10.0,
            "local_lhat_geometry_preset": (
                "k500_internal_bounded_augmax_lam35_eps10_v1"
            ),
        }
        for name, expected in expected_geometry.items():
            actual = method.contracts.get(name)
            matches = (
                _same_finite_float(actual, expected)
                if isinstance(expected, float)
                else actual == expected
            )
            if not matches:
                raise ValueError(
                    f"{method_id} bounded AugMax field {name} drifted: "
                    f"{actual!r}"
                )
        if method.contracts.get("full_k500_refit_supported") is not True:
            raise ValueError(
                f"{method_id} must declare full-K500 refit support"
            )
        if method.contracts.get("tuning_partition_only") != "k500":
            raise ValueError(
                f"{method_id} must use the complete K500 adaptation partition"
            )
        if method_id == VAE_LHAT_BOUNDED_AUGMAX_POLISH_METHOD_ID:
            if (
                method.contracts.get("matched_control_method_id")
                != CLEAN_BOUNDED_AUGMAX_CONTROL_METHOD_ID
            ):
                raise ValueError(
                    f"{method_id} matched control identity drifted"
                )
            for forbidden in (
                "matched_candidate_method_id",
                "matched_chain3_control_mode",
                "control_executes_full_lhat_search",
            ):
                if forbidden in method.contracts:
                    raise ValueError(
                        f"{method_id} candidate unexpectedly declares "
                        f"control field {forbidden}"
                    )
        else:
            if (
                method.contracts.get("matched_candidate_method_id")
                != VAE_LHAT_BOUNDED_AUGMAX_POLISH_METHOD_ID
                or method.contracts.get("matched_chain3_control_mode")
                != "replace_lhat_with_clean_after_full_search_and_qc"
                or method.contracts.get("control_executes_full_lhat_search")
                is not True
            ):
                raise ValueError(
                    f"{method_id} matched full-search control contract drifted"
                )
            if "matched_control_method_id" in method.contracts:
                raise ValueError(
                    f"{method_id} control unexpectedly declares a control id"
                )

    lowdose_auxiliary_specs = {
        VAE_LHAT_TEACHER_DISTILL_METHOD_ID: (
            "lhat_teacher_distill_bce",
            "bce",
            ("lhat_view",),
        ),
        CLEAN_TEACHER_DISTILL_CONTROL_METHOD_ID: (
            "clean_teacher_distill_control_bce",
            "bce",
            ("lhat_control_view",),
        ),
        VAE_LHAT_HARD_BCE_METHOD_ID: (
            "lhat_hard_bce",
            "bce",
            ("lhat_view",),
        ),
        CLEAN_HARD_BCE_CONTROL_METHOD_ID: (
            "clean_hard_control_bce",
            "bce",
            ("lhat_control_view",),
        ),
        VAE_LHAT_PATH_BCE_METHOD_ID: (
            "lhat_path_hard_bce",
            "bce",
            ("lhat_view",),
        ),
        CLEAN_LHAT_PATH_CONTROL_METHOD_ID: (
            "clean_lhat_path_control_bce",
            "bce",
            ("lhat_control_view",),
        ),
        VAE_LHAT_CONSISTENCY_METHOD_ID: (
            "clean_lhat_consistency_jsd",
            "bernoulli_jsd",
            ("clean_view", "lhat_view"),
        ),
        CLEAN_CONSISTENCY_CONTROL_METHOD_ID: (
            "clean_identity_consistency_jsd",
            "bernoulli_jsd",
            ("clean_view", "lhat_control_view"),
        ),
        VAE_LHAT_FEATURE_INVARIANCE_METHOD_ID: (
            "clean_lhat_feature_cosine",
            "bernoulli_jsd",
            ("clean_view", "lhat_view"),
        ),
        CLEAN_FEATURE_INVARIANCE_CONTROL_METHOD_ID: (
            "clean_identity_feature_cosine",
            "bernoulli_jsd",
            ("clean_view", "lhat_control_view"),
        ),
        VAE_LHAT_LOCAL_ANCHOR_SOFT_METHOD_ID: (
            "lhat_anchor_soft_bce",
            "bce",
            ("lhat_view",),
        ),
        CLEAN_LOCAL_ANCHOR_SOFT_CONTROL_METHOD_ID: (
            "clean_anchor_soft_control_bce",
            "bce",
            ("lhat_control_view",),
        ),
        VAE_CALIBRATED_LOCAL_ANCHOR_SOFT_METHOD_ID: (
            "lhat_anchor_soft_bce",
            "bce",
            ("lhat_view",),
        ),
        CLEAN_CALIBRATED_LOCAL_ANCHOR_SOFT_CONTROL_METHOD_ID: (
            "clean_anchor_soft_control_bce",
            "bce",
            ("lhat_control_view",),
        ),
    }
    lowdose_method = method_id in lowdose_auxiliary_specs
    lowdose_auxiliary_weight: float | None = None
    if lowdose_method:
        raw_weight = method.contracts.get("auxiliary_effective_weight")
        if (
            isinstance(raw_weight, bool)
            or not isinstance(raw_weight, (int, float))
            or not math.isfinite(float(raw_weight))
            or not 0.0 < float(raw_weight) < 0.5
        ):
            raise ValueError(
                f"{method_id} auxiliary_effective_weight must lie in (0,0.5)"
            )
        lowdose_auxiliary_weight = float(raw_weight)
        auxiliary_name, auxiliary_kind, auxiliary_views = (
            lowdose_auxiliary_specs[method_id]
        )
        if auxiliary_kind == "bernoulli_jsd":
            expected_objective = (
                ("clean_bce", "bce", ("clean_view",), 1.0),
                (
                    "clean_preserve_bce",
                    "bce",
                    ("clean_view",),
                    1.0 - 2.0 * lowdose_auxiliary_weight,
                ),
                (
                    auxiliary_name,
                    auxiliary_kind,
                    auxiliary_views,
                    2.0 * lowdose_auxiliary_weight,
                ),
            )
        else:
            expected_objective = (
                ("clean_bce", "bce", ("clean_view",), 1.0),
                (
                    "clean_preserve_bce",
                    "bce",
                    ("clean_view",),
                    1.0 - 2.0 * lowdose_auxiliary_weight,
                ),
                (
                    auxiliary_name,
                    auxiliary_kind,
                    auxiliary_views,
                    2.0 * lowdose_auxiliary_weight,
                ),
            )
    else:
        expected_objective = _EXPECTED_OBJECTIVES[method_id]
    actual_objective = tuple(
        (term.name, term.kind, tuple(term.views), float(term.weight))
        for term in method.objective.terms
    )
    if actual_objective != expected_objective:
        raise ValueError(
            f"{method_id} objective does not match the frozen scale-only contract"
        )

    actual_bn = method.contracts.get("batch_norm_objective_view_weights")
    if not isinstance(actual_bn, Mapping):
        raise ValueError(
            f"{method_id} BatchNorm objective-view weights must be a mapping"
        )
    if lowdose_method:
        assert lowdose_auxiliary_weight is not None
        _, _, auxiliary_views = lowdose_auxiliary_specs[method_id]
        auxiliary_view = auxiliary_views[-1]
        expected_bn = {
            "clean_view": 1.0 - lowdose_auxiliary_weight,
            auxiliary_view: lowdose_auxiliary_weight,
        }
    else:
        expected_bn = _EXPECTED_BN_WEIGHTS[method_id]
    if set(actual_bn) != set(expected_bn) or any(
        not _same_finite_float(actual_bn[name], expected)
        for name, expected in expected_bn.items()
    ):
        raise ValueError(
            f"{method_id} BatchNorm weights do not match the frozen objective"
        )
    expected_effective: Mapping[str, float] | None = None
    if method_id == VAE_LHAT_CLEAN_POLISH_METHOD_ID:
        expected_effective = MappingProxyType(
            {
                "clean_bce": 0.50,
                "lhat_direct_bce": 0.25,
                "clean_lhat_jsd": 0.25,
            }
        )
    elif method_id in {
        VAE_LHAT_CHAIN3_POLISH_METHOD_ID,
        CLEAN_CHAIN3_POLISH_CONTROL_METHOD_ID,
        J4_VAE_LHAT_CHAIN3_METHOD_ID,
        J4_CLEAN_CHAIN3_CONTROL_METHOD_ID,
    }:
        expected_effective = MappingProxyType(
            {
                "clean_bce": 0.50,
                "clean_preserve_bce": 0.25,
                "augmix_bce": 0.125,
                "consistency_jsd": 0.125,
            }
        )
    elif method_id == VAE_LHAT_TEACHER_DISTILL_METHOD_ID:
        assert lowdose_auxiliary_weight is not None
        expected_effective = MappingProxyType(
            {
                "clean_bce": 0.50,
                "clean_preserve_bce": 0.50 - lowdose_auxiliary_weight,
                "lhat_teacher_distill_bce": lowdose_auxiliary_weight,
            }
        )
    elif method_id == CLEAN_TEACHER_DISTILL_CONTROL_METHOD_ID:
        assert lowdose_auxiliary_weight is not None
        expected_effective = MappingProxyType(
            {
                "clean_bce": 0.50,
                "clean_preserve_bce": 0.50 - lowdose_auxiliary_weight,
                "clean_teacher_distill_control_bce": (
                    lowdose_auxiliary_weight
                ),
            }
        )
    elif method_id == VAE_LHAT_HARD_BCE_METHOD_ID:
        assert lowdose_auxiliary_weight is not None
        expected_effective = MappingProxyType(
            {
                "clean_bce": 0.50,
                "clean_preserve_bce": 0.50 - lowdose_auxiliary_weight,
                "lhat_hard_bce": lowdose_auxiliary_weight,
            }
        )
    elif method_id == CLEAN_HARD_BCE_CONTROL_METHOD_ID:
        assert lowdose_auxiliary_weight is not None
        expected_effective = MappingProxyType(
            {
                "clean_bce": 0.50,
                "clean_preserve_bce": 0.50 - lowdose_auxiliary_weight,
                "clean_hard_control_bce": lowdose_auxiliary_weight,
            }
        )
    elif method_id == VAE_LHAT_PATH_BCE_METHOD_ID:
        assert lowdose_auxiliary_weight is not None
        expected_effective = MappingProxyType(
            {
                "clean_bce": 0.50,
                "clean_preserve_bce": 0.50 - lowdose_auxiliary_weight,
                "lhat_path_hard_bce": lowdose_auxiliary_weight,
            }
        )
    elif method_id == CLEAN_LHAT_PATH_CONTROL_METHOD_ID:
        assert lowdose_auxiliary_weight is not None
        expected_effective = MappingProxyType(
            {
                "clean_bce": 0.50,
                "clean_preserve_bce": 0.50 - lowdose_auxiliary_weight,
                "clean_lhat_path_control_bce": lowdose_auxiliary_weight,
            }
        )
    elif method_id == VAE_LHAT_CONSISTENCY_METHOD_ID:
        assert lowdose_auxiliary_weight is not None
        expected_effective = MappingProxyType(
            {
                "clean_bce": 0.50,
                "clean_preserve_bce": 0.50 - lowdose_auxiliary_weight,
                "clean_lhat_consistency_jsd": lowdose_auxiliary_weight,
            }
        )
    elif method_id == CLEAN_CONSISTENCY_CONTROL_METHOD_ID:
        assert lowdose_auxiliary_weight is not None
        expected_effective = MappingProxyType(
            {
                "clean_bce": 0.50,
                "clean_preserve_bce": 0.50 - lowdose_auxiliary_weight,
                "clean_identity_consistency_jsd": lowdose_auxiliary_weight,
            }
        )
    elif method_id == VAE_LHAT_FEATURE_INVARIANCE_METHOD_ID:
        assert lowdose_auxiliary_weight is not None
        expected_effective = MappingProxyType(
            {
                "clean_bce": 0.50,
                "clean_preserve_bce": 0.50 - lowdose_auxiliary_weight,
                "clean_lhat_feature_cosine": lowdose_auxiliary_weight,
            }
        )
    elif method_id == CLEAN_FEATURE_INVARIANCE_CONTROL_METHOD_ID:
        assert lowdose_auxiliary_weight is not None
        expected_effective = MappingProxyType(
            {
                "clean_bce": 0.50,
                "clean_preserve_bce": 0.50 - lowdose_auxiliary_weight,
                "clean_identity_feature_cosine": lowdose_auxiliary_weight,
            }
        )
    elif method_id in LOCAL_ANCHOR_SOFT_CANDIDATE_IDS:
        assert lowdose_auxiliary_weight is not None
        expected_effective = MappingProxyType(
            {
                "clean_bce": 0.50,
                "clean_preserve_bce": 0.50 - lowdose_auxiliary_weight,
                "lhat_anchor_soft_bce": lowdose_auxiliary_weight,
            }
        )
    elif method_id in LOCAL_ANCHOR_SOFT_CONTROL_IDS:
        assert lowdose_auxiliary_weight is not None
        expected_effective = MappingProxyType(
            {
                "clean_bce": 0.50,
                "clean_preserve_bce": 0.50 - lowdose_auxiliary_weight,
                "clean_anchor_soft_control_bce": lowdose_auxiliary_weight,
            }
        )
    if expected_effective is not None:
        actual_effective = method.contracts.get("effective_objective_weights")
        if not isinstance(actual_effective, Mapping) or (
            set(actual_effective) != set(expected_effective)
            or any(
                not _same_finite_float(actual_effective[name], expected)
                for name, expected in expected_effective.items()
            )
        ):
            raise ValueError(
                f"{method_id} effective objective weights do not match the "
                "frozen scale-only contract"
            )
    if method_id in {
        J4_VAE_LHAT_CHAIN3_METHOD_ID,
        J4_CLEAN_CHAIN3_CONTROL_METHOD_ID,
    }:
        expected_teacher_contract = {
            "auxiliary_label_policy": (
                "teacher_mix_times_stage2_initializer_probability_plus_"
                "hard_remainder"
            ),
            "auxiliary_teacher_policy": (
                "frozen_stage2_initializer_snapshot_before_first_optimizer_step"
            ),
            "auxiliary_teacher_mix": 0.4,
            "auxiliary_teacher_scope": "augmix_bce_only",
            "non_augmix_bce_hard_labels_preserved": True,
        }
        for name, expected in expected_teacher_contract.items():
            actual = method.contracts.get(name)
            matches = (
                _same_finite_float(actual, expected)
                if isinstance(expected, float)
                else actual == expected
            )
            if not matches:
                raise ValueError(
                    f"{method_id} J4 teacher contract {name} drifted: {actual!r}"
                )
    if method_id in {
        VAE_LHAT_TEACHER_DISTILL_METHOD_ID,
        CLEAN_TEACHER_DISTILL_CONTROL_METHOD_ID,
    }:
        expected_scope = (
            "lhat_distill_bce_only"
            if method_id == VAE_LHAT_TEACHER_DISTILL_METHOD_ID
            else "clean_identity_distill_control_bce_only"
        )
        expected_teacher_contract = {
            "auxiliary_label_policy": (
                "frozen_stage2_initializer_probability_only"
            ),
            "auxiliary_teacher_policy": (
                "frozen_stage2_initializer_snapshot_before_first_optimizer_step"
            ),
            "auxiliary_teacher_mix": 1.0,
            "auxiliary_teacher_scope": expected_scope,
            "auxiliary_soft_bce_pos_weight": None,
            "clean_hard_label_bce_preserved": True,
        }
        for name, expected in expected_teacher_contract.items():
            actual = method.contracts.get(name)
            matches = (
                _same_finite_float(actual, expected)
                if isinstance(expected, float)
                else actual == expected
            )
            if not matches:
                raise ValueError(
                    f"{method_id} teacher-distill contract {name} drifted: "
                    f"{actual!r}"
                )
    if method_id in {
        VAE_LHAT_HARD_BCE_METHOD_ID,
        CLEAN_HARD_BCE_CONTROL_METHOD_ID,
        VAE_LHAT_PATH_BCE_METHOD_ID,
        CLEAN_LHAT_PATH_CONTROL_METHOD_ID,
    }:
        if (
            method.contracts.get("hard_auxiliary_target_policy")
            != "anchor_ground_truth_multihot"
        ):
            raise ValueError(
                f"{method_id} must use anchor ground-truth multihot targets"
            )
        for forbidden in (
            "auxiliary_label_policy",
            "auxiliary_teacher_policy",
            "auxiliary_teacher_mix",
            "auxiliary_teacher_scope",
            "auxiliary_soft_bce_pos_weight",
            "auxiliary_teacher_initializer_sha256",
        ):
            if forbidden in method.contracts:
                raise ValueError(
                    f"{method_id} hard-BCE contract unexpectedly contains "
                    f"teacher field {forbidden}"
                )
    if method_id in {
        VAE_LHAT_PATH_BCE_METHOD_ID,
        CLEAN_LHAT_PATH_CONTROL_METHOD_ID,
    }:
        expected_path_contract = {
            "enabled": True,
            "t_values": list(LHAT_PATH_T_VALUES),
            "sampling": "uniform_valid_per_eligible_record_per_epoch",
            "geometry": "standardized_clean_to_lhat_ray",
            "residual_correction": "linear_clean_hard_endpoint",
            "candidate_training_view": "one_sampled_path_point",
            "matched_compute_in_control": True,
        }
        actual_path_contract = method.contracts.get("lhat_path_polish")
        if not isinstance(actual_path_contract, Mapping) or dict(
            actual_path_contract
        ) != expected_path_contract:
            raise ValueError(
                f"{method_id} LHAT path-polish contract drifted"
            )
    if method_id in {
        VAE_LHAT_CONSISTENCY_METHOD_ID,
        CLEAN_CONSISTENCY_CONTROL_METHOD_ID,
    }:
        if (
            method.contracts.get("clean_lhat_consistency_policy")
            != "symmetric_bernoulli_jsd_current_model"
        ):
            raise ValueError(
                f"{method_id} must use current-model Bernoulli JSD consistency"
            )
        if method.contracts.get("lhat_direct_label_supervision") is not False:
            raise ValueError(
                f"{method_id} must not directly supervise the LHAT endpoint"
            )
        for forbidden in (
            "auxiliary_label_policy",
            "auxiliary_teacher_policy",
            "auxiliary_teacher_mix",
            "auxiliary_teacher_scope",
            "auxiliary_soft_bce_pos_weight",
            "auxiliary_teacher_initializer_sha256",
            "hard_auxiliary_target_policy",
        ):
            if forbidden in method.contracts:
                raise ValueError(
                    f"{method_id} consistency contract unexpectedly contains "
                    f"target field {forbidden}"
                )
    if method_id in {
        VAE_LHAT_FEATURE_INVARIANCE_METHOD_ID,
        CLEAN_FEATURE_INVARIANCE_CONTROL_METHOD_ID,
    }:
        expected_policy = {
            "space": "managed_classifier_penultimate_feature",
            "distance": "one_minus_cosine_similarity",
            "validity": "exact_matched_eligibility_intersection",
            "projection_head": False,
            "direct_lhat_label_supervision": False,
            "typed_objective_slot": (
                "bernoulli_jsd_replaced_before_objective_scaling"
            ),
            "matched_compute_in_control": True,
        }
        actual_policy = method.contracts.get("feature_invariance_polish")
        if not isinstance(actual_policy, Mapping) or dict(
            actual_policy
        ) != expected_policy:
            raise ValueError(
                f"{method_id} feature-invariance contract drifted"
            )
        for forbidden in (
            "auxiliary_label_policy",
            "auxiliary_teacher_policy",
            "auxiliary_teacher_mix",
            "auxiliary_teacher_scope",
            "auxiliary_soft_bce_pos_weight",
            "auxiliary_teacher_initializer_sha256",
            "hard_auxiliary_target_policy",
            "clean_lhat_consistency_policy",
        ):
            if forbidden in method.contracts:
                raise ValueError(
                    f"{method_id} feature-invariance contract unexpectedly "
                    f"contains field {forbidden}"
                )
    if method_id in LOCAL_ANCHOR_SOFT_METHOD_IDS:
        calibrated = method_id in {
            VAE_CALIBRATED_LOCAL_ANCHOR_SOFT_METHOD_ID,
            CLEAN_CALIBRATED_LOCAL_ANCHOR_SOFT_CONTROL_METHOD_ID,
        }
        expected_preset = (
            "historical_gate2_calibrated_v1"
            if calibrated
            else "local_anchor_soft_v1"
        )
        expected_policy = {
            "candidate_policy": "exact_positive_set_nonself",
            "candidate_count": 20,
            "candidate_includes_anchor": False,
            "anchor_in_outer_interpolation": True,
            "local_pool_size": 80,
            "hull_lambda": 0.20 if calibrated else 0.05,
            "pgd_epsilon_l2_standardized": 4.0 if calibrated else 2.0,
            "steps": 10 if calibrated else 3,
            "learning_rate": 0.40 if calibrated else 0.25,
            "minimum_effective_anchor_mass": 0.80 if calibrated else 0.95,
            "matched_compute_in_control": True,
        }
        actual_policy = method.contracts.get("local_anchor_soft_polish")
        if not isinstance(actual_policy, Mapping) or dict(
            actual_policy
        ) != expected_policy:
            raise ValueError(
                f"{method_id} local anchor-soft contract drifted"
            )
        expected_fields = {
            "local_lhat_geometry_preset": expected_preset,
            "anchor_soft_target_policy": (
                "anchor_soft_exact_positive_set_no_class_admission"
            ),
            "auxiliary_positive_value": 0.95,
            "auxiliary_negative_value": 0.0,
            "auxiliary_soft_bce_pos_weight": None,
            "clean_hard_label_bce_preserved": True,
        }
        for name, expected in expected_fields.items():
            actual = method.contracts.get(name)
            matches = (
                _same_finite_float(actual, expected)
                if isinstance(expected, float)
                else actual == expected
            )
            if not matches:
                raise ValueError(
                    f"{method_id} local anchor-soft field {name} drifted: "
                    f"{actual!r}"
                )
        for forbidden in (
            "auxiliary_label_policy",
            "auxiliary_teacher_policy",
            "auxiliary_teacher_mix",
            "auxiliary_teacher_scope",
            "auxiliary_teacher_initializer_sha256",
            "hard_auxiliary_target_policy",
            "clean_lhat_consistency_policy",
            "feature_invariance_polish",
        ):
            if forbidden in method.contracts:
                raise ValueError(
                    f"{method_id} local anchor-soft contract unexpectedly "
                    f"contains field {forbidden}"
                )
        control_executes_search = method.contracts.get(
            "control_executes_full_lhat_search"
        )
        if method_id in LOCAL_ANCHOR_SOFT_CONTROL_IDS:
            if control_executes_search is not True:
                raise ValueError(
                    f"{method_id} must execute the full LHAT search"
                )
        elif control_executes_search is not None:
            raise ValueError(
                f"{method_id} candidate unexpectedly declares control search"
            )


@contextmanager
def patch_online_trainer_runtime() -> Iterator[None]:
    """Validate allowlisted factory calls and scale objectives exactly once."""

    import core.online_trainer as online_trainer

    original_factory = online_trainer.build_method_runtime
    original_compute_objective = online_trainer._compute_objective

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
    ) -> MethodViewRuntime:
        if _declares_scale_only_adapter(method):
            validate_scale_only_method(method)
        if method.profile_name in {
            J4_VAE_LHAT_CHAIN3_METHOD_ID,
            J4_CLEAN_CHAIN3_CONTROL_METHOD_ID,
            VAE_LHAT_TEACHER_DISTILL_METHOD_ID,
            CLEAN_TEACHER_DISTILL_CONTROL_METHOD_ID,
            VAE_LHAT_HARD_BCE_METHOD_ID,
            CLEAN_HARD_BCE_CONTROL_METHOD_ID,
            VAE_LHAT_PATH_BCE_METHOD_ID,
            CLEAN_LHAT_PATH_CONTROL_METHOD_ID,
            VAE_LHAT_CONSISTENCY_METHOD_ID,
            CLEAN_CONSISTENCY_CONTROL_METHOD_ID,
            VAE_LHAT_FEATURE_INVARIANCE_METHOD_ID,
            CLEAN_FEATURE_INVARIANCE_CONTROL_METHOD_ID,
            VAE_LHAT_LOCAL_ANCHOR_SOFT_METHOD_ID,
            CLEAN_LOCAL_ANCHOR_SOFT_CONTROL_METHOD_ID,
            VAE_CALIBRATED_LOCAL_ANCHOR_SOFT_METHOD_ID,
            CLEAN_CALIBRATED_LOCAL_ANCHOR_SOFT_CONTROL_METHOD_ID,
        }:
            if latent_pool is None:
                raise ValueError(
                    "matched eligibility runtime requires a latent pool"
                )
            return _J4MatchedEligibilityRuntime(
                method,
                model_name=model_name,
                config_root=Path(config_root).expanduser().resolve(),
                latent_pool=latent_pool,
                encoder=encoder,
                decoder=decoder,
                minimum_std_mV=minimum_std_mV,
                maximum_abs_mV=maximum_abs_mV,
            )
        if (
            method.profile_name
            in {A5_M20_LHAT_AUGMIX_METHOD_ID, *PURE_M20_RUNTIME_METHOD_IDS}
            and latent_pool is not None
            and decoder is not None
        ):
            return _A5CalibratedMixRuntime(
                method,
                model_name=model_name,
                config_root=Path(config_root).expanduser().resolve(),
                latent_pool=latent_pool,
                encoder=encoder,
                decoder=decoder,
                minimum_std_mV=minimum_std_mV,
                maximum_abs_mV=maximum_abs_mV,
            )
        return original_factory(
            method,
            model_name=model_name,
            config_root=config_root,
            latent_pool=latent_pool,
            encoder=encoder,
            decoder=decoder,
            minimum_std_mV=minimum_std_mV,
            maximum_abs_mV=maximum_abs_mV,
        )

    def patched_compute_objective(**kwargs: Any):
        method = kwargs.get("method")
        should_scale = False
        feature_invariance = (
            isinstance(method, CompiledMethod)
            and method.profile_name
            in {
                VAE_LHAT_FEATURE_INVARIANCE_METHOD_ID,
                CLEAN_FEATURE_INVARIANCE_CONTROL_METHOD_ID,
            }
        )
        local_anchor_soft = (
            isinstance(method, CompiledMethod)
            and method.profile_name in LOCAL_ANCHOR_SOFT_METHOD_IDS
        )
        if isinstance(method, CompiledMethod) and _declares_scale_only_adapter(method):
            validate_scale_only_method(method)
            should_scale = True
        objective_kwargs = dict(kwargs)
        if (
            isinstance(method, CompiledMethod)
            and method.profile_name
            in {A5_M20_LHAT_AUGMIX_METHOD_ID, *PURE_M20_RUNTIME_METHOD_IDS}
        ):
            model = objective_kwargs.get("model")
            if not isinstance(model, nn.Module):
                raise TypeError("A5-M20 objective requires a torch.nn.Module model")
            objective_kwargs["model"] = _CheckpointedModelForward(model)
        capture_model: _PenultimateFeatureCaptureModel | None = None
        logit_capture_model: _LogitCaptureModel | None = None
        if feature_invariance:
            model = objective_kwargs.get("model")
            if not isinstance(model, nn.Module):
                raise TypeError(
                    "feature-invariance objective requires a torch.nn.Module model"
                )
            capture_model = _PenultimateFeatureCaptureModel(model)
            objective_kwargs["model"] = capture_model
        elif local_anchor_soft:
            model = objective_kwargs.get("model")
            if not isinstance(model, nn.Module):
                raise TypeError(
                    "local anchor-soft objective requires a torch.nn.Module model"
                )
            logit_capture_model = _LogitCaptureModel(model)
            objective_kwargs["model"] = logit_capture_model
        try:
            result = original_compute_objective(**objective_kwargs)
        finally:
            if capture_model is not None:
                capture_model.close()
        if feature_invariance:
            assert isinstance(method, CompiledMethod)
            assert capture_model is not None
            bundle = objective_kwargs.get("bundle")
            if not isinstance(bundle, ViewBundle):
                raise TypeError(
                    "feature-invariance objective requires a typed ViewBundle"
                )
            auxiliary_view_name = (
                "lhat_view"
                if method.profile_name
                == VAE_LHAT_FEATURE_INVARIANCE_METHOD_ID
                else "lhat_control_view"
            )
            auxiliary_term_name = (
                "clean_lhat_feature_cosine"
                if method.profile_name
                == VAE_LHAT_FEATURE_INVARIANCE_METHOD_ID
                else "clean_identity_feature_cosine"
            )
            clean = bundle.require("clean_view")
            auxiliary = bundle.require(auxiliary_view_name)
            if not isinstance(clean, WaveformView) or not isinstance(
                auxiliary, WaveformView
            ):
                raise TypeError(
                    "feature-invariance objective views must be WaveformView"
                )
            if len(capture_model.features) != 2:
                raise RuntimeError(
                    "feature-invariance objective expected exactly two model "
                    f"forwards, captured {len(capture_model.features)}"
                )
            clean_features, auxiliary_features = capture_model.features
            batch_size = clean.batch_size
            if (
                clean_features.ndim != 2
                or auxiliary_features.ndim != 2
                or clean_features.shape != auxiliary_features.shape
                or clean_features.shape[0] != batch_size
            ):
                raise RuntimeError(
                    "feature-invariance objective lost full-batch feature alignment"
                )
            common = clean.valid_mask & auxiliary.valid_mask
            positions = torch.nonzero(common, as_tuple=False).flatten()
            count = int(positions.numel())
            if count <= 0:
                raise RuntimeError(
                    "feature-invariance objective has no eligible records"
                )
            clean_selected = clean_features.index_select(0, positions).float()
            auxiliary_selected = auxiliary_features.index_select(
                0, positions
            ).float()
            cosine = F.cosine_similarity(
                clean_selected,
                auxiliary_selected,
                dim=1,
                eps=1.0e-8,
            )
            feature_loss = (1.0 - cosine).mean()
            if not bool(torch.isfinite(feature_loss)):
                raise FloatingPointError(
                    "feature-invariance cosine distance became NaN or Inf"
                )
            terms = {
                term.name: term for term in method.objective.terms
            }
            term = terms.get(auxiliary_term_name)
            if (
                term is None
                or term.kind != "bernoulli_jsd"
                or tuple(term.views)
                != ("clean_view", auxiliary_view_name)
            ):
                raise RuntimeError(
                    "feature-invariance typed objective slot drifted"
                )
            contribution = (
                float(term.weight)
                * (float(count) / float(batch_size))
                * feature_loss
            )
            raw_terms = dict(result.raw_terms)
            weighted_terms = dict(result.weighted_terms)
            valid_counts = dict(result.valid_counts)
            if (
                auxiliary_term_name not in raw_terms
                or auxiliary_term_name not in weighted_terms
                or valid_counts.get(auxiliary_term_name) != count
            ):
                raise RuntimeError(
                    "feature-invariance placeholder objective accounting drifted"
                )
            total = (
                result.total
                - weighted_terms[auxiliary_term_name]
                + contribution
            )
            raw_terms[auxiliary_term_name] = feature_loss
            weighted_terms[auxiliary_term_name] = contribution
            result_type = type(result)
            result = result_type(
                total=total,
                raw_terms=raw_terms,
                weighted_terms=weighted_terms,
                valid_counts=valid_counts,
            )
        if local_anchor_soft:
            assert isinstance(method, CompiledMethod)
            assert logit_capture_model is not None
            if objective_kwargs.get("pos_weight") is not None:
                raise ValueError(
                    "local anchor-soft auxiliary BCE forbids pos_weight"
                )
            bundle = objective_kwargs.get("bundle")
            if not isinstance(bundle, ViewBundle):
                raise TypeError(
                    "local anchor-soft objective requires a typed ViewBundle"
                )
            auxiliary_view_name = (
                "lhat_view"
                if method.profile_name in LOCAL_ANCHOR_SOFT_CANDIDATE_IDS
                else "lhat_control_view"
            )
            auxiliary_term_name = (
                "lhat_anchor_soft_bce"
                if method.profile_name in LOCAL_ANCHOR_SOFT_CANDIDATE_IDS
                else "clean_anchor_soft_control_bce"
            )
            clean = bundle.require("clean_view")
            auxiliary = bundle.require(auxiliary_view_name)
            if not isinstance(clean, WaveformView) or not isinstance(
                auxiliary, WaveformView
            ):
                raise TypeError(
                    "local anchor-soft objective views must be WaveformView"
                )
            if (
                clean.sample_ids != auxiliary.sample_ids
                or not torch.equal(clean.labels, auxiliary.labels)
                or not bool(clean.valid_mask.all())
            ):
                raise RuntimeError(
                    "local anchor-soft objective lost origin, label, or clean "
                    "validity alignment"
                )
            if len(logit_capture_model.logits) != 2:
                raise RuntimeError(
                    "local anchor-soft objective expected exactly two model "
                    f"forwards, captured {len(logit_capture_model.logits)}"
                )
            batch_size = clean.batch_size
            positions = torch.nonzero(
                auxiliary.valid_mask, as_tuple=False
            ).flatten()
            count = int(positions.numel())
            if count <= 0:
                raise RuntimeError(
                    "local anchor-soft objective has no eligible records"
                )
            auxiliary_logits = logit_capture_model.logits[1]
            if (
                not isinstance(auxiliary_logits, torch.Tensor)
                or auxiliary_logits.ndim != 2
                or auxiliary_logits.shape[1] != 5
            ):
                raise RuntimeError(
                    "local anchor-soft objective captured invalid logits"
                )
            if auxiliary_logits.shape[0] == batch_size:
                selected_logits = auxiliary_logits.index_select(0, positions)
            elif auxiliary_logits.shape[0] == count:
                selected_logits = auxiliary_logits
            else:
                raise RuntimeError(
                    "local anchor-soft auxiliary logits lost batch alignment"
                )
            anchor_targets = torch.where(
                clean.labels > 0.5,
                torch.full_like(clean.labels, 0.95),
                torch.zeros_like(clean.labels),
            )
            selected_targets = anchor_targets.index_select(0, positions).to(
                device=selected_logits.device,
                dtype=torch.float32,
            )
            soft_bce = F.binary_cross_entropy_with_logits(
                selected_logits.float(),
                selected_targets,
            )
            if not bool(torch.isfinite(soft_bce)):
                raise FloatingPointError(
                    "local anchor-soft BCE became NaN or Inf"
                )
            terms = {term.name: term for term in method.objective.terms}
            term = terms.get(auxiliary_term_name)
            if (
                term is None
                or term.kind != "bce"
                or tuple(term.views) != (auxiliary_view_name,)
            ):
                raise RuntimeError(
                    "local anchor-soft typed objective slot drifted"
                )
            contribution = (
                float(term.weight)
                * (float(count) / float(batch_size))
                * soft_bce
            )
            raw_terms = dict(result.raw_terms)
            weighted_terms = dict(result.weighted_terms)
            valid_counts = dict(result.valid_counts)
            if (
                auxiliary_term_name not in raw_terms
                or auxiliary_term_name not in weighted_terms
                or valid_counts.get(auxiliary_term_name) != count
            ):
                raise RuntimeError(
                    "local anchor-soft objective accounting drifted"
                )
            total = (
                result.total
                - weighted_terms[auxiliary_term_name]
                + contribution
            )
            raw_terms[auxiliary_term_name] = soft_bce
            weighted_terms[auxiliary_term_name] = contribution
            result_type = type(result)
            result = result_type(
                total=total,
                raw_terms=raw_terms,
                weighted_terms=weighted_terms,
                valid_counts=valid_counts,
            )
        if not should_scale:
            return result
        result_type = type(result)
        return result_type(
            total=result.total * OBJECTIVE_GLOBAL_SCALE,
            raw_terms=dict(result.raw_terms),
            weighted_terms={
                name: value * OBJECTIVE_GLOBAL_SCALE
                for name, value in result.weighted_terms.items()
            },
            valid_counts=dict(result.valid_counts),
        )

    online_trainer.build_method_runtime = patched_factory
    online_trainer._compute_objective = patched_compute_objective
    try:
        yield
    finally:
        online_trainer._compute_objective = original_compute_objective
        online_trainer.build_method_runtime = original_factory


install_objective_scale_adapter = patch_online_trainer_runtime


__all__ = [
    "D8_METHOD_ID",
    "D9_METHOD_ID",
    "R4_RESIDUAL_LATENT2_METHOD_ID",
    "R4_LHAT_AUGMIX_LOWDOSE_METHOD_ID",
    "R8_RESIDUAL_LATENT1_CLEAN55_METHOD_ID",
    "PURE_M20_LHAT_AUGMIX_METHOD_ID",
    "VAE_LHAT_BOUNDED_AUGMAX_POLISH_METHOD_ID",
    "CLEAN_BOUNDED_AUGMAX_CONTROL_METHOD_ID",
    "VAE_LHAT_CLEAN_POLISH_METHOD_ID",
    "VAE_LHAT_CHAIN3_POLISH_METHOD_ID",
    "CLEAN_CHAIN3_POLISH_CONTROL_METHOD_ID",
    "J4_VAE_LHAT_CHAIN3_METHOD_ID",
    "J4_CLEAN_CHAIN3_CONTROL_METHOD_ID",
    "VAE_LHAT_TEACHER_DISTILL_METHOD_ID",
    "CLEAN_TEACHER_DISTILL_CONTROL_METHOD_ID",
    "VAE_LHAT_HARD_BCE_METHOD_ID",
    "CLEAN_HARD_BCE_CONTROL_METHOD_ID",
    "VAE_LHAT_PATH_BCE_METHOD_ID",
    "CLEAN_LHAT_PATH_CONTROL_METHOD_ID",
    "LHAT_PATH_T_VALUES",
    "VAE_LHAT_CONSISTENCY_METHOD_ID",
    "CLEAN_CONSISTENCY_CONTROL_METHOD_ID",
    "VAE_LHAT_FEATURE_INVARIANCE_METHOD_ID",
    "CLEAN_FEATURE_INVARIANCE_CONTROL_METHOD_ID",
    "VAE_LHAT_LOCAL_ANCHOR_SOFT_METHOD_ID",
    "CLEAN_LOCAL_ANCHOR_SOFT_CONTROL_METHOD_ID",
    "VAE_CALIBRATED_LOCAL_ANCHOR_SOFT_METHOD_ID",
    "CLEAN_CALIBRATED_LOCAL_ANCHOR_SOFT_CONTROL_METHOD_ID",
    "OBJECTIVE_GLOBAL_SCALE",
    "SCALE_ONLY_METHOD_IDS",
    "install_objective_scale_adapter",
    "patch_online_trainer_runtime",
    "validate_scale_only_method",
]

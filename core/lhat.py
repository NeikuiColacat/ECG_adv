"""Repaired ECGTwin VAE latent-hull online adversarial sample generation.

The module owns attack geometry and diagnostics, not the outer training loop.
Only per-batch hull logits are optimized.  Classifier and decoder parameters
remain untouched while gradients flow through both models to those hull logits.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from models.vae import decode_to_ptbxl_waveform
from util.config_bundle import require_mapping as _mapping, resolve_config_reference, resolve_entry_config_path
from util.random_seed import load_random_seed_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LHAT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train" / "lhat.yaml"
ATTACK_OBJECTIVES = frozenset(
    {
        "maximize_multilabel_bce_with_logits",
        "maximize_equal_positive_negative_bce_with_logits",
        "maximize_bernoulli_kl_from_decoded_anchor",
        "compute_matched_uniform_latent_control",
    }
)
SUPPORTED_NUM_CANDIDATES = frozenset({5, 10, 20})


@dataclass(frozen=True)
class AttackDomain:
    sampling_rate_hz: int
    points: int


@dataclass(frozen=True)
class AttackThenContractConfig:
    enabled: bool
    version: str
    t_values: tuple[float, ...]
    margin_retention: float
    selection: str
    margin_reference: str
    endpoint_residual_correction: str
    raw_search_diagnostics_retained: bool
    uses_heldout_target: bool


@dataclass(frozen=True)
class LHATConfig:
    config_path: Path
    random_seed_config_path: Path
    random_namespace: str
    standardizer_epsilon: float
    num_candidates: int
    candidate_mode: str
    local_pool_size: int
    include_anchor: bool
    init_logit_gap: float
    hull_lambda: float
    steps: int
    learning_rate: float
    pgd_epsilon: float
    attack_weight_mode: str
    attack_objective: str
    normalization_epsilon: float
    canonical_domain: AttackDomain
    model_domains: dict[str, AttackDomain]
    attack_then_contract: AttackThenContractConfig


@dataclass(frozen=True)
class LatentStandardizer:
    """Coordinate-wise train-pool latent standardizer with stable identity."""

    mean: torch.Tensor
    scale: torch.Tensor
    count: int
    epsilon: float
    identity_sha256: str

    @classmethod
    def fit(
        cls,
        latents: torch.Tensor,
        *,
        epsilon: float = 1e-6,
    ) -> "LatentStandardizer":
        if not isinstance(latents, torch.Tensor) or latents.ndim < 2:
            raise TypeError("latents must be a rank-2-or-higher torch.Tensor")
        if latents.shape[0] < 2:
            raise ValueError("at least two train-pool latents are required")
        if not latents.is_floating_point() or not bool(torch.isfinite(latents).all()):
            raise ValueError("latents must be finite floating-point values")
        if float(epsilon) <= 0.0:
            raise ValueError("standardizer epsilon must be positive")
        values = latents.detach().to(dtype=torch.float32)
        mean = values.mean(dim=0, keepdim=True)
        scale = values.std(dim=0, correction=0, keepdim=True).clamp_min(
            float(epsilon)
        )
        digest = hashlib.sha256()
        digest.update(str(int(values.shape[0])).encode("ascii"))
        digest.update(mean.cpu().contiguous().numpy().tobytes())
        digest.update(scale.cpu().contiguous().numpy().tobytes())
        return cls(
            mean=mean,
            scale=scale,
            count=int(values.shape[0]),
            epsilon=float(epsilon),
            identity_sha256=digest.hexdigest(),
        )

    def _on(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.mean.to(device=value.device, dtype=value.dtype),
            self.scale.to(device=value.device, dtype=value.dtype),
        )

    def transform(self, value: torch.Tensor) -> torch.Tensor:
        mean, scale = self._on(value)
        return (value - mean) / scale

    def inverse_transform(self, value: torch.Tensor) -> torch.Tensor:
        mean, scale = self._on(value)
        return value * scale + mean

    def describe(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "latent_shape": list(self.mean.shape[1:]),
            "epsilon": self.epsilon,
            "identity_sha256": self.identity_sha256,
        }


@dataclass(frozen=True)
class LHATDiagnostics:
    raw_clean_bce: torch.Tensor
    decoded_anchor_bce: torch.Tensor
    initial_bce: torch.Tensor
    final_bce: torch.Tensor
    initial_attack_objective: torch.Tensor
    final_attack_objective: torch.Tensor
    attack_objective_gain: torch.Tensor
    reconstruction_delta: torch.Tensor
    adversarial_delta: torch.Tensor
    total_delta: torch.Tensor
    gain_vs_initial: torch.Tensor
    normalized_gain: torch.Tensor
    loss_gain: torch.Tensor
    atk_init_l2: torch.Tensor
    atk_anchor_l2: torch.Tensor
    sample_anyflip_eligible: torch.Tensor
    sample_anyflip_success: torch.Tensor
    positive_hide_numerator: torch.Tensor
    positive_hide_denominator: torch.Tensor
    negative_add_numerator: torch.Tensor
    negative_add_denominator: torch.Tensor
    coefficient_entropy: torch.Tensor
    coefficient_top1: torch.Tensor
    projection_scale: torch.Tensor
    effective_lambda: torch.Tensor
    effective_anchor_share: torch.Tensor
    decoded_invalid: torch.Tensor
    decoded_max_abs_mV: torch.Tensor

    def __post_init__(self) -> None:
        values = tuple(getattr(self, item.name) for item in fields(self))
        if not values or any(
            not isinstance(value, torch.Tensor) or value.ndim != 1
            for value in values
        ):
            raise TypeError("LHAT diagnostics must be rank-1 tensors")
        length = values[0].shape[0]
        if any(value.shape[0] != length for value in values):
            raise ValueError("LHAT diagnostic tensors must share one batch length")
        if any(value.requires_grad for value in values):
            raise ValueError("LHAT diagnostic tensors must be detached")

    def select(self, mask: torch.Tensor) -> "LHATDiagnostics":
        if (
            not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.bool
            or mask.ndim != 1
            or mask.shape[0] != self.raw_clean_bce.shape[0]
        ):
            raise ValueError("LHAT diagnostic selection mask must be bool shape (B,)")
        return LHATDiagnostics(
            **{
                item.name: getattr(self, item.name)[mask].detach()
                for item in fields(self)
            }
        )

    def sample_tensor_dict(self) -> dict[str, torch.Tensor]:
        return {
            "raw_clean_bce": self.raw_clean_bce,
            "decoded_anchor_bce": self.decoded_anchor_bce,
            "initial_bce": self.initial_bce,
            "final_bce": self.final_bce,
            "initial_attack_objective": self.initial_attack_objective,
            "final_attack_objective": self.final_attack_objective,
            "attack_objective_gain": self.attack_objective_gain,
            "reconstruction_delta": self.reconstruction_delta,
            "adversarial_delta": self.adversarial_delta,
            "total_delta": self.total_delta,
            "gain_vs_initial": self.gain_vs_initial,
            "normalized_gain": self.normalized_gain,
            "atk_init_l2": self.atk_init_l2,
            "atk_anchor_l2": self.atk_anchor_l2,
            "sample_anyflip_eligible": self.sample_anyflip_eligible.float(),
            "sample_anyflip_success": self.sample_anyflip_success.float(),
            "positive_hide_numerator": self.positive_hide_numerator.float(),
            "positive_hide_denominator": self.positive_hide_denominator.float(),
            "negative_add_numerator": self.negative_add_numerator.float(),
            "negative_add_denominator": self.negative_add_denominator.float(),
        }

    def mean_tensor_dict(self) -> dict[str, torch.Tensor]:
        ordinary = {
            "raw_clean_bce": self.raw_clean_bce,
            "decoded_anchor_bce": self.decoded_anchor_bce,
            "initial_bce": self.initial_bce,
            "final_bce": self.final_bce,
            "initial_attack_objective": self.initial_attack_objective,
            "final_attack_objective": self.final_attack_objective,
            "attack_objective_gain": self.attack_objective_gain,
            "reconstruction_delta": self.reconstruction_delta,
            "adversarial_delta": self.adversarial_delta,
            "total_delta": self.total_delta,
            "gain_vs_initial": self.gain_vs_initial,
            "normalized_gain": self.normalized_gain,
            # Retain the established name with its now-explicit meaning:
            # final hard BCE minus decoded-anchor BCE.
            "loss_gain": self.loss_gain,
            "atk_init_l2": self.atk_init_l2,
            "atk_anchor_l2": self.atk_anchor_l2,
            "coefficient_entropy": self.coefficient_entropy,
            "coefficient_top1": self.coefficient_top1,
            "projection_scale": self.projection_scale,
            "effective_lambda": self.effective_lambda,
            "effective_anchor_share": self.effective_anchor_share,
            "decoded_invalid_rate": self.decoded_invalid.float(),
            "decoded_max_abs_mV": self.decoded_max_abs_mV,
        }
        result = {
            name: value.float().mean().detach() for name, value in ordinary.items()
        }
        result.update(
            {
                "sample_anyflip_numerator": self.sample_anyflip_success.float()
                .sum()
                .detach(),
                "sample_anyflip_denominator": self.sample_anyflip_eligible.float()
                .sum()
                .detach(),
                "positive_hide_numerator": self.positive_hide_numerator.float()
                .sum()
                .detach(),
                "positive_hide_denominator": self.positive_hide_denominator.float()
                .sum()
                .detach(),
                "negative_add_numerator": self.negative_add_numerator.float()
                .sum()
                .detach(),
                "negative_add_denominator": self.negative_add_denominator.float()
                .sum()
                .detach(),
            }
        )
        return result


@dataclass(frozen=True)
class LHATResult:
    waveform_raw: torch.Tensor
    anchor_waveform_raw: torch.Tensor
    anchor_standardized: torch.Tensor
    latent_standardized: torch.Tensor
    weights: torch.Tensor
    diagnostics: LHATDiagnostics


@dataclass(frozen=True)
class AttackThenContractDiagnostics:
    accepted: torch.Tensor
    selected_t: torch.Tensor
    raw_clean_bce: torch.Tensor
    selected_bce: torch.Tensor
    bce_gain: torch.Tensor
    path_valid_count: torch.Tensor
    path_preserving_count: torch.Tensor
    clean_correct_class_count: torch.Tensor
    training_anyflip_success: torch.Tensor

    def __post_init__(self) -> None:
        values = tuple(getattr(self, item.name) for item in fields(self))
        if not values or any(
            not isinstance(value, torch.Tensor)
            or value.ndim != 1
            or value.requires_grad
            for value in values
        ):
            raise TypeError(
                "attack-then-contract diagnostics must be detached rank-1 tensors"
            )
        if any(value.shape != values[0].shape for value in values[1:]):
            raise ValueError(
                "attack-then-contract diagnostics must share one batch length"
            )

    def select(self, mask: torch.Tensor) -> "AttackThenContractDiagnostics":
        if (
            not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.bool
            or mask.ndim != 1
            or mask.shape != self.accepted.shape
        ):
            raise ValueError("contract diagnostic mask must be bool shape (B,)")
        return AttackThenContractDiagnostics(
            **{
                item.name: getattr(self, item.name)[mask].detach()
                for item in fields(self)
            }
        )

    def sample_tensor_dict(self) -> dict[str, torch.Tensor]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
        }

    def mean_tensor_dict(self) -> dict[str, torch.Tensor]:
        return {
            "contract_acceptance_rate": self.accepted.float().mean().detach(),
            "contract_selected_t": self.selected_t.float().mean().detach(),
            "contract_raw_clean_bce": self.raw_clean_bce.float().mean().detach(),
            "contract_selected_bce": self.selected_bce.float().mean().detach(),
            "contract_bce_gain": self.bce_gain.float().mean().detach(),
            "contract_path_valid_count": self.path_valid_count.float().mean().detach(),
            "contract_path_preserving_count": (
                self.path_preserving_count.float().mean().detach()
            ),
            "contract_clean_correct_class_count": (
                self.clean_correct_class_count.float().mean().detach()
            ),
            "contract_training_anyflip_numerator": (
                self.training_anyflip_success.float().sum().detach()
            ),
            "contract_training_anyflip_denominator": torch.tensor(
                float(self.training_anyflip_success.numel()),
                device=self.training_anyflip_success.device,
            ),
        }


@dataclass(frozen=True)
class AttackThenContractResult:
    waveform_raw: torch.Tensor
    valid_mask: torch.Tensor
    diagnostics: AttackThenContractDiagnostics


def load_lhat_config(
    path: str | Path = DEFAULT_LHAT_CONFIG_PATH,
    *,
    config_root: str | Path | None = None,
) -> LHATConfig:
    config_path = resolve_entry_config_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"LHAT config not found: {config_path}")
    payload = _mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")), "LHAT config"
    )
    expected_root_keys = {
        "schema_version",
        "method",
        "latent_standardization",
        "candidates",
        "hull_attack",
        "attack_then_contract",
        "decoder_bridge",
        "diagnostics",
    }
    if set(payload) != expected_root_keys:
        raise ValueError("LHAT config keys are incomplete or unexpected")
    if payload.get("schema_version") != 1:
        raise ValueError("LHAT config schema_version must be 1")
    standardization = _mapping(
        payload.get("latent_standardization"), "latent_standardization"
    )
    method = _mapping(payload.get("method"), "method")
    if set(method) != {"name", "random_seed_file", "random_namespace"}:
        raise ValueError("LHAT method keys are incomplete or unexpected")
    candidates = _mapping(payload.get("candidates"), "candidates")
    attack = _mapping(payload.get("hull_attack"), "hull_attack")
    raw_contract = _mapping(
        payload.get("attack_then_contract"), "attack_then_contract"
    )
    expected_contract_keys = {
        "enabled",
        "version",
        "t_values",
        "margin_retention",
        "selection",
        "margin_reference",
        "endpoint_residual_correction",
        "raw_search_diagnostics_retained",
        "uses_heldout_target",
    }
    if set(raw_contract) != expected_contract_keys:
        raise ValueError("attack_then_contract keys are incomplete or unexpected")
    expected_attack_keys = {
        "weight_mode",
        "init_logit_gap",
        "hull_lambda",
        "steps",
        "learning_rate",
        "pgd_epsilon_l2_standardized",
        "objective",
        "optimizer",
    }
    if set(attack) != expected_attack_keys:
        raise ValueError("LHAT hull_attack keys are incomplete or unexpected")
    bridge = _mapping(payload.get("decoder_bridge"), "decoder_bridge")
    if set(bridge) != {
        "native_points",
        "canonical_domain",
        "interpolation",
        "model_domains",
        "normalization_after_decode",
        "normalization_epsilon",
    }:
        raise ValueError("LHAT decoder_bridge keys are incomplete or unexpected")
    raw_canonical_domain = _mapping(
        bridge.get("canonical_domain"), "canonical_domain"
    )
    canonical_domain = AttackDomain(
        sampling_rate_hz=int(raw_canonical_domain.get("sampling_rate_hz", 0)),
        points=int(raw_canonical_domain.get("points", 0)),
    )
    raw_domains = _mapping(bridge.get("model_domains"), "model_domains")
    domains = {
        name: AttackDomain(
            sampling_rate_hz=int(_mapping(value, name).get("sampling_rate_hz", 0)),
            points=int(_mapping(value, name).get("points", 0)),
        )
        for name, value in raw_domains.items()
    }
    seed_config_path = resolve_config_reference(
        method.get("random_seed_file"),
        owner_config_path=config_path,
        config_root=config_root,
        description="method.random_seed_file",
        must_exist=True,
    )
    load_random_seed_config(seed_config_path)
    config = LHATConfig(
        config_path=config_path,
        random_seed_config_path=seed_config_path,
        random_namespace=str(method.get("random_namespace", "")),
        standardizer_epsilon=float(standardization.get("epsilon", 0.0)),
        num_candidates=int(candidates.get("num_candidates", 0)),
        candidate_mode=str(candidates.get("mode", "")),
        local_pool_size=int(candidates.get("local_pool_size", 0)),
        include_anchor=bool(candidates.get("include_anchor", True)),
        init_logit_gap=float(attack.get("init_logit_gap", float("nan"))),
        hull_lambda=float(attack.get("hull_lambda", 0.0)),
        steps=int(attack.get("steps", 0)),
        learning_rate=float(attack.get("learning_rate", 0.0)),
        pgd_epsilon=float(attack.get("pgd_epsilon_l2_standardized", 0.0)),
        attack_weight_mode=str(attack.get("weight_mode", "")),
        attack_objective=str(attack.get("objective", "")),
        normalization_epsilon=float(bridge.get("normalization_epsilon", 0.0)),
        canonical_domain=canonical_domain,
        model_domains=domains,
        attack_then_contract=AttackThenContractConfig(
            enabled=bool(raw_contract.get("enabled", False)),
            version=str(raw_contract.get("version", "")),
            t_values=tuple(float(value) for value in raw_contract.get("t_values", ())),
            margin_retention=float(raw_contract.get("margin_retention", 0.0)),
            selection=str(raw_contract.get("selection", "")),
            margin_reference=str(raw_contract.get("margin_reference", "")),
            endpoint_residual_correction=str(
                raw_contract.get("endpoint_residual_correction", "")
            ),
            raw_search_diagnostics_retained=bool(
                raw_contract.get("raw_search_diagnostics_retained", False)
            ),
            uses_heldout_target=bool(
                raw_contract.get("uses_heldout_target", True)
            ),
        ),
    )
    if candidates.get("label_policy") != "exact_positive_set":
        raise ValueError("main LHAT candidate policy must be exact_positive_set")
    if not config.random_namespace:
        raise ValueError("LHAT random namespace must be recorded")
    if config.num_candidates not in SUPPORTED_NUM_CANDIDATES:
        raise ValueError(
            "main LHAT contract requires num_candidates in "
            f"{sorted(SUPPORTED_NUM_CANDIDATES)}"
        )
    if config.include_anchor:
        raise ValueError("repaired LHAT contract requires include_anchor=false")
    if config.init_logit_gap != 0.0:
        raise ValueError("repaired LHAT contract requires init_logit_gap=0.0")
    if not 0.0 < config.hull_lambda <= 1.0:
        raise ValueError("hull_lambda must be in (0, 1]")
    if config.steps < 1 or config.learning_rate <= 0.0 or config.pgd_epsilon <= 0.0:
        raise ValueError("LHAT steps, learning_rate and pgd epsilon must be positive")
    if config.attack_weight_mode not in {
        "optimized_softmax",
        "uniform_compute_matched",
    }:
        raise ValueError("LHAT attack weight_mode is unsupported")
    if config.attack_objective not in ATTACK_OBJECTIVES:
        raise ValueError("LHAT attack objective is unsupported")
    if attack.get("optimizer") != "adam":
        raise ValueError("LHAT attack optimizer must be adam")
    is_uniform_control = config.attack_weight_mode == "uniform_compute_matched"
    if is_uniform_control != (
        config.attack_objective == "compute_matched_uniform_latent_control"
    ):
        raise ValueError(
            "uniform compute-matched weight mode and objective must be paired"
        )
    if config.candidate_mode not in {"nearest", "local_random"}:
        raise ValueError("candidate mode must be nearest or local_random")
    if config.local_pool_size < config.num_candidates:
        raise ValueError("local_pool_size must be at least num_candidates")
    expected_domains = {
        "efficientnet1dv2": AttackDomain(100, 1000),
        "ecgfounder": AttackDomain(500, 5000),
    }
    if domains != expected_domains:
        raise ValueError("LHAT model-domain contract does not match both backbones")
    if canonical_domain != AttackDomain(100, 1000):
        raise ValueError("LHAT decoded waveform must use the canonical 100 Hz domain")
    if bridge.get("interpolation") != "linear_align_corners":
        raise ValueError("LHAT domain bridges must use linear align_corners interpolation")
    contract = config.attack_then_contract
    if (
        not contract.enabled
        or contract.version != "preflip_maxloss_grid_v1"
        or contract.t_values != (0.25, 0.5, 0.75, 1.0)
        or contract.margin_retention != 0.5
        or contract.selection != "maximum_bce_without_new_clean_correct_flip"
        or contract.margin_reference != "signed_raw_clean_logit"
        or contract.endpoint_residual_correction
        != "linear_clean_hard_endpoint"
        or not contract.raw_search_diagnostics_retained
        or contract.uses_heldout_target
    ):
        raise ValueError("attack_then_contract does not match the frozen contract")
    return config


def _projected_hull(
    anchor: torch.Tensor,
    candidates: torch.Tensor,
    weight_logits: torch.Tensor,
    *,
    hull_lambda: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = torch.softmax(weight_logits, dim=1)
    view_shape = (weights.shape[0], weights.shape[1]) + (1,) * (
        candidates.ndim - 2
    )
    mixed = (weights.view(view_shape) * candidates).sum(dim=1)
    preprojected = anchor + float(hull_lambda) * (mixed - anchor)
    delta = preprojected - anchor
    norm = delta.flatten(1).norm(p=2, dim=1)
    scale = torch.clamp(float(epsilon) / norm.clamp_min(1e-12), max=1.0)
    broadcast = scale.view((scale.shape[0],) + (1,) * (anchor.ndim - 1))
    adversarial = anchor + delta * broadcast
    effective_lambda = scale * float(hull_lambda)
    return adversarial, weights, scale, effective_lambda


def _global_zscore_bct(raw_btc: torch.Tensor, epsilon: float) -> torch.Tensor:
    flat = raw_btc.flatten(1)
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, correction=0, keepdim=True).clamp_min(float(epsilon))
    normalized = (flat - mean) / std
    return normalized.view_as(raw_btc).transpose(1, 2).contiguous()


def _canonical_to_attack_domain(
    raw_canonical_btc: torch.Tensor,
    domain: AttackDomain,
    canonical_domain: AttackDomain,
) -> torch.Tensor:
    if tuple(raw_canonical_btc.shape[1:]) != (canonical_domain.points, 12):
        raise ValueError("decoded LHAT waveform is outside the canonical 100 Hz domain")
    if domain == canonical_domain:
        return raw_canonical_btc.contiguous()
    if domain != AttackDomain(500, 5000):
        raise ValueError("unsupported LHAT attack-domain bridge")
    return F.interpolate(
        raw_canonical_btc.transpose(1, 2),
        size=domain.points,
        mode="linear",
        align_corners=True,
    ).transpose(1, 2).contiguous()


def _bce_per_sample(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError(
            f"classifier logits/targets shape mismatch: {logits.shape}/{targets.shape}"
        )
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(
        dim=1
    )


def _equal_positive_negative_bce_per_sample(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Give each record's positive and negative label families equal mass."""

    if logits.shape != targets.shape:
        raise ValueError(
            f"classifier logits/targets shape mismatch: {logits.shape}/{targets.shape}"
        )
    elementwise = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    positive = targets >= 0.5
    negative = ~positive
    positive_count = positive.sum(dim=1)
    negative_count = negative.sum(dim=1)
    positive_mean = (elementwise * positive).sum(dim=1) / positive_count.clamp_min(1)
    negative_mean = (elementwise * negative).sum(dim=1) / negative_count.clamp_min(1)
    both = (positive_count > 0) & (negative_count > 0)
    return torch.where(
        both,
        0.5 * (positive_mean + negative_mean),
        torch.where(positive_count > 0, positive_mean, negative_mean),
    )


def _bernoulli_kl_from_reference_per_sample(
    logits: torch.Tensor,
    reference_logits: torch.Tensor,
) -> torch.Tensor:
    """Independent-Bernoulli KL(reference || current), averaged over labels."""

    if logits.shape != reference_logits.shape:
        raise ValueError(
            "classifier/reference logits shape mismatch: "
            f"{logits.shape}/{reference_logits.shape}"
        )
    reference = reference_logits.detach()
    probability = torch.sigmoid(reference)
    reference_log_positive = F.logsigmoid(reference)
    reference_log_negative = F.logsigmoid(-reference)
    current_log_positive = F.logsigmoid(logits)
    current_log_negative = F.logsigmoid(-logits)
    divergence = probability * (
        reference_log_positive - current_log_positive
    ) + (1.0 - probability) * (
        reference_log_negative - current_log_negative
    )
    return divergence.mean(dim=1)


def _attack_objective_per_sample(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    objective: str,
    reference_logits: torch.Tensor,
) -> torch.Tensor:
    if objective in {
        "maximize_multilabel_bce_with_logits",
        "compute_matched_uniform_latent_control",
    }:
        return _bce_per_sample(logits, targets)
    if objective == "maximize_equal_positive_negative_bce_with_logits":
        return _equal_positive_negative_bce_per_sample(logits, targets)
    if objective == "maximize_bernoulli_kl_from_decoded_anchor":
        return _bernoulli_kl_from_reference_per_sample(logits, reference_logits)
    raise ValueError(f"unsupported LHAT attack objective: {objective!r}")


def _freeze_parameter_gradients(
    *modules: nn.Module,
) -> tuple[tuple[nn.Parameter, bool], ...]:
    """Freeze module weights while preserving gradients to their inputs."""

    seen: set[int] = set()
    states: list[tuple[nn.Parameter, bool]] = []
    for module in modules:
        for parameter in module.parameters():
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            states.append((parameter, bool(parameter.requires_grad)))
            parameter.requires_grad_(False)
    return tuple(states)


def _restore_parameter_gradients(
    states: tuple[tuple[nn.Parameter, bool], ...],
) -> None:
    for parameter, requires_grad in states:
        parameter.requires_grad_(requires_grad)


def generate_lhat_adversarial(
    *,
    classifier: nn.Module,
    decoder: nn.Module,
    anchor_standardized: torch.Tensor,
    candidates_standardized: torch.Tensor,
    raw_clean_waveform: torch.Tensor,
    targets: torch.Tensor,
    standardizer: LatentStandardizer,
    model_name: str,
    config: LHATConfig | None = None,
) -> LHATResult:
    """Generate one VAE-LHAT raw waveform batch for AugMix chain 3."""

    resolved = load_lhat_config() if config is None else config
    if model_name not in resolved.model_domains:
        raise ValueError(f"unsupported LHAT model_name: {model_name!r}")
    domain = resolved.model_domains[model_name]
    if anchor_standardized.ndim < 2 or candidates_standardized.ndim != (
        anchor_standardized.ndim + 1
    ):
        raise ValueError("anchor/candidate latent ranks are inconsistent")
    batch = anchor_standardized.shape[0]
    if candidates_standardized.shape[:2] != (batch, resolved.num_candidates):
        raise ValueError(
            f"candidates must have shape (B,{resolved.num_candidates},...), "
            f"got {tuple(candidates_standardized.shape)}"
        )
    if tuple(candidates_standardized.shape[2:]) != tuple(
        anchor_standardized.shape[1:]
    ):
        raise ValueError("candidate latent shape must match anchor latent shape")
    if targets.shape != (batch, 5):
        raise ValueError("LHAT targets must have shape (B,5)")
    if tuple(raw_clean_waveform.shape) != (
        batch,
        resolved.canonical_domain.points,
        12,
    ):
        raise ValueError("LHAT raw_clean_waveform must be canonical shape (B,1000,12)")
    inputs_finite = (
        torch.isfinite(anchor_standardized).all()
        & torch.isfinite(candidates_standardized).all()
        & torch.isfinite(raw_clean_waveform).all()
        & torch.isfinite(targets).all()
    )
    if not bool(inputs_finite.item()):
        raise ValueError("LHAT inputs contain NaN or Inf")

    classifier_was_training = classifier.training
    decoder_was_training = decoder.training
    parameter_grad_states = _freeze_parameter_gradients(classifier, decoder)
    classifier.eval()
    decoder.eval()
    weight_logits = torch.zeros(
        batch,
        resolved.num_candidates,
        device=anchor_standardized.device,
        dtype=anchor_standardized.dtype,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([weight_logits], lr=resolved.learning_rate)

    def decoded_from_standardized(value: torch.Tensor) -> torch.Tensor:
        latent = standardizer.inverse_transform(value)
        return decode_to_ptbxl_waveform(decoder, latent)

    def logits_from_standardized(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_canonical = decoded_from_standardized(value)
        raw_model = _canonical_to_attack_domain(
            raw_canonical,
            domain,
            resolved.canonical_domain,
        )
        model_input = _global_zscore_bct(raw_model, resolved.normalization_epsilon)
        return classifier(model_input), raw_canonical

    try:
        with torch.no_grad():
            raw_clean_model = _canonical_to_attack_domain(
                raw_clean_waveform,
                domain,
                resolved.canonical_domain,
            )
            raw_clean_logits = classifier(
                _global_zscore_bct(
                    raw_clean_model,
                    resolved.normalization_epsilon,
                )
            )
            raw_clean_bce = _bce_per_sample(raw_clean_logits, targets)
            initial_latent, _, _, _ = _projected_hull(
                anchor_standardized,
                candidates_standardized,
                weight_logits,
                hull_lambda=resolved.hull_lambda,
                epsilon=resolved.pgd_epsilon,
            )
            # Both probes are inference-only and independent along the batch
            # axis.  Decode/classify them in one 2B launch to reduce kernel
            # launch overhead while preserving the exact anchor/initial
            # tensors, ordering and loss definitions.
            paired_logits, paired_raw = logits_from_standardized(
                torch.cat((anchor_standardized, initial_latent), dim=0)
            )
            anchor_logits, initial_logits = paired_logits.split(batch, dim=0)
            anchor_raw, _initial_raw = paired_raw.split(batch, dim=0)
            decoded_anchor_bce = _bce_per_sample(anchor_logits, targets)
            initial_bce = _bce_per_sample(initial_logits, targets)
            initial_attack_objective = _attack_objective_per_sample(
                initial_logits,
                targets,
                objective=resolved.attack_objective,
                reference_logits=anchor_logits,
            )

        for _ in range(resolved.steps):
            optimizer.zero_grad(set_to_none=True)
            attack_latent, _, _, _ = _projected_hull(
                anchor_standardized,
                candidates_standardized,
                weight_logits,
                hull_lambda=resolved.hull_lambda,
                epsilon=resolved.pgd_epsilon,
            )
            attack_logits, _ = logits_from_standardized(attack_latent)
            objective = _attack_objective_per_sample(
                attack_logits,
                targets,
                objective=resolved.attack_objective,
                reference_logits=anchor_logits,
            ).mean()
            (gradient,) = torch.autograd.grad(objective, weight_logits)
            weight_logits.grad = -gradient
            optimizer.step()

        with torch.no_grad():
            final_latent, weights, projection_scale, effective_lambda = (
                _projected_hull(
                    anchor_standardized,
                    candidates_standardized,
                    weight_logits,
                    hull_lambda=resolved.hull_lambda,
                    epsilon=resolved.pgd_epsilon,
                )
            )
            final_logits, final_raw = logits_from_standardized(final_latent)
            final_bce = _bce_per_sample(final_logits, targets)
            final_attack_objective = _attack_objective_per_sample(
                final_logits,
                targets,
                objective=resolved.attack_objective,
                reference_logits=anchor_logits,
            )
            final_prediction = torch.sigmoid(final_logits) >= 0.5
            anchor_prediction = torch.sigmoid(anchor_logits) >= 0.5
            truth = targets >= 0.5
            decoded_anchor_correct = anchor_prediction == truth
            final_wrong = final_prediction != truth
            sample_anyflip_eligible = decoded_anchor_correct.all(dim=1)
            sample_anyflip_success = sample_anyflip_eligible & final_wrong.any(dim=1)
            positive_hide_eligible = truth & anchor_prediction
            negative_add_eligible = (~truth) & (~anchor_prediction)
            positive_hide_numerator = (
                positive_hide_eligible & (~final_prediction)
            ).sum(dim=1)
            positive_hide_denominator = positive_hide_eligible.sum(dim=1)
            negative_add_numerator = (
                negative_add_eligible & final_prediction
            ).sum(dim=1)
            negative_add_denominator = negative_add_eligible.sum(dim=1)
            coefficient_entropy = -(
                weights * weights.clamp_min(1e-12).log()
            ).sum(dim=1)
            decoded_invalid = ~torch.isfinite(final_raw).flatten(1).all(dim=1)
            decoded_max_abs = torch.nan_to_num(final_raw.abs(), nan=float("inf")).flatten(
                1
            ).amax(dim=1)
            diagnostics = LHATDiagnostics(
                raw_clean_bce=raw_clean_bce.detach(),
                decoded_anchor_bce=decoded_anchor_bce.detach(),
                initial_bce=initial_bce.detach(),
                final_bce=final_bce.detach(),
                initial_attack_objective=initial_attack_objective.detach(),
                final_attack_objective=final_attack_objective.detach(),
                attack_objective_gain=(
                    final_attack_objective - initial_attack_objective
                ).detach(),
                reconstruction_delta=(decoded_anchor_bce - raw_clean_bce).detach(),
                adversarial_delta=(final_bce - decoded_anchor_bce).detach(),
                total_delta=(final_bce - raw_clean_bce).detach(),
                gain_vs_initial=(final_bce - initial_bce).detach(),
                normalized_gain=(
                    (final_bce - decoded_anchor_bce)
                    / decoded_anchor_bce.clamp_min(1.0e-6)
                ).detach(),
                loss_gain=(final_bce - decoded_anchor_bce).detach(),
                atk_init_l2=(final_latent - initial_latent)
                .flatten(1)
                .norm(p=2, dim=1)
                .detach(),
                atk_anchor_l2=(final_latent - anchor_standardized)
                .flatten(1)
                .norm(p=2, dim=1)
                .detach(),
                sample_anyflip_eligible=sample_anyflip_eligible.detach(),
                sample_anyflip_success=sample_anyflip_success.detach(),
                positive_hide_numerator=positive_hide_numerator.detach(),
                positive_hide_denominator=positive_hide_denominator.detach(),
                negative_add_numerator=negative_add_numerator.detach(),
                negative_add_denominator=negative_add_denominator.detach(),
                coefficient_entropy=coefficient_entropy.detach(),
                coefficient_top1=weights.max(dim=1).values.detach(),
                projection_scale=projection_scale.detach(),
                effective_lambda=effective_lambda.detach(),
                effective_anchor_share=(1.0 - effective_lambda).detach(),
                decoded_invalid=decoded_invalid.detach(),
                decoded_max_abs_mV=decoded_max_abs.detach(),
            )
            return LHATResult(
                waveform_raw=final_raw.detach().contiguous(),
                anchor_waveform_raw=anchor_raw.detach().contiguous(),
                anchor_standardized=anchor_standardized.detach().contiguous(),
                latent_standardized=final_latent.detach().contiguous(),
                weights=weights.detach().contiguous(),
                diagnostics=diagnostics,
            )
    finally:
        _restore_parameter_gradients(parameter_grad_states)
        classifier.train(classifier_was_training)
        decoder.train(decoder_was_training)


def apply_linear_endpoint_residual_correction(
    decoded_paths: torch.Tensor,
    *,
    t_values: torch.Tensor,
    raw_clean_waveform: torch.Tensor,
    decoded_anchor_waveform: torch.Tensor,
    raw_hard_waveform: torch.Tensor,
    decoded_hard_waveform: torch.Tensor,
) -> torch.Tensor:
    """Make the decoded latent path meet both observed waveform endpoints."""

    if decoded_paths.ndim != 4:
        raise ValueError("decoded_paths must have shape (B,T,time,12)")
    batch, count, points, leads = decoded_paths.shape
    expected = (batch, points, leads)
    endpoints = (
        raw_clean_waveform,
        decoded_anchor_waveform,
        raw_hard_waveform,
        decoded_hard_waveform,
    )
    if any(tuple(value.shape) != expected for value in endpoints):
        raise ValueError("residual-correction endpoint shapes are inconsistent")
    if tuple(t_values.shape) != (count,):
        raise ValueError("t_values must contain one value per decoded path point")
    t = t_values.to(
        device=decoded_paths.device,
        dtype=decoded_paths.dtype,
    ).view(1, count, 1, 1)
    clean_residual = raw_clean_waveform - decoded_anchor_waveform
    hard_residual = raw_hard_waveform - decoded_hard_waveform
    return (
        decoded_paths
        + (1.0 - t) * clean_residual[:, None]
        + t * hard_residual[:, None]
    ).contiguous()


def contract_lhat_adversarial(
    *,
    classifier: nn.Module,
    decoder: nn.Module,
    attack: LHATResult,
    raw_clean_waveform: torch.Tensor,
    targets: torch.Tensor,
    standardizer: LatentStandardizer,
    model_name: str,
    minimum_std_mV: float,
    maximum_abs_mV: float,
    config: LHATConfig | None = None,
) -> AttackThenContractResult:
    """Select the hardest label-preserving point on the clean-to-hard path."""

    resolved = load_lhat_config() if config is None else config
    contract = resolved.attack_then_contract
    if not contract.enabled:
        raise ValueError("attack-then-contract is disabled")
    if model_name not in resolved.model_domains:
        raise ValueError(f"unsupported LHAT model_name: {model_name!r}")
    domain = resolved.model_domains[model_name]
    batch = int(raw_clean_waveform.shape[0])
    if tuple(raw_clean_waveform.shape) != (
        batch,
        resolved.canonical_domain.points,
        12,
    ):
        raise ValueError("contract raw clean waveform must be canonical (B,1000,12)")
    if targets.shape != (batch, 5):
        raise ValueError("contract targets must have shape (B,5)")
    if attack.anchor_standardized.shape != attack.latent_standardized.shape:
        raise ValueError("contract latent endpoints must have identical shape")
    if attack.anchor_standardized.shape[0] != batch:
        raise ValueError("contract latent batch differs from raw clean batch")
    if minimum_std_mV <= 0.0 or maximum_abs_mV <= 0.0:
        raise ValueError("contract quality thresholds must be positive")

    classifier_was_training = classifier.training
    decoder_was_training = decoder.training
    parameter_grad_states = _freeze_parameter_gradients(classifier, decoder)
    classifier.eval()
    decoder.eval()
    t_values = torch.tensor(
        contract.t_values,
        device=attack.latent_standardized.device,
        dtype=attack.latent_standardized.dtype,
    )
    path_count = int(t_values.numel())
    latent_delta = attack.latent_standardized - attack.anchor_standardized
    path_standardized = (
        attack.anchor_standardized[:, None]
        + t_values.view((1, path_count) + (1,) * (latent_delta.ndim - 1))
        * latent_delta[:, None]
    )
    try:
        with torch.no_grad():
            decoded_paths = decode_to_ptbxl_waveform(
                decoder,
                standardizer.inverse_transform(
                    path_standardized.flatten(0, 1)
                ),
            ).reshape(
                batch,
                path_count,
                resolved.canonical_domain.points,
                12,
            )
            corrected = apply_linear_endpoint_residual_correction(
                decoded_paths,
                t_values=t_values,
                raw_clean_waveform=raw_clean_waveform,
                decoded_anchor_waveform=attack.anchor_waveform_raw,
                raw_hard_waveform=attack.waveform_raw,
                decoded_hard_waveform=decoded_paths[:, -1],
            )
            flat = corrected.flatten(2)
            finite = torch.isfinite(flat).all(dim=2)
            safe = torch.nan_to_num(
                flat, nan=0.0, posinf=0.0, neginf=0.0
            )
            valid = (
                finite
                & (safe.std(dim=2, correction=0) >= float(minimum_std_mV))
                & (safe.abs().amax(dim=2) <= float(maximum_abs_mV))
            )

            clean_model = _canonical_to_attack_domain(
                raw_clean_waveform,
                domain,
                resolved.canonical_domain,
            )
            clean_logits = classifier(
                _global_zscore_bct(
                    clean_model, resolved.normalization_epsilon
                )
            )
            path_model = _canonical_to_attack_domain(
                corrected.flatten(0, 1),
                domain,
                resolved.canonical_domain,
            )
            path_logits = classifier(
                _global_zscore_bct(
                    path_model, resolved.normalization_epsilon
                )
            ).reshape(batch, path_count, 5)
            expanded_targets = targets[:, None].expand(-1, path_count, -1)
            path_bce = F.binary_cross_entropy_with_logits(
                path_logits,
                expanded_targets,
                reduction="none",
            ).mean(dim=2)
            clean_bce = _bce_per_sample(clean_logits, targets)

            truth_sign = targets.mul(2.0).sub(1.0)
            clean_signed = clean_logits * truth_sign
            path_signed = path_logits * truth_sign[:, None]
            clean_correct = clean_signed >= 0.0
            retained_margin = (
                clean_signed.clamp_min(0.0)
                * float(contract.margin_retention)
            )
            preserving = (
                (~clean_correct[:, None])
                | (path_signed >= retained_margin[:, None])
            ).all(dim=2)
            allowed = valid & preserving
            ranked = path_bce + t_values.view(1, -1) * 1.0e-8
            masked = ranked.masked_fill(~allowed, -torch.inf)
            accepted = allowed.any(dim=1)
            selected_index = masked.argmax(dim=1)
            rows = torch.arange(batch, device=corrected.device)
            selected = corrected[rows, selected_index]
            selected_logits = path_logits[rows, selected_index]
            selected_bce = path_bce[rows, selected_index]
            selected_t = t_values.index_select(0, selected_index)
            selected = torch.where(
                accepted.view(-1, 1, 1),
                selected,
                raw_clean_waveform,
            )
            selected_bce = torch.where(accepted, selected_bce, clean_bce)
            selected_t = torch.where(
                accepted, selected_t, torch.zeros_like(selected_t)
            )
            selected_prediction = selected_logits >= 0.0
            clean_prediction = clean_logits >= 0.0
            training_flip = (
                clean_correct
                & (selected_prediction != clean_prediction)
            ).any(dim=1)
            training_flip &= accepted
            diagnostics = AttackThenContractDiagnostics(
                accepted=accepted.detach(),
                selected_t=selected_t.detach(),
                raw_clean_bce=clean_bce.detach(),
                selected_bce=selected_bce.detach(),
                bce_gain=(selected_bce - clean_bce).detach(),
                path_valid_count=valid.sum(dim=1).detach(),
                path_preserving_count=(valid & preserving).sum(dim=1).detach(),
                clean_correct_class_count=clean_correct.sum(dim=1).detach(),
                training_anyflip_success=training_flip.detach(),
            )
            return AttackThenContractResult(
                waveform_raw=selected.detach().contiguous(),
                valid_mask=accepted.detach(),
                diagnostics=diagnostics,
            )
    finally:
        _restore_parameter_gradients(parameter_grad_states)
        classifier.train(classifier_was_training)
        decoder.train(decoder_was_training)


__all__ = [
    "LHATConfig",
    "LatentStandardizer",
    "contract_lhat_adversarial",
    "generate_lhat_adversarial",
    "load_lhat_config",
]

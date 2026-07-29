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
from util.config_bundle import resolve_config_reference, resolve_entry_config_path
from util.random_seed import load_random_seed_config, make_torch_generator


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


@dataclass(frozen=True)
class AttackDomain:
    sampling_rate_hz: int
    points: int


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

    def mean_dict(self) -> dict[str, float]:
        tensors = self.mean_tensor_dict()
        names = tuple(tensors)
        values = torch.stack(
            [tensors[name].float() for name in names]
        ).detach().cpu().tolist()
        return dict(zip(names, (float(value) for value in values), strict=True))


@dataclass(frozen=True)
class LHATResult:
    waveform_raw: torch.Tensor
    anchor_waveform_raw: torch.Tensor
    latent_standardized: torch.Tensor
    weights: torch.Tensor
    diagnostics: LHATDiagnostics


def _mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a mapping")
    return value


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
    )
    if candidates.get("label_policy") != "exact_positive_set":
        raise ValueError("main LHAT candidate policy must be exact_positive_set")
    if not config.random_namespace:
        raise ValueError("LHAT random namespace must be recorded")
    if config.num_candidates != 20:
        raise ValueError("main LHAT contract requires exactly 20 candidates")
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
    return config


def make_lhat_generator(
    device: str | torch.device,
    *identity: Any,
    config: LHATConfig | None = None,
) -> torch.Generator:
    """Create the isolated candidate-sampling RNG declared by LHAT YAML."""

    resolved = load_lhat_config() if config is None else config
    return make_torch_generator(
        device,
        resolved.random_namespace,
        *identity,
        config_path=resolved.random_seed_config_path,
    )


def _validate_generator(generator: torch.Generator, device: torch.device) -> None:
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    generator_device = torch.device(generator.device)
    if generator_device.type != device.type:
        raise ValueError("generator device must match latent device")
    if device.type == "cuda":
        current_index = torch.cuda.current_device()
        latent_index = current_index if device.index is None else device.index
        generator_index = (
            current_index
            if generator_device.index is None
            else generator_device.index
        )
        if generator_index != latent_index:
            raise ValueError("generator CUDA index must match latent CUDA index")


def select_exact_label_candidates(
    standardized_latents: torch.Tensor,
    labels: torch.Tensor,
    *,
    anchor_indices: torch.Tensor,
    num_candidates: int,
    mode: str,
    local_pool_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Select distinct exact-label, non-self candidates from a train-only pool."""

    if standardized_latents.ndim < 2 or labels.ndim != 2:
        raise ValueError("latent pool and labels must have batch axes")
    if standardized_latents.shape[0] != labels.shape[0]:
        raise ValueError("latent pool and labels must have equal record counts")
    if anchor_indices.ndim != 1:
        raise ValueError("anchor_indices must be one-dimensional")
    if mode not in {"nearest", "local_random"}:
        raise ValueError("mode must be nearest or local_random")
    if int(num_candidates) < 1 or int(local_pool_size) < int(num_candidates):
        raise ValueError("candidate counts are invalid")
    if not bool(torch.isfinite(standardized_latents).all()):
        raise ValueError("standardized latent pool contains NaN or Inf")
    _validate_generator(generator, standardized_latents.device)
    labels_bool = labels > 0.5
    outputs: list[torch.Tensor] = []
    for raw_anchor in anchor_indices.tolist():
        anchor = int(raw_anchor)
        if not 0 <= anchor < standardized_latents.shape[0]:
            raise IndexError(f"anchor index out of range: {anchor}")
        eligible = torch.all(labels_bool == labels_bool[anchor], dim=1)
        eligible[anchor] = False
        pool = torch.nonzero(eligible, as_tuple=False).flatten()
        if pool.numel() < int(num_candidates):
            raise ValueError(
                f"anchor {anchor} has {pool.numel()} distinct non-self candidates; "
                f"requires {num_candidates}"
            )
        delta = standardized_latents[pool] - standardized_latents[anchor]
        distance = delta.flatten(1).norm(p=2, dim=1)
        ordered = pool[torch.argsort(distance, stable=True)]
        local = ordered[: min(int(local_pool_size), int(ordered.numel()))]
        if mode == "nearest":
            selected = local[: int(num_candidates)]
        else:
            order = torch.randperm(
                local.numel(), device=local.device, generator=generator
            )
            selected = local[order[: int(num_candidates)]]
        outputs.append(selected)
    return torch.stack(outputs, dim=0)


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
        return decode_to_ptbxl_waveform(
            decoder, latent, target_points=resolved.canonical_domain.points
        )

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
                latent_standardized=final_latent.detach().contiguous(),
                weights=weights.detach().contiguous(),
                diagnostics=diagnostics,
            )
    finally:
        classifier.train(classifier_was_training)
        decoder.train(decoder_was_training)


__all__ = [
    "AttackDomain",
    "DEFAULT_LHAT_CONFIG_PATH",
    "LHATConfig",
    "LHATDiagnostics",
    "LHATResult",
    "LatentStandardizer",
    "generate_lhat_adversarial",
    "load_lhat_config",
    "make_lhat_generator",
    "select_exact_label_candidates",
]

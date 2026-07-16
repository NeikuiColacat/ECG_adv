"""Three-chain ECG AugMix sample generation for online adversarial training.

Chains 1 and 2 are independent configured depth-2/3 corruption compositions.
Chain 3 is the VAE-LHAT raw waveform exactly as supplied: no extra corruption
is permitted on that branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import torch
import yaml

from util.config_bundle import resolve_config_reference, resolve_entry_config_path
from util.random_seed import load_random_seed_config, make_torch_generator
from util.augmentations.torch_operators import (
    baseline_shift,
    baseline_wander,
    emg_noise,
    powerline_noise,
    random_leads_masking,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUGMIX_CONFIG_PATH = PROJECT_ROOT / "configs" / "train" / "augmix.yaml"
CANONICAL_OPERATORS = (
    "powerline_noise",
    "emg_noise",
    "baseline_wander",
    "baseline_shift",
    "random_leads_masking",
)
OPERATOR_FUNCTIONS: dict[str, Callable[..., torch.Tensor]] = {
    "powerline_noise": powerline_noise,
    "emg_noise": emg_noise,
    "baseline_wander": baseline_wander,
    "baseline_shift": baseline_shift,
    "random_leads_masking": random_leads_masking,
}


@dataclass(frozen=True)
class AugMixConfig:
    config_path: Path
    random_seed_config_path: Path
    random_namespace: str
    width: int
    chain_roles: tuple[str, str, str]
    allowed_sampling_rates_hz: tuple[int, ...]
    operator_config_path: Path
    operator_profile: str
    severity: int
    depths: tuple[int, ...]
    canonical_operators: tuple[str, ...]
    dirichlet_alpha: float
    beta_alpha: float
    vae_chain_weight_cap: float | None
    normalization_epsilon: float


@dataclass(frozen=True)
class AugMixBatch:
    chain1_raw: torch.Tensor
    chain2_raw: torch.Tensor
    chain3_raw: torch.Tensor
    mixed_raw: torch.Tensor
    mixed_normalized: torch.Tensor
    mixture_weights: torch.Tensor
    augmented_strength: torch.Tensor
    chain1_composition_index: torch.Tensor
    chain2_composition_index: torch.Tensor
    chain1_depth: torch.Tensor
    chain2_depth: torch.Tensor


def _mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a mapping")
    return value


def load_augmix_config(
    path: str | Path = DEFAULT_AUGMIX_CONFIG_PATH,
    *,
    config_root: str | Path | None = None,
) -> AugMixConfig:
    config_path = resolve_entry_config_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"AugMix config not found: {config_path}")
    payload = _mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")), "AugMix config"
    )
    if payload.get("schema_version") != 1:
        raise ValueError("AugMix config schema_version must be 1")
    method = _mapping(payload.get("method"), "method")
    topology = _mapping(payload.get("topology"), "topology")
    corruptions = _mapping(payload.get("corruption_chains"), "corruption_chains")
    mixing = _mapping(payload.get("mixing"), "mixing")
    output = _mapping(payload.get("output"), "output")
    cap = mixing.get("vae_chain_weight_cap")
    seed_config_path = resolve_config_reference(
        method.get("random_seed_file"),
        owner_config_path=config_path,
        config_root=config_root,
        description="method.random_seed_file",
        must_exist=True,
    )
    load_random_seed_config(seed_config_path)
    config = AugMixConfig(
        config_path=config_path,
        random_seed_config_path=seed_config_path,
        random_namespace=str(method.get("random_namespace", "")),
        width=int(topology.get("width", 0)),
        chain_roles=tuple(str(value) for value in topology.get("chain_roles", ())),
        allowed_sampling_rates_hz=tuple(
            int(value) for value in topology.get("allowed_sampling_rates_hz", ())
        ),
        operator_config_path=resolve_config_reference(
            corruptions.get("operator_config"),
            owner_config_path=config_path,
            config_root=config_root,
            description="corruption_chains.operator_config",
            must_exist=True,
        ),
        operator_profile=str(corruptions.get("profile", "")),
        severity=int(corruptions.get("severity", 0)),
        depths=tuple(int(value) for value in corruptions.get("depths", ())),
        canonical_operators=tuple(
            str(value)
            for value in corruptions.get("canonical_operator_order", ())
        ),
        dirichlet_alpha=float(mixing.get("dirichlet_alpha", 0.0)),
        beta_alpha=float(mixing.get("beta_alpha", 0.0)),
        vae_chain_weight_cap=None if cap is None else float(cap),
        normalization_epsilon=float(output.get("normalization_epsilon", 0.0)),
    )
    expected_roles = (
        "paper_anchored_s5_corruption_chain",
        "paper_anchored_s5_corruption_chain",
        "vae_lhat_adversarial_waveform",
    )
    if config.width != 3 or config.chain_roles != expected_roles:
        raise ValueError("main AugMix topology must be the locked three-chain design")
    if bool(topology.get("chain3_additional_corruption", True)):
        raise ValueError("AugMix chain 3 must not receive an additional corruption")
    if bool(topology.get("raw_supervised_side_branch", True)):
        raise ValueError("main AugMix must not add a raw-supervised side branch")
    if config.allowed_sampling_rates_hz != (100, 500):
        raise ValueError("AugMix must support the declared 100 Hz and 500 Hz domains")
    if config.depths != (2, 3):
        raise ValueError("main AugMix corruption depths must be [2, 3]")
    if config.canonical_operators != CANONICAL_OPERATORS:
        raise ValueError("AugMix canonical operator order is invalid")
    if corruptions.get("sampling") != "uniform_over_all_depth2_depth3_compositions":
        raise ValueError("AugMix composition sampler must be uniform over depth2+3")
    if (
        corruptions.get("profile_status")
        != "project_defined_paper_anchored_not_official_preset"
    ):
        raise ValueError("AugMix profile status must not mislabel the custom preset")
    if bool(corruptions.get("replacement_within_chain", True)):
        raise ValueError("operators may not repeat within one corruption chain")
    if not bool(corruptions.get("independent_chains", False)):
        raise ValueError("AugMix corruption chains must be independently sampled")
    # The first locked implementation intentionally uses alpha=1, which can be
    # sampled with isolated Torch generators without touching global RNG state.
    if config.dirichlet_alpha != 1.0 or config.beta_alpha != 1.0:
        raise ValueError("manual AugMix v1 currently locks Dirichlet/Beta alpha to 1")
    if config.vae_chain_weight_cap is not None:
        raise ValueError("main AugMix v1 uses uncapped transparent Dirichlet weights")
    if config.normalization_epsilon <= 0.0:
        raise ValueError("AugMix normalization epsilon must be positive")
    if not config.random_namespace:
        raise ValueError("AugMix random namespace must be recorded")
    return config


@lru_cache(maxsize=8)
def _load_operator_profile(config: AugMixConfig) -> dict[str, dict[str, Any]]:
    if not config.operator_config_path.is_file():
        raise FileNotFoundError(
            f"augmentation operator config not found: {config.operator_config_path}"
        )
    payload = _mapping(
        yaml.safe_load(config.operator_config_path.read_text(encoding="utf-8")),
        "operator config",
    )
    metadata = _mapping(payload.get("metadata"), "operator config.metadata")
    operator_seed_path = resolve_config_reference(
        metadata.get("random_seed_file"),
        owner_config_path=config.operator_config_path,
        description="operator config.metadata.random_seed_file",
        must_exist=True,
    )
    if operator_seed_path != config.random_seed_config_path:
        raise ValueError("AugMix and operator config must use the same seed YAML")
    profiles = _mapping(payload.get("profiles"), "operator config.profiles")
    profile = _mapping(
        profiles.get(config.operator_profile),
        f"operator profile {config.operator_profile}",
    )
    resolved: dict[str, dict[str, Any]] = {}
    for operator in config.canonical_operators:
        severities = _mapping(profile.get(operator), f"profile.{operator}")
        params = severities.get(config.severity)
        if params is None:
            params = severities.get(str(config.severity))
        resolved[operator] = _mapping(
            params, f"profile.{operator}.{config.severity}"
        )
    return resolved


def _validate_waveform(value: torch.Tensor, description: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{description} must be a torch.Tensor")
    if value.ndim != 3 or value.shape[0] < 1 or value.shape[1] < 2 or value.shape[2] != 12:
        raise ValueError(f"{description} must have shape (B,time,12)")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{description} must be finite floating-point raw mV")
    return value


def _validate_generator(generator: torch.Generator, device: torch.device) -> None:
    generator_device = torch.device(generator.device)
    if generator_device.type != device.type:
        raise ValueError("AugMix generator device must match waveform device")
    if device.type == "cuda" and generator_device.index != device.index:
        raise ValueError("AugMix generator CUDA index must match waveform CUDA index")


def _all_compositions(config: AugMixConfig) -> tuple[tuple[str, ...], ...]:
    output: list[tuple[str, ...]] = []
    for depth in config.depths:
        output.extend(combinations(config.canonical_operators, depth))
    return tuple(output)


def _apply_operator(
    signal: torch.Tensor,
    operator: str,
    params: dict[str, Any],
    *,
    sampling_rate_hz: int,
    generator: torch.Generator,
) -> torch.Tensor:
    kwargs = dict(params)
    kwargs["rng"] = generator
    if operator in {"powerline_noise", "baseline_wander", "baseline_shift"}:
        kwargs["freq"] = float(sampling_rate_hz)
    return OPERATOR_FUNCTIONS[operator](signal, **kwargs)


def _apply_sampled_compositions(
    signal: torch.Tensor,
    composition_indices: torch.Tensor,
    *,
    compositions: tuple[tuple[str, ...], ...],
    operator_params: dict[str, dict[str, Any]],
    sampling_rate_hz: int,
    generator: torch.Generator,
) -> torch.Tensor:
    output = torch.empty_like(signal, dtype=torch.float32)
    for raw_index in torch.unique(composition_indices, sorted=True).tolist():
        index = int(raw_index)
        mask = composition_indices == index
        branch = signal[mask]
        for operator in compositions[index]:
            branch = _apply_operator(
                branch,
                operator,
                operator_params[operator],
                sampling_rate_hz=sampling_rate_hz,
                generator=generator,
            )
        output[mask] = branch
    return output.contiguous()


def _dirichlet_one(
    batch: int,
    width: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    uniform = torch.rand(
        batch,
        width,
        device=device,
        dtype=torch.float32,
        generator=generator,
    ).clamp_min(torch.finfo(torch.float32).tiny)
    exponentials = -uniform.log()
    return exponentials / exponentials.sum(dim=1, keepdim=True)


def _global_zscore_btc(signal: torch.Tensor, epsilon: float) -> torch.Tensor:
    mean = signal.mean(dim=(1, 2), keepdim=True)
    std = signal.std(dim=(1, 2), correction=0, keepdim=True)
    return ((signal - mean) / std.clamp_min(float(epsilon))).contiguous()


def make_augmix_generator(
    device: str | torch.device,
    *identity: Any,
    config: AugMixConfig | None = None,
) -> torch.Generator:
    """Create one persistent isolated AugMix stream for a run/epoch/worker."""

    resolved = load_augmix_config() if config is None else config
    return make_torch_generator(
        device,
        resolved.random_namespace,
        *identity,
        config_path=resolved.random_seed_config_path,
    )


def generate_three_chain_augmix(
    clean_raw: torch.Tensor,
    vae_lhat_raw: torch.Tensor,
    *,
    sampling_rate_hz: int,
    config: AugMixConfig | None = None,
    generator: torch.Generator | None = None,
) -> AugMixBatch:
    """Generate one raw and normalized locked three-chain AugMix view."""

    resolved = load_augmix_config() if config is None else config
    clean = _validate_waveform(clean_raw, "clean_raw")
    adversarial = _validate_waveform(vae_lhat_raw, "vae_lhat_raw")
    if clean.shape != adversarial.shape:
        raise ValueError("clean and VAE-LHAT waveforms must have identical shape")
    if int(sampling_rate_hz) not in resolved.allowed_sampling_rates_hz:
        raise ValueError(
            f"sampling_rate_hz must be one of {resolved.allowed_sampling_rates_hz}"
        )
    expected_points = int(sampling_rate_hz) * 10
    if clean.shape[1] != expected_points:
        raise ValueError(
            f"{sampling_rate_hz} Hz AugMix expects {expected_points} points, "
            f"got {clean.shape[1]}"
        )
    if generator is None:
        raise ValueError(
            "generator is required; create one persistent stream with "
            "make_augmix_generator"
        )
    _validate_generator(generator, clean.device)

    compositions = _all_compositions(resolved)
    operator_params = _load_operator_profile(resolved)
    batch = int(clean.shape[0])
    chain1_indices = torch.randint(
        0,
        len(compositions),
        (batch,),
        device=clean.device,
        generator=generator,
    )
    chain2_indices = torch.randint(
        0,
        len(compositions),
        (batch,),
        device=clean.device,
        generator=generator,
    )
    chain1 = _apply_sampled_compositions(
        clean,
        chain1_indices,
        compositions=compositions,
        operator_params=operator_params,
        sampling_rate_hz=int(sampling_rate_hz),
        generator=generator,
    )
    chain2 = _apply_sampled_compositions(
        clean,
        chain2_indices,
        compositions=compositions,
        operator_params=operator_params,
        sampling_rate_hz=int(sampling_rate_hz),
        generator=generator,
    )
    # Chain 3 is deliberately copied directly.  No operator is called on it.
    chain3 = adversarial.to(dtype=torch.float32).clone().contiguous()
    weights = _dirichlet_one(
        batch, resolved.width, device=clean.device, generator=generator
    )
    augmented_strength = torch.rand(
        batch,
        device=clean.device,
        dtype=torch.float32,
        generator=generator,
    )
    mixture = (
        weights[:, 0].view(-1, 1, 1) * chain1
        + weights[:, 1].view(-1, 1, 1) * chain2
        + weights[:, 2].view(-1, 1, 1) * chain3
    )
    strength = augmented_strength.view(-1, 1, 1)
    mixed = (1.0 - strength) * clean.to(dtype=torch.float32) + strength * mixture
    depths = torch.tensor(
        [len(value) for value in compositions],
        device=clean.device,
        dtype=torch.int64,
    )
    return AugMixBatch(
        chain1_raw=chain1,
        chain2_raw=chain2,
        chain3_raw=chain3,
        mixed_raw=mixed.contiguous(),
        mixed_normalized=_global_zscore_btc(
            mixed, resolved.normalization_epsilon
        ),
        mixture_weights=weights.contiguous(),
        augmented_strength=augmented_strength.contiguous(),
        chain1_composition_index=chain1_indices,
        chain2_composition_index=chain2_indices,
        chain1_depth=depths[chain1_indices],
        chain2_depth=depths[chain2_indices],
    )


def multilabel_jsd(logits_views: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Jensen-Shannon divergence over per-class Bernoulli predictions."""

    if len(logits_views) < 2:
        raise ValueError("multilabel_jsd requires at least two views")
    first_shape = logits_views[0].shape
    if len(first_shape) != 2 or any(value.shape != first_shape for value in logits_views):
        raise ValueError("all JSD logits must share shape (B,classes)")
    epsilon = torch.finfo(logits_views[0].dtype).eps
    probabilities = torch.stack(
        [torch.sigmoid(value).clamp(epsilon, 1.0 - epsilon) for value in logits_views],
        dim=0,
    )
    mean = probabilities.mean(dim=0).clamp(epsilon, 1.0 - epsilon)
    divergence = probabilities * (probabilities / mean).log()
    divergence += (1.0 - probabilities) * (
        (1.0 - probabilities) / (1.0 - mean)
    ).log()
    return divergence.mean()


__all__ = [
    "AugMixBatch",
    "AugMixConfig",
    "CANONICAL_OPERATORS",
    "DEFAULT_AUGMIX_CONFIG_PATH",
    "generate_three_chain_augmix",
    "load_augmix_config",
    "make_augmix_generator",
    "multilabel_jsd",
]
